"""
03_modeling.py  (v2 — Optuna-tuned + stacking ensemble)
=========================================================
Proposal mapping: Section 10 (Model Development), Section 11 (Time-Consistent
Validation).

IMPROVEMENTS over v1:
  1. Enriched feature set from 02_feature_engineering.py v2 (rolling-10/30,
     days_since_last_late, lead_time_ratio, interaction features)
  2. Optuna hyperparameter tuning for LightGBM and CatBoost, 60 trials each,
     optimizing ROC-AUC on the CALIBRATION fold so test remains a clean holdout
  3. Stacking meta-ensemble: base-model probabilities blended via logistic
     meta-learner fitted on calibration fold
  4. class_weight="balanced" and scale_pos_weight for classifiers

TEMPORAL SPLIT:
  Train: 2023-01->2023-12 | Calib: 2024-01->2024-03 | Test: 2024-04->2024-06 | Fold2: 2024-07+

Outputs:
  - results/model_metrics.csv
  - results/predictions_test.csv / predictions_fold2.csv
  - results/fitted_feature_columns.json
  - results/optuna_best_params.json
"""

import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
    brier_score_loss, mean_absolute_error, mean_squared_error)
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_SEED = 42
DATA_DIR    = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

TRAIN_END   = pd.Timestamp("2023-12-31")
CALIB_START = pd.Timestamp("2024-01-01"); CALIB_END  = pd.Timestamp("2024-03-31")
TEST_START  = pd.Timestamp("2024-04-01"); TEST_END   = pd.Timestamp("2024-06-30")
FOLD2_START = pd.Timestamp("2024-07-01")

NON_FEATURE_COLUMNS = [
    "po_id","order_date","promised_delivery_date","actual_delivery_date",
    "actual_lead_time","delay","delay_days","late",
]
CATEGORICAL_CANDIDATES = ["supplier_id","material_or_category","order_priority_or_type","plant_id","buyer_id"]
ZERO_IMPUTE_FEATURES   = ["is_urgent","is_q4","cold_start","lead_time_tight","supplier_workload_30d","workload_x_tight_lead"]


def load_synthetic():
    return pd.read_csv(DATA_DIR/"synthetic_features.csv",
        parse_dates=["order_date","promised_delivery_date","actual_delivery_date"])


def split_temporal(df):
    train = df[df["order_date"] <= TRAIN_END].copy()
    calib = df[(df["order_date"] >= CALIB_START) & (df["order_date"] <= CALIB_END)].copy()
    test  = df[(df["order_date"] >= TEST_START)  & (df["order_date"] <= TEST_END)].copy()
    fold2 = df[df["order_date"] >= FOLD2_START].copy()
    print(f"Split: train={len(train)} calib={len(calib)} test={len(test)} fold2={len(fold2)}")
    return train, calib, test, fold2


def build_feature_matrix(train, *other_frames):
    cat_cols     = [c for c in CATEGORICAL_CANDIDATES if c in train.columns]
    feature_cols = [c for c in train.columns if c not in NON_FEATURE_COLUMNS]
    num_cols     = [c for c in feature_cols if c not in cat_cols]

    train_enc    = pd.get_dummies(train[feature_cols], columns=cat_cols, dummy_na=False)
    train_medians = train[num_cols].median()
    for col in num_cols:
        train_enc[col] = train[col].fillna(0 if col in ZERO_IMPUTE_FEATURES else train_medians[col])

    encoded = [train_enc]
    for frame in other_frames:
        f_enc = pd.get_dummies(frame[feature_cols], columns=cat_cols, dummy_na=False)
        for col in num_cols:
            val = frame[col] if col in frame.columns else pd.Series(np.nan, index=frame.index)
            f_enc[col] = val.fillna(0 if col in ZERO_IMPUTE_FEATURES else train_medians[col])
        f_enc = f_enc.reindex(columns=train_enc.columns, fill_value=0)
        encoded.append(f_enc)
    return encoded, list(train_enc.columns), train_medians


def clf_metrics(y_true, y_prob):
    return {"roc_auc": roc_auc_score(y_true, y_prob) if len(set(y_true))>1 else np.nan,
            "pr_auc":  average_precision_score(y_true, y_prob),
            "f1":      f1_score(y_true, (y_prob>=0.5).astype(int)),
            "brier":   brier_score_loss(y_true, y_prob)}


def reg_metrics(y_true, y_pred):
    return {"mae": mean_absolute_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred))}


def tune_lgbm_clf(train_X, y_train, calib_X, y_calib, n_trials=60):
    def obj(trial):
        p = dict(n_estimators=trial.suggest_int("n_estimators",200,800),
                 learning_rate=trial.suggest_float("learning_rate",0.01,0.2,log=True),
                 num_leaves=trial.suggest_int("num_leaves",20,150),
                 max_depth=trial.suggest_int("max_depth",3,10),
                 min_child_samples=trial.suggest_int("min_child_samples",10,80),
                 subsample=trial.suggest_float("subsample",0.5,1.0),
                 colsample_bytree=trial.suggest_float("colsample_bytree",0.5,1.0),
                 reg_alpha=trial.suggest_float("reg_alpha",1e-4,10.0,log=True),
                 reg_lambda=trial.suggest_float("reg_lambda",1e-4,10.0,log=True),
                 class_weight="balanced", random_state=RANDOM_SEED, verbosity=-1)
        m = LGBMClassifier(**p); m.fit(train_X, y_train)
        return roc_auc_score(y_calib, m.predict_proba(calib_X)[:,1])
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(obj, n_trials=n_trials)
    print(f"  LightGBM clf best calib AUC: {study.best_value:.4f}")
    return study.best_params


def tune_catb_clf(train_X, y_train, calib_X, y_calib, n_trials=60):
    def obj(trial):
        p = dict(iterations=trial.suggest_int("iterations",200,800),
                 learning_rate=trial.suggest_float("learning_rate",0.01,0.2,log=True),
                 depth=trial.suggest_int("depth",4,10),
                 l2_leaf_reg=trial.suggest_float("l2_leaf_reg",1.0,20.0),
                 bagging_temperature=trial.suggest_float("bagging_temperature",0.0,1.0),
                 random_strength=trial.suggest_float("random_strength",0.0,2.0),
                 border_count=trial.suggest_int("border_count",32,255),
                 auto_class_weights="Balanced", random_seed=RANDOM_SEED, verbose=False)
        m = CatBoostClassifier(**p); m.fit(train_X, y_train)
        return roc_auc_score(y_calib, m.predict_proba(calib_X)[:,1])
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(obj, n_trials=n_trials)
    print(f"  CatBoost clf best calib AUC: {study.best_value:.4f}")
    return study.best_params


def tune_lgbm_reg(train_X, y_train, calib_X, y_calib, n_trials=40):
    def obj(trial):
        p = dict(n_estimators=trial.suggest_int("n_estimators",200,800),
                 learning_rate=trial.suggest_float("learning_rate",0.01,0.2,log=True),
                 num_leaves=trial.suggest_int("num_leaves",20,150),
                 max_depth=trial.suggest_int("max_depth",3,10),
                 min_child_samples=trial.suggest_int("min_child_samples",10,80),
                 subsample=trial.suggest_float("subsample",0.5,1.0),
                 colsample_bytree=trial.suggest_float("colsample_bytree",0.5,1.0),
                 random_state=RANDOM_SEED, verbosity=-1)
        m = LGBMRegressor(**p); m.fit(train_X, y_train)
        return mean_absolute_error(y_calib, np.clip(m.predict(calib_X),0,None))
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(obj, n_trials=n_trials)
    print(f"  LightGBM reg best calib MAE: {study.best_value:.4f}")
    return study.best_params


def main():
    df = load_synthetic()
    train, calib, test, fold2 = split_temporal(df)
    (train_X, calib_X, test_X, fold2_X), feat_cols, train_medians = \
        build_feature_matrix(train, calib, test, fold2)

    y_train_clf = train["late"].values;        y_calib_clf = calib["late"].values
    y_train_reg = train["delay_days"].values;  y_calib_reg = calib["delay_days"].values
    pos_weight  = (y_train_clf==0).sum() / (y_train_clf==1).sum()

    all_metrics       = []
    test_preds        = test[["po_id","supplier_id","order_date","late","delay_days"]].copy()
    fold2_preds       = fold2[["po_id","supplier_id","order_date","late","delay_days"]].copy()

    # baselines
    sup_delay = train.groupby("supplier_id")["delay_days"].mean()
    sup_late  = train.groupby("supplier_id")["late"].mean()
    for fold_name, frame, preds in [("test",test,test_preds),("fold2",fold2,fold2_preds)]:
        hd = frame["supplier_id"].map(sup_delay).fillna(train["delay_days"].mean()).values
        hl = frame["supplier_id"].map(sup_late).fillna(train["late"].mean()).values
        preds["pred_late_prob__erp_baseline"]   = 0.0
        preds["pred_late_prob__historical_avg"] = hl
        preds["pred_delay__erp_baseline"]        = 0.0
        preds["pred_delay__historical_avg"]      = hd
        for m,yp in [("erp_baseline",np.zeros(len(frame))),("historical_avg",hl)]:
            all_metrics.append({"fold":fold_name,"target":"classification","model":m,**clf_metrics(frame["late"].values,yp)})
        for m,yp in [("erp_baseline",np.zeros(len(frame))),("historical_avg",hd)]:
            all_metrics.append({"fold":fold_name,"target":"regression","model":m,**reg_metrics(frame["delay_days"].values,yp)})

    # Optuna tuning
    print("\nOptuna tuning (this takes ~4-6 mins)...")
    lgbm_clf_p = tune_lgbm_clf(train_X, y_train_clf, calib_X, y_calib_clf, n_trials=60)
    catb_clf_p = tune_catb_clf(train_X, y_train_clf, calib_X, y_calib_clf, n_trials=60)
    lgbm_reg_p = tune_lgbm_reg(train_X, y_train_reg, calib_X, y_calib_reg, n_trials=40)

    classifiers = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED),
        "random_forest":       RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                                       max_features="sqrt", min_samples_leaf=5,
                                                       random_state=RANDOM_SEED, n_jobs=-1),
        "xgboost":             XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=6,
                                              subsample=0.8, colsample_bytree=0.8,
                                              scale_pos_weight=pos_weight,
                                              random_state=RANDOM_SEED, eval_metric="logloss", verbosity=0),
        "lightgbm":            LGBMClassifier(**{**lgbm_clf_p, "random_state":RANDOM_SEED, "verbosity":-1}),
        "catboost":            CatBoostClassifier(**{**catb_clf_p, "random_seed":RANDOM_SEED, "verbose":False}),
    }

    base_calib = {}; base_test = {}; base_fold2 = {}
    print("\nTraining classifiers...")
    for name, clf in classifiers.items():
        clf.fit(train_X, y_train_clf)
        pc = clf.predict_proba(calib_X)[:,1]
        pt = clf.predict_proba(test_X)[:,1]
        pf = clf.predict_proba(fold2_X)[:,1]
        base_calib[name]=pc; base_test[name]=pt; base_fold2[name]=pf
        test_preds[f"pred_late_prob__{name}"]  = pt
        fold2_preds[f"pred_late_prob__{name}"] = pf
        for fn,y,p in [("test",test["late"].values,pt),("fold2",fold2["late"].values,pf)]:
            all_metrics.append({"fold":fn,"target":"classification","model":name,**clf_metrics(y,p)})
        print(f"  {name}: test ROC-AUC={roc_auc_score(test['late'].values, pt):.4f}")

    # stacking ensemble
    meta_X_calib = np.column_stack(list(base_calib.values()))
    meta_X_test  = np.column_stack(list(base_test.values()))
    meta_X_fold2 = np.column_stack(list(base_fold2.values()))
    meta = LogisticRegression(C=1.0, random_state=RANDOM_SEED, max_iter=500)
    meta.fit(meta_X_calib, y_calib_clf)
    st  = meta.predict_proba(meta_X_test)[:,1]
    sf  = meta.predict_proba(meta_X_fold2)[:,1]
    test_preds["pred_late_prob__stacking_ensemble"]  = st
    fold2_preds["pred_late_prob__stacking_ensemble"] = sf
    for fn,y,p in [("test",test["late"].values,st),("fold2",fold2["late"].values,sf)]:
        all_metrics.append({"fold":fn,"target":"classification","model":"stacking_ensemble",**clf_metrics(y,p)})
    print(f"  stacking_ensemble: test ROC-AUC={roc_auc_score(test['late'].values, st):.4f}")

    regressors = {
        "linear_regression": LinearRegression(),
        "random_forest":     RandomForestRegressor(n_estimators=500, min_samples_leaf=5,
                                                    random_state=RANDOM_SEED, n_jobs=-1),
        "xgboost":           XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=6,
                                           subsample=0.8, colsample_bytree=0.8,
                                           random_state=RANDOM_SEED, verbosity=0),
        "lightgbm":          LGBMRegressor(**{**lgbm_reg_p, "random_state":RANDOM_SEED, "verbosity":-1}),
        "catboost":          CatBoostRegressor(n_estimators=400, learning_rate=0.05, depth=6,
                                                random_seed=RANDOM_SEED, verbose=False),
    }
    print("\nTraining regressors...")
    for name, reg in regressors.items():
        reg.fit(train_X, y_train_reg)
        pt = np.clip(reg.predict(test_X),0,None)
        pf = np.clip(reg.predict(fold2_X),0,None)
        test_preds[f"pred_delay__{name}"]  = pt
        fold2_preds[f"pred_delay__{name}"] = pf
        for fn,y,p in [("test",test["delay_days"].values,pt),("fold2",fold2["delay_days"].values,pf)]:
            all_metrics.append({"fold":fn,"target":"regression","model":name,**reg_metrics(y,p)})
        print(f"  {name}: test MAE={mean_absolute_error(test['delay_days'].values, pt):.4f}")

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(RESULTS_DIR/"model_metrics.csv", index=False)
    test_preds.to_csv(RESULTS_DIR/"predictions_test.csv", index=False)
    fold2_preds.to_csv(RESULTS_DIR/"predictions_fold2.csv", index=False)
    json.dump({"lightgbm_clf":lgbm_clf_p,"catboost_clf":catb_clf_p,"lightgbm_reg":lgbm_reg_p},
              open(RESULTS_DIR/"optuna_best_params.json","w"), indent=2)
    json.dump({"feature_columns":feat_cols,"train_medians":train_medians.to_dict(),"random_seed":RANDOM_SEED},
              open(RESULTS_DIR/"fitted_feature_columns.json","w"), indent=2, default=str)

    print("\n=== FINAL SCORES (test fold, classification) ===")
    r = metrics_df[(metrics_df["fold"]=="test")&(metrics_df["target"]=="classification")]
    print(r[["model","roc_auc","pr_auc","f1","brier"]].sort_values("roc_auc",ascending=False).to_string(index=False))
    print("\n=== FINAL SCORES (test fold, regression) ===")
    r = metrics_df[(metrics_df["fold"]=="test")&(metrics_df["target"]=="regression")]
    print(r[["model","mae","rmse"]].sort_values("mae").to_string(index=False))


if __name__ == "__main__":
    main()
