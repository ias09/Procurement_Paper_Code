import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, brier_score_loss,
    mean_absolute_error, mean_squared_error,
)
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

warnings.filterwarnings("ignore")

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except ImportError as e:
    raise ImportError(
        "catboost is required for the DataCo cross-domain validation. "
        "tab:dataco_clf and tab:dataco_reg report 5 models including CatBoost, "
        "and the cross-dataset comparison (tab:cross_dataset) assumes this row "
        "is present. The previous version of this script silently skipped "
        "CatBoost and continued with 4 models if this import failed, which "
        "would silently produce a results table that does not match the "
        "published one. Install it with `pip install catboost` before "
        "running this script."
    ) from e

RANDOM_SEED        = 42
ALPHA              = 0.10
BETA               = 0.95
HOLDING_COST_RATE  = 0.0005
SHORTAGE_COST_RATE = 0.02
BUFFER_GRID        = np.arange(0, 11, 1)

DATA_DIR    = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

NON_FEATURE_COLUMNS = [
    "po_id", "order_date", "promised_delivery_date", "actual_delivery_date",
    "actual_lead_time", "delay", "delay_days", "late", "entity_id",
]
CATEGORICAL_CANDIDATES = [
    "category_or_material", "shipping_mode_or_priority", "market_or_plant",
]

def preprocess():
    raw = pd.read_csv(DATA_DIR / "dataco_raw.csv", encoding="latin1")
    raw = raw[~raw["Order Status"].isin(["CANCELED", "SUSPECTED_FRAUD"])].copy()

    df = pd.DataFrame({
        "po_id":                   raw["Order Item Id"].astype(str),
        "entity_id":               raw["Customer Id"].astype(str),
        "category_or_material":    raw["Category Name"],
        "order_date":              pd.to_datetime(raw["order date (DateOrders)"]),
        "promised_lead_time":      raw["Days for shipment (scheduled)"],
        "actual_lead_time":        raw["Days for shipping (real)"],
        "quantity":                raw["Order Item Quantity"],
        "unit_price":              raw["Product Price"],
        "total_spend":             raw["Order Item Total"],
        "shipping_mode_or_priority": raw["Shipping Mode"],
        "market_or_plant":         raw["Market"],
    })
    df["promised_delivery_date"] = df["order_date"] + pd.to_timedelta(
        df["promised_lead_time"], unit="D")
    df["actual_delivery_date"] = df["order_date"] + pd.to_timedelta(
        df["actual_lead_time"], unit="D")
    df["delay"]      = df["actual_lead_time"] - df["promised_lead_time"]
    df["delay_days"] = df["delay"].clip(lower=0)
    df["late"]       = (df["delay_days"] > 0).astype(int)
    df = df.dropna(subset=["order_date", "promised_lead_time", "actual_lead_time"])

    print(f"[dataco preprocessing] rows={len(df)} | late rate={df['late'].mean():.4f}")
    df.to_csv(DATA_DIR / "dataco_clean.csv", index=False)
    return df


def _strictly_past_indices(dates, i):
    boundary = np.searchsorted(dates, dates[i], side="left")
    return np.arange(boundary)


def _entity_group_features(group):
    group  = group.sort_values(["order_date", "po_id"]).reset_index(drop=True)
    dates  = group["order_date"].values
    late   = group["late"].values.astype(float)
    delay  = group["delay_days"].values.astype(float)
    n      = len(group)

    ontime5       = np.full(n, np.nan)
    avgdelay5     = np.full(n, np.nan)
    avgdelay90    = np.full(n, np.nan)
    ontime90      = np.full(n, np.nan)
    order_count   = np.zeros(n, dtype=int)
    ninety_days   = np.timedelta64(90, "D")

    for i in range(n):
        past_idx = _strictly_past_indices(dates, i)
        order_count[i] = len(past_idx)
        if len(past_idx) == 0:
            continue
        last5 = past_idx[-5:]
        ontime5[i]    = 1 - late[last5].mean()
        avgdelay5[i]  = delay[last5].mean()
        cutoff = dates[i] - ninety_days
        win90  = past_idx[dates[past_idx] >= cutoff]
        if len(win90) > 0:
            ontime90[i]   = 1 - late[win90].mean()
            avgdelay90[i] = delay[win90].mean()

    group["entity_ontime_rate_5"]   = ontime5
    group["entity_avg_delay_5"]     = avgdelay5
    group["entity_ontime_rate_90"]  = ontime90
    group["entity_avg_delay_90"]    = avgdelay90
    group["entity_order_count_so_far"] = order_count
    return group


def build_features(df):
    df = df.copy()
    df["order_month"]   = df["order_date"].dt.month
    df["order_quarter"] = df["order_date"].dt.quarter
    parts = [_entity_group_features(g) for _, g in df.groupby("entity_id")]
    return pd.concat(parts, ignore_index=True)


def split_temporal(df):
    total    = len(df)
    n_train  = int(total * 0.60)
    n_calib  = int(total * 0.15)
    df_s     = df.sort_values("order_date").reset_index(drop=True)
    train    = df_s.iloc[:n_train].copy()
    calib    = df_s.iloc[n_train: n_train + n_calib].copy()
    test     = df_s.iloc[n_train + n_calib:].copy()
    print(f"[dataco split] train={len(train)} calib={len(calib)} test={len(test)}")
    return train, calib, test


def build_feature_matrix(train, *other_frames):
    categorical_cols = [c for c in CATEGORICAL_CANDIDATES if c in train.columns]
    feature_cols     = [c for c in train.columns if c not in NON_FEATURE_COLUMNS]
    numeric_cols     = [c for c in feature_cols if c not in categorical_cols]

    train_enc    = pd.get_dummies(train[feature_cols],
                                   columns=categorical_cols, dummy_na=False)
    train_medians = train[numeric_cols].median()
    train_enc[numeric_cols] = train[numeric_cols].fillna(train_medians)

    encoded = [train_enc]
    for frame in other_frames:
        f_enc = pd.get_dummies(frame[feature_cols],
                                columns=categorical_cols, dummy_na=False)
        f_enc[numeric_cols] = frame[numeric_cols].fillna(train_medians)
        f_enc = f_enc.reindex(columns=train_enc.columns, fill_value=0)
        encoded.append(f_enc)
    return encoded

def classification_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else np.nan,
        "pr_auc":  average_precision_score(y_true, y_prob),
        "f1":      f1_score(y_true, y_pred),
        "brier":   brier_score_loss(y_true, y_prob),
    }


def regression_metrics(y_true, y_pred):
    return {
        "mae":  mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
    }


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """ECE -- identical to 04_conformal_prediction.py implementation."""
    bins    = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_prob, bins, right=True) - 1, 0, n_bins - 1)
    ece     = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(y_true)) * abs(
            y_prob[mask].mean() - y_true[mask].mean())
    return ece


def evaluate_intervals(y_true, lower, upper, alpha):
    """Full interval evaluation suite -- mirrors 04_conformal_prediction.py."""
    covered    = (y_true >= lower) & (y_true <= upper)
    coverage   = covered.mean()
    width      = (upper - lower).mean()
    efficiency = coverage / width if width > 0 else np.nan
    width_cov  = (upper - lower)[covered].mean()  if covered.sum()  > 0 else np.nan
    width_miss = (upper - lower)[~covered].mean() if (~covered).sum() > 0 else np.nan
    penalty    = (2.0 / alpha) * (
        np.maximum(0, lower - y_true) + np.maximum(0, y_true - upper))
    winkler    = (upper - lower + penalty).mean()
    return {
        "empirical_coverage":  coverage,
        "mean_width":          width,
        "coverage_efficiency": efficiency,
        "mean_width_covered":  width_cov,
        "mean_width_missed":   width_miss,
        "winkler_score":       winkler,
    }


def main():
    raw_clean = preprocess()
    features  = build_features(raw_clean)
    features.to_csv(DATA_DIR / "dataco_features.csv", index=False)
    print(f"[dataco features] shape: {features.shape}")

    train, calib, test = split_temporal(features)
    train_X, calib_X, test_X = build_feature_matrix(train, calib, test)

    y_train_clf = train["late"].values
    y_train_reg = train["delay_days"].values

    metrics_rows    = []
    test_predictions = test[["po_id", "entity_id", "order_date",
                               "late", "delay_days", "unit_price"]].copy()

    classifiers = {
        "logistic_regression": LogisticRegression(max_iter=1000,
                                                   random_state=RANDOM_SEED),
        "random_forest":       RandomForestClassifier(n_estimators=150,
                                                       random_state=RANDOM_SEED,
                                                       n_jobs=-1),
        "xgboost":             XGBClassifier(n_estimators=150,
                                              random_state=RANDOM_SEED,
                                              eval_metric="logloss",
                                              verbosity=0),
        "lightgbm":            LGBMClassifier(n_estimators=150,
                                               random_state=RANDOM_SEED,
                                               verbosity=-1),
        "catboost":            CatBoostClassifier(
                                   n_estimators=150, random_seed=RANDOM_SEED, verbose=False),
    }

    print("\n=== DataCo Classification ===")
    for name, clf in classifiers.items():
        clf.fit(train_X, y_train_clf)
        prob  = clf.predict_proba(test_X)[:, 1]
        test_predictions[f"pred_late_prob__{name}"] = prob
        clf_m = classification_metrics(test["late"].values, prob)
        ece   = expected_calibration_error(test["late"].values, prob)
        metrics_rows.append({"target": "classification", "model": name,
                              **clf_m, "ece": ece})
        print(f"  {name:<28} ROC-AUC={clf_m['roc_auc']:.4f}  "
              f"PR-AUC={clf_m['pr_auc']:.4f}  F1={clf_m['f1']:.4f}  "
              f"Brier={clf_m['brier']:.4f}  ECE={ece:.4f}")

    regressors = {
        "linear_regression": LinearRegression(),
        "random_forest":     RandomForestRegressor(n_estimators=150,
                                                    random_state=RANDOM_SEED,
                                                    n_jobs=-1),
        "xgboost":           XGBRegressor(n_estimators=150,
                                           random_state=RANDOM_SEED,
                                           verbosity=0),
        "lightgbm":          LGBMRegressor(n_estimators=150,
                                            random_state=RANDOM_SEED,
                                            verbosity=-1),
        "catboost":          CatBoostRegressor(
                                 n_estimators=150, random_seed=RANDOM_SEED, verbose=False),
    }

    print("\n=== DataCo Regression ===")
    for name, reg in regressors.items():
        reg.fit(train_X, y_train_reg)
        pred = np.clip(reg.predict(test_X), 0, None)
        test_predictions[f"pred_delay__{name}"] = pred
        reg_m = regression_metrics(test["delay_days"].values, pred)
        metrics_rows.append({"target": "regression", "model": name, **reg_m})
        print(f"  {name:<28} MAE={reg_m['mae']:.4f}  RMSE={reg_m['rmse']:.4f}")

    pd.DataFrame(metrics_rows).to_csv(
        RESULTS_DIR / "dataco_model_metrics.csv", index=False)
    test_predictions.to_csv(
        RESULTS_DIR / "dataco_predictions_test.csv", index=False)

    print("\n=== DataCo CQR Conformal Prediction ===")
    y_calib_reg = calib["delay_days"].values
    y_test_reg  = test["delay_days"].values

    lo_model = LGBMRegressor(objective="quantile", alpha=ALPHA / 2,
                              n_estimators=150, random_state=RANDOM_SEED,
                              verbosity=-1)
    hi_model = LGBMRegressor(objective="quantile", alpha=1 - ALPHA / 2,
                              n_estimators=150, random_state=RANDOM_SEED,
                              verbosity=-1)
    lo_model.fit(train_X, y_train_reg)
    hi_model.fit(train_X, y_train_reg)

    q_lo_c = lo_model.predict(calib_X)
    q_hi_c = hi_model.predict(calib_X)
    nonconf = np.maximum(q_lo_c - y_calib_reg, y_calib_reg - q_hi_c)
    n_c     = len(nonconf)
    level   = min(1.0, np.ceil((n_c + 1) * (1 - ALPHA)) / n_c)
    Q       = np.quantile(nonconf, level)

    q_lo_test = np.clip(lo_model.predict(test_X) - Q, 0, None)
    q_hi_test = np.clip(hi_model.predict(test_X) + Q, 0, None)

    conf_m = evaluate_intervals(y_test_reg, q_lo_test, q_hi_test, ALPHA)
    conf_m.update({"nominal_coverage": 1 - ALPHA, "calibration_Q": Q})
    pd.DataFrame([conf_m]).to_csv(
        RESULTS_DIR / "dataco_conformal_metrics.csv", index=False)

    print(f"  calibration Q      : {Q:.3f}")
    print(f"  empirical coverage : {conf_m['empirical_coverage']:.4f} (target {1-ALPHA:.2f})")
    print(f"  mean width         : {conf_m['mean_width']:.2f} days")
    print(f"  coverage efficiency: {conf_m['coverage_efficiency']:.4f}")
    print(f"  winkler score      : {conf_m['winkler_score']:.4f}")

    print("\n=== DataCo Policy Comparison (5 policies) ===")
    median_model = LGBMRegressor(objective="quantile", alpha=0.50,
                                  n_estimators=150,
                                  random_state=RANDOM_SEED, verbosity=-1)
    median_model.fit(train_X, y_train_reg)
    calib_residuals  = y_calib_reg - median_model.predict(calib_X)
    median_test_pred = median_model.predict(test_X)

    unit_price = test["unit_price"].values
    c_hold     = HOLDING_COST_RATE  * unit_price
    c_short    = SHORTAGE_COST_RATE * unit_price

    rng = np.random.default_rng(RANDOM_SEED)
    calib_res_sample = (rng.choice(calib_residuals, size=2000, replace=False)
                        if len(calib_residuals) > 2000 else calib_residuals)

    simulated         = np.clip(median_test_pred[:, None]
                                + calib_res_sample[None, :], 0, None)
    best_cost         = np.full(len(test), np.inf)
    cost_opt_buf      = np.zeros(len(test))
    svc_buf           = np.full(len(test), BUFFER_GRID.max())
    sl_set            = np.zeros(len(test), dtype=bool)
    hist_buf          = calib["delay_days"].mean() * np.ones(len(test))

    for b in BUFFER_GRID:
        shortage     = np.maximum(0, simulated - b)
        exp_cost     = c_hold * b + c_short * shortage.mean(axis=1)
        improve      = exp_cost < best_cost
        cost_opt_buf[improve] = b
        best_cost[improve]    = exp_cost[improve]
        coverage_b   = (simulated <= b).mean(axis=1)
        newly_sl     = (~sl_set) & (coverage_b >= BETA)
        svc_buf[newly_sl] = b
        sl_set       |= newly_sl

    realized_delay = test["delay_days"].values
    
    q95_buf = np.clip(hi_model.predict(test_X), 0, None)

    policies = {
        "ERP Baseline":          np.zeros(len(test)),
        "Historical Buffer":      hist_buf,
        "Quantile Buffer (p95)":  q95_buf,
        "Conformal Buffer":       q_hi_test,
        "Cost-Optimal Buffer":    cost_opt_buf,
        "Service-Level Buffer":   svc_buf,
    }

    policy_rows = []
    for name, buf in policies.items():
        cost       = c_hold * buf + c_short * np.maximum(0, realized_delay - buf)
        svc_met    = (realized_delay <= buf).astype(int)
        policy_rows.append({
            "policy":                name,
            "mean_buffer_days":      buf.mean(),
            "mean_realized_cost":    cost.mean(),
            "service_level":         svc_met.mean(),
            "shortage_event_rate":   1 - svc_met.mean(),
        })

    policy_df = pd.DataFrame(policy_rows)
    policy_df.to_csv(RESULTS_DIR / "dataco_policy_comparison.csv", index=False)
    print(policy_df.to_string(index=False))

    print("\nSaved:")
    print("  dataco_model_metrics.csv      (+ ECE inline)")
    print("  dataco_conformal_metrics.csv  (+ efficiency, winkler)")
    print("  dataco_policy_comparison.csv  (all 6 policies)")
    print("  dataco_predictions_test.csv")
    print("\nREMINDER: DataCo tests pipeline MECHANICS under domain shift "
          "(Customer Id as entity, e-commerce shipping) -- not the B2B "
          "procurement supplier-reliability claim directly.")


if __name__ == "__main__":
    main()