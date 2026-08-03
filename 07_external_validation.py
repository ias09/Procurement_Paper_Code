"""
07_external_validation.py
============================
Proposal mapping: "Dataset Design and Experimental Setup" doc, Section 2.5-2.6
(Public Procurement Dataset / external validation), tying back to Sections
10-12 (Model Development, Calibration) for the metrics computed here.

DESIGN: models are trained on the FULL synthetic dataset (all of 2023-2024,
not just the internal train fold) because by this stage internal temporal
validation (forward-chaining, Sections 11/03_modeling.py) is already done and
passed. The genuine generalization test here is a DIFFERENT external dataset
(Kaggle Dataset 1) with an entirely different supplier population, so there
is no leakage risk in using all synthetic data to train -- Dataset 1's
suppliers/POs were never seen during synthetic-data generation or training.

CRITICAL HONESTY POINT: Dataset 1 has its own supplier IDs (not S01-S40),
its own category labels, and no plant_id/buyer_id at all. When we align
Dataset 1's encoded features onto the synthetic model's training columns,
every supplier_id_*/material_or_category_* dummy that doesn't exactly match
a synthetic-side label becomes all-zero for Dataset 1 -- meaning the model
CANNOT use supplier identity or any synthetic-specific category dummy for
Dataset 1 at all. It can only generalize via: (a) numeric rolling-history
features (computed independently, with the same leakage-safe logic, on
Dataset 1's own order sequence), and (b) shared numeric order-level features
(quantity, unit_price, total_spend, promised_lead_time, order_month, etc).
This is the correct and honest test of transfer -- a model that only worked
by memorizing synthetic supplier IDs would score near-baseline here, exposing
that weakness rather than hiding it.

Outputs:
  - results/external_validation_metrics.csv   (classification + regression)
  - results/external_validation_calibration.csv
  - results/external_validation_predictions.csv
"""

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
from catboost import CatBoostClassifier, CatBoostRegressor

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

NON_FEATURE_COLUMNS = [
    "po_id", "order_date", "promised_delivery_date", "actual_delivery_date",
    "actual_lead_time", "delay", "delay_days", "late",
]
CATEGORICAL_CANDIDATES = [
    "supplier_id", "material_or_category", "order_priority_or_type",
    "plant_id", "buyer_id",
]


def load_data():
    synthetic = pd.read_csv(
        DATA_DIR / "synthetic_features.csv",
        parse_dates=["order_date", "promised_delivery_date", "actual_delivery_date"],
    )
    ds1 = pd.read_csv(
        DATA_DIR / "dataset1_features.csv",
        parse_dates=["order_date", "promised_delivery_date", "actual_delivery_date"],
    )
    return synthetic, ds1


def build_feature_matrices(synthetic, ds1):
    """Fits one-hot encoding + median imputation on the FULL synthetic dataset,
    then aligns Dataset 1 onto those exact columns (unseen categories -> 0)."""
    categorical_cols = [c for c in CATEGORICAL_CANDIDATES if c in synthetic.columns]
    feature_cols = [c for c in synthetic.columns if c not in NON_FEATURE_COLUMNS]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    synthetic_enc = pd.get_dummies(synthetic[feature_cols], columns=categorical_cols, dummy_na=False)
    synthetic_medians = synthetic[numeric_cols].median()
    synthetic_enc[numeric_cols] = synthetic[numeric_cols].fillna(synthetic_medians)

    # Dataset 1 may lack some categorical columns entirely (plant_id, buyer_id) --
    # build only from what it actually has, then reindex onto synthetic's columns.
    ds1_categorical_cols = [c for c in categorical_cols if c in ds1.columns]
    ds1_feature_cols = [c for c in feature_cols if c in ds1.columns]
    ds1_numeric_cols = [c for c in ds1_feature_cols if c not in ds1_categorical_cols]

    ds1_enc = pd.get_dummies(ds1[ds1_feature_cols], columns=ds1_categorical_cols, dummy_na=False)
    ds1_enc[ds1_numeric_cols] = ds1[ds1_numeric_cols].fillna(synthetic_medians[ds1_numeric_cols])
    ds1_enc = ds1_enc.reindex(columns=synthetic_enc.columns, fill_value=0)

    n_matched_dummy_cols = sum(
        1 for c in synthetic_enc.columns
        if any(c.startswith(cat + "_") for cat in categorical_cols) and ds1_enc[c].sum() > 0
    )
    print(f"Diagnostic: {n_matched_dummy_cols} one-hot category columns have ANY nonzero "
          f"overlap between synthetic training categories and Dataset 1 (expect this to be "
          f"low/zero for supplier_id and plant_id/buyer_id, since those don't overlap by design).")

    return synthetic_enc, ds1_enc


def classification_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else np.nan,
        "pr_auc": average_precision_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred),
        "brier": brier_score_loss(y_true, y_prob),
    }


def regression_metrics(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
    }


def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_prob, bins, right=True) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(y_true)) * abs(y_prob[mask].mean() - y_true[mask].mean())
    return ece


def get_classifiers():
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1),
        "xgboost": XGBClassifier(n_estimators=300, random_state=RANDOM_SEED, eval_metric="logloss", verbosity=0),
        "lightgbm": LGBMClassifier(n_estimators=300, random_state=RANDOM_SEED, verbosity=-1),
        "catboost": CatBoostClassifier(n_estimators=300, random_seed=RANDOM_SEED, verbose=False),
    }


def get_regressors():
    return {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1),
        "xgboost": XGBRegressor(n_estimators=300, random_state=RANDOM_SEED, verbosity=0),
        "lightgbm": LGBMRegressor(n_estimators=300, random_state=RANDOM_SEED, verbosity=-1),
        "catboost": CatBoostRegressor(n_estimators=300, random_seed=RANDOM_SEED, verbose=False),
    }


def main():
    synthetic, ds1 = load_data()
    synthetic_X, ds1_X = build_feature_matrices(synthetic, ds1)

    y_train_clf = synthetic["late"].values
    y_train_reg = synthetic["delay_days"].values
    y_ext_clf = ds1["late"].values
    y_ext_reg = ds1["delay_days"].values

    metrics_rows = []
    calib_rows = []
    predictions = ds1[["po_id", "supplier_id", "order_date", "late", "delay_days"]].copy()

    # --- Baselines for context ---
    erp_pred_late = np.zeros(len(ds1))
    erp_pred_delay = np.zeros(len(ds1))
    predictions["pred_late_prob__erp_baseline"] = erp_pred_late
    predictions["pred_delay__erp_baseline"] = erp_pred_delay
    metrics_rows.append({"target": "classification", "model": "erp_baseline",
                          **classification_metrics(y_ext_clf, erp_pred_late)})
    metrics_rows.append({"target": "regression", "model": "erp_baseline",
                          **regression_metrics(y_ext_reg, erp_pred_delay)})

    # historical-average baseline using Dataset 1's OWN supplier history (computed
    # in 02_feature_engineering.py, leakage-safe) rather than synthetic suppliers,
    # since synthetic supplier IDs don't exist in Dataset 1 at all
    hist_pred_delay = ds1["supplier_avg_delay_90"].fillna(ds1["delay_days"].mean()).values
    hist_pred_late = (1 - ds1["supplier_ontime_rate_90"].fillna(1 - ds1["late"].mean())).values
    predictions["pred_late_prob__historical_avg"] = hist_pred_late
    predictions["pred_delay__historical_avg"] = hist_pred_delay
    metrics_rows.append({"target": "classification", "model": "historical_avg",
                          **classification_metrics(y_ext_clf, hist_pred_late)})
    metrics_rows.append({"target": "regression", "model": "historical_avg",
                          **regression_metrics(y_ext_reg, hist_pred_delay)})

    # --- ML classifiers (trained on FULL synthetic data, evaluated on Dataset 1) ---
    for name, clf in get_classifiers().items():
        clf.fit(synthetic_X, y_train_clf)
        prob = clf.predict_proba(ds1_X)[:, 1]
        predictions[f"pred_late_prob__{name}"] = prob
        metrics_rows.append({"target": "classification", "model": name,
                              **classification_metrics(y_ext_clf, prob)})
        ece = expected_calibration_error(y_ext_clf, prob)
        calib_rows.append({"model": name, "brier_score": brier_score_loss(y_ext_clf, prob), "ece": ece})
        print(f"[classification] {name}: ROC-AUC={roc_auc_score(y_ext_clf, prob):.4f} on Dataset 1")

    # --- ML regressors ---
    for name, reg in get_regressors().items():
        reg.fit(synthetic_X, y_train_reg)
        pred = np.clip(reg.predict(ds1_X), 0, None)
        predictions[f"pred_delay__{name}"] = pred
        metrics_rows.append({"target": "regression", "model": name,
                              **regression_metrics(y_ext_reg, pred)})
        print(f"[regression] {name}: MAE={mean_absolute_error(y_ext_reg, pred):.3f} on Dataset 1")

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(RESULTS_DIR / "external_validation_metrics.csv", index=False)
    pd.DataFrame(calib_rows).to_csv(RESULTS_DIR / "external_validation_calibration.csv", index=False)
    predictions.to_csv(RESULTS_DIR / "external_validation_predictions.csv", index=False)

    print("\n=== External validation (Dataset 1) -- classification ===")
    print(metrics_df[metrics_df["target"] == "classification"].to_string(index=False))
    print("\n=== External validation (Dataset 1) -- regression ===")
    print(metrics_df[metrics_df["target"] == "regression"].to_string(index=False))
    print("\nSaved: results/external_validation_metrics.csv, "
          "external_validation_calibration.csv, external_validation_predictions.csv")


if __name__ == "__main__":
    main()
