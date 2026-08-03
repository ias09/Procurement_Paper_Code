"""
08_cross_domain_validation.py
================================
PURPOSE (read this before using these results in the manuscript):
This script is NOT a test of "does our supplier-delay framework predict real
supplier delays." It is a CROSS-DOMAIN ROBUSTNESS CHECK: does the framework's
machinery -- leakage-safe rolling features, forward-chaining temporal splits,
classification + regression modeling, conformalized quantile regression, and
cost-sensitive/service-level decision policies -- behave correctly and
produce sensible, well-calibrated results on a DIFFERENT dataset with genuine
learnable signal, where the entity being tracked is a CUSTOMER (DataCo's
e-commerce buyers) rather than a SUPPLIER.

WHY THIS DATASET, AND WHY IT'S A DOMAIN SHIFT (documented for the manuscript):
  - DataCo Global Supply Chain Dataset (Constante, Silva, Pereira; Mendeley,
    2019) is a B2C e-commerce shipping dataset: one company (DataCo) ships
    orders out to many customers. There is no multi-supplier structure --
    DataCo itself is the only "shipper." This is fundamentally different from
    the proposal's B2B procurement setting (many suppliers shipping TO one
    buyer organization).
  - We substitute Customer Id for supplier_id throughout. This is an explicit,
    labeled substitution, not a claim that customers and suppliers are
    conceptually equivalent -- it lets us test whether ROLLING-ENTITY-HISTORY
    FEATURES (the actual novel mechanism in Section 8.2-8.3 of the proposal)
    behave correctly on an entity that does have real, learnable behavioral
    variation (recall: Dataset 1 in 07_external_validation.py did NOT have
    this, which is why that test underperformed -- this script isolates
    whether the MECHANISM works when given an entity with real signal).
  - "promised_lead_time" = "Days for shipment (scheduled)"
    "actual_lead_time"   = "Days for shipping (real)"
    both fields already exist natively in DataCo, so Late/DelayDays/PLT/ALT
    (eqs. 1-5) transfer directly without invented thresholds.

DATA CLEANING: rows with Order Status == 'CANCELED' or 'SUSPECTED_FRAUD' are
dropped before computing targets, since these orders never completed
fulfillment (same logic as excluding cancelled POs in procurement data).

OUTPUTS:
  - data/dataco_clean.csv, data/dataco_features.csv
  - results/dataco_model_metrics.csv
  - results/dataco_conformal_metrics.csv
  - results/dataco_policy_comparison.csv
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
from lightgbm import LGBMClassifier, LGBMRegressor, LGBMRegressor as LGBMQuantile
from catboost import CatBoostClassifier, CatBoostRegressor

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
ALPHA = 0.10
BETA = 0.95
HOLDING_COST_RATE = 0.0005
SHORTAGE_COST_RATE = 0.02
BUFFER_GRID = np.arange(0, 11, 1)  # DataCo delays are much smaller scale (0-5 days) than synthetic

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

NON_FEATURE_COLUMNS = [
    "po_id", "order_date", "promised_delivery_date", "actual_delivery_date",
    "actual_lead_time", "delay", "delay_days", "late", "entity_id",
]
CATEGORICAL_CANDIDATES = ["category_or_material", "shipping_mode_or_priority", "market_or_plant"]


# ---------------------------------------------------------------------------
# STAGE 1: preprocessing (analog of 01_preprocessing.py)
# ---------------------------------------------------------------------------
def preprocess():
    raw = pd.read_csv(DATA_DIR / "dataco_raw.csv", encoding="latin1")
    raw = raw[~raw["Order Status"].isin(["CANCELED", "SUSPECTED_FRAUD"])].copy()

    df = pd.DataFrame({
        "po_id": raw["Order Item Id"].astype(str),
        "entity_id": raw["Customer Id"].astype(str),  # supplier_id ANALOG -- see docstring
        "category_or_material": raw["Category Name"],
        "order_date": pd.to_datetime(raw["order date (DateOrders)"]),
        "promised_lead_time": raw["Days for shipment (scheduled)"],
        "actual_lead_time": raw["Days for shipping (real)"],
        "quantity": raw["Order Item Quantity"],
        "unit_price": raw["Product Price"],
        "total_spend": raw["Order Item Total"],
        "shipping_mode_or_priority": raw["Shipping Mode"],
        "market_or_plant": raw["Market"],
    })
    df["promised_delivery_date"] = df["order_date"] + pd.to_timedelta(df["promised_lead_time"], unit="D")
    df["actual_delivery_date"] = df["order_date"] + pd.to_timedelta(df["actual_lead_time"], unit="D")
    df["delay"] = df["actual_lead_time"] - df["promised_lead_time"]
    df["delay_days"] = df["delay"].clip(lower=0)
    df["late"] = (df["delay_days"] > 0).astype(int)

    df = df.dropna(subset=["order_date", "promised_lead_time", "actual_lead_time"])
    print(f"[dataco preprocessing] rows after cleaning: {len(df)} | late rate: {df['late'].mean():.4f}")
    df.to_csv(DATA_DIR / "dataco_clean.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# STAGE 2: feature engineering (analog of 02_feature_engineering.py, same
# strict date-boundary leakage-safety rule, just keyed on entity_id instead
# of supplier_id)
# ---------------------------------------------------------------------------
def _strictly_past_indices(dates, i):
    boundary = np.searchsorted(dates, dates[i], side="left")
    return np.arange(boundary)


def _entity_group_features(group):
    group = group.sort_values(["order_date", "po_id"]).reset_index(drop=True)
    dates = group["order_date"].values
    late = group["late"].values.astype(float)
    delay = group["delay_days"].values.astype(float)
    n = len(group)

    ontime5 = np.full(n, np.nan)
    avgdelay5 = np.full(n, np.nan)
    avgdelay90 = np.full(n, np.nan)
    ontime90 = np.full(n, np.nan)
    order_count_so_far = np.zeros(n, dtype=int)
    ninety_days = np.timedelta64(90, "D")

    for i in range(n):
        past_idx = _strictly_past_indices(dates, i)
        order_count_so_far[i] = len(past_idx)
        if len(past_idx) == 0:
            continue
        last5 = past_idx[-5:]
        ontime5[i] = 1 - late[last5].mean()
        avgdelay5[i] = delay[last5].mean()

        cutoff = dates[i] - ninety_days
        win90 = past_idx[dates[past_idx] >= cutoff]
        if len(win90) > 0:
            ontime90[i] = 1 - late[win90].mean()
            avgdelay90[i] = delay[win90].mean()

    group["entity_ontime_rate_5"] = ontime5
    group["entity_avg_delay_5"] = avgdelay5
    group["entity_ontime_rate_90"] = ontime90
    group["entity_avg_delay_90"] = avgdelay90
    group["entity_order_count_so_far"] = order_count_so_far
    return group


def build_features(df):
    df = df.copy()
    df["order_month"] = df["order_date"].dt.month
    df["order_quarter"] = df["order_date"].dt.quarter

    parts = [_entity_group_features(g) for _, g in df.groupby("entity_id")]
    out = pd.concat(parts, ignore_index=True)
    return out


# ---------------------------------------------------------------------------
# STAGE 3: temporal split + feature matrix (analogs of 03/04/05's shared logic)
# ---------------------------------------------------------------------------
def split_temporal(df):
    # DataCo spans 2015-01 to 2018-01 (~37 months). Mirror the proposal's
    # train/calib/test ratios proportionally: ~24mo train, ~6mo calib, ~6mo test.
    train_end = pd.Timestamp("2016-12-31")
    calib_start, calib_end = pd.Timestamp("2017-01-01"), pd.Timestamp("2017-06-30")
    test_start, test_end = pd.Timestamp("2017-07-01"), pd.Timestamp("2018-01-31")

    train = df[df["order_date"] <= train_end].copy()
    calib = df[(df["order_date"] >= calib_start) & (df["order_date"] <= calib_end)].copy()
    test = df[(df["order_date"] >= test_start) & (df["order_date"] <= test_end)].copy()
    print(f"[dataco split] train={len(train)} calib={len(calib)} test={len(test)}")
    return train, calib, test


def build_feature_matrix(train, *other_frames):
    categorical_cols = [c for c in CATEGORICAL_CANDIDATES if c in train.columns]
    feature_cols = [c for c in train.columns if c not in NON_FEATURE_COLUMNS]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    train_enc = pd.get_dummies(train[feature_cols], columns=categorical_cols, dummy_na=False)
    train_medians = train[numeric_cols].median()
    train_enc[numeric_cols] = train[numeric_cols].fillna(train_medians)

    encoded = [train_enc]
    for frame in other_frames:
        f_enc = pd.get_dummies(frame[feature_cols], columns=categorical_cols, dummy_na=False)
        f_enc[numeric_cols] = frame[numeric_cols].fillna(train_medians)
        f_enc = f_enc.reindex(columns=train_enc.columns, fill_value=0)
        encoded.append(f_enc)
    return encoded


def classification_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else np.nan,
        "pr_auc": average_precision_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred),
        "brier": brier_score_loss(y_true, y_prob),
    }


def regression_metrics(y_true, y_pred):
    return {"mae": mean_absolute_error(y_true, y_pred), "rmse": np.sqrt(mean_squared_error(y_true, y_pred))}


def main():
    raw_clean = preprocess()
    features = build_features(raw_clean)
    features.to_csv(DATA_DIR / "dataco_features.csv", index=False)
    print(f"[dataco features] shape: {features.shape}")

    train, calib, test = split_temporal(features)
    train_X, calib_X, test_X = build_feature_matrix(train, calib, test)
    y_train_clf, y_train_reg = train["late"].values, train["delay_days"].values

    # --- 03 analog: classification + regression models ---
    metrics_rows = []
    classifiers = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        "random_forest": RandomForestClassifier(n_estimators=150, random_state=RANDOM_SEED, n_jobs=-1),
        "xgboost": XGBClassifier(n_estimators=150, random_state=RANDOM_SEED, eval_metric="logloss", verbosity=0),
        "lightgbm": LGBMClassifier(n_estimators=150, random_state=RANDOM_SEED, verbosity=-1),
        "catboost": CatBoostClassifier(n_estimators=150, random_seed=RANDOM_SEED, verbose=False),
    }
    regressors = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=150, random_state=RANDOM_SEED, n_jobs=-1),
        "xgboost": XGBRegressor(n_estimators=150, random_state=RANDOM_SEED, verbosity=0),
        "lightgbm": LGBMRegressor(n_estimators=150, random_state=RANDOM_SEED, verbosity=-1),
        "catboost": CatBoostRegressor(n_estimators=150, random_seed=RANDOM_SEED, verbose=False),
    }

    test_predictions = test[["po_id", "entity_id", "order_date", "late", "delay_days", "unit_price"]].copy()

    for name, clf in classifiers.items():
        clf.fit(train_X, y_train_clf)
        prob = clf.predict_proba(test_X)[:, 1]
        test_predictions[f"pred_late_prob__{name}"] = prob
        metrics_rows.append({"target": "classification", "model": name, **classification_metrics(test["late"].values, prob)})
        print(f"[dataco classification] {name}: ROC-AUC={roc_auc_score(test['late'].values, prob):.4f}")

    for name, reg in regressors.items():
        reg.fit(train_X, y_train_reg)
        pred = np.clip(reg.predict(test_X), 0, None)
        test_predictions[f"pred_delay__{name}"] = pred
        metrics_rows.append({"target": "regression", "model": name, **regression_metrics(test["delay_days"].values, pred)})
        print(f"[dataco regression] {name}: MAE={mean_absolute_error(test['delay_days'].values, pred):.4f}")

    pd.DataFrame(metrics_rows).to_csv(RESULTS_DIR / "dataco_model_metrics.csv", index=False)
    test_predictions.to_csv(RESULTS_DIR / "dataco_predictions_test.csv", index=False)

    # --- 04 analog: CQR conformal prediction ---
    y_calib = calib["delay_days"].values
    lo_model = LGBMQuantile(objective="quantile", alpha=ALPHA / 2, n_estimators=150, random_state=RANDOM_SEED, verbosity=-1)
    hi_model = LGBMQuantile(objective="quantile", alpha=1 - ALPHA / 2, n_estimators=150, random_state=RANDOM_SEED, verbosity=-1)
    lo_model.fit(train_X, y_train_reg)
    hi_model.fit(train_X, y_train_reg)

    q_lo_calib = lo_model.predict(calib_X)
    q_hi_calib = hi_model.predict(calib_X)
    nonconformity = np.maximum(q_lo_calib - y_calib, y_calib - q_hi_calib)
    n_c = len(nonconformity)
    level = min(1.0, np.ceil((n_c + 1) * (1 - ALPHA)) / n_c)
    Q = np.quantile(nonconformity, level)

    q_lo_test = lo_model.predict(test_X) - Q
    q_hi_test = hi_model.predict(test_X) + Q
    y_test_reg = test["delay_days"].values
    coverage = ((y_test_reg >= q_lo_test) & (y_test_reg <= q_hi_test)).mean()
    width = (q_hi_test - q_lo_test).mean()
    print(f"[dataco conformal] calibration Q={Q:.3f} | empirical coverage={coverage:.4f} (target {1-ALPHA:.2f}) | mean width={width:.2f}")
    pd.DataFrame([{"nominal_coverage": 1 - ALPHA, "empirical_coverage": coverage,
                    "mean_interval_width": width, "calibration_Q": Q}]).to_csv(
        RESULTS_DIR / "dataco_conformal_metrics.csv", index=False
    )

    # --- 05 analog: cost-sensitive decision layer ---
    median_model = LGBMQuantile(objective="quantile", alpha=0.50, n_estimators=150, random_state=RANDOM_SEED, verbosity=-1)
    median_model.fit(train_X, y_train_reg)
    median_calib_pred = median_model.predict(calib_X)
    calib_residuals = y_calib - median_calib_pred
    median_test_pred = median_model.predict(test_X)

    unit_price = test["unit_price"].values
    c_hold = HOLDING_COST_RATE * unit_price
    c_short = SHORTAGE_COST_RATE * unit_price

    # subsample calibration residuals to keep the simulated-delay matrix a
    # manageable size (23k test POs x full 29k calib residuals would need
    # ~5GB; 2000 residual samples is more than enough to approximate the
    # distribution while keeping memory reasonable)
    rng = np.random.default_rng(RANDOM_SEED)
    if len(calib_residuals) > 2000:
        calib_residuals_sample = rng.choice(calib_residuals, size=2000, replace=False)
    else:
        calib_residuals_sample = calib_residuals

    simulated = np.clip(median_test_pred[:, None] + calib_residuals_sample[None, :], 0, None)
    best_cost = np.full(len(test), np.inf)
    cost_optimal_buffer = np.zeros(len(test))
    service_level_buffer = np.full(len(test), BUFFER_GRID.max())
    sl_set = np.zeros(len(test), dtype=bool)

    for b in BUFFER_GRID:
        shortage = np.maximum(0, simulated - b)
        expected_cost = c_hold * b + c_short * shortage.mean(axis=1)
        improve = expected_cost < best_cost
        cost_optimal_buffer[improve] = b
        best_cost[improve] = expected_cost[improve]

        coverage_b = (simulated <= b).mean(axis=1)
        newly_satisfied = (~sl_set) & (coverage_b >= BETA)
        service_level_buffer[newly_satisfied] = b
        sl_set |= newly_satisfied

    realized_delay = test["delay_days"].values
    policies = {
        "erp_baseline": np.zeros(len(test)),
        "quantile_buffer_p90": np.clip(hi_model.predict(test_X), 0, None),  # reuse q=0.95 upper as p90-ish proxy
        "conformal_buffer": np.clip(q_hi_test, 0, None),
        "cost_optimal_buffer": cost_optimal_buffer,
        "service_level_buffer": service_level_buffer,
    }

    policy_rows = []
    for name, buffer in policies.items():
        cost = c_hold * buffer + c_short * np.maximum(0, realized_delay - buffer)
        service_met = (realized_delay <= buffer).astype(int)
        policy_rows.append({
            "policy": name, "mean_buffer_days": buffer.mean(), "mean_realized_cost": cost.mean(),
            "service_level_attained": service_met.mean(), "shortage_event_rate": 1 - service_met.mean(),
        })
    policy_df = pd.DataFrame(policy_rows)
    policy_df.to_csv(RESULTS_DIR / "dataco_policy_comparison.csv", index=False)

    print("\n=== DataCo cross-domain policy comparison ===")
    print(policy_df.to_string(index=False))
    print("\nSaved: dataco_model_metrics.csv, dataco_conformal_metrics.csv, "
          "dataco_policy_comparison.csv, dataco_predictions_test.csv")
    print("\nREMINDER: these results test pipeline MECHANICS under domain shift "
          "(Customer Id as entity, e-commerce shipping), NOT the proposal's "
          "supplier-procurement claim directly -- see this script's docstring.")


if __name__ == "__main__":
    main()
