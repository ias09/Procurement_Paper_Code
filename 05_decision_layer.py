"""
05_decision_layer.py
======================
Proposal mapping: Sections 13-15 (Cost-Sensitive Optimization, Expected-Cost
Optimization, Service-Level-Constrained Optimization), Section 17 (Decision
Layer / six policies), Section 18.2 (Decision Metrics).

COST MODEL (as agreed with user -- scaled per PO by unit_price, not fixed
constants, since orders vary widely in value):
    c_hold_i  = 0.05% of unit_price_i   per buffer day
    c_short_i = 2%    of unit_price_i   per late day beyond the buffer
    (≈40:1 ratio, matching the spirit of the proposal's worked example)

SIX POLICIES (Section 17, Table 4/12/13):
  1. ERP Baseline        -- buffer = 0 (trust promised date only)
  2. Historical Buffer   -- buffer = supplier's TRAIN-period average delay
                             (reuses 03_modeling.py's historical_avg baseline)
  3. Quantile Buffer     -- buffer = predicted 90th-percentile delay
                             (new dedicated q=0.90 LightGBM quantile model)
  4. Conformal Buffer    -- buffer = conformal upper bound from
                             04_conformal_prediction.py (90% coverage)
  5. Cost-Optimal Buffer -- buffer minimizing E[cost] (Section 14), found via
                             grid search using a calibration-residual-based
                             simulated delay distribution per PO (see below)
  6. Service-Level Buffer-- smallest buffer with P(delay <= buffer) >= beta
                             (beta = 0.95, Section 15), same simulated
                             distribution as policy 5

HOW WE SIMULATE THE PER-PO DELAY DISTRIBUTION FOR POLICIES 5 & 6:
  We fit one more quantile model at q=0.50 (median) on TRAIN. On the
  CALIBRATION fold we compute residuals = actual_delay - median_prediction.
  For a given test PO, we approximate its predicted delay distribution as
  {median_prediction(x) + r : r in calibration residuals}, clipped at 0.
  This reuses the calibration fold's empirical error distribution (already
  reserved exclusively for calibration, never touched in training) rather
  than assuming a parametric distribution -- consistent with the proposal's
  distribution-free philosophy in Section 9.

EVALUATION (Section 18.2): for each policy and each PO, we compute the
REALIZED cost and REALIZED service-level outcome using the PO's actual
delay_days (ground truth, available in this offline evaluation), so the six
policies can be compared on equal footing.

Outputs:
  - results/buffer_decisions_test.csv
  - results/buffer_decisions_fold2.csv
  - results/policy_comparison.csv      (mean cost, mean buffer, service-level
                                         attainment per policy, per fold)
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
BETA = 0.95          # required service level (Section 15)
HOLDING_COST_RATE = 0.0005   # 0.05% of unit_price per buffer day
SHORTAGE_COST_RATE = 0.02    # 2% of unit_price per late day beyond buffer
BUFFER_GRID = np.arange(0, 31, 1)  # candidate buffers to search, in days

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

TRAIN_END = pd.Timestamp("2023-12-31")
CALIB_START, CALIB_END = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-03-31")
TEST_START, TEST_END = pd.Timestamp("2024-04-01"), pd.Timestamp("2024-06-30")
FOLD2_START = pd.Timestamp("2024-07-01")

NON_FEATURE_COLUMNS = [
    "po_id", "order_date", "promised_delivery_date", "actual_delivery_date",
    "actual_lead_time", "delay", "delay_days", "late",
]
CATEGORICAL_CANDIDATES = [
    "supplier_id", "material_or_category", "order_priority_or_type",
    "plant_id", "buyer_id",
]


def load_synthetic():
    return pd.read_csv(
        DATA_DIR / "synthetic_features.csv",
        parse_dates=["order_date", "promised_delivery_date", "actual_delivery_date"],
    )


def split_temporal(df):
    train = df[df["order_date"] <= TRAIN_END].copy()
    calib = df[(df["order_date"] >= CALIB_START) & (df["order_date"] <= CALIB_END)].copy()
    test = df[(df["order_date"] >= TEST_START) & (df["order_date"] <= TEST_END)].copy()
    fold2 = df[df["order_date"] >= FOLD2_START].copy()
    return train, calib, test, fold2


def build_feature_matrix(train, *other_frames):
    categorical_cols = [c for c in CATEGORICAL_CANDIDATES if c in train.columns]
    feature_cols = [c for c in train.columns if c not in NON_FEATURE_COLUMNS]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    train_enc = pd.get_dummies(train[feature_cols], columns=categorical_cols, dummy_na=False)
    train_medians = train[numeric_cols].median()
    train_enc[numeric_cols] = train[numeric_cols].fillna(train_medians)

    encoded_frames = [train_enc]
    for frame in other_frames:
        f_enc = pd.get_dummies(frame[feature_cols], columns=categorical_cols, dummy_na=False)
        f_enc[numeric_cols] = frame[numeric_cols].fillna(train_medians)
        f_enc = f_enc.reindex(columns=train_enc.columns, fill_value=0)
        encoded_frames.append(f_enc)

    return encoded_frames


def cost_function(buffer, realized_delay, c_hold, c_short):
    """Eq. in Section 13: C_i = c_hold * Buffer_i + c_short * max(0, Delay_i - Buffer_i)"""
    return c_hold * buffer + c_short * np.maximum(0, realized_delay - buffer)


def cost_optimal_and_service_level_buffers(median_pred, calib_residuals, c_hold, c_short, beta=BETA):
    """For each PO (vectorized over POs), search BUFFER_GRID to find:
       - the cost-minimizing buffer (Section 14)
       - the smallest buffer satisfying the service-level constraint (Section 15)
    using the simulated distribution {median_pred + r, r in calib_residuals}, clipped at 0."""
    n_po = len(median_pred)
    n_resid = len(calib_residuals)

    # simulated_delay[i, j] = simulated delay for PO i under residual sample j
    simulated_delay = np.clip(median_pred[:, None] + calib_residuals[None, :], 0, None)  # (n_po, n_resid)

    cost_optimal_buffer = np.zeros(n_po)
    service_level_buffer = np.zeros(n_po)
    best_cost_so_far = np.full(n_po, np.inf)

    for b in BUFFER_GRID:
        shortage = np.maximum(0, simulated_delay - b)  # (n_po, n_resid)
        expected_cost_b = c_hold * b + c_short * shortage.mean(axis=1)  # (n_po,)
        improve_mask = expected_cost_b < best_cost_so_far
        cost_optimal_buffer[improve_mask] = b
        best_cost_so_far[improve_mask] = expected_cost_b[improve_mask]

        coverage_b = (simulated_delay <= b).mean(axis=1)  # (n_po,)
        # first buffer (smallest b, since grid ascending) hitting the service level
        not_yet_set = service_level_buffer == 0
        newly_satisfied = not_yet_set & (coverage_b >= beta) & (b > 0)
        # handle b=0 case separately (can satisfy at b=0 if all simulated delays are 0)
        if b == 0:
            satisfied_at_zero = coverage_b >= beta
            service_level_buffer[satisfied_at_zero] = 0
        else:
            service_level_buffer[newly_satisfied] = b

    # POs whose service level was never satisfied within the grid: use the grid max
    unmet = (service_level_buffer == 0) & (
        (np.clip(median_pred[:, None] + calib_residuals[None, :], 0, None) <= BUFFER_GRID.max()).mean(axis=1) < beta
    )
    service_level_buffer[unmet] = BUFFER_GRID.max()

    return cost_optimal_buffer, service_level_buffer


def evaluate_policy(realized_delay, buffer, c_hold, c_short):
    realized_cost = cost_function(buffer, realized_delay, c_hold, c_short)
    service_level_met = (realized_delay <= buffer).astype(int)
    return realized_cost, service_level_met


def main():
    df = load_synthetic()
    train, calib, test, fold2 = split_temporal(df)
    train_X, calib_X, test_X, fold2_X = build_feature_matrix(train, calib, test, fold2)

    y_train = train["delay_days"].values
    y_calib = calib["delay_days"].values

    # --- dedicated quantile models for this script: q=0.50 (median) and q=0.90 ---
    median_model = LGBMRegressor(objective="quantile", alpha=0.50,
                                  n_estimators=300, random_state=RANDOM_SEED, verbosity=-1)
    q90_model = LGBMRegressor(objective="quantile", alpha=0.90,
                               n_estimators=300, random_state=RANDOM_SEED, verbosity=-1)
    median_model.fit(train_X, y_train)
    q90_model.fit(train_X, y_train)

    median_calib_pred = median_model.predict(calib_X)
    calib_residuals = y_calib - median_calib_pred
    print(f"Calibration residuals: mean={calib_residuals.mean():.3f}, "
          f"std={calib_residuals.std():.3f}, n={len(calib_residuals)}")

    conformal_test = pd.read_csv(RESULTS_DIR / "conformal_intervals_test.csv")
    conformal_fold2 = pd.read_csv(RESULTS_DIR / "conformal_intervals_fold2.csv")
    hist_preds_test = pd.read_csv(RESULTS_DIR / "predictions_test.csv")
    hist_preds_fold2 = pd.read_csv(RESULTS_DIR / "predictions_fold2.csv")

    policy_summary = []

    for fold_name, frame, X, conf_df, hist_df in [
        ("test", test, test_X, conformal_test, hist_preds_test),
        ("fold2", fold2, fold2_X, conformal_fold2, hist_preds_fold2),
    ]:
        realized_delay = frame["delay_days"].values
        unit_price = frame["unit_price"].values
        c_hold = HOLDING_COST_RATE * unit_price
        c_short = SHORTAGE_COST_RATE * unit_price

        median_pred = median_model.predict(X)
        q90_pred = np.clip(q90_model.predict(X), 0, None)

        cost_optimal_buf, service_level_buf = cost_optimal_and_service_level_buffers(
            median_pred, calib_residuals, c_hold, c_short, beta=BETA
        )

        buffers = {
            "erp_baseline": np.zeros(len(frame)),
            "historical_buffer": hist_df.set_index("po_id").loc[frame["po_id"], "pred_delay__historical_avg"].values,
            "quantile_buffer_p90": q90_pred,
            "conformal_buffer": conf_df.set_index("po_id").loc[frame["po_id"], "conformal_buffer"].values,
            "cost_optimal_buffer": cost_optimal_buf,
            "service_level_buffer": service_level_buf,
        }

        decisions = frame[["po_id", "supplier_id", "order_date", "delay_days", "unit_price"]].copy()
        for policy_name, buffer in buffers.items():
            realized_cost, service_met = evaluate_policy(realized_delay, buffer, c_hold, c_short)
            decisions[f"buffer__{policy_name}"] = buffer
            decisions[f"cost__{policy_name}"] = realized_cost
            decisions[f"service_met__{policy_name}"] = service_met

            policy_summary.append({
                "fold": fold_name,
                "policy": policy_name,
                "mean_buffer_days": buffer.mean(),
                "mean_realized_cost": realized_cost.mean(),
                "total_realized_cost": realized_cost.sum(),
                "service_level_attained": service_met.mean(),
                "shortage_event_rate": 1 - service_met.mean(),
            })

        decisions.to_csv(RESULTS_DIR / f"buffer_decisions_{fold_name}.csv", index=False)
        print(f"\n[{fold_name}] saved buffer_decisions_{fold_name}.csv")

    summary_df = pd.DataFrame(policy_summary)
    summary_df.to_csv(RESULTS_DIR / "policy_comparison.csv", index=False)

    print("\n=== Policy comparison (test fold) ===")
    print(summary_df[summary_df["fold"] == "test"].to_string(index=False))
    print("\nSaved: results/policy_comparison.csv, buffer_decisions_test.csv, buffer_decisions_fold2.csv")


if __name__ == "__main__":
    main()
