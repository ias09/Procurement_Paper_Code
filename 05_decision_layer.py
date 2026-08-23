
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
BETA = 0.95          
HOLDING_COST_RATE = 0.0005   
SHORTAGE_COST_RATE = 0.02    
BUFFER_GRID = np.arange(0, 31, 1)  

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

    n_po = len(median_pred)
    n_resid = len(calib_residuals)


    simulated_delay = np.clip(median_pred[:, None] + calib_residuals[None, :], 0, None) 

    cost_optimal_buffer = np.zeros(n_po)
    service_level_buffer = np.zeros(n_po)
    best_cost_so_far = np.full(n_po, np.inf)
    resolved = np.zeros(n_po, dtype=bool)

    for b in BUFFER_GRID:
        shortage = np.maximum(0, simulated_delay - b) 
        expected_cost_b = c_hold * b + c_short * shortage.mean(axis=1)  
        improve_mask = expected_cost_b < best_cost_so_far
        cost_optimal_buffer[improve_mask] = b
        best_cost_so_far[improve_mask] = expected_cost_b[improve_mask]

        coverage_b = (simulated_delay <= b).mean(axis=1)  
       
        newly_satisfied = (~resolved) & (coverage_b >= beta)
        service_level_buffer[newly_satisfied] = b
        resolved |= newly_satisfied

    
    service_level_buffer[~resolved] = BUFFER_GRID.max()

    n_zero = int((resolved & (service_level_buffer == 0)).sum())
    n_unmet = int((~resolved).sum())
    print(f"Service-level buffers: {n_zero} POs satisfied at 0 days, "
          f"{n_unmet} POs unmet within grid (capped at {BUFFER_GRID.max()} days)")

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
        print(f"[{fold_name}] raw (unclipped) median_pred: min={median_pred.min():.2f}  "
              f"max={median_pred.max():.2f}  "
              f"(sentinel bug required min < ~{-1.645*calib_residuals.std():.2f} "
              f"given calib residual std={calib_residuals.std():.3f})")
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