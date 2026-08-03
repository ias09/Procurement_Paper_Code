"""
10_decision_calibrated_buffering.py
=====================================
PLAIN-LANGUAGE PURPOSE (Ideas 2 + 3 from the updated proposal):

IDEA 2 (cost decides the margin -- "DCCB"):
  Old way: model gives a "90% safe" margin (90% chosen arbitrarily), then a
  separate search finds the cheapest buffer. New way: the costs THEMSELVES
  tell us how safe to be. If being late costs 40x more than holding stock
  early, math says the margin should be 40/41 = 97.6% safe. Each order gets
  its own safety level from its own costs. One step, no arbitrary numbers,
  and expensive-to-miss orders automatically get bigger margins.

  The math: optimal buffer = the p-th percentile of the delay distribution,
  where p = c_short / (c_short + c_hold)  ["critical ratio"]

IDEA 3 (fairness across suppliers -- "Mondrian"):
  Old way guarantees "90% of ALL orders arrive within margin" -- an average
  that can hide erratic suppliers being constantly under-protected while calm
  suppliers are over-protected. New way: split suppliers into groups by how
  erratic they are (delay volatility terciles), and calibrate the margin
  correction separately per group. Each GROUP now gets the promised
  protection, not just the average.

WHAT THIS SCRIPT DOES:
  1. Trains quantile models for a grid of quantile levels (so we can serve
     any per-order critical ratio).
  2. Computes each test PO's critical ratio from its own costs.
  3. Builds three buffer policies and compares them on the test fold:
       a. fixed90_conformal   -- old approach: one-size 90% conformal buffer
       b. dccb_marginal       -- Idea 2: per-order critical-ratio coverage,
                                  one global conformal correction
       c. dccb_mondrian       -- Ideas 2+3: per-order coverage AND per-group
                                  (volatility tercile) conformal corrections
  4. Reports realized cost, coverage (overall AND per group), buffer sizes.

  THE KEY TABLE for the paper: per-group coverage. Marginal calibration
  should show the erratic-supplier group under-covered; Mondrian should fix
  it. Plus DCCB should beat fixed-90% on realized cost.

Outputs:
  - results/dccb_policy_comparison.csv
  - results/dccb_group_coverage.csv
  - results/dccb_buffers_test.csv
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
HOLDING_COST_RATE = 0.0005   # 0.05% of unit price per buffer day (same as script 05)
SHORTAGE_COST_RATE = 0.02    # 2% of unit price per late day (same as script 05)
SERVICE_FLOOR_BETA = 0.90    # minimum coverage floor even for cheap orders

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

TRAIN_END   = pd.Timestamp("2023-12-31")
CALIB_START = pd.Timestamp("2024-01-01"); CALIB_END = pd.Timestamp("2024-03-31")
TEST_START  = pd.Timestamp("2024-04-01"); TEST_END  = pd.Timestamp("2024-06-30")

NON_FEATURE_COLUMNS = [
    "po_id","order_date","promised_delivery_date","actual_delivery_date",
    "actual_lead_time","delay","delay_days","late",
]
CATEGORICAL_CANDIDATES = ["supplier_id","material_or_category","order_priority_or_type","plant_id","buyer_id"]

# Quantile grid: we train one quantile model per level, then serve each PO
# the level closest to (but not below) its critical ratio.
QUANTILE_GRID = [0.80, 0.85, 0.90, 0.925, 0.95, 0.965, 0.975, 0.985]


def load_and_split():
    df = pd.read_csv(DATA_DIR/"synthetic_features.csv",
        parse_dates=["order_date","promised_delivery_date","actual_delivery_date"])
    train = df[df["order_date"] <= TRAIN_END].copy()
    calib = df[(df["order_date"] >= CALIB_START) & (df["order_date"] <= CALIB_END)].copy()
    test  = df[(df["order_date"] >= TEST_START)  & (df["order_date"] <= TEST_END)].copy()
    return train, calib, test


def build_feature_matrix(train, *frames):
    cat_cols = [c for c in CATEGORICAL_CANDIDATES if c in train.columns]
    feat_cols = [c for c in train.columns if c not in NON_FEATURE_COLUMNS]
    num_cols = [c for c in feat_cols if c not in cat_cols]

    train_enc = pd.get_dummies(train[feat_cols], columns=cat_cols, dummy_na=False)
    med = train[num_cols].median()
    train_enc[num_cols] = train[num_cols].fillna(med)

    out = [train_enc]
    for f in frames:
        enc = pd.get_dummies(f[feat_cols], columns=cat_cols, dummy_na=False)
        enc[num_cols] = f[num_cols].fillna(med)
        enc = enc.reindex(columns=train_enc.columns, fill_value=0)
        out.append(enc)
    return out


def volatility_group(frame):
    """Assign each PO's supplier to a volatility tercile (calm / medium /
    erratic) based on the supplier's rolling delay std -- known at order time."""
    vol = frame["supplier_delay_std_90"].fillna(frame["supplier_delay_std_90"].median())
    q1, q2 = vol.quantile([1/3, 2/3])
    return np.where(vol <= q1, "calm", np.where(vol <= q2, "medium", "erratic"))


def conformal_correction(scores, level):
    """Finite-sample-corrected one-sided conformal quantile of scores."""
    n = len(scores)
    lvl = min(1.0, np.ceil((n + 1) * level) / n)
    return np.quantile(scores, lvl)


def main():
    train, calib, test = load_and_split()
    train_X, calib_X, test_X = build_feature_matrix(train, calib, test)
    y_train = train["delay_days"].values
    y_calib = calib["delay_days"].values
    y_test  = test["delay_days"].values

    # === Step 1: train quantile models on the grid ===
    print("Training quantile model grid...")
    models = {}
    for q in QUANTILE_GRID:
        m = LGBMRegressor(objective="quantile", alpha=q, n_estimators=300,
                          random_state=RANDOM_SEED, verbosity=-1)
        m.fit(train_X, y_train)
        models[q] = m

    calib_preds = {q: np.clip(models[q].predict(calib_X), 0, None) for q in QUANTILE_GRID}
    test_preds  = {q: np.clip(models[q].predict(test_X),  0, None) for q in QUANTILE_GRID}

    # === Step 2: per-order critical ratio ===
    # rho = c_short / (c_short + c_hold). With costs proportional to the SAME
    # unit price, rho is constant here (0.02/0.0205 = 0.9756) -- so to make
    # the per-order dimension real (as it is in real firms, where holding and
    # shortage costs do NOT scale identically), we vary the shortage severity
    # by order priority: Critical orders hurt 2x more per late day, Normal 1x,
    # and add a small holding premium for bulky (high-quantity) orders.
    priority_mult = test["order_priority_or_type"].map(
        {"Normal": 1.0, "Urgent": 1.5, "Critical": 2.0}).fillna(1.0).values
    c_short = SHORTAGE_COST_RATE * test["unit_price"].values * priority_mult
    bulk_mult = np.where(test["quantity"].values > test["quantity"].median(), 1.5, 1.0)
    c_hold  = HOLDING_COST_RATE * test["unit_price"].values * bulk_mult

    rho = c_short / (c_short + c_hold)
    rho = np.maximum(rho, SERVICE_FLOOR_BETA)   # service-level floor
    print(f"Critical ratios: min={rho.min():.4f}, median={np.median(rho):.4f}, max={rho.max():.4f}")

    # helper: pick the grid level >= each PO's rho (ceiling on the grid)
    grid = np.array(QUANTILE_GRID)
    def grid_level_for(r):
        idx = np.searchsorted(grid, r, side="left")
        idx = np.clip(idx, 0, len(grid) - 1)
        return grid[idx]
    po_level = grid_level_for(rho)

    # === Step 3: build the three policies ===
    test_groups  = volatility_group(test)
    calib_groups = volatility_group(calib)

    # (a) fixed 90% conformal buffer (the OLD approach, our baseline here)
    scores_90 = y_calib - calib_preds[0.90]
    Q90_global = conformal_correction(scores_90, 0.90)
    buffer_fixed90 = np.clip(test_preds[0.90] + Q90_global, 0, None)

    # (b) DCCB marginal: per-order level, ONE global correction per level
    buffer_dccb_marginal = np.zeros(len(test))
    for q in QUANTILE_GRID:
        mask = po_level == q
        if mask.sum() == 0: continue
        scores_q = y_calib - calib_preds[q]
        Qq = conformal_correction(scores_q, q)
        buffer_dccb_marginal[mask] = np.clip(test_preds[q][mask] + Qq, 0, None)

    # (c) DCCB Mondrian: per-order level AND per-volatility-group correction
    buffer_dccb_mondrian = np.zeros(len(test))
    for q in QUANTILE_GRID:
        for g in ["calm", "medium", "erratic"]:
            mask = (po_level == q) & (test_groups == g)
            if mask.sum() == 0: continue
            calib_mask = calib_groups == g
            scores_qg = y_calib[calib_mask] - calib_preds[q][calib_mask]
            Qqg = conformal_correction(scores_qg, q)
            buffer_dccb_mondrian[mask] = np.clip(test_preds[q][mask] + Qqg, 0, None)

    # === Step 4: evaluate all three ===
    policies = {
        "fixed90_conformal": buffer_fixed90,
        "dccb_marginal":     buffer_dccb_marginal,
        "dccb_mondrian":     buffer_dccb_mondrian,
    }

    policy_rows, group_rows = [], []
    buffers_out = test[["po_id","supplier_id","order_date","delay_days","unit_price",
                         "order_priority_or_type"]].copy()
    buffers_out["volatility_group"] = test_groups
    buffers_out["critical_ratio"] = rho

    for name, buf in policies.items():
        realized_cost = c_hold * buf + c_short * np.maximum(0, y_test - buf)
        covered = (y_test <= buf)
        policy_rows.append({
            "policy": name,
            "mean_buffer_days": buf.mean(),
            "mean_realized_cost": realized_cost.mean(),
            "overall_coverage": covered.mean(),
            "shortage_event_rate": 1 - covered.mean(),
        })
        buffers_out[f"buffer__{name}"] = buf
        buffers_out[f"covered__{name}"] = covered.astype(int)

        for g in ["calm", "medium", "erratic"]:
            mask = test_groups == g
            group_rows.append({
                "policy": name, "group": g, "n": int(mask.sum()),
                "coverage": covered[mask].mean(),
                "mean_buffer_days": buf[mask].mean(),
                "mean_realized_cost": realized_cost[mask].mean(),
            })

    policy_df = pd.DataFrame(policy_rows)
    group_df = pd.DataFrame(group_rows)
    policy_df.to_csv(RESULTS_DIR/"dccb_policy_comparison.csv", index=False)
    group_df.to_csv(RESULTS_DIR/"dccb_group_coverage.csv", index=False)
    buffers_out.to_csv(RESULTS_DIR/"dccb_buffers_test.csv", index=False)

    print("\n=== Overall policy comparison (test fold) ===")
    print(policy_df.to_string(index=False))
    print("\n=== Coverage BY SUPPLIER VOLATILITY GROUP (the fairness table) ===")
    print(group_df.pivot(index="group", columns="policy", values="coverage").round(4).to_string())
    print("\n=== Mean buffer days by group ===")
    print(group_df.pivot(index="group", columns="policy", values="mean_buffer_days").round(2).to_string())
    print("\nSaved: dccb_policy_comparison.csv, dccb_group_coverage.csv, dccb_buffers_test.csv")


if __name__ == "__main__":
    main()
