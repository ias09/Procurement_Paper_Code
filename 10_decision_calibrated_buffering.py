import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")

RANDOM_SEED        = 42
HOLDING_COST_RATE  = 0.0005
SHORTAGE_COST_RATE = 0.02
SERVICE_FLOOR_BETA = 0.90

DATA_DIR    = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

TRAIN_END   = pd.Timestamp("2023-12-31")
CALIB_START = pd.Timestamp("2024-01-01"); CALIB_END = pd.Timestamp("2024-03-31")
TEST_START  = pd.Timestamp("2024-04-01"); TEST_END  = pd.Timestamp("2024-06-30")

NON_FEATURE_COLUMNS = [
    "po_id", "order_date", "promised_delivery_date", "actual_delivery_date",
    "actual_lead_time", "delay", "delay_days", "late",
]
CATEGORICAL_CANDIDATES = [
    "supplier_id", "material_or_category", "order_priority_or_type",
    "plant_id", "buyer_id",
]

QUANTILE_GRID = [0.80, 0.85, 0.90, 0.925, 0.95, 0.965, 0.975, 0.985]
N_GROUPS_GRID = [2, 3, 4, 5]    


def load_and_split():
    df    = pd.read_csv(DATA_DIR / "synthetic_features.csv",
                        parse_dates=["order_date", "promised_delivery_date",
                                     "actual_delivery_date"])
    train = df[df["order_date"] <= TRAIN_END].copy()
    calib = df[(df["order_date"] >= CALIB_START) &
               (df["order_date"] <= CALIB_END)].copy()
    test  = df[(df["order_date"] >= TEST_START) &
               (df["order_date"] <= TEST_END)].copy()
    return train, calib, test


def build_feature_matrix(train, *frames):
    cat_cols  = [c for c in CATEGORICAL_CANDIDATES if c in train.columns]
    feat_cols = [c for c in train.columns if c not in NON_FEATURE_COLUMNS]
    num_cols  = [c for c in feat_cols if c not in cat_cols]

    train_enc = pd.get_dummies(train[feat_cols], columns=cat_cols, dummy_na=False)
    med       = train[num_cols].median()
    train_enc[num_cols] = train[num_cols].fillna(med)

    out = [train_enc]
    for f in frames:
        enc = pd.get_dummies(f[feat_cols], columns=cat_cols, dummy_na=False)
        enc[num_cols] = f[num_cols].fillna(med)
        enc = enc.reindex(columns=train_enc.columns, fill_value=0)
        out.append(enc)
    return out


def assign_volatility_group(frame, n_groups=3):
  
    vol  = frame["supplier_delay_std_90"].fillna(
        frame["supplier_delay_std_90"].median())
    quantiles = np.linspace(0, 1, n_groups + 1)
    cuts      = vol.quantile(quantiles[1:-1]).values   

    if n_groups == 2:
        labels = ["calm", "erratic"]
    elif n_groups == 3:
        labels = ["calm", "medium", "erratic"]
    elif n_groups == 4:
        labels = ["calm", "medium_low", "medium_high", "erratic"]
    else:
        labels = [f"g{i+1}" for i in range(n_groups)]

    band_idx = np.digitize(vol.values, cuts, right=True)
    groups   = np.array(labels)[band_idx]
    return groups


def conformal_correction(scores, level):
    n   = len(scores)
    lv  = min(1.0, np.ceil((n + 1) * level) / n)
    return np.quantile(scores, lv)


def run_dccb_mondrian(test, calib,
                      test_X, calib_X,
                      models, calib_preds, test_preds,
                      c_hold, c_short, y_test,
                      po_level, grid,
                      n_groups=3):
  
    test_groups  = assign_volatility_group(test,  n_groups=n_groups)
    calib_groups = assign_volatility_group(calib, n_groups=n_groups)
    group_names  = np.unique(test_groups)

    y_calib = calib["delay_days"].values
    buffer  = np.zeros(len(test))

    for q in grid:
        for g in group_names:
            mask       = (po_level == q) & (test_groups == g)
            if mask.sum() == 0:
                continue
            calib_mask = calib_groups == g
            scores_qg  = y_calib[calib_mask] - calib_preds[q][calib_mask]
            Qqg        = conformal_correction(scores_qg, q)
            buffer[mask] = np.clip(test_preds[q][mask] + Qqg, 0, None)

    covered = (y_test <= buffer)
    return buffer, covered, test_groups


def main():
    train, calib, test = load_and_split()
    train_X, calib_X, test_X = build_feature_matrix(train, calib, test)
    y_train = train["delay_days"].values
    y_calib = calib["delay_days"].values
    y_test  = test["delay_days"].values

    print("Training quantile model grid...")
    models      = {}
    calib_preds = {}
    test_preds  = {}
    for q in QUANTILE_GRID:
        m = LGBMRegressor(objective="quantile", alpha=q,
                          n_estimators=300, random_state=RANDOM_SEED,
                          verbosity=-1)
        m.fit(train_X, y_train)
        models[q]      = m
        calib_preds[q] = np.clip(m.predict(calib_X), 0, None)
        test_preds[q]  = np.clip(m.predict(test_X),  0, None)

    priority_mult = test["order_priority_or_type"].map(
        {"Normal": 1.0, "Urgent": 1.5, "Critical": 2.0}).fillna(1.0).values
    c_short = SHORTAGE_COST_RATE * test["unit_price"].values * priority_mult
    bulk_m  = np.where(test["quantity"].values > test["quantity"].median(),
                       1.5, 1.0)
    c_hold  = HOLDING_COST_RATE * test["unit_price"].values * bulk_m

    rho     = c_short / (c_short + c_hold)
    rho     = np.maximum(rho, SERVICE_FLOOR_BETA)
    print(f"Critical ratios: min={rho.min():.4f}  "
          f"median={np.median(rho):.4f}  max={rho.max():.4f}")

    grid_arr = np.array(QUANTILE_GRID)
    def grid_level_for(r):
        idx = np.searchsorted(grid_arr, r, side="left")
        return grid_arr[np.clip(idx, 0, len(grid_arr) - 1)]
    po_level = grid_level_for(rho)

    scores_90    = y_calib - calib_preds[0.90]
    Q90_global   = conformal_correction(scores_90, 0.90)
    buf_fixed90  = np.clip(test_preds[0.90] + Q90_global, 0, None)

    buf_dccb_m   = np.zeros(len(test))
    for q in QUANTILE_GRID:
        mask   = po_level == q
        if mask.sum() == 0:
            continue
        scores_q = y_calib - calib_preds[q]
        Qq       = conformal_correction(scores_q, q)
        buf_dccb_m[mask] = np.clip(test_preds[q][mask] + Qq, 0, None)

    buf_dccb_mond, _, test_groups_3 = run_dccb_mondrian(
        test, calib, test_X, calib_X,
        models, calib_preds, test_preds,
        c_hold, c_short, y_test, po_level, QUANTILE_GRID, n_groups=3)

    policies = {
        "fixed90_conformal": buf_fixed90,
        "dccb_marginal":     buf_dccb_m,
        "dccb_mondrian":     buf_dccb_mond,
    }

    policy_rows = []
    group_rows  = []
    buffers_out = test[["po_id", "supplier_id", "order_date", "delay_days",
                         "unit_price", "order_priority_or_type"]].copy()
    buffers_out["volatility_group"] = test_groups_3
    buffers_out["critical_ratio"]   = rho

    for name, buf in policies.items():
        realized_cost = c_hold * buf + c_short * np.maximum(0, y_test - buf)
        covered       = (y_test <= buf)
        policy_rows.append({
            "policy":              name,
            "mean_buffer_days":    buf.mean(),
            "mean_realized_cost":  realized_cost.mean(),
            "overall_coverage":    covered.mean(),
            "shortage_event_rate": 1 - covered.mean(),
        })
        buffers_out[f"buffer__{name}"]  = buf
        buffers_out[f"covered__{name}"] = covered.astype(int)

        for g in ["calm", "medium", "erratic"]:
            mask = test_groups_3 == g
            if mask.sum() == 0:
                continue
            group_rows.append({
                "policy":             name,
                "group":              g,
                "n":                  int(mask.sum()),
                "coverage":           covered[mask].mean(),
                "mean_buffer_days":   buf[mask].mean(),
                "mean_realized_cost": realized_cost[mask].mean(), 
            })

    policy_df = pd.DataFrame(policy_rows)
    group_df  = pd.DataFrame(group_rows)

    policy_df.to_csv(RESULTS_DIR / "dccb_policy_comparison.csv", index=False)
    group_df.to_csv(RESULTS_DIR / "dccb_group_coverage.csv",     index=False)
    buffers_out.to_csv(RESULTS_DIR / "dccb_buffers_test.csv",    index=False)

    print("\n=== Overall policy comparison ===")
    print(policy_df.to_string(index=False))
    print("\n=== Coverage by supplier volatility group ===")
    print(group_df.pivot(index="group", columns="policy",
                          values="coverage").round(4).to_string())
    print("\n=== Mean buffer days by group ===")
    print(group_df.pivot(index="group", columns="policy",
                          values="mean_buffer_days").round(2).to_string())
    print("\n=== Mean realized cost by group (NEW) ===")
    print(group_df.pivot(index="group", columns="policy",
                          values="mean_realized_cost").round(4).to_string())

    print(f"\nMondrian group-count sensitivity over n_groups={N_GROUPS_GRID}...")
    sens_rows = []
    for ng in N_GROUPS_GRID:
        buf_ng, cov_ng, grp_ng = run_dccb_mondrian(
            test, calib, test_X, calib_X,
            models, calib_preds, test_preds,
            c_hold, c_short, y_test, po_level, QUANTILE_GRID, n_groups=ng)

        rc_ng = c_hold * buf_ng + c_short * np.maximum(0, y_test - buf_ng)

        group_covs = []
        for g in np.unique(grp_ng):
            mask = grp_ng == g
            if mask.sum() > 0:
                group_covs.append(cov_ng[mask].mean())
        coverage_gap = max(group_covs) - min(group_covs) if group_covs else np.nan

        sens_rows.append({
            "n_groups":           ng,
            "overall_coverage":   cov_ng.mean(),
            "max_coverage_gap":   coverage_gap,
            "mean_buffer_days":   buf_ng.mean(),
            "mean_realized_cost": rc_ng.mean(),
        })
        print(f"  n_groups={ng} | overall_cov={cov_ng.mean():.4f} | "
              f"max_gap={coverage_gap:.4f} | "
              f"mean_buf={buf_ng.mean():.2f}d | "
              f"mean_cost={rc_ng.mean():.4f}")

    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(RESULTS_DIR / "mondrian_groupcount_sensitivity.csv",
                   index=False)

    print("\n=== Mondrian group-count sensitivity ===")
    print(sens_df.to_string(index=False))

    print("\nSaved:")
    print("  dccb_policy_comparison.csv")
    print("  dccb_group_coverage.csv             (+ mean_realized_cost per group)")
    print("  dccb_buffers_test.csv")
    print("  mondrian_groupcount_sensitivity.csv  (NEW)")


if __name__ == "__main__":
    main()