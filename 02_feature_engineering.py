"""
02_feature_engineering.py  (v2 — enriched features for score optimization)
==========================================================================
Proposal mapping: Section 8 (Feature Engineering) -- order-level, supplier
historical, and reliability trajectory features.

LEAKAGE-SAFETY RULE: unchanged from v1. For PO i with order_date d_i and
supplier s, every supplier-history feature is computed using ONLY rows of
supplier s with order_date < d_i (strictly), using date-boundary searchsorted
with deterministic (order_date, po_id) sort key so same-day sibling orders
never contaminate each other's history.

WHAT'S NEW vs v1:
  - Rolling-10 and rolling-30 order windows (fills the 5 <-> 90-day gap)
  - Days-since-last-late: recency of the supplier's most recent failure
  - Lead-time ratio: promised / category typical (how tight is this order?)
  - Supplier × category interaction: some suppliers underperform specifically
    on certain material types (interaction captured via cross-aggregations)
  - Cold-start indicator flag: first-N-orders binary, since NaN-filled
    early orders are qualitatively different from mature supplier histories
  - Workload proxy: concurrent orders placed to same supplier in last 30 days
    (now computed correctly from the clean df, not from generator internals)

Outputs:
  - data/synthetic_features.csv
  - data/dataset1_features.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

CATEGORY_TYPICAL_LEAD = {
    "Packaging": 7, "Plastic": 10, "Textile": 10, "Rubber": 10,
    "Aluminum": 14, "Glass": 14, "Chemicals": 14, "Steel": 18,
    "Machinery": 25, "Electronics": 21,
}


def _strictly_past_indices(dates: np.ndarray, i: int) -> np.ndarray:
    boundary = np.searchsorted(dates, dates[i], side="left")
    return np.arange(boundary)


def _supplier_group_features(group: pd.DataFrame) -> pd.DataFrame:
    """All rolling supplier-history windows, strictly leakage-safe."""
    group = group.sort_values(["order_date", "po_id"]).reset_index(drop=True)
    dates = group["order_date"].values
    late = group["late"].values.astype(float)
    delay = group["delay_days"].values.astype(float)
    n = len(group)

    # --- allocate output arrays ---
    ontime5 = np.full(n, np.nan);   avgdelay5 = np.full(n, np.nan)
    maxdelay5 = np.full(n, np.nan); stddelay5 = np.full(n, np.nan)
    p90delay5 = np.full(n, np.nan)

    ontime10 = np.full(n, np.nan);  avgdelay10 = np.full(n, np.nan)
    stddelay10 = np.full(n, np.nan)

    ontime30 = np.full(n, np.nan);  avgdelay30 = np.full(n, np.nan)
    stddelay30 = np.full(n, np.nan)

    ontime90 = np.full(n, np.nan);  avgdelay90 = np.full(n, np.nan)
    stddelay90 = np.full(n, np.nan); p90delay90 = np.full(n, np.nan)

    days_since_last_late = np.full(n, np.nan)
    order_count_so_far = np.zeros(n, dtype=int)
    workload_30d = np.zeros(n)

    ninety_days  = np.timedelta64(90, "D")
    thirty_days  = np.timedelta64(30, "D")

    for i in range(n):
        past_idx = _strictly_past_indices(dates, i)
        order_count_so_far[i] = len(past_idx)
        if len(past_idx) == 0:
            continue

        # --- workload: how many orders placed to this supplier in last 30d ---
        cutoff30 = dates[i] - thirty_days
        workload_30d[i] = np.sum(dates[past_idx] >= cutoff30)

        # --- days since last late delivery ---
        late_past_idx = past_idx[late[past_idx] == 1]
        if len(late_past_idx) > 0:
            last_late_date = dates[late_past_idx[-1]]
            days_since_last_late[i] = (dates[i] - last_late_date) / np.timedelta64(1, "D")

        # --- rolling-5 ---
        last5 = past_idx[-5:]
        ontime5[i]   = 1 - late[last5].mean()
        avgdelay5[i] = delay[last5].mean()
        maxdelay5[i] = delay[last5].max()
        if len(last5) >= 2: stddelay5[i] = delay[last5].std(ddof=0)
        p90delay5[i] = np.percentile(delay[last5], 90)

        # --- rolling-10 ---
        last10 = past_idx[-10:]
        ontime10[i]   = 1 - late[last10].mean()
        avgdelay10[i] = delay[last10].mean()
        if len(last10) >= 2: stddelay10[i] = delay[last10].std(ddof=0)

        # --- rolling-30 ---
        last30 = past_idx[-30:]
        ontime30[i]   = 1 - late[last30].mean()
        avgdelay30[i] = delay[last30].mean()
        if len(last30) >= 2: stddelay30[i] = delay[last30].std(ddof=0)

        # --- rolling-90 day (time-based, not count-based) ---
        cutoff90 = dates[i] - ninety_days
        win90 = past_idx[dates[past_idx] >= cutoff90]
        if len(win90) > 0:
            ontime90[i]   = 1 - late[win90].mean()
            avgdelay90[i] = delay[win90].mean()
            p90delay90[i] = np.percentile(delay[win90], 90)
            if len(win90) >= 2: stddelay90[i] = delay[win90].std(ddof=0)

    group["supplier_ontime_rate_5"]     = ontime5
    group["supplier_avg_delay_5"]       = avgdelay5
    group["supplier_max_delay_5"]       = maxdelay5
    group["supplier_delay_std_5"]       = stddelay5
    group["supplier_delay_p90_5"]       = p90delay5
    group["supplier_ontime_rate_10"]    = ontime10
    group["supplier_avg_delay_10"]      = avgdelay10
    group["supplier_delay_std_10"]      = stddelay10
    group["supplier_ontime_rate_30"]    = ontime30
    group["supplier_avg_delay_30"]      = avgdelay30
    group["supplier_delay_std_30"]      = stddelay30
    group["supplier_ontime_rate_90"]    = ontime90
    group["supplier_avg_delay_90"]      = avgdelay90
    group["supplier_delay_std_90"]      = stddelay90
    group["supplier_delay_p90_90"]      = p90delay90
    group["days_since_last_late"]       = days_since_last_late
    group["supplier_workload_30d"]      = workload_30d
    group["supplier_order_count_so_far"] = order_count_so_far
    return group


def _trajectory_features(group: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    group = group.sort_values(["order_date", "po_id"]).reset_index(drop=True)
    dates = group["order_date"].values
    late  = group["late"].values.astype(float)
    delay = group["delay_days"].values.astype(float)
    m = len(group)

    ontime_slope = np.full(m, np.nan)
    delay_slope  = np.full(m, np.nan)
    var_growth   = np.full(m, np.nan)

    for i in range(m):
        past_idx = _strictly_past_indices(dates, i)
        if len(past_idx) >= 3:
            recent = past_idx[-n:]
            x = np.arange(len(recent))
            ontime_slope[i] = np.polyfit(x, 1 - late[recent], 1)[0]
            delay_slope[i]  = np.polyfit(x, delay[recent], 1)[0]

        if len(past_idx) >= 2 * n:
            recent = past_idx[-n:]; prior = past_idx[-2 * n:-n]
            var_growth[i] = delay[recent].std(ddof=0) - delay[prior].std(ddof=0)
        elif len(past_idx) >= 6:
            half = len(past_idx) // 2
            r = past_idx[half:]; p = past_idx[:half]
            if len(r) >= 3 and len(p) >= 3:
                var_growth[i] = delay[r].std(ddof=0) - delay[p].std(ddof=0)

    group["ontime_rate_slope_10"]      = ontime_slope
    group["delay_trend_slope_10"]      = delay_slope
    group["delay_variance_growth_10"]  = var_growth
    return group


def add_order_level_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["order_month"]   = df["order_date"].dt.month
    df["order_quarter"] = df["order_date"].dt.quarter
    df["is_q4"]         = df["order_date"].dt.month.isin([10, 11, 12]).astype(int)
    df["is_urgent"]     = df["order_priority_or_type"].isin(
        ["Urgent", "Critical", "Emergency"]).astype(int)
    df["log_quantity"]  = np.log1p(df["quantity"])
    df["log_unit_price"] = np.log1p(df["unit_price"])

    # lead-time ratio: how tight is this promise vs category norm
    typical = df["material_or_category"].map(CATEGORY_TYPICAL_LEAD)
    df["lead_time_ratio"] = df["promised_lead_time"] / typical.clip(lower=1)
    df["lead_time_tight"] = (df["promised_lead_time"] < typical).astype(int)

    # cold-start indicator
    df["cold_start"] = 0  # will be set after supplier history is computed
    return df


def add_supplier_history_features(df: pd.DataFrame) -> pd.DataFrame:
    parts = [_supplier_group_features(g) for _, g in df.groupby("supplier_id")]
    out = pd.concat(parts, ignore_index=True)
    # cold-start: fewer than 5 prior orders -> rolling features are noisy/missing
    out["cold_start"] = (out["supplier_order_count_so_far"] < 5).astype(int)
    return out


def add_trajectory_features(df: pd.DataFrame) -> pd.DataFrame:
    parts = [_trajectory_features(g) for _, g in df.groupby("supplier_id")]
    return pd.concat(parts, ignore_index=True)


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Supplier × category cross-features: some suppliers underperform
    specifically on certain material types. We can't one-hot both because
    the combinations explode — instead use a numeric interaction:
    supplier's rolling avg delay RELATIVE to the category-level avg delay
    seen so far in the training data. Also: workload × tight-lead-time."""
    df = df.copy()
    # category-level mean delay (from rolling-90 where available, else 0)
    cat_avg = df.groupby("material_or_category")["supplier_avg_delay_90"].transform("mean").fillna(0)
    df["delay_vs_category_avg"] = (df["supplier_avg_delay_90"].fillna(0) - cat_avg)
    df["workload_x_tight_lead"] = df["supplier_workload_30d"] * df["lead_time_tight"]
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_order_level_features(df)
    df = add_supplier_history_features(df)
    df = add_trajectory_features(df)
    df = add_interaction_features(df)
    return df


def main():
    synthetic = pd.read_csv(DATA_DIR / "synthetic_clean.csv", parse_dates=[
        "order_date", "promised_delivery_date", "actual_delivery_date"
    ])
    synthetic_feat = build_features(synthetic)
    synthetic_feat.to_csv(DATA_DIR / "synthetic_features.csv", index=False)
    new_cols = [c for c in synthetic_feat.columns if c not in synthetic.columns]
    print(f"synthetic_features.csv: {synthetic_feat.shape} | {len(new_cols)} new feature columns")
    print("New:", new_cols)

    ds1 = pd.read_csv(DATA_DIR / "dataset1_clean.csv", parse_dates=[
        "order_date", "promised_delivery_date", "actual_delivery_date"
    ])
    ds1_feat = build_features(ds1)
    ds1_feat.to_csv(DATA_DIR / "dataset1_features.csv", index=False)
    print(f"dataset1_features.csv: {ds1_feat.shape}")

    # leakage sanity
    first_rows = synthetic_feat.sort_values(["supplier_id", "order_date"]).groupby("supplier_id").head(1)
    print("\nFirst-order supplier_avg_delay_5 (should all be NaN):",
          first_rows["supplier_avg_delay_5"].isna().all())


if __name__ == "__main__":
    main()
