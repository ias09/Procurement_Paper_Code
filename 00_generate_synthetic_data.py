"""
generate_synthetic_data.py
============================
Rebuilds the synthetic ERP-style procurement dataset with a REAL causal
structure linking features to delay, instead of near-random noise (v1 had
correlations near 0 for every feature except a weak supplier effect -- see
diagnosis in chat). This generator bakes in a known, interpretable data-
generating process so that:
  (a) the downstream ML pipeline has real signal to learn,
  (b) we know the "ground truth" effects, useful for sanity-checking SHAP
      results later in 06_explainability.py (explanations should roughly
      recover these known effects).

REPRODUCIBILITY: fixed RANDOM_SEED. Re-running this script regenerates an
identical dataset byte-for-byte.

Causal structure (all effects in days, added together then noise added):
  - supplier_base_tendency:   each of 40 suppliers has a fixed base delay
                               tendency, ~N(0, 3.5) days -- wide spread so
                               supplier identity is a strong predictor
  - supplier_trend:           each supplier drifts (improving/worsening)
                               linearly over the 2-year window -- creates
                               REAL signal for the reliability-trajectory
                               features in 02_feature_engineering.py
  - category_complexity:      10 material categories, each with a fixed
                               complexity effect (e.g. Electronics/Machinery
                               harder to deliver on time than Packaging)
  - quantity_effect:          larger orders -> modestly higher delay risk
  - tight_lead_time_effect:   promised lead time shorter than the category's
                               typical lead time -> higher delay risk
  - urgency_effect:           Urgent/Critical orders get expedited
                               (negative effect, i.e. less delay) -- models
                               a buyer pushing harder on rush orders
  - workload_congestion:      more concurrent orders for the same supplier
                               in the trailing 30 days -> more delay risk
  - seasonal_effect:          Q4 (Oct-Dec) congestion -> more delay risk
  - noise:                    remaining randomness, kept moderate so the
                               signal isn't drowned out, but real-world
                               unpredictability remains (this is NOT a
                               deterministic system)

Output: data/synthetic_raw.xlsx (overwrites v1; v1 backed up as
        data/synthetic_raw_v1_weak_signal.xlsx for reference)
"""

import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

N_ORDERS = 10_000
N_SUPPLIERS = 40
N_MATERIALS = 25
N_PLANTS = 5
N_BUYERS = 10
START_DATE = pd.Timestamp("2023-01-01")
END_DATE = pd.Timestamp("2024-12-30")

CATEGORIES = ["Packaging", "Plastic", "Steel", "Glass", "Machinery",
              "Chemicals", "Aluminum", "Textile", "Electronics", "Rubber"]
# Fixed, interpretable complexity effect per category (days added to delay tendency)
CATEGORY_COMPLEXITY = {
    "Packaging": -1.5, "Plastic": -0.5, "Textile": -0.8, "Rubber": -0.3,
    "Aluminum": 0.5, "Glass": 0.8, "Chemicals": 1.2, "Steel": 1.0,
    "Machinery": 2.5, "Electronics": 2.0,
}
# Typical promised lead time per category (days) -- used to define "tight" lead times
CATEGORY_TYPICAL_LEAD = {
    "Packaging": 7, "Plastic": 10, "Textile": 10, "Rubber": 10,
    "Aluminum": 14, "Glass": 14, "Chemicals": 14, "Steel": 18,
    "Machinery": 25, "Electronics": 21,
}

PRIORITIES = ["Normal", "Urgent", "Critical"]


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    # --- Entity setup ---
    supplier_ids = [f"S{str(i+1).zfill(2)}" for i in range(N_SUPPLIERS)]
    material_ids = [f"M{str(i+1).zfill(2)}" for i in range(N_MATERIALS)]
    material_category_map = {m: CATEGORIES[i % len(CATEGORIES)] for i, m in enumerate(material_ids)}
    plant_ids = [f"P{i+1}" for i in range(N_PLANTS)]
    buyer_ids = [f"B{i+1}" for i in range(N_BUYERS)]

    # Wide supplier base-tendency spread (this is what makes supplier_id a strong predictor)
    supplier_base_tendency = {s: rng.normal(0, 3.5) for s in supplier_ids}
    # Linear drift per supplier over the ~730-day window (some improving, some worsening)
    total_days = (END_DATE - START_DATE).days
    supplier_trend_per_day = {s: rng.normal(0, 0.006) for s in supplier_ids}  # up to ~+/-4.4 days drift over 2 yrs
    # Per-supplier noise volatility (some suppliers more erratic than others)
    supplier_volatility = {s: rng.uniform(2.0, 5.0) for s in supplier_ids}

    # --- Generate order dates first (needed for workload congestion calc) ---
    order_dates = pd.to_datetime(
        START_DATE + pd.to_timedelta(rng.integers(0, total_days + 1, size=N_ORDERS), unit="D")
    )
    order_supplier = rng.choice(supplier_ids, size=N_ORDERS)
    order_material = rng.choice(material_ids, size=N_ORDERS)
    order_category = np.array([material_category_map[m] for m in order_material])
    order_plant = rng.choice(plant_ids, size=N_ORDERS)
    order_buyer = rng.choice(buyer_ids, size=N_ORDERS)
    order_priority = rng.choice(PRIORITIES, size=N_ORDERS, p=[0.6, 0.3, 0.1])
    order_quantity = rng.integers(10, 500, size=N_ORDERS)
    order_unit_price = np.round(rng.uniform(5, 500, size=N_ORDERS), 2)
    order_total_spend = np.round(order_quantity * order_unit_price, 2)

    # Promised lead time: category-typical +/- noise, tightened sometimes for urgent orders
    typical_lead = np.array([CATEGORY_TYPICAL_LEAD[c] for c in order_category])
    lead_noise = rng.integers(-3, 4, size=N_ORDERS)
    urgent_tighten = np.where(order_priority == "Critical", -4,
                       np.where(order_priority == "Urgent", -2, 0))
    promised_lead_time = np.clip(typical_lead + lead_noise + urgent_tighten, 3, None)

    df = pd.DataFrame({
        "order_id": [f"PO{100000+i}" for i in range(N_ORDERS)],
        "supplier_id": order_supplier,
        "material_id": order_material,
        "material_category": order_category,
        "order_date": order_dates,
        "plant_id": order_plant,
        "buyer_id": order_buyer,
        "order_priority": order_priority,
        "quantity": order_quantity,
        "unit_price": order_unit_price,
        "total_spend": order_total_spend,
        "promised_lead_time": promised_lead_time,
    })
    df["promised_delivery_date"] = df["order_date"] + pd.to_timedelta(df["promised_lead_time"], unit="D")

    # --- Workload congestion: count of same-supplier orders within the trailing 30 days ---
    df = df.sort_values(["supplier_id", "order_date"]).reset_index(drop=True)
    workload = np.zeros(len(df))
    for supplier, group in df.groupby("supplier_id"):
        dates = group["order_date"].values
        idx = group.index.values
        for pos, i in enumerate(idx):
            window_start = dates[pos] - np.timedelta64(30, "D")
            count = np.sum((dates >= window_start) & (dates < dates[pos]))
            workload[i] = count
    df["_workload_30d"] = workload

    # --- Compute the true delay-generating predictor ---
    day_index = (df["order_date"] - START_DATE).dt.days.values

    base = np.array([supplier_base_tendency[s] for s in df["supplier_id"]])
    trend = np.array([supplier_trend_per_day[s] for s in df["supplier_id"]]) * day_index
    category_fx = np.array([CATEGORY_COMPLEXITY[c] for c in df["material_category"]])
    quantity_fx = (df["quantity"].values / 100.0) * 0.35
    typical_lead_arr = np.array([CATEGORY_TYPICAL_LEAD[c] for c in df["material_category"]])
    tight_lead_fx = np.clip(typical_lead_arr - df["promised_lead_time"].values, 0, None) * 0.4
    urgency_fx = np.where(df["order_priority"] == "Critical", -2.5,
                  np.where(df["order_priority"] == "Urgent", -1.2, 0.0))
    workload_fx = df["_workload_30d"].values * 0.18
    seasonal_fx = np.where(df["order_date"].dt.month.isin([10, 11, 12]), 1.5, 0.0)

    volatility = np.array([supplier_volatility[s] for s in df["supplier_id"]])
    noise = rng.normal(0, 1, size=len(df)) * volatility

    predictor = (base + trend + category_fx + quantity_fx + tight_lead_fx
                 + urgency_fx + workload_fx + seasonal_fx)

    # Recalibrate intercept so late rate lands near the proposal's target (~30%),
    # WITHOUT changing the relative strength of any effect above (we only shift
    # the whole predictor down by a constant, found via a small search).
    def late_rate_for_shift(shift):
        delay_raw_trial = (predictor - shift) + noise
        actual_lead_trial = np.clip(df["promised_lead_time"].values + delay_raw_trial, 1, None).round()
        delay_trial = actual_lead_trial - df["promised_lead_time"].values
        return (np.clip(delay_trial, 0, None) > 0).mean()

    lo, hi = 0.0, 15.0
    for _ in range(40):
        mid = (lo + hi) / 2
        rate = late_rate_for_shift(mid)
        if rate > 0.30:
            lo = mid
        else:
            hi = mid
    intercept_shift = (lo + hi) / 2
    print(f"Calibrated intercept shift: {intercept_shift:.3f} days (targeting ~30% late rate)")

    predictor = predictor - intercept_shift
    delay_raw = predictor + noise

    df["actual_lead_time"] = np.clip(
        df["promised_lead_time"].values + delay_raw, 1, None
    ).round().astype(int)
    df["actual_delivery_date"] = df["order_date"] + pd.to_timedelta(df["actual_lead_time"], unit="D")

    df["delay"] = df["actual_lead_time"] - df["promised_lead_time"]
    df["delay_days"] = df["delay"].clip(lower=0)
    df["late"] = (df["delay_days"] > 0).astype(int)

    df = df.drop(columns=["_workload_30d"])
    df = df.sort_values("order_date").reset_index(drop=True)

    print(f"Generated {len(df)} POs | late rate: {df['late'].mean():.4f} | "
          f"date range: {df['order_date'].min()} -> {df['order_date'].max()}")

    # quick signal check before saving
    print("\nQuick signal check (should now be well above v1's ~0.01-0.18):")
    print("corr(quantity, delay_days):", round(df["quantity"].corr(df["delay_days"]), 3))
    print("corr(promised_lead_time, delay_days):", round(df["promised_lead_time"].corr(df["delay_days"]), 3))
    sup_oracle_mean = df.groupby("supplier_id")["late"].transform("mean")
    print("corr(supplier oracle late-rate, late):", round(sup_oracle_mean.corr(df["late"]), 3))
    cat_oracle_mean = df.groupby("material_category")["late"].transform("mean")
    print("corr(category oracle late-rate, late):", round(cat_oracle_mean.corr(df["late"]), 3))

    # backup v1 if present and not already backed up
    v1_path = DATA_DIR / "synthetic_raw.xlsx"
    v1_backup = DATA_DIR / "synthetic_raw_v1_weak_signal.xlsx"
    if v1_path.exists() and not v1_backup.exists():
        v1_path.rename(v1_backup)
        print(f"\nBacked up old (weak-signal) dataset to {v1_backup.name}")

    df.to_excel(DATA_DIR / "synthetic_raw.xlsx", index=False)
    print(f"Saved new dataset to {DATA_DIR / 'synthetic_raw.xlsx'}")


if __name__ == "__main__":
    main()
