"""
11_adaptive_conformal_drift.py
================================
PLAIN-LANGUAGE PURPOSE (Idea 1 from the updated proposal -- the headline
experiment):

Suppliers change over time. A supplier that was reliable when we calibrated
our margins can start deteriorating (capacity loss, financial trouble). The
OLD system keeps using the margins calibrated on the past -- and quietly
starts failing: deliveries land outside the margin more and more often, and
nobody notices because the system never re-checks itself.

THE FIX ("self-correcting margins", technically Adaptive Conformal Inference,
Gibbs & Candes 2021): after every delivery, check -- did it land inside the
margin? If it missed, make the next margin for that supplier a bit wider.
If it landed inside, relax a tiny bit. One line of math:

    alpha_next = alpha_now + gamma * (alpha_target - miss)

where miss = 1 if the delivery exceeded the buffer, else 0. Over time this
provably keeps the promised protection level NO MATTER HOW the supplier's
behavior changes.

THE EXPERIMENT (two scenarios):
  A. NATURAL: the synthetic data as-is (its generator already includes mild
     supplier drift). Compare static vs self-correcting margins over the
     evaluation period (2024-04 to 2024-12), tracking protection over time.
  B. STRESS: we simulate a sudden regime change -- from July 2024, ten
     suppliers start deteriorating (delays grow by +0.04 days per day,
     reaching ~+7 days by December). This is the "supplier in trouble"
     scenario. The static system should visibly collapse for these
     suppliers; the self-correcting one should dip and then recover.

WHAT TO LOOK AT IN THE OUTPUT:
  - rolling coverage curves (the paper's key figure): static line falls off
    a cliff in the stress scenario; adaptive line returns to target.
  - the price paid: adaptive margins get wider for the troubled suppliers
    (that's the point -- protection costs buffer days).

Outputs:
  - results/aci_coverage_timeline.csv    (rolling coverage, both scenarios)
  - results/aci_summary.csv              (overall metrics)
  - results/aci_coverage_plot.png        (the key figure)
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
ALPHA_TARGET = 0.10        # target: 90% of deliveries inside the buffer
GAMMA = 0.05               # ACI step size (how fast margins react)
N_STRESS_SUPPLIERS = 10    # how many suppliers deteriorate in scenario B
STRESS_START = pd.Timestamp("2024-07-01")
STRESS_RATE = 0.04         # extra delay days per calendar day after stress start
ROLL_WINDOW = 200          # rolling window (orders) for the coverage curves

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

TRAIN_END   = pd.Timestamp("2023-12-31")
CALIB_START = pd.Timestamp("2024-01-01"); CALIB_END = pd.Timestamp("2024-03-31")
EVAL_START  = pd.Timestamp("2024-04-01")   # test + fold2 together = Apr-Dec 2024

NON_FEATURE_COLUMNS = [
    "po_id","order_date","promised_delivery_date","actual_delivery_date",
    "actual_lead_time","delay","delay_days","late",
]
CATEGORICAL_CANDIDATES = ["supplier_id","material_or_category","order_priority_or_type","plant_id","buyer_id"]


def load_and_split():
    df = pd.read_csv(DATA_DIR/"synthetic_features.csv",
        parse_dates=["order_date","promised_delivery_date","actual_delivery_date"])
    train = df[df["order_date"] <= TRAIN_END].copy()
    calib = df[(df["order_date"] >= CALIB_START) & (df["order_date"] <= CALIB_END)].copy()
    evalf = df[df["order_date"] >= EVAL_START].copy()
    evalf = evalf.sort_values(["order_date", "po_id"]).reset_index(drop=True)
    return train, calib, evalf


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


def run_scenario(evalf, base_pred, calib_scores, delays, label):
    """Simulate the evaluation period chronologically twice:
    once with STATIC margins, once with SELF-CORRECTING (ACI) margins.

    base_pred:    model's raw 90th-percentile delay prediction per eval PO
    calib_scores: calibration nonconformity scores (y - pred), from which the
                  margin correction Q is looked up at any level
    delays:       realized delay days per eval PO (possibly stress-modified)
    """
    n = len(evalf)
    suppliers = evalf["supplier_id"].values

    # STATIC: one fixed correction from calibration, never updated
    n_c = len(calib_scores)
    lvl = min(1.0, np.ceil((n_c + 1) * (1 - ALPHA_TARGET)) / n_c)
    Q_static = np.quantile(calib_scores, lvl)
    buffer_static = np.clip(base_pred + Q_static, 0, None)
    covered_static = delays <= buffer_static

    # ADAPTIVE (ACI): per-supplier alpha_t, updated after every delivery
    alpha_state = {s: ALPHA_TARGET for s in np.unique(suppliers)}
    buffer_aci = np.zeros(n)
    covered_aci = np.zeros(n, dtype=bool)
    sorted_scores = np.sort(calib_scores)

    def q_at(alpha):
        """Margin correction at protection level 1-alpha (clipped to sane range)."""
        a = float(np.clip(alpha, 0.001, 0.5))
        lvl = min(1.0, np.ceil((n_c + 1) * (1 - a)) / n_c)
        return np.quantile(sorted_scores, lvl)

    for i in range(n):
        s = suppliers[i]
        a = alpha_state[s]
        buffer_aci[i] = max(0.0, base_pred[i] + q_at(a))
        miss = 1.0 if delays[i] > buffer_aci[i] else 0.0
        covered_aci[i] = miss == 0.0
        # the one-line self-correction:
        alpha_state[s] = float(np.clip(a + GAMMA * (ALPHA_TARGET - miss), 0.001, 0.5))

    timeline = pd.DataFrame({
        "scenario": label,
        "order_index": np.arange(n),
        "order_date": evalf["order_date"].values,
        "supplier_id": suppliers,
        "delay_days": delays,
        "buffer_static": buffer_static, "covered_static": covered_static.astype(int),
        "buffer_aci": buffer_aci,       "covered_aci": covered_aci.astype(int),
    })
    timeline["rolling_cov_static"] = timeline["covered_static"].rolling(ROLL_WINDOW, min_periods=50).mean()
    timeline["rolling_cov_aci"]    = timeline["covered_aci"].rolling(ROLL_WINDOW, min_periods=50).mean()

    summary = {
        "scenario": label,
        "overall_coverage_static": covered_static.mean(),
        "overall_coverage_aci": covered_aci.mean(),
        "target_coverage": 1 - ALPHA_TARGET,
        "mean_buffer_static": buffer_static.mean(),
        "mean_buffer_aci": buffer_aci.mean(),
        "min_rolling_cov_static": timeline["rolling_cov_static"].min(),
        "min_rolling_cov_aci": timeline["rolling_cov_aci"].min(),
    }
    return timeline, summary


def main():
    train, calib, evalf = load_and_split()
    train_X, calib_X, eval_X = build_feature_matrix(train, calib, evalf)

    # base quantile model (90th percentile of delay)
    model = LGBMRegressor(objective="quantile", alpha=1 - ALPHA_TARGET,
                          n_estimators=300, random_state=RANDOM_SEED, verbosity=-1)
    model.fit(train_X, train["delay_days"].values)

    calib_scores = calib["delay_days"].values - np.clip(model.predict(calib_X), 0, None)
    base_pred = np.clip(model.predict(eval_X), 0, None)

    # === Scenario A: NATURAL (data as generated, mild built-in drift) ===
    delays_A = evalf["delay_days"].values.astype(float)
    timeline_A, summary_A = run_scenario(evalf, base_pred, calib_scores, delays_A, "natural")

    # === Scenario B: STRESS (10 suppliers deteriorate from July 2024) ===
    rng = np.random.default_rng(RANDOM_SEED)
    stress_suppliers = rng.choice(evalf["supplier_id"].unique(), size=N_STRESS_SUPPLIERS, replace=False)
    days_after = (evalf["order_date"] - STRESS_START).dt.days.clip(lower=0).values
    is_stressed = evalf["supplier_id"].isin(stress_suppliers).values
    extra = np.where(is_stressed, days_after * STRESS_RATE, 0.0)
    delays_B = delays_A + extra
    print(f"Stress scenario: {N_STRESS_SUPPLIERS} suppliers deteriorate from {STRESS_START.date()}, "
          f"max added delay {extra.max():.1f} days")
    timeline_B, summary_B = run_scenario(evalf, base_pred, calib_scores, delays_B, "stress")

    # save
    pd.concat([timeline_A, timeline_B], ignore_index=True).to_csv(
        RESULTS_DIR/"aci_coverage_timeline.csv", index=False)
    summary_df = pd.DataFrame([summary_A, summary_B])
    summary_df.to_csv(RESULTS_DIR/"aci_summary.csv", index=False)

    # the key figure: rolling coverage, both scenarios
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, tl, title in [(axes[0], timeline_A, "Scenario A: natural drift"),
                           (axes[1], timeline_B, "Scenario B: stress (10 suppliers deteriorate from Jul 2024)")]:
        ax.plot(tl["order_index"], tl["rolling_cov_static"], label="Static margins (old)", lw=1.8)
        ax.plot(tl["order_index"], tl["rolling_cov_aci"], label="Self-correcting margins (ACI)", lw=1.8)
        ax.axhline(1 - ALPHA_TARGET, color="k", ls="--", lw=1, label="Target 90%")
        if title.startswith("Scenario B"):
            stress_idx = tl.index[tl["order_date"] >= STRESS_START]
            if len(stress_idx) > 0:
                ax.axvline(stress_idx[0], color="red", ls=":", lw=1.5, label="Deterioration begins")
        ax.set_title(title); ax.set_xlabel(f"Order sequence (rolling {ROLL_WINDOW}-order coverage)")
        ax.set_ylim(0.5, 1.02); ax.legend(fontsize=8, loc="lower left")
    axes[0].set_ylabel("Fraction of deliveries inside buffer")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR/"aci_coverage_plot.png", dpi=150)
    plt.close()

    print("\n=== SUMMARY ===")
    print(summary_df.to_string(index=False))
    print("\nSaved: aci_coverage_timeline.csv, aci_summary.csv, aci_coverage_plot.png")


if __name__ == "__main__":
    main()
