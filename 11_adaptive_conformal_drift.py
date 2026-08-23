import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")

RANDOM_SEED       = 42
ALPHA_TARGET      = 0.10
GAMMA             = 0.05         
GAMMA_GRID        = [0.01, 0.02, 0.05, 0.10, 0.20]   
N_STRESS_SUPPLIERS = 10
STRESS_START      = pd.Timestamp("2024-07-01")
STRESS_RATE       = 0.04          
ROLL_WINDOW       = 200           
RECOVERY_TOL      = 0.05          

DATA_DIR    = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

TRAIN_END   = pd.Timestamp("2023-12-31")
CALIB_START = pd.Timestamp("2024-01-01")
CALIB_END   = pd.Timestamp("2024-03-31")
EVAL_START  = pd.Timestamp("2024-04-01")

NON_FEATURE_COLUMNS = [
    "po_id", "order_date", "promised_delivery_date", "actual_delivery_date",
    "actual_lead_time", "delay", "delay_days", "late",
]
CATEGORICAL_CANDIDATES = [
    "supplier_id", "material_or_category", "order_priority_or_type",
    "plant_id", "buyer_id",
]


def load_and_split():
    df    = pd.read_csv(DATA_DIR / "synthetic_features.csv",
                        parse_dates=["order_date", "promised_delivery_date",
                                     "actual_delivery_date"])
    train = df[df["order_date"] <= TRAIN_END].copy()
    calib = df[(df["order_date"] >= CALIB_START) &
               (df["order_date"] <= CALIB_END)].copy()
    evalf = df[df["order_date"] >= EVAL_START].copy()
    evalf = evalf.sort_values(["order_date", "po_id"]).reset_index(drop=True)
    return train, calib, evalf


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


def run_scenario(evalf, base_pred, calib_scores, delays, label, gamma=GAMMA):

    n         = len(evalf)
    suppliers = evalf["supplier_id"].values

    # STATIC
    n_c      = len(calib_scores)
    lvl      = min(1.0, np.ceil((n_c + 1) * (1 - ALPHA_TARGET)) / n_c)
    Q_static = np.quantile(calib_scores, lvl)
    buf_s    = np.clip(base_pred + Q_static, 0, None)
    cov_s    = (delays <= buf_s)

    # ACI
    sorted_scores = np.sort(calib_scores)
    alpha_state   = {s: ALPHA_TARGET for s in np.unique(suppliers)}
    buf_a         = np.zeros(n)
    cov_a         = np.zeros(n, dtype=bool)

    def q_at(alpha):
        a   = float(np.clip(alpha, 0.001, 0.5))
        lv  = min(1.0, np.ceil((n_c + 1) * (1 - a)) / n_c)
        return np.quantile(sorted_scores, lv)

    for i in range(n):
        s       = suppliers[i]
        a       = alpha_state[s]
        buf_a[i] = max(0.0, base_pred[i] + q_at(a))
        miss    = 1.0 if delays[i] > buf_a[i] else 0.0
        cov_a[i] = (miss == 0.0)
        alpha_state[s] = float(
            np.clip(a + gamma * (ALPHA_TARGET - miss), 0.001, 0.5))

    tl = pd.DataFrame({
        "scenario":       label,
        "order_index":    np.arange(n),
        "order_date":     evalf["order_date"].values,
        "supplier_id":    suppliers,
        "delay_days":     delays,
        "buffer_static":  buf_s,
        "covered_static": cov_s.astype(int),
        "buffer_aci":     buf_a,
        "covered_aci":    cov_a.astype(int),
    })
    tl["rolling_cov_static"] = (tl["covered_static"]
                                 .rolling(ROLL_WINDOW, min_periods=50).mean())
    tl["rolling_cov_aci"]    = (tl["covered_aci"]
                                 .rolling(ROLL_WINDOW, min_periods=50).mean())

    summary = {
        "scenario":              label,
        "gamma":                 gamma,
        "overall_coverage_static": cov_s.mean(),
        "overall_coverage_aci":    cov_a.mean(),
        "target_coverage":         1 - ALPHA_TARGET,
        "mean_buffer_static":      buf_s.mean(),
        "mean_buffer_aci":         buf_a.mean(),
        "min_rolling_cov_static":  tl["rolling_cov_static"].min(),
        "min_rolling_cov_aci":     tl["rolling_cov_aci"].min(),
    }
    return tl, summary


def compute_recovery_speed(timeline_B, stress_suppliers, evalf):
    
    recovery_target = (1 - ALPHA_TARGET) - RECOVERY_TOL  
    rows = []
    stress_start_idx = evalf.index[evalf["order_date"] >= STRESS_START]
    if len(stress_start_idx) == 0:
        return pd.DataFrame()
    first_stress_idx = stress_start_idx[0]

    for sup in stress_suppliers:
       
        sub = timeline_B[
            (timeline_B["supplier_id"] == sup) &
            (timeline_B["order_index"] >= first_stress_idx)
        ].copy().reset_index(drop=True)

        if len(sub) < 10:
            rows.append({"supplier_id": sup, "n_orders_to_recovery": np.nan,
                         "n_orders_after_stress": len(sub),
                         "final_rolling_cov_aci": np.nan})
            continue

        sub["sup_rolling_aci"] = (sub["covered_aci"]
                                   .rolling(min(20, len(sub)), min_periods=5)
                                   .mean())
        recovered = sub[sub["sup_rolling_aci"] >= recovery_target]
        n_to_rec  = recovered.index[0] + 1 if len(recovered) > 0 else np.nan
        rows.append({
            "supplier_id":            sup,
            "n_orders_to_recovery":   n_to_rec,
            "n_orders_after_stress":  len(sub),
            "final_rolling_cov_aci":  sub["sup_rolling_aci"].iloc[-1],
        })

    return pd.DataFrame(rows)


def main():
    train, calib, evalf = load_and_split()
    train_X, calib_X, eval_X = build_feature_matrix(train, calib, evalf)

   
    model = LGBMRegressor(objective="quantile", alpha=1 - ALPHA_TARGET,
                           n_estimators=300, random_state=RANDOM_SEED,
                           verbosity=-1)
    model.fit(train_X, train["delay_days"].values)

    calib_scores = (calib["delay_days"].values
                    - np.clip(model.predict(calib_X), 0, None))
    base_pred    = np.clip(model.predict(eval_X), 0, None)

    rng = np.random.default_rng(RANDOM_SEED)
    stress_suppliers = rng.choice(evalf["supplier_id"].unique(),
                                   size=N_STRESS_SUPPLIERS, replace=False)
    days_after = (evalf["order_date"] - STRESS_START).dt.days.clip(lower=0).values
    is_stressed = evalf["supplier_id"].isin(stress_suppliers).values
    extra       = np.where(is_stressed, days_after * STRESS_RATE, 0.0)

    delays_A = evalf["delay_days"].values.astype(float)
    delays_B = delays_A + extra

    print(f"Stress scenario: {N_STRESS_SUPPLIERS} suppliers deteriorating "
          f"from {STRESS_START.date()} at +{STRESS_RATE} days/day | "
          f"max added delay {extra.max():.1f} days")


    print(f"\nPrimary ACI run (gamma={GAMMA})...")
    tl_A, sum_A = run_scenario(evalf, base_pred, calib_scores,
                                delays_A, "natural", gamma=GAMMA)
    tl_B, sum_B = run_scenario(evalf, base_pred, calib_scores,
                                delays_B, "stress",  gamma=GAMMA)

    pd.concat([tl_A, tl_B], ignore_index=True).to_csv(
        RESULTS_DIR / "aci_coverage_timeline.csv", index=False)
    pd.DataFrame([sum_A, sum_B]).to_csv(
        RESULTS_DIR / "aci_summary.csv", index=False)

    print("\n=== ACI Summary ===")
    print(pd.DataFrame([sum_A, sum_B]).to_string(index=False))

    print("\nComputing per-supplier recovery speed (Scenario B)...")
    recovery_df = compute_recovery_speed(tl_B, stress_suppliers, evalf)
    if not recovery_df.empty:
        recovery_df.to_csv(RESULTS_DIR / "aci_recovery_speed.csv", index=False)
        valid = recovery_df["n_orders_to_recovery"].dropna()
        print(f"  Orders to recovery -- "
              f"mean={valid.mean():.1f}  "
              f"median={valid.median():.1f}  "
              f"min={valid.min():.0f}  "
              f"max={valid.max():.0f}  "
              f"({len(valid)}/{N_STRESS_SUPPLIERS} suppliers recovered "
              f"within evaluation window)")
        agg = {
            "mean_orders_to_recovery":   valid.mean(),
            "median_orders_to_recovery": valid.median(),
            "min_orders_to_recovery":    valid.min(),
            "max_orders_to_recovery":    valid.max(),
            "pct_recovered":             len(valid) / N_STRESS_SUPPLIERS,
        }
        pd.DataFrame([agg]).to_csv(
            RESULTS_DIR / "aci_recovery_summary.csv", index=False)


    print(f"\nGamma sensitivity analysis over {GAMMA_GRID}...")
    sens_rows = []
    for g in GAMMA_GRID:
        
        tl_g, sum_g = run_scenario(evalf, base_pred, calib_scores,
                                   delays_B, f"stress_gamma_{g}", gamma=g)
        rec_g = compute_recovery_speed(tl_g, stress_suppliers, evalf)
        mean_rec = rec_g["n_orders_to_recovery"].dropna().mean() if not rec_g.empty else np.nan
        sens_rows.append({
            "gamma":                      g,
            "overall_coverage_aci":       sum_g["overall_coverage_aci"],
            "min_rolling_cov_aci":        sum_g["min_rolling_cov_aci"],
            "mean_buffer_aci_days":       sum_g["mean_buffer_aci"],
            "mean_orders_to_recovery":    mean_rec,
        })
        print(f"  gamma={g:.2f} | overall_cov={sum_g['overall_coverage_aci']:.4f} | "
              f"min_rolling={sum_g['min_rolling_cov_aci']:.4f} | "
              f"mean_buf={sum_g['mean_buffer_aci']:.2f}d | "
              f"orders_to_rec={mean_rec:.1f}")

    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(RESULTS_DIR / "aci_gamma_sensitivity.csv", index=False)

    print("\n=== Gamma Sensitivity Summary ===")
    print(sens_df.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, tl, title in [
        (axes[0], tl_A, "Scenario A: natural drift"),
        (axes[1], tl_B, f"Scenario B: stress "
                        f"({N_STRESS_SUPPLIERS} suppliers deteriorate "
                        f"from Jul 2024)"),
    ]:
        ax.plot(tl["order_index"], tl["rolling_cov_static"],
                label="Static margins", lw=1.8)
        ax.plot(tl["order_index"], tl["rolling_cov_aci"],
                label=f"ACI margins (gamma={GAMMA})", lw=1.8)
        ax.axhline(1 - ALPHA_TARGET, color="k", ls="--", lw=1,
                   label=f"Target {int((1-ALPHA_TARGET)*100)}%")
        if "Scenario B" in title:
            stress_idx = tl.index[tl["order_date"] >= STRESS_START]
            if len(stress_idx) > 0:
                ax.axvline(stress_idx[0], color="red", ls=":", lw=1.5,
                           label="Deterioration begins")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(f"Order sequence (rolling {ROLL_WINDOW}-order coverage)")
        ax.set_ylim(0.5, 1.02)
        ax.legend(fontsize=8, loc="lower left")
    axes[0].set_ylabel("Fraction of deliveries inside buffer")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "aci_coverage_plot.png", dpi=150)
    plt.close()

    print("\nSaved:")
    print("  aci_coverage_timeline.csv")
    print("  aci_summary.csv")
    print("  aci_coverage_plot.png")
    print("  aci_recovery_speed.csv      (NEW -- per-supplier orders to recovery)")
    print("  aci_recovery_summary.csv    (NEW -- aggregate recovery statistics)")
    print("  aci_gamma_sensitivity.csv   (NEW -- sensitivity to step-size choice)")


if __name__ == "__main__":
    main()