import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
ALPHA = 0.10
ALPHA_GRID = [0.05, 0.10, 0.15, 0.20]  

DATA_DIR    = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

TRAIN_END                  = pd.Timestamp("2023-12-31")
CALIB_START, CALIB_END     = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-03-31")
TEST_START,  TEST_END      = pd.Timestamp("2024-04-01"), pd.Timestamp("2024-06-30")
FOLD2_START                = pd.Timestamp("2024-07-01")

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
    test  = df[(df["order_date"] >= TEST_START)  & (df["order_date"] <= TEST_END)].copy()
    fold2 = df[df["order_date"] >= FOLD2_START].copy()
    return train, calib, test, fold2


def build_feature_matrix(train, *other_frames):
    categorical_cols = [c for c in CATEGORICAL_CANDIDATES if c in train.columns]
    feature_cols     = [c for c in train.columns if c not in NON_FEATURE_COLUMNS]
    numeric_cols     = [c for c in feature_cols if c not in categorical_cols]

    train_enc    = pd.get_dummies(train[feature_cols], columns=categorical_cols, dummy_na=False)
    train_medians = train[numeric_cols].median()
    train_enc[numeric_cols] = train[numeric_cols].fillna(train_medians)

    encoded_frames = [train_enc]
    for frame in other_frames:
        f_enc = pd.get_dummies(frame[feature_cols], columns=categorical_cols, dummy_na=False)
        f_enc[numeric_cols] = frame[numeric_cols].fillna(train_medians)
        f_enc = f_enc.reindex(columns=train_enc.columns, fill_value=0)
        encoded_frames.append(f_enc)

    return encoded_frames


def fit_cqr(train_X, y_train, calib_X, y_calib, alpha):
    """Fit CQR quantile models at given alpha and return calibrated Q."""
    lo = LGBMRegressor(objective="quantile", alpha=alpha / 2,
                       n_estimators=300, random_state=RANDOM_SEED, verbosity=-1)
    hi = LGBMRegressor(objective="quantile", alpha=1 - alpha / 2,
                       n_estimators=300, random_state=RANDOM_SEED, verbosity=-1)
    lo.fit(train_X, y_train)
    hi.fit(train_X, y_train)

    q_lo_c = lo.predict(calib_X)
    q_hi_c = hi.predict(calib_X)
    nonconformity = np.maximum(q_lo_c - y_calib, y_calib - q_hi_c)

    n = len(nonconformity)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    Q = np.quantile(nonconformity, level)
    print(f"  alpha={alpha:.2f} | n_calib={n} | Q={Q:.3f} days")
    return lo, hi, Q


def evaluate_intervals(y_true, lower, upper, alpha):
   
    covered      = (y_true >= lower) & (y_true <= upper)
    coverage     = covered.mean()
    width        = (upper - lower).mean()
    efficiency   = coverage / width if width > 0 else np.nan
    width_cov    = (upper - lower)[covered].mean()  if covered.sum()  > 0 else np.nan
    width_miss   = (upper - lower)[~covered].mean() if (~covered).sum() > 0 else np.nan
    penalty      = (2.0 / alpha) * (
        np.maximum(0, lower - y_true) + np.maximum(0, y_true - upper)
    )
    winkler      = (upper - lower + penalty).mean()
    return {
        "empirical_coverage":   coverage,
        "mean_width":           width,
        "coverage_efficiency":  efficiency,
        "mean_width_covered":   width_cov,
        "mean_width_missed":    width_miss,
        "winkler_score":        winkler,
    }


def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins    = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_prob, bins, right=True) - 1, 0, n_bins - 1)
    ece     = 0.0
    rows    = []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        bin_conf = y_prob[mask].mean()
        bin_acc  = y_true[mask].mean()
        weight   = mask.sum() / len(y_true)
        ece     += weight * abs(bin_conf - bin_acc)
        rows.append({
            "bin": b,
            "bin_range": f"[{bins[b]:.1f},{bins[b+1]:.1f}]",
            "n": int(mask.sum()),
            "mean_predicted_prob": bin_conf,
            "observed_rate": bin_acc,
        })
    return ece, pd.DataFrame(rows)


def main():
    df = load_synthetic()
    train, calib, test, fold2 = split_temporal(df)
    train_X, calib_X, test_X, fold2_X = build_feature_matrix(train, calib, test, fold2)

    y_train = train["delay_days"].values
    y_calib = calib["delay_days"].values

    print(f"\nFitting primary CQR (alpha={ALPHA})...")
    lo_model, hi_model, Q = fit_cqr(train_X, y_train, calib_X, y_calib, ALPHA)

    conformal_metrics = []
    for fold_name, frame, X in [("test", test, test_X), ("fold2", fold2, fold2_X)]:
        lower = np.clip(lo_model.predict(X) - Q, 0, None)
        upper = np.clip(hi_model.predict(X) + Q, 0, None)
        y_true = frame["delay_days"].values

        metrics = evaluate_intervals(y_true, lower, upper, ALPHA)
        row = {"fold": fold_name, "nominal_coverage": 1 - ALPHA,
               "calibration_Q": Q, **metrics}
        conformal_metrics.append(row)

        print(f"  [{fold_name}] coverage={metrics['empirical_coverage']:.4f} "
              f"(target={1-ALPHA:.2f}) | width={metrics['mean_width']:.2f}d | "
              f"efficiency={metrics['coverage_efficiency']:.4f} | "
              f"winkler={metrics['winkler_score']:.4f}")
        print(f"           width|covered={metrics['mean_width_covered']:.2f}d  "
              f"width|missed={metrics['mean_width_missed']:.2f}d")

        out = frame[["po_id", "supplier_id", "order_date", "delay_days", "late"]].copy()
        out["conformal_lower"]  = lower
        out["conformal_upper"]  = upper
        out["conformal_buffer"] = upper
        out.to_csv(RESULTS_DIR / f"conformal_intervals_{fold_name}.csv", index=False)

    pd.DataFrame(conformal_metrics).to_csv(RESULTS_DIR / "conformal_metrics.csv", index=False)

    print(f"\nPareto analysis across alpha = {ALPHA_GRID}...")
    pareto_rows = []
    for a in ALPHA_GRID:
        lo_a, hi_a, Q_a = fit_cqr(train_X, y_train, calib_X, y_calib, a)
        for fold_name, frame, X in [("test", test, test_X), ("fold2", fold2, fold2_X)]:
            lower_a = np.clip(lo_a.predict(X) - Q_a, 0, None)
            upper_a = np.clip(hi_a.predict(X) + Q_a, 0, None)
            y_true  = frame["delay_days"].values
            m = evaluate_intervals(y_true, lower_a, upper_a, a)
            pareto_rows.append({
                "fold": fold_name,
                "alpha": a,
                "nominal_coverage": 1 - a,
                **m,
            })

    pareto_df = pd.DataFrame(pareto_rows)
    pareto_df.to_csv(RESULTS_DIR / "conformal_pareto.csv", index=False)

    print("\n=== Pareto summary (test fold) ===")
    test_pareto = pareto_df[pareto_df["fold"] == "test"]
    print(test_pareto[["alpha", "nominal_coverage", "empirical_coverage",
                        "mean_width", "coverage_efficiency",
                        "winkler_score"]].to_string(index=False))

    print("\nComputing probability calibration metrics...")
    preds_test = pd.read_csv(RESULTS_DIR / "predictions_test.csv")
    prob_cols   = [c for c in preds_test.columns if c.startswith("pred_late_prob__")]
    y_true_test = preds_test["late"].values

    calib_rows       = []
    reliability_rows = []
    for col in prob_cols:
        model_name = col.replace("pred_late_prob__", "")
        y_prob     = preds_test[col].values
        brier      = np.mean((y_prob - y_true_test) ** 2)
        ece, rel_df = expected_calibration_error(y_true_test, y_prob)
        rel_df["model"] = model_name
        reliability_rows.append(rel_df)
        calib_rows.append({"model": model_name, "brier_score": brier, "ece": ece})
        print(f"  {model_name:<30} Brier={brier:.4f}  ECE={ece:.4f}")

    pd.DataFrame(calib_rows).to_csv(
        RESULTS_DIR / "calibration_metrics.csv", index=False)
    pd.concat(reliability_rows, ignore_index=True).to_csv(
        RESULTS_DIR / "reliability_diagram_data.csv", index=False)

    print("\nSaved:")
    print("  conformal_intervals_test.csv / fold2.csv")
    print("  conformal_metrics.csv  (+ efficiency, winkler)")
    print("  conformal_pareto.csv   (NEW -- Pareto frontier across alpha)")
    print("  calibration_metrics.csv")
    print("  reliability_diagram_data.csv")


if __name__ == "__main__":
    main()
