"""
04_conformal_prediction.py
============================
Proposal mapping: Section 9 (Conformal Prediction Layer), Section 12
(Probability Calibration).

METHOD: Conformalized Quantile Regression (CQR), as agreed with the user
instead of plain split-conformal, because it gives feature-adaptive interval
widths (a risky PO gets a wider interval than a low-risk one) rather than a
single fixed-width correction for every PO.

CQR procedure:
  1. Train two LightGBM quantile regressors on TRAIN: one for the lower
     quantile (alpha/2) and one for the upper quantile (1 - alpha/2) of
     delay_days. We use alpha = 0.10 -> nominal 90% coverage (matches the
     worked example in the proposal's methodology doc).
  2. On the CALIBRATION fold (2024 Q1, untouched by training), compute the
     nonconformity score for each PO:
         E_i = max(q_lo(x_i) - y_i, y_i - q_hi(x_i))
  3. Take Q = the (1-alpha)-quantile of {E_i} (with finite-sample correction).
  4. For any new PO: interval = [q_lo(x) - Q, q_hi(x) + Q].
     This interval has a distribution-free, finite-sample coverage guarantee
     under exchangeability of calib/test (Section 9, eq. 6).

Reuses the EXACT SAME temporal split and feature-encoding logic as
03_modeling.py (copied here, not imported, since 03_modeling.py is not a
valid Python module name starting with a digit -- kept byte-identical to
avoid any train/test contamination drift between scripts).

Also computes classification probability-calibration metrics (Brier score,
Expected Calibration Error, reliability-diagram bins) for the classifiers
already scored in 03_modeling.py, reusing their saved test-fold predictions
(no need to refit).

Outputs:
  - results/conformal_intervals_test.csv
  - results/conformal_intervals_fold2.csv
  - results/conformal_metrics.csv      (coverage, mean width, by fold)
  - results/calibration_metrics.csv    (Brier, ECE per classifier)
  - results/reliability_diagram_data.csv
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
ALPHA = 0.10  # 90% nominal coverage, matches proposal's worked example
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


def conformalized_quantile_regression(train_X, y_train, calib_X, y_calib, eval_X_dict, alpha=ALPHA):
    """Returns dict {name: (lower, upper)} for each frame in eval_X_dict, plus
    the calibration nonconformity quantile Q and the raw (pre-conformal)
    quantile predictions for diagnostics."""
    lo_model = LGBMRegressor(objective="quantile", alpha=alpha / 2,
                              n_estimators=300, random_state=RANDOM_SEED, verbosity=-1)
    hi_model = LGBMRegressor(objective="quantile", alpha=1 - alpha / 2,
                              n_estimators=300, random_state=RANDOM_SEED, verbosity=-1)
    lo_model.fit(train_X, y_train)
    hi_model.fit(train_X, y_train)

    q_lo_calib = lo_model.predict(calib_X)
    q_hi_calib = hi_model.predict(calib_X)
    nonconformity = np.maximum(q_lo_calib - y_calib, y_calib - q_hi_calib)

    n = len(nonconformity)
    # finite-sample-corrected quantile level (standard split-conformal correction)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    Q = np.quantile(nonconformity, level)
    print(f"Calibration set size: {n} | nonconformity quantile Q = {Q:.3f} days")

    results = {}
    for name, X in eval_X_dict.items():
        q_lo = lo_model.predict(X) - Q
        q_hi = hi_model.predict(X) + Q
        results[name] = (q_lo, q_hi)
    return results, Q, lo_model, hi_model


def evaluate_intervals(y_true, lower, upper):
    covered = (y_true >= lower) & (y_true <= upper)
    coverage = covered.mean()
    width = (upper - lower).mean()
    return coverage, width


def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(y_prob, bins, right=True) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    ece = 0.0
    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        bin_conf = y_prob[mask].mean()
        bin_acc = y_true[mask].mean()
        weight = mask.sum() / len(y_true)
        ece += weight * abs(bin_conf - bin_acc)
        rows.append({"bin": b, "bin_range": f"[{bins[b]:.1f},{bins[b+1]:.1f}]",
                     "n": int(mask.sum()), "mean_predicted_prob": bin_conf, "observed_rate": bin_acc})
    return ece, pd.DataFrame(rows)


def main():
    df = load_synthetic()
    train, calib, test, fold2 = split_temporal(df)
    train_X, calib_X, test_X, fold2_X = build_feature_matrix(train, calib, test, fold2)

    y_train = train["delay_days"].values
    y_calib = calib["delay_days"].values

    interval_results, Q, lo_model, hi_model = conformalized_quantile_regression(
        train_X, y_train, calib_X, y_calib,
        eval_X_dict={"test": test_X, "fold2": fold2_X},
    )

    conformal_metrics = []
    for fold_name, frame in [("test", test), ("fold2", fold2)]:
        lower, upper = interval_results[fold_name]
        y_true = frame["delay_days"].values
        coverage, width = evaluate_intervals(y_true, lower, upper)
        conformal_metrics.append({
            "fold": fold_name, "nominal_coverage": 1 - ALPHA,
            "empirical_coverage": coverage, "mean_interval_width": width,
            "calibration_Q": Q,
        })
        print(f"[{fold_name}] empirical coverage: {coverage:.4f} (target {1-ALPHA:.2f}) | "
              f"mean interval width: {width:.2f} days")

        out = frame[["po_id", "supplier_id", "order_date", "delay_days", "late"]].copy()
        out["conformal_lower"] = np.clip(lower, 0, None)
        out["conformal_upper"] = np.clip(upper, 0, None)
        out["conformal_buffer"] = out["conformal_upper"]  # upper bound = procurement buffer (Section 9, eq. 8)
        out.to_csv(RESULTS_DIR / f"conformal_intervals_{fold_name}.csv", index=False)

    pd.DataFrame(conformal_metrics).to_csv(RESULTS_DIR / "conformal_metrics.csv", index=False)

    # --- Classification calibration metrics (Section 12) ---
    # Reuses predictions already produced by 03_modeling.py -- no refitting needed.
    preds_test = pd.read_csv(RESULTS_DIR / "predictions_test.csv")
    prob_cols = [c for c in preds_test.columns if c.startswith("pred_late_prob__")]
    y_true_test = preds_test["late"].values

    calib_rows = []
    reliability_rows = []
    for col in prob_cols:
        model_name = col.replace("pred_late_prob__", "")
        y_prob = preds_test[col].values
        brier = np.mean((y_prob - y_true_test) ** 2)
        ece, reliability_df = expected_calibration_error(y_true_test, y_prob)
        reliability_df["model"] = model_name
        reliability_rows.append(reliability_df)
        calib_rows.append({"model": model_name, "brier_score": brier, "ece": ece})
        print(f"[calibration] {model_name}: Brier={brier:.4f}, ECE={ece:.4f}")

    pd.DataFrame(calib_rows).to_csv(RESULTS_DIR / "calibration_metrics.csv", index=False)
    pd.concat(reliability_rows, ignore_index=True).to_csv(RESULTS_DIR / "reliability_diagram_data.csv", index=False)

    print("\nSaved: conformal_intervals_test.csv, conformal_intervals_fold2.csv, "
          "conformal_metrics.csv, calibration_metrics.csv, reliability_diagram_data.csv")


if __name__ == "__main__":
    main()
