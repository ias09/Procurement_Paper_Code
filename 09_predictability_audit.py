"""
09_predictability_audit.py
============================
PLAIN-LANGUAGE PURPOSE (Idea 4 from the updated proposal):
Before building any model on a procurement dataset, run a 5-minute health
check that answers: "does lateness in this data actually follow any pattern,
or is it basically random?" If it's random, no model -- however fancy --
can predict it, and the company should know that BEFORE spending weeks on ML.

The check computes, for each dataset:

  1. SUPPLIER SPREAD: how different are suppliers from each other?
     If every supplier is late ~30% of the time, supplier identity tells you
     nothing. If some are late 10% and others 45%, there's real signal.
     (statistic: standard deviation + min/max of per-supplier late rates)

  2. ORACLE CORRELATION: the "cheating ceiling". We let ourselves cheat and
     use each supplier's FULL history (including the future) as a predictor.
     No honest model can beat this cheat. If even the cheat correlates ~0
     with outcomes, the dataset is unpredictable, full stop.

  3. FEATURE CORRELATIONS: do basic order facts (quantity, price, lead time)
     relate to delay at all?

  4. PREDICTED AUC BAND: a rough empirical mapping from the above statistics
     to the ROC-AUC a good model can realistically achieve. Calibrated from
     our own three datasets + published benchmarks.

We then VALIDATE the audit by comparing its predictions against the actual
model results we already obtained in scripts 03/07/08:
  synthetic  -> audit says "learnable"   -> actual AUC 0.80  (correct?)
  dataset1   -> audit says "random-like" -> actual AUC 0.49  (correct?)
  dataco     -> audit says "learnable"   -> actual AUC 0.75  (correct?)

Outputs:
  - results/predictability_audit.csv
  - results/predictability_audit_report.txt  (human-readable verdicts)
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def audit_dataset(df: pd.DataFrame, entity_col: str, name: str) -> dict:
    """Run the full predictability audit on one dataset."""
    out = {"dataset": name, "n_rows": len(df), "n_entities": df[entity_col].nunique(),
           "late_rate": df["late"].mean()}

    # --- 1. Supplier (entity) spread ---
    by_entity = df.groupby(entity_col)["late"].agg(["mean", "count"])
    stable = by_entity[by_entity["count"] >= 20]
    if len(stable) < 3:  # entities have few orders each (e.g. dataco customers)
        stable = by_entity[by_entity["count"] >= 5]
    out["entity_late_rate_std"] = stable["mean"].std()
    out["entity_late_rate_min"] = stable["mean"].min()
    out["entity_late_rate_max"] = stable["mean"].max()
    out["entity_late_rate_range"] = out["entity_late_rate_max"] - out["entity_late_rate_min"]

    # --- 2. Oracle correlation (the cheating ceiling) ---
    oracle_entity = df.groupby(entity_col)["late"].transform("mean")
    out["oracle_entity_corr"] = oracle_entity.corr(df["late"])

    cat_col = "material_or_category" if "material_or_category" in df.columns else None
    if cat_col:
        oracle_cat = df.groupby(cat_col)["late"].transform("mean")
        out["oracle_category_corr"] = oracle_cat.corr(df["late"])
    else:
        out["oracle_category_corr"] = np.nan

    # --- 3. Basic feature correlations with delay magnitude ---
    for col in ["quantity", "unit_price", "promised_lead_time"]:
        if col in df.columns:
            out[f"corr_{col}_delay"] = df[col].corr(df["delay_days"])
        else:
            out[f"corr_{col}_delay"] = np.nan

    # --- 4. Predicted achievable AUC band ---
    # The oracle entity correlation is the dominant indicator, with the
    # category oracle as a secondary source of signal.
    r = max(out["oracle_entity_corr"] if not pd.isna(out["oracle_entity_corr"]) else 0,
            out["oracle_category_corr"] if not pd.isna(out["oracle_category_corr"]) else 0)
    if r < 0.10:
        band = (0.50, 0.55); verdict = ("RANDOM-LIKE: no meaningful predictive signal. "
                                          "ML buffering NOT recommended on this data.")
    elif r < 0.25:
        band = (0.55, 0.65); verdict = "WEAK: marginal signal. ML may barely beat naive baselines."
    elif r < 0.40:
        band = (0.65, 0.80); verdict = "MODERATE: usable signal. ML buffering viable with careful features."
    else:
        band = (0.75, 0.85); verdict = "STRONG: clear entity-level structure. ML buffering recommended."
    out["predicted_auc_low"], out["predicted_auc_high"] = band
    out["verdict"] = verdict
    return out


def main():
    audits = []

    synthetic = pd.read_csv(DATA_DIR / "synthetic_clean.csv")
    audits.append(audit_dataset(synthetic, "supplier_id", "synthetic"))

    ds1 = pd.read_csv(DATA_DIR / "dataset1_clean.csv")
    audits.append(audit_dataset(ds1, "supplier_id", "dataset1_public_procurement"))

    dataco = pd.read_csv(DATA_DIR / "dataco_clean.csv")
    audits.append(audit_dataset(dataco, "entity_id", "dataco_crossdomain"))

    audit_df = pd.DataFrame(audits)
    audit_df.to_csv(RESULTS_DIR / "predictability_audit.csv", index=False)

    # --- validate audit predictions against realized model results ---
    realized = {}
    try:
        m = pd.read_csv(RESULTS_DIR / "model_metrics.csv")
        realized["synthetic"] = m[(m["fold"]=="test") & (m["target"]=="classification") &
                 (~m["model"].isin(["erp_baseline","historical_avg"]))]["roc_auc"].max()
    except Exception:
        realized["synthetic"] = np.nan
    try:
        m = pd.read_csv(RESULTS_DIR / "external_validation_metrics.csv")
        realized["dataset1_public_procurement"] = m[(m["target"]=="classification") &
                 (~m["model"].isin(["erp_baseline","historical_avg"]))]["roc_auc"].max()
    except Exception:
        realized["dataset1_public_procurement"] = np.nan
    try:
        m = pd.read_csv(RESULTS_DIR / "dataco_model_metrics.csv")
        realized["dataco_crossdomain"] = m[m["target"]=="classification"]["roc_auc"].max()
    except Exception:
        realized["dataco_crossdomain"] = np.nan

    lines = ["="*70, "DATA PREDICTABILITY AUDIT REPORT", "="*70, ""]
    for _, row in audit_df.iterrows():
        name = row["dataset"]
        actual = realized.get(name, np.nan)
        if not pd.isna(actual):
            within = (row["predicted_auc_low"] - 0.03) <= actual <= (row["predicted_auc_high"] + 0.03)
            actual_line = f"  ACTUAL best model AUC:    {actual:.3f}"
            correct_line = f"  Audit prediction correct? {'YES' if within else 'NO'}"
        else:
            actual_line = "  ACTUAL best model AUC:    (not yet run)"
            correct_line = "  Audit prediction correct? n/a"
        lines += [
            f"Dataset: {name}",
            f"  Rows: {row['n_rows']}, entities: {row['n_entities']}, late rate: {row['late_rate']:.3f}",
            f"  Entity late-rate spread: std={row['entity_late_rate_std']:.3f}, "
            f"range=[{row['entity_late_rate_min']:.3f}, {row['entity_late_rate_max']:.3f}]",
            f"  Oracle entity correlation (cheating ceiling): {row['oracle_entity_corr']:.3f}",
            f"  Oracle category correlation: {row['oracle_category_corr']:.3f}" if not pd.isna(row['oracle_category_corr']) else "  Oracle category correlation: n/a",
            f"  Predicted achievable AUC: {row['predicted_auc_low']:.2f}-{row['predicted_auc_high']:.2f}",
            actual_line,
            correct_line,
            f"  VERDICT: {row['verdict']}",
            "",
        ]
    report = "\n".join(lines)
    (RESULTS_DIR / "predictability_audit_report.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
