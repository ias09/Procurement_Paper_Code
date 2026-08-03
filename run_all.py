"""
run_all.py
===========
Single-command orchestrator for the full procurement decision intelligence
pipeline. Runs scripts 00-08 in order, passing results between them via the
data/ and results/ directories.

Usage:
    python run_all.py                   # full pipeline including data generation
    python run_all.py --skip-generate   # skip 00 (use existing synthetic data)
    python run_all.py --skip-external   # skip 07 + 08 (skip external validation)
    python run_all.py --from 03         # restart from a specific script number

All output files end up in:
    data/       -- cleaned CSVs and feature CSVs
    results/    -- metrics, predictions, plots, SHAP outputs
"""

import subprocess
import sys
import time
import argparse
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
PROJECT_DIR = Path(__file__).resolve().parent

PIPELINE = [
    ("00", "00_generate_synthetic_data.py", "Generating synthetic ERP dataset"),
    ("01", "01_preprocessing.py",           "Preprocessing both datasets"),
    ("02", "02_feature_engineering.py",     "Engineering leakage-safe features"),
    ("03", "03_modeling.py",                "Training + tuning ML models (Optuna + stacking)"),
    ("04", "04_conformal_prediction.py",    "Conformal prediction intervals (CQR) + calibration"),
    ("05", "05_decision_layer.py",          "Cost-sensitive + service-level buffering policies"),
    ("06", "06_explainability.py",          "SHAP explainability analysis"),
    ("07", "07_external_validation.py",     "External validation: Dataset 1 (same-domain)"),
    ("08", "08_cross_domain_validation.py", "Cross-domain robustness: DataCo (domain-shift check)"),
    ("09", "09_predictability_audit.py",    "Data predictability audit (pre-deployment health check)"),
    ("10", "10_decision_calibrated_buffering.py", "Decision-calibrated buffering (DCCB) + Mondrian groups"),
    ("11", "11_adaptive_conformal_drift.py", "Adaptive conformal drift experiment (self-correcting margins)"),
]


def run_script(script_name: str, description: str) -> bool:
    script_path = SCRIPTS_DIR / script_name
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"  Script: {script_name}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_DIR),
        capture_output=False,
    )
    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"\n  ✓ Completed in {elapsed:.1f}s")
        return True
    else:
        print(f"\n  ✗ FAILED after {elapsed:.1f}s (return code {result.returncode})")
        return False


def parse_args():
    parser = argparse.ArgumentParser(description="Run the full procurement ML pipeline")
    parser.add_argument("--skip-generate", action="store_true",
                        help="Skip script 00 (use existing synthetic data)")
    parser.add_argument("--skip-external", action="store_true",
                        help="Skip scripts 07 and 08 (external validation)")
    parser.add_argument("--from", dest="from_script", default="00",
                        help="Start from this script number (e.g. --from 03)")
    return parser.parse_args()


def print_summary(results: dict):
    print(f"\n{'='*60}")
    print("  PIPELINE SUMMARY")
    print(f"{'='*60}")
    for script_num, name, desc, success, elapsed in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}  [{script_num}] {desc} ({elapsed:.1f}s)")

    failed = [r for r in results if not r[3]]
    if not failed:
        print(f"\n  All {len(results)} scripts completed successfully.")
        print(f"  Results: {PROJECT_DIR / 'results/'}")
        print(f"  Data:    {PROJECT_DIR / 'data/'}")
    else:
        print(f"\n  {len(failed)} script(s) failed. Check output above for details.")


def print_key_results():
    """Read and print the most important metrics after a successful run."""
    results_dir = PROJECT_DIR / "results"
    print(f"\n{'='*60}")
    print("  KEY RESULTS")
    print(f"{'='*60}")
    try:
        import pandas as pd, json

        m = pd.read_csv(results_dir / "model_metrics.csv")
        clf = m[(m["fold"]=="test") & (m["target"]=="classification")].sort_values("roc_auc", ascending=False)
        reg = m[(m["fold"]=="test") & (m["target"]=="regression")].sort_values("mae")

        print("\n  Classification (test fold):")
        for _, row in clf.iterrows():
            print(f"    {row['model']:<28} ROC-AUC={row['roc_auc']:.4f}  PR-AUC={row['pr_auc']:.4f}  F1={row['f1']:.4f}")

        print("\n  Regression (test fold):")
        for _, row in reg.iterrows():
            print(f"    {row['model']:<28} MAE={row['mae']:.4f}  RMSE={row['rmse']:.4f}")

        cm = pd.read_csv(results_dir / "conformal_metrics.csv")
        print("\n  Conformal prediction:")
        for _, row in cm.iterrows():
            print(f"    [{row['fold']}] coverage={row['empirical_coverage']:.4f} "
                  f"(target={row['nominal_coverage']:.2f})  width={row['mean_interval_width']:.2f} days")

        pc = pd.read_csv(results_dir / "policy_comparison.csv")
        print("\n  Procurement policy comparison (test fold):")
        for _, row in pc[pc["fold"]=="test"].iterrows():
            print(f"    {row['policy']:<30} cost={row['mean_realized_cost']:.3f}  "
                  f"service_level={row['service_level_attained']:.3f}  "
                  f"buffer={row['mean_buffer_days']:.1f}d")

    except Exception as e:
        print(f"  (Could not read results: {e})")


def main():
    args = parse_args()
    start_num = args.from_script.zfill(2)

    print(f"\n{'='*60}")
    print("  PROCUREMENT DECISION INTELLIGENCE PIPELINE")
    print("  Shawon -- Q1 Journal Submission Pipeline")
    print(f"{'='*60}")
    print(f"  Project dir:   {PROJECT_DIR}")
    print(f"  Python:        {sys.executable}")
    if args.skip_generate:  print("  [--skip-generate] Skipping script 00")
    if args.skip_external:  print("  [--skip-external] Skipping scripts 07 and 08")
    if start_num != "00":   print(f"  [--from {start_num}] Starting from script {start_num}")

    results = []
    pipeline_start = time.time()

    for script_num, script_name, description in PIPELINE:
        # skip rules
        if script_num < start_num:
            continue
        if args.skip_generate and script_num == "00":
            print(f"\n  [SKIP] {description}")
            continue
        if args.skip_external and script_num in ("07", "08"):
            print(f"\n  [SKIP] {description}")
            continue

        t0 = time.time()
        success = run_script(script_name, description)
        elapsed = time.time() - t0
        results.append((script_num, script_name, description, success, elapsed))

        if not success:
            print(f"\n  Pipeline HALTED at script {script_num}. Fix the error above and re-run.")
            print(f"  Tip: use --from {script_num} to restart from this point.")
            print_summary(results)
            sys.exit(1)

    total_elapsed = time.time() - pipeline_start
    print(f"\n  Total pipeline time: {total_elapsed/60:.1f} minutes")
    print_summary(results)
    print_key_results()


if __name__ == "__main__":
    main()
