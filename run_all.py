import subprocess
import sys
import time
import argparse
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
PROJECT_DIR = Path(__file__).resolve().parent

PIPELINE = [
    ("00", "00_generate_synthetic_data.py",
     "Generating synthetic ERP dataset"),
    ("01", "01_preprocessing.py",
     "Preprocessing both datasets"),
    ("02", "02_feature_engineering.py",
     "Engineering leakage-safe features"),
    ("03", "03_modeling.py",
     "Training + tuning ML models (Optuna + stacking)"),
    ("04", "04_conformal_prediction.py",
     "CQR conformal intervals + calibration + Pareto analysis"),
    ("05", "05_decision_layer.py",
     "Cost-sensitive + service-level buffering policies"),
    ("06", "06_explainability.py",
     "SHAP explainability analysis"),
    ("07", "07_external_validation.py",
     "External validation: KagProc + CQR conformal on KagProc"),
    ("08", "08_cross_domain_validation.py",
     "Cross-domain robustness: DataCo (with ECE + full policy table)"),
    ("09", "09_predictability_audit.py",
     "Data predictability audit (pre-deployment health check)"),
    ("10", "10_decision_calibrated_buffering.py",
     "DCCB + Mondrian (with cost breakdown + group-count sensitivity)"),
    ("11", "11_adaptive_conformal_drift.py",
     "ACI drift experiment (recovery speed + gamma sensitivity)"),
]


def run_script(script_name: str, description: str) -> bool:
    script_path = SCRIPTS_DIR / script_name
    print(f"\n{'='*62}")
    print(f"  {description}")
    print(f"  Script: {script_name}")
    print(f"{'='*62}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_DIR),
        capture_output=False,
    )
    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"\n  OK  Completed in {elapsed:.1f}s")
        return True
    else:
        print(f"\n  FAILED after {elapsed:.1f}s "
              f"(return code {result.returncode})")
        return False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full procurement ML pipeline")
    parser.add_argument("--skip-generate", action="store_true",
                        help="Skip script 00 (use existing synthetic data)")
    parser.add_argument("--skip-external", action="store_true",
                        help="Skip scripts 07 and 08 (external validation)")
    parser.add_argument("--from", dest="from_script", default="00",
                        help="Start from this script number (e.g. --from 03)")
    return parser.parse_args()


def print_summary(results: list):
    print(f"\n{'='*62}")
    print("  PIPELINE SUMMARY")
    print(f"{'='*62}")
    for script_num, name, desc, success, elapsed in results:
        status = "PASS" if success else "FAIL"
        print(f"  {status}  [{script_num}] {desc} ({elapsed:.1f}s)")

    failed = [r for r in results if not r[3]]
    if not failed:
        print(f"\n  All {len(results)} scripts completed successfully.")
        print(f"  Results: {PROJECT_DIR / 'results/'}")
        print(f"  Data:    {PROJECT_DIR / 'data/'}")
    else:
        print(f"\n  {len(failed)} script(s) failed.")


def print_key_results():
    """Print the most important metrics after a successful run."""
    results_dir = PROJECT_DIR / "results"
    print(f"\n{'='*62}")
    print("  KEY RESULTS")
    print(f"{'='*62}")
    try:
        import pandas as pd

        m = pd.read_csv(results_dir / "model_metrics.csv")
        clf = (m[(m["fold"] == "test") & (m["target"] == "classification")]
               .sort_values("roc_auc", ascending=False))
        reg = (m[(m["fold"] == "test") & (m["target"] == "regression")]
               .sort_values("mae"))
        print("\n  Classification (synthetic test fold):")
        for _, r in clf.iterrows():
            print(f"    {r['model']:<28} AUC={r['roc_auc']:.4f}  "
                  f"PR-AUC={r['pr_auc']:.4f}  F1={r['f1']:.4f}")
        print("\n  Regression (synthetic test fold):")
        for _, r in reg.iterrows():
            print(f"    {r['model']:<28} MAE={r['mae']:.4f}  "
                  f"RMSE={r['rmse']:.4f}")

        cm = pd.read_csv(results_dir / "conformal_metrics.csv")
        print("\n  Conformal prediction (synthetic):")
        for _, r in cm.iterrows():
            print(f"    [{r['fold']}] cov={r['empirical_coverage']:.4f} "
                  f"(target={r['nominal_coverage']:.2f})  "
                  f"width={r['mean_width']:.2f}d  "
                  f"efficiency={r['coverage_efficiency']:.4f}  "
                  f"winkler={r['winkler_score']:.4f}")

        try:
            par = pd.read_csv(results_dir / "conformal_pareto.csv")
            print("\n  Conformal Pareto (test fold):")
            for _, r in par[par["fold"] == "test"].iterrows():
                print(f"    alpha={r['alpha']:.2f}  "
                      f"nom_cov={r['nominal_coverage']:.2f}  "
                      f"emp_cov={r['empirical_coverage']:.4f}  "
                      f"width={r['mean_width']:.2f}d  "
                      f"efficiency={r['coverage_efficiency']:.4f}")
        except FileNotFoundError:
            pass

        pc = pd.read_csv(results_dir / "policy_comparison.csv")
        print("\n  Procurement policy comparison (synthetic test fold):")
        for _, r in pc[pc["fold"] == "test"].iterrows():
            print(f"    {r['policy']:<30} cost={r['mean_realized_cost']:.3f}  "
                  f"svc={r['service_level_attained']:.3f}  "
                  f"buf={r['mean_buffer_days']:.1f}d")

        try:
            aci = pd.read_csv(results_dir / "aci_summary.csv")
            print("\n  ACI drift experiment:")
            for _, r in aci.iterrows():
                print(f"    [{r['scenario']}]  "
                      f"static_cov={r['overall_coverage_static']:.4f}  "
                      f"aci_cov={r['overall_coverage_aci']:.4f}  "
                      f"min_rolling_static={r['min_rolling_cov_static']:.4f}  "
                      f"min_rolling_aci={r['min_rolling_cov_aci']:.4f}")
        except FileNotFoundError:
            pass

        try:
            rec = pd.read_csv(results_dir / "aci_recovery_summary.csv")
            r   = rec.iloc[0]
            print(f"\n  ACI recovery speed (Scenario B stressed suppliers):")
            print(f"    mean orders to recovery : {r['mean_orders_to_recovery']:.1f}")
            print(f"    median                  : {r['median_orders_to_recovery']:.1f}")
            print(f"    pct recovered in window : {r['pct_recovered']*100:.0f}%")
        except FileNotFoundError:
            pass

        try:
            gs = pd.read_csv(results_dir / "aci_gamma_sensitivity.csv")
            print("\n  ACI gamma sensitivity (Scenario B):")
            for _, r in gs.iterrows():
                print(f"    gamma={r['gamma']:.2f}  "
                      f"overall_cov={r['overall_coverage_aci']:.4f}  "
                      f"min_rolling={r['min_rolling_cov_aci']:.4f}  "
                      f"buf={r['mean_buffer_aci_days']:.2f}d  "
                      f"orders_to_rec={r['mean_orders_to_recovery']:.1f}")
        except FileNotFoundError:
            pass

        try:
            dccb = pd.read_csv(results_dir / "dccb_policy_comparison.csv")
            print("\n  DCCB policy comparison:")
            for _, r in dccb.iterrows():
                print(f"    {r['policy']:<22} cov={r['overall_coverage']:.4f}  "
                      f"cost={r['mean_realized_cost']:.4f}  "
                      f"buf={r['mean_buffer_days']:.2f}d")
        except FileNotFoundError:
            pass

        try:
            mg = pd.read_csv(results_dir / "mondrian_groupcount_sensitivity.csv")
            print("\n  Mondrian group-count sensitivity:")
            for _, r in mg.iterrows():
                print(f"    n_groups={int(r['n_groups'])}  "
                      f"overall_cov={r['overall_coverage']:.4f}  "
                      f"max_gap={r['max_coverage_gap']:.4f}  "
                      f"buf={r['mean_buffer_days']:.2f}d  "
                      f"cost={r['mean_realized_cost']:.4f}")
        except FileNotFoundError:
            pass

        try:
            ev = pd.read_csv(results_dir / "external_validation_metrics.csv")
            clf_ev = (ev[ev["target"] == "classification"]
                      .sort_values("roc_auc", ascending=False))
            print("\n  KagProc external validation (classification):")
            for _, r in clf_ev.iterrows():
                ece_str = f"  ECE={r['ece']:.4f}" if "ece" in r and not pd.isna(r.get("ece")) else ""
                print(f"    {r['model']:<28} AUC={r['roc_auc']:.4f}  "
                      f"F1={r['f1']:.4f}  Brier={r['brier']:.4f}{ece_str}")
        except FileNotFoundError:
            pass

        try:
            kc = pd.read_csv(results_dir / "kagproc_conformal_metrics.csv")
            r  = kc.iloc[0]
            print(f"\n  KagProc conformal (should show wide intervals / trivial coverage):")
            print(f"    empirical coverage : {r['empirical_coverage']:.4f}")
            print(f"    mean width         : {r['mean_width']:.2f} days")
            print(f"    coverage efficiency: {r['coverage_efficiency']:.4f}")
        except FileNotFoundError:
            pass

        try:
            dc = pd.read_csv(results_dir / "dataco_model_metrics.csv")
            clf_dc = (dc[dc["target"] == "classification"]
                      .sort_values("roc_auc", ascending=False))
            print("\n  DataCo cross-domain (classification):")
            for _, r in clf_dc.iterrows():
                ece_str = f"  ECE={r['ece']:.4f}" if "ece" in r and not pd.isna(r.get("ece")) else ""
                print(f"    {r['model']:<28} AUC={r['roc_auc']:.4f}  "
                      f"F1={r['f1']:.4f}  Brier={r['brier']:.4f}{ece_str}")
        except FileNotFoundError:
            pass

        try:
            dcp = pd.read_csv(results_dir / "dataco_policy_comparison.csv")
            print("\n  DataCo policy comparison:")
            for _, r in dcp.iterrows():
                print(f"    {r['policy']:<26} cost={r['mean_realized_cost']:.3f}  "
                      f"svc={r['service_level']:.3f}  "
                      f"buf={r['mean_buffer_days']:.1f}d")
        except FileNotFoundError:
            pass

    except Exception as e:
        print(f"  (Could not read results: {e})")


def main():
    args       = parse_args()
    start_num  = args.from_script.zfill(2)

    print(f"\n{'='*62}")
    print("  PROCUREMENT DECISION INTELLIGENCE PIPELINE")
    print("  Q1 Journal Submission -- Full Reproducible Run")
    print(f"{'='*62}")
    print(f"  Project dir : {PROJECT_DIR}")
    print(f"  Python      : {sys.executable}")
    if args.skip_generate: print("  [--skip-generate] Skipping script 00")
    if args.skip_external: print("  [--skip-external] Skipping scripts 07 and 08")
    if start_num != "00":  print(f"  [--from {start_num}] Starting from script {start_num}")

    results        = []
    pipeline_start = time.time()

    for script_num, script_name, description in PIPELINE:
        if script_num < start_num:
            continue
        if args.skip_generate and script_num == "00":
            print(f"\n  [SKIP] {description}")
            continue
        if args.skip_external and script_num in ("07", "08"):
            print(f"\n  [SKIP] {description}")
            continue

        t0      = time.time()
        success = run_script(script_name, description)
        elapsed = time.time() - t0
        results.append((script_num, script_name, description, success, elapsed))

        if not success:
            print(f"\n  Pipeline HALTED at script {script_num}.")
            print(f"  Tip: use --from {script_num} to restart from this point.")
            print_summary(results)
            sys.exit(1)

    total_elapsed = time.time() - pipeline_start
    print(f"\n  Total pipeline time: {total_elapsed/60:.1f} minutes")
    print_summary(results)
    print_key_results()


if __name__ == "__main__":
    main()
