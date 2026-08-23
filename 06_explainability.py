
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from catboost import CatBoostClassifier, CatBoostRegressor

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

TRAIN_END = pd.Timestamp("2023-12-31")
TEST_START, TEST_END = pd.Timestamp("2024-04-01"), pd.Timestamp("2024-06-30")

NON_FEATURE_COLUMNS = [
    "po_id", "order_date", "promised_delivery_date", "actual_delivery_date",
    "actual_lead_time", "delay", "delay_days", "late",
]
CATEGORICAL_CANDIDATES = [
    "supplier_id", "material_or_category", "order_priority_or_type",
    "plant_id", "buyer_id",
]

CATBOOST_REG_PARAMS = {"n_estimators": 400, "learning_rate": 0.05, "depth": 6}


def load_tuned_classifier_params():
   
    params_path = RESULTS_DIR / "optuna_best_params.json"
    if not params_path.exists():
        raise FileNotFoundError(
            f"{params_path} not found. Run 03_modeling.py before this script -- "
            "SHAP must explain the tuned CatBoost classifier that 03 reports as "
            "the best model, not an untuned default."
        )
    with open(params_path) as fh:
        best = json.load(fh)
    if "catboost_clf" not in best:
        raise KeyError(
            "optuna_best_params.json has no 'catboost_clf' key. "
            "Regenerate it by re-running 03_modeling.py."
        )
    return best["catboost_clf"]


def load_synthetic():
    return pd.read_csv(
        DATA_DIR / "synthetic_features.csv",
        parse_dates=["order_date", "promised_delivery_date", "actual_delivery_date"],
    )


def split_temporal(df):
    train = df[df["order_date"] <= TRAIN_END].copy()
    test = df[(df["order_date"] >= TEST_START) & (df["order_date"] <= TEST_END)].copy()
    return train, test


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


def aggregate_onehot_importance(mean_abs_shap, columns):
  
    grouped = {}
    for col, val in zip(columns, mean_abs_shap):
        base = col
        for cat in CATEGORICAL_CANDIDATES:
            if col.startswith(cat + "_"):
                base = cat
                break
        grouped[base] = grouped.get(base, 0.0) + val
    return pd.Series(grouped).sort_values(ascending=False)


def run_shap(model, X, model_label, sample_size=500, random_seed=RANDOM_SEED):
    
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):      
        shap_values = shap_values[1]

    
    rng     = np.random.default_rng(random_seed)
    idx     = rng.choice(len(X), size=min(sample_size, len(X)), replace=False)
    X_s     = X.iloc[idx].reset_index(drop=True)
    sv_s    = shap_values[idx]

    mean_abs            = np.abs(shap_values).mean(axis=0)
    pd.Series(mean_abs, index=X.columns).sort_values(ascending=False).to_csv(
    RESULTS_DIR / f"shap_importance_{model_label}_raw_dummies.csv")
    importance_grouped  = aggregate_onehot_importance(mean_abs, X.columns)
    importance_grouped.to_csv(
        RESULTS_DIR / f"shap_importance_{model_label}.csv",
        header=["mean_abs_shap"]
    )
    top10        = importance_grouped.head(10)
    top10_groups = set(top10.index.tolist())

    def col_group(col):
        for cat in CATEGORICAL_CANDIDATES:
            if col.startswith(cat + "_"):
                return cat
        return col

    top10_mask   = [col_group(c) in top10_groups for c in X.columns]
    X_top10      = X_s.loc[:, top10_mask]
    sv_top10     = sv_s[:, top10_mask]
    disp_names   = [col_group(c) for c in X_top10.columns]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(
        top10.index[::-1],
        top10.values[::-1],
        color="#2171b5",
        edgecolor="white",
        linewidth=0.4
    )
    ax.set_xlabel("Mean |SHAP value| (aggregated over one-hot dummies)",
                  fontsize=10)
    ax.set_title(f"Feature importance — {model_label}", fontsize=10)
    ax.tick_params(axis="y", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(RESULTS_DIR / f"shap_bar_{model_label}.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    shap.summary_plot(
        sv_top10, X_top10,
        feature_names=disp_names,
        max_display=10,
        show=False,
        plot_size=None,
        color_bar_label="Feature value"
    )
    plt.title(f"SHAP beeswarm — {model_label}", fontsize=10, pad=10)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(RESULTS_DIR / f"shap_beeswarm_{model_label}.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close()

    shap_sums    = shap_values.sum(axis=1)
    median_shap  = np.median(shap_sums)
    rep_idx      = int(np.argmin(np.abs(shap_sums - median_shap)))

    expected_val = float(explainer.expected_value[1]) \
                   if isinstance(explainer.expected_value, (list, np.ndarray)) \
                   else float(explainer.expected_value)

    expl_obj = shap.Explanation(
        values         = shap_values[rep_idx],
        base_values    = expected_val,
        data           = X.iloc[rep_idx].values,
        feature_names  = X.columns.tolist()
    )
    plt.figure(figsize=(8, 5))
    shap.plots.waterfall(expl_obj, max_display=10, show=False)
    plt.title(f"SHAP waterfall (representative order) — {model_label}",
              fontsize=10, pad=10)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(RESULTS_DIR / f"shap_waterfall_{model_label}.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close()

    top_feature   = top10.index[0]
    top_feat_cols = [c for c in X.columns if col_group(c) == top_feature]

    if top_feat_cols:
      
        if len(top_feat_cols) == 1:
            feat_vals  = X_s[top_feat_cols[0]].values
            feat_label = top_feature
        else:
          
            feat_vals  = X_s[top_feat_cols].sum(axis=1).values
            feat_label = f"{top_feature} (dummy sum)"

        top_feat_sv = sv_s[:, [c in top_feat_cols for c in X.columns]].sum(axis=1)

        fig, ax = plt.subplots(figsize=(6, 4))
        sc = ax.scatter(
            feat_vals, top_feat_sv,
            c=feat_vals, cmap="coolwarm",
            alpha=0.5, s=12, linewidths=0
        )
        plt.colorbar(sc, ax=ax, label=feat_label)
        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_xlabel(feat_label, fontsize=10)
        ax.set_ylabel(f"SHAP value for {top_feature}", fontsize=10)
        ax.set_title(f"SHAP dependence — {top_feature} | {model_label}",
                     fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        for ext in ("pdf", "png"):
            plt.savefig(RESULTS_DIR / f"shap_scatter_{model_label}.{ext}",
                        dpi=300, bbox_inches="tight")
        plt.close()

    print(f"\n=== Top 10 SHAP drivers ({model_label}) ===")
    print(importance_grouped.head(10).to_string())
    return importance_grouped


def main():
    df = load_synthetic()
    train, test = split_temporal(df)
    train_X, test_X = build_feature_matrix(train, test)

    y_train_clf = train["late"].values
    y_train_reg = train["delay_days"].values

    catb_clf_p = load_tuned_classifier_params()
    print(f"Loaded Optuna-tuned CatBoost classifier params from 03_modeling.py: "
          f"{catb_clf_p}")
    clf = CatBoostClassifier(**{**catb_clf_p, "random_seed": RANDOM_SEED,
                                "verbose": False})
    clf.fit(train_X, y_train_clf)
    imp_clf = run_shap(clf, test_X, "classification")

    print(f"Using 03_modeling.py's fixed CatBoost regressor params: "
          f"{CATBOOST_REG_PARAMS}")
    reg = CatBoostRegressor(**{**CATBOOST_REG_PARAMS, "random_seed": RANDOM_SEED,
                               "verbose": False})
    reg.fit(train_X, y_train_reg)
    imp_reg = run_shap(reg, test_X, "regression")

    top_n   = 10
    labels  = imp_clf.head(top_n).index.tolist()
    clf_v   = imp_clf.reindex(labels).fillna(0).values
    reg_v   = imp_reg.reindex(labels).fillna(0).values

    x       = np.arange(top_n)
    width   = 0.38

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(x + width / 2, clf_v[::-1], width,
            label="Classification (P(late))",
            color="#2171b5", edgecolor="white", linewidth=0.4)
    ax.barh(x - width / 2, reg_v[::-1], width,
            label="Regression (delay days)",
            color="#6baed6", edgecolor="white", linewidth=0.4)
    ax.set_yticks(x)
    ax.set_yticklabels(labels[::-1], fontsize=8)
    ax.set_xlabel("Mean |SHAP value|", fontsize=10)
    ax.set_title("SHAP feature importance: classification vs regression",
                 fontsize=10)
    ax.legend(fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(RESULTS_DIR / f"shap_comparison.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close()

    print("\nSaved all SHAP figures to results/")


if __name__ == "__main__":
    main()