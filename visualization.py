"""Publication-quality visualization suite for SHAP explanation stability and model evaluation.

Addresses SRS Requirements:
- REQ-F-56: Comprehensive tabular exports.
- REQ-F-57: Visualizations of ROC/PR curves, SHAP rankings, consistency heatmaps, and recommendation distributions.
- REQ-F-58: Output serialization to the results directory.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve

logger = logging.getLogger(__name__)

# Configure clean aesthetic styles
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11


def plot_roc_pr_curves(
    predictions_df: pd.DataFrame,
    output_dir: Path,
):
    """Plot ROC and Precision-Recall curves comparing all models."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    y_true = predictions_df["actual_is_churn"].values
    prob_cols = [c for c in predictions_df.columns if c.endswith("_prob")]

    palette = {"RandomForest": "#2b5c8f", "XGBoost": "#d95f02", "CatBoost": "#2ca02c"}

    # 1. ROC Curves
    for col in prob_cols:
        model_name = col.replace("_prob", "")
        y_prob = predictions_df[col].values
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        color = palette.get(model_name, "#333333")
        axes[0].plot(fpr, tpr, label=model_name, color=color, linewidth=2)

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Guess")
    axes[0].set_title("ROC Curves Comparison (Test Set)")
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[0].set_ylabel("True Positive Rate (Recall)")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # 2. Precision-Recall Curves
    baseline = np.mean(y_true)
    for col in prob_cols:
        model_name = col.replace("_prob", "")
        y_prob = predictions_df[col].values
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        color = palette.get(model_name, "#333333")
        axes[1].plot(recall, precision, label=model_name, color=color, linewidth=2)

    axes[1].axhline(baseline, color="k", linestyle="--", alpha=0.6, label=f"Baseline ({baseline:.2f})")
    axes[1].set_title("Precision-Recall Curves Comparison")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig_path = output_dir / "model_roc_pr_curves.png"
    fig.savefig(fig_path)
    plt.close(fig)
    logger.info(f"Saved ROC/PR curves to {fig_path}")


def plot_global_shap_importance(
    importance_df: pd.DataFrame,
    output_dir: Path,
    top_n: int = 12,
):
    """Plot comparative horizontal bar chart of top mean absolute SHAP values."""
    # Compute average rank or mean importance across models to determine top features
    avg_imp = importance_df.mean(axis=1).sort_values(ascending=False)
    top_features = avg_imp.head(top_n).index.tolist()[::-1]  # reverse for bottom-up plot

    plot_df = importance_df.loc[top_features]

    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    y = np.arange(len(top_features))
    height = 0.25

    colors = ["#2b5c8f", "#d95f02", "#2ca02c"]
    models = importance_df.columns.tolist()

    for idx, model in enumerate(models):
        offset = (idx - 1) * height
        ax.barh(y + offset, plot_df[model], height=height, label=model, color=colors[idx % len(colors)], alpha=0.88)

    ax.set_yticks(y)
    ax.set_yticklabels(top_features, fontsize=10)
    ax.set_xlabel("Mean Absolute SHAP Value (Impact on Churn Probability)", fontsize=11)
    ax.set_title(f"Global SHAP Feature Importance: Top {top_n} Features Across Models", fontsize=13, pad=12)
    ax.legend(title="Model", loc="lower right")
    ax.grid(axis="x", linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig_path = output_dir / "global_shap_importance_comparison.png"
    fig.savefig(fig_path)
    plt.close(fig)
    logger.info(f"Saved SHAP importance chart to {fig_path}")


def plot_consistency_heatmaps(
    spearman_matrix: pd.DataFrame,
    jaccard_matrix: pd.DataFrame,
    output_dir: Path,
):
    """Plot pairwise consistency matrices for Spearman rank correlation and top-10 Jaccard similarity."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

    # Spearman Heatmap
    sns.heatmap(
        spearman_matrix,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"label": "Spearman Rho"},
        ax=axes[0],
        square=True,
    )
    axes[0].set_title("Global Ranking Consistency\n(Spearman Rank Correlation)", fontsize=11, pad=10)

    # Jaccard Heatmap
    sns.heatmap(
        jaccard_matrix,
        annot=True,
        fmt=".3f",
        cmap="Oranges",
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"label": "Jaccard Index"},
        ax=axes[1],
        square=True,
    )
    axes[1].set_title("Top-10 Feature Set Consistency\n(Jaccard Similarity)", fontsize=11, pad=10)

    plt.tight_layout()
    fig_path = output_dir / "shap_consistency_matrices.png"
    fig.savefig(fig_path)
    plt.close(fig)
    logger.info(f"Saved consistency matrices to {fig_path}")


def plot_recommendation_concordance(
    rec_df: pd.DataFrame,
    summary: Dict[str, Any],
    output_dir: Path,
):
    """Plot retention action distributions and concordance rates."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # 1. Action distribution per model
    dist = summary.get("action_distribution_per_model", {})
    dist_df = pd.DataFrame(dist).fillna(0) * 100

    dist_df.plot(
        kind="bar",
        ax=axes[0],
        colormap="tab10",
        alpha=0.85,
        width=0.75,
    )
    axes[0].set_title("Retention Action Distribution by Model (% of Cohort)", fontsize=12)
    axes[0].set_ylabel("Proportion of Cohort (%)")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=25, ha="right", fontsize=9)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    axes[0].legend(title="Model")

    # 2. Pairwise Recommendation Agreement
    pairwise = summary.get("pairwise_action_agreement_pct", {})
    pairs = list(pairwise.keys())
    rates = list(pairwise.values())

    bars = axes[1].bar(
        [p.replace("_vs_", " vs\n") for p in pairs],
        rates,
        color=["#1f77b4", "#ff7f0e", "#2ca02c"],
        alpha=0.85,
        width=0.5,
    )
    axes[1].set_title("Pairwise Recommendation Action Agreement (%)", fontsize=12)
    axes[1].set_ylabel("Agreement Rate (%)")
    axes[1].set_ylim(0, 105)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        yval = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 1.5,
            f"{yval:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    plt.tight_layout()
    fig_path = output_dir / "retention_recommendation_concordance.png"
    fig.savefig(fig_path)
    plt.close(fig)
    logger.info(f"Saved recommendation concordance chart to {fig_path}")


def generate_all_visualizations(
    predictions_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    spearman_matrix: pd.DataFrame,
    jaccard_matrix: pd.DataFrame,
    rec_df: pd.DataFrame,
    rec_summary: Dict[str, Any],
    output_dir: Path,
):
    """Generate and save all publication-grade figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_roc_pr_curves(predictions_df, output_dir)
    plot_global_shap_importance(importance_df, output_dir)
    plot_consistency_heatmaps(spearman_matrix, jaccard_matrix, output_dir)
    plot_recommendation_concordance(rec_df, rec_summary, output_dir)
    logger.info("All visualizations successfully created.")
