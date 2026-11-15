"""Cross-model explanation stability evaluation module.

Addresses SRS Requirements:
- REQ-F-43: Jaccard similarity J(A, B) = |A ∩ B| / |A ∪ B| for top-K features (K=5, 10, 15).
- REQ-F-44: Spearman rank correlation of global feature importance rankings across model pairs.
- REQ-F-45: Per-customer SHAP vector correlation, top-5 overlap, and sign agreement.
- REQ-F-46: Grouped breakdown of stability measures across customer subgroups.
- REQ-F-47: Separate reporting of prediction agreement vs explanation agreement.
- REQ-F-48: Pairwise consistency matrices generation.
- REQ-F-49: Rigorous empirical reporting of stability measurements.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

logger = logging.getLogger(__name__)


def compute_jaccard_similarity(set_a: set, set_b: set) -> float:
    """Calculate Jaccard index J(A, B) = |A ∩ B| / |A ∪ B|."""
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return round(float(intersection / max(1, union)), 4)


def compute_global_stability(
    importance_df: pd.DataFrame,
    top_k_values: List[int] = [5, 10, 15],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute global Jaccard similarities, Spearman rank correlations, and consistency matrices (REQ-F-43, REQ-F-44)."""
    models = importance_df.columns.tolist()
    pairs = [
        ("RandomForest", "XGBoost"),
        ("RandomForest", "CatBoost"),
        ("XGBoost", "CatBoost"),
    ]

    records = []
    spearman_matrix = pd.DataFrame(1.0, index=models, columns=models)
    jaccard_matrices = {k: pd.DataFrame(1.0, index=models, columns=models) for k in top_k_values}

    for a, b in pairs:
        if a not in importance_df.columns or b not in importance_df.columns:
            continue

        rank_a = importance_df[a].rank(ascending=False)
        rank_b = importance_df[b].rank(ascending=False)
        rho = spearmanr(rank_a, rank_b).statistic
        spearman_matrix.loc[a, b] = round(float(rho), 4)
        spearman_matrix.loc[b, a] = round(float(rho), 4)

        row = {
            "model_pair": f"{a} vs {b}",
            "spearman_rank_correlation": round(float(rho), 4),
        }

        # Jaccard similarities at top-K
        ranked_a = importance_df[a].sort_values(ascending=False).index.tolist()
        ranked_b = importance_df[b].sort_values(ascending=False).index.tolist()

        for k in top_k_values:
            top_a_set = set(ranked_a[:k])
            top_b_set = set(ranked_b[:k])
            j_score = compute_jaccard_similarity(top_a_set, top_b_set)
            row[f"jaccard_top_{k}"] = j_score
            row[f"overlap_top_{k}"] = len(top_a_set.intersection(top_b_set))
            jaccard_matrices[k].loc[a, b] = j_score
            jaccard_matrices[k].loc[b, a] = j_score

        records.append(row)

    stability_df = pd.DataFrame(records)
    return stability_df, spearman_matrix, jaccard_matrices[10]


def compute_per_customer_stability(
    shap_matrices: Dict[str, np.ndarray],
    feature_names: List[str],
    selected_customers_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate per-customer vector correlation, top-5 overlap, and sign agreement (REQ-F-45)."""
    models = list(shap_matrices.keys())
    pairs = [
        ("RandomForest", "XGBoost"),
        ("RandomForest", "CatBoost"),
        ("XGBoost", "CatBoost"),
    ]

    n_customers = len(selected_customers_df)
    customer_ids = selected_customers_df.index.tolist()

    records = []

    for i in range(n_customers):
        msno = customer_ids[i]
        c_row = {"msno": msno}

        for a, b in pairs:
            if a not in shap_matrices or b not in shap_matrices:
                continue

            v_a = shap_matrices[a][i]
            v_b = shap_matrices[b][i]

            # Vector Pearson correlation
            std_a, std_b = np.std(v_a), np.std(v_b)
            if std_a > 1e-7 and std_b > 1e-7:
                vec_corr = float(pearsonr(v_a, v_b)[0])
            else:
                vec_corr = 0.0

            # Sign agreement: fraction of features where sign(SHAP_a) == sign(SHAP_b)
            sign_a = np.sign(v_a)
            sign_b = np.sign(v_b)
            sign_agree = float(np.mean(sign_a == sign_b))

            # Top-5 positive churn drivers overlap
            top5_a = set(np.argsort(-v_a)[:5])
            top5_b = set(np.argsort(-v_b)[:5])
            top5_jaccard = compute_jaccard_similarity(top5_a, top5_b)
            top5_overlap = len(top5_a.intersection(top5_b))

            c_row[f"{a}_{b}_vector_corr"] = round(vec_corr, 4)
            c_row[f"{a}_{b}_sign_agreement"] = round(sign_agree, 4)
            c_row[f"{a}_{b}_top5_overlap"] = top5_overlap
            c_row[f"{a}_{b}_top5_jaccard"] = top5_jaccard

        records.append(c_row)

    per_customer_df = pd.DataFrame(records).set_index("msno")

    # Merge metadata for subgroup aggregation (REQ-F-46)
    merged_df = per_customer_df.join(selected_customers_df)

    # Subgroup summary statistics
    subgroup_summaries = []
    subgroups = {
        "Overall Cohort": merged_df,
        "Full Model Agreement": merged_df[merged_df.get("full_agreement", False)],
        "Model Disagreement": merged_df[merged_df.get("disagreement", False)],
        "Actual Churners": merged_df[merged_df.get("actual_is_churn", 0) == 1],
        "Actual Non-Churners": merged_df[merged_df.get("actual_is_churn", 0) == 0],
    }

    numeric_cols = [c for c in per_customer_df.columns if c != "msno"]

    for grp_name, grp_df in subgroups.items():
        if len(grp_df) == 0:
            continue
        means = grp_df[numeric_cols].mean().round(4).to_dict()
        subgroup_summaries.append({
            "subgroup": grp_name,
            "sample_size": len(grp_df),
            **means,
        })

    subgroup_summary_df = pd.DataFrame(subgroup_summaries)
    return per_customer_df, subgroup_summary_df


def run_stability_evaluation(
    importance_df: pd.DataFrame,
    shap_matrices: Dict[str, np.ndarray],
    selected_customers_df: pd.DataFrame,
    feature_names: List[str],
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute complete explanation stability evaluation."""
    logger.info("Executing explanation stability evaluation...")
    global_stab_df, spearman_matrix, jaccard_matrix = compute_global_stability(importance_df)
    per_cust_df, subgroup_summary_df = compute_per_customer_stability(
        shap_matrices, feature_names, selected_customers_df
    )

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        global_stab_df.to_csv(output_dir / "global_stability_results.csv", index=False)
        spearman_matrix.to_csv(output_dir / "spearman_consistency_matrix.csv")
        jaccard_matrix.to_csv(output_dir / "jaccard_top10_consistency_matrix.csv")
        per_cust_df.to_csv(output_dir / "per_customer_stability.csv")
        subgroup_summary_df.to_csv(output_dir / "subgroup_stability_summary.csv", index=False)

    return {
        "global_stability": global_stab_df,
        "spearman_matrix": spearman_matrix,
        "jaccard_matrix": jaccard_matrix,
        "per_customer": per_cust_df,
        "subgroup_summary": subgroup_summary_df,
    }
