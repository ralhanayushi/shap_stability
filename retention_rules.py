"""Rule-based retention recommendation engine translating SHAP explanations into operational interventions.

Addresses SRS Requirements:
- REQ-F-51: Transparent rule-based mapping from top positive churn drivers to retention actions.
- REQ-F-52: Documented justification for each business intervention rule.
- REQ-F-53: Independent recommendation generation per model (global and customer level).
- REQ-F-54: Recommendation concordance analysis across models.
- REQ-F-55: Advisory disclaimer (associational model explanation, not causal guarantee).
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from .config import RETENTION_ACTIONS

logger = logging.getLogger(__name__)


def map_feature_to_action(feature_name: str) -> Tuple[str, str]:
    """Map a specific churn driver feature to an operational retention action and justification (REQ-F-51, REQ-F-52)."""
    f = feature_name.lower()

    if f.startswith("log_"):
        action = "engagement_campaign"
        justification = (
            f"Feature '{feature_name}' indicates diminishing listening frequency, song completion, "
            "or prolonged inactivity. Automated playlist recommendations, artist updates, and push notifications "
            "can revive streaming engagement."
        )
    elif any(term in f for term in ["renew", "cancel", "expiry", "expire"]):
        action = "renewal_support"
        justification = (
            f"Feature '{feature_name}' points to subscription lapse risk, disabled auto-renewal, "
            "or pending membership expiration. Targeted renewal alerts, payment retry mechanisms, or auto-renew "
            "incentives are recommended."
        )
    elif any(term in f for term in ["price", "paid", "discount", "plan", "payment"]):
        action = "plan_or_price_communication"
        justification = (
            f"Feature '{feature_name}' reflects price sensitivity, plan tier misalignment, or payment method friction. "
            "Offering flexible tier options, promotional pricing, or preferred payment options can lower barrier to retention."
        )
    else:
        action = "general_review"
        justification = (
            f"Feature '{feature_name}' relates to demographic tenure or account characteristics. "
            "A standard loyalty check-in or service satisfaction outreach is advised."
        )

    return action, justification


def derive_customer_recommendations(
    shap_matrices: Dict[str, np.ndarray],
    feature_names: List[str],
    selected_customers_df: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Derive rule-based retention actions for each customer under each model's explanations (REQ-F-53, REQ-F-54)."""
    models = list(shap_matrices.keys())
    n_customers = len(selected_customers_df)
    customer_ids = selected_customers_df.index.tolist()

    records = []

    for i in range(n_customers):
        msno = customer_ids[i]
        row = {"msno": msno}
        actions_for_customer = {}

        for name in models:
            shap_vec = shap_matrices[name][i]
            max_idx = int(np.argmax(shap_vec))
            max_val = float(shap_vec[max_idx])

            # If even the highest SHAP contribution is <= 0, the model sees no positive churn pressure
            if max_val <= 0:
                top_feature = "none"
                act = "no_positive_driver"
                just = "No feature contributed positively to churn risk."
            else:
                top_feature = feature_names[max_idx]
                act, just = map_feature_to_action(top_feature)

            actions_for_customer[name] = act
            row[f"{name}_top_feature"] = top_feature
            row[f"{name}_shap_value"] = round(max_val, 4)
            row[f"{name}_action"] = act
            row[f"{name}_justification"] = just

        # Multi-model recommendation concordance
        unique_actions = set(actions_for_customer.values())
        row["all_models_agree_action"] = len(unique_actions) == 1
        row["concordant_action"] = list(unique_actions)[0] if len(unique_actions) == 1 else "divergent"
        records.append(row)

    rec_df = pd.DataFrame(records).set_index("msno")

    # Concordance summary
    total = len(rec_df)
    full_agreement_pct = round(float(rec_df["all_models_agree_action"].mean() * 100), 2)

    pairwise_agreements = {}
    pairs = [
        ("RandomForest", "XGBoost"),
        ("RandomForest", "CatBoost"),
        ("XGBoost", "CatBoost"),
    ]
    for a, b in pairs:
        pair_agree = (rec_df[f"{a}_action"] == rec_df[f"{b}_action"]).mean() * 100
        pairwise_agreements[f"{a}_vs_{b}"] = round(float(pair_agree), 2)

    action_distribution = {}
    for name in models:
        action_distribution[name] = rec_df[f"{name}_action"].value_counts(normalize=True).round(4).to_dict()

    summary = {
        "total_evaluated_customers": total,
        "full_concordance_rate_pct": full_agreement_pct,
        "pairwise_action_agreement_pct": pairwise_agreements,
        "action_distribution_per_model": action_distribution,
        "advisory_disclaimer": (
            "Retention recommendations are generated based on associative feature attributions from "
            "trained machine learning models. They do not constitute proven causal guarantees."
        ),
    }

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        rec_df.to_csv(output_dir / "retention_recommendations.csv")
        with open(output_dir / "recommendation_concordance_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    return rec_df, summary
