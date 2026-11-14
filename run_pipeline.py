"""End-to-End Orchestrator for SHAP Explanation Stability Evaluation System.

Runs the entire research pipeline:
1. Data inspection and leakage-safe cohort extraction.
2. Chunked feature engineering across transactions, listening logs, and member profiles.
3. Matrix transformation and median imputation (fit strictly on training data).
4. Hyperparameter search and training for Random Forest, XGBoost, and CatBoost.
5. Model evaluation (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix).
6. TreeExplainer SHAP computation for global and representative customer cohorts.
7. Cross-model stability metrics (Jaccard top-K, Spearman rho, per-customer vector correlation).
8. Rule-based retention recommendation concordance analysis.
9. Publication-ready chart generation.
"""
import argparse
import json
import logging
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Project modules
from src.config import EARLY_CUTOFF, LATE_CUTOFF, LOOKBACK_WINDOW_DAYS, SEED
from src.data_loader import (
    inspect_raw_files,
    stream_transactions_for_users,
    stream_user_logs_for_users,
)
from src.evaluation import evaluate_models
from src.feature_engineering import (
    build_customer_dataset,
    generate_feature_dictionary,
    prepare_feature_matrices,
)
from src.models import ChurnModelSuite
from src.retention_rules import derive_customer_recommendations
from src.shap_explainer import run_shap_analysis
from src.stability import run_stability_evaluation
from src.visualization import generate_all_visualizations

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("SHAP_Pipeline")


def parse_args():
    parser = argparse.ArgumentParser(description="SHAP Explanation Stability Evaluation Pipeline")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Directory containing raw KKBOX files")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to store research outputs")
    parser.add_argument("--max-users", type=int, default=25000, help="Number of customer labels to sample (per cohort)")
    parser.add_argument("--max-tx-rows", type=int, default=2000000, help="Max transaction rows to process")
    parser.add_argument("--max-log-rows", type=int, default=3000000, help="Max log rows to stream")
    parser.add_argument("--background-size", type=int, default=100, help="Background sample size for SHAP TreeExplainer")
    parser.add_argument("--shap-sample-size", type=int, default=100, help="Number of test customers for local SHAP explanations")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for reproducibility")
    return parser.parse_args()


def main():
    args = parse_args()
    t_start = time.time()

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("STARTING SHAP EXPLANATION STABILITY EVALUATION PIPELINE")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Max sampled users: {args.max_users}")
    logger.info("=" * 80)

    # ---------------------------------------------------------
    # Stage 1: Inspect Raw Data & Generate Audit Report
    # ---------------------------------------------------------
    logger.info("\n--- STAGE 1: RAW DATA INSPECTION ---")
    inspection_report = inspect_raw_files(data_dir, output_file=output_dir / "inspection_report.json")
    logger.info(f"Found raw files: {list(inspection_report['files'].keys())}")

    # ---------------------------------------------------------
    # Stage 2: Cohort Selection & Leakage Isolation
    # ---------------------------------------------------------
    logger.info("\n--- STAGE 2: COHORT SELECTION & LEAKAGE ISOLATION ---")
    train_labels_file = data_dir / "train.csv"
    train_v2_labels_file = data_dir / "train_v2.csv"

    if not train_labels_file.exists() or not train_v2_labels_file.exists():
        raise FileNotFoundError("Missing train.csv or train_v2.csv in data directory!")

    df_early_labels = pd.read_csv(train_labels_file, usecols=["msno", "is_churn"]).drop_duplicates("msno")
    df_late_labels = pd.read_csv(train_v2_labels_file, usecols=["msno", "is_churn"]).drop_duplicates("msno")

    # Sample if requested
    if args.max_users and len(df_early_labels) > args.max_users:
        df_early_labels = df_early_labels.sample(n=args.max_users, random_state=args.seed)
    if args.max_users and len(df_late_labels) > args.max_users:
        df_late_labels = df_late_labels.sample(n=args.max_users, random_state=args.seed)

    # Strict Leakage Isolation: remove any early cohort customer who also appears in late cohort
    # to guarantee zero customer identity overlap between training and out-of-time evaluation
    df_early_labels = df_early_labels[~df_early_labels["msno"].isin(set(df_late_labels["msno"]))]

    logger.info(f"Early cohort (development): {len(df_early_labels)} customers (Cutoff: {EARLY_CUTOFF.date()})")
    logger.info(f"Late cohort (out-of-time test): {len(df_late_labels)} customers (Cutoff: {LATE_CUTOFF.date()})")

    all_target_users = set(df_early_labels["msno"]).union(set(df_late_labels["msno"]))
    logger.info(f"Total unique target users across both cohorts: {len(all_target_users)}")

    # ---------------------------------------------------------
    # Stage 3: Load Members Demographics
    # ---------------------------------------------------------
    logger.info("\n--- STAGE 3: LOADING DEMOGRAPHICS ---")
    members_file = data_dir / "members_v3.csv"
    if not members_file.exists():
        raise FileNotFoundError(f"Missing {members_file}")

    members_df = pd.read_csv(
        members_file,
        usecols=["msno", "city", "bd", "gender", "registered_via", "registration_init_time"],
    )
    members_df = members_df[members_df["msno"].isin(all_target_users)]
    logger.info(f"Loaded demographic records for {len(members_df)} target customers.")

    # ---------------------------------------------------------
    # Stage 4: Stream Transactions & Listening Logs (Leakage-Safe)
    # ---------------------------------------------------------
    logger.info("\n--- STAGE 4: STREAMING TRANSACTIONS & LISTENING LOGS ---")
    tx_files = [data_dir / "transactions.csv", data_dir / "transactions_v2.csv"]
    log_files = [data_dir / "user_logs.csv.7z", data_dir / "user_logs.csv", data_dir / "user_logs_v2.csv"]

    # 4A. Early Cohort Streams (cutoff <= 2017-01-31)
    logger.info("Extracting early cohort transactions (<= 2017-01-31)...")
    early_tx = stream_transactions_for_users(
        tx_files,
        set(df_early_labels["msno"]),
        cutoff_date=EARLY_CUTOFF,
        max_rows=args.max_tx_rows,
    )
    logger.info(f"Early transactions extracted: {len(early_tx)} rows.")

    logger.info("Extracting early cohort user logs (<= 2017-01-31)...")
    early_logs = stream_user_logs_for_users(
        log_files,
        set(df_early_labels["msno"]),
        cutoff_date=EARLY_CUTOFF,
        lookback_window_days=LOOKBACK_WINDOW_DAYS,
        max_rows=args.max_log_rows,
    )
    logger.info(f"Early user logs extracted: {len(early_logs)} rows.")

    # 4B. Late Cohort Streams (cutoff <= 2017-02-28)
    logger.info("Extracting late cohort transactions (<= 2017-02-28)...")
    late_tx = stream_transactions_for_users(
        tx_files,
        set(df_late_labels["msno"]),
        cutoff_date=LATE_CUTOFF,
        max_rows=args.max_tx_rows,
    )
    logger.info(f"Late transactions extracted: {len(late_tx)} rows.")

    logger.info("Extracting late cohort user logs (<= 2017-02-28)...")
    late_logs = stream_user_logs_for_users(
        log_files,
        set(df_late_labels["msno"]),
        cutoff_date=LATE_CUTOFF,
        lookback_window_days=LOOKBACK_WINDOW_DAYS,
        max_rows=args.max_log_rows,
    )
    logger.info(f"Late user logs extracted: {len(late_logs)} rows.")

    # ---------------------------------------------------------
    # Stage 5: Feature Engineering & Customer Dataset Assembly
    # ---------------------------------------------------------
    logger.info("\n--- STAGE 5: FEATURE ENGINEERING ---")
    early_dataset, audit_early = build_customer_dataset(
        df_early_labels, members_df, early_tx, early_logs, EARLY_CUTOFF
    )
    late_dataset, audit_late = build_customer_dataset(
        df_late_labels, members_df, late_tx, late_logs, LATE_CUTOFF
    )

    with open(output_dir / "cohort_audit.json", "w") as f:
        json.dump([audit_early, audit_late], f, indent=2)

    # ---------------------------------------------------------
    # Stage 6: Train / Validation Split & Feature Matrix Encoding
    # ---------------------------------------------------------
    logger.info("\n--- STAGE 6: DATA SPLITTING & IMPUTATION ---")
    y_early = early_dataset["is_churn"].astype(int)
    y_late = late_dataset["is_churn"].astype(int)

    tr_idx, val_idx = train_test_split(
        np.arange(len(early_dataset)),
        test_size=0.20,
        random_state=args.seed,
        stratify=y_early,
    )

    dev_train = early_dataset.iloc[tr_idx]
    dev_val = early_dataset.iloc[val_idx]
    test_oot = late_dataset

    logger.info(f"Development Train: {len(dev_train)} customers (Churn: {dev_train['is_churn'].mean():.3f})")
    logger.info(f"Development Val:   {len(dev_val)} customers (Churn: {dev_val['is_churn'].mean():.3f})")
    logger.info(f"Out-of-time Test:  {len(test_oot)} customers (Churn: {test_oot['is_churn'].mean():.3f})")

    # Fit preprocessing strictly on dev_train!
    X_train, X_val, X_test, imputer, feature_cols = prepare_feature_matrices(
        train_df=dev_train,
        val_df=dev_val,
        test_df=test_oot,
    )

    y_train = dev_train["is_churn"].astype(int)
    y_val = dev_val["is_churn"].astype(int)
    y_test = test_oot["is_churn"].astype(int)

    # Save imputer and feature dictionary
    joblib.dump(imputer, output_dir / "imputer.joblib")
    with open(output_dir / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    feat_dict = generate_feature_dictionary(feature_cols)
    with open(output_dir / "feature_dictionary.json", "w") as f:
        json.dump(feat_dict, f, indent=2)

    logger.info(f"Engineered {len(feature_cols)} features per customer.")

    # ---------------------------------------------------------
    # Stage 7: Model Training & Hyperparameter Search
    # ---------------------------------------------------------
    logger.info("\n--- STAGE 7: MODEL TRAINING & HYPERPARAMETER SEARCH ---")
    model_suite = ChurnModelSuite(random_state=args.seed)
    models = model_suite.train_all(X_train, y_train, X_val, y_val)
    model_suite.save(output_dir)

    # ---------------------------------------------------------
    # Stage 8: Evaluation & Threshold Optimization
    # ---------------------------------------------------------
    logger.info("\n--- STAGE 8: MODEL EVALUATION & THRESHOLD OPTIMIZATION ---")
    metrics_df, test_preds, confusion_matrices = evaluate_models(
        models=models,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        output_dir=output_dir,
    )
    print("\nModel Evaluation Performance Summary:")
    print(metrics_df[metrics_df["split"] == "test"].to_string(index=False))

    # ---------------------------------------------------------
    # Stage 9: SHAP Explanation Generation
    # ---------------------------------------------------------
    logger.info("\n--- STAGE 9: SHAP EXPLANATION COMPUTATION ---")
    importance_df, shap_matrices, selected_customers_df, cohort_counts = run_shap_analysis(
        models=models,
        X_train=X_train,
        X_test=X_test,
        predictions_df=test_preds,
        background_size=args.background_size,
        sample_size=args.shap_sample_size,
        output_dir=output_dir,
    )
    print("\nTop 10 Global SHAP Feature Importances:")
    print(importance_df.head(10).to_string())

    # ---------------------------------------------------------
    # Stage 10: Explanation Stability Analysis
    # ---------------------------------------------------------
    logger.info("\n--- STAGE 10: EXPLANATION STABILITY EVALUATION ---")
    stability_res = run_stability_evaluation(
        importance_df=importance_df,
        shap_matrices=shap_matrices,
        selected_customers_df=selected_customers_df,
        feature_names=feature_cols,
        output_dir=output_dir,
    )
    print("\nGlobal Stability Metrics:")
    print(stability_res["global_stability"].to_string(index=False))
    print("\nSpearman Consistency Matrix:")
    print(stability_res["spearman_matrix"].to_string())
    print("\nSubgroup Stability Summary:")
    print(stability_res["subgroup_summary"].to_string(index=False))

    # ---------------------------------------------------------
    # Stage 11: Rule-Based Retention Recommendations
    # ---------------------------------------------------------
    logger.info("\n--- STAGE 11: RULE-BASED RETENTION RECOMMENDATIONS ---")
    rec_df, rec_summary = derive_customer_recommendations(
        shap_matrices=shap_matrices,
        feature_names=feature_cols,
        selected_customers_df=selected_customers_df,
        output_dir=output_dir,
    )
    print(f"\nRetention Recommendation Full Concordance Rate: {rec_summary['full_concordance_rate_pct']}%")
    print("Pairwise Action Concordance:", rec_summary["pairwise_action_agreement_pct"])

    # ---------------------------------------------------------
    # Stage 12: Publication Visualizations
    # ---------------------------------------------------------
    logger.info("\n--- STAGE 12: VISUALIZATION GENERATION ---")
    generate_all_visualizations(
        predictions_df=test_preds,
        importance_df=importance_df,
        spearman_matrix=stability_res["spearman_matrix"],
        jaccard_matrix=stability_res["jaccard_matrix"],
        rec_df=rec_df,
        rec_summary=rec_summary,
        output_dir=output_dir,
    )

    # ---------------------------------------------------------
    # Stage 13: Experiment Metadata & Reproducibility Logging
    # ---------------------------------------------------------
    logger.info("\n--- STAGE 13: REPRODUCIBILITY LOGGING ---")
    experiment_config = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(),
        "python_version": sys.version,
        "seed": args.seed,
        "early_cutoff": str(EARLY_CUTOFF.date()),
        "late_cutoff": str(LATE_CUTOFF.date()),
        "lookback_window_days": LOOKBACK_WINDOW_DAYS,
        "max_sampled_users": args.max_users,
        "background_size": args.background_size,
        "shap_sample_size": args.shap_sample_size,
        "total_runtime_seconds": round(time.time() - t_start, 2),
        "disclaimer": (
            "This system is a research prototype. Prediction performance confirms that models function as churn predictors; "
            "it is not the main research contribution. SHAP explains the internal associative behavior of trained models "
            "and does not prove causal mechanisms of customer churn."
        ),
    }
    with open(output_dir / "experiment_config.json", "w") as f:
        json.dump(experiment_config, f, indent=2)

    logger.info("=" * 80)
    logger.info(f"PIPELINE COMPLETED SUCCESSFULLY IN {time.time() - t_start:.2f} SECONDS!")
    logger.info(f"All research artifacts and figures saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
