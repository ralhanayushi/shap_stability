"""Leakage-safe feature engineering pipeline for KKBOX churn prediction.

Addresses SRS Requirements:
- REQ-F-09: Prediction cutoff enforcement (strictly records <= cutoff).
- REQ-F-10: Configurable look-back window.
- REQ-F-11: Automated leakage check ensuring max(record_date) <= cutoff.
- REQ-F-12: Aggregating transaction and user-log rows to customer level.
- REQ-F-13: Handling missing values and duplicate rows.
- REQ-F-14: Age filtering (10 <= bd <= 100, others set to NaN).
- REQ-F-15: Demographics, subscription, and listening features.
- REQ-F-16: Equivalent features across models via matrix transformation and imputation.
- REQ-F-19: Feature dictionary generation and audit logging.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from .config import CATEGORICAL_COLS, SEED

logger = logging.getLogger(__name__)


def clean_members(members_df: pd.DataFrame, cutoff_date: pd.Timestamp) -> pd.DataFrame:
    """Clean member demographics, filter implausible ages, and calculate tenure."""
    df = members_df.copy().drop_duplicates("msno")

    # Filter implausible age values (REQ-F-14: 10 <= bd <= 100)
    df.loc[(df["bd"] < 10) | (df["bd"] > 100), "bd"] = np.nan

    # Registration tenure in days up to cutoff (REQ-F-15)
    df["registration_init_time"] = pd.to_datetime(
        df["registration_init_time"].astype("Int64").astype(str),
        format="%Y%m%d",
        errors="coerce",
    )
    df["tenure_days"] = (cutoff_date - df["registration_init_time"]).dt.days
    # Negative tenure (registered after cutoff) indicates invalid/post-cutoff registration
    df.loc[df["tenure_days"] < 0, "tenure_days"] = np.nan

    df = df.drop(columns=["registration_init_time"], errors="ignore")
    return df.set_index("msno")


def compute_transaction_features(tx_df: pd.DataFrame, cutoff_date: pd.Timestamp) -> pd.DataFrame:
    """Aggregate transaction history into customer-level subscription features (Leakage-Safe)."""
    if tx_df.empty:
        return pd.DataFrame(index=pd.Index([], name="msno"))

    # Automated leakage verification (REQ-F-11)
    max_tx_date = tx_df["transaction_date"].max()
    if pd.notna(max_tx_date) and max_tx_date > cutoff_date:
        raise ValueError(
            f"LEAKAGE DETECTED in transactions! Max transaction date {max_tx_date} > cutoff {cutoff_date}"
        )

    tx_df = tx_df.sort_values(["msno", "transaction_date"])
    g = tx_df.groupby("msno", sort=False)

    # Historical aggregations
    agg_df = g.agg(
        tx_count=("transaction_date", "size"),
        tx_last_date=("transaction_date", "max"),
        tx_total_paid=("actual_amount_paid", "sum"),
        tx_mean_paid=("actual_amount_paid", "mean"),
        tx_mean_plan_days=("payment_plan_days", "mean"),
        tx_mean_list_price=("plan_list_price", "mean"),
        tx_cancel_rate=("is_cancel", "mean"),
        tx_auto_renew_rate=("is_auto_renew", "mean"),
    )

    agg_df["tx_days_since_last"] = (cutoff_date - agg_df.pop("tx_last_date")).dt.days

    # Discount rate feature
    price_diff = tx_df["plan_list_price"] - tx_df["actual_amount_paid"]
    discount_ratio = price_diff / (tx_df["plan_list_price"] + 1e-5)
    tx_df["discount_ratio"] = discount_ratio.clip(lower=0.0)
    agg_df["tx_mean_discount_rate"] = tx_df.groupby("msno", sort=False)["discount_ratio"].mean()

    # Most recent transaction characteristics
    last_tx = g.tail(1).set_index("msno")
    for col in ["payment_method_id", "payment_plan_days", "actual_amount_paid", "is_auto_renew", "is_cancel"]:
        col_name = f"last_{col}" if col != "payment_method_id" else col
        agg_df[col_name] = last_tx[col]

    # Expiry relationship relative to cutoff
    agg_df["tx_days_until_expiry"] = (last_tx["membership_expire_date"] - cutoff_date).dt.days
    agg_df["is_expired_at_cutoff"] = (agg_df["tx_days_until_expiry"] < 0).astype(int)

    return agg_df


def compute_log_features(logs_df: pd.DataFrame, cutoff_date: pd.Timestamp) -> pd.DataFrame:
    """Aggregate listening log history into customer-level activity features (Leakage-Safe)."""
    if logs_df.empty:
        return pd.DataFrame(index=pd.Index([], name="msno"))

    # Automated leakage verification (REQ-F-11)
    max_log_date = logs_df["date"].max()
    if pd.notna(max_log_date) and max_log_date > cutoff_date:
        raise ValueError(
            f"LEAKAGE DETECTED in user logs! Max log date {max_log_date} > cutoff {cutoff_date}"
        )

    # Compute song counts
    total_songs = (
        logs_df["num_25"]
        + logs_df["num_50"]
        + logs_df["num_75"]
        + logs_df["num_985"]
        + logs_df["num_100"]
    )
    logs_df["total_songs"] = total_songs

    g = logs_df.groupby("msno", sort=False)
    agg_df = g.agg(
        log_active_days=("date", "nunique"),
        log_last_active_date=("date", "max"),
        log_total_secs=("total_secs", "sum"),
        log_mean_secs_per_day=("total_secs", "mean"),
        log_total_unq=("num_unq", "sum"),
        log_total_songs=("total_songs", "sum"),
        log_sum_num_100=("num_100", "sum"),
        log_sum_num_25=("num_25", "sum"),
    )

    agg_df["log_days_since_last"] = (cutoff_date - agg_df.pop("log_last_active_date")).dt.days
    agg_df["log_completion_ratio"] = (
        agg_df.pop("log_sum_num_100") / (agg_df["log_total_songs"] + 1e-5)
    ).clip(0.0, 1.0)
    agg_df["log_skip_ratio"] = (
        agg_df.pop("log_sum_num_25") / (agg_df["log_total_songs"] + 1e-5)
    ).clip(0.0, 1.0)

    return agg_df


def build_customer_dataset(
    labels_df: pd.DataFrame,
    members_df: pd.DataFrame,
    tx_df: pd.DataFrame,
    logs_df: pd.DataFrame,
    cutoff_date: pd.Timestamp,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Assemble customer-level dataset joined by msno."""
    labels = labels_df.drop_duplicates("msno").set_index("msno")

    cleaned_members = clean_members(members_df, cutoff_date)
    tx_features = compute_transaction_features(tx_df, cutoff_date)
    log_features = compute_log_features(logs_df, cutoff_date)

    # Left join all feature sets onto labels
    dataset = labels.join(cleaned_members, how="left")
    dataset = dataset.join(tx_features, how="left")
    dataset = dataset.join(log_features, how="left")

    audit = {
        "cutoff_date": str(cutoff_date.date()),
        "total_customers": len(dataset),
        "churn_rate": float(dataset["is_churn"].mean()),
        "feature_count": int(dataset.shape[1] - 1),  # excluding is_churn
        "tx_feature_coverage": float((~dataset["tx_count"].isna()).mean()),
        "log_feature_coverage": float((~dataset["log_active_days"].isna()).mean()),
        "member_feature_coverage": float((~dataset["city"].isna()).mean()),
    }
    logger.info(f"Dataset assembled: {audit}")
    return dataset, audit


def prepare_feature_matrices(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
    imputer: Optional[SimpleImputer] = None,
    feature_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame], SimpleImputer, List[str]]:
    """Encode categoricals, align columns across splits, and impute missing values (REQ-F-16)."""
    def _encode_single(df: pd.DataFrame) -> pd.DataFrame:
        X = df.drop(columns=["is_churn"], errors="ignore").copy()
        for c in CATEGORICAL_COLS:
            if c in X.columns:
                X[c] = X[c].astype("string").fillna("Unknown")
        X = pd.get_dummies(X, columns=[c for c in CATEGORICAL_COLS if c in X.columns], dtype=np.int8)
        X = X.replace([np.inf, -np.inf], np.nan)
        return X

    X_train_raw = _encode_single(train_df)

    if feature_columns is None:
        feature_columns = X_train_raw.columns.tolist()

    X_train_aligned = X_train_raw.reindex(columns=feature_columns, fill_value=0)

    if imputer is None:
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        X_train_mat = imputer.fit_transform(X_train_aligned)
    else:
        X_train_mat = imputer.transform(X_train_aligned)

    X_train = pd.DataFrame(X_train_mat, index=train_df.index, columns=feature_columns).astype(np.float32)

    X_val = None
    if val_df is not None:
        X_val_raw = _encode_single(val_df).reindex(columns=feature_columns, fill_value=0)
        X_val_mat = imputer.transform(X_val_raw)
        X_val = pd.DataFrame(X_val_mat, index=val_df.index, columns=feature_columns).astype(np.float32)

    X_test = None
    if test_df is not None:
        X_test_raw = _encode_single(test_df).reindex(columns=feature_columns, fill_value=0)
        X_test_mat = imputer.transform(X_test_raw)
        X_test = pd.DataFrame(X_test_mat, index=test_df.index, columns=feature_columns).astype(np.float32)

    return X_train, X_val, X_test, imputer, feature_columns


def generate_feature_dictionary(columns: List[str]) -> Dict[str, Dict[str, str]]:
    """Generate comprehensive feature dictionary describing each engineered feature (REQ-F-19)."""
    dictionary = {}
    for col in columns:
        if col.startswith("tx_"):
            table = "transactions"
            desc = f"Aggregated subscription metric: {col.replace('tx_', '')}"
        elif col.startswith("last_"):
            table = "transactions"
            desc = f"Latest subscription status: {col.replace('last_', '')}"
        elif col.startswith("log_"):
            table = "user_logs"
            desc = f"Aggregated listening metric: {col.replace('log_', '')}"
        elif col in ("tenure_days", "bd"):
            table = "members"
            desc = "Member demographic characteristic: " + ("Tenure in days" if col == "tenure_days" else "Age in years")
        elif any(col.startswith(cat + "_") for cat in CATEGORICAL_COLS):
            table = "members/transactions"
            desc = f"One-hot indicator for category: {col}"
        else:
            table = "computed"
            desc = f"Derived churn feature: {col}"

        dictionary[col] = {
            "source_table": table,
            "description": desc,
            "data_type": "float32",
        }
    return dictionary
