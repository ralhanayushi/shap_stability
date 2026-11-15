"""Central configuration for SHAP Explanation Stability Evaluation System."""
from pathlib import Path
import pandas as pd

# Random Seed for Reproducibility (REQ-F-24, REQ-F-27)
SEED = 42

# Prediction Cutoffs (REQ-F-09)
# Early cohort: features strictly <= 2017-01-31, predicting churn in Feb 2017
EARLY_CUTOFF = pd.Timestamp("2017-01-31")
# Late cohort (out-of-time evaluation): features strictly <= 2017-02-28, predicting churn in Mar 2017
LATE_CUTOFF = pd.Timestamp("2017-02-28")

# Look-back window in days (REQ-F-10)
LOOKBACK_WINDOW_DAYS = 60

# Categorical columns to be dummy-encoded or handled
CATEGORICAL_COLS = ["city", "gender", "registered_via", "payment_method_id"]

# Raw columns required from tables
TRANSACTION_COLS = [
    "msno",
    "payment_method_id",
    "payment_plan_days",
    "plan_list_price",
    "actual_amount_paid",
    "is_auto_renew",
    "transaction_date",
    "membership_expire_date",
    "is_cancel",
]

LOG_COLS = [
    "msno",
    "date",
    "num_25",
    "num_50",
    "num_75",
    "num_985",
    "num_100",
    "num_unq",
    "total_secs",
]

MEMBER_COLS = [
    "msno",
    "city",
    "bd",
    "gender",
    "registered_via",
    "registration_init_time",
]

# Hyperparameter Search Grids for Models (REQ-F-23)
RF_PARAM_GRID = [
    {"n_estimators": 100, "max_depth": 10, "min_samples_leaf": 5},
    {"n_estimators": 160, "max_depth": 14, "min_samples_leaf": 4},
    {"n_estimators": 200, "max_depth": 16, "min_samples_leaf": 2},
    {"n_estimators": 250, "max_depth": 18, "min_samples_leaf": 2},
]

XGB_PARAM_GRID = [
    {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
    {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
    {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.03, "subsample": 0.85, "colsample_bytree": 0.85},
    {"n_estimators": 350, "max_depth": 7, "learning_rate": 0.03, "subsample": 0.9, "colsample_bytree": 0.8},
]

CAT_PARAM_GRID = [
    {"iterations": 250, "depth": 5, "learning_rate": 0.05},
    {"iterations": 400, "depth": 6, "learning_rate": 0.05},
    {"iterations": 500, "depth": 7, "learning_rate": 0.04},
    {"iterations": 600, "depth": 6, "learning_rate": 0.03},
]

# Retention Action Definitions (REQ-F-51, REQ-F-52)
RETENTION_ACTIONS = {
    "engagement_campaign": "Target with personalized playlists, push notifications, and high-energy music recommendations to revitalize declining listening activity.",
    "renewal_support": "Deploy renewal incentives, automated auto-renew opt-in prompts, or proactive grace-period notices to prevent lapse at expiration.",
    "plan_or_price_communication": "Offer targeted subscription discounts, flexible tenure plans, or alternative payment methods to resolve pricing friction.",
    "general_review": "Standard low-friction customer service check-in and product satisfaction outreach.",
    "no_positive_driver": "No actionable positive churn risk driver identified by model.",
}
