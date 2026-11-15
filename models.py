"""Tree-based churn prediction models with hyperparameter search and class imbalance handling.

Addresses SRS Requirements:
- REQ-F-21: Random Forest, XGBoost, and CatBoost classifiers on identical training data.
- REQ-F-22: Class weighting to handle imbalance (balanced_subsample, scale_pos_weight, auto_class_weights).
- REQ-F-23: Small documented hyperparameter search on training and validation splits.
- REQ-F-24: Full reproducibility with fixed random seed and trial tracking.
- REQ-F-25: Test split isolation (never touched during tuning).
- REQ-F-26: Model and hyperparameter persistence.
- REQ-F-27: Explicit random seed across all libraries.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, log_loss, f1_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from .config import SEED, RF_PARAM_GRID, XGB_PARAM_GRID, CAT_PARAM_GRID

logger = logging.getLogger(__name__)


class ChurnModelSuite:
    """Encapsulates training, hyperparameter optimization, and saving of the 3 tree models."""

    def __init__(self, random_state: int = SEED):
        self.random_state = random_state
        self.models: Dict[str, Any] = {}
        self.best_params: Dict[str, Dict[str, Any]] = {}
        self.search_logs: List[Dict[str, Any]] = []

    def _compute_imbalance_ratio(self, y_train: pd.Series) -> float:
        num_neg = (y_train == 0).sum()
        num_pos = (y_train == 1).sum()
        return float(num_neg / max(1, num_pos))

    def tune_and_train_rf(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        param_grid: Optional[List[Dict[str, Any]]] = None,
    ) -> RandomForestClassifier:
        """Hyperparameter search and training for Random Forest (REQ-F-21, REQ-F-22, REQ-F-23)."""
        logger.info("Tuning Random Forest...")
        grid = param_grid or RF_PARAM_GRID
        best_score = -1.0
        best_cfg = grid[0]

        for trial_idx, cfg in enumerate(grid):
            rf = RandomForestClassifier(
                n_estimators=cfg["n_estimators"],
                max_depth=cfg["max_depth"],
                min_samples_leaf=cfg["min_samples_leaf"],
                class_weight="balanced_subsample",
                random_state=self.random_state,
                n_jobs=-1,
            )
            rf.fit(X_train, y_train)
            val_probs = rf.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, val_probs)
            f1 = f1_score(y_val, (val_probs >= 0.5).astype(int), zero_division=0)
            loss = log_loss(y_val, val_probs, labels=[0, 1])

            self.search_logs.append({
                "model": "RandomForest",
                "trial": trial_idx + 1,
                "params": cfg,
                "val_roc_auc": round(auc, 4),
                "val_f1": round(f1, 4),
                "val_log_loss": round(loss, 4),
            })
            if auc > best_score:
                best_score = auc
                best_cfg = cfg

        logger.info(f"Best Random Forest: AUC={best_score:.4f} with {best_cfg}")
        final_rf = RandomForestClassifier(
            n_estimators=best_cfg["n_estimators"],
            max_depth=best_cfg["max_depth"],
            min_samples_leaf=best_cfg["min_samples_leaf"],
            class_weight="balanced_subsample",
            random_state=self.random_state,
            n_jobs=-1,
        )
        final_rf.fit(X_train, y_train)
        self.models["RandomForest"] = final_rf
        self.best_params["RandomForest"] = best_cfg
        return final_rf

    def tune_and_train_xgb(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        param_grid: Optional[List[Dict[str, Any]]] = None,
    ) -> XGBClassifier:
        """Hyperparameter search and training for XGBoost (REQ-F-21, REQ-F-22, REQ-F-23)."""
        logger.info("Tuning XGBoost...")
        grid = param_grid or XGB_PARAM_GRID
        imbalance_ratio = self._compute_imbalance_ratio(y_train)

        best_score = -1.0
        best_cfg = grid[0]

        for trial_idx, cfg in enumerate(grid):
            xgb = XGBClassifier(
                n_estimators=cfg["n_estimators"],
                max_depth=cfg["max_depth"],
                learning_rate=cfg["learning_rate"],
                subsample=cfg.get("subsample", 0.8),
                colsample_bytree=cfg.get("colsample_bytree", 0.8),
                scale_pos_weight=imbalance_ratio,
                tree_method="hist",
                eval_metric="logloss",
                random_state=self.random_state,
                n_jobs=-1,
            )
            xgb.fit(X_train, y_train)
            val_probs = xgb.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, val_probs)
            f1 = f1_score(y_val, (val_probs >= 0.5).astype(int), zero_division=0)
            loss = log_loss(y_val, val_probs, labels=[0, 1])

            self.search_logs.append({
                "model": "XGBoost",
                "trial": trial_idx + 1,
                "params": cfg,
                "val_roc_auc": round(auc, 4),
                "val_f1": round(f1, 4),
                "val_log_loss": round(loss, 4),
            })
            if auc > best_score:
                best_score = auc
                best_cfg = cfg

        logger.info(f"Best XGBoost: AUC={best_score:.4f} with {best_cfg}")
        final_xgb = XGBClassifier(
            n_estimators=best_cfg["n_estimators"],
            max_depth=best_cfg["max_depth"],
            learning_rate=best_cfg["learning_rate"],
            subsample=best_cfg.get("subsample", 0.8),
            colsample_bytree=best_cfg.get("colsample_bytree", 0.8),
            scale_pos_weight=imbalance_ratio,
            tree_method="hist",
            eval_metric="logloss",
            random_state=self.random_state,
            n_jobs=-1,
        )
        final_xgb.fit(X_train, y_train)
        self.models["XGBoost"] = final_xgb
        self.best_params["XGBoost"] = best_cfg
        return final_xgb

    def tune_and_train_catboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        param_grid: Optional[List[Dict[str, Any]]] = None,
    ) -> CatBoostClassifier:
        """Hyperparameter search and training for CatBoost (REQ-F-21, REQ-F-22, REQ-F-23)."""
        logger.info("Tuning CatBoost...")
        grid = param_grid or CAT_PARAM_GRID
        best_score = -1.0
        best_cfg = grid[0]

        for trial_idx, cfg in enumerate(grid):
            cat = CatBoostClassifier(
                iterations=cfg["iterations"],
                depth=cfg["depth"],
                learning_rate=cfg["learning_rate"],
                loss_function="Logloss",
                auto_class_weights="Balanced",
                verbose=False,
                random_seed=self.random_state,
                thread_count=-1,
            )
            cat.fit(X_train, y_train)
            val_probs = cat.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, val_probs)
            f1 = f1_score(y_val, (val_probs >= 0.5).astype(int), zero_division=0)
            loss = log_loss(y_val, val_probs, labels=[0, 1])

            self.search_logs.append({
                "model": "CatBoost",
                "trial": trial_idx + 1,
                "params": cfg,
                "val_roc_auc": round(auc, 4),
                "val_f1": round(f1, 4),
                "val_log_loss": round(loss, 4),
            })
            if auc > best_score:
                best_score = auc
                best_cfg = cfg

        logger.info(f"Best CatBoost: AUC={best_score:.4f} with {best_cfg}")
        final_cat = CatBoostClassifier(
            iterations=best_cfg["iterations"],
            depth=best_cfg["depth"],
            learning_rate=best_cfg["learning_rate"],
            loss_function="Logloss",
            auto_class_weights="Balanced",
            verbose=False,
            random_seed=self.random_state,
            thread_count=-1,
        )
        final_cat.fit(X_train, y_train)
        self.models["CatBoost"] = final_cat
        self.best_params["CatBoost"] = best_cfg
        return final_cat

    def train_all(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Dict[str, Any]:
        """Run hyperparameter search and training across all three models."""
        self.tune_and_train_rf(X_train, y_train, X_val, y_val)
        self.tune_and_train_xgb(X_train, y_train, X_val, y_val)
        self.tune_and_train_catboost(X_train, y_train, X_val, y_val)
        return self.models

    def save(self, output_dir: Path):
        """Save trained models and hyperparameter search results (REQ-F-26)."""
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, model in self.models.items():
            joblib.dump(model, output_dir / f"{name}.joblib")
            logger.info(f"Saved {name} model to {output_dir / f'{name}.joblib'}")

        with open(output_dir / "best_hyperparameters.json", "w") as f:
            json.dump(self.best_params, f, indent=2)

        pd.DataFrame(self.search_logs).to_csv(output_dir / "hyperparameter_search_log.csv", index=False)
        logger.info("Saved hyperparameter search logs.")
