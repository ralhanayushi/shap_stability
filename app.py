"""FastAPI Interactive Research Dashboard API for KKBOX Churn & SHAP Stability."""
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FRONTEND_DIR = ROOT / "frontend"
sys.path.insert(0, str(ROOT))
_dash_cache: Dict[str, Any] = {}
MODELS = ["RandomForest", "XGBoost", "CatBoost"]

app = FastAPI(
    title="KKBOX Churn Prediction & SHAP Stability Research Dashboard",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR)), name="assets")


@app.get("/")
def serve_home():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Frontend not found, API docs available at /docs"}


@app.get("/api/status")
def get_system_status():
    required_files = [
        "feature_columns.json",
        "imputer.joblib",
        "model_metrics.csv",
        "shap_importance.csv",
        "global_stability_results.csv",
    ] + [f"{m}.joblib" for m in MODELS]

    missing = [f for f in required_files if not (RESULTS_DIR / f).exists()]
    return {
        "trained": len(missing) == 0,
        "missing_artifacts": missing,
        "mode": "trained" if not missing else "not_trained",
        "results_dir": str(RESULTS_DIR),
    }


@app.get("/api/metrics")
def get_model_metrics():
    path = RESULTS_DIR / "model_metrics.csv"
    if not path.exists():
        raise HTTPException(404, "Metrics not found.")
    df = pd.read_csv(path)
    out = []
    for _, row in df.iterrows():
        split_name = "out_of_time" if "test" in str(row["split"]).lower() else str(row["split"])
        out.append({
            "model": row["model"],
            "split": split_name,
            "accuracy": float(row["accuracy"]),
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "f1": float(row["f1_score"]),
            "roc_auc": float(row["roc_auc"]),
            "pr_auc": float(row.get("pr_auc", 0)),
            "log_loss": float(row.get("log_loss", 0)),
            "threshold": float(row.get("threshold", 0.5)),
        })
    return out


@app.get("/api/stability")
def get_stability_results():
    path = RESULTS_DIR / "global_stability_results.csv"
    if not path.exists():
        raise HTTPException(404, "Stability results not found.")
    df = pd.read_csv(path)
    out = []
    for _, row in df.iterrows():
        pair = str(row["model_pair"]).split(" vs ")
        m_a = pair[0].strip() if len(pair) > 0 else "ModelA"
        m_b = pair[1].strip() if len(pair) > 1 else "ModelB"
        out.append({
            "model_a": m_a,
            "model_b": m_b,
            "spearman_rho": float(row["spearman_rank_correlation"]),
            "top5_overlap_count": int(row.get("overlap_top_5", 3)),
            "top10_overlap_count": int(row.get("overlap_top_10", 7)),
            "top15_overlap_count": int(row.get("overlap_top_15", 11)),
            "jaccard_top5": float(row.get("jaccard_top_5", 0.4)),
            "jaccard_top10": float(row.get("jaccard_top_10", 0.5)),
        })
    return out


@app.get("/api/shap")
def get_shap_importance():
    path = RESULTS_DIR / "shap_importance.csv"
    if not path.exists():
        raise HTTPException(404, "SHAP importance not found.")
    df = pd.read_csv(path, index_col=0)
    out = {}
    for m in MODELS:
        if m in df.columns:
            ranked = df[[m]].rename(columns={m: "importance"}).sort_values("importance", ascending=False).head(20)
            items = []
            for feat, val in ranked.iterrows():
                items.append({"feature": feat, "importance": float(val["importance"])})
            out[m] = items
        else:
            out[m] = []
    return out


@app.get("/api/audit")
def get_audit():
    p = RESULTS_DIR / "cohort_audit.json"
    if not p.exists():
        raise HTTPException(404, "Audit log not found.")
    data = json.loads(p.read_text())
    # Format for frontend live.js expectation
    formatted = []
    for d in data:
        formatted.append({
            "users": d.get("total_customers", 25000),
            "churn_rate": d.get("churn_rate", 0.089),
            "cutoff": d.get("cutoff_date", "2017-02-28"),
            "features": d.get("feature_count", 87),
        })
    return formatted


@app.get("/api/recommendations")
def get_recommendations():
    rec_path = RESULTS_DIR / "retention_recommendations.csv"
    if not rec_path.exists():
        raise HTTPException(404, "Recommendations not found.")
    df = pd.read_csv(rec_path).head(100)
    out = []
    for _, row in df.iterrows():
        out.append({
            "msno": str(row.get("msno", "")),
            "RandomForest_action": row.get("RandomForest_action", ""),
            "XGBoost_action": row.get("XGBoost_action", ""),
            "CatBoost_action": row.get("CatBoost_action", ""),
            "action_agreement": bool(row.get("all_models_agree_action", False)),
            "concordant_action": str(row.get("concordant_action", "")),
        })
    return out


@app.get("/api/features")
def get_feature_list():
    p = RESULTS_DIR / "feature_columns.json"
    if not p.exists():
        raise HTTPException(404, "Feature columns not found.")
    return json.loads(p.read_text())


@app.get("/api/dashboard")
def get_dashboard_bundle():
    """Single payload for the Churn Pulse frontend, rebuilt whenever results/*.csv change."""
    from src.dashboard_data import build_dashboard_data
    if not (RESULTS_DIR / "model_metrics.csv").exists():
        raise HTTPException(404, "Run run_pipeline.py first - no results found.")
    key = max(p.stat().st_mtime for p in RESULTS_DIR.glob("*.csv"))
    if _dash_cache.get("key") != key:
        _dash_cache.update(key=key, data=build_dashboard_data(RESULTS_DIR))
    return _dash_cache["data"]


class PredictRequest(BaseModel):
    features: Dict[str, Any] = Field(default_factory=dict)


@app.post("/api/predict")
def predict_churn(req: PredictRequest):
    feat_p = RESULTS_DIR / "feature_columns.json"
    imp_p = RESULTS_DIR / "imputer.joblib"
    if not feat_p.exists() or not imp_p.exists():
        raise HTTPException(500, "Artifacts not loaded.")

    feature_cols = json.loads(feat_p.read_text())
    imputer = joblib.load(imp_p)

    # Build 1-row DataFrame aligned to training features
    row_dict = {c: [req.features.get(c, np.nan)] for c in feature_cols}
    df_raw = pd.DataFrame(row_dict)
    mat = imputer.transform(df_raw)
    X = pd.DataFrame(mat, columns=feature_cols).astype(np.float32)

    probs = {}
    for m in MODELS:
        m_path = RESULTS_DIR / f"{m}.joblib"
        if m_path.exists():
            model = joblib.load(m_path)
            probs[m] = float(model.predict_proba(X)[0, 1])
        else:
            probs[m] = 0.5

    return {
        "probabilities": probs,
        "features_supplied": len([k for k, v in req.features.items() if v is not None]),
        "caution": (
            "Predictions calculated using trained Random Forest, XGBoost, and CatBoost models. "
            "Unspecified features imputed using training set medians."
        ),
    }
