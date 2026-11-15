"""Builds the single JSON bundle consumed by the Churn Pulse dashboard (frontend/index.html).

Everything is derived from files the pipeline writes to results/, so re-running
run_pipeline.py automatically refreshes the dashboard.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = ["RandomForest", "XGBoost", "CatBoost"]


def build_dashboard_data(results_dir) -> dict:
    R = Path(results_dir)
    rd = lambda f: pd.read_csv(R / f)
    M = MODELS

    met = rd("model_metrics.csv")
    met = met[met.split.astype(str).str.contains("test")]
    tp = rd("test_predictions.csv")
    y = tp.actual_is_churn.values

    sweep, hist = {}, {"bins": [f"{i/10:.1f}" for i in range(10)]}
    for m in M:
        p = tp[m + "_prob"].values
        rows = []
        for t in np.arange(0.0, 1.0001, 0.01):
            pr = p >= t
            rows.append([round(float(t), 2), int((pr & (y == 1)).sum()), int((pr & (y == 0)).sum()),
                         int((~pr & (y == 1)).sum()), int((~pr & (y == 0)).sum())])
        sweep[m] = rows
        hist[m] = np.histogram(p, bins=10, range=(0, 1))[0].tolist()

    imp = rd("shap_importance.csv").set_index("feature")
    shap = {m: [[k, float(v)] for k, v in imp[m].sort_values(ascending=False).head(15).items()] for m in M}
    shap_all = {m: imp[m].sort_values(ascending=False).index.tolist()[:60] for m in M}
    fd = json.loads((R / "feature_dictionary.json").read_text())
    fdict = {k: v.get("description", "") for k, v in fd.items()}

    sel, ps, rec = rd("selected_customers_metadata.csv"), rd("per_customer_stability.csv"), rd("retention_recommendations.csv")
    sv = {m: rd(f"shap_values_{m}.csv").set_index("msno") for m in M}
    cust = []
    for i, r in sel.iterrows():
        pred = [int(r[m + "_pred"]) for m in M]
        maj, a = int(sum(pred) >= 2), int(r.actual_is_churn)
        g = ("TP" if a else "FP") if maj else ("FN" if a else "TN")
        top = {}
        for m in M:
            s = sv[m].loc[r.msno]
            top[m] = [[k, round(float(s[k]), 4)] for k in s.abs().sort_values(ascending=False).head(8).index]
        st, rr = ps[ps.msno == r.msno].iloc[0], rec[rec.msno == r.msno].iloc[0]
        cust.append(dict(i=int(i), actual=a, g=g, p=[round(float(r[m + "_prob"]), 4) for m in M], pred=pred,
                         agree=bool(r.all_models_agree), top=top,
                         st=[round(float(st[c]), 3) for c in ["RandomForest_XGBoost_vector_corr",
                             "RandomForest_CatBoost_vector_corr", "XGBoost_CatBoost_vector_corr"]],
                         act=[str(rr[m + "_action"]) for m in M]))

    mat = lambda f: pd.read_csv(R / f, index_col=0).values.round(4).tolist()
    sp = rd("global_stability_results.csv")

    def jm(k):
        d = {}
        for _, r in sp.iterrows():
            a, b = r.model_pair.split(" vs ")
            d[(a, b)] = d[(b, a)] = float(r[f"jaccard_top_{k}"])
        return [[1.0 if a == b else d[(a, b)] for b in M] for a in M]

    D = dict(audit=json.loads((R / "cohort_audit.json").read_text()), test=met.to_dict("records"),
             conf=json.loads((R / "confusion_matrices.json").read_text()), sweep=sweep, hist=hist,
             shap=shap, shapAll=shap_all, dict=fdict, stab=sp.to_dict("records"),
             spearman=mat("spearman_consistency_matrix.csv"), jac5=jm(5), jac10=jm(10), jac15=jm(15),
             sub=rd("subgroup_stability_summary.csv").to_dict("records"), cust=cust,
             rec=json.loads((R / "recommendation_concordance_summary.json").read_text()),
             consensus=float(tp.all_models_agree.mean()),
             disclaimer=json.loads((R / "experiment_config.json").read_text())["disclaimer"])
    return json.loads(json.dumps(D, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    print(json.dumps(build_dashboard_data(root / "results"))[:300])
