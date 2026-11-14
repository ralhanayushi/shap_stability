# Visual Studio Code Guide: Running & Exploring the Project

Welcome! This guide explains how to open, view, run, and present the **SHAP Explanation Stability Evaluation System** directly inside **Visual Studio Code (VS Code)**.

---

## 1. Opening the Project in VS Code

1. Launch **Visual Studio Code**.
2. Click **File** > **Open Folder...** (or press `Ctrl + K, Ctrl + O`).
3. Select the project folder:
   ```
   C:\Users\bhumi w\.gemini\antigravity\scratch\shap_churn_stability
   ```
4. If prompted with *"Do you trust the authors of the files in this folder?"*, click **Yes, I trust the authors**.

---

## 2. Setting Up the Python Environment in VS Code

1. Open the Integrated Terminal in VS Code:
   - Press **`Ctrl + ~`** (or go to **Terminal** > **New Terminal**).
2. If you haven't set up your virtual environment yet, run:
   ```powershell
   .\setup_env.bat
   ```
   *(Or activate your existing Python installation where dependencies are installed).*
3. Select the Python Interpreter in VS Code:
   - Press **`Ctrl + Shift + P`** to open the Command Palette.
   - Type `Python: Select Interpreter` and press `Enter`.
   - Choose the recommended interpreter (either your `.venv` or your system Python 3.13).

---

## 3. How to View and Understand All Codes

Use the VS Code **Explorer sidebar** (`Ctrl + Shift + E`) to navigate the codebase:

| Directory / File | What It Contains & What to View |
| :--- | :--- |
| [`src/config.py`](file:///C:/Users/bhumi%20w/.gemini/antigravity/scratch/shap_churn_stability/src/config.py) | Cutoff dates (`2017-01-31`, `2017-02-28`), seeds (`42`), hyperparameter search grids, action definitions. |
| [`src/data_loader.py`](file:///C:/Users/bhumi%20w/.gemini/antigravity/scratch/shap_churn_stability/src/data_loader.py) | High-speed streaming parser for transactions and user logs directly from `.7z` archives without disk bloat. |
| [`src/feature_engineering.py`](file:///C:/Users/bhumi%20w/.gemini/antigravity/scratch/shap_churn_stability/src/feature_engineering.py) | Leakage-safe customer aggregations, age anomaly filtering ($10 \le \text{bd} \le 100$), median imputation. |
| [`src/models.py`](file:///C:/Users/bhumi%20w/.gemini/antigravity/scratch/shap_churn_stability/src/models.py) | Model implementations for **Random Forest**, **XGBoost**, and **CatBoost** with class weighting & tuning. |
| [`src/evaluation.py`](file:///C:/Users/bhumi%20w/.gemini/antigravity/scratch/shap_churn_stability/src/evaluation.py) | Precision, Recall, $F_1$, ROC-AUC, threshold optimization on validation data, out-of-time test scoring. |
| [`src/shap_explainer.py`](file:///C:/Users/bhumi%20w/.gemini/antigravity/scratch/shap_churn_stability/src/shap_explainer.py) | TreeExplainer SHAP values, probability additivity verification, and 100-customer stratified cohort selection. |
| [`src/stability.py`](file:///C:/Users/bhumi%20w/.gemini/antigravity/scratch/shap_churn_stability/src/stability.py) | **Core Research Module**: Spearman rank correlation, Top-$K$ Jaccard similarity, per-customer vector correlations. |
| [`src/retention_rules.py`](file:///C:/Users/bhumi%20w/.gemini/antigravity/scratch/shap_churn_stability/src/retention_rules.py) | Transparent rule-based retention mapping and multi-model action concordance calculation. |
| [`src/visualization.py`](file:///C:/Users/bhumi%20w/.gemini/antigravity/scratch/shap_churn_stability/src/visualization.py) | Generation of publication-quality ROC/PR curves, SHAP comparison bars, consistency heatmaps. |
| [`backend/app.py`](file:///C:/Users/bhumi%20w/.gemini/antigravity/scratch/shap_churn_stability/backend/app.py) | FastAPI REST API serving live endpoints (`/api/status`, `/api/metrics`, `/api/stability`, `/api/predict`). |
| [`frontend/index.html`](file:///C:/Users/bhumi%20w/.gemini/antigravity/scratch/shap_churn_stability/frontend/index.html) | Interactive web interface displaying live metrics, SHAP bars, stability matrices, and test prediction form. |
| [`run_pipeline.py`](file:///C:/Users/bhumi%20w/.gemini/antigravity/scratch/shap_churn_stability/run_pipeline.py) | Complete CLI orchestrator running all 13 research stages end-to-end. |

---

## 4. Running the Code in VS Code (Press F5)

Pre-configured launch profiles have been added to [`.vscode/launch.json`](file:///C:/Users/bhumi%20w/.gemini/antigravity/scratch/shap_churn_stability/.vscode/launch.json):

1. Press **`Ctrl + Shift + D`** (or click the **Run & Debug** play icon on the left bar).
2. In the top dropdown menu, select one of the three profiles:
   - **`1. Launch Web Dashboard (FastAPI)`**: Starts the interactive server at `http://127.0.0.1:8000`.
   - **`2. Run Full Research Pipeline (KKBOX Data)`**: Runs the complete training, SHAP, and stability pipeline.
   - **`3. Run Fast Pipeline (Sample Data)`**: Runs an instant 5-second sanity test on the sample dataset.
3. Press **`F5`** (or click the green **Play** button) to start running or debugging!

---

## 5. Viewing & Running the Jupyter Notebook in VS Code

1. In the Explorer, open [`notebooks/01_shap_stability_evaluation.ipynb`](file:///C:/Users/bhumi%20w/.gemini/antigravity/scratch/shap_churn_stability/notebooks/01_shap_stability_evaluation.ipynb).
2. VS Code will render the interactive Jupyter Notebook interface natively.
3. In the top right corner of the notebook, click **Select Kernel** > **Python Environments** > select your Python interpreter.
4. Click **Run All** (or press `Shift + Enter` on individual cells) to execute cells, display metrics tables, and render plots inline.

---

## 6. How to Showcase This to Others

### Presentation Routine
1. **Open VS Code** on your screen with [`README.md`](file:///C:/Users/bhumi%20w/.gemini/antigravity/scratch/shap_churn_stability/README.md) or [`shap_stability_evaluation_report.md`](file:///C:/Users/bhumi%20w/.gemini/antigravity/brain/d33548b6-0531-4fe7-ad09-38aa0688d5c6/shap_stability_evaluation_report.md) previewed (`Ctrl + Shift + V`).
2. **Launch the Dashboard**: Double-click `run_dashboard.bat` (or select profile 1 in VS Code and press F5).
3. **Open Browser to `http://127.0.0.1:8000`**:
   - Point out the **Model Accuracy (~90%)** across XGBoost, CatBoost, and Random Forest.
   - Switch between models on the **SHAP Importance** chart to show top features (`last_is_auto_renew`, `tx_days_since_last`, `log_skip_ratio`).
   - Highlight the **Spearman Rank Correlation ($0.9601$)** and **Top-5 Overlap ($100\%$)** between XGBoost and CatBoost.
   - Demonstrate the **Live Predictor** card by clicking "Predict with trained models".
4. **Show the Code Structure**:
   - Open [`src/stability.py`](file:///C:/Users/bhumi%20w/.gemini/antigravity/scratch/shap_churn_stability/src/stability.py) to show the Jaccard, Spearman, and per-customer vector correlation functions.
   - Open [`src/feature_engineering.py`](file:///C:/Users/bhumi%20w/.gemini/antigravity/scratch/shap_churn_stability/src/feature_engineering.py) to show the leakage-safe cutoff asserts.
