# 🧠 Bank Customer Churn Prediction — End-to-End ML + MLflow Project

## 📌 Overview
This repository implements a complete **Machine Learning workflow** for predicting customer churn in a retail banking dataset.  
It combines traditional ML with modern MLOps tools such as **MLflow**, **Optuna**, and structured artifact tracking — designed for interview-ready, production-grade pipelines.

---

## 🗂️ Project Folder Structure

```bash
bank_churn_rate/
├── data/
│   ├── raw/                        # Original dataset (churn.csv)
│   └── processed/                  # SMOTE / ADASYN balanced + validation splits
│
├── models/
│   ├── ml_models/
│   │   ├── SMOTE/                 # Baseline (untuned) SMOTE models (.pkl)
│   │   └── ADASYN/                # Baseline (untuned) ADASYN models (.pkl)
│   ├── tuned/
│   │   ├── SMOTE/                 # Best tuned SMOTE models + params (.pkl / .json)
│   │   └── ADASYN/
│   ├── tuned_models/
│   │   ├── SMOTE/                 # Best tuned SMOTE models + params (.pkl / .json)
│   │   └── ADASYN/                # Best tuned ADASYN models + params (.pkl / .json)
│   └── feature_engineering/       # Preprocessor pickle (preprocessor.pkl)
│
├── reports/
│   ├── eda/                       # Exploratory data analysis plots (.png)
│   ├── evaluate/
│   │   ├── SMOTE/                 # Confusion matrix, ROC, metrics per model
│   │   └── ADASYN/
│   └── tuning/
│       ├── SMOTE/                 # Optuna trial CSVs + trend plots
│       └── ADASYN/
│
├── src/
│   ├── utils/                     # Utility scripts (I/O, plotting, JSON helpers)
│   │   ├── file_utils.py
│   │   ├── plot_utils.py
│   │   └── misc_utils.py
│   ├── workflows/   
│   │   ├── data_loader.py             # Data import and preprocessing helpers
│   │   ├── feature_engineering.py     # Encoding, scaling, resampling
│   │   ├── model_training.py          # Baseline model training & evaluation
│   │   └── hyperparameter_tuning.py   # Optuna-based hyperparameter optimization
│   └── monitoring/
│       └── mlflow_runner.py       # MLflow experiment logging
│
├── mlruns/                        # MLflow local tracking directory
│   ├── bank_churn_rate_using_SMOTE_oversampling/
│   └── bank_churn_rate_using_ADASYN_oversampling/
│
├── requirements.txt
└── README.md

```

---

## 🔍 Workflow Summary

### 1️⃣ Data Loading & EDA (`eda.ipynb`)
- Loads and inspects raw data  
- Explores class balance, distributions, correlations, and target rates  
- Automatically saves plots and summary JSON under `reports/eda/`

### 2️⃣ Feature Engineering (`feature_engineering.py`)
- Drops ID columns, encodes categoricals, scales numericals  
- Handles class imbalance using **SMOTE** and **ADASYN**  
- Saves processed CSVs + preprocessor pickle for reproducibility  

### 3️⃣ Model Training (`model_training.py`)
- Trains multiple models (Logistic Regression, Random Forest, XGBoost, etc.)  
- Evaluates via accuracy, F1-macro, ROC-AUC  
- Saves all metrics JSONs and model pickle files  

### 4️⃣ Hyperparameter Tuning (`hyperparameter_tuning.py`)
- Uses **Optuna** with 50 trials per model  
- Objective = 0.5 × F1_macro + 0.5 × ROC_AUC  
- Stores per-trial CSVs and generates combined parameter-trend dashboards  
  (numeric → line, categorical → scatter colored by objective value)

### 5️⃣ MLflow Tracking (`monitoring/mlflow_runner.py`)
- Creates clean experiments:
  - `bank_churn_rate_using_SMOTE_oversampling`
  - `bank_churn_rate_using_ADASYN_oversampling`
- Logs for each run:
  - Parameters, metrics, and tags (baseline / tuned)
  - All artifacts: pickled models, Optuna plots, evaluation charts
- Models registered under `mlruns/<experiment>`
- Metrics & parameters browsable via **MLflow UI** (`localhost:5000`)

### 6️⃣ Explainability (coming soon)
- Integrate **SHAP** for feature-level explanations  
- Visualize key driver importance and per-customer SHAP values  

### 7️⃣ LLM-Driven Insights (coming soon)
- Use **LangGraph + LangFuse** to build an LLM agent that:
  - Reads artifacts, plots & metrics  
  - Generates natural-language markdown insights (`insights.md`)  
  - Optionally exposed via **Streamlit** dashboard  

---

## ⚙️ Setup

```bash
# 1. Create environment
conda create -n churn python=3.10
conda activate churn

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch MLflow UI
mlflow ui --backend-store-uri file:///absolute/path/to/mlruns --port 5000

# Step 1 – Preprocess
python src/feature_engineering.py

# Step 2 – Train models
python src/model_training.py

# Step 3 – Tune & evaluate
python src/hyperparameter_tuning.py

# Step 4 – Log to MLflow
python src/monitoring/mlflow_runner.py

📈 Metrics Tracked
Metric	Description
accuracy	Overall correctness
precision, recall, f1_macro	Class-wise balance indicators
roc_auc	Overall ranking quality
optuna_value	Combined tuning score (F1 + ROC)

💡 Interesting Questions to Explore

Which features most strongly drive churn across demographics?

How do SMOTE vs ADASYN impact recall and precision?

How stable are model rankings between baseline and tuned runs?

Could adding temporal features improve predictive power?

How would LLM-based reasoning (e.g., feature description generation) enhance interpretability?

🏁 Next Steps

✅ Add SHAP explanations
✅ Build LLM Insights Agent using LangGraph + LangFuse
✅ Create Streamlit Dashboard for interactive exploration