import streamlit as st
import os
import sys 
main=os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(main)
print(main)
from streamlit_app.ui_utils import ROOT

st.set_page_config(page_title="Bank Churn Rate - ML System",
                   layout="wide")

st.title("🏦 Bank Churn Rate - ML System")
st.write("""
This is a modular end-to-end project with:
- Feature Engineering Pipeline (preprocessor.pkl)
- SMOTE/ADASYN Sampling
- Baseline + Optuna-tuned models
- Evaluation plots
- SHAP Explainability
- MLFlow Experiment Tracking
""")

st.subheader("Project Health Checks")

checks = {
    "Raw data": os.path.join(ROOT, "data", "raw"),
    "EDA reports": os.path.join(ROOT, "reports", "eda"),
    "Evaluation Reports": os.path.join(ROOT, "reports", "evaluate"),
    "Explainability reports": os.path.join(ROOT, "reports", "explainability"),
    "Preprocessor": os.path.join(ROOT, "models", "feature_engineering", "preprocessor.pkl"),
    "Tuned models": os.path.join(ROOT, "models", "tuning_models")}

for name, path in checks.items():
    ok = os.path.exists(path)
    st.write(f"{'✅' if ok else '❌'} **{name}** - `{path}`")

st.info("Use the left sidebar to open pages: EDA, Evaluation, Predict, SHAP, MLFlow.")



