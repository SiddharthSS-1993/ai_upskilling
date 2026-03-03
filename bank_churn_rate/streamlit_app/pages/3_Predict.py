import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
main=os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(main)
from src.utils.file_utils import save_csv
from streamlit_app.ui_utils import(ROOT,
                                   load_preprocessor,
                                   load_tuned_model,
                                   available_models_in_directory,
                                   normalize_columns,
                                   build_single_row_template,
                                   predict_proba)

sampling_method = ["SMOTE", "ADASYN"]
st.set_page_config(page_title="Predict", layout="wide")
st.title("🔮 Predict Churn")

st.sidebar.subheader("Model Selection")
method = st.selectbox("Sampling Method", sampling_method, index=0)
MODEL_DIRECTORY = os.path.join(ROOT, "models", "tuning_models", method)
model_names = available_models_in_directory(MODEL_DIRECTORY)
if not model_names:
    st.error(f"No tuned models found in: {MODEL_DIRECTORY}")
    st.stop()

model_name = st.sidebar.selectbox("Model Name", model_names, index=0)

preprocessor = load_preprocessor()
model = load_tuned_model(method, model_name)

st.write(f"Using **{method}** / **{model_name}** (Tuned model).")
tab_1, tab_2 = st.tabs(["Upload CSV", "Single Prediction Form"])

with tab_1:
    st.subheader("Batch Prediction from CSV")
    uploaded = st.file_uploader("Upload a CSV with raw columns", type=["csv"])
    if uploaded is not None:
        data = pd.read_csv(uploaded)
        data = normalize_columns(data)

        st.write("Preview:")
        st.dataframe(data.head(10))

        # If Exited column exists
        if "Exited" in data.columns:
            feature_data = data.drop(columns=["Exited"])
        else:
            feature_data = data.copy()

        
        try:
            X_transformed = preprocessor.transform(feature_data)
            proba = predict_proba(model, X_transformed)
            output = data.copy()
            output["Churn_Probability"] = proba
            output["Churn_Prediction"] = (proba >= 0.5).astype(int)

            st.success("Prediction Generated!!!")
            st.dataframe(output.head(20))
            csv_bytes = output.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions CSV", data=csv_bytes, file_name="predictions.csv")
        except Exception as e:
            st.error("Prediction Failed. Most common cause column mismatch vs training")
            st.exception(e)
    
with tab_2:
    st.subheader("Single-row prediction")
    template = build_single_row_template()

    # Editable inputs
    column_a, column_b, column_c = st.columns(3)

    with column_a:
        credit_score = st.number_input("CreditScore",
                                       value=int(template.loc[0, "CreditScore"]),
                                       min_value=300,
                                       max_value=900)
        age = st.number_input("Age",
                              value=int(template.loc[0, "Age"]),
                              min_value=18,
                              max_value=100)
        tenure = st.number_input("Tenure",
                                 value=int(template.loc[0, "Tenure"]),
                                 min_value=0,
                                 max_value=20)
        balance = st.number_input("Balance",
                                  value=float(template.loc[0, "Balance"]),
                                  min_value=0.0)
        
    with column_b:
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"], index=0)
        gender = st.selectbox("Gender", ["Male", "Female"], index=0)
        num_products = st.number_input("NumOfProducts",
                              value=int(template.loc[0, "NumOfProducts"]),
                              min_value=1,
                              max_value=10)
        has_card = st.selectbox("HasCrCard", [0, 1], index=1)

    with column_c:
        is_active = st.selectbox("IsActiveMember", [0, 1], index=1)
        salary = st.number_input("EstimatedSalary",
                              value=float(template.loc[0, "EstimatedSalary"]),
                              min_value=0.0)
        # Keep id columns optional
        customer_id = st.number_input("CustomerId",
                              value=int(template.loc[0, "CustomerId"]),
                              min_value=0)
        surname = st.text_input("Surname", value=str(template.loc[0, "Surname"]))

    row = template.copy()
    row.loc[0, "CreditScore"] = credit_score
    row.loc[0, "Age"] = age
    row.loc[0, "Tenure"] = tenure
    row.loc[0, "Balance"] = balance
    row.loc[0, "Geography"] = geography
    row.loc[0, "Gender"] = gender
    row.loc[0, "NumOfProducts"] = num_products
    row.loc[0, "HasCrCard"] = has_card
    row.loc[0, "IsActiveMember"] = is_active
    row.loc[0, "EstimatedSalary"] = salary
    row.loc[0, "CustomerId"] = customer_id
    row.loc[0, "Surname"] = surname

    st.write("Input Row:")
    st.dataframe(row)

    if st.button("Predict Churn Probability"):
        try:
            X_transformed = preprocessor.transform(row)
            proba = float(predict_proba(model, X_transformed)[0])
            predict = 1 if proba >= 0.5 else 0
            st.metric("Churn Probability", f"{proba:.4f}")
            st.write("Prediction:", "🚨 Churn(1)" if predict ==1 else "✅ Not Churn (0)")
        except Exception as e:
            st.error("Prediction Failed Likely due to column Mismatch")
            st.exception(e)







                                    
        
        



