########################################
# This file explains reasoning for feature importance
# of individual samples vs whole using SHAPley's Additive properyty.
########################################

########################################
# 1. Import headers
########################################


import os
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
main=os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(main)
print(main)
from commons import ensure_directory
from src.utils.file_utils import load_pickle, save_csv
from src.workflows.model_training import validate_balance
import mlflow, mlflow.sklearn

########################################
# 2. Load Preprocessor, Model and Data
########################################

ROOT = Path(__file__).resolve().parents[2]

X_validation = load_pickle(os.path.join(ROOT, "data", "processed", "X_validation.pkl"))

def load_artifacts(method: str, model_name: str):
    """
    mathod = 'SMOTE' or 'ADASYN'
    model_name = 'LogisticRegression', 'Randomforest'...etc. 
    """
    PREPROCESSOR_PATH = os.path.join(ROOT, "models", "feature_engineering", "preprocessor.pkl")
    MODEL_PATH = os.path.join(ROOT, "models", "tuning_models", method, f"{model_name}_best.pkl")
    preprocessor = load_pickle(PREPROCESSOR_PATH)
    model = load_pickle(MODEL_PATH)
    X_validation_transformed = preprocessor.transform(X_validation)
    return model, X_validation_transformed, preprocessor

########################################
# 3. SHAP explainer
########################################
def get_explainer(model, X_validation_transformed):
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_validation_transformed)
    except:
        background = shap.sample(X_validation_transformed, 100)
        explainer = shap.KernelExplainer(model.predict_proba,
                                         background)
        shap_values = explainer.shap_values(X_validation_transformed)
    return explainer, shap_values

########################################
# 4. Generate All SHAP Plots
########################################
def create_shap_plots(method: str,
                      model_name: str,
                      X_validation_transformed: pd.DataFrame,
                      shap_values,
                      feature_names):
    save_directory = os.path.join(ROOT, "reports", "explainability", method, model_name)
    ensure_directory(save_directory)
    if hasattr(shap_values, "values"):
        shap_array = shap_values.values
    elif isinstance(shap_values, list):
        if len(shap_values) > 1:
            shap_array = np.array(shap_values[1])
        else:
            shap_array = np.array(shap_values[0])
    else:
        shap_array = np.array(shap_values)

    # 1. Summary Plot
    plt.figure()
    shap.summary_plot(shap_array,
                      features=X_validation_transformed,
                      feature_names=feature_names,
                      show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, "shap_summary.png"))
    plt.close()

    # 2. Bar Plot
    plt.figure()
    shap.summary_plot(shap_array,
                      X_validation_transformed,
                      feature_names=feature_names,
                      plot_type="bar",
                      show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, "shap_bar.png"))
    plt.close()

    # 3. Top 3 Dependency Plots
    mean_abs = np.mean(np.abs(shap_array), axis=0)
    top_indices = np.argsort(mean_abs)[-3:][::-1]
    importance_path = os.path.join(save_directory, "SHAP_importance.csv")
    importance = pd.DataFrame({"feature": feature_names,
                               "importance": mean_abs})
    save_csv(importance, importance_path, index=False)

    mlflow.log_artifact(importance_path)

    for index in top_indices:
        index = int(index)
        feature_name = feature_names[index]
        
        plt.figure()
        shap.dependence_plot(index,
                             shap_array,
                             X_validation_transformed,
                             feature_names=feature_names,
                             interaction_index=None,
                             show=False
                             )
        plt.tight_layout()
        plt.savefig(os.path.join(save_directory, f"shap_dependence_{feature_name}.png"))
        plt.close('all')
         
########################################
# 5. Run SHAP
########################################
def run_shap(method: str, model_name: str):
    print(f"Generating SHAP for {method} -> {model_name}")

    model, X_validation_transformed, preprocessor = load_artifacts(method, model_name)
    
    with mlflow.start_run(run_name=f"SHAP_{method}_{model_name}"):
        mlflow.log_param("method", method)
        mlflow.log_param("model_name", model_name)

        feature_names = preprocessor.get_feature_names_out().tolist()
    
        explainer, shap_array = get_explainer(model, X_validation_transformed)

        # log raw shap arrays
        np.save("shap_values.npy", shap_array)
        mlflow.log_artifact("shap_values.npy")
        if shap_array.ndim == 3:
            shap_array = shap_array[:, :, 1]

        shap_values = shap.Explanation(values=shap_array,
                                       base_values=explainer.expected_value,
                                       data=X_validation_transformed,
                                       feature_names=feature_names)
        create_shap_plots(method=method,
                          model_name=model_name,
                          X_validation_transformed=X_validation_transformed,
                          shap_values=shap_values,
                          feature_names=feature_names)
    
        for file in os.listdir(os.path.join(ROOT, "reports", "explainability", method, model_name)):
            if file.endswith(".png"):
                mlflow.log_artifact(os.path.join(ROOT, "reports", "explainability", method, model_name, file))
    
    print(f"SHAP plots saved at ROOT/reports/explainability/{method}/{model_name}")

if __name__ == "__main__":
    over_sampling_methods = ["SMOTE", "ADASYN"]
    models = ["logistic_regression",
              "random_forest",
              "xgboost",
              "light_gbm",
              "cat_boost"]
    
    for method in over_sampling_methods:
        for model_name in models:
            run_shap(method, model_name)


        

    
        






