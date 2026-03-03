#==============================================
# 1. Imports Cofigurations and Load Balanced Data
# =============================================
# Imports
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
main=os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(main)
from src.utils.file_utils import (save_pickle,
                              load_pickle,
                              save_csv,
                              save_json)
from commons import ensure_directory 
from typing import Any

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Metrics and Plots
from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score,
                             roc_auc_score,
                             confusion_matrix,
                             RocCurveDisplay,
                             ConfusionMatrixDisplay,
                             classification_report)

import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]

def validate_balance(balance_version: str) -> Any:
    # Validate paths/balance version
    if balance_version not in ["SMOTE", "ADASYN"]:
        raise ValueError("BALANCE_VERSION must be SMOTE or ADASYN!!!")

    data_pkl_path = os.path.join(PROCESSED_DATA_PATH, f"TRAIN_{balance_version}.pkl")
    (X_train, y_train, X_validation, y_validation) = load_pickle(data_pkl_path)
    return (X_train, y_train, X_validation, y_validation)

# Configurations

PROCESSED_DATA_PATH = os.path.join(ROOT, "data", "processed")
MODELS_PATH = os.path.join(ROOT, "models")
MODELS_ML_PATH = os.path.join(ROOT,"models", "ml_models")
REPORTS_EVALUATE_PATH = os.path.join(ROOT,"reports", "evaluate")
RANDOM_STATE = 42

BALANCE_VERSION = "SMOTE" # Change this variable to ADASYN when needed.
(X_smote, y_smote, X_validation, y_validation) = validate_balance(BALANCE_VERSION)
BALANCE_VERSION = "ADASYN"
(X_adasyn, y_adasyn, X_validation, y_validation) = validate_balance(BALANCE_VERSION)


print("Loaded Balance Training!!!")
print("Train SMOTE: ", X_smote.shape, np.bincount(np.array(y_smote).astype(int)))
print("Validation SMOTE: ", X_validation.shape, np.bincount(np.array(y_validation).astype(int)))

print("Train ADASYN: ", X_adasyn.shape, np.bincount(np.array(y_adasyn).astype(int)))
print("Validation ADASYN: ", X_validation.shape, np.bincount(np.array(y_validation).astype(int)))

#==============================================
# 2. Model Zoo
# =============================================
def get_models():
    models = {"logistic_regression": LogisticRegression(max_iter=2000,
                                                        n_jobs=None,
                                                        solver="lbfgs",
                                                        class_weight="balanced"),
              "random_forest": RandomForestClassifier(n_estimators=400,
                                                      max_depth=None,
                                                      min_samples_split=2,
                                                      n_jobs=-1,
                                                      random_state=RANDOM_STATE,
                                                      class_weight="balanced_subsample"),
              "xgboost": XGBClassifier(n_estimators=400,
                                       max_depth=5,
                                       learning_rate=0.05,
                                       subsample=0.9,
                                       colsample_bytree=0.9,
                                       reg_lambda=1.0,
                                       random_state=RANDOM_STATE,
                                       n_jobs=-1,
                                       eval_metric="logloss"),
              "light_gbm": LGBMClassifier(n_estimators=500,
                                          max_depth=-1,
                                          learning_rate=0.05,
                                          subsample=0.9,
                                          colsample_bytree=0.9,
                                          reg_lambda=0.0,
                                          random_state=RANDOM_STATE,
                                          n_jobs=-1),
              "cat_boost": CatBoostClassifier(iterations=400,
                                              depth=6,
                                              learning_rate=0.05,
                                              random_seed=RANDOM_STATE,
                                              verbose=False)}
    return models

#==============================================
# 3. Train and Evaluate on Sampling Data
#==============================================
ensure_directory(MODELS_ML_PATH)
ensure_directory(REPORTS_EVALUATE_PATH)

# Get model from model zoo
models = get_models()

# Fit evaluate and save artifacts
def evaluate_save(classifier,
                  model_name: str,
                  X_validation: pd.DataFrame,
                  y_validation: pd.Series,
                  output_directory: str):
    """
    Evaluate a fitted classifier of (X and y validations), save ROC and confusion matrices.
    Print a short report and return a metrics dictionary + file paths.
    """
    # predictios
    y_predict = classifier.predict(X_validation)
    # Decision scores/probabilities for ROC AUC
    try:
        y_score = classifier.predict_proba(X_validation)[:, 1]
    except Exception:
        # Some linear models expose decision functions instead
        y_score = classifier.decision_function(X_validation)

    # Evaluation Metrics
    accuracy = accuracy_score(y_validation, y_predict)
    precision = precision_score(y_validation, y_predict)
    recall = recall_score(y_validation, y_predict)
    f1 = f1_score(y_validation, y_predict)
    roc = roc_auc_score(y_validation, y_score)

    # Plots folder
    ensure_directory(output_directory)

    # ROC Curve
    roc_path = os.path.join(output_directory, f"roc_{model_name}.png")
    plt.figure()

    RocCurveDisplay.from_predictions(y_validation, y_score)
    plt.title(f"ROC - {model_name}")
    plt.savefig(roc_path, bbox_inches="tight")
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y_validation, y_predict)
    cm_path = os.path.join(output_directory, f"confusion_matrix_{model_name}.png")

    plt.figure()
    ConfusionMatrixDisplay(cm).plot(values_format="d")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()

    # Console Summary
    print(f"    Saved: {roc_path}")
    print(f"    Saved: {cm_path}")
    print("classification report:")
    print(classification_report(y_validation, y_predict, digits=4))

    return {"accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc,
            "roc_path": roc_path,
            "confusion_matrix": cm,
            "confusion_matrix_path": cm_path}

# Model Training driver for given method
def train_and_evaluate(method: str,
                       X_train: pd.DataFrame,
                       y_train: pd.DataFrame,
                       X_validation: pd.DataFrame,
                       y_validation: pd.DataFrame):
    """
    For a resampling Method(SMOTE/ADASYN):
    - Loads all models from model zoo.
    - Trains models.
    - Saves pickles under models/ml_models/<method>
    - Saves plots under reports/evaluate/<method>
    Returns a list[dict] of metrics rows.
    """
    # Outputs
    plot_directory = os.path.join(REPORTS_EVALUATE_PATH, method)
    pickle_directory = os.path.join(MODELS_ML_PATH, method)
    ensure_directory(plot_directory)
    ensure_directory(pickle_directory)
    
    rows = []
    
    for model_name, classifier in models.items():
        print(f"Training {model_name} on {method}...")
        classifier.fit(X_train, y_train)

        # evaluate + save plots
        metrics = evaluate_save(classifier,
                                model_name=model_name,
                                X_validation=X_validation,
                                y_validation=y_validation,
                                output_directory=plot_directory)
        
        # save_pickle
        save_pickle(classifier, os.path.join(MODELS_ML_PATH, method, f"{model_name}.pkl"))
        
        print(f"    Saved model: {pickle_directory}")

        # Add row
        row = {"model": model_name,
               "method": method,
               **metrics
               }
        rows.append(row)
    return rows

#==============================================
# 4. Run Methods + Leaderboard CSV
#==============================================

all_rows = []

# 1. SMOTE
rows_smote = train_and_evaluate("SMOTE",
                                X_smote,
                                y_smote,
                                X_validation,
                                y_validation)

all_rows.extend(rows_smote)

# 2. ADASYN
rows_adasyn = train_and_evaluate("ADASYN",
                                X_adasyn,
                                y_adasyn,
                                X_validation,
                                y_validation)


all_rows.extend(rows_adasyn)

# Leaderboard One CSV Comparing all
leaderboard = pd.DataFrame(all_rows) \
                           .sort_values(by=["method", "roc_auc", "f1_score"],
                                        ascending=[True, False, False])

save_csv(leaderboard, os.path.join(REPORTS_EVALUATE_PATH, "model_results.csv"))

print("\nLeaderboard saved -> ", os.path.join(REPORTS_EVALUATE_PATH))
print("\nTop rows:\n", leaderboard.head(10).to_string(index=False))

print("Training and evaluation completed.")
print(f"Plots at {os.path.join(REPORTS_EVALUATE_PATH, "SMOTE")} and {os.path.join(REPORTS_EVALUATE_PATH, "ADASYN")}")
print(f"Pickles at {os.path.join(MODELS_ML_PATH, "SMOTE")} and {os.path.join(MODELS_ML_PATH, "ADASYN")}")

#==============================================
# 5. Model Comparison to send best models to Optuna
#==============================================
def compare_models(results_list, balance_version: str):
    """
    Consolidate all model metrics into a single comparison dataframe.
    """
    data_results = pd.DataFrame(results_list)
    print(data_results.head(2))

    data_results = data_results.sort_values(by="accuracy", ascending=False).reset_index(drop=True)

    print(f"Model Leaderboard for {BALANCE_VERSION}")
    print(data_results[["model", "accuracy", "precision", "recall", "f1_score"]])

    summary_path = os.path.join(REPORTS_EVALUATE_PATH, balance_version)
    # Bar chart Accuracy 
    plt.figure(figsize=(10,5))
    sns.barplot(data=data_results,
                x="model",
                y="accuracy",
                palette="viridis")
    plt.title(f"Model Accuracy Comparison - {BALANCE_VERSION}")
    plt.ylabel("Accuracy")
    plt.ylim(0,1)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(summary_path, "model_comparison_by_accuracy.png"))
    plt.close()
    
    save_csv(data_results, os.path.join(summary_path, "model_comparison.csv"))
    data_results_json = data_results.to_json()
    save_json(data_results_json, os.path.join(summary_path, "model_comparison.json"))
    print(f"Model comparison saved at: {summary_path}")
    
    return data_results

compare_models(rows_smote, "SMOTE")
compare_models(rows_adasyn, "ADASYN")


                 









