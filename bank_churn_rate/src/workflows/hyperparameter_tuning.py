#==============================================
# 1. Imports Configurations and Load Artifacts
# Process
# - Tune All Models using Optuna
# - Optimizes on 50% F1_Macro and 50% ROC_AUC_Curve.
# - Save tuned models.
# =============================================
# Imports
import os
os.environ["MPLBACKEND"] = "Agg"
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import optuna
from optuna.importance import get_param_importances
import sys
main=os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(main)
import numpy as np
import pandas as pd

from src.workflows.model_training import get_models, validate_balance
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (f1_score, roc_auc_score)
from typing import Any

from src.utils.file_utils import (save_pickle,
                              load_pickle,
                              save_csv,
                              save_json)
from commons import ensure_directory 

# Configurations
ROOT = Path(__file__).resolve().parents[2]
TUNING_DIRECTORY = os.path.join(ROOT, "models", "tuned") 
TUNING_MODELS_DIRECTORY = os.path.join(ROOT,"models", "tuning_models")
TUNING_REPORTS_DIRECTORY = os.path.join(ROOT,"reports", "tuning")
PROCESSED_DATA_PATH = os.path.join(ROOT,"data", "processed")

BALANCE_VERSION = "SMOTE" # Change this variable to ADASYN when needed.
(X_smote, y_smote, X_validation, y_validation) = validate_balance(BALANCE_VERSION)
BALANCE_VERSION = "ADASYN"
(X_adasyn, y_adasyn, X_validation, y_validation) = validate_balance(BALANCE_VERSION)

# Composite Metric for tuning
def composite_score(y_validation, y_predict, y_score):
    f1_macro = f1_score(y_validation, y_predict, average="macro")
    roc_auc = roc_auc_score(y_validation, y_score)

    return 0.5*f1_macro + 0.5*roc_auc, f1_macro, roc_auc

#==============================================
# 2. Search spaces per model.
# =============================================
def suggest_parameters(trial: optuna.trial.Trial,
                       model_key: str):
    if model_key == "logistic_regression":
        return {"C": trial.suggest_float("C", 1e-3, 50.0, log=True),
                "penalty": "l2",
                "solver": "lbfgs",
                "max_iter": trial.suggest_int("max_iter", 200, 2000, step=200)}
    if model_key == "random_forest":
        return {"n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
                "n_jobs": -1,
                "random_state": 42}
    if model_key == "xgboost":
        return {"n_estimators": trial.suggest_int("n_estimators", 150,900, step=50),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "random_state": 42,
                "n_jobs": -1,
                "tree_method": "hist",
                "eval_metric": "logloss"}
    if model_key == "light_gbm":
        return {"n_estimators": trial.suggest_int("n_estimators", 150, 1000, step=50),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 255),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
                "min_child_samples":trial.suggest_int("min_child_samples", 5, 50),
                "random_state": 42,
                "n_jobs":-1}
    if model_key == "cat_boost":
        return {"depth": trial.suggest_int("depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                "iterations": trial.suggest_int("iterations", 300, 1200, step=100),
                "random_seed": 42,
                "verbose": False,
                "allow_writing_files": False,
                "train_dir": None}
    raise ValueError(f"Unknown Model Key: {model_key}")

# Factory to build model with parameters
def build_model(model_key: str, parameters: dict):
    if model_key == "logistic_regression":
        return LogisticRegression(**parameters)
    if model_key == "random_forest":
        return RandomForestClassifier(**parameters)
    if model_key == "xgboost":
        return XGBClassifier(**parameters)
    if model_key == "light_gbm":
        return LGBMClassifier(**parameters)
    if model_key == "cat_boost":
        return CatBoostClassifier(**parameters)
    raise ValueError(f"Unknown Model Key: {model_key}")

# Tune one model per method (SMOTE/ADASYN)
def tune_one_model(X_train,
                   y_train,
                   X_validation,
                   y_validation,
                   method: str,
                   model_key: str,
                   n_trials: int=50,
                   ):
    study_directory = os.path.join(TUNING_DIRECTORY, method)
    tuned_directory = os.path.join(TUNING_MODELS_DIRECTORY, method)
    report_directory = os.path.join(TUNING_REPORTS_DIRECTORY, method)

    ensure_directory(study_directory)
    ensure_directory(tuned_directory)
    ensure_directory(report_directory)

    def objective(trial: optuna.trial.Trial):
        parameters = suggest_parameters(trial, model_key)
        classifier = build_model(model_key, parameters)
        classifier.fit(X_train, y_train)

        # predictions + Score
        y_predict = classifier.predict(X_validation)
        try:
            y_score = classifier.predict_proba(X_validation)[:, 1]
        except Exception:
            y_score = classifier.decision_function(X_validation)

        composite, f1_macro, auc = composite_score(y_validation, y_predict, y_score)
        trial.set_user_attr("f1_macro", float(f1_macro))
        trial.set_user_attr("roc_auc", float(auc))
        return composite
    
    study_name = f"{model_key}_{method}"
    study = optuna.create_study(direction="maximize", study_name=study_name)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # save trials into csv
    trials_list = []
    for trials in study.trials:
        trials_dict = trials.params.copy()
        trials_dict["value"] = trials.value
        trials_dict["trial_number"] = trials.number
        trials_list.append(trials_dict)

    trials_df = pd.DataFrame(trials_list)
    csv_path = os.path.join(study_directory, f"{model_key}_trials.csv")
    save_csv(trials_df, csv_path)
    print(f"Saved {len(trials_df)} trials to {csv_path}")

    # Save best parameters + study
    best_parameters_path = os.path.join(study_directory, f"{model_key}_best_parameters.json")
    save_json(study.best_params, best_parameters_path)

    study_path = os.path.join(study_directory, f"{model_key}_study.pkl")
    save_pickle(study, study_path)

    # Retrain with best parameters and save tuned_model
    best_model = build_model(model_key, study.best_params)
    best_model.fit(X_train, y_train)
    tuned_pickle = os.path.join(tuned_directory, f"{model_key}_best.pkl")
    save_pickle(best_model, tuned_pickle)

    # Evaluation with best model
    y_predict = best_model.predict(X_validation)
    try:
            y_score = best_model.predict_proba(X_validation)[:, 1]
    except Exception:
            y_score = best_model.decision_function(X_validation)
    composite, f1_macro, auc = composite_score(y_validation, y_predict, y_score)

    summary = {
        "method": method,
        "model": model_key,
        "best_value_composite": float(study.best_value),
        "post_fit_composite": float(composite),
        "f1_macro": float(f1_macro),
        "roc_auc": float(auc),
        "best_parameters_path": str(best_parameters_path),
        "study_path": str(study_path),
        "tuned_model_path": str(tuned_pickle)
    }
    save_json(summary, os.path.join(report_directory, f"{model_key}_tuning_summary.json"))
    print(f"Tuned {model_key} ({method}) - Composite = {composite:.4f}, F1_Macro={f1_macro:.4f}, AUC={auc:.4f}")    
    return summary, study

#==============================================
# 3. Run All models.
# =============================================
ALL_MODELS = ["logistic_regression", "random_forest", "xgboost", "light_gbm", "cat_boost"]

def save_parameter_importance_plot(study, method: str, model_key: str):
    output_directory = os.path.join(TUNING_REPORTS_DIRECTORY, method)
    ensure_directory(output_directory)
    try:
        importance = get_param_importances(study)
        labels = list(importance.keys())
        values = [importance[key] for key in labels]
        plt.figure(figsize=(8, 4))
        plt.bar(labels, values)
        plt.xticks(rotation=30, ha="right")
        plt.title(f"Parameter Importance - {model_key.upper()} ({method})")
        plt.tight_layout()
        path = os.path.join(output_directory, f"{model_key}_parameter_importance.png")
        plt.savefig(path)
        plt.close()

        print(f"Saved parameter importance -> {path}")
    except Exception as e:
        print(f"Skippedv importance for {model_key}/{method}: {e}")

def run_all_tuning(n_trials=50):
    summaries = []
    studies = {}
    for method in ["SMOTE", "ADASYN"]:
        for model in ALL_MODELS:
            print(f"Tuning: {model.upper()} on {method}")
            if method == "SMOTE":
                summary, study = tune_one_model(X_smote,
                                                y_smote,
                                                X_validation,
                                                y_validation,
                                                method,
                                                model,
                                                n_trials=n_trials,
                                                )
            elif method == "ADASYN":
                summary, study = tune_one_model(X_adasyn,
                                                y_adasyn,
                                                X_validation,
                                                y_validation,
                                                method,
                                                model,
                                                n_trials=n_trials,
                                                )
            else: raise ValueError(f"method {method} incorrect.")
            summaries.append(summary)
            studies[(method, model)] = study

    # Save leaderboard
    data = pd.DataFrame(summaries) \
                        .sort_values(by=["method",
                                         "post_fit_composite",
                                         "roc_auc",
                                         "f1_macro"],
                                         ascending=[True, False, False, False])
      
    leaderboard_csv = os.path.join(TUNING_REPORTS_DIRECTORY, "tuning_leaderboard.csv")
    save_csv(data, leaderboard_csv)
    print("\nTuning Leaderboard Saved: ", leaderboard_csv)

    # Parameter Importance plots
    for (method, model), study in studies.items():
        save_parameter_importance_plot(study, method, model)
    return data, studies

if __name__=="__main__":
    # you requested 50 trials
    tuning_data, tuning_studies = run_all_tuning(n_trials=50)
    print("\n Top tuned models per method:")
    print(tuning_data.sort_values(["method", "post_fit_composite"], ascending=[True, False])    
                     .groupby("method").head(3)[["method", "model", "post_fit_composite", "roc_auc", "f1_macro"]]
                     .to_string(index=False))   

         


