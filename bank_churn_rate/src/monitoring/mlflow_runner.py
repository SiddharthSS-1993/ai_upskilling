#============================================
# Centralized MLFlow tracking for model experiments
#============================================
"""
This file initialised MLFlow, registers experiments (SMPOTE and ADASYN)
and validates if all required artifacts exist before logging.
"""
import os
import sys
import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlflow.models import evaluate
from pathlib import Path
main=os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(main)
from src.utils.file_utils import load_json, load_csv, load_pickle
from src.workflows.model_training import validate_balance
import glob
from mlflow.sklearn import log_model
from mlflow.system_metrics import enable_system_metrics_logging
from commons import ensure_directory
import math

#=============================================
# 1. Project Paths
#=============================================
ROOT = Path(__file__).resolve().parents[2]
print(ROOT)
DATA_PROCESSED_DIRECTORY = os.path.join(ROOT, "data", "processed")

BASELINE_MODEL_DIRECTORY = os.path.join(ROOT, "models", "ml_models")
BEST_TUNED_MODELS_DIRECTORY = os.path.join(ROOT, "models", "tuning_models")
BEST_PARAMETERS_DIRECTORY = os.path.join(ROOT, "models", "tuned")

REPORTS_EVALUATE_DIRECTORY = os.path.join(ROOT, "reports", "evaluate")
REPORTS_TUNING_DIRECTORY = os.path.join(ROOT, "reports", "tuning")

MLFLOW_DIRECTORY = os.path.join(ROOT, "mlruns")

#=============================================
# 2. Initialize MLflow
#=============================================

mlflow.set_tracking_uri(f"http://localhost:5000")
# print(f"MLflow Tracking URI set at: {MLFLOW_DIRECTORY}")

#=============================================
# 3. Register Experiments
#=============================================
EXPERIMENTS = ["bank_churn_SMOTE_oversampling", "bank_churn_ADASYN_oversampling"]
SAMPLING_METHODS = ["SMOTE", "ADASYN"]
MODES = ["baseline", "tuned"]
MODEL_LIST = ["logistic_regression", "random_forest", "xgboost", "light_gbm", "cat_boost"]
EXPERIMENT_IDS = {}

for experiment in EXPERIMENTS:
    try:
        exp_id = mlflow.create_experiment(experiment)
        print(f"Created new experiment: {experiment} (id={exp_id})")
    except Exception:
        exp_id = mlflow.get_experiment_by_name(experiment).experiment_id
        print(f"Experiment already exists!!! {experiment} (id={exp_id})")

#=============================================
# 4. Fail fast if directories to be loaded are missing.
#=============================================
REQUIRED_DIRECTORIES = [BASELINE_MODEL_DIRECTORY,
                        BEST_TUNED_MODELS_DIRECTORY,
                        BEST_PARAMETERS_DIRECTORY,
                        REPORTS_EVALUATE_DIRECTORY,
                        REPORTS_TUNING_DIRECTORY]

missing = [str(p) for p in REQUIRED_DIRECTORIES if not p]
if missing:
    print("\nRequired folders are missing before MLflow Logging!!!")
    for m in missing:
        print(f"Fix {m} directory before running the mlflow logging phase")
    sys.exit(1)

print("All required folders exist!!!. Ready for MLflow logging.")
(X_train, y_train, X_validation, y_validation) = validate_balance("SMOTE")
X_validation = pd.DataFrame(X_validation).reset_index(drop=True)
y_validation = pd.Series(y_validation).reset_index(drop=True)
X_validation["Exited"] = y_validation

#=============================================
# 5. Get trials evolution per sampler model from optuna tuning
#=============================================
def plot_optuna_trial_trends(csv_path: str, output_directory: str):
    trials = load_csv(csv_path)
    exclude_columns = ["trial_number"]
    parameter_columns = [c for c in trials.columns if c not in exclude_columns]

    numeric_parameters = [c for c in parameter_columns if pd.api.types.is_numeric_dtype(trials[c])]
    categorical_parameters = [c for c in parameter_columns if c not in numeric_parameters]

    n_parameters = len(parameter_columns)
    n_columns = 3
    n_rows = math.ceil(n_parameters / n_columns)

    figure, axes = plt.subplots(n_rows, n_columns, figsize=(18, n_rows * 4))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    # Next each parameter trend:
    for index, parameter in enumerate(parameter_columns):
        ax = axes[index]
        if parameter in numeric_parameters:
            sns.lineplot(data=trials,
                         x="trial_number",
                         y=parameter,
                         marker="o",
                         color="teal",
                         ax=ax)
        else:
            sns.stripplot(data=trials,
                            x="trial_number",
                            y=parameter,
                            ax=ax,
                            jitter=True)
        ax.set_title(f"{parameter} Trend", fontsize=10)
        ax.set_xlabel("Trial Number")
        ax.set_ylabel(parameter)

    # Hide unused subplots if any
    for j in range(len(parameter_columns), len(axes)):
        figure.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(output_directory)
    plt.close(figure)
    print(f"Combined parameter trend plot saved: {output_directory}")        


#=============================================
# 6. Main logging function
#=============================================
def log_run_to_mlflow(sampling_method, model_name, mode):
    for experiment in EXPERIMENTS:
        mlflow.set_experiment(experiment)
        run_name = f"{model_name}_{mode}"
        print(f"Starting MLFlow Run: {run_name}")

        with mlflow.start_run(run_name=run_name, log_system_metrics=True):
            # --- Tags for c(lean UI
            mlflow.set_tag("model", model_name)
            mlflow.set_tag("mode", mode)
            if sampling_method in experiment:
                mlflow.set_tag("sampler", sampling_method)
            mlflow.set_tag("author", "Siddharth")

            if mode == "baseline":
                # log model paths
                model_path = os.path.join(BASELINE_MODEL_DIRECTORY, sampling_method, f"{model_name}.pkl")
                
                model = load_pickle(model_path)
                print("\n", model_path)
                mlflow.log_artifact(model_path, artifact_path="model")
                model = log_model(model, name="model")
                eval_results = mlflow.evaluate(model=model.model_uri.replace("runs:/", "mlruns/"),
                                               data=X_validation,
                                               targets="Exited",
                                               model_type="classifier",
                                               evaluators=["default"])
                print(f"Logged model: {model_path}")
                # log metric plots
                plot_path = os.path.join(REPORTS_EVALUATE_DIRECTORY, sampling_method)
                confusion_png_files = glob.glob(os.path.join(plot_path, f"confusion_matrix_{model_name}.png"))
                roc_png_files = glob.glob(os.path.join(plot_path, f"roc_{model_name}.png"))
                png_files = confusion_png_files + roc_png_files
                print(png_files)
                for file in png_files:
                    mlflow.log_artifact(file, artifact_path="plots")
                    print(f"Logged plots: {file}")
                
            if mode == "tuned":
                # log model paths
                model_path = os.path.join(BEST_TUNED_MODELS_DIRECTORY, sampling_method, f"{model_name}_best.pkl")
                print("\n", model_path)
                model = load_pickle(model_path)
                mlflow.log_artifact(model_path, artifact_path="model")
                model = log_model(model, name="model")
                eval_results = mlflow.evaluate(model=model.model_uri.replace("runs:/", "mlruns/"),
                                               data=X_validation,
                                               targets="Exited",
                                               model_type="classifier",
                                               evaluators=["default"])
                print(f"Logged model: {model_path}")
                # log best parameters
                parameters_path = os.path.join(BEST_PARAMETERS_DIRECTORY, sampling_method, f"{model_name}_best_parameters.json")
                print("\n", parameters_path)
                mlflow.log_artifact(parameters_path, artifact_path="params")
                print(f"Logged parameters: {parameters_path}")
                # log metric plots
                plot_path = os.path.join(REPORTS_TUNING_DIRECTORY, sampling_method)
                png_files = glob.glob(os.path.join(plot_path, f"{model_name}_parameter_importance.png"))
                for file in png_files:
                    print("\n", os.path.join(plot_path, file))
                    mlflow.log_artifact(file, artifact_path="plots")
                    print(f"Logged plots: {file}")
                # log tuned metrics
                json_files = glob.glob(os.path.join(plot_path, f"{model_name}_tuning_summary.json"))
                for file in json_files:
                    print("\n", os.path.join(plot_path, file))
                    metrics_dict = load_json(file)
                    numeric_metrics = {k: v for k, v in metrics_dict.items() if isinstance(v, (int, float))}
                    print(numeric_metrics)
                    mlflow.log_metrics(numeric_metrics)
                    print(f"logged metrics: {list(metrics_dict.keys())}")
                
                csv_path = os.path.join(BEST_PARAMETERS_DIRECTORY, sampling_method, f"{model_name}_trials.csv")
                plot_trial_evolution_path = os.path.join(REPORTS_TUNING_DIRECTORY, sampling_method, f"{model_name}_parameter_per_trial_trend.png")
                plot_optuna_trial_trends(csv_path=csv_path, output_directory=plot_trial_evolution_path)
                mlflow.log_artifact(plot_trial_evolution_path, artifact_path="trial_plots")
                print(f"Logged plots: {plot_trial_evolution_path}")


            mlflow.end_run()
        print(f"mlflow run completed: {run_name}")

#=============================================
# 6. Driver function
#=============================================
if __name__ == "__main__":
    for sampler in SAMPLING_METHODS:
        for model in MODEL_LIST:
            for mode in MODES:
                log_run_to_mlflow(sampler, model, mode)

    print("\nAll mlflow runs completed!!!")
    print("\nMLflow UI: mlflow ui --backend-store-uri mlruns")
    
            
            
                    



