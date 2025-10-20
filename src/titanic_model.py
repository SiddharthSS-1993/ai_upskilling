#Data Ingestion and Description
import pandas as pd

# DATA HANDLING
import scipy.stats as stats
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import optuna
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
# Explainability 
import shap
# mlflow
import mlflow, mlflow.sklearn

# Visualization
import seaborn as sns
import matplotlib.pyplot as  plt
import warnings
warnings.filterwarnings("ignore")

def build_pipeline(trial: optuna.Trial) -> Pipeline:
    # Search Space for Random Forest
    n_estimators = trial.suggest_int("n_estimators", 100, 600, step=50)
    max_depth = trial.suggest_int("max_depth", 3, 20)
    min_sample_split = trial.suggest_int("min_sample_split", 2, 20)
    min_sample_leaf = trial.suggest_int("min_sample_leaf", 1, 10)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])

    model = RandomForestClassifier(n_estimators=n_estimators,
                                   max_depth=max_depth,
                                   min_samples_split = min_sample_split,
                                   min_samples_leaf=min_sample_leaf,
                                   max_features=max_features,
                                   random_state=42,
                                   n_jobs=-1)
    pipe = Pipeline([("preprocess", preprocessor),
                     ("model", model)])
    return pipe

def objective(trial: optuna.Trial) -> float:
    pipe = build_pipeline(trial)
    # Use ROC-AUC (higher is better)
    scores = cross_val_score(pipe, X, Y, scoring="roc_auc", cv=cv, n_jobs=-1)
    return scores.mean()

# Initialize optuna study
study = optuna.create_study(direction="maximize", study_name="roc_auc_study")
# MLFLOW Experiment For ML Pipeline
mlflow.set_experiment("TitanicPipeline")
with mlflow.start_run(run_name="Data_Cleaning_v1") as run:
    # 1. Data Ingestion
    data = pd.read_csv("../data/archive/Titanic-Dataset.csv")
    mlflow.log_artifact("../data/archive/Titanic-Dataset.csv")
    
    # 1.1 Data description 
    print(data.head())
    print(data.describe())
    print(data.isnull().sum())
    print(data.isna().sum())

    # 2 Data Cleaning
    numerical_features = ["Age", "Fare"]
    categorical_features = ["Sex", "Embarked"]
    
    data["Age"] = data["Age"].fillna(data["Age"].median())
    data["Fare"] = data["Fare"].fillna(data["Fare"].median())

    for column_name in categorical_features:
        data[column_name].fillna("Unknown", inplace=True)

    mlflow.log_param("imputation_method", "median")

    data["Survived"] = data["Survived"].astype('category')
    data.drop_duplicates(inplace=True)

    # 2.1 Removing Outliers from Inter Quartile Range
    q1, q3 = data["Fare"].quantile([0.25, 0.75])
    iqr = q3-q1
    print(str(iqr) + ' ' + str(q1) + ' ' + str(q3))
    
    data = data[(data["Fare"] >= q1 - 1.5*iqr) & \
                (data["Fare"] <= q3 + 1.5*iqr)]
    mlflow.log_param("outlier_method", "IQR")
    
    # 2.2 Data Visualisation
    
    sns.histplot(data["Age"], kde=True)
    plt.savefig("histplot.png")
    mlflow.log_param("Frequency Visualisations", "histplot")

    sns.boxplot(x=data["Fare"])
    plt.savefig("boxplot.png")
    mlflow.log_param("Outlier Visualisation", "Boxplot")

    stats.probplot(data["Age"], dist="norm", plot=plt)
    plt.savefig("probplot.png", dpi=160, bbox_inches="tight")
    print("skewness: ", data["Age"].skew())
    skewed = data["Age"].skew()
    print("kurtosis: ", data["Age"].kurtosis())
    kurt = data["Age"].kurtosis()
    mlflow.log_param("Normalization Visualisations", "Probability plot")

    sns.pairplot(data, hue="Survived")
    plt.savefig("pairplot.png")
    mlflow.log_param("Feature Correlation Visualisations", "Pairplot")

    mlflow.log_metric("rows after cleaning", len(data))
    plt.show()

    # 2.3 Confidence Interval
    mean, sd, counts = data["Age"].mean(), data["Age"].std(), len(data["Age"])
    ci = stats.norm.interval(0.95, loc=mean, scale = sd/np.sqrt(counts))
    (lower, upper) = ci
    print(f"Confidence Interval at 95% has a lower bound: {lower} and upper bound {upper}")

    # 3 Feature Selection
    # For numerical and categorical features look at 2. Data cleaning section
    # 3.1 Prepare features(X) and target(Y) Datasets
    target_feature = "Survived"

    X = data[numerical_features + categorical_features].copy()
    Y = data[target_feature].copy()

    # 3.1 Train Test Split
    # Random state = seed is used to keep the randomness constant instead of rerandomizing at every build or new call
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # 4. Define Preprocessor
    preprocessor = ColumnTransformer([("numericals", StandardScaler(), numerical_features),
                                      ("categoricals", OneHotEncoder(handle_unknown="ignore"), categorical_features),
                                    ])
    
    # 5. Cross Validation and Hyper Parameter Tuning with Optuna
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    study.optimize(objective, n_trials=25, show_progress_bar=False)
 
    # Log best trial
    best = study.best_trial
    print(study.best_value)
    print("\n" + str(best.params))
    mlflow.log_metric("best_cv_auc", study.best_value)
    mlflow.log_params(best.params)
    
    # 5. Build Model Pipeline Static
    pipeline = Pipeline([("preprocess", preprocessor),
                         ("model", RandomForestClassifier(n_estimators=300,
                                                          random_state=42,
                                                          n_jobs=1))])
    pipeline.fit(X_train, Y_train)
    predictor = pipeline.predict(X_test)
    probsX = pipeline.predict_proba(X_test)[:, 1]

    # Build Model Pipeline with best hyperparameters
    final_pipe = build_pipeline(best)
    final_pipe.fit(X, Y)

    mlflow.sklearn.log_model(final_pipe, "final_rf_model")

    # Visualize Optuna History Locally and save artifact
    optimization_history = optuna.visualization.matplotlib.plot_optimization_history(study)
    [optimization_history.write_image("optuna_history.png"
                    ) if hasattr(optimization_history, "write_image")
                      else plt.savefig("optuna_history.png")]
    parameter_importance = optuna.visualization.matplotlib.plot_param_importances(study)
    plt.savefig("optuna_parameter_importance.png")
    
    parallel_coordinates = optuna.visualization.matplotlib.plot_parallel_coordinate(study)
    plt.savefig("optuna_parallel_coordinates.png")
    
    
    mlflow.log_artifact("optuna_history.png")
    mlflow.log_artifact("optuna_parameter_importance.png")
    mlflow.log_artifact("optuna_parallel_coordinates.png")

    # 6. Model Evaluation
    accuracy = accuracy_score(Y_test, predictor)
    auc = roc_auc_score(Y_test, probsX)

    print("Best AUC", study.best_value)
    print("Best Hyper Parameters", study.best_trial.params)
    print(f"Accuracy: {accuracy: 3f}")
    print(f"ROC-AUC: {auc: 3f}")
    print(classification_report(Y_test, predictor))
    
    # Explainability with SHAP
    # Get processed test data and model

    rf_model = pipeline.named_steps["model"]
    preprocessed = pipeline.named_steps["preprocess"]
    X_test_prepared = preprocessed.transform(X_test)

    # Retrieve final feature names
    ohe = preprocessed.named_transformers_["categoricals"]
    feature_names = numerical_features + list(ohe.get_feature_names_out(categorical_features))
    X_test_df = pd.DataFrame(X_test_prepared, columns=feature_names)
    # Create SHAP Explainer and Values
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test_prepared)
    print(shap_values.shape)
    # Global Summary Plot
    shap.summary_plot(shap_values, X_test_df, feature_names)

    # Local Plot
    i = 0 # Pick a row index

    shap.force_plot(explainer.expected_value[1], shap_values[:, :, 1][i, :], feature_names)
    
    run = mlflow.end_run()


# MLFLOW Experiment for Statistical Analysis
mlflow.set_experiment("Titanic Stats")
with mlflow.start_run(run_name="Statistical Analysis") as run:
    mlflow.log_metric("mean_age", mean)
    mlflow.log_metric("age_ standard_deviation", sd)
    mlflow.log_metric("age_skew", skewed)
    mlflow.log_metric("age_kurtosis", kurt)
    run = mlflow.end_run()


# MLFLOW Experiment for Model Evaluation and Tracking
mlflow.set_tracking_uri("file:///C:/MachineLearningCource/AI Upskilling/ai upskilling/mlruns/results_evaluation_tracking")
mlflow.set_experiment("feature_evaluation_xai")

with mlflow.start_run(run_name="Random_Forest_Baseline_Titanic"):
    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", 300)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("ROC_AUC", auc)
    mlflow.sklearn.log_model(pipeline, "models")

    # Save and log SHAP Summary as Artifact
    plt.figure()

    shap.summary_plot(shap_values, X_test_df, feature_names, show=False)
    plt.savefig("shap_summary.png", dpi=160, bbox_inches="tight")

    plt.close()

    mlflow.log_artifact("shap_summary.png")
    mlflow.end_run()    


