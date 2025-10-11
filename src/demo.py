import pandas as pd
import seaborn as sns
import mlflow

mlflow.set_experiment("TitanicPipeline")
with mlflow.start_run(run_name="Data_Cleaning_v1") as run:
    # Data Ingestion
    data = pd.read_csv("../data/archive/Titanic-Dataset.csv")

    # Data description 
    print(data.head())
    print(data.describe())
    print(data.isnull().sum())
    print(data.isna().sum())

    # Data Cleaning
    data["Age"] = data["Age"].fillna(data["Age"].median())
    mlflow.log_param("imputation_method", "median")

    data["Survived"] = data["Survived"].astype('category')
    data.drop_duplicates(inplace=True)

    # Removing Outliers from Inter Quartile Range
    q1, q3 = data["Fare"].quantile([0.25, 0.75])
    iqr = q3-q1
    print(str(iqr) + ' ' + str(q1) + ' ' + str(q3))
    
    data = data[(data["Fare"] >= q1 - 1.5*iqr) & \
                (data["Fare"] <= q3 + 1.5*iqr)]
    mlflow.log_param("outlier_method", "IQR")
    
    # Data Visualisation
    sns.histplot(data["Age"], kde=True)
    mlflow.log_metric("rows after cleaning", len(data))
    run =mlflow.end_run()
