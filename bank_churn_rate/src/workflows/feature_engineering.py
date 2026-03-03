#=====================================
# 1. Imports
#=====================================
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
main=os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(main)
from src.workflows.data_loader import DataLoader
from commons import ensure_directory
from src.utils.file_utils import save_pickle, save_csv 
from imblearn.over_sampling import SMOTE, ADASYN
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


id_columns = ["RowNumber", "CustomerId", "Surname"]

#=====================================
# 2. Load Raw Data
#=====================================
FILE = os.path.join(ROOT, "data", "raw", "churn.csv")
loader = DataLoader(filepath=FILE)
data = loader.load(sample_fraction=1.0)
print("Data loaded successfully: ", FILE)
data.head()

#=====================================
# 3. Drop ID Like Columns
#=====================================
data = data.drop(columns=[column for column in id_columns if column in data.columns], errors="ignore")
print(f"\nDropped ID Columns: {id_columns}\n")

#=====================================
# 4. Handle Missing Values
#=====================================
missing_percentage = data.isna().mean().sort_values(ascending=False)
print("Missing values percentage per column:")
print(missing_percentage)

numeric_columns = data.select_dtypes(include=["int64", "Float64"]).columns.to_list()
categorical_columns = data.select_dtypes(exclude=["int64", "Float64"]).columns.to_list()
TARGET_COLUMN = "Exited"

# If no missing values skip imputation
if missing_percentage.sum() == 0:
    print("No missing values, Skipping Imputation.")
else:
    # Split numeric and categorical columns to treat missing values accordingly
    # Fill numeric columns with median and categorical columns with mode(highest
    # frequency).
    for column in numeric_columns:
        data[column].fillna(data[column].median(), inplace=True)

    for column in categorical_columns:
        data[column].fillna(data[column].mode()[0], inplace=True)

    print("Missing values Handled -> numeric = median, categorical = mode\n")

#=====================================
# 5. Encoding + Scaling Setup
#=====================================
# Exclude target 
numeric_columns = [column for column in numeric_columns if column not in [TARGET_COLUMN]]
categorical_columns = [column for column in categorical_columns if column not in [TARGET_COLUMN]]

print("Numeric columns: ", numeric_columns)
print("Categorical columns: ", categorical_columns)

# Build Preprocessing Pipeline
preprocessor = ColumnTransformer(transformers=[("numeric_columns", StandardScaler(), numeric_columns),
                                               ("categorical_columns", OneHotEncoder(), categorical_columns)])

print("Column Transformer Created!!!")

#=====================================
# 6. Apply Preprocessing and on Feature Data
#=====================================
X = data.drop(columns=[TARGET_COLUMN])
y = data[TARGET_COLUMN]

print("Feature Matrix shape before transformation: ", X.shape)
print("Target shape: ", y.shape)


#=====================================
# 7. Class Imbalance Handling SMOTE + ADASYN
#=====================================
# Train Validation split before resampling
X_train, X_validation, y_train, y_validation = train_test_split(X,
                                                                y,
                                                                test_size=0.2,
                                                                stratify=y,
                                                                random_state=42)
save_pickle(X_train, os.path.join(ROOT, "data", "processed", "X_train.pkl"))
save_pickle(X_validation, os.path.join(ROOT, "data", "processed", "X_validation.pkl"))
X_train = preprocessor.fit_transform(X_train)
X_validation = preprocessor.transform(X_validation)


print("Transformation Complete!! \nShape after encoding +scaling: ", X_train.shape)
print("\nBefore resampling!!!")
print(f"\nTrain class distribution: {y_train.value_counts(normalize=True)}")
print(f"\nValidation class distribution: {y_validation.value_counts(normalize=True)}")

# Apply SMOTE and ADASYN only on training data
smote = SMOTE(random_state=42)
adasyn = ADASYN(random_state=42)

X_smote, y_smote = smote.fit_resample(X_train, y_train)
X_adasyn, y_adasyn = adasyn.fit_resample(X_train, y_train)

print("\nAfter SMOTE Resampling!!!")
print(f"\nTrain Class distribution: {y_smote.value_counts(normalize=True)}")


print("\nAfter ADASYN Resampling!!!")
print(f"\nTrain Class distribution: {y_adasyn.value_counts(normalize=True)}")

# Save results
# Save fitted transformer for reuse
ensure_directory(os.path.join(ROOT, "data", "processed"))
save_pickle((X_smote, y_smote, X_validation, y_validation) , os.path.join(ROOT, "data", "processed", "train_SMOTE.pkl"))
save_pickle((X_adasyn, y_adasyn, X_validation, y_validation) , os.path.join(ROOT, "data", "processed", "train_ADASYN.pkl"))

ensure_directory(os.path.join(ROOT,"models"))
save_pickle(preprocessor, os.path.join(ROOT, "models", "feature_engineering", "preprocessor.pkl"))

print("processor saved to ROOT/models/feature_engineering/")
print("train_SMOTE saved to ROOT/data/processed/")
print("train_ADASYN saved to ROOT/data/processed/")













 



