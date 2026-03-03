import streamlit as st
from pathlib import Path
import os
from typing import Dict, List, Optional, Tuple
from src.utils.file_utils import load_pickle

ROOT = Path(__file__).resolve().parents[1]
import numpy as np
import pandas as pd
import glob

def list_images(directory: Path, extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg")) -> List[Path]:
    directory = Path(directory)
    if not directory.exists():
        return []
    files = []
    for extension in extensions:
        files.extend(directory.glob(f"*{extension}"))
    return sorted(files)

def available_models_in_directory(method_directory: Path) -> List[str]:
    method_directory = Path(method_directory)
    if not method_directory.exists():
        return []
    pickles = sorted(method_directory.glob("*.pkl"))
    names = []
    for p in pickles:
        name = p.stem
        if name.endswith("_best"):
            name = name[: -len("_best")]
        names.append(name)
    return names

def load_preprocessor() -> object:
    directory = os.path.join(ROOT, "models", "feature_engineering", "preprocessor.pkl")
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Preprocessor not found at {directory}")
    return load_pickle(directory)

def load_tuned_model(method: str, model_name:str):
    directory = os.path.join(ROOT, "models", "tuning_models", method, f"{model_name}_best.pkl")
    if not os.path.exists(directory):
        alternate_directory = os.path.join(ROOT, "models", "tuning_models", method, f"{model_name}.pkl")
        if os.path.exists(alternate_directory):
            directory = alternate_directory
        else: raise FileNotFoundError(f"Model not found at {directory}")
    return load_pickle(directory)

def get_expected_raw_columns() -> List[str]:
    return ["RowNumber",
            "CustomerId",
            "Surname",
            "CreditScore",
            "Geography",
            "Gender",
            "Age",
            "Tenure",
            "Balance",
            "NumOfProducts",
            "HasCrCard",
            "IsActiveMember",
            "EstimatedSalary"]

def normalize_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Fix CSV Column Issues
    """
    rename_map = {}

    for column in data.columns:
        column_stripped = column.strip()
        rename_map[column] = column_stripped

    data = data.rename(columns=rename_map)

    # Common variants
    variants = {
        "Estimated Salary": "EstimatedSalary",
        "Estimated_Salary": "EstimatedSalary",
        "estimatedsalary": "EstimatedSalary",
        "Exited ": "Exited" 
    }
    for old_name, new_name in variants.items():
        if old_name in data.columns and new_name not in data.columns:
            data = data.rename(columns={old_name: new_name})
    return data

def build_single_row_template() -> pd.DataFrame:
    """
    A safe default single row template for user inputs.
    """
    data = {"RowNumber": 1,
            "CustomerId": 0,
            "Surname": "Unknown",
            "CreditScore": 650,
            "Geography": "France",
            "Gender": "Male",
            "Age": 35,
            "Tenure": 5,
            "Balance": 50000.0,
            "NumOfProducts": 1,
            "HasCrCard": 1,
            "IsActiveMember": 1,
            "EstimatedSalary": 60000.0,
            }
    return pd.DataFrame([data])

def predict_proba(model, X_transformed: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_transformed)
        # Binary classification -> proba[:,1]
        return proba[:, 1]
    # Sigmoid classification
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_transformed)
        # logistic transform
        return 1.0 / (1.0 + np.exp(-scores))
    # predict 0/1 directly
    predictions = model.predict(X_transformed)
    return np.asarray(predictions, dtype=float)



           





