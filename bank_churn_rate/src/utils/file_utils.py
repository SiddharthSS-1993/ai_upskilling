"""
Reusable Helper functions for reading/writing files (CSV, JSON Pickle) and
Handling safe paths.

Used across EDA, Feature Engineering, Training, LLM Modules etc.
"""
import os
import sys
import json
import pickle
import pandas as pd
from typing import Any
main=os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(main)
from commons import ensure_directory

###############################################
# CSV Helpers
###############################################
def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV Not Found: {path}")
    else:
        return pd.read_csv(path)
    
def save_csv(data: pd.DataFrame, path: str, index: bool =False):
    ensure_directory(os.path.dirname(path))
    data.to_csv(path, index=index)
    print(f"Saved CSV: {path}")

###############################################
# Pickle Helpers (Models, Objects)
###############################################
def save_pickle(object: Any, path: str):
    ensure_directory(os.path.dirname(path))
    with open(path, "wb") as file:
         pickle.dump(object, file)
    print(f"Pickle Saved: {path}")

def load_pickle(path: str) -> Any:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pickle Not Found: {path}")
    else:
        with open(path, "rb") as file:
            return pickle.load(file)

###############################################
# JSON Helpers (Metadata, Reports)
###############################################
def save_json(data: dict, path: str):
    ensure_directory(os.path.dirname(path))
    with open(path, "w") as file:
         json.dump(data, file, indent=4)
    print(f"JSON Saved: {path}")

def load_json(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON Not Found: {path}")
    else:
        with open(path, "r") as file:
            return json.load(file)
