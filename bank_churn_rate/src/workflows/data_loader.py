"""
Loads Raw Data into a pandas Dataframe with basic validatio. Used by EDA, Feature
Engineering, training and scoring pipelines.
"""
import os
import pandas as pd
from typing import Optional

class DataLoader:
    """
    Reusable class for loading csv data with schema validationand optional sampling
    """
    def __init__(self,
                 filepath: str,
                 required_columns: Optional[list] = None):
        self.filepath = filepath
        self.required_columns = required_columns

        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Dataset not found at: {self.filepath}")

    def load(self, sample_fraction: float = 1.0) -> pd.DataFrame:
        """
        Load CSV File and optionally return a sample for fast prototyping.
        sample_fractiion=1.0 -> full data by default
        """
        data = pd.read_csv(self.filepath)
        print(f"Loaded Dataset: {self.filepath}")
        print(f"Shape of the dataset is: {data.shape}")

        if self.required_columns:
            self.validate_schema(data)

        if sample_fraction < 1.0:
           data = data.sample(frac=sample_fraction,
                              random_state=42)
           print(f"Returning Sample: {data.shape}")
        return data
    
    def validate_schema(self, data: pd.DataFrame):
        """
        Ensure Required columns exist in dataset.
        """
        missing_columns = [column for column in self.required_columns if column not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing Requiredd columns: {missing_columns}")
        else:
            print(f"Schema validated: All required columns present.")
    
if __name__ == "__main__":
    # Example Test Run 
    FILE = "../data/raw/churn.csv"
    REQUIRED = ["CustomerId", "Age", "Exited"]
    # for minimal schema check

    loader = DataLoader(filepath=FILE,
                        required_columns=REQUIRED)
    data = loader.load(sample_fraction=1.0)
    print("\nPreview:")
    print(data.head())    




