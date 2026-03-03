"""
Reusable Helper functions for plotting different charts(Histograms +
Kernel density function, Box plots, Bar charts, plot grid) and
Handling safe paths.

Used across EDA, LLM Modules etc.
"""
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Optional
import math
import numpy as np
main=os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(main)
from commons import ensure_directory

###############################################
# Save Figure Helper
###############################################
def save_figure(figure, filename: str):
    """
    Saves a matplotlib figure to reports/figures/subfolder/filename.png
    """
    folder = os.path.join("../reports", "eda")
    ensure_directory(folder)
    full_path = os.path.join(folder, f"{filename}.png")
    figure.savefig(full_path, bbox_inches="tight", dpi=150)
    plt.close(figure)
    print(f"Figure Saved: {full_path}")

###############################################
# Single Feature Plot
###############################################
def plot_numeric_distribution(data: pd.DataFrame,
                              numeric_column: List[str],
                              target_column: Optional[str] = None):
    """
    Histogram + Kernel Density Equation with target hue.
    """
    n = len(numeric_column)
    rows = math.ceil(n / 3)
    figure, axes = plt.subplots(rows, 3, figsize=(18, 5*rows))
    axes = axes.flatten()

    for i, column in enumerate(numeric_column):
        ax = axes[i]
        if target_column:
            sns.histplot(data,
                         x=column,
                         hue=target_column,
                         kde=True,
                         stat="density",
                         ax=ax)
        else:
            sns.histplot(data,
                         x=column,
                         kde=True,
                         stat="density",
                         ax=ax)
        ax.set_title(f"{column}")

    # Remove unused Sub plots.
    for j in range(i + 1, len(axes)):
        figure.delaxes(axes[j])

    figure.suptitle("Numeric Distribution By Target", fontsize=18)
    save_figure(figure, "numeric_distribution")

###############################################
# Boxplots
###############################################
    
def plot_boxplots(data: pd.DataFrame,
                  numeric_column: List[str],
                  target_column: str):
    """
    Box plot for numeric column by target class.
    """
    n = len(numeric_column)
    rows = math.ceil(n / 3)
    figure, axes = plt.subplots(rows, 3, figsize=(18, 5*rows))
    axes = axes.flatten()

    for i, column in enumerate(numeric_column):
        ax = axes[i]
        sns.boxplot(data=data,
                    x=target_column,
                    y=column,
                    ax=ax)
        ax.set_title(f"{column} vs {target_column}")

    # Remove unused Sub plots.
    for j in range(i + 1, len(axes)):
        figure.delaxes(axes[j])

    figure.suptitle("Numeric Boxplots By Target", fontsize=18)
    save_figure(figure, "numeric_boxplot")

###############################################
# Categorical Plots
###############################################
def plot_categorical_bars(data: pd.DataFrame,
                          categorical_column: List[str],
                          target_column: str):
    """
    Plots Churn Rate Bar chart of all categorical columns.
    """
    n = len(categorical_column)
    columns = 3
    rows = math.ceil(n / columns)
    
    figure, axes = plt.subplots(rows, 3, figsize=(18, 5*rows))
    axes = np.array(axes).reshape(-1)

    for i, column in enumerate(categorical_column):
        ax = axes[i]
        grouped = (data.groupby(column)[target_column]
                    .mean()
                    .sort_values(ascending=False)
                    .rename("churn_rate"))
        sns.barplot(x=grouped.index, y=grouped.values, ax=ax)
        ax.set_title(f"{column} vs {target_column}")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    # Remove unused Sub plots.
    for j in range(i + 1, len(axes)):
        figure.delaxes(axes[j])

    figure.suptitle("Categorical Churn Rates", fontsize=18)
    save_figure(figure, "plot_categorical_bars")

###############################################
# Correlation Heatmap
###############################################
def plot_correlation_heatmap(data: pd.DataFrame,
                             numeric_columns: list[str],
                             target_column: str):
    """
    Correlation Heatmap including target. 
    """
    correlation = data[numeric_columns + [target_column]].corr(numeric_only=True)

    figure, axes = plt.subplots(figsize=(12, 8))
    sns.heatmap(correlation,
                cmap="coolwarm",
                center=0,
                annot=True,
                fmt=".2f",
                ax=axes)

    axes.set_title("Correlation Heatmap")
    save_figure(figure, "plot_correlation_heatmap")            




