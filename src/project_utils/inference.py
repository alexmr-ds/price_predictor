"""
Utility functions required to support the car price prediction web application (app.py)
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable

import numpy as np
import joblib
import math
import pandas as pd
from pathlib import Path

from .paths import PROJ_ROOT, CLEANED_DATA_DIR

# Locations
MODELS_DIR = PROJ_ROOT / "src" / "models"
PREPROCESSOR_PATH = PROJ_ROOT / "src" / "project_utils" / "pre_pipeline.joblib"
CLEANED_MODELS_PATH = PROJ_ROOT / "src" / "project_utils" / "cleaned_models.json"
FINAL_MODEL_PATH = MODELS_DIR / "final_stacking_model.joblib"

# Raw schema expected by the preprocessing pipeline.
RAW_FEATURE_COLUMNS = [
    "Brand",
    "model",
    "year",
    "transmission",
    "mileage",
    "fuelType",
    "engineSize",
    "hasDamage",
    "tax",
    "previousOwners",
    "mpg",
]


def load_final_model(path: Path = FINAL_MODEL_PATH):
    """Load the final stacking/ensemble model."""
    return joblib.load(path)


def to_int_bounds(lo, hi, median):
    """
    Convert numeric bounds to integer values suitable for UI components
    """
    lo = int(math.floor(lo))
    hi = int(math.ceil(hi))
    median = int(round(median))
    return lo, hi, median


def categorical_options(
    df: pd.DataFrame, cat_cols: list, brand_col: str = "Brand", model_col: str = "model"
):
    """
    Extract available categorical feature values grouped by brand-model pairs.

     Parameters
    ----------
    df : pandas.DataFrame
        Input dataset containing brand, model, and categorical feature columns.
    cat_cols : list of str
        Names of categorical columns for which available options should be extracted.
    brand_col : str, default="Brand"
        Column name identifying the vehicle brand.
    model_col : str, default="model"
        Column name identifying the vehicle model.

    Returns
    -------
    cat_cols_dict: dict
        Dictionary indexed by (brand, model) tuples, where each value is a dictionary
        mapping categorical column names to lists of available category values.
    """

    df = df.dropna().copy()
    cat_cols_dict = {}
    for (b, m), g in df.groupby([brand_col, model_col]):
        cats = {col: g[col].unique().tolist() for col in cat_cols}
        cat_cols_dict[(b, m)] = cats

    return cat_cols_dict


def load_brand_model_mapping(
    path: Path = CLEANED_MODELS_PATH,
) -> Dict[str, Iterable[str]]:
    """Load dictionary of allowed models per brand."""
    with open(path, "r") as f:
        return json.load(f)


def build_brand_model_ranges(
    df: pd.DataFrame = pd.read_csv(
        os.path.join(CLEANED_DATA_DIR, "non_engineered_train_data.csv")
    ),
    brand_col: str = "Brand",
    model_col: str = "model",
    q_low: float = 0.01,
    q_high: float = 0.99,
    min_group_size: int = 5,
    cat_cols: list = ["transmission", "fuelType"],
):
    """
    Compute brand-model-specific ranges and categorical options.

    Parameters
    ----------
    df : pandas.DataFrame, optional
        Input dataset used to compute feature ranges. If not provided, the cleaned
        training dataset is loaded from disk.
    brand_col : str, default="Brand"
        Column name identifying the vehicle brand.
    model_col : str, default="model"
        Column name identifying the vehicle model.
    q_low : float, default=0.01
        Lower quantile used to define the minimum bound of numeric features.
    q_high : float, default=0.99
        Upper quantile used to define the maximum bound of numeric features.
    min_group_size : int, default=5
        Minimum number of observations required to compute brand–model–specific
        ranges. Groups smaller than this threshold fall back to global ranges.
    cat_cols : list of str, default=["transmission", "fuelType"]
        Names of categorical columns for which available options are extracted.

    Returns:
        ranges; dict
            ranges[(brand, model)][col] = {"min", "max", "median"}
            Uses quantiles to avoid ouliers runining slider limits.
            Falls back to global ranges if group is too small.
        cat_cols_dict
            cat_cols_dict[(brand, model)][col] = []
    """

    # Stores categorical options
    cat_cols_dict = categorical_options(df, cat_cols=cat_cols)

    # DataFrame and list without 'transmission' and 'fuelType' columns
    df = df.drop(columns=cat_cols).dropna().copy()

    excluded_cols = set(cat_cols) | {brand_col, model_col}

    feature_cols = [col for col in RAW_FEATURE_COLUMNS if col not in excluded_cols]

    # Global fallback ranges
    global_ranges = {}
    for col in feature_cols:
        # DataFrame with only numeric columns
        df_num = df.drop(columns=[brand_col, model_col]).copy()
        lo, hi, med = to_int_bounds(
            df_num[col].quantile(q_low),
            df_num[col].quantile(q_high),
            df_num[col].median(),
        )
        global_ranges[col] = {"min": lo, "max": hi, "default": med}

    # Dictionary of feature value ranges grouped by (brand, model)
    ranges = {}
    grouped = df.groupby([brand_col, model_col])
    for (b, m), g in grouped:
        if len(g) < min_group_size:
            # too few points -> use global
            ranges[(b, m)] = global_ranges
            continue

        r = {}
        for col in feature_cols:
            lo, hi, med = to_int_bounds(
                g[col].quantile(q_low),
                g[col].quantile(q_high),
                g[col].median(),
            )
            r[col] = {"min": lo, "max": hi, "default": med}
        ranges[(b, m)] = r

    return ranges, cat_cols_dict


def build_raw_input(payload: Dict[str, Any]) -> pd.DataFrame:
    """
    Build a single-row DataFrame matching the raw feature schema.
    Unknown keys are ignored; missing keys become NaN.
    """
    cols = {col: payload.get(col) for col in RAW_FEATURE_COLUMNS}
    df = pd.DataFrame([cols])
    # Ensure we have exactly the right columns in the right order
    df = df[RAW_FEATURE_COLUMNS]

    return df


def predict_price(model, features_df: pd.DataFrame) -> float:
    """Run price predicitions."""

    pred = model.predict(features_df)

    return float(np.ravel(pred)[0])
