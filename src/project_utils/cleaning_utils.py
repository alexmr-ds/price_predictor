"""
Categorical text standardisation utilities and feature engineering.

This module provides helper functions for cleaning and standardising categorical
string variables using controlled vocabularies and fuzzy string matching. It is
primarily designed for preprocessing high-cardinality categorical features, such
as vehicle models, where spelling variations, abbreviations, and inconsistencies
are common.

In addition, the module includes custom functions implementing the feature
engineering procedures adopted in this project, ensuring consistency and
reproducibility across training and evaluation datasets.
"""

import pandas as pd
import numpy as np
from difflib import get_close_matches


def category_analyzer(category: str, valid_categories: list, cutoff: int = 0):
    """Cleans and standardizes categorical text values by matching them to a predefined list
    of valid categories using fuzzy string matching.
    Missing values are preserved
    Parameters
    ----------
    category : str or object
        The category value to clean or match. Can be a string, None, or NaN.
    valid_categories : list of str
        A list of accepted category names to which the input value will be compared.
    cutoff : float, optional (default=0)
        The minimum similarity ratio (0–1) required for a match using `difflib.get_close_matches`.
        Lower values allow looser matches.

    Returns
    -------
    str or np.nan
        - The cleaned and matched category name in lowercase, if a close match is found or the
          value is already valid.
        - `np.nan` if the input is missing.

    """
    # Handle missing values
    if pd.isna(category):
        return np.nan

    # If already valid, keep it
    if category in valid_categories:
        return category

    # Otherwise find closest valid match
    matches = get_close_matches(category, valid_categories, n=1, cutoff=cutoff)

    if matches:
        valid_match = matches[0]
        return valid_match


def model_cleaner(df: pd.DataFrame, dct: dict, group: list):
    """
    Clean the 'model' column brand by brand using 'category_analyzer',
    based on a dictionary of allowed/cleaned model names per brand.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least 'Brand' and 'model' columns.
    dct : dict
        Dictionary mapping each brand (key) to a list of valid model names (value).
    group : list
        Brands to process (exact matches to values in df['Brand'])

    Returns
    -------
    pd.Series
        A Series of cleaned model names, aligned with df.index.
    """
    # Iterates over each brand and applies category cleaning fuction
    cleaned_series_list = []
    for g in group:
        # Subset by brand
        brand_subset = df.query("Brand == @g")["model"]

        # Applies cleaning function
        cleaned_series = brand_subset.apply(
            lambda model: category_analyzer(model, dct[g])
        )
        cleaned_series_list.append(cleaned_series)

    # Concatenate all Series into one, preserving original index order
    clean_models = pd.concat(cleaned_series_list).reindex(df.index)
    return clean_models


def engineer_features(
    df: pd.DataFrame,
    log_features: list = [
        "mileage",
        "mpg",
    ],
    current_year: int = 2020,
) -> pd.DataFrame:
    """
    Apply feature engineering steps.

    This function performs logarithmic transformations on selected skewed
    features, constructs composite categorical variables, derives age-based
    features, and engineers interaction features related to fuel type and
    engine size.

    Parameters
    ----------
    df : pd.DataFrame
        Input test dataset.
    log_features : list of str
        Names of numerical features to be log-transformed.
    current_year : int, default=2020
        Reference year used to compute vehicle age.

    Returns
    -------
    pd.DataFrame
        Test dataset with engineered features.
    """
    df = df.copy()

    # --- Log transformations for skewed numerical features ---
    for feature in log_features:
        if feature in df.columns:
            df[f"log_{feature}"] = np.log1p(df[feature])
            df.drop(columns=feature, inplace=True)

    # --- Brand–model composite feature ---
    if {"Brand", "model"}.issubset(df.columns):
        df["brand_model"] = df["Brand"].astype(str) + "_" + df["model"].astype(str)
        df.drop(columns=["Brand", "model"], inplace=True)

    # --- Age-based feature engineering ---
    if "year" in df.columns:
        df["age"] = current_year - df["year"]
        df.drop(columns="year", inplace=True)

    # --- Categorical feature interactions ---
    if {"fuelType", "brand_model"}.issubset(df.columns):
        df["fuelType_model"] = df["fuelType"] + "_" + df["brand_model"]

    # --- Engine size binning ---
    if "engineSize" in df.columns:
        df["engineSize_binned"] = pd.cut(
            df["engineSize"],
            bins=[0, 1.0, 1.5, 2.0, 2.5, 5.0],
            labels=["tiny", "small", "medium", "large", "huge"],
        )

        # Fuel type × engine size interaction
        if "fuelType" in df.columns:
            df["fuelType_engineSize"] = (
                df["fuelType"] + "_" + df["engineSize_binned"].astype(str)
            )

        df.drop(columns="engineSize", inplace=True)

    return df
