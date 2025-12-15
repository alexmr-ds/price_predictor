"""
Categorical text standardisation utilities.

This module provides helper functions for cleaning and standardising categorical
string variables using controlled vocabularies and fuzzy string matching.
It is primarily designed for preprocessing high-cardinality categorical features
such as vehicle models, where spelling variations, abbreviations, or inconsistencies
are common.

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
