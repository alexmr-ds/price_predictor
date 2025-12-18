"""
Utility functions used across the project for exploratory checks, preprocessing export,
and model evaluation.
"""

# Locate project root dir, enable package imports from src/
import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJ_ROOT / "src"))
# Load processed data directory path
from project_utils.paths import PRE_PIPELINE_DIR

# General imports
import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)


def count_outliers(df: pd.DataFrame, iqr_range: float = 2):
    """
    Count the number of outliers in a DataFrame or Series using the IQR rule per column.
    Returns both counts and a boolean mask.

    Parameters
    ----------
    df : pandas.DataFrame or pandas.Series
        Numeric data for outlier detection. For DataFrames, outliers are
        detected independently for each column.
    iqr_range : float, default=1.5
        IQR multiplier that defines the outlier bounds. Common values:

    1.5: Standard outlier detection (Tukey's fences)
    3.0: Conservative detection (extreme outliers only)
    Larger values result in fewer points being marked as outliers.

    Returns
    -------
    (counts, mask) : tuple
        counts:
            - pd.Series of integers outlier counts indexed by the numeric columns.
            - panda.DataFrame[bool] aligned with numeric coumns (True if outlier)
    """
    # Calculate quartiles
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1

    # Define bounds
    lower = q1 - iqr_range * iqr
    upper = q3 + iqr_range * iqr

    # Create mask for outliers
    outlier_mask = (df < lower) | (df > upper)

    # Count outliers per column
    outlier_counts = outlier_mask.sum()

    return outlier_counts, outlier_mask


def conditional_variance_comparison(df: pd.DataFrame, groups: list):
    """
    Compare each numeric feature's overall (global) variance with its conditional
    variance after grouping by one or more categorical columns

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing numerical and categorical features.

    groups : list
        A list of column names used to define the grouping structure. Must be a list
        of existing columns.

    Returns
    -------
    pd.DataFrame
        A DataFrame indexed by numerical feature names with two columns:
            - 'global_variance': variance computed over the entire dataset.
            - 'grouped_variance': mean variance across groups defined by 'groups'.

    Raises
    ------
    TypeError
        If 'groups' is not provided as a list

    """
    if not isinstance(groups, list):
        raise TypeError("Groups must be provided as a list of column names.")

    var_dict = {}
    for feature in df.select_dtypes(exclude="object").columns:
        global_var = df[feature].var()
        group_var = df.groupby(groups)[feature].var().mean()
        var_dict[feature] = (global_var, group_var)
    df_var = pd.DataFrame.from_dict(
        var_dict, columns=["global_variance", "grouped_variance"], orient="index"
    )
    return df_var


def column_cleaner(columns: list):
    """
    Remove transformer prefixes (e.g., 'num__', 'cat__') from feature names.

    This utility function is designed to clean the feature names produced by
    scikit-learn's `ColumnTransformer`, which typically prefixes output
    columns with strings such as 'num__' or 'cat__' to indicate their origin.
    For readability and downstream compatibility, this function strips these
    prefixes and returns only the original feature names.

    Parameters
    ----------
    columns : iterable of str
        The raw feature names returned by 'ColumnTransformer.get_feature_names_out()'.

    Returns
    -------
    list of str
        A list of cleaned feature names with transformer prefixes removed.
    """
    return [
        str(name).split("__", 1)[1] if "__" in str(name) else str(name)
        for name in columns
    ]


def get_column_transformer_from_pipeline(pipe: Pipeline):
    """
    Return the ColumnTransformer inside a pipeline, whether it is at the top level
    or nested under a 'preprocess' step.
    This function was specifically designed to enable the exportation of not only the 'pre-pipeline'
    but also the 'full_preprocessor' ('pre-pipeline') with filter methods.

    Parameters
    ----------
    pipe : Pipeline
        A scikit-learn Pipeline object that contains, either directly or nested,
        a step named 'column_transformer'.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        The first 'Columntransformer' can be found within the pipeline.

    Raises
    ------
    ValueError
        If no step named `"column_transformer"` is found at the top level or
        within the `"preprocess"` sub-pipeline.
    """
    #  Pre_pipeline case
    if "column_transformer" in pipe.named_steps:
        return pipe.named_steps["column_transformer"]

    # Full_preprocessor case
    if "preprocess" in pipe.named_steps:
        inner = pipe.named_steps["preprocess"]
        if hasattr(inner, "named_steps") and "column_transformer" in inner.named_steps:
            return inner.named_steps["column_transformer"]

    # If we reach here, we didn’t find it
    raise ValueError("No 'column_transformer' step found in the given pipeline.")


def preprocess_export_splits(
    pre_pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame | None = None,
    X_test: pd.DataFrame | None = None,
    y_train: pd.Series | None = None,
    clean_columns: bool = True,
    csv: bool = False,
):
    """
    Fit a preprocessing pipeline on the training data and transform the train/val/test
    splits. Returns the transformed datasets as pandas DataFrames and can optionally
    export them as CSV files.

    The function works with unsupervised preprocessing (e.g., imputers, scalers,
    one-hot encoders) and supervised encoders (e.g., target encoding) by passing
    'y_train' to 'fit' when provided.

    The workflow performed is as follows:
    1. Validate that any provided splits (val/test) have the same columns and order as 'X_train'.
    2. Fit 'pre_pipeline' on 'X_train' (and 'y_train' if provided).
    3. Transform 'X_train', and any provided 'X_val'/'X_test'.
    4. Derive output feature names from the pipeline’s 'ColumnTransformer'; if a downstream
       dropper step (e.g., 'drop_filtered_features') exposes 'get_feature_names_out',
       apply it to the base names. If 'clean_columns=True', strip transformer prefixes.
    5. Assemble dense DataFrames with preserved indices.
    6. Optionally export the DataFrames to CSV files under 'PRE_PIPELINE_DIR' (must be defined).

    Parameters
    ----------
    pre_pipeline : Pipeline
        A scikit-learn Pipeline object containing the preprocessing steps
        (e.g., imputers, encoders, scalers).
    X_train : pd.DataFrame
        Feature matrix used to fit the preprocessing pipeline.
    X_val : pd.DataFrame
        Validation feature matrix transformed using the fitted pipeline.
    X_test : pd.DataFrame
        Test feature matrix transformed using the fitted pipeline.
    y_train : pd.Series or None, optional (default=None)
        Target vector required for supervised encoding steps such as
        `TargetEncoder`. When None, the pipeline is fitted without `y`.
    clean_columns : bool, optional (default=True)
        If True, transformer prefixes (e.g., 'num__', 'cat__') are removed
        from the output feature names for improved readability.
    csv : bool, optional (default=False)
        If True, write CSVs to ``PRE_PIPELINE_DIR``:
        - ``X_train_preprocessed.csv``
        - ``X_val_preprocessed.csv`` (if applicable)
        - ``X_test_preprocessed.csv`` (if applicable)

    Returns
    -------
    df_train_pre : pd.DataFrame
        Preprocessed training set.
    df_val_pre : pd.DataFrame or None
        Preprocessed validtion set, or None if 'X_val' was None.
    df_test_pre : pd.DataFrame or None
        Preprocessed test set, or None if 'X_test' was None.

    Raises
    ------
    ValueError
        If 'X_train' vs 'X_val'/ 'X_test' columns differ in names or order.
    """
    # --- Validate column consistency only for datasets that exist --- #
    if X_val is not None:
        if not (X_train.columns == X_val.columns).all():
            raise ValueError("X_train and X_val have mismatching columns.")

    if X_test is not None:
        if not (X_train.columns == X_test.columns).all():
            raise ValueError("X_train and X_test have mismatching columns.")

    # -- Fit pre-processing pipeline to X_train and transform all sets --#
    if y_train is not None:  # For supervised encoders
        pre_pipeline.fit(X_train, y_train)
    else:
        pre_pipeline.fit(X_train)

    # Extracts feature names after the pipeline transformation
    ct = get_column_transformer_from_pipeline(pre_pipeline)
    raw_feature_names = ct.get_feature_names_out()
    if "drop_filtered_features" in pre_pipeline.named_steps:
        raw_feature_names = raw_feature_names[
            ~np.isin(
                raw_feature_names,
                np.array(
                    ["num__paintQuality%", "num__previousOwners", "num__hasDamage"]
                ),
            )
        ]
    feature_names = (
        column_cleaner(raw_feature_names) if clean_columns else raw_feature_names
    )

    # Training set
    X_train_preprocessed = pre_pipeline.transform(X_train)
    df_train_pre = pd.DataFrame(
        X_train_preprocessed, columns=feature_names, index=X_train.index
    )

    # Validation set
    if X_val is not None:
        X_val_preprocessed = pre_pipeline.transform(X_val)
        df_val_pre = pd.DataFrame(
            X_val_preprocessed, columns=feature_names, index=X_val.index
        )
    else:
        df_val_pre = None

    # Test set
    if X_test is not None:
        X_test_preprocessed = pre_pipeline.transform(X_test)
        df_test_pre = pd.DataFrame(
            X_test_preprocessed, columns=feature_names, index=X_test.index
        )
    else:
        df_test_pre = None

    # Export to CSV (optional)
    if csv:
        df_train_pre.to_csv(
            os.path.join(PRE_PIPELINE_DIR, "X_train_preprocessed.csv"),
            index=True,
        )
        if df_val_pre is not None:
            df_val_pre.to_csv(
                os.path.join(PRE_PIPELINE_DIR, "X_val_preprocessed.csv"),
                index=True,
            )
        if df_test_pre is not None:
            df_test_pre.to_csv(
                os.path.join(PRE_PIPELINE_DIR, "X_test_preprocessed.csv"),
                index=True,
            )
    return df_train_pre, df_val_pre, df_test_pre


def linear_evaluation_metrics(y: pd.Series, y_pred: np.ndarray, verbose=False):
    """
    Compute core performance metrics for regression models.

    This function calculates three standard indicators of regression performance:
    the coefficient of determination (R²), the mean absolute error (MAE),
    the mean absolute percentage error (MAPE) and the root mean squared error (RMSE).
    Together, these metrics quantify the predictive accuracy of the model and the magnitude
    of deviations between predicted and true target values.

    If 'verbose=True', a formatted summary of the computed metrics is printed to
    the console. When 'verbose=False' (default), the function simply returns the
    metric values without producing any output.

    Parameters
    ----------
    y : pd.Series
        The ground-truth target values.
    y_pred : np.ndarray
        The predicted target values generated by the regression model.
    verbose : bool, optional (default=False)
        If True, prints a formatted summary of the metrics. This option affects only
        the display behavior and does not alter the returned values.

    Returns
    -------
    (r2, mae, mape, rmse) : tuple of float
        r2
            Coefficient of determination
        mae
            Mean absolute error.
        mape
            Mean absolute percentage error
        rmse
            Root mean squared error
    """
    # Evaluation metrics
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    mape = mean_absolute_percentage_error(y, y_pred) * 100
    r_mse = root_mean_squared_error(y, y_pred)
    if verbose:
        print(f"R²: {r2:.3f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%, RMSE: {r_mse:.2f}")
        print("-" * 55)

    return r2, mae, mape, r_mse
