import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class BrandModelGroupImputer(BaseEstimator, TransformerMixin):
    """
    Imputes missing values using brand-model group statistics.

    This transformer performs group-wise imputation based on combinations of 'brand'
    and model identifiers. For each group, the transformer computes:

      - Numeric columns → imputed with the median of the corresponding group
      - Categorical columns → imputed with the mode of the corresponding group

    If a Brand–Model group contains no non-missing values for a given column,
    the transformer falls back to the global median (numeric) or global mode (categorical).

    Parameters
    ----------
    brand_col : str, default="Brand"
        Name of the column identifying the brand.
    model_col : str, default="model"
        Name of the column identifying the model.

    Attributes
    ----------
    group_medians_ : pd.DataFrame
        Median values of numeric columns aggregated by Brand–Model groups.
    group_modes_ : pd.DataFrame
        Mode values of categorical columns aggregated by Brand–Model groups.
    global_medians_ : pd.Series
        Column-wise medians used as fallbacks when group medians are missing.
    global_modes_ : pd.Series
        Column-wise modes used as fallbacks when group modes are missing.

    Returns
    -------
    X_imputed : pd.DataFrame
        A copy of the input with missing values imputed according to the
        Brand–Model hierarchical scheme.
    """

    def __init__(self, brand_model_col: str = "brand_model"):
        self.brand_model_col = brand_model_col

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()

        # Global medians and modes (fallbacks)
        self.global_medians_ = X.select_dtypes(include=[np.number]).median()
        if not X.select_dtypes(exclude=[np.number]).empty:
            self.global_modes_ = (
                X.select_dtypes(exclude=[np.number]).mode(dropna=True)
            ).iloc[0]
        else:
            self.global_modes_ = pd.Series(dtype=object)

        # Compute group-level medians and modes
        grouped = X.groupby([self.brand_model_col], dropna=False)

        self.group_medians_ = grouped.median(numeric_only=True)

        def safe_mode(s):
            m = s.mode(dropna=True)
            return m.iloc[0] if not m.empty else np.nan

        self.group_modes_ = grouped.agg(safe_mode)

        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        # Merge in group medians and modes using a vectorized join
        merged = X.merge(
            self.group_medians_.add_suffix("_median"),
            how="left",
            left_on=[self.brand_model_col],
            right_index=True,
        )
        merged = merged.merge(
            self.group_modes_.add_suffix("_mode"),
            how="left",
            left_on=[self.brand_model_col],
            right_index=True,
        )

        for col in X.columns:
            if col in [self.brand_model_col]:
                continue  # skip grouping columns

            if col in X.select_dtypes(include=[np.number]).columns:
                median_col = col + "_median"
                X[col] = X[col].fillna(merged[median_col])
                X[col] = X[col].fillna(self.global_medians_.get(col, np.nan))
            else:
                mode_col = col + "_mode"
                X[col] = X[col].fillna(merged[mode_col])
                X[col] = X[col].fillna(self.global_modes_.get(col, "Unknown"))

        return X

    def set_output(self, transform=None):
        """
        Compatibility hook for sklearn's set_output.
        This transformer always returns a pandas DataFrame, so we ignore `transform`.
        This transformer always returns a pandas DataFrame, so the `transform`
        argument is ignored. The method exists solely to make the class
        compatible with Pipeline.set_output
        """
        return self


class FilterFeatureDropper(BaseEstimator, TransformerMixin):
    """
    Transformer that drops specified features from an array after preprocessing.

    This custom transformer is inteded to be used as a pipeline step after the preprocessing
    pipeline has produced its feature matrix. It identifies the indices of the columns
    corresponding to features previously flagged for removal during filter-based
    feature selection, and deletes those columns from the array.

    Parameters
    ----------
    columns_to_drop: list of str.
         Column names to remove.

    Returns
        -------
        np.ndarray of str
            Remaining feature names after removal.

    Notes
    -----
    This transformer must be placed *after* the preprocessing pipeline
    that generates the feature matrix. The preprocessing pipeline must
    support 'get_feature_names_out()'.
    """

    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        self.get_feature_names_in_ = np.array(X.columns)
        return self

    def transform(self, X):
        # Drop columns if they exist
        X_dropped = X.drop(
            columns=[col for col in self.columns_to_drop if col in X.columns]
        )
        # Converts DataFrame back into array
        return X_dropped.to_numpy()

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.feature_names_in_

        # Return remaining feature names
        return np.array([f for f in input_features if f not in self.columns_to_drop])
