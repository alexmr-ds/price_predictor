import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class BrandModelGroupImputer(BaseEstimator, TransformerMixin):
    """
    A scikit-learn–compatible transformer that imputes missing values using
    brand–model group statistics.

    For each feature, missing values are imputed hierarchically:
    1. Using the median (numeric) or mode (categorical) within the
       corresponding (brand, model) group.
    2. Falling back to the global median or mode computed across the
       entire dataset when group-level statistics are unavailable.

    This approach preserves brand- and model-specific characteristics while
    remaining robust to sparse or unseen groups at inference time.

    Parameters
    ----------
    brand_col : str, default="Brand"
        Name of the column identifying the vehicle brand.

    model_col : str, default="model"
        Name of the column identifying the vehicle model.
    """
    def __init__(self, brand_col="Brand", model_col="model"):
        self.brand_col = brand_col
        self.model_col = model_col

    def fit(self, X, y=None):
        """
        Fit the imputer by computing global and group-level statistics.

        Computes:
        - Global medians for numeric features
        - Global modes for categorical features
        - Per-(brand, model) medians for numeric features
        - Per-(brand, model) modes for categorical features

        The computed statistics are stored and reused during transformation.

        Parameters
        ----------
        X : pandas.DataFrame or array-like
            Input feature matrix.

        y : array-like, optional
            Target values (ignored).

        Returns
        -------
        self : BrandModelGroupImputer
            Fitted transformer.
        """
        X = pd.DataFrame(X).copy()
        # compute group medians & modes
        self.global_medians_ = X.select_dtypes(include=[np.number]).median()
        if not X.select_dtypes(exclude=[np.number]).empty:
            self.global_modes_ = X.select_dtypes(exclude=[np.number]).mode(dropna=True).iloc[0]
        else:
            self.global_modes_ = pd.Series(dtype=object)

        grouped = X.groupby([self.brand_col, self.model_col], dropna=False)
        self.group_medians_ = grouped.median(numeric_only=True)
        def safe_mode(s):
            m = s.mode(dropna=True)
            return m.iloc[0] if not m.empty else np.nan
        self.group_modes_ = grouped.agg(safe_mode)
        return self

    def transform(self, X):
        """
        Impute missing values using brand–model group statistics.

        For each feature (excluding the brand and model identifier columns):
        - Numeric features are imputed with the group median, falling back
          to the global median if necessary.
        - Categorical features are imputed with the group mode, falling back
          to the global mode if necessary.

        This ensures consistent imputation for both training and unseen
        data while avoiding target leakage.

        Parameters
        ----------
        X : pandas.DataFrame or array-like
            Input feature matrix with possible missing values.

        Returns
        -------
        pandas.DataFrame
            DataFrame with missing values imputed.
        """
        X = pd.DataFrame(X).copy()
        merged = X.merge(self.group_medians_.add_suffix("_median"),
                         how="left", left_on=[self.brand_col, self.model_col], right_index=True)
        merged = merged.merge(self.group_modes_.add_suffix("_mode"),
                              how="left", left_on=[self.brand_col, self.model_col], right_index=True)

        for col in X.columns:
            if col in [self.brand_col, self.model_col]:
                continue
            if col in X.select_dtypes(include=[np.number]).columns:
                median_col = col + "_median"
                X[col] = X[col].fillna(merged.get(median_col))
                X[col] = X[col].fillna(self.global_medians_.get(col, np.nan))
            else:
                mode_col = col + "_mode"
                X[col] = X[col].fillna(merged.get(mode_col))
                X[col] = X[col].fillna(self.global_modes_.get(col, "Unknown"))
        return X