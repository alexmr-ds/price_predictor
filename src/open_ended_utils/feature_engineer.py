import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    A scikit-learn–compatible transformer that performs feature engineering
    for used-car price regression tasks.

    This transformer creates derived features that capture:
    - vehicle age and usage intensity
    - nonlinear mileage effects
    - ownership structure
    - engine efficiency and taxation interactions
    - row-level data quality signals (missing values and outliers)
    - combined categorical identifiers for brand and model

    The transformer is stateless and safe for use inside a scikit-learn
    Pipeline, ensuring consistent feature generation across training
    and inference.
    
    Parameters
    ----------
    reference_year : int, default=2020
        Reference year used to compute vehicle age from the manufacturing
        year. Should be fixed to avoid temporal leakage.
    """

    def __init__(self, reference_year = 2020):
        self.reference_year = reference_year

    def fit(self, X, y=None):
        """
        Fit the transformer.

        This transformer does not learn parameters from data.
        The method exists to satisfy the scikit-learn estimator interface.

        Parameters
        ----------
        X : pandas.DataFrame
            Input feature matrix.

        y : array-like, optional
            Target values (ignored).

        Returns
        -------
        self : FeatureEngineer
            Fitted transformer (self).
        """
        return self
    
    def add_missing_and_outlier_flags(
        self,
        df: pd.DataFrame,
        iqr_multiplier: float = 1.5
    ) -> pd.DataFrame:
        """
        Adds two boolean columns to the dataframe:
        - 'has_missing': True if the row contains any missing value
        - 'has_outliers': True if the row contains any numeric outlier (IQR-based)

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with mixed feature types
        iqr_multiplier : float, default=1.5
            Multiplier for IQR-based outlier detection

        Returns
        -------
        pd.DataFrame
            Copy of df with two additional boolean columns
        """

        df_out = df.copy()

        # -----------------------------
        # Missing value flag (row-wise)
        # -----------------------------
        df_out["has_missing"] = df_out.isna().any(axis=1)

        # -----------------------------------
        # Outlier flag (numeric columns only)
        # -----------------------------------
        numeric_cols = df_out.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            # No numeric features → no outliers possible
            df_out["has_outliers"] = False
            return df_out

        Q1 = df_out[numeric_cols].quantile(0.25)
        Q3 = df_out[numeric_cols].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR

        outlier_mask = (
            (df_out[numeric_cols] < lower_bound) |
            (df_out[numeric_cols] > upper_bound)
        )

        # Row-wise: any numeric feature is an outlier
        df_out["has_outliers"] = outlier_mask.any(axis=1)

        return df_out

    def transform(self, X):
        """
        Generate engineered features from raw vehicle attributes.

        The following features are created:
        - 'brand_model': concatenation of brand and model identifiers
        - 'age': vehicle age relative to `reference_year`
        - 'miles_per_year': mileage normalized by vehicle age
        - 'log_mileage': log-transformed mileage
        - 'multiple_owners': indicator for more than one previous owner
        - 'engine_efficiency': fuel efficiency relative to engine size
        - 'tax_engine_size': tax burden relative to engine size
        - 'has_missing': row-level missing value indicator
        - 'has_outliers': row-level numeric outlier indicator

        Parameters
        ----------
        X : pandas.DataFrame
            Input feature matrix.

        Returns
        -------
        pandas.DataFrame
            Feature-engineered dataframe suitable for downstream modeling.
        """
        X = X.copy()
        
        X["brand_model"] = (
            X["Brand"].fillna("unknown")
            + "_"
            + X["model"].fillna("unknown")
        )
        X["age"] = (self.reference_year - X["year"])
        X["miles_per_year"] = X['mileage'].div(X['age'] + 1).replace([np.inf, -np.inf], 0)
        X["log_mileage"] = np.log1p(X["mileage"])
        X["multiple_owners"] = (X["previousOwners"] > 1)
        X["engine_efficiency"] = 0.264172052 * X['mpg'].div(X['engineSize']).replace([np.inf, -np.inf], 0)
        X["tax_engine_size"] = X['tax'].div(X['engineSize']).replace([np.inf, -np.inf], 0)
        X = self.add_missing_and_outlier_flags(X)
        
        return X