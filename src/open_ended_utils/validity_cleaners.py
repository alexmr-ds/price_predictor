import numpy as np
import pandas as pd
import difflib
from sklearn.base import BaseEstimator, TransformerMixin

    

class NumericValidityCleaner(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        ranges={"year": (0, 2020), "mileage": (0, None), "engineSize": (0, None), "mpg": (0, 300), "tax": (0, None), "previousOwners": (0, None)},
        policy="Wipe",
    ):
        self.policy = policy
        self.ranges = ranges


    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        for key, rang in self.ranges.items():
            str_aux = key + "_invalid"
            cond_lower = (X[key] < rang[0]) if rang[0] is not None else False
            cond_upper = (X[key] > rang[1]) if rang[1] is not None else False
            X[str_aux] = cond_lower | cond_upper
            if self.policy == "wipe":
                    X[key] = X[key].mask(X[str_aux], other=np.nan)
            elif self.policy == "abs_clip":
                    X[key] = X[key].mask(X[str_aux], other=X[key].abs().clip(lower=rang[0], upper=rang[1]))
            elif self.policy == "clip":
                    X[key] = X[key].mask(X[str_aux], other=X[key].clip(lower=rang[0], upper=rang[1]))
        return X



class CategoricalValidityCleaner(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        valids: dict[str, list],
        min_similarity=0.9,
        unknown_label="unknown",
        add_invalid_flag=True,
        normalize_strings=True,
    ):
        self.valids = valids
        self.min_similarity = min_similarity
        self.unknown_label = unknown_label
        self.add_invalid_flag = add_invalid_flag
        self.normalize_strings = normalize_strings

    def fit(self, X, y=None):
        return self

    def _normalize(self, s):
        if pd.isna(s):
            return s
        s = str(s).strip().lower()
        return s

    def _best_fuzzy_match(self, value, candidates):
        """
        Returns (best_match, similarity_score) or (None, 0.0)
        """
        matches = difflib.get_close_matches(
            value,
            candidates,
            n=1,
            cutoff=self.min_similarity
        )
        if matches:
            best = matches[0]
            score = difflib.SequenceMatcher(None, value, best).ratio()
            return best, score
        return None, 0.0

    def transform(self, X):
        X = X.copy()
        invalid_flags = []

        for col, valid_values in self.valids.items():
            if col not in X.columns:
                continue

            # Normalize canonical values once
            if self.normalize_strings:
                canonicals = [self._normalize(v) for v in valid_values]
            else:
                canonicals = list(valid_values)

            invalid_mask = pd.Series(False, index=X.index)

            for idx, val in enumerate(X[col]):
                if pd.isna(val):
                    continue

                val_norm = self._normalize(val) if self.normalize_strings else val
                match, score = self._best_fuzzy_match(val_norm, canonicals)

                if match is not None:
                    X.at[idx, col] = match
                else:
                    X.at[idx, col] = self.unknown_label
                    invalid_mask.at[idx] = True

            invalid_flags.append(invalid_mask)

        if self.add_invalid_flag and invalid_flags:
            X["entry_was_invalid_cat"] = np.logical_or.reduce(invalid_flags)

        return X


class FullValidityCleaner(BaseEstimator, TransformerMixin):
    """
    A scikit-learn–compatible transformer that enforces value validity for
    numeric and categorical features in tabular data.

    This transformer:
    - Validates numeric columns against predefined ranges
    - Applies configurable correction policies for invalid numeric values
    - Normalizes and fuzzy-matches categorical values against allowed sets
    - Replaces invalid categorical entries with a fallback token
    - Optionally adds row-level invalidity flags for numeric and categorical data

    The transformer is designed to be used safely inside a scikit-learn
    Pipeline, ensuring consistent preprocessing between training and inference.

    Parameters
    ----------
    valids : dict[str, tuple | list]
        Mapping of categorical column names to allowed canonical values.
        Values not matching these sets (optionally via fuzzy matching) are
        replaced according to `cat_replace_with`.

    ranges : dict[str, tuple[int | None, int | None]]
        Mapping of numeric column names to (min, max) validity bounds.
        Use None for unbounded sides of the range.

    numeric_policy : {"wipe", "clip", "abs_clip"}, default="wipe"
        Strategy for handling invalid numeric values:
        - "wipe": replace invalid values with NaN
        - "clip": clip values to the specified bounds
        - "abs_clip": apply absolute value, then clip to bounds

    min_similarity : float, default=0.9
        Minimum similarity threshold for fuzzy matching categorical values.

    cat_replace_with : str, default="unknown"
        Replacement value for categorical entries that cannot be matched
        to any valid category.

    add_invalid_flag : bool, default=True
        Whether to add boolean indicator columns capturing whether any
        categorical or numeric value in a row was invalid.

    normalize_strings : bool, default=True
        Whether to normalize categorical strings (lowercase, strip whitespace)
        before matching.
    """


    def __init__(
        self,
        valids: dict[str, tuple|list],
        ranges: dict[str, tuple[int | None,int | None]],
        numeric_policy="wipe",
        min_similarity=0.9,
        cat_replace_with="unknown",
        add_invalid_flag=True,
        normalize_strings=True,
    ):
        self.valids = valids
        self.ranges = ranges
        self.numeric_policy = numeric_policy
        self.min_similarity = min_similarity
        self.cat_replace_with = cat_replace_with
        self.add_invalid_flag = add_invalid_flag
        self.normalize_strings = normalize_strings

    def fit(self, X, y=None):
        """
        Fit the transformer.

        This transformer is stateless and does not learn parameters from data.
        The method exists to satisfy the scikit-learn estimator interface.

        Parameters
        ----------
        X : pandas.DataFrame
            Input feature matrix.

        y : array-like, optional
            Target values (ignored).

        Returns
        -------
        self : FullValidityCleaner
            Fitted transformer (self).
        """
        return self
    
    def _normalize(self, s):
        """
        Normalize a string value for comparison.

        Applies stripping and lowercasing. NaN values are returned unchanged.

        Parameters
        ----------
        s : object
            Input value.

        Returns
        -------
        object
            Normalized string or NaN.
        """
        if pd.isna(s):
            return s
        s = str(s).strip().lower()
        return s


    def _best_fuzzy_match(self, value, candidates):
        """
        Find the closest fuzzy match for a value among a set of candidates.

        Parameters
        ----------
        value : str
            Normalized input value.

        candidates : list[str]
            List of valid normalized candidate strings.

        Returns
        -------
        tuple[str | None, float]
            Best matching candidate and its similarity score, or (None, 0.0)
            if no candidate meets the similarity threshold.
        """
        matches = difflib.get_close_matches(
            value,
            candidates,
            n=1,
            cutoff=self.min_similarity
        )
        if matches:
            best = matches[0]
            score = difflib.SequenceMatcher(None, value, best).ratio()
            return best, score
        return None, 0.0

    def transform(self, X):
        """
        Validate and clean numeric and categorical features.

        Numeric columns are checked against predefined ranges and corrected
        according to `numeric_policy`. Categorical columns are normalized and
        fuzzy-matched against valid value sets, with unmatched entries replaced.

        Optionally, row-level boolean indicators are added to signal whether
        any numeric or categorical value in a row was invalid.

        Parameters
        ----------
        X : pandas.DataFrame
            Input feature matrix.

        Returns
        -------
        pandas.DataFrame
            Cleaned feature matrix with corrected values and optional
            invalidity indicator columns.
        """
        X = X.copy()

        for key, rang in self.ranges.items():
            str_aux = key + "_invalid"
            cond_lower = (X[key] < rang[0]) if rang[0] is not None else False
            cond_upper = (X[key] > rang[1]) if rang[1] is not None else False
            X[str_aux] = cond_lower | cond_upper
            if self.numeric_policy == "wipe":
                    X[key] = X[key].mask(X[str_aux], other=np.nan)
                    
            elif self.numeric_policy == "abs_clip":
                    X[key] = X[key].mask(X[str_aux], other=X[key].abs().clip(lower=rang[0], upper=rang[1]))
            elif self.numeric_policy == "clip":
                    X[key] = X[key].mask(X[str_aux], other=X[key].clip(lower=rang[0], upper=rang[1]))
            X[key] = pd.to_numeric(X[key], errors="coerce")

        invalid_flags = []

        for col, valid_values in self.valids.items():
            if col not in X.columns:
                continue

            canonicals = (
                [self._normalize(v) for v in valid_values]
                if self.normalize_strings else list(valid_values)
            )

            invalid_mask = pd.Series(False, index=X.index)

            for idx, val in X[col].items():
                if pd.isna(val):
                    continue

                val_norm = self._normalize(val) if self.normalize_strings else val
                match, _ = self._best_fuzzy_match(val_norm, canonicals)

                if match is not None:
                    X.at[idx, col] = match
                else:
                    X.at[idx, col] = self.cat_replace_with
                    invalid_mask.at[idx] = True

            invalid_flags.append(invalid_mask)

        if self.add_invalid_flag and invalid_flags:
            X["entry_was_invalid_cat"] = pd.concat(invalid_flags, axis=1).any(axis=1)

        # --- numeric invalid aggregation ---
        invalid_cols = X.columns[X.columns.str.contains("_invalid")].to_list()
        if invalid_cols:
            X["entry_was_invalid"] = X[invalid_cols].any(axis=1)
        else:
            X["entry_was_invalid"] = False

        # drop helper columns
        X.drop(columns=invalid_cols, inplace=True)


        return X
