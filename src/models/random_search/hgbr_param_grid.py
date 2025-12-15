"""
Hyperparameter search spaces for HistGradientBoostingRegressor
Regime-based to avoid pathological parameter combinations.
"""

# --- Hyperparameter search space (regime-based to avoid bad combos) --- #
param_distributions = [
    # Regime A
    {
        # --- Boosting dynamics ---
        "regressor__regressor__learning_rate": [0.03, 0.05, 0.08],
        "regressor__regressor__max_iter": [600, 1000, 2000],
        # --- Tree capacity / interaction control ---
        "regressor__regressor__max_depth": [3, 5, 8],
        "regressor__regressor__max_leaf_nodes": [31, 63, 127, 255],
        # --- Regularization (biggest lever for MAE stability) ---
        "regressor__regressor__min_samples_leaf": [20, 40, 80, 120],
        "regressor__regressor__l2_regularization": [1e-3, 1e-2, 1e-1, 1.0],
        # --- Histogram resolution ---
        "regressor__regressor__max_bins": [128, 255],
        # --- Feature subsampling (variance control) ---
        "regressor__regressor__max_features": [0.6, 0.8, 1.0],
        # --- Early stopping controls ---
        "regressor__regressor__validation_fraction": [0.1, 0.15],
        "regressor__regressor__n_iter_no_change": [20, 30],
        "regressor__regressor__early_stopping": [True],
    },
    # Regime B
    {
        "regressor__regressor__learning_rate": [0.02, 0.03, 0.05],
        "regressor__regressor__max_iter": [1000, 2000],
        "regressor__regressor__max_depth": [None, 8, 12],
        "regressor__regressor__max_leaf_nodes": [127, 255, 511],
        "regressor__regressor__min_samples_leaf": [10, 20, 40],
        "regressor__regressor__l2_regularization": [0.0, 1e-4, 1e-3, 1e-2],
        "regressor__regressor__max_bins": [128, 192, 255],
        "regressor__regressor__max_features": [0.6, 0.8, 1.0],
        "regressor__regressor__validation_fraction": [0.1, 0.15, 0.2],
        "regressor__regressor__n_iter_no_change": [10, 20],
        "regressor__regressor__early_stopping": [True],
    },
    # Regime C
    {
        "regressor__regressor__learning_rate": [0.05, 0.08, 0.1],
        "regressor__regressor__max_iter": [300, 600, 1000],
        "regressor__regressor__max_depth": [3, 5, 8],
        "regressor__regressor__max_leaf_nodes": [31, 63, 127],
        "regressor__regressor__min_samples_leaf": [20, 40, 80],
        "regressor__regressor__l2_regularization": [1e-3, 1e-2, 1e-1],
        "regressor__regressor__max_bins": [128, 255],
        "regressor__regressor__max_features": [0.6, 0.8, 1.0],
        "regressor__regressor__validation_fraction": [0.1],
        "regressor__regressor__n_iter_no_change": [20],
        "regressor__regressor__early_stopping": [True],
    },
]
