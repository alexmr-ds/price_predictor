"""
Extensive hyperparameter search space for HistGradientBoostingRegressor.

The search space is defined using a regime-based strategy to avoid
pathological or unstable parameter combinations. Each regime represents
a coherent trade-off between model capacity, regularization strength,
and boosting dynamics.

This configuration is designed for extensive continuous random search
targeting approximately 3 hours of runtime.

Randomized hyperparameter search configuration:
- Search method: RandomizedSearchCV
- Number of sampled configurations: 1000 (estimated for 5-hour runtime
- Cross-validation: 10-fold K-Fold (with shuffling)
- Scoring metric: Mean Absolute Error (MAE)
- Parallelization: n_jobs = -1
- Random seed: random_state = 0

Early stopping is enabled in all regimes to improve computational
efficiency and prevent overfitting.
"""

# --- Extensive Hyperparameter search space (regime-based to avoid bad combos) --- #
# Using scipy.stats distributions for continuous random sampling
from scipy.stats import uniform, randint

param_distributions = [
    # Regime A: Conservative learning with strong regularization
    {
        # --- Boosting dynamics ---
        "regressor__regressor__learning_rate": uniform(0.01, 0.05),  # 0.01 to 0.06
        "regressor__regressor__max_iter": randint(500, 3000),  # 500 to 3000
        # --- Tree capacity / interaction control ---
        "regressor__regressor__max_depth": [3, 4, 5, 6, 7, 8, None],
        "regressor__regressor__max_leaf_nodes": [31, 63, 127, 255, 511],
        # --- Regularization (biggest lever for MAE stability) ---
        "regressor__regressor__min_samples_leaf": randint(20, 200),  # 20 to 200
        "regressor__regressor__l2_regularization": uniform(1e-4, 1.0),  # 1e-4 to 1.0
        # --- Histogram resolution ---
        "regressor__regressor__max_bins": [64, 128, 192, 255],
        # --- Feature subsampling (variance control) ---
        "regressor__regressor__max_features": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        # --- Early stopping controls ---
        "regressor__regressor__validation_fraction": [0.08, 0.1, 0.12, 0.15, 0.18],
        "regressor__regressor__n_iter_no_change": [10, 15, 20, 25, 30, 40],
        "regressor__regressor__early_stopping": [True],
        "regressor__regressor__loss": ["absolute_error"],
    },
    # Regime B: Moderate learning with balanced regularization
    {
        "regressor__regressor__learning_rate": uniform(0.02, 0.08),  # 0.02 to 0.10
        "regressor__regressor__max_iter": randint(800, 2500),  # 800 to 2500
        "regressor__regressor__max_depth": [None, 5, 6, 8, 10, 12, 15],
        "regressor__regressor__max_leaf_nodes": [63, 127, 255, 511, 1023],
        "regressor__regressor__min_samples_leaf": randint(10, 150),  # 10 to 150
        "regressor__regressor__l2_regularization": uniform(0.0, 0.5),  # 0.0 to 0.5
        "regressor__regressor__max_bins": [96, 128, 160, 192, 224, 255],
        "regressor__regressor__max_features": [0.6, 0.7, 0.8, 0.9, 1.0],
        "regressor__regressor__validation_fraction": [0.1, 0.12, 0.15, 0.18, 0.2],
        "regressor__regressor__n_iter_no_change": [10, 15, 20, 25, 30],
        "regressor__regressor__early_stopping": [True],
        "regressor__regressor__loss": ["absolute_error"],
    },
    # Regime C: Aggressive learning with lighter regularization
    {
        "regressor__regressor__learning_rate": uniform(0.05, 0.1),  # 0.05 to 0.15
        "regressor__regressor__max_iter": randint(300, 1500),  # 300 to 1500
        "regressor__regressor__max_depth": [3, 4, 5, 6, 8, 10, None],
        "regressor__regressor__max_leaf_nodes": [31, 63, 127, 255, 511],
        "regressor__regressor__min_samples_leaf": randint(15, 120),  # 15 to 120
        "regressor__regressor__l2_regularization": uniform(1e-4, 0.2),  # 1e-4 to 0.2
        "regressor__regressor__max_bins": [128, 160, 192, 224, 255],
        "regressor__regressor__max_features": [0.7, 0.8, 0.9, 1.0],
        "regressor__regressor__validation_fraction": [0.1, 0.12, 0.15],
        "regressor__regressor__n_iter_no_change": [15, 20, 25, 30],
        "regressor__regressor__early_stopping": [True],
        "regressor__regressor__loss": ["absolute_error"],
    },
    # Regime D: Very conservative with deep regularization
    {
        "regressor__regressor__learning_rate": uniform(0.005, 0.03),  # 0.005 to 0.035
        "regressor__regressor__max_iter": randint(1000, 4000),  # 1000 to 4000
        "regressor__regressor__max_depth": [3, 4, 5, 6, None],
        "regressor__regressor__max_leaf_nodes": [31, 63, 127, 255],
        "regressor__regressor__min_samples_leaf": randint(30, 250),  # 30 to 250
        "regressor__regressor__l2_regularization": uniform(1e-3, 2.0),  # 1e-3 to 2.0
        "regressor__regressor__max_bins": [128, 192, 255],
        "regressor__regressor__max_features": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "regressor__regressor__validation_fraction": [0.1, 0.15, 0.2],
        "regressor__regressor__n_iter_no_change": [20, 30, 40, 50],
        "regressor__regressor__early_stopping": [True],
        "regressor__regressor__loss": ["absolute_error"],
    },
    # Regime E: High capacity trees with feature subsampling
    {
        "regressor__regressor__learning_rate": uniform(0.03, 0.07),  # 0.03 to 0.10
        "regressor__regressor__max_iter": randint(600, 2000),  # 600 to 2000
        "regressor__regressor__max_depth": [None, 8, 10, 12, 15, 20],
        "regressor__regressor__max_leaf_nodes": [255, 511, 1023, 2047],
        "regressor__regressor__min_samples_leaf": randint(5, 80),  # 5 to 80
        "regressor__regressor__l2_regularization": uniform(0.0, 0.1),  # 0.0 to 0.1
        "regressor__regressor__max_bins": [192, 224, 255],
        "regressor__regressor__max_features": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "regressor__regressor__validation_fraction": [0.1, 0.12, 0.15, 0.18],
        "regressor__regressor__n_iter_no_change": [10, 15, 20, 25],
        "regressor__regressor__early_stopping": [True],
        "regressor__regressor__loss": ["absolute_error"],
    },
    # Regime F: Fine-grained histogram resolution focus
    {
        "regressor__regressor__learning_rate": uniform(0.04, 0.08),  # 0.04 to 0.12
        "regressor__regressor__max_iter": randint(400, 1800),  # 400 to 1800
        "regressor__regressor__max_depth": [4, 5, 6, 8, None],
        "regressor__regressor__max_leaf_nodes": [63, 127, 255, 511],
        "regressor__regressor__min_samples_leaf": randint(20, 100),  # 20 to 100
        "regressor__regressor__l2_regularization": uniform(1e-3, 0.5),  # 1e-3 to 0.5
        "regressor__regressor__max_bins": [
            160,
            192,
            224,
            255,
        ],  # Focus on higher resolution
        "regressor__regressor__max_features": [0.7, 0.8, 0.9, 1.0],
        "regressor__regressor__validation_fraction": [0.1, 0.15],
        "regressor__regressor__n_iter_no_change": [20, 25, 30],
        "regressor__regressor__early_stopping": [True],
        "regressor__regressor__loss": ["absolute_error"],
    },
    # Regime G: Balanced exploration across all dimensions
    {
        "regressor__regressor__learning_rate": uniform(0.02, 0.1),  # 0.02 to 0.12
        "regressor__regressor__max_iter": randint(500, 2500),  # 500 to 2500
        "regressor__regressor__max_depth": [None, 4, 5, 6, 8, 10, 12],
        "regressor__regressor__max_leaf_nodes": [63, 127, 255, 511, 1023],
        "regressor__regressor__min_samples_leaf": randint(10, 180),  # 10 to 180
        "regressor__regressor__l2_regularization": uniform(0.0, 1.0),  # 0.0 to 1.0
        "regressor__regressor__max_bins": [128, 160, 192, 224, 255],
        "regressor__regressor__max_features": [0.6, 0.7, 0.8, 0.9, 1.0],
        "regressor__regressor__validation_fraction": [0.08, 0.1, 0.12, 0.15, 0.18, 0.2],
        "regressor__regressor__n_iter_no_change": [10, 15, 20, 25, 30, 40],
        "regressor__regressor__early_stopping": [True],
        "regressor__regressor__loss": ["absolute_error"],
    },
    # Regime H: Extreme regularization for stability
    {
        "regressor__regressor__learning_rate": uniform(0.01, 0.04),  # 0.01 to 0.05
        "regressor__regressor__max_iter": randint(1500, 4000),  # 1500 to 4000
        "regressor__regressor__max_depth": [3, 4, 5, 6, None],
        "regressor__regressor__max_leaf_nodes": [31, 63, 127, 255],
        "regressor__regressor__min_samples_leaf": randint(50, 300),  # 50 to 300
        "regressor__regressor__l2_regularization": uniform(0.1, 3.0),  # 0.1 to 3.0
        "regressor__regressor__max_bins": [128, 192, 255],
        "regressor__regressor__max_features": [0.6, 0.7, 0.8, 0.9, 1.0],
        "regressor__regressor__validation_fraction": [0.12, 0.15, 0.18, 0.2],
        "regressor__regressor__n_iter_no_change": [25, 30, 40, 50],
        "regressor__regressor__early_stopping": [True],
        "regressor__regressor__loss": ["absolute_error"],
    },
]
