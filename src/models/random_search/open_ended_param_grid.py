"""
Hyperparameter search space for the full pipeline implemented at the open ended section.

The search space is defined using a regime-based strategy to avoid
pathological or unstable parameter combinations. Each regime represents
a coherent trade-off between model capacity, regularization strength,
and boosting dynamics.

This configuration is designed for extensive continuous random search
targeting approximately 3 hours of runtime.

Randomized hyperparameter search configuration:
- Search method: RandomizedSearchCV
- Number of sampled configurations: 100
- Cross-validation: 5-fold K-Fold (with shuffling)
- Scoring metric: Mean Absolute Error (MAE)
- Parallelization: n_jobs = -1
- Random seed: random_state = 0

Early stopping is enabled in all regimes to improve computational
efficiency and prevent overfitting.
"""

# --- Extensive Hyperparameter search space (regime-based to avoid bad combos) --- #

distribution = {
    "cleaner__numeric_policy": ["wipe", "abs_clip", "clip"],
    "cleaner__min_similarity": [0.85, 0.9, 0.95],
    "cleaner__cat_replace_with": ["unknown", np.nan],
    "imputer": [
        "passthrough",
        ml_utils.BrandModelGroupImputer(brand_col="Brand", model_col="model"),
    ],
    "model__early_stopping": [True],
    "model__max_depth": [None] + list(range(5, 80, 5)),
    "model__random_state": randint(1, 100),
    "model__learning_rate": loguniform(0.01, 0.2),
    "model__max_iter": randint(100, 20000),
    "model__max_leaf_nodes": randint(32, 5000),
    "model__min_samples_leaf": randint(5, 500),
    "model__l2_regularization": loguniform(1e-3, 0.5),
}
