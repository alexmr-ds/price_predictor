"""
Centralized path configuration for the project.

This module defines all directory paths used throughout the project,
including raw, cleaned, processed data locations, preprocessing artifacts,
model outputs, and notebooks. Paths are constructed dynamically from the
project root to ensure portability across environments and operating systems.

The project root is inferred relative to the location of this file, allowing
all downstream modules to import consistent and absolute paths without
hardcoding directory structures.
"""

from pathlib import Path
from loguru import logger

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[2]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

# Data directories
DATA_DIR = PROJ_ROOT / "data"
IMAGES_DIR = PROJ_ROOT / "images"
RAW_DATA_DIR = DATA_DIR / "raw"
CLEANED_DATA_DIR = DATA_DIR / "cleaned"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PRE_PIPELINE_DIR = PROCESSED_DATA_DIR / "pre_pipeline"
FULL_PREPROCESSOR_DIR = PROCESSED_DATA_DIR / "full_preprocessor"
TEST_PREDICTIONS_DIR = PROCESSED_DATA_DIR / "test_predictions"

# Notebook directories
NOTEBOOKS_DIR = PROJ_ROOT / "notebooks"
DATA_CLEANING_DIR = PROJ_ROOT / "data_cleaning"
MODELS_DIR = NOTEBOOKS_DIR / "models"
