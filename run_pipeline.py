#!/usr/bin/env python
"""
Script to run all notebooks in the correct order for the car price prediction pipeline.

Execution order:
1. data_cleaning.ipynb - Cleans the raw data
2. LR.ipynb - Creates preprocessing pipeline and baseline model
3. TBEM.ipynb - Trains tree-based ensemble models
4. BR.ipynb - Trains bagging regressor
5. Ensemble.ipynb - Creates final stacking ensemble

Usage:
    python run_pipeline.py                    # Interactive mode (asks for confirmation)
    python run_pipeline.py --yes              # Non-interactive mode (no confirmation)
    python run_pipeline.py -y                 # Short form for non-interactive mode
    python run_pipeline.py --setup-env        # Set up conda environment from environment.yml
    python run_pipeline.py --skip-env          # Skip environment setup, use current Python
    python run_pipeline.py --setup-env --yes    # Set up environment in non-interactive mode
"""

import sys
import subprocess
from pathlib import Path
import time
import argparse
import tempfile
import shutil
import threading
import itertools
import os

# Get project root directory
PROJ_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJ_ROOT / "environment.yml"
ENV_NAME = "car_price_env"

# Define notebooks in execution order
NOTEBOOKS = [
    {
        "name": "data_cleaning.ipynb",
        "path": PROJ_ROOT / "notebooks" / "data_cleaning" / "data_cleaning.ipynb",
        "description": "Data Cleaning",
    },
    {
        "name": "LR.ipynb",
        "path": PROJ_ROOT / "notebooks" / "models" / "LR.ipynb",
        "description": "Linear Regression (Baseline Model)",
    },
    {
        "name": "TBEM.ipynb",
        "path": PROJ_ROOT / "notebooks" / "models" / "TBEM.ipynb",
        "description": "Tree-Based Ensemble Methods",
    },
    {
        "name": "BR.ipynb",
        "path": PROJ_ROOT / "notebooks" / "models" / "BR.ipynb",
        "description": "Bagging Regressor",
    },
    {
        "name": "Ensemble.ipynb",
        "path": PROJ_ROOT / "notebooks" / "models" / "Ensemble.ipynb",
        "description": "Stacking Ensemble",
    },
]


def check_conda_available():
    """Check if conda is available in the system."""
    try:
        result = subprocess.run(
            ["conda", "--version"], capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def check_env_exists(env_name):
    """Check if a conda environment exists."""
    try:
        result = subprocess.run(
            ["conda", "env", "list", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            import json

            envs_data = json.loads(result.stdout)
            envs = envs_data.get("envs", [])
            # Check if any environment path contains the env name
            for env_path in envs:
                if env_name in env_path:
                    return True
        return False
    except Exception:
        return False


def check_current_env():
    """Check if we're currently in the expected conda environment."""
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    if conda_env == ENV_NAME:
        return True
    # Also check if CONDA_PREFIX contains the env name
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if ENV_NAME in conda_prefix:
        return True
    return False


def setup_conda_environment(env_name, env_file, skip_prompt=False):
    """
    Set up conda environment and provide activation instructions.

    Args:
        env_name: Name of the conda environment
        env_file: Path to environment.yml file
        skip_prompt: If True, skip user prompts

    Returns:
        True if environment setup is complete or user chose to skip, False if user needs to activate
    """
    if not check_conda_available():
        print("WARNING: Conda is not available in your system.")
        print("  Please install conda/miniconda or use your own Python environment.")
        print("  Continuing with current Python environment...\n")
        return True  # Continue anyway

    if not env_file.exists():
        print(f"WARNING: Environment file not found at {env_file}")
        print("  Continuing with current Python environment...\n")
        return True  # Continue anyway

    # Check if already in the right environment
    if check_current_env():
        print(f"Already in conda environment '{env_name}'")
        print("  Ready to proceed!\n")
        return True

    env_exists = check_env_exists(env_name)

    if env_exists:
        print(f"✓ Conda environment '{env_name}' exists.")
        print(f"  Current environment: {os.environ.get('CONDA_DEFAULT_ENV', 'None')}")
        if not skip_prompt:
            response = (
                input(f"  Do you want to use it? (yes/no) [yes]: ").strip().lower()
            )
            if response and response not in ["yes", "y"]:
                print("  Using current Python environment.\n")
                return True
    else:
        print(f"ℹ Conda environment '{env_name}' does not exist.")
        if not skip_prompt:
            response = (
                input(
                    f"  Do you want to create it from {env_file.name}? (yes/no) [yes]: "
                )
                .strip()
                .lower()
            )
            if response and response not in ["yes", "y"]:
                print("  Using current Python environment.\n")
                return True

        # Create environment
        print(f"\n  Creating conda environment '{env_name}' from {env_file.name}...")
        print("  This may take several minutes. Please wait...\n")

        with Spinner("Creating conda environment"):
            result = subprocess.run(
                ["conda", "env", "create", "-f", str(env_file), "-n", env_name, "-y"],
                capture_output=True,
                text=True,
                cwd=str(PROJ_ROOT),
            )

        if result.returncode != 0:
            print(f"✗ Failed to create conda environment")
            if result.stderr:
                print(f"  Error: {result.stderr[:500]}")  # Limit error output
            print("\n  Continuing with current Python environment...\n")
            return True  # Continue anyway
        else:
            print(f"✓ Successfully created conda environment '{env_name}'\n")

    # Provide activation instructions
    print(f"{'=' * 70}")
    print("ENVIRONMENT ACTIVATION REQUIRED")
    print(f"{'=' * 70}")
    print(f"\n⚠ To use the conda environment '{env_name}', please:")
    print(f"  1. Activate it in your terminal:")
    print(f"     conda activate {env_name}")
    print(f"  2. Then run this script again from that terminal.\n")
    print(f"  Alternatively, you can continue with your current Python environment.\n")
    print(f"{'=' * 70}\n")

    if not skip_prompt:
        response = (
            input("Continue with current environment? (yes/no) [yes]: ").strip().lower()
        )
        if response and response not in ["yes", "y"]:
            print("\nPlease activate the environment and run the script again.")
            return False

    print("  Continuing with current Python environment...\n")
    return True


class Spinner:
    """A simple spinner animation for terminal output."""

    def __init__(self, message="Processing", delay=0.1):
        self.spinner_chars = itertools.cycle(
            ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        )
        self.message = message
        self.delay = delay
        self.stop_spinner = False
        self.spinner_thread = None

    def _spin(self):
        """Internal method to run the spinner animation."""
        while not self.stop_spinner:
            sys.stdout.write(f"\r{next(self.spinner_chars)} {self.message}...")
            sys.stdout.flush()
            time.sleep(self.delay)
        # Clear the spinner line
        sys.stdout.write("\r" + " " * (len(self.message) + 10) + "\r")
        sys.stdout.flush()

    def start(self):
        """Start the spinner animation."""
        self.stop_spinner = False
        self.spinner_thread = threading.Thread(target=self._spin, daemon=True)
        self.spinner_thread.start()

    def stop(self, success=True):
        """Stop the spinner animation."""
        self.stop_spinner = True
        if self.spinner_thread:
            self.spinner_thread.join(timeout=0.5)
        # Show completion status
        status = "✓" if success else "✗"
        sys.stdout.write(f"\r{status} {self.message}\n")
        sys.stdout.flush()

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop(success=(exc_type is None))


def run_notebook(notebook_path, description):
    """
    Execute a Jupyter notebook using nbconvert.
    Output is redirected to a temporary directory and cleaned up after execution.

    Args:
        notebook_path: Path to the notebook file
        description: Human-readable description of the notebook

    Returns:
        True if execution succeeded, False otherwise
    """
    if not notebook_path.exists():
        print(f"ERROR: Notebook not found at {notebook_path}")
        return False

    start_time = time.time()

    # Create a temporary directory for nbconvert output
    temp_output_dir = tempfile.mkdtemp(prefix="nbconvert_")

    try:
        # Start spinner animation
        with Spinner(f"Executing {description}"):
            # Execute notebook using nbconvert
            # --ExecutePreprocessor.timeout=600 sets a 10-minute timeout per cell
            # --output-dir redirects output to temp directory (will be cleaned up)
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "notebook",
                    "--execute",
                    "--ExecutePreprocessor.timeout=600",
                    "--ExecutePreprocessor.kernel_name=python3",
                    "--output-dir",
                    temp_output_dir,
                    str(notebook_path),
                ],
                capture_output=True,
                text=True,
                cwd=str(PROJ_ROOT),
            )

        elapsed_time = time.time() - start_time

        if result.returncode == 0:
            print(
                f"  Execution time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)\n"
            )
            return True
        else:
            print(f"  Execution time: {elapsed_time:.2f} seconds")
            print(f"\n  ERROR DETAILS:")
            print(f"  Return code: {result.returncode}")
            if result.stdout:
                print(f"\n  STDOUT:\n{result.stdout}")
            if result.stderr:
                print(f"\n  STDERR:\n{result.stderr}")
            print()
            return False

    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"  Execution time: {elapsed_time:.2f} seconds")
        print(f"\n  EXCEPTION: {str(e)}\n")
        return False

    finally:
        # Clean up temporary output directory
        try:
            shutil.rmtree(temp_output_dir)
        except Exception:
            pass  # Ignore cleanup errors


def main():
    """Main execution function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run all notebooks in the car price prediction pipeline in order"
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Run in non-interactive mode (skip confirmation prompts)",
    )
    parser.add_argument(
        "--setup-env",
        action="store_true",
        help="Set up conda environment from environment.yml",
    )
    parser.add_argument(
        "--skip-env",
        action="store_true",
        help="Skip environment setup and use current Python environment",
    )
    args = parser.parse_args()

    print(f"Project root: {PROJ_ROOT}\n")
    print("=" * 70)
    print("CAR PRICE PREDICTION PIPELINE")
    print("=" * 70)

    # Environment setup
    if args.skip_env:
        print("  Skipping environment setup (using current Python environment)\n")
    elif args.setup_env:
        # User explicitly requested environment setup
        if not setup_conda_environment(ENV_NAME, ENV_FILE, skip_prompt=args.yes):
            return 1  # Exit if user needs to activate environment
    elif check_conda_available() and ENV_FILE.exists():
        # Conda and env file available - ask user
        if not args.yes:
            print(f"\nEnvironment file detected: {ENV_FILE.name}")
            print(f"Environment name: {ENV_NAME}\n")
            response = (
                input(
                    "Do you want to set up/use the conda environment? (yes/no) [no]: "
                )
                .strip()
                .lower()
            )
            if response in ["yes", "y"]:
                if not setup_conda_environment(ENV_NAME, ENV_FILE, skip_prompt=False):
                    return 1  # Exit if user needs to activate environment
            else:
                print("  Using current Python environment.\n")
        # In non-interactive mode with --yes, skip environment setup

    print(f"\nThis script will execute {len(NOTEBOOKS)} notebooks in order.\n")
    print("Execution order:")
    for i, nb in enumerate(NOTEBOOKS, 1):
        print(f"  {i}. {nb['description']} ({nb['name']})")
    print("\n" + "=" * 70 + "\n")

    # Confirm execution (unless --yes flag is used)
    if not args.yes:
        response = input("Do you want to proceed? (yes/no): ").strip().lower()
        if response not in ["yes", "y"]:
            print("Execution cancelled.")
            return 0

    print("\nStarting pipeline execution...\n")
    overall_start_time = time.time()

    # Execute notebooks in order
    failed_notebooks = []
    for i, notebook in enumerate(NOTEBOOKS, 1):
        print(f"\n[{i}/{len(NOTEBOOKS)}] ", end="")
        print(f"Running: {notebook['description']}")
        print(f"Notebook: {notebook['name']}")
        print(f"{'=' * 70}")
        success = run_notebook(notebook["path"], notebook["description"])

        if not success:
            failed_notebooks.append(notebook["name"])
            print(f"\n⚠ WARNING: Pipeline stopped due to failure in {notebook['name']}")
            print("Subsequent notebooks depend on this one and will not be executed.\n")
            break

    overall_elapsed_time = time.time() - overall_start_time

    # Summary
    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 70)

    if not failed_notebooks:
        print("✓ All notebooks executed successfully!")
        print(
            f"  Total execution time: {overall_elapsed_time:.2f} seconds ({overall_elapsed_time / 60:.2f} minutes)"
        )
    else:
        print(f"✗ Pipeline execution failed at: {', '.join(failed_notebooks)}")
        print(f"  Execution time before failure: {overall_elapsed_time:.2f} seconds")
        print(
            "\nPlease check the error messages above and fix the issues before re-running."
        )

    print("=" * 70 + "\n")

    return 0 if not failed_notebooks else 1


if __name__ == "__main__":
    sys.exit(main())
