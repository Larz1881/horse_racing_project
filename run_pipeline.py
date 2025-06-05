#!/usr/bin/env python
"""
Main pipeline orchestrator for the Horse Racing Analysis project.
This script sequences the parsing, transformation, and feature engineering steps.
It is intended to be run from the project root directory.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional  # Added for type hinting

# --- Setup Project Root and System Path ---
# This script (run_pipeline.py) is in the project root.
# The PROJECT_ROOT is the directory containing this script.
try:
    PROJECT_ROOT: Path = Path(__file__).resolve().parent
    # Add PROJECT_ROOT to sys.path to allow imports like 'config.settings'
    # and 'horse_racing.parsers...', 'horse_racing.transformers...'
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
except NameError:
    # Fallback if __file__ is not defined
    PROJECT_ROOT: Path = Path.cwd()
    if not (PROJECT_ROOT / "config").exists() or not (PROJECT_ROOT / "horse_racing").exists():
        print("Warning: PROJECT_ROOT might not be correctly set. Assuming current working directory.")
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

# --- Imports ---
# These imports rely on PROJECT_ROOT being correctly added to sys.path.
try:
    from config.settings import (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        CACHE_DIR,
        DRF_PATTERN,
    )
    # bris_spec_new.py is in horse_racing/parsers/
    from horse_racing.parsers.bris_spec_new import main as parse_bris_main

    from horse_racing.transformers.current_race_info import main as create_current_info_main
    from horse_racing.transformers.long_format_transformer import (
        transform_workouts,
        transform_past_starts,
    )
    from horse_racing.transformers.feature_engineering import main as engineer_features_main
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure that:")
    print(f"1. The project root ({PROJECT_ROOT}) is correctly identified and added to sys.path.")
    print("2. 'bris_spec_new.py' is in the 'horse_racing/parsers/' directory and is importable.")
    print("3. Other 'horse_racing' script files (current_race_info.py, etc.) exist in their 'horse_racing/transformers/' subdirectories.")
    print("4. Each of these scripts has a callable 'main()' function.")
    print(
        "5. 'config/settings.py' exists in the 'config/' directory and defines necessary variables (RAW_DATA_DIR, etc.).")
    print(
        "6. Ensure necessary '__init__.py' files are present in 'horse_racing/', 'horse_racing/parsers/', and 'horse_racing/transformers/' to make them packages.")
    sys.exit(1)

# --- Logging Setup ---
LOG_DIR: Path = PROJECT_ROOT / "logs"  # Logs directory in project root
try:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE_PATH: Path = LOG_DIR / f'pipeline_{datetime.now():%Y%m%d_%H%M%S}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE_PATH),
            logging.StreamHandler(sys.stdout)
        ]
    )
except Exception as log_e:
    print(f"Error setting up logging: {log_e}. Logging to console only.")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

logger = logging.getLogger(__name__)


# --- Helper Function to Find DRF File ---
def find_latest_drf_file() -> Path:
    """Finds the most recent DRF file in the raw data directory."""
    logger.info(f"Searching for DRF files in: {RAW_DATA_DIR} with pattern: '{DRF_PATTERN}'")
    if not RAW_DATA_DIR.exists():
        msg = f"Raw data directory does not exist: {RAW_DATA_DIR}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    drf_files = list(RAW_DATA_DIR.glob(DRF_PATTERN))
    if not drf_files:
        msg = f"No DRF files found matching pattern '{DRF_PATTERN}' in {RAW_DATA_DIR}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    latest_file = max(drf_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Using DRF file: {latest_file.name}")
    return latest_file


# --- Main Pipeline Function ---
def run_complete_pipeline():
    """Runs the complete data processing pipeline."""
    logger.info("=== Starting Horse Racing Data Pipeline ===")
    try:
        drf_to_process = find_latest_drf_file()

        logger.info("--- Step 1: Parsing DRF file (using horse_racing/parsers/bris_spec_new.py) ---")
        # Assuming bris_spec_new.main (now parse_bris_main) can accept the DRF file path
        parse_bris_main(drf_file_path_arg=drf_to_process)
        logger.info("--- Step 1: Parsing DRF file completed ---")

        logger.info("--- Step 2: Creating current race info ---")
        create_current_info_main()
        logger.info("--- Step 2: Creating current race info completed ---")

        logger.info("--- Step 3: Transforming workouts ---")
        transform_workouts()
        logger.info("--- Step 3: Transforming workouts completed ---")

        logger.info("--- Step 4: Transforming past starts ---")
        transform_past_starts()
        logger.info("--- Step 4: Transforming past starts completed ---")

        logger.info("--- Step 5: Engineering features ---")
        engineer_features_main()
        logger.info("--- Step 5: Engineering features completed ---")

        logger.info("Running advanced fitness metrics...")
        from horse_racing.transformers.advanced_fitness_metrics import main as run_fitness_metrics
        run_fitness_metrics()

        logger.info("Running sophisticated workout analysis...")
        from horse_racing.transformers.sophisticated_workout_analysis import main as run_workout_analysis
        run_workout_analysis()

        logger.info("Running advanced pace projection...")
        from horse_racing.transformers.advanced_pace_projection import main as run_pace_analysis
        run_pace_analysis()

        logger.info("Running multi-dimensional class assessment...")
        from horse_racing.transformers.multi_dimensional_class_assessment import main as run_class_assessment
        run_class_assessment()

        logger.info("Running form cycle detection...")
        from horse_racing.transformers.form_cycle_detector import main as run_form_cycle
        run_form_cycle()

        logger.info("Running integrated analytics system...")
        from horse_racing.transformers.integrated_analytics_system import main as run_integrated_analytics
        run_integrated_analytics()

        logger.info("=== Pipeline completed successfully! ===")

    except FileNotFoundError as fnf_error:
        logger.error(f"Pipeline aborted: A required file was not found. Details: {fnf_error}", exc_info=False)
    except AttributeError as attr_error:
        logger.error(
            f"Pipeline aborted: An attribute error occurred (often due to missing or incorrectly named main function in a module). Details: {attr_error}",
            exc_info=True)
    except ImportError as import_err:
        logger.error(f"Pipeline aborted: An import error occurred. Details: {import_err}", exc_info=True)
    except Exception as e:
        logger.error(f"Pipeline failed with an unexpected error: {e}", exc_info=True)


if __name__ == "__main__":
    logger.info(f"Pipeline started. Project Root: {PROJECT_ROOT}")
    logger.info(
        f"Using settings: RAW_DATA_DIR={RAW_DATA_DIR}, PROCESSED_DATA_DIR={PROCESSED_DATA_DIR}, CACHE_DIR={CACHE_DIR}")

    run_complete_pipeline()