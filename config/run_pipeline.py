#!/usr/bin/env python
"""Main pipeline to process current DRF file."""

import sys
from pathlib import Path
import logging
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent))

from config.settings import *
from src.parsers.bris_spec import main as parse_bris
from src.transformers.current_race_info import create_current_info
from src.transformers.transform_workouts import transform_workouts
from src.transformers.transform_past_starts import transform_past_starts
from src.transformers.feature_engineering import engineer_features

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'pipeline_{datetime.now():%Y%m%d_%H%M%S}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def find_current_drf():
    """Find the most recent DRF file in raw data directory."""
    drf_files = list(RAW_DATA_DIR.glob(DRF_PATTERN))
    if not drf_files:
        raise FileNotFoundError(f"No DRF files found in {RAW_DATA_DIR}")
    # Return the most recent file
    return max(drf_files, key=lambda p: p.stat().st_mtime)

def main():
    """Run the complete pipeline."""
    try:
        # Find current DRF
        current_drf = find_current_drf()
        logger.info(f"Processing: {current_drf.name}")

        # Step 1: Parse DRF
        logger.info("Step 1: Parsing DRF file...")
        parse_bris(current_drf)

        # Step 2: Create current race info
        logger.info("Step 2: Creating current race info...")
        create_current_info()

        # Step 3: Transform workouts
        logger.info("Step 3: Transforming workouts...")
        transform_workouts()

        # Step 4: Transform past starts
        logger.info("Step 4: Transforming past starts...")
        transform_past_starts()

        # Step 5: Feature engineering
        logger.info("Step 5: Engineering features...")
        engineer_features()

        logger.info("Pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()