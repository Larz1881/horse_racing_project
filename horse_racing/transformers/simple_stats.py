import pandas as pd
import logging
import sys
from pathlib import Path

from config.settings import PARSED_RACE_DATA, PROCESSED_DATA_DIR
from horse_racing.transformers.long_format_transformer import (
    compute_fractional_splits,
    calculate_record_statistics,
    calculate_jockey_performance,
)

# --- Input file path from centralized settings ---
INPUT_PARQUET_PATH: Path = PARSED_RACE_DATA



def main():
    """
    Main function to process feature engineering.
    This function will be called by run_pipeline.py.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"--- Starting Feature Engineering ({pd.Timestamp.now(tz='America/New_York').strftime('%Y-%m-%d %H:%M:%S %Z')}) ---")

    # Check if input file exists
    if not INPUT_PARQUET_PATH.exists():
        logger.error(f"Error: Input file not found at {INPUT_PARQUET_PATH}")
        return

    logger.info(f"Loading data from: {INPUT_PARQUET_PATH}")
    df = pd.read_parquet(INPUT_PARQUET_PATH)
    logger.info(f"Loaded data with shape: {df.shape}")

    logger.info("Computing record statistics...")
    df_perf = calculate_record_statistics(df)
    logger.info(f"Created performance DataFrame with shape: {df_perf.shape}")

    # Compute fractional splits
    logger.info("Computing fractional splits...")
    df_splits = compute_fractional_splits(df)
    logger.info(f"Created splits DataFrame with shape: {df_splits.shape}")

    # Output to CSV in the processed data directory
    output_perf_csv = PROCESSED_DATA_DIR / "horse_performance.csv"
    output_splits_csv = PROCESSED_DATA_DIR / "splits.csv"

    logger.info(f"Saving horse_performance.csv to: {output_perf_csv}")
    df_perf.to_csv(output_perf_csv, index=False)
    
    logger.info(f"Saving splits.csv to: {output_splits_csv}")
    df_splits.to_csv(output_splits_csv, index=False)

    logger.info("Processing jockey statistics...")
    df_jockey = calculate_jockey_performance(df)
    if not df_jockey.empty:
        output_jockey_parquet = PROCESSED_DATA_DIR / "jockey_performance.parquet"
        logger.info(f"Saving jockey_performance.parquet to: {output_jockey_parquet}")
        df_jockey.to_parquet(output_jockey_parquet, index=False)

    logger.info("--- Feature Engineering Script Finished ---")


if __name__ == "__main__":
    # Setup basic logging if this script is run directly
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
    main()



