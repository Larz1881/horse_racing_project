import numpy as np
import pandas as pd
import logging
import sys
from pathlib import Path
from typing import List, Final, Optional

from config.settings import PARSED_RACE_DATA, PROCESSED_DATA_DIR
from horse_racing.transformers.long_format_transformer import compute_fractional_splits

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

    # define each record group by its start, win, place, show, earnings columns
    record_groups = {
        "distance":       ("starts_pos_65",  "wins_pos_66",  "places_pos_67",  "shows_pos_68",  "earnings_pos_69"),
        "track":          ("starts_pos_70",  "wins_pos_71",  "places_pos_72",  "shows_pos_73",  "earnings_pos_74"),
        "turf":           ("starts_pos_75",  "wins_pos_76",  "places_pos_77",  "shows_pos_78",  "earnings_pos_79"),
        "mud":            ("starts_pos_80",  "wins_pos_81",  "places_pos_82",  "shows_pos_83",  "earnings_pos_84"),
        "current_year":   ("starts_pos_86",  "wins_pos_87",  "places_pos_88",  "shows_pos_89",  "earnings_pos_90"),
        "previous_year":  ("starts_pos_92",  "wins_pos_93",  "places_pos_94",  "shows_pos_95",  "earnings_pos_96"),
        "lifetime":       ("starts_pos_97",  "wins_pos_98",  "places_pos_99",  "shows_pos_100", "earnings_pos_101"),
    }

    # loop and compute record statistics
    logger.info("Computing record statistics...")
    for name, (col_start, col_win, col_place, col_show, col_earn) in record_groups.items():
        # Check if columns exist
        missing_cols = [col for col in [col_start, col_win, col_place, col_show, col_earn] if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing columns for {name} record group: {missing_cols}")
            continue
            
        # avoid division-by-zero
        starts = df[col_start].replace({0: np.nan})

        df[f"{name}_win_pct"]            = df[col_win]   / starts
        df[f"{name}_itm_pct"]            = (df[col_win] + df[col_place] + df[col_show]) / starts
        df[f"{name}_earnings_per_start"] = df[col_earn]  / starts

    # build the list of metric column names
    record_groups_list = ["distance", "track", "turf", "mud", "current_year", "previous_year", "lifetime"]
    metrics = []
    for name in record_groups_list:
        metrics += [f"{name}_win_pct", f"{name}_itm_pct", f"{name}_earnings_per_start"]

    # select only the identifiers + your new metrics
    base_cols = ["race", "post_position", "morn_line_odds_if_available", "horse_name"]
    available_metrics = [col for col in metrics if col in df.columns]
    available_base_cols = [col for col in base_cols if col in df.columns]
    
    df_perf = df[available_base_cols + available_metrics].copy()
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

    # Ensure all Jockey stats columns are numeric
    jockey_cols = [
        'wins_4_pos_1158',
        'jockey_sts_current_year',
        'wins_4_pos_1163',
        'jockey_sts_previous_year',
        'jockey_wins_current_meet',
        'jockey_sts_current_meet',
        't_j_combo_wins_meet',
        't_j_combo_starts_meet',
    ]
    
    logger.info("Processing jockey statistics...")
    for col in jockey_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Build new DataFrame for jockey performance
    jockey_base_cols = ['today_s_jockey', 'jockey_1']
    available_jockey_base = [col for col in jockey_base_cols if col in df.columns]
    
    if available_jockey_base and 'today_s_jockey' in df.columns:
        df_jockey = pd.DataFrame({
            'Jockey': df['today_s_jockey'],
            'Change': df['today_s_jockey'].eq(df.get('jockey_1', df['today_s_jockey'])).map({True: 'N', False: 'Y'}),
        })
        
        # Add percentage columns if data is available
        if 'wins_4_pos_1158' in df.columns and 'jockey_sts_current_year' in df.columns:
            df_jockey['Win %'] = df['wins_4_pos_1158'] / df['jockey_sts_current_year']
        if 'wins_4_pos_1163' in df.columns and 'jockey_sts_previous_year' in df.columns:
            df_jockey['Last Yr %'] = df['wins_4_pos_1163'] / df['jockey_sts_previous_year']
        if 'jockey_wins_current_meet' in df.columns and 'jockey_sts_current_meet' in df.columns:
            df_jockey['Meet'] = df['jockey_wins_current_meet'] / df['jockey_sts_current_meet']
        if 't_j_combo_wins_meet' in df.columns and 't_j_combo_starts_meet' in df.columns:
            df_jockey['T/J'] = df['t_j_combo_wins_meet'] / df['t_j_combo_starts_meet']

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



