# -*- coding: utf-8 -*-
"""
Transforms workout history data from a "wide" format (found in the parsed
Brisnet data) into a "long" format (one row per workout per horse).

This script uses the Brisnet specification (via bris_spec.pkl) to map
field numbers to the actual (potentially messy) column names found in the
wide data file, allowing it to reshape data even with inconsistent naming
conventions for workouts #2 through #12.

Reads:
- parsed_race_data_full.parquet (output of the modified bris_spec.py)
- bris_spec.pkl (cache mapping Brisnet Field# to actual column labels)

Outputs:
- workouts_long_format.parquet (long format workout data)
"""
from __future__ import annotations

import pandas as pd
import logging
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Final, Optional, Set
import numpy as np # For potential numeric cleaning

# --- Configuration ---

# 1. Define paths relative to the script location
SCRIPT_DIR: Final[Path] = Path(__file__).parent.resolve()
PROJECT_ROOT: Final[Path] = SCRIPT_DIR.parent.parent    # .../horse_racing_project/

# 2. Input files
WIDE_DATA_PARQUET_FILENAME: Final[str] = "parsed_race_data_full.parquet"
SPEC_CACHE_FILENAME: Final[str] = "bris_spec.pkl"

WIDE_DATA_FILE_PATH: Final[Path] = PROJECT_ROOT / "data" / "processed" / WIDE_DATA_PARQUET_FILENAME
SPEC_CACHE_FILE_PATH: Final[Path] = PROJECT_ROOT / "data" / "cache" / SPEC_CACHE_FILENAME

# 3. Output file for the transformed "long" workout data
LONG_WORKOUTS_FILENAME: Final[str] = "workouts_long_format.parquet"
LONG_WORKOUTS_FILE_PATH: Final[Path] = PROJECT_ROOT / "data" / "processed" / LONG_WORKOUTS_FILENAME

# 4. Define ID variables: Columns identifying the *current* horse/race entry.
#    *** Updated based on your confirmation ***
ID_VARIABLES: Final[List[str]] = [
    'track', # Today's track code
    'race', # Today's race number
    # 'date', # Add if available and needed as part of the unique ID
    'post_position', # Today's post position
    'horse_name' # Today's horse name
    # 'program_number_if_available' # Removed as per your list, add back if needed
]

# 5. Define the mapping from desired clean metric names to the
#    Brisnet Field Number of the *first* workout (#1) for that metric.
#    (This map remains the same as it's based on the Brisnet spec)
WORKOUT_METRIC_MAP: Final[Dict[str, int]] = {
    'work_date': 102,
    'work_time': 114,
    'work_track': 126,
    'work_distance': 138,
    'work_track_condition': 150,
    'work_description': 162,      # e.g., 'H', 'B', 'Hg'
    'work_surface_type': 174,     # e.g., 'T', 'IT', 'TT'
    'work_num_at_dist': 186,      # Number of works that day/distance
    'work_rank': 198              # Rank of the work
}

# --- End Configuration ---


def load_data(parquet_path: Path, pkl_path: Path) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Loads the wide Parquet data and the spec cache PKL."""
    wide_df: Optional[pd.DataFrame] = None
    spec_df: Optional[pd.DataFrame] = None

    # Load Wide Data
    if not parquet_path.exists():
        print(f"Error: Input Parquet file not found at {parquet_path}")
        return None, None
    try:
        print(f"Loading wide format data from: {parquet_path}")
        wide_df = pd.read_parquet(parquet_path, engine='pyarrow')
        print(f"Loaded wide data with shape: {wide_df.shape}")
    except Exception as e:
        print(f"Error loading Parquet file {parquet_path}: {e}")
        return None, None

    # Load Spec Cache
    if not pkl_path.exists():
        print(f"Error: Specification cache file not found at {pkl_path}")
        return wide_df, None
    try:
        print(f"Loading specification cache from: {pkl_path}")
        spec_df = pd.read_pickle(pkl_path)
        # Ensure Field Number is usable (as index or column 'field_number')
        if 'field_number' not in spec_df.columns and spec_df.index.name != 'field_number':
             spec_df['field_number'] = spec_df.index # Assume index is field number
        if 'label' not in spec_df.columns:
             print(f"Error: Spec cache '{pkl_path.name}' must contain a 'label' column (the actual column names).")
             return wide_df, None
        # Set field_number as index for easy lookup using .loc
        # Important: Verify that the index IS the field number after loading.
        if 'field_number' in spec_df.columns and spec_df.index.name != 'field_number':
             print("Setting 'field_number' column as index for spec cache.")
             spec_df = spec_df.set_index('field_number', drop=False) # Keep column too if needed

        if spec_df.index.name != 'field_number':
             print("Warning: Spec cache index is not named 'field_number'. Lookup using .loc might fail if index isn't the field number.")

        print(f"Loaded spec cache for {len(spec_df)} fields.")
    except Exception as e:
        print(f"Error loading specification cache file {pkl_path}: {e}")
        return wide_df, None

    return wide_df, spec_df


def clean_workout_data(df: pd.DataFrame) -> pd.DataFrame:
    """Applies specific cleaning and type conversion to long workout data."""
    print("Cleaning and converting types in long workout data...")
    if df.empty:
        return df

    df = df.copy() # Work on a copy to avoid SettingWithCopyWarning

    # Convert workout date
    if 'work_date' in df.columns:
        # Assuming dates were read as YYYYMMDD strings initially
        df['work_date'] = pd.to_datetime(df['work_date'], format='%Y%m%d', errors='coerce')
        # Remove rows where the essential workout date is missing after conversion
        original_rows = len(df)
        df.dropna(subset=['work_date'], inplace=True)
        if len(df) < original_rows:
            print(f"  Dropped {original_rows - len(df)} rows with invalid work_date.")

    # Convert numeric fields
    numeric_cols = ['work_time', 'work_distance', 'work_num_at_dist', 'work_rank']
    for col in numeric_cols:
        if col in df.columns:
            # Convert to numeric, coercing errors (non-numeric values become NaN)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Optional: Further cleaning based on NaN values in key numeric fields if desired
    # original_rows = len(df)
    # df.dropna(subset=['work_time', 'work_distance'], inplace=True) # Example
    # if len(df) < original_rows:
    #     print(f"  Dropped {original_rows - len(df)} rows with missing work_time or work_distance.")

    # Ensure workout_num is integer
    if 'workout_num' in df.columns:
        # Convert after potential NaNs from previous steps are handled or accepted
        if df['workout_num'].notna().all():
             df['workout_num'] = df['workout_num'].astype(int)
        else:
             print("Warning: NaNs found in workout_num column before integer conversion.")

    print(f"Workout data cleaned. Shape after cleaning: {df.shape}")
    return df


def main():
    """
    Main function to transform workout data.
    This function will be called by run_pipeline.py.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"--- Starting Workout Data Reshaping ({pd.Timestamp.now(tz='America/New_York').strftime('%Y-%m-%d %H:%M:%S %Z')}) ---")

    wide_df, spec_df = load_data(WIDE_DATA_FILE_PATH, SPEC_CACHE_FILE_PATH)

    if wide_df is None or spec_df is None:
        logger.error("Failed to load necessary data. Aborting.")
        return

    # Verify ID variables exist in the loaded wide_df
    actual_id_vars = [col for col in ID_VARIABLES if col in wide_df.columns]
    if len(actual_id_vars) != len(ID_VARIABLES):
        missing_ids = [col for col in ID_VARIABLES if col not in actual_id_vars]
        logger.warning(f"Configuration Warning: The following specified ID_VARIABLES were not found in {WIDE_DATA_PARQUET_FILENAME}: {missing_ids}")
        if not actual_id_vars:
            logger.error("Error: No valid ID variables found to identify rows. Aborting.")
            return
    logger.info(f"Using ID Variables: {actual_id_vars}")

    # --- Reshaping Loop ---
    all_workouts_long: List[pd.DataFrame] = []
    logger.info("Reshaping workout data using specification (Field# -> Actual Column Label)...")

    # Loop through potential workouts 1 to 12
    for i in range(12):
        workout_num = i + 1
        # This dictionary maps the actual (messy) column name found in wide_df
        # to the desired clean metric name for THIS specific workout number.
        current_workout_actual_cols_map: Dict[str, str] = {}
        # This list stores the actual (messy) column names to select from wide_df
        actual_cols_to_select_for_this_num: List[str] = []

        # Find actual column names for this specific workout number using the spec cache
        for clean_name, base_field in WORKOUT_METRIC_MAP.items():
            field_num = base_field + i
            try:
                # Lookup the actual column name ('label') from the spec cache using field number
                # Assumes spec_df index is the field number
                actual_col_name = spec_df.loc[field_num, 'label']
                # Check if this actual column name exists in the wide dataframe
                if actual_col_name in wide_df.columns:
                    current_workout_actual_cols_map[actual_col_name] = clean_name
                    actual_cols_to_select_for_this_num.append(actual_col_name)
                # else: # Debug: Column defined in spec but not found in Parquet
                #    print(f"Debug: Col '{actual_col_name}' (Field {field_num}) not in {WIDE_DATA_PARQUET_FILENAME}")
            except KeyError:
                # Field number might not be present in the spec_df index (e.g., it was reserved/skipped)
                # print(f"Debug: Field number {field_num} not found in spec_df index.")
                pass
            except Exception as e:
                # Catch other potential lookup errors
                logger.warning(f"Warning: Error looking up field {field_num} label in spec_df: {e}")

        # --- Process data if columns were found for this workout number ---
        if actual_cols_to_select_for_this_num:
            # Select ID vars + the specific messy workout columns for THIS workout num
            cols_to_grab = actual_id_vars + actual_cols_to_select_for_this_num
            # Ensure unique column names in list
            cols_to_grab = list(dict.fromkeys(cols_to_grab))

            # Create a temporary DataFrame with just the needed columns
            temp_df = wide_df[cols_to_grab].copy()

            # Rename the messy workout columns to their clean metric names
            temp_df.rename(columns=current_workout_actual_cols_map, inplace=True)

            # Add the workout number identifier
            temp_df['workout_num'] = workout_num

            # Define the final list of columns we want in the long format df
            # (IDs, clean metric names defined in WORKOUT_METRIC_MAP, workout_num)
            final_cols_for_concat = actual_id_vars + list(WORKOUT_METRIC_MAP.keys()) + ['workout_num']
            # Filter this list based on columns that *actually exist* in temp_df after renaming
            final_cols_for_concat = [c for c in final_cols_for_concat if c in temp_df.columns]

            # Append the processed (selected and renamed) data for this workout number
            all_workouts_long.append(temp_df[final_cols_for_concat])

    # --- Concatenate and Finalize ---
    if all_workouts_long:
        logger.info("Concatenating data for all workouts...")
        long_workouts_df = pd.concat(all_workouts_long, ignore_index=True, sort=False)
        logger.info(f"Raw concatenated workout data shape: {long_workouts_df.shape}")

        # Apply cleaning and type conversion to the combined DataFrame
        long_workouts_df = clean_workout_data(long_workouts_df)

        # --- Save Results ---
        if not long_workouts_df.empty:
            logger.info(f"Saving long format workout data to: {LONG_WORKOUTS_FILE_PATH}")
            try:
                long_workouts_df.to_parquet(LONG_WORKOUTS_FILE_PATH, index=False, engine='pyarrow')
                logger.info("Save complete.")
                logger.info("Output DataFrame Info:")
                long_workouts_df.info(verbose=False, show_counts=True)
            except ImportError:
                logger.error("Error: 'pyarrow' library not found. Cannot save to Parquet.")
                logger.error("Please install it: pip install pyarrow")
            except Exception as e:
                logger.error(f"Error saving final data to Parquet file {LONG_WORKOUTS_FILE_PATH}: {e}")
        else:
            logger.warning("No valid workout data remained after cleaning. Output file not saved.")

    else:
        logger.warning("No workout data columns found or processed based on spec. No output file created.")

    logger.info(f"--- Workout Reshaping Script Finished ({pd.Timestamp.now(tz='America/New_York').strftime('%Y-%m-%d %H:%M:%S %Z')}) ---")


if __name__ == "__main__":
    # Setup basic logging if this script is run directly
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
    main()