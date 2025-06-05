# -*- coding: utf-8 -*-
"""
Parses a Brisnet comma-delimited race data file (e.g., CDX0502.DRF)
using column names derived from the Brisnet specification dictionary,
attempts type conversions based on bris_dict.txt, keeps all parsed
columns (except 'reserved'), and saves the final DataFrame to an
Apache Parquet file.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any, Final, Optional, Set
import re  # For parsing bris_dict.txt
import logging  # Import logging

from config.settings import (
    BRIS_SPEC_CACHE,
    BRIS_DICT,
    PARSED_RACE_DATA,
    PROCESSED_DATA_DIR,
    CACHE_DIR,
)

# Default DRF if run directly and no argument is passed, can be overridden by main's argument
# RACE_DATA_FILENAME_DEFAULT: Final[str] = "PIM0509.DRF"

SPEC_CACHE_FILE_PATH_BRIS: Final[Path] = BRIS_SPEC_CACHE
BRIS_DICT_FILE_PATH_BRIS: Final[Path] = BRIS_DICT
OUTPUT_PARQUET_FILE_PATH_BRIS: Final[Path] = PARSED_RACE_DATA

# --- Helper Functions (load_specification_cache, parse_bris_dict_types, etc. remain the same) ---
def load_specification_cache(spec_cache_path: Path) -> Optional[pd.DataFrame]:
    """Loads the column label specification DataFrame from the cache file."""
    logger = logging.getLogger(__name__) # Get logger instance
    if not spec_cache_path.exists():
        logger.error(f"Error: Specification cache file not found at {spec_cache_path}")
        logger.error("Please ensure bris_spec.pkl exists (or run the script that generates it).")
        return None
    try:
        logger.info(f"Loading specification cache from: {spec_cache_path}")
        spec_df = pd.read_pickle(spec_cache_path)
        spec_df = spec_df.sort_index() # Assume index is Field Number
        if 'label' not in spec_df.columns:
             logger.error("Error: Spec cache DataFrame must contain a 'label' column.")
             return None
        if not spec_df['label'].is_unique:
             logger.error("Error: Labels loaded from cache are not unique!")
             return None
        if spec_df.index.name != 'field_number':
            spec_df['field_number'] = spec_df.index
        return spec_df
    except Exception as e:
        logger.error(f"Error loading specification cache file: {e}")
        return None

def parse_bris_dict_types(dict_path: Path) -> Dict[int, str]:
    """
    Parses bris_dict.txt primarily to get the declared TYPE for each Field #.
    Returns a dictionary mapping {field_number: 'TYPE'}.
    """
    logger = logging.getLogger(__name__)
    field_types: Dict[int, str] = {}
    pattern = re.compile(r"^\s*(\d+)\s+(.*?)\s{2,}(CHARACTER|NUMERIC|DATE)\s+", re.MULTILINE)
    logger.info(f"Parsing field types from: {dict_path}")
    if not dict_path.exists():
        logger.error(f"Error: {dict_path} not found.")
        return field_types
    try:
        with open(dict_path, 'r', encoding='iso-8859-1') as f:
            content = f.read()
        for match in pattern.finditer(content):
            field_num = int(match.group(1))
            type_ = match.group(3).strip()
            field_types[field_num] = type_
        if 2 in field_types and field_types[2] == 'CHARACTER': field_types[2] = 'DATE_STR'
        for fn in range(102, 114): field_types[fn] = 'DATE_STR'
        for fn in range(256, 266): field_types[fn] = 'DATE_STR'
        logger.info(f"Parsed types for {len(field_types)} fields from {dict_path.name}.")
    except Exception as e:
        logger.error(f"Error parsing {dict_path.name}: {e}")
    return field_types

def identify_column_types_from_spec(spec_df: pd.DataFrame, field_type_map: Dict[int, str]) -> Tuple[Set[str], Set[str]]:
    """Identifies numeric and date columns based on the parsed field types."""
    logger = logging.getLogger(__name__)
    numeric_cols: Set[str] = set()
    date_cols: Set[str] = set()
    logger.info("\nIdentifying column types using parsed bris_dict.txt info...")
    if 'field_number' not in spec_df.columns or 'label' not in spec_df.columns:
         logger.error("Error: spec_df DataFrame must contain 'field_number' and 'label' columns.")
         return numeric_cols, date_cols
    if not field_type_map:
        logger.warning("Warning: Field type map is empty. Cannot reliably identify types.")
        return numeric_cols, date_cols
    for _, row in spec_df.iterrows():
        field_num = row.get('field_number')
        label = row.get('label')
        if label is None or field_num is None or label.startswith("reserved"): continue
        field_type = field_type_map.get(field_num)
        if field_type == 'NUMERIC': numeric_cols.add(label)
        elif field_type == 'DATE_STR': date_cols.add(label)
    logger.info(f"Identified {len(numeric_cols)} numeric columns using spec.")
    logger.info(f"Identified {len(date_cols)} date columns using spec.")
    return numeric_cols, date_cols

def parse_brisnet_csv_data(data_file_to_parse: Path, column_names_all: List[str]) -> Optional[pd.DataFrame]:
    """Parses a comma-delimited Brisnet data file."""
    logger = logging.getLogger(__name__)
    if not data_file_to_parse.exists():
        logger.error(f"Error: Race data file not found at {data_file_to_parse}")
        return None
    column_names_filtered = [name for name in column_names_all if not name.startswith("reserved")]
    cols_to_use_indices = [i for i, name in enumerate(column_names_all) if not name.startswith("reserved")]
    if not column_names_filtered:
        logger.error("Error: No non-reserved columns found in the specification.")
        return None
    logger.info(f"\nAttempting to parse {data_file_to_parse.name} as comma-delimited...")
    logger.info(f"Excluding columns starting with 'reserved'. Using {len(column_names_filtered)} out of {len(column_names_all)} columns.")
    try:
        df = pd.read_csv(
            data_file_to_parse, sep=',', header=None, names=column_names_filtered,
            usecols=cols_to_use_indices, dtype=str, quotechar='"',
            on_bad_lines='warn', low_memory=False, encoding='iso-8859-1',
            skipinitialspace=True
        )
        logger.info(f"Parsing successful. Read {len(df)} lines from {data_file_to_parse.name}.")
        return df
    except Exception as e:
        logger.error(f"An unexpected error occurred during parsing of {data_file_to_parse.name}: {e}", exc_info=True)
        return None

def convert_data_types(df: pd.DataFrame, numeric_cols_labels: Set[str], date_cols_labels: Set[str]) -> pd.DataFrame:
    """Attempts to convert columns to appropriate numeric and date types."""
    logger = logging.getLogger(__name__)
    logger.info("\nAttempting data type conversions...")
    df_copy = df.copy()
    numeric_cols_present = [col for col in numeric_cols_labels if col in df_copy.columns]
    logger.info(f"Converting {len(numeric_cols_present)} columns to numeric...")
    converted_numeric_count = 0
    for col in numeric_cols_present:
        try:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
            converted_numeric_count += 1
        except Exception as e:
            logger.warning(f"  - Warning: Failed to convert column '{col}' to numeric: {e}")
    logger.info(f"Successfully converted {converted_numeric_count} numeric columns.")
    date_cols_present = [col for col in date_cols_labels if col in df_copy.columns]
    logger.info(f"Converting {len(date_cols_present)} columns to datetime...")
    converted_date_count = 0
    brisnet_date_format = '%Y%m%d'
    for col in date_cols_present:
        try:
            df_copy[col] = pd.to_datetime(df_copy[col], format=brisnet_date_format, errors='coerce')
            converted_date_count += 1
        except Exception as e:
            try:
                 logger.info(f"  - Retrying datetime conversion for '{col}' without specific format.")
                 df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
            except Exception as e2:
                 logger.warning(f"  - Warning: Failed to convert column '{col}' to datetime: {e2}")
    logger.info(f"Successfully converted {converted_date_count} date columns.")
    return df_copy

# --- NEW Main Function ---
def main(drf_file_path_arg: Optional[Path] = None):
    """
    Main processing logic for parsing a DRF file.
    Accepts a DRF file path as an argument.
    """
    # Get a logger specific to this module (or use a globally configured one)
    logger = logging.getLogger(__name__) # Ensures this main function uses logging

    # Ensure parent directories for output exist (moved from global scope for clarity)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)  # Cache dir might be for reading too

    logger.info("--- bris_spec_new.main() starting ---")

    # Determine which DRF file to process
    if drf_file_path_arg:
        current_drf_file_to_process = drf_file_path_arg
        logger.info(f"Processing DRF file provided as argument: {current_drf_file_to_process}")
    else:
        # Fallback: if no argument, use the default defined globally or error
        # This part depends on how you want `bris_spec_new.py` to behave if run directly
        # without `run_pipeline.py` providing the path.
        # For now, let's assume `run_pipeline.py` will always provide it.
        # If run directly, it might need to find the file or use a hardcoded default.
        # The find_latest_drf_file logic is better placed in run_pipeline.py.
        logger.error("DRF file path argument (drf_file_path_arg) is required for bris_spec_new.main().")
        return # Or raise an error

    # 1. Load the column name specification cache (Field# -> Label)
    spec_df = load_specification_cache(SPEC_CACHE_FILE_PATH_BRIS)
    if spec_df is None:
        logger.error("Aborting bris_spec_new.main() due to failure in loading spec cache.")
        return # Or exit(1)

    # 2. Parse the bris_dict.txt file to get Field# -> Type mapping
    field_type_map = parse_bris_dict_types(BRIS_DICT_FILE_PATH_BRIS)
    if not field_type_map:
        logger.warning("Could not parse field types from bris_dict.txt. Type conversion may be inaccurate.")
        # Decide if this is a critical error that should stop execution
        # For now, it proceeds with a warning.

    # 3. Extract the full ordered list of labels (column names) from the cache
    all_ordered_column_labels = spec_df['label'].tolist()

    # 4. Parse the actual race data file, excluding only reserved columns
    # Use the determined DRF file path
    race_data_df = parse_brisnet_csv_data(current_drf_file_to_process, all_ordered_column_labels)

    if race_data_df is not None and not race_data_df.empty:
        # 5. Identify numeric and date columns using the parsed spec info
        numeric_labels, date_labels = identify_column_types_from_spec(spec_df, field_type_map)

        # 6. Perform Type Conversions
        race_data_df = convert_data_types(race_data_df, numeric_labels, date_labels)

        # 7. Basic Integrity Check
        RACE_COL = 'race' 
        try:
             race_col_label = spec_df[spec_df['field_number'] == 3]['label'].iloc[0]
        except (KeyError, IndexError):
             logger.warning("Warning: Could not find label for Field 3 (Race #) in spec cache. Using default 'race'.")
             race_col_label = RACE_COL
        if race_col_label not in race_data_df.columns:
            logger.warning(f"\nWarning: Race column ('{race_col_label}') not found. Output may lack structure.")
        else:
            original_rows = len(race_data_df)
            if race_col_label in numeric_labels:
                 race_data_df = race_data_df.dropna(subset=[race_col_label])
                 if not race_data_df.empty:
                     race_data_df[race_col_label] = race_data_df[race_col_label].astype(int)
                 rows_after_dropna = len(race_data_df)
                 if rows_after_dropna < original_rows:
                     logger.info(f"\nNote: Removed {original_rows - rows_after_dropna} rows with invalid/missing '{race_col_label}' values.")
            else:
                 logger.warning(f"\nWarning: '{race_col_label}' was not identified as numeric, skipping integrity check.")
        
        #9. Label each race as Sprint vs. Route
        def label_dist(d):
            try: val = float(d)
            except Exception: return np.nan
            return "Sprint" if val <= 1540 else "Route"

        # 10. current race
        if "distance_in_yards" in race_data_df.columns:
            race_data_df["distance_type"] = race_data_df["distance_in_yards"].apply(label_dist)
        else:
            logger.error("No 'distance_in_yards' column found in DataFrame. Cannot create 'distance_type'.")
            # Decide if this is critical. For now, it would raise KeyError later if not handled.
            # To prevent error, you could add: race_data_df["distance_type"] = np.nan
            # or raise a more controlled error here.
            # For now, let's allow the KeyError to be caught by run_pipeline.py's general error handler if it occurs.

        # 11. prior races 1–10
        for i in range(1, 11):
            yard_col = f"distance_in_yards_{i}"
            type_col = f"distance_type_{i}"
            if yard_col in race_data_df.columns: # Check if column exists
                race_data_df[type_col] = race_data_df[yard_col].apply(label_dist)
            else:
                # If column doesn't exist, create it with NaNs or log warning
                logger.warning(f"Column '{yard_col}' not found for past race {i}. '{type_col}' will not be created or will be all NaN.")
                # race_data_df[type_col] = np.nan # Optionally create it as NaN

        # 8. Save the final DataFrame to Parquet
        logger.info(f"\nSaving the processed data to Parquet file: {OUTPUT_PARQUET_FILE_PATH_BRIS}")
        try:
            # Ensure the output directory exists
            OUTPUT_PARQUET_FILE_PATH_BRIS.parent.mkdir(parents=True, exist_ok=True)
            race_data_df.to_parquet(OUTPUT_PARQUET_FILE_PATH_BRIS, index=False, engine='pyarrow')
            logger.info(f"\nSuccessfully saved final data to: {OUTPUT_PARQUET_FILE_PATH_BRIS}")
            logger.info(f"Final DataFrame shape: {race_data_df.shape}")
            logger.info("Final DataFrame Info:")
            # Redirect .info() to logger if possible, or capture and log
            # For simplicity, if run directly, it prints to stdout. If imported, this print might be hidden.
            # race_data_df.info(verbose=False) 
        except ImportError:
            logger.error("\nError: 'pyarrow' library not found. Cannot save to Parquet.")
            logger.error("Please install it: pip install pyarrow")
        except Exception as e:
            logger.error(f"\nError saving final data to Parquet file {OUTPUT_PARQUET_FILE_PATH_BRIS}: {e}", exc_info=True)
    else:
        logger.warning("\nNo data parsed or DataFrame is empty after initial parsing. Cannot save Parquet file.")
    
    logger.info("--- bris_spec_new.main() finished ---")


# --- Main Execution (for direct script run) ---
if __name__ == "__main__":
    # Setup basic logging if this script is run directly
    # This allows you to see logs when testing bris_spec_new.py on its own.
    # It won't interfere if run_pipeline.py has already configured logging.
    if not logging.getLogger().hasHandlers():
        # Basic console logging for direct execution
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
    
    # For direct execution, you might want to define a default DRF file
    # or use a file finding mechanism similar to run_pipeline.py's find_latest_drf_file()
    # For example:
    # default_drf = PROJECT_ROOT_BRIS / "data" / "raw" / "PIM0509.DRF" # Adjust as needed
    # if default_drf.exists():
    #     main(drf_file_path_arg=default_drf)
    # else:
    #     logging.getLogger(__name__).error(f"Default DRF file {default_drf} not found for direct run.")
    # Or, for simplicity when run directly, it might process a fixed file or none if no arg given.
    # The `run_pipeline.py` will be responsible for passing the correct DRF path.
    # If called directly without arguments, the current main() will log an error and return.
    # You can adapt this part as needed for direct testing.
    print("bris_spec_new.py executed directly. Attempting to run main() without DRF argument (will show error or use internal default).")
    main() # This will currently cause an error message from main() if no default is handled internally for drf_file_path_arg