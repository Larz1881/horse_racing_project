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
import re # For parsing bris_dict.txt

# --- Configuration ---

# 1. Define paths relative to the script location
SCRIPT_DIR: Final[Path] = Path(__file__).parent.resolve()

# 2. Input files  
SPEC_CACHE_FILENAME: Final[str] = "bris_spec.pkl" # Assumes this maps Field# -> 'label'
BRIS_DICT_FILENAME: Final[str] = "bris_dict.txt" # The spec file itself
RACE_DATA_FILENAME: Final[str] = "PIM0509.DRF" # Or your specific data filename

SPEC_CACHE_FILE_PATH: Final[Path] = SCRIPT_DIR / SPEC_CACHE_FILENAME
BRIS_DICT_FILE_PATH: Final[Path] = SCRIPT_DIR / BRIS_DICT_FILENAME
RACE_DATA_FILE_PATH: Final[Path] = SCRIPT_DIR / RACE_DATA_FILENAME

# 3. Output file name for the full, parsed data in Parquet format
OUTPUT_PARQUET_FILENAME: Final[str] = "parsed_race_data_full.parquet"
OUTPUT_PARQUET_FILE_PATH: Final[Path] = SCRIPT_DIR / OUTPUT_PARQUET_FILENAME

# --- End Configuration ---

def load_specification_cache(spec_cache_path: Path) -> Optional[pd.DataFrame]:
    """Loads the column label specification DataFrame from the cache file."""
    if not spec_cache_path.exists():
        print(f"Error: Specification cache file not found at {spec_cache_path}")
        print("Please ensure bris_spec.pkl exists (or run the script that generates it).")
        return None
    try:
        print(f"Loading specification cache from: {spec_cache_path}")
        spec_df = pd.read_pickle(spec_cache_path)
        spec_df = spec_df.sort_index() # Assume index is Field Number
        if 'label' not in spec_df.columns:
             print("Error: Spec cache DataFrame must contain a 'label' column.")
             return None
        if not spec_df['label'].is_unique:
             print("Error: Labels loaded from cache are not unique!")
             return None
        # Add field number as a column if it's not the index
        if spec_df.index.name != 'field_number':
            spec_df['field_number'] = spec_df.index
        return spec_df
    except Exception as e:
        print(f"Error loading specification cache file: {e}")
        return None

def parse_bris_dict_types(dict_path: Path) -> Dict[int, str]:
    """
    Parses bris_dict.txt primarily to get the declared TYPE for each Field #.
    Returns a dictionary mapping {field_number: 'TYPE'}.
    Handles specific fields known to represent dates despite CHARACTER type.
    """
    field_types: Dict[int, str] = {}
    # Regex captures Field #, Description (non-greedy), and Type (CHARACTER/NUMERIC/DATE)
    # It handles potential extra spaces and requires at least two spaces before TYPE
    pattern = re.compile(r"^\s*(\d+)\s+(.*?)\s{2,}(CHARACTER|NUMERIC|DATE)\s+", re.MULTILINE)

    print(f"Parsing field types from: {dict_path}")
    if not dict_path.exists():
        print(f"Error: {dict_path} not found.")
        return field_types

    try:
        with open(dict_path, 'r', encoding='iso-8859-1') as f: # Use common encoding
            content = f.read()

        for match in pattern.finditer(content):
            field_num = int(match.group(1))
            # desc = match.group(2).strip() # Description not needed here
            type_ = match.group(3).strip()
            field_types[field_num] = type_

        # --- Manual Overrides/Refinements for Date Fields ---
        # Field 2 (Date) is CHARACTER but should be treated as Date
        if 2 in field_types and field_types[2] == 'CHARACTER':
            field_types[2] = 'DATE_STR'
        # Fields 102-113 (Workout Date) - Field 102 is DATE, others CHARACTER
        # Mark all as DATE_STR for consistent handling
        for fn in range(102, 114):
            field_types[fn] = 'DATE_STR'
        # Fields 256-265 (Past Race Date) are CHARACTER but represent dates
        for fn in range(256, 266):
            field_types[fn] = 'DATE_STR'

        print(f"Parsed types for {len(field_types)} fields from {dict_path.name}.")

    except Exception as e:
        print(f"Error parsing {dict_path.name}: {e}")

    return field_types


def identify_column_types_from_spec(spec_df: pd.DataFrame, field_type_map: Dict[int, str]) -> Tuple[Set[str], Set[str]]:
    """
    Identifies numeric and date columns based on the parsed field types from bris_dict.txt.

    Args:
        spec_df: DataFrame loaded from bris_spec.pkl, mapping Field# to 'label'.
                 Must contain 'field_number' and 'label' columns.
        field_type_map: Dictionary mapping Field# to parsed 'TYPE' ('NUMERIC', 'CHARACTER', 'DATE_STR').

    Returns:
        Tuple containing (set_of_numeric_labels, set_of_date_labels).
    """
    numeric_cols: Set[str] = set()
    date_cols: Set[str] = set()
    print("\nIdentifying column types using parsed bris_dict.txt info...")

    if 'field_number' not in spec_df.columns or 'label' not in spec_df.columns:
         print("Error: spec_df DataFrame must contain 'field_number' and 'label' columns.")
         return numeric_cols, date_cols

    if not field_type_map:
        print("Warning: Field type map is empty. Cannot reliably identify types.")
        return numeric_cols, date_cols

    for _, row in spec_df.iterrows():
        # Use .get for safer access in case columns are missing
        field_num = row.get('field_number')
        label = row.get('label')

        if label is None or field_num is None or label.startswith("reserved"):
            continue

        # Get the parsed type for this field number
        field_type = field_type_map.get(field_num)

        if field_type == 'NUMERIC':
            numeric_cols.add(label)
        elif field_type == 'DATE_STR': # Use our custom type for date-like strings/fields
            date_cols.add(label)
        # Add more checks here if needed, e.g., for specific CHARACTER fields
        # that might represent numbers (though NUMERIC type should cover most)

    print(f"Identified {len(numeric_cols)} numeric columns using spec.")
    print(f"Identified {len(date_cols)} date columns using spec.")
    return numeric_cols, date_cols


def parse_brisnet_csv_data(
    data_file_path: Path,
    column_names_all: List[str] # Takes all names initially
) -> Optional[pd.DataFrame]:
    """
    Parses a comma-delimited Brisnet data file using provided column names,
    excluding only columns marked as "reserved". Reads all as string initially.
    """
    # (This function remains the same as the previous version)
    if not data_file_path.exists():
        print(f"Error: Race data file not found at {data_file_path}")
        print(f"Looked for: {data_file_path.resolve()}")
        return None
    column_names_filtered = [ name for name in column_names_all if not name.startswith("reserved") ]
    cols_to_use_indices = [ i for i, name in enumerate(column_names_all) if not name.startswith("reserved") ]
    if not column_names_filtered:
        print("Error: No non-reserved columns found in the specification.")
        return None
    print(f"\nAttempting to parse {data_file_path} as comma-delimited...")
    print(f"Excluding columns starting with 'reserved'. Using {len(column_names_filtered)} out of {len(column_names_all)} columns.")
    try:
        df = pd.read_csv(
            data_file_path, sep=',', header=None, names=column_names_filtered,
            usecols=cols_to_use_indices, dtype=str, quotechar='"',
            on_bad_lines='warn', low_memory=False, encoding='iso-8859-1',
            skipinitialspace=True
        )
        print(f"Parsing successful. Read {len(df)} lines.")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {data_file_path}")
        return None
    except pd.errors.ParserError as e:
         print(f"Error parsing CSV file: {e}")
         print("Check file format, encoding, and separators.")
         return None
    except Exception as e:
        print(f"An unexpected error occurred during parsing: {e}")
        return None


def convert_data_types(df: pd.DataFrame, numeric_cols_labels: Set[str], date_cols_labels: Set[str]) -> pd.DataFrame:
    """Attempts to convert columns to appropriate numeric and date types."""
    # (This function remains the same as the previous version)
    print("\nAttempting data type conversions...")
    df_copy = df.copy()
    # --- Numeric Conversion ---
    numeric_cols_present = [col for col in numeric_cols_labels if col in df_copy.columns]
    print(f"Converting {len(numeric_cols_present)} columns to numeric...")
    converted_numeric_count = 0
    for col in numeric_cols_present:
        try:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
            converted_numeric_count += 1
        except Exception as e:
            print(f"  - Warning: Failed to convert column '{col}' to numeric: {e}")
    print(f"Successfully converted {converted_numeric_count} numeric columns.")
    # --- Date Conversion ---
    date_cols_present = [col for col in date_cols_labels if col in df_copy.columns]
    print(f"Converting {len(date_cols_present)} columns to datetime...")
    converted_date_count = 0
    brisnet_date_format = '%Y%m%d' # Brisnet dates are often YYYYMMDD
    for col in date_cols_present:
        try:
            df_copy[col] = pd.to_datetime(df_copy[col], format=brisnet_date_format, errors='coerce')
            converted_date_count += 1
        except Exception as e:
            # Fallback for potentially different date formats if needed
            try:
                 print(f"  - Retrying datetime conversion for '{col}' without specific format.")
                 df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
            except Exception as e2:
                 print(f"  - Warning: Failed to convert column '{col}' to datetime: {e2}")
    print(f"Successfully converted {converted_date_count} date columns.")
    return df_copy


# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load the column name specification cache (Field# -> Label)
    spec_df = load_specification_cache(SPEC_CACHE_FILE_PATH)
    if spec_df is None:
        exit()

    # 2. Parse the bris_dict.txt file to get Field# -> Type mapping
    field_type_map = parse_bris_dict_types(BRIS_DICT_FILE_PATH)
    if not field_type_map:
        print("Could not parse field types from bris_dict.txt. Type conversion may be inaccurate.")
        # Optionally exit or proceed with caution
        # exit()

    # 3. Extract the full ordered list of labels (column names) from the cache
    all_ordered_column_labels = spec_df['label'].tolist()

    # 4. Parse the actual race data file, excluding only reserved columns
    race_data_df = parse_brisnet_csv_data(RACE_DATA_FILE_PATH, all_ordered_column_labels)

    if race_data_df is not None and not race_data_df.empty:

        # 5. Identify numeric and date columns using the parsed spec info
        numeric_labels, date_labels = identify_column_types_from_spec(spec_df, field_type_map)

        # 6. Perform Type Conversions
        race_data_df = convert_data_types(race_data_df, numeric_labels, date_labels)

        # 7. Basic Integrity Check (ensure 'race' column exists and is valid)
        RACE_COL = 'race' # Assuming 'race' is the label for Field 3
        # Find the actual label for Field 3 from spec_df if necessary
        try:
             race_col_label = spec_df[spec_df['field_number'] == 3]['label'].iloc[0]
        except (KeyError, IndexError):
             print("Warning: Could not find label for Field 3 (Race #) in spec cache. Using default 'race'.")
             race_col_label = RACE_COL # Fallback

        if race_col_label not in race_data_df.columns:
            print(f"\nWarning: Race column ('{race_col_label}') not found. Output may lack structure.")
        else:
            original_rows = len(race_data_df)
            # Ensure it was identified as numeric before attempting conversion/dropna
            if race_col_label in numeric_labels:
                 # Drop rows where race number couldn't be converted to numeric
                 race_data_df = race_data_df.dropna(subset=[race_col_label])
                 # Convert valid ones to integer
                 if not race_data_df.empty:
                     race_data_df[race_col_label] = race_data_df[race_col_label].astype(int)
                 rows_after_dropna = len(race_data_df)
                 if rows_after_dropna < original_rows:
                     print(f"\nNote: Removed {original_rows - rows_after_dropna} rows with invalid/missing '{race_col_label}' values for basic integrity.")
            else:
                 print(f"\nWarning: '{race_col_label}' was not identified as numeric based on spec, skipping integrity check based on it.")

        #9. Label each race as Sprint vs. Route
        def label_dist(d):
            # coerce to float; return NaN if conversion fails
            try:
                val = float(d)
            except Exception:
                return np.nan
            return "Sprint" if val <= 1540 else "Route"

        # 10. current race
        if "distance_in_yards" in race_data_df.columns:
            race_data_df["distance_type"] = race_data_df["distance_in_yards"].apply(label_dist)
        else:
            raise KeyError("No 'distance_in_yards' column found in DataFrame")

        # 11. prior races 1–10
        for i in range(1, 11):
            yard_col = f"distance_in_yards_{i}"
            type_col = f"distance_type_{i}"
            if yard_col not in race_data_df.columns:
                raise KeyError(f"No column named '{yard_col}'")
            race_data_df[type_col] = race_data_df[yard_col].apply(label_dist)

        # 8. Save the final DataFrame to Parquet
        print(f"\nSaving the processed data to Parquet file: {OUTPUT_PARQUET_FILE_PATH}")
        try:
            race_data_df.to_parquet(OUTPUT_PARQUET_FILE_PATH, index=False, engine='pyarrow')
            print(f"\nSuccessfully saved final data to: {OUTPUT_PARQUET_FILE_PATH}")
            print(f"Final DataFrame shape: {race_data_df.shape}")
            #print(f"Final DataFrame Memory Usage: {race_data_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB") # Can be slow
            print("Final DataFrame Info:")
            race_data_df.info(verbose=False) # Print summary info

        except ImportError:
            print("\nError: 'pyarrow' library not found. Cannot save to Parquet.")
            print("Please install it: pip install pyarrow")
        except Exception as e:
            print(f"\nError saving final data to Parquet file {OUTPUT_PARQUET_FILE_PATH}: {e}")

    else:
        print("\nNo data parsed or DataFrame  is empty after initial parsing. Cannot save Parquet file.")



