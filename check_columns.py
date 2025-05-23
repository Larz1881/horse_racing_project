import pyarrow.parquet as pq
from pathlib import Path

# --- Configuration ---
SCRIPT_DIR = Path(__file__).parent.resolve() # Or set your path directly
PARQUET_FILENAME = "workouts_long_format.parquet"
PARQUET_FILE_PATH = SCRIPT_DIR / PARQUET_FILENAME
# --- End Configuration ---

if not PARQUET_FILE_PATH.exists():
    print(f"Error: File not found at {PARQUET_FILE_PATH}")
else:
    try:
        # Open the Parquet file and read its schema
        parquet_file = pq.ParquetFile(PARQUET_FILE_PATH)
        schema = parquet_file.schema

        # Get the column names from the schema
        column_names = schema.names

        print(f"Columns found in '{PARQUET_FILENAME}' (using PyArrow schema):")
        print("-" * 30)
        for name in column_names:
            print(name)
        print("-" * 30)
        print(f"Total columns found: {len(column_names)}")

        # --- Verification (same as Method 1) ---
        ID_VARIABLES_TO_CHECK = [
            'race',
            # 'track', # Uncomment if you expect it
            # 'date', # Uncomment if you expect it
            'post_position',
            'horse_name',
            'program_number_if_available'
        ]

        print("\nVerifying your ID_VARIABLES list:")
        missing_ids = []
        found_ids = []
        for var in ID_VARIABLES_TO_CHECK:
            if var in column_names:
                found_ids.append(var)
            else:
                missing_ids.append(var)

        if found_ids:
            print(f"  Found: {found_ids}")
        if missing_ids:
            print(f"  *** MISSING ***: {missing_ids}")
            print("  >>> Please update the ID_VARIABLES list in your script!")
        else:
            print("  All specified ID_VARIABLES were found.")

    except Exception as e:
        print(f"An error occurred while reading the Parquet schema with PyArrow: {e}")