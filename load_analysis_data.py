# -*- coding: utf-8 -*-
"""
Script to load the three analysis-ready Parquet files
(current race info, long-format workouts, long-format past starts)
into separate Pandas DataFrames and perform basic Exploratory Data Analysis (EDA).
"""
import pandas as pd
from pathlib import Path
from typing import Optional, List
import numpy as np
import matplotlib.pyplot as plt # Import for plotting histograms

# --- Configuration ---
# Define paths relative to the script location
SCRIPT_DIR = Path(__file__).parent.resolve()

# Define the filenames
CURRENT_INFO_FILENAME = "current_race_info.parquet"
WORKOUTS_FILENAME = "workouts_long_format.parquet"
PAST_STARTS_FILENAME = "past_starts_long_format.parquet"

CURRENT_INFO_FILE_PATH = SCRIPT_DIR / CURRENT_INFO_FILENAME
WORKOUTS_FILE_PATH = SCRIPT_DIR / WORKOUTS_FILENAME
PAST_STARTS_FILE_PATH = SCRIPT_DIR / PAST_STARTS_FILENAME

# Define key columns for EDA (adjust based on actual column names)
CURRENT_CATEGORICAL_COLS = ['track', 'surface', 'race_type', 'sex', 'today_s_medication', 'equipment_change']
CURRENT_NUMERIC_COLS = ['race', 'distance_in_yards', 'purse', 'claiming_price', 'morn_line_odds', 'weight', 'bris_prime_power_rating']

WORKOUT_CATEGORICAL_COLS = ['work_track', 'work_track_condition', 'work_description', 'work_surface_type']
WORKOUT_NUMERIC_COLS = ['work_time', 'work_distance', 'work_num_at_dist', 'work_rank']

PAST_STARTS_CATEGORICAL_COLS = ['pp_track_code_bris', 'pp_track_condition', 'pp_surface', 'pp_race_type', 'pp_finish_pos'] # Finish pos can be non-numeric
PAST_STARTS_NUMERIC_COLS = ['pp_distance', 'pp_odds', 'pp_bris_speed_rating', 'pp_bris_late_pace', 'pp_days_since_prev', 'pp_purse']

# --- End Configuration ---

def load_parquet_file(file_path: Path, description: str) -> Optional[pd.DataFrame]:
    """
    Loads a single Parquet file into a Pandas DataFrame with error handling.
    """
    if not file_path.exists():
        print(f"Error: {description} file not found at {file_path}")
        return None
    try:
        print(f"Loading {description} data from: {file_path}")
        df = pd.read_parquet(file_path, engine='pyarrow')
        print(f"Successfully loaded {description}. Shape: {df.shape}")
        return df
    except ImportError:
        print("\nError: 'pyarrow' library not found. Cannot load Parquet.")
        print("Please install it: pip install pyarrow")
        return None
    except Exception as e:
        print(f"Error loading {description} file {file_path}: {e}")
        return None

def perform_eda(df: Optional[pd.DataFrame], df_name: str, categorical_cols: List[str], numeric_cols: List[str]):
    """Performs and prints basic EDA for a given DataFrame."""
    if df is None:
        print(f"\nSkipping EDA for {df_name} as DataFrame failed to load.")
        return

    print(f"\n--- EDA for {df_name} ---")
    print(f"Shape: {df.shape}")

    # 1. Basic Info & Dtypes
    print("\n1. Info & Data Types:")
    df.info(verbose=False, show_counts=True) # Concise info

    # 2. Summary Statistics
    print("\n2. Summary Statistics (Numeric & Object):")
    try:
        # Transpose for better readability with many columns
        print(df.describe(include='all').transpose())
    except Exception as e:
        print(f"  Could not generate describe() stats: {e}")


    # 3. Missing Values Analysis
    print("\n3. Missing Values (Top 15 columns with missing data):")
    missing_counts = df.isnull().sum()
    missing_percent = (missing_counts / len(df)) * 100
    missing_df = pd.DataFrame({'count': missing_counts, 'percentage': missing_percent})
    missing_df = missing_df[missing_df['count'] > 0].sort_values(by='percentage', ascending=False)
    if not missing_df.empty:
        print(missing_df.head(15))
    else:
        print("  No missing values found.")

    # 4. Value Counts for Key Categorical Columns
    print("\n4. Value Counts for Key Categorical Columns:")
    actual_categorical = [col for col in categorical_cols if col in df.columns]
    if actual_categorical:
        for col in actual_categorical:
            print(f"\n  Column: '{col}'")
            try:
                print(df[col].value_counts(dropna=False)) # Include NaN counts
            except Exception as e:
                print(f"    Could not get value counts: {e}")
    else:
        print("  No specified categorical columns found in DataFrame.")

    # 5. Distributions for Key Numeric Columns (Histograms)
    print("\n5. Distributions for Key Numeric Columns:")
    actual_numeric = [col for col in numeric_cols if col in df.columns]
    if actual_numeric:
        num_plots = len(actual_numeric)
        # Simple subplot layout (adjust rows/cols as needed)
        ncols = 3
        nrows = (num_plots + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))
        axes = axes.flatten() # Flatten to easily iterate

        for i, col in enumerate(actual_numeric):
            ax = axes[i]
            try:
                df[col].hist(ax=ax, bins=20) # Use pandas histogram on the subplot axis
                ax.set_title(f"Distribution of '{col}'")
                ax.set_xlabel(col)
                ax.set_ylabel("Frequency")
            except TypeError:
                 print(f"  Skipping histogram for '{col}' due to non-numeric data.")
            except Exception as e:
                print(f"  Could not plot histogram for '{col}': {e}")

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle(f'Histograms for Key Numeric Columns in {df_name}', fontsize=16, y=1.02)
        plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout
        # Consider saving the plot instead of showing if running non-interactively
        # plt.savefig(f"eda_histograms_{df_name}.png")
        # plt.close(fig) # Close the figure after saving
        print(f"  Histograms generated (display may depend on environment).")

    else:
        print("  No specified numeric columns found in DataFrame for histograms.")

    print(f"--- End EDA for {df_name} ---")


# --- Main Loading and EDA Section ---
if __name__ == "__main__":
    print("--- Loading Analysis DataFrames ---")
    df_current = load_parquet_file(CURRENT_INFO_FILE_PATH, "Current Race Info")
    df_workouts = load_parquet_file(WORKOUTS_FILE_PATH, "Workouts (Long Format)")
    df_past_starts = load_parquet_file(PAST_STARTS_FILE_PATH, "Past Starts (Long Format)")
    print("-" * 50)

    # --- Perform EDA ---
    perform_eda(df_current, "Current Race Info", CURRENT_CATEGORICAL_COLS, CURRENT_NUMERIC_COLS)
    print("-" * 50)
    perform_eda(df_workouts, "Workouts (Long Format)", WORKOUT_CATEGORICAL_COLS, WORKOUT_NUMERIC_COLS)
    print("-" * 50)
    perform_eda(df_past_starts, "Past Starts (Long Format)", PAST_STARTS_CATEGORICAL_COLS, PAST_STARTS_NUMERIC_COLS)
    print("-" * 50)

    # --- Verification ---
    if df_current is not None and df_workouts is not None and df_past_starts is not None:
        print("\nAll three DataFrames loaded successfully! EDA complete.")
        # Now you can proceed with merging and analysis using:
        # df_current
        # df_workouts
        # df_past_starts
    else:
        print("\nOne or more DataFrames failed to load. EDA skipped for missing DataFrames.")

    print("\n--- Loading & EDA Script Finished ---")
    



