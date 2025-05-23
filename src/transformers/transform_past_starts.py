# -*- coding: utf-8 -*-
"""
Transforms Past Performance history data (last 10 races) from a "wide"
format into a "long" format (one row per past start per horse).

It uses the Brisnet specification (via bris_spec.pkl) to map field
numbers to the actual column names found in the wide data file. This
allows reshaping even with potentially inconsistent column naming.

Reads:
- parsed_race_data_full.parquet (output of the modified bris_spec.py)
- bris_spec.pkl (cache mapping Brisnet Field# to actual column labels)

Outputs:
- past_starts_long_format.parquet (long format past performance data)
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Any, Final, Optional, Set
import numpy as np # For potential numeric cleaning

# --- Configuration ---

# 1. Define paths relative to the script location
SCRIPT_DIR: Final[Path] = Path(__file__).parent.resolve()

# 2. Input files
WIDE_DATA_PARQUET_FILENAME: Final[str] = "parsed_race_data_full.parquet"
SPEC_CACHE_FILENAME: Final[str] = "bris_spec.pkl"

WIDE_DATA_FILE_PATH: Final[Path] = SCRIPT_DIR / WIDE_DATA_PARQUET_FILENAME
SPEC_CACHE_FILE_PATH: Final[Path] = SCRIPT_DIR / SPEC_CACHE_FILENAME

# 3. Output file for the transformed "long" past performance data
LONG_PAST_STARTS_FILENAME: Final[str] = "past_starts_long_format.parquet"
LONG_PAST_STARTS_FILE_PATH: Final[Path] = SCRIPT_DIR / LONG_PAST_STARTS_FILENAME

# 4. Define ID variables: Columns identifying the *current* horse/race entry.
#    *** Based on your confirmation ***
ID_VARIABLES: Final[List[str]] = [
    'track', # Today's track code
    'race', # Today's race number
    # 'date', # Add if available and needed as part of the unique ID
    'post_position', # Today's post position
    'horse_name' # Today's horse name
]

# 5. Define the mapping from desired clean metric names to the
#    Brisnet Field Number of the *first* past race (#1) for that metric.
#    *** Carefully review and complete this map based on bris_dict.txt ***
PAST_RACE_METRIC_MAP: Final[Dict[str, int]] = {
    'pp_race_date': 256,                 # Past Perf Race Date
    'pp_days_since_prev': 266,          # Note: Field 275 is reserved
    'pp_track_code': 276,               # Track Code (full name)
    'pp_track_code_bris': 286,          # BRIS Track Code (3-char)
    'pp_race_num': 296,                 # Past Race Number
    'pp_track_condition': 306,
    'pp_distance': 316,                 # In yards
    'pp_surface': 326,
    'pp_chute_indicator': 336,
    'pp_num_entrants': 346,
    'pp_post_position': 356,
    'pp_equipment': 366,
    'pp_racename': 376,
    'pp_medication': 386,
    'pp_trip_comment': 396,
    'pp_winner_name': 406,
    'pp_second_name': 416,
    'pp_third_name': 426,
    'pp_winner_weight': 436,
    'pp_second_weight': 446,
    'pp_third_weight': 456,
    'pp_winner_margin': 466,
    'pp_second_margin': 476,
    'pp_third_margin': 486,
    'pp_extra_comment': 496,
    'pp_weight_carried': 506,
    'pp_odds': 516,
    'pp_entry_indicator': 526,
    'pp_race_classification': 536,
    'pp_claiming_price': 546,
    'pp_purse': 556,
    'pp_start_call_pos': 566,
    'pp_first_call_pos': 576,
    'pp_second_call_pos': 586,
    'pp_gate_call_pos': 596,
    'pp_stretch_pos': 606,
    'pp_finish_pos': 616,               # Official Finish Position
    'pp_money_pos': 626,                # Position for money
    'pp_start_lengths_leader': 636,
    'pp_start_lengths_behind': 646,
    'pp_first_call_lengths_leader': 656,
    'pp_first_call_lengths_behind': 666,
    'pp_second_call_lengths_leader': 676,
    'pp_second_call_lengths_behind': 686,
    'pp_bris_shape_1st_call': 696,      # Note: 706-715 reserved
    'pp_stretch_lengths_leader': 716,
    'pp_stretch_lengths_behind': 726,
    'pp_finish_lengths_winner': 736,
    'pp_finish_lengths_behind': 746,
    'pp_bris_shape_2nd_call': 756,
    'pp_bris_2f_pace': 766,
    'pp_bris_4f_pace': 776,
    'pp_bris_6f_pace': 786,
    'pp_bris_8f_pace': 796,
    'pp_bris_10f_pace': 806,
    'pp_bris_late_pace': 816,           # Note: 826-845 reserved
    'pp_bris_speed_rating': 846,
    'pp_speed_rating_alt': 856,         # Alternative Speed Rating
    'pp_track_variant': 866,
    'pp_frac_2f': 876,
    'pp_frac_3f': 886,
    'pp_frac_4f': 896,
    'pp_frac_5f': 906,
    'pp_frac_6f': 916,
    'pp_frac_7f': 926,
    'pp_frac_8f': 936,
    'pp_frac_10f': 946,
    'pp_frac_12f': 956,
    'pp_frac_14f': 966,
    'pp_frac_16f': 976,
    'pp_fraction_1': 986,              # Generic Fraction 1
    'pp_fraction_2': 996,              # Generic Fraction 2
    'pp_fraction_3': 1006,             # Generic Fraction 3 (Note 1016-1035 reserved)
    'pp_final_time': 1036,
    'pp_claimed_code': 1046,
    'pp_trainer': 1056,
    'pp_jockey': 1066,
    'pp_apprentice_allow': 1076,
    'pp_race_type': 1086,
    'pp_age_sex_restrict': 1096,
    'pp_statebred_flag': 1106,
    'pp_restricted_flag': 1116,
    'pp_favorite_indicator': 1126,
    'pp_front_bandages': 1136,          # Note: 1146 reserved
    'pp_bris_speed_par': 1167,          # Speed Par for class level of past race
    'pp_bar_shoe': 1182,
    'pp_company_line': 1192,
    'pp_low_claiming': 1202,            # Low claiming price of past race
    'pp_high_claiming': 1212,           # High claiming price of past race
    'pp_misc_code': 1254,               # Misc code (e.g., nasal strip) for past race
    'pp_ext_start_comment': 1383,       # Extended Start Comment
    'pp_sealed_track': 1393,            # Sealed Track Indicator
    'pp_aw_surface_flag': 1403,         # All-Weather Surface Flag
    'pp_eqb_conditions': 1419,          # Equibase Abbrev Conditions
}

def label_dist(d):
    """Return 'Sprint' if distance ≤ 1540 yards, else 'Route'."""
    try:
        val = float(d)
    except (ValueError, TypeError):
        return np.nan
    return "Sprint" if val <= 1540 else "Route"

# --- End Configuration ---

# --- Helper Functions ---
# Reuse load_data from previous script
def load_data(parquet_path: Path, pkl_path: Path) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Loads the wide Parquet data and the spec cache PKL."""
    # (Same implementation as in transform_workouts.py)
    wide_df: Optional[pd.DataFrame] = None
    spec_df: Optional[pd.DataFrame] = None
    # Load Wide Data
    if not parquet_path.exists(): print(f"Error: Input Parquet file not found at {parquet_path}"); return None, None
    try:
        print(f"Loading wide format data from: {parquet_path}"); wide_df = pd.read_parquet(parquet_path, engine='pyarrow'); print(f"Loaded wide data with shape: {wide_df.shape}")
    except Exception as e: print(f"Error loading Parquet file {parquet_path}: {e}"); return None, None
    # Load Spec Cache
    if not pkl_path.exists(): print(f"Error: Specification cache file not found at {pkl_path}"); return wide_df, None
    try:
        print(f"Loading specification cache from: {pkl_path}"); spec_df = pd.read_pickle(pkl_path)
        if 'field_number' not in spec_df.columns and spec_df.index.name != 'field_number': spec_df['field_number'] = spec_df.index
        if 'label' not in spec_df.columns: print(f"Error: Spec cache '{pkl_path.name}' must contain a 'label' column."); return wide_df, None
        if 'field_number' in spec_df.columns and spec_df.index.name != 'field_number': spec_df = spec_df.set_index('field_number', drop=False)
        if spec_df.index.name != 'field_number': print("Warning: Spec cache index is not named 'field_number'.")
        print(f"Loaded spec cache for {len(spec_df)} fields.")
    except Exception as e: print(f"Error loading specification cache file {pkl_path}: {e}"); return wide_df, None
    return wide_df, spec_df


def clean_past_starts_data(df: pd.DataFrame) -> pd.DataFrame:
    """Applies specific cleaning and type conversion to long past starts data."""
    print("Cleaning and converting types in long past starts data...")
    if df.empty: return df
    df = df.copy()

    # Identify potential date, numeric, object columns based on clean names
    date_cols = [col for col in df.columns if 'date' in col.lower()] # Simple check
    numeric_cols = [
        col for col in df.columns if any(k in col.lower() for k in [
            'days', 'distance', 'entrants', 'position', 'weight', 'margin',
            'odds', 'claiming', 'purse', 'lengths', 'shape', 'pace', 'speed',
            'variant', 'frac', 'time', 'allow', 'indic', 'par', 'low', 'high'
        ]) and col not in date_cols
    ]

    # --- Ensure pp_distance is numeric ---
    if 'pp_distance' in df.columns:
        df['pp_distance'] = pd.to_numeric(df['pp_distance'], errors='coerce')
    else:
        print("  Warning: 'pp_distance' column not found during cleaning.")
        df['pp_distance'] = np.nan

    # Add specific known numeric cols if missed
    numeric_cols.extend(['pp_race_num', 'pp_medication', 'morn_line_odds_if_available', 'pp_purse']) # Medication is numeric code
    numeric_cols = list(set(numeric_cols)) # Unique list

    # Convert dates
    if 'pp_race_date' in df.columns:
        df['pp_race_date'] = pd.to_datetime(df['pp_race_date'], format='%Y%m%d', errors='coerce')
        original_rows = len(df)
        df.dropna(subset=['pp_race_date'], inplace=True) # Drop starts without a valid date
        if len(df) < original_rows: print(f"  Dropped {original_rows - len(df)} rows with invalid pp_race_date.")

    # Convert numeric
    print(f"Attempting numeric conversion for {len(numeric_cols)} columns...")
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Convert position/finish fields to numeric where possible (handle non-numeric like 'F', 'E')
    pos_cols = [col for col in df.columns if 'pos' in col.lower() or 'finish' in col.lower()]
    for col in pos_cols:
         if col in df.columns and df[col].dtype == 'object': # Only try if not already numeric
              df[col] = pd.to_numeric(df[col], errors='coerce') # Coerce 'F', 'E' etc. to NaN

    # Ensure past_race_num is integer
    if 'past_race_num' in df.columns:
        if df['past_race_num'].notna().all():
            df['past_race_num'] = df['past_race_num'].astype(int)

    print(f"Past starts data cleaned. Shape after cleaning: {df.shape}")
    return df

def calculate_splits_for_long_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates split times from cumulative fractional times in a long-format DataFrame.
    Assumes fractional columns like 'pp_frac_2f', 'pp_frac_4f', etc., already exist
    and have been converted to numeric types (e.g., seconds).
    """
    print("Calculating split times for long-format past performance data...")

    # Map of shorthand -> column name in your PAST_RACE_METRIC_MAP clean names
    cumulative_time_cols = {
        'c2': 'pp_frac_2f', 'c4': 'pp_frac_4f', 'c6': 'pp_frac_6f',
        'c8': 'pp_frac_8f', 'c10': 'pp_frac_10f', 'c12': 'pp_frac_12f',
        'c5': 'pp_frac_5f', 'c7': 'pp_frac_7f', 'c9': 'pp_frac_9f',
        'final': 'pp_final_time'
    }

    # Ensure cols are numeric
    for key, col in cumulative_time_cols.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            print(f"  Warning: '{col}' not found—skipping.")

    # Standard even-furlong splits
    if 'pp_frac_2f' in df: df['pp_split_0_2f_secs'] = df['pp_frac_2f']
    if all(c in df for c in ('pp_frac_4f','pp_frac_2f')):
        df['pp_split_2f_4f_secs'] = df['pp_frac_4f'] - df['pp_frac_2f']
    if all(c in df for c in ('pp_frac_6f','pp_frac_4f')):
        df['pp_split_4f_6f_secs'] = df['pp_frac_6f'] - df['pp_frac_4f']
    if all(c in df for c in ('pp_frac_8f','pp_frac_6f')):
        df['pp_split_6f_8f_secs'] = df['pp_frac_8f'] - df['pp_frac_6f']
    if all(c in df for c in ('pp_frac_10f','pp_frac_8f')):
        df['pp_split_8f_10f_secs'] = df['pp_frac_10f'] - df['pp_frac_8f']
    if all(c in df for c in ('pp_frac_12f','pp_frac_10f')):
        df['pp_split_10f_12f_secs'] = df['pp_frac_12f'] - df['pp_frac_10f']

    # Split to finish (last even furlong to final time)
    if 'pp_final_time' in df.columns:
        if 'pp_frac_10f' in df.columns:
            df['pp_split_10f_finish_secs'] = df['pp_final_time'] - df['pp_frac_10f']
        elif 'pp_frac_8f' in df.columns:
            df['pp_split_8f_finish_secs']  = df['pp_final_time'] - df['pp_frac_8f']
        elif 'pp_frac_6f' in df.columns:
            df['pp_split_6f_finish_secs']  = df['pp_final_time'] - df['pp_frac_6f']
        elif 'pp_frac_4f' in df.columns:
            df['pp_split_4f_finish_secs']  = df['pp_final_time'] - df['pp_frac_4f']
        elif 'pp_frac_2f' in df.columns:
            df['pp_split_2f_finish_secs']  = df['pp_final_time'] - df['pp_frac_2f']

    # Odd-furlong splits if you have them
    if all(c in df for c in ('pp_frac_5f','pp_frac_4f')):
        df['pp_split_4f_5f_secs'] = df['pp_frac_5f'] - df['pp_frac_4f']
    if all(c in df for c in ('pp_frac_7f','pp_frac_6f')):
        df['pp_split_6f_7f_secs'] = df['pp_frac_7f'] - df['pp_frac_6f']
    if all(c in df for c in ('pp_frac_9f','pp_frac_8f')):
        df['pp_split_8f_9f_secs'] = df['pp_frac_9f'] - df['pp_frac_8f']

    print("Split time calculation complete.")
    return df

def calculate_position_changes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates positions gained/lost between points of call.
    Assumes call-position columns have been converted to numeric.
    """
    print("Calculating position changes between calls...")

    pos_col_map = {
        'start':   'pp_start_call_pos',
        'c1':      'pp_first_call_pos',
        'c2':      'pp_second_call_pos',
        'stretch': 'pp_stretch_pos',
        'finish':  'pp_finish_pos'
    }

    # Ensure numeric dtype (or coerce)
    for key, col in pos_col_map.items():
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                print(f"  Warning: '{col}' not numeric—coercing.")
                df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            print(f"  Warning: '{col}' missing—filling with NaN.")
            df[col] = np.nan

    # Compute position-gains (prev_pos – curr_pos)
    if 'pp_start_call_pos' in df and 'pp_first_call_pos' in df:
        df['pp_pos_gain_start_to_c1'] = df['pp_start_call_pos'] - df['pp_first_call_pos']
    if 'pp_first_call_pos' in df and 'pp_second_call_pos' in df:
        df['pp_pos_gain_c1_to_c2']    = df['pp_first_call_pos'] - df['pp_second_call_pos']
    if 'pp_second_call_pos' in df and 'pp_stretch_pos' in df:
        df['pp_pos_gain_c2_to_stretch'] = df['pp_second_call_pos'] - df['pp_stretch_pos']
    if 'pp_stretch_pos' in df and 'pp_finish_pos' in df:
        df['pp_pos_gain_stretch_to_finish'] = df['pp_stretch_pos'] - df['pp_finish_pos']
    if 'pp_start_call_pos' in df and 'pp_finish_pos' in df:
        df['pp_pos_gain_start_to_finish'] = df['pp_start_call_pos'] - df['pp_finish_pos']

    print("Position change calculation complete.")
    return df

def calculate_lengths_behind_changes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the change in lengths behind the leader between points of call.
    Assumes lengths-behind columns have been converted to numeric.
    """
    print("Calculating changes in lengths behind leader between calls...")

    lengths_col_map = {
        'start':   'pp_start_lengths_leader',
        'c1':      'pp_first_call_lengths_leader',
        'c2':      'pp_second_call_lengths_leader',
        'stretch': 'pp_stretch_lengths_leader',
        'finish':  'pp_finish_lengths_winner'
    }

    # Defensive numeric conversion
    for key, col in lengths_col_map.items():
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                print(f"  Warning: '{col}' not numeric—coercing.")
                df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            print(f"  Warning: '{col}' missing—filling with NaN.")
            df[col] = np.nan

    # Compute change in lengths behind leader (later – earlier)
    if 'pp_start_lengths_leader' in df and 'pp_first_call_lengths_leader' in df:
        df['pp_chg_len_ldr_start_to_c1'] = (
            df['pp_first_call_lengths_leader'] - df['pp_start_lengths_leader']
        )
    if 'pp_first_call_lengths_leader' in df and 'pp_second_call_lengths_leader' in df:
        df['pp_chg_len_ldr_c1_to_c2'] = (
            df['pp_second_call_lengths_leader'] - df['pp_first_call_lengths_leader']
        )
    if 'pp_second_call_lengths_leader' in df and 'pp_stretch_lengths_leader' in df:
        df['pp_chg_len_ldr_c2_to_stretch'] = (
            df['pp_stretch_lengths_leader'] - df['pp_second_call_lengths_leader']
        )
    if 'pp_stretch_lengths_leader' in df and 'pp_finish_lengths_winner' in df:
        df['pp_chg_len_ldr_stretch_to_finish'] = (
            df['pp_finish_lengths_winner'] - df['pp_stretch_lengths_leader']
        )
    if 'pp_start_lengths_leader' in df and 'pp_finish_lengths_winner' in df:
        df['pp_chg_len_ldr_start_to_finish'] = (
            df['pp_finish_lengths_winner'] - df['pp_start_lengths_leader']
        )

    print("Change in lengths behind leader calculation complete.")
    return df


def calculate_early_late_pace_figures(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates E1, E2, Turn Time, and Combined Pace figures based on race type
    and BRIS pace figures for each past performance.
    """
    print("Calculating E1, E2, Turn Time, and Combined Pace figures...")

    pace_cols = {
        '2f':   'pp_bris_2f_pace',
        '4f':   'pp_bris_4f_pace',
        '6f':   'pp_bris_6f_pace',
        'late': 'pp_bris_late_pace'
    }

    # Defensive numeric conversion
    for key, col in pace_cols.items():
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                print(f"  Warning: '{col}' not numeric—coercing.")
                df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            print(f"  Warning: '{col}' missing—filling with NaN.")
            df[col] = np.nan

    if 'pp_distance_type' not in df.columns:
        print("  Critical: 'pp_distance_type' missing—skipping pace calculations.")
        df['pp_e1_pace']        = np.nan
        df['pp_e2_pace']        = np.nan
        df['pp_turn_time']      = np.nan
        df['pp_combined_pace']  = np.nan
        return df

    # Initialize
    df['pp_e1_pace']       = np.nan
    df['pp_e2_pace']       = np.nan

    sprint = df['pp_distance_type'] == 'Sprint'
    route  = df['pp_distance_type'] == 'Route'

    # E1: to 1st call
    if 'pp_bris_2f_pace' in df:
        df.loc[sprint, 'pp_e1_pace'] = df.loc[sprint, 'pp_bris_2f_pace']
    if 'pp_bris_4f_pace' in df:
        df.loc[route,  'pp_e1_pace'] = df.loc[route,  'pp_bris_4f_pace']

    # E2: to 2nd call
    if 'pp_bris_4f_pace' in df:
        df.loc[sprint, 'pp_e2_pace'] = df.loc[sprint, 'pp_bris_4f_pace']
    if 'pp_bris_6f_pace' in df:
        df.loc[route,  'pp_e2_pace'] = df.loc[route,  'pp_bris_6f_pace']

    # Turn Time
    df['pp_turn_time'] = df['pp_e2_pace'] - df['pp_e1_pace']

    # Combined Pace = E2 + Late
    if 'pp_bris_late_pace' in df:
        df['pp_combined_pace'] = df['pp_e2_pace'] + df['pp_bris_late_pace']
    else:
        df['pp_combined_pace'] = np.nan

    print("Pace figure calculations complete.")
    return df

def calculate_avg_best2_bris_speed(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each row, takes the two largest of
      'bris_speed_rating_1', 'bris_speed_rating_2', 'bris_speed_rating_3'
    and returns their average.  If fewer than two non-null values exist,
    returns NaN.
    """
    speed_cols = ['bris_speed_rating_1', 'bris_speed_rating_2', 'bris_speed_rating_3']

    # defensive: ensure numeric
    for c in speed_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        else:
            # if you really want to force the column in so you never KeyError later:
            df[c] = np.nan

    # now compute
    df['pp_avg_best2_bris_speed'] = (
        df[speed_cols]
          .apply(lambda row: row.nlargest(2).mean() if row.count() >= 2 else np.nan,
                 axis=1)
    )

    return df

def calculate_avg_best2_recent(df: pd.DataFrame,
                               id_vars: list[str],
                               metrics: list[str]) -> pd.DataFrame:
    """
    For each group defined by `id_vars`, sorts by pp_race_date descending,
    then for each metric in `metrics` takes the first 3 rows, selects the
    two largest non-null values, and stores their mean in a new column
    named f"avg_best2_recent_{metric}".
    """
    # Ensure metrics are numeric
    for m in metrics:
        if m in df.columns:
            df[m] = pd.to_numeric(df[m], errors='coerce')
        else:
            df[m] = np.nan

    # Sort entire DataFrame so head(3) on each group is the 3 most recent
    df = df.sort_values("pp_race_date", ascending=False)

    # Compute and broadcast the per-group averages
    for m in metrics:
        out_col = f"avg_best2_recent_{m}"
        df[out_col] = (
            df
            .groupby(id_vars)[m]
            .transform(lambda s: (
                s.head(3)            # 3 most recent
                 .nlargest(2)         # pick top 2
                 .mean()              # average them
            ))
        )

    return df

# ─── New Section: Class-Level & Odds Metrics ────────────────────────────────
def add_class_and_odds_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add average purse size and betting odds for the most-recent
    5, 3, and 1 past starts of every horse on today’s card.

    Requires:
        • pp_race_date : datetime or yyyymmdd int
        • pp_purse     : numeric
        • pp_odds      : numeric
    """
    out = df.copy()

    # Ensure the newest start is *last* inside each horse group
    sort_cols = ['track', 'race', 'horse_name', 'pp_race_date']
    out.sort_values(sort_cols, inplace=True)

    group_cols = ['track', 'race', 'horse_name']

    # Helper: mean of last *n* items in a Series
    def _tail_mean(series, n):
        return series.tail(n).mean()

    # ── Purse size metrics ──────────────────────────────────────────────────
    out['avg_purse_last_5'] = (
        out.groupby(group_cols)['pp_purse']
           .transform(lambda s: _tail_mean(s, 5))
    )
    out['avg_purse_last_3'] = (
        out.groupby(group_cols)['pp_purse']
           .transform(lambda s: _tail_mean(s, 3))
    )
    out['purse_last_1'] = (
        out.groupby(group_cols)['pp_purse']
           .transform(lambda s: s.iloc[-1])
    )

    # ── Betting-odds metrics ────────────────────────────────────────────────
    out['avg_odds_last_5'] = (
        out.groupby(group_cols)['pp_odds']
           .transform(lambda s: _tail_mean(s, 5))
    )
    out['avg_odds_last_3'] = (
        out.groupby(group_cols)['pp_odds']
           .transform(lambda s: _tail_mean(s, 3))
    )
    out['odds_last_1'] = (
        out.groupby(group_cols)['pp_odds']
           .transform(lambda s: s.iloc[-1])
    )

    return out
# ────────────────────────────────────────────────────────────────────────────



# --- Main Execution ---
if __name__ == "__main__":
    print(f"--- Starting Past Performance Data Reshaping ({pd.Timestamp.now(tz='America/New_York').strftime('%Y-%m-%d %H:%M:%S %Z')}) ---")

    wide_df, spec_df = load_data(WIDE_DATA_FILE_PATH, SPEC_CACHE_FILE_PATH)

    if wide_df is None or spec_df is None:
        print("Failed to load necessary data. Aborting.")
        exit()

    # Verify ID variables exist
    actual_id_vars = [col for col in ID_VARIABLES if col in wide_df.columns]
    if len(actual_id_vars) != len(ID_VARIABLES):
        missing_ids = [col for col in ID_VARIABLES if col not in actual_id_vars]
        print(f"Configuration Warning: ID_VARIABLES not found: {missing_ids}")
        if not actual_id_vars: print("Error: No valid ID variables found. Aborting."); exit()
    print(f"Using ID Variables: {actual_id_vars}")

    # --- Reshaping using Iterative Melting ---
    all_past_races_melted: Dict[str, pd.DataFrame] = {}
    print("\nReshaping past performance data using specification and pd.melt...")

    # Loop through each defined metric in the map
    for clean_name, base_field in PAST_RACE_METRIC_MAP.items():
        actual_cols_for_metric: List[str] = []
        col_to_race_num_map: Dict[str, int] = {} # Maps actual col name -> past race num (1-10)

        # Find the 10 actual column names for this metric
        for i in range(10): # Past races 1 to 10
            past_race_num = i + 1
            field_num = base_field + i
            try:
                actual_col_name = spec_df.loc[field_num, 'label']
                if actual_col_name in wide_df.columns:
                    actual_cols_for_metric.append(actual_col_name)
                    col_to_race_num_map[actual_col_name] = past_race_num
            except KeyError: # Skip if field number not in spec cache (e.g., reserved Field 275)
                pass
            except Exception as e:
                print(f"Warning: Error looking up field {field_num}: {e}")

        # Perform melt if we found columns for this metric
        if actual_cols_for_metric:
            # print(f"  Melting metric: '{clean_name}' using columns: {actual_cols_for_metric}") # Verbose
            try:
                melted_metric_df = wide_df[actual_id_vars + actual_cols_for_metric].melt(
                    id_vars=actual_id_vars,
                    value_vars=actual_cols_for_metric,
                    var_name='_actual_col_name', # Temp col holding the messy name
                    value_name=clean_name        # The desired clean metric name
                )
                # Create the past_race_num column (1-10)
                melted_metric_df['past_race_num'] = melted_metric_df['_actual_col_name'].map(col_to_race_num_map)
                # Drop the temporary column and rows where mapping failed (shouldn't happen)
                melted_metric_df.drop(columns=['_actual_col_name'], inplace=True)
                melted_metric_df.dropna(subset=['past_race_num'], inplace=True)

                # Optional: Drop rows where the metric value itself is null/blank here?
                # Depends if you want to keep record of the past race even if this specific metric is missing.
                # Example: melted_metric_df.dropna(subset=[clean_name], inplace=True)

                # Store the result
                all_past_races_melted[clean_name] = melted_metric_df
            except Exception as e:
                 print(f"Error melting metric '{clean_name}': {e}")
                 print(f"  Columns attempted: {actual_cols_for_metric}")


    # --- Join Melted DataFrames ---
    long_past_starts_df: Optional[pd.DataFrame] = None
    if all_past_races_melted:
        print(f"\nJoining {len(all_past_races_melted)} melted metric DataFrames...")
        # Get the list of metric names successfully melted
        melted_keys = list(all_past_races_melted.keys())
        # Start with the first melted DataFrame
        base_metric = melted_keys[0]
        long_past_starts_df = all_past_races_melted[base_metric]
        print(f"  Starting join with '{base_metric}' ({len(long_past_starts_df)} rows)")

        # Iteratively merge the rest
        join_keys = actual_id_vars + ['past_race_num']
        for i in range(1, len(melted_keys)):
            metric_name = melted_keys[i]
            df_to_merge = all_past_races_melted[metric_name]
            # Select only join keys and the metric value to avoid duplicate ID cols
            cols_to_select = join_keys + [metric_name]
            df_to_merge = df_to_merge[cols_to_select]
            print(f"  Merging with '{metric_name}' ({len(df_to_merge)} rows)")
            try:
                long_past_starts_df = pd.merge(
                    long_past_starts_df,
                    df_to_merge,
                    on=join_keys,
                    how='outer' # Use outer join to keep all race records even if a metric is missing
                )
                print(f"  Shape after merge: {long_past_starts_df.shape}")
            except Exception as e:
                 print(f"Error merging metric '{metric_name}': {e}")
                 # Consider logging which columns were in each df before merge
                 # print("Columns in base df:", long_past_starts_df.columns)
                 # print("Columns in df_to_merge:", df_to_merge.columns)


        if long_past_starts_df is not None:
            # Apply final cleaning and type conversion
            long_past_starts_df = clean_past_starts_data(long_past_starts_df)

            # --- NEW: classify each past start as Sprint vs. Route ---
            if 'pp_distance' in long_past_starts_df.columns:
                long_past_starts_df['pp_distance_type'] = (
                    long_past_starts_df['pp_distance']
                    .apply(label_dist)
                )
                print("  Added 'pp_distance_type' classification to past starts.")
            else:
                print("  Critical: 'pp_distance' missing—cannot create 'pp_distance_type'.")
                long_past_starts_df['pp_distance_type'] = np.nan

            # Calculate splits on the cleaned data
            long_past_starts_df = calculate_splits_for_long_format(long_past_starts_df)
            
            # Compute position-gains between calls ---
            long_past_starts_df = calculate_position_changes(long_past_starts_df)
            
            # Calculate changes in lengths behind leader ---
            long_past_starts_df = calculate_lengths_behind_changes(long_past_starts_df)

            # Calculate Pace figures ---
            long_past_starts_df = calculate_early_late_pace_figures(long_past_starts_df)

            # Calculate average of best 2 BRIS speed ratings ---
            long_past_starts_df = calculate_avg_best2_bris_speed(long_past_starts_df)

            # Calculate average of best 2 recent metrics ---
            id_vars = ID_VARIABLES  
            metrics = [
                "pp_e1_pace",
                "pp_turn_time",
                "pp_e2_pace",
                "pp_bris_late_pace",
                "pp_combined_pace",
            ]
            long_past_starts_df = calculate_avg_best2_recent(long_past_starts_df, id_vars, metrics)

            long_past_starts_df=add_class_and_odds_metrics(long_past_starts_df)
            
            long_past_starts_df = long_past_starts_df.copy()

            static_cols = ['track', 'race', 'horse_name', 'post_position',
                           'morn_line_odds_if_available', 'bris_run_style_designation',
                           'quirin_style_speed_points',
                           ]
            
            static_info = wide_df[static_cols].drop_duplicates()

            long_past_starts_df = long_past_starts_df.merge(
                static_info,
                on=['track', 'race', 'post_position', 'horse_name'],
                how='left')
        else:
             print("Join process resulted in an empty DataFrame.")

    else:
        print("\nNo past performance metrics were successfully melted. Cannot proceed.")
        exit()

    # --- Save Results ---
    if long_past_starts_df is not None and not long_past_starts_df.empty:
        print(f"\nSaving long format past performance data to: {LONG_PAST_STARTS_FILE_PATH}")
        try:
            long_past_starts_df.to_parquet(LONG_PAST_STARTS_FILE_PATH, index=False, engine='pyarrow')
            print("Save complete.")
            print("\nOutput DataFrame Info:")
            long_past_starts_df.info(verbose=False, show_counts=True)
        except ImportError:
            print("\nError: 'pyarrow' library not found.")
        except Exception as e:
            print(f"\nError saving final data to Parquet file {LONG_PAST_STARTS_FILE_PATH}: {e}")
    else:
        print("\nNo valid past performance data remained after processing. Output file not saved.")

    print(f"\n--- Past Performance Reshaping Script Finished ({pd.Timestamp.now(tz='America/New_York').strftime('%Y-%m-%d %H:%M:%S %Z')}) ---")