import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Final, Optional

# --- Define standardized paths ---
SCRIPT_DIR: Path = Path(__file__).parent.resolve() # .../src/transformers/
PROJECT_ROOT: Path = SCRIPT_DIR.parent.parent    # .../horse_racing_project/
PROCESSED_DATA_DIR: Path = PROJECT_ROOT / "data" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True) # Ensure directory exists

# --- Load your full parsed DataFrame ---
# Adjust to read from the processed data directory
INPUT_PARQUET_PATH: Path = PROCESSED_DATA_DIR / "parsed_race_data_full.parquet"
if not INPUT_PARQUET_PATH.exists():
    print(f"Error: Input file not found at {INPUT_PARQUET_PATH}")
    exit()
print(f"Loading data from: {INPUT_PARQUET_PATH}")
df = pd.read_parquet(INPUT_PARQUET_PATH)

# ——————————————————————————
def compute_fractional_splits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Efficiently build all split columns for races 1–10 without fragmenting df.
    Returns a NEW DataFrame, `splits_df`, containing only the split columns.
    """
    splits = {}
    for i in range(1, 11):
        # source columns
        c2, c4, c6 = f"2f_fraction_if_any_{i}", f"4f_fraction_if_any_{i}", f"6f_fraction_if_any_{i}"
        c8, c10, c12 = f"8f_fraction_if_any_{i}", f"10f_fraction_if_any_{i}", f"12f_fraction_if_any_{i}"
        c5, c7, c9 = f"5f_fraction_if_any_{i}", f"7f_fraction_if_any_{i}", f"9f_fraction_if_any_{i}"

        # coerce once
        for col in (c2, c4, c6, c8, c10, c12, c5, c7, c9):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # even splits
        if c2 in df: splits[f"fraction_1_{i}"] = df[c2]
        if c2 in df and c4 in df: splits[f"fraction_2_{i}"] = df[c4] - df[c2]
        if c4 in df and c6 in df: splits[f"fraction_3_{i}"] = df[c6] - df[c4]
        if c6 in df and c8 in df: splits[f"fraction_4_{i}"] = df[c8] - df[c6]
        if c8 in df and c10 in df: splits[f"fraction_5_{i}"] = df[c10] - df[c8]
        if c10 in df and c12 in df: splits[f"fraction_6_{i}"] = df[c12] - df[c10]

        # odd‐furlong splits
        if c5 in df and c4 in df: splits[f"fraction_5f_{i}"] = df[c5] - df[c4]
        if c7 in df and c6 in df: splits[f"fraction_7f_{i}"] = df[c7] - df[c6]
        if c9 in df and c8 in df: splits[f"fraction_9f_{i}"] = df[c9] - df[c8]

    # one big concat
    splits_df = pd.DataFrame(splits, index=df.index)
    return splits_df


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

# loop and compute
for name, (col_start, col_win, col_place, col_show, col_earn) in record_groups.items():
    # avoid division-by-zero
    starts = df[col_start].replace({0: np.nan})

    df[f"{name}_win_pct"]            = df[col_win]   / starts
    df[f"{name}_itm_pct"]            = (df[col_win] + df[col_place] + df[col_show]) / starts
    df[f"{name}_earnings_per_start"] = df[col_earn]  / starts

# assume `df` already has all the *_win_pct, *_itm_pct, *_earnings_per_start cols

# 1. build the list of metric column names
record_groups = ["distance", "track", "turf", "mud", "current_year", "previous_year", "lifetime"]
metrics = []
for name in record_groups:
    metrics += [f"{name}_win_pct", f"{name}_itm_pct", f"{name}_earnings_per_start"]

# 2. select only the identifiers + your new metrics
df_perf = df[["race", "post_position", "morn_line_odds_if_available", "horse_name"] + metrics].copy()
df_splits = compute_fractional_splits(df)


# 3. Output to CSV in the processed data directory
output_perf_csv = PROCESSED_DATA_DIR / "horse_performance.csv"
output_splits_csv = PROCESSED_DATA_DIR / "splits.csv"

print(f"Saving horse_performance.csv to: {output_perf_csv}")
df_perf.to_csv(output_perf_csv, index=False)
print(f"Saving splits.csv to: {output_splits_csv}")
df_splits.to_csv(output_splits_csv, index=False)

# ────────────────────────────────────────────────
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
for col in jockey_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
# ────────────────────────────────────────────────


# 4. Build new DataFrame for jockey performance
df_jockey = pd.DataFrame({
    'Jockey': df['today_s_jockey'],
    'Change': df['today_s_jockey'].eq(df['jockey_1']).map({True: 'N', False: 'Y'}),
    'Win %': df['wins_4_pos_1158'] / df['jockey_sts_current_year'],
    'Last Yr %': df['wins_4_pos_1163'] / df['jockey_sts_previous_year'],
    'Meet': df['jockey_wins_current_meet'] / df['jockey_sts_current_meet'],
    'T/J': df['t_j_combo_wins_meet'] / df['t_j_combo_starts_meet'],
})

# 3) Examine the first rows
print(df_splits.head())

# (Optional) save to its own file
output_jockey_parquet = PROCESSED_DATA_DIR / "jockey_performance.parquet"
# print(f"Saving jockey_performance.parquet to: {output_jockey_parquet}")
# df_jockey.to_parquet(output_jockey_parquet, index=False) # Currently commented out

print("\n--- Feature Engineering Script Finished ---")



