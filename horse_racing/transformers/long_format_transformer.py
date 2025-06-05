# -*- coding: utf-8 -*-
"""Utility functions for converting wide race data into long formats."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import (
    PARSED_RACE_DATA,
    BRIS_SPEC_CACHE,
    WORKOUTS_LONG,
    PAST_STARTS_LONG,
)

# ---------------------------------------------------------------------------
# ID variables shared by both transformations
ID_VARIABLES: List[str] = [
    "track",
    "race",
    "post_position",
    "horse_name",
]

# Mapping from clean workout metric names to the field number of workout #1
WORKOUT_METRIC_MAP: Dict[str, int] = {
    "work_date": 102,
    "work_time": 114,
    "work_track": 126,
    "work_distance": 138,
    "work_track_condition": 150,
    "work_description": 162,
    "work_surface_type": 174,
    "work_num_at_dist": 186,
    "work_rank": 198,
}

# Mapping from clean past start metric names to the field number of race #1
PAST_RACE_METRIC_MAP: Dict[str, int] = {
    "pp_race_date": 256,
    "pp_days_since_prev": 266,
    "pp_track_code": 276,
    "pp_track_code_bris": 286,
    "pp_race_num": 296,
    "pp_track_condition": 306,
    "pp_distance": 316,
    "pp_surface": 326,
    "pp_chute_indicator": 336,
    "pp_num_entrants": 346,
    "pp_post_position": 356,
    "pp_equipment": 366,
    "pp_racename": 376,
    "pp_medication": 386,
    "pp_trip_comment": 396,
    "pp_winner_name": 406,
    "pp_second_name": 416,
    "pp_third_name": 426,
    "pp_winner_weight": 436,
    "pp_second_weight": 446,
    "pp_third_weight": 456,
    "pp_winner_margin": 466,
    "pp_second_margin": 476,
    "pp_third_margin": 486,
    "pp_extra_comment": 496,
    "pp_weight_carried": 506,
    "pp_odds": 516,
    "pp_entry_indicator": 526,
    "pp_race_classification": 536,
    "pp_claiming_price": 546,
    "pp_purse": 556,
    "pp_start_call_pos": 566,
    "pp_first_call_pos": 576,
    "pp_second_call_pos": 586,
    "pp_gate_call_pos": 596,
    "pp_stretch_pos": 606,
    "pp_finish_pos": 616,
    "pp_money_pos": 626,
    "pp_start_lengths_leader": 636,
    "pp_start_lengths_behind": 646,
    "pp_first_call_lengths_leader": 656,
    "pp_first_call_lengths_behind": 666,
    "pp_second_call_lengths_leader": 676,
    "pp_second_call_lengths_behind": 686,
    "pp_bris_shape_1st_call": 696,
    "pp_stretch_lengths_leader": 716,
    "pp_stretch_lengths_behind": 726,
    "pp_finish_lengths_winner": 736,
    "pp_finish_lengths_behind": 746,
    "pp_bris_shape_2nd_call": 756,
    "pp_bris_2f_pace": 766,
    "pp_bris_4f_pace": 776,
    "pp_bris_6f_pace": 786,
    "pp_bris_8f_pace": 796,
    "pp_bris_10f_pace": 806,
    "pp_bris_late_pace": 816,
    "pp_bris_speed_rating": 846,
    "pp_speed_rating_alt": 856,
    "pp_track_variant": 866,
    "pp_frac_2f": 876,
    "pp_frac_3f": 886,
    "pp_frac_4f": 896,
    "pp_frac_5f": 906,
    "pp_frac_6f": 916,
    "pp_frac_7f": 926,
    "pp_frac_8f": 936,
    "pp_frac_10f": 946,
    "pp_frac_12f": 956,
    "pp_frac_14f": 966,
    "pp_frac_16f": 976,
    "pp_fraction_1": 986,
    "pp_fraction_2": 996,
    "pp_fraction_3": 1006,
    "pp_final_time": 1036,
    "pp_claimed_code": 1046,
    "pp_trainer": 1056,
    "pp_jockey": 1066,
    "pp_apprentice_allow": 1076,
    "pp_race_type": 1086,
    "pp_age_sex_restrict": 1096,
    "pp_statebred_flag": 1106,
    "pp_restricted_flag": 1116,
    "pp_favorite_indicator": 1126,
    "pp_front_bandages": 1136,
    "pp_bris_speed_par": 1167,
    "pp_bar_shoe": 1182,
    "pp_company_line": 1192,
    "pp_low_claiming": 1202,
    "pp_high_claiming": 1212,
    "pp_misc_code": 1254,
    "pp_ext_start_comment": 1383,
    "pp_sealed_track": 1393,
    "pp_aw_surface_flag": 1403,
    "pp_eqb_conditions": 1419,
}

# ---------------------------------------------------------------------------
# Generic helpers

def load_data(parquet_path: Path, pkl_path: Path) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load wide-format race data and the specification cache."""
    wide_df: Optional[pd.DataFrame] = None
    spec_df: Optional[pd.DataFrame] = None

    if not parquet_path.exists():
        logging.error("Input Parquet file not found at %s", parquet_path)
        return None, None
    try:
        wide_df = pd.read_parquet(parquet_path, engine="pyarrow")
    except Exception as exc:
        logging.error("Error loading Parquet file %s: %s", parquet_path, exc)
        return None, None

    if not pkl_path.exists():
        logging.error("Specification cache file not found at %s", pkl_path)
        return wide_df, None
    try:
        spec_df = pd.read_pickle(pkl_path)
        if "field_number" not in spec_df.columns and spec_df.index.name != "field_number":
            spec_df["field_number"] = spec_df.index
        if "label" not in spec_df.columns:
            logging.error("Spec cache '%s' must contain a 'label' column.", pkl_path.name)
            return wide_df, None
        if "field_number" in spec_df.columns and spec_df.index.name != "field_number":
            spec_df = spec_df.set_index("field_number", drop=False)
    except Exception as exc:
        logging.error("Error loading specification cache file %s: %s", pkl_path, exc)
        return wide_df, None

    return wide_df, spec_df


def validate_id_vars(df: pd.DataFrame, id_vars: List[str], logger: logging.Logger) -> List[str]:
    """Ensure that ID columns exist in the DataFrame."""
    actual = [c for c in id_vars if c in df.columns]
    if len(actual) != len(id_vars):
        missing = [c for c in id_vars if c not in actual]
        logger.warning("ID variables missing from data: %s", missing)
        if not actual:
            raise ValueError("No valid ID variables found.")
    return actual


def wide_to_long_iterative(
    wide_df: pd.DataFrame,
    spec_df: pd.DataFrame,
    id_vars: List[str],
    metric_map: Dict[str, int],
    iter_col: str,
    n_iters: int,
) -> pd.DataFrame:
    """Iteratively reshape sets of numbered columns to a long format."""
    out_frames: List[pd.DataFrame] = []
    for i in range(n_iters):
        num = i + 1
        col_map: Dict[str, str] = {}
        cols: List[str] = []
        for clean, base in metric_map.items():
            field_num = base + i
            try:
                actual = spec_df.loc[field_num, "label"]
            except Exception:
                continue
            if actual in wide_df.columns:
                col_map[actual] = clean
                cols.append(actual)
        if not cols:
            continue
        temp = wide_df[id_vars + cols].copy()
        temp.rename(columns=col_map, inplace=True)
        temp[iter_col] = num
        final_cols = id_vars + list(metric_map.keys()) + [iter_col]
        final_cols = [c for c in final_cols if c in temp.columns]
        out_frames.append(temp[final_cols])
    if not out_frames:
        return pd.DataFrame(columns=id_vars + list(metric_map.keys()) + [iter_col])
    return pd.concat(out_frames, ignore_index=True, sort=False)


def wide_to_long_melt(
    wide_df: pd.DataFrame,
    spec_df: pd.DataFrame,
    id_vars: List[str],
    metric_map: Dict[str, int],
    iter_col: str,
    n_iters: int,
) -> pd.DataFrame:
    """Reshape numbered columns to long format using pd.melt."""
    melted: Dict[str, pd.DataFrame] = {}
    for clean, base in metric_map.items():
        actual_cols: List[str] = []
        col_to_iter: Dict[str, int] = {}
        for i in range(n_iters):
            iter_num = i + 1
            field_num = base + i
            try:
                actual = spec_df.loc[field_num, "label"]
            except Exception:
                continue
            if actual in wide_df.columns:
                actual_cols.append(actual)
                col_to_iter[actual] = iter_num
        if not actual_cols:
            continue
        mdf = wide_df[id_vars + actual_cols].melt(
            id_vars=id_vars,
            value_vars=actual_cols,
            var_name="_actual",
            value_name=clean,
        )
        mdf[iter_col] = mdf["_actual"].map(col_to_iter)
        mdf.drop(columns=["_actual"], inplace=True)
        mdf.dropna(subset=[iter_col], inplace=True)
        melted[clean] = mdf
    if not melted:
        return pd.DataFrame()
    keys = list(melted.keys())
    df_long = melted[keys[0]]
    join_keys = id_vars + [iter_col]
    for key in keys[1:]:
        df_long = df_long.merge(
            melted[key][join_keys + [key]],
            on=join_keys,
            how="outer",
        )
    return df_long

# ---------------------------------------------------------------------------
# Workout-specific helpers

def clean_workout_data(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if "work_date" in df.columns:
        df["work_date"] = pd.to_datetime(df["work_date"], format="%Y%m%d", errors="coerce")
        df.dropna(subset=["work_date"], inplace=True)
    numeric_cols = ["work_time", "work_distance", "work_num_at_dist", "work_rank"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "workout_num" in df.columns and df["workout_num"].notna().all():
        df["workout_num"] = df["workout_num"].astype(int)
    return df

# ---------------------------------------------------------------------------
# Past starts specific helpers

def label_dist(d):
    try:
        val = float(d)
    except (ValueError, TypeError):
        return np.nan
    return "Sprint" if val <= 1540 else "Route"


def clean_past_starts_data(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    date_cols = [col for col in df.columns if "date" in col.lower()]
    numeric_cols = [
        col
        for col in df.columns
        if any(k in col.lower() for k in [
            "days",
            "distance",
            "entrants",
            "position",
            "weight",
            "margin",
            "odds",
            "claiming",
            "purse",
            "lengths",
            "shape",
            "pace",
            "speed",
            "variant",
            "frac",
            "time",
            "allow",
            "indic",
            "par",
            "low",
            "high",
        ])
        and col not in date_cols
    ]
    if "pp_distance" in df.columns:
        df["pp_distance"] = pd.to_numeric(df["pp_distance"], errors="coerce")
    numeric_cols.extend(["pp_race_num", "pp_medication", "pp_purse"])
    numeric_cols = list(set(numeric_cols))
    if "pp_race_date" in df.columns:
        df["pp_race_date"] = pd.to_datetime(df["pp_race_date"], format="%Y%m%d", errors="coerce")
        df.dropna(subset=["pp_race_date"], inplace=True)
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    pos_cols = [c for c in df.columns if "pos" in c.lower() or "finish" in c.lower()]
    for col in pos_cols:
        if col in df.columns and df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "past_race_num" in df.columns and df["past_race_num"].notna().all():
        df["past_race_num"] = df["past_race_num"].astype(int)
    return df


def calculate_splits_for_long_format(df: pd.DataFrame) -> pd.DataFrame:
    cumulative = {
        "c2": "pp_frac_2f",
        "c4": "pp_frac_4f",
        "c6": "pp_frac_6f",
        "c8": "pp_frac_8f",
        "c10": "pp_frac_10f",
        "c12": "pp_frac_12f",
        "c5": "pp_frac_5f",
        "c7": "pp_frac_7f",
        "c9": "pp_frac_9f",
        "final": "pp_final_time",
    }
    for col in cumulative.values():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "pp_frac_2f" in df:
        df["pp_split_0_2f_secs"] = df["pp_frac_2f"]
    if all(c in df for c in ("pp_frac_4f", "pp_frac_2f")):
        df["pp_split_2f_4f_secs"] = df["pp_frac_4f"] - df["pp_frac_2f"]
    if all(c in df for c in ("pp_frac_6f", "pp_frac_4f")):
        df["pp_split_4f_6f_secs"] = df["pp_frac_6f"] - df["pp_frac_4f"]
    if all(c in df for c in ("pp_frac_8f", "pp_frac_6f")):
        df["pp_split_6f_8f_secs"] = df["pp_frac_8f"] - df["pp_frac_6f"]
    if all(c in df for c in ("pp_frac_10f", "pp_frac_8f")):
        df["pp_split_8f_10f_secs"] = df["pp_frac_10f"] - df["pp_frac_8f"]
    if all(c in df for c in ("pp_frac_12f", "pp_frac_10f")):
        df["pp_split_10f_12f_secs"] = df["pp_frac_12f"] - df["pp_frac_10f"]
    if "pp_final_time" in df.columns:
        if "pp_frac_10f" in df.columns:
            df["pp_split_10f_finish_secs"] = df["pp_final_time"] - df["pp_frac_10f"]
        elif "pp_frac_8f" in df.columns:
            df["pp_split_8f_finish_secs"] = df["pp_final_time"] - df["pp_frac_8f"]
        elif "pp_frac_6f" in df.columns:
            df["pp_split_6f_finish_secs"] = df["pp_final_time"] - df["pp_frac_6f"]
        elif "pp_frac_4f" in df.columns:
            df["pp_split_4f_finish_secs"] = df["pp_final_time"] - df["pp_frac_4f"]
        elif "pp_frac_2f" in df.columns:
            df["pp_split_2f_finish_secs"] = df["pp_final_time"] - df["pp_frac_2f"]
    if all(c in df for c in ("pp_frac_5f", "pp_frac_4f")):
        df["pp_split_4f_5f_secs"] = df["pp_frac_5f"] - df["pp_frac_4f"]
    if all(c in df for c in ("pp_frac_7f", "pp_frac_6f")):
        df["pp_split_6f_7f_secs"] = df["pp_frac_7f"] - df["pp_frac_6f"]
    if all(c in df for c in ("pp_frac_9f", "pp_frac_8f")):
        df["pp_split_8f_9f_secs"] = df["pp_frac_9f"] - df["pp_frac_8f"]
    return df


def calculate_position_changes(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {
        "start": "pp_start_call_pos",
        "c1": "pp_first_call_pos",
        "c2": "pp_second_call_pos",
        "stretch": "pp_stretch_pos",
        "finish": "pp_finish_pos",
    }
    for col in col_map.values():
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "pp_start_call_pos" in df and "pp_first_call_pos" in df:
        df["pp_pos_gain_start_to_c1"] = df["pp_start_call_pos"] - df["pp_first_call_pos"]
    if "pp_first_call_pos" in df and "pp_second_call_pos" in df:
        df["pp_pos_gain_c1_to_c2"] = df["pp_first_call_pos"] - df["pp_second_call_pos"]
    if "pp_second_call_pos" in df and "pp_stretch_pos" in df:
        df["pp_pos_gain_c2_to_stretch"] = df["pp_second_call_pos"] - df["pp_stretch_pos"]
    if "pp_stretch_pos" in df and "pp_finish_pos" in df:
        df["pp_pos_gain_stretch_to_finish"] = df["pp_stretch_pos"] - df["pp_finish_pos"]
    if "pp_start_call_pos" in df and "pp_finish_pos" in df:
        df["pp_pos_gain_start_to_finish"] = df["pp_start_call_pos"] - df["pp_finish_pos"]
    return df


def calculate_lengths_behind_changes(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {
        "start": "pp_start_lengths_leader",
        "c1": "pp_first_call_lengths_leader",
        "c2": "pp_second_call_lengths_leader",
        "stretch": "pp_stretch_lengths_leader",
        "finish": "pp_finish_lengths_winner",
    }
    for col in col_map.values():
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "pp_start_lengths_leader" in df and "pp_first_call_lengths_leader" in df:
        df["pp_chg_len_ldr_start_to_c1"] = df["pp_first_call_lengths_leader"] - df["pp_start_lengths_leader"]
    if "pp_first_call_lengths_leader" in df and "pp_second_call_lengths_leader" in df:
        df["pp_chg_len_ldr_c1_to_c2"] = df["pp_second_call_lengths_leader"] - df["pp_first_call_lengths_leader"]
    if "pp_second_call_lengths_leader" in df and "pp_stretch_lengths_leader" in df:
        df["pp_chg_len_ldr_c2_to_stretch"] = df["pp_stretch_lengths_leader"] - df["pp_second_call_lengths_leader"]
    if "pp_stretch_lengths_leader" in df and "pp_finish_lengths_winner" in df:
        df["pp_chg_len_ldr_stretch_to_finish"] = df["pp_finish_lengths_winner"] - df["pp_stretch_lengths_leader"]
    if "pp_start_lengths_leader" in df and "pp_finish_lengths_winner" in df:
        df["pp_chg_len_ldr_start_to_finish"] = df["pp_finish_lengths_winner"] - df["pp_start_lengths_leader"]
    return df


def calculate_early_late_pace_figures(df: pd.DataFrame) -> pd.DataFrame:
    pace_cols = {
        "2f": "pp_bris_2f_pace",
        "4f": "pp_bris_4f_pace",
        "6f": "pp_bris_6f_pace",
        "late": "pp_bris_late_pace",
    }
    for col in pace_cols.values():
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "pp_distance_type" not in df.columns:
        df["pp_e1_pace"] = np.nan
        df["pp_e2_pace"] = np.nan
        df["pp_turn_time"] = np.nan
        df["pp_combined_pace"] = np.nan
        return df
    df["pp_e1_pace"] = np.nan
    df["pp_e2_pace"] = np.nan
    sprint = df["pp_distance_type"] == "Sprint"
    route = df["pp_distance_type"] == "Route"
    if "pp_bris_2f_pace" in df:
        df.loc[sprint, "pp_e1_pace"] = df.loc[sprint, "pp_bris_2f_pace"]
    if "pp_bris_4f_pace" in df:
        df.loc[route, "pp_e1_pace"] = df.loc[route, "pp_bris_4f_pace"]
    if "pp_bris_4f_pace" in df:
        df.loc[sprint, "pp_e2_pace"] = df.loc[sprint, "pp_bris_4f_pace"]
    if "pp_bris_6f_pace" in df:
        df.loc[route, "pp_e2_pace"] = df.loc[route, "pp_bris_6f_pace"]
    df["pp_turn_time"] = df["pp_e2_pace"] - df["pp_e1_pace"]
    if "pp_bris_late_pace" in df:
        df["pp_combined_pace"] = df["pp_e2_pace"] + df["pp_bris_late_pace"]
    else:
        df["pp_combined_pace"] = np.nan
    return df


def calculate_avg_best2_bris_speed(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["bris_speed_rating_1", "bris_speed_rating_2", "bris_speed_rating_3"]
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = np.nan
    df["pp_avg_best2_bris_speed"] = df[cols].apply(
        lambda r: r.nlargest(2).mean() if r.count() >= 2 else np.nan,
        axis=1,
    )
    return df


def calculate_avg_best2_recent(df: pd.DataFrame, id_vars: List[str], metrics: List[str]) -> pd.DataFrame:
    for m in metrics:
        if m in df.columns:
            df[m] = pd.to_numeric(df[m], errors="coerce")
        else:
            df[m] = np.nan
    df = df.sort_values("pp_race_date", ascending=False)
    for m in metrics:
        out_col = f"avg_best2_recent_{m}"
        df[out_col] = (
            df.groupby(id_vars)[m]
            .transform(lambda s: s.head(3).nlargest(2).mean())
        )
    return df


def add_class_and_odds_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    sort_cols = ["track", "race", "horse_name", "pp_race_date"]
    out.sort_values(sort_cols, inplace=True)
    group_cols = ["track", "race", "horse_name"]
    def _tail_mean(series: pd.Series, n: int):
        return series.tail(n).mean()
    out["avg_purse_last_5"] = out.groupby(group_cols)["pp_purse"].transform(lambda s: _tail_mean(s, 5))
    out["avg_purse_last_3"] = out.groupby(group_cols)["pp_purse"].transform(lambda s: _tail_mean(s, 3))
    out["purse_last_1"] = out.groupby(group_cols)["pp_purse"].transform(lambda s: s.iloc[-1])
    out["avg_odds_last_5"] = out.groupby(group_cols)["pp_odds"].transform(lambda s: _tail_mean(s, 5))
    out["avg_odds_last_3"] = out.groupby(group_cols)["pp_odds"].transform(lambda s: _tail_mean(s, 3))
    out["odds_last_1"] = out.groupby(group_cols)["pp_odds"].transform(lambda s: s.iloc[-1])
    return out

# ---------------------------------------------------------------------------
# High level transformation functions

def transform_workouts() -> None:
    logger = logging.getLogger(__name__)
    logger.info("--- Transforming workout data ---")
    wide_df, spec_df = load_data(PARSED_RACE_DATA, BRIS_SPEC_CACHE)
    if wide_df is None or spec_df is None:
        logger.error("Failed to load necessary data. Aborting.")
        return
    actual_id_vars = validate_id_vars(wide_df, ID_VARIABLES, logger)
    long_df = wide_to_long_iterative(
        wide_df,
        spec_df,
        actual_id_vars,
        WORKOUT_METRIC_MAP,
        "workout_num",
        12,
    )
    long_df = clean_workout_data(long_df)
    if long_df.empty:
        logger.warning("No valid workout data found. Output file not saved.")
        return
    long_df.to_parquet(WORKOUTS_LONG, index=False, engine="pyarrow")
    logger.info("Saved workout data to %s", WORKOUTS_LONG)


def transform_past_starts() -> None:
    logger = logging.getLogger(__name__)
    logger.info("--- Transforming past performance data ---")
    wide_df, spec_df = load_data(PARSED_RACE_DATA, BRIS_SPEC_CACHE)
    if wide_df is None or spec_df is None:
        logger.error("Failed to load necessary data. Aborting.")
        return
    actual_id_vars = validate_id_vars(wide_df, ID_VARIABLES, logger)
    long_df = wide_to_long_melt(
        wide_df,
        spec_df,
        actual_id_vars,
        PAST_RACE_METRIC_MAP,
        "past_race_num",
        10,
    )
    if long_df.empty:
        logger.error("No past performance metrics were successfully melted.")
        return
    long_df = clean_past_starts_data(long_df)
    if "pp_distance" in long_df.columns:
        long_df["pp_distance_type"] = long_df["pp_distance"].apply(label_dist)
    else:
        logger.warning("'pp_distance' missing—cannot create 'pp_distance_type'.")
        long_df["pp_distance_type"] = np.nan
    long_df = calculate_splits_for_long_format(long_df)
    long_df = calculate_position_changes(long_df)
    long_df = calculate_lengths_behind_changes(long_df)
    long_df = calculate_early_late_pace_figures(long_df)
    long_df = calculate_avg_best2_bris_speed(long_df)
    metrics = [
        "pp_e1_pace",
        "pp_turn_time",
        "pp_e2_pace",
        "pp_bris_late_pace",
        "pp_combined_pace",
    ]
    long_df = calculate_avg_best2_recent(long_df, ID_VARIABLES, metrics)
    long_df = add_class_and_odds_metrics(long_df)
    long_df = long_df.copy()
    static_cols = [
        "track",
        "race",
        "horse_name",
        "post_position",
        "morn_line_odds_if_available",
        "bris_run_style_designation",
        "quirin_style_speed_points",
    ]
    static_info = wide_df[static_cols].drop_duplicates()
    long_df = long_df.merge(
        static_info,
        on=["track", "race", "post_position", "horse_name"],
        how="left",
    )
    if long_df.empty:
        logger.warning("No valid past performance data remained. Output not saved.")
        return
    long_df.to_parquet(PAST_STARTS_LONG, index=False, engine="pyarrow")
    logger.info("Saved past performance data to %s", PAST_STARTS_LONG)

