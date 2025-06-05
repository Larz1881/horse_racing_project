"""Unified module for various race data views."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import *

class RaceDataViews:
    """Provides various analytical views of race data."""

    def __init__(self, current_path: Optional[Path] = None,
                 past_path: Optional[Path] = None):
        """Initialize with data paths."""
        self.current_path = current_path or CURRENT_RACE_INFO
        self.past_path = past_path or PAST_STARTS_LONG

        # Load data
        self.current_df = pd.read_parquet(self.current_path)
        self.past_df = None  # Lazy load when needed

        # Cache for computed metrics
        self._metrics_cache = {}

    def _load_past_data(self):
        """Lazy load past performance data when needed."""
        if self.past_df is None and self.past_path.exists():
            self.past_df = pd.read_parquet(self.past_path)

    def get_race_metadata(self, race_num: int) -> Dict[str, str]:
        """Get race condition details for display."""
        race_data = self.current_df[self.current_df['race'] == race_num]
        if race_data.empty:
            return {}

        first_row = race_data.iloc[0]
        return {
            'conditions': first_row.get('race_conditions', ''),
            'classification': first_row.get('today_s_race_classification', ''),
            'purse': first_row.get('purse', ''),
            'distance': first_row.get('distance_in_yards', ''),
            'surface': first_row.get('surface', ''),
            'furlongs': first_row.get('furlongs', '')  # If you calculate this
        }

    def get_contender_summary(self, race_num: int) -> pd.DataFrame:
        """Get the contender summary view (from app.py)."""
        # Load past data for metrics
        self._load_past_data()

        # Get current race data
        race_current = self.current_df[self.current_df['race'] == race_num].copy()

        if not race_current.empty and self.past_df is not None:
            # Calculate metrics if not cached
            cache_key = f'metrics_{race_num}'
            if cache_key not in self._metrics_cache:
                # Get past performance metrics
                race_past = self.past_df[self.past_df['race'] == race_num]

                # Calculate avg best 2 speed and combined pace
                id_vars = ["race", "post_position", "horse_name"]
                metrics = race_past.groupby(id_vars).agg({
                    'pp_bris_speed_rating': lambda x: x.nlargest(2).mean() if len(x) >= 2 else x.mean(),
                    'pp_combined_pace': 'mean'  # Or however you calculate this
                }).reset_index()

                metrics.columns = ['race', 'post_position', 'horse_name',
                                 'pp_avg_best2_bris_speed', 'pp_combined_pace']

                self._metrics_cache[cache_key] = metrics

            # Merge metrics
            race_current = race_current.merge(
                self._metrics_cache[cache_key],
                on=['race', 'post_position', 'horse_name'],
                how='left'
            )

        # Select and rename columns
        columns = {
            'post_position': 'Post',
            'horse_name': 'Horse',
            'morn_line_odds_if_available': 'M/L Odds',
            'bris_prime_power_rating': 'Prime Power',
            'pp_avg_best2_bris_speed': 'Speed 2/3',
            'pp_combined_pace': 'Combined'
        }

        # Ensure only existing columns are selected to avoid KeyErrors
        existing_keys = [col for col in columns.keys() if col in race_current.columns]
        result = race_current[existing_keys].rename(columns=columns)

        if 'Prime Power' in result.columns:
            return result.sort_values('Prime Power', ascending=False)
        return result


    def get_connections_view(self, race_num: int) -> pd.DataFrame:
        """Get connections (owner, breeder, auction info)."""
        columns = {
            'program_number_if_available': 'Number',
            'horse_name': 'Name',
            'morn_line_odds_if_available': 'M/L Odds',
            'today_s_owner': 'Owner',
            'auction_price': 'Auction Price',
            'breeder': 'Breeder',
            'sire_stud_fee_current': 'Stud Fee'
        }

        race_data = self.current_df[self.current_df['race'] == race_num]
        existing_cols = [col for col in columns.keys() if col in race_data.columns]

        result = race_data[existing_cols].rename(columns=columns)

        if 'M/L Odds' in result.columns:
            return result.sort_values('M/L Odds', ascending=True)
        return result


    def get_trainer_jockey_view(self, race_num: int) -> pd.DataFrame:
        """Get trainer/jockey statistics view."""
        columns = {
            'program_number_if_available': 'Number',
            'horse_name': 'Name',
            'morn_line_odds_if_available': 'M/L Odds',
            'today_s_trainer': 'Trainer',
            'today_s_jockey': 'Jockey',
            't_j_combo_starts_365d': 'Starts',
            't_j_combo_wins_365d': 'Wins',
            't_j_combo_2_roi_365d': 'ROI'
        }

        race_data = self.current_df[self.current_df['race'] == race_num]
        existing_cols = [col for col in columns.keys() if col in race_data.columns]

        result = race_data[existing_cols].rename(columns=columns)

        # Add win percentage if we have the data
        if 'Starts' in result.columns and 'Wins' in result.columns:
            # Ensure 'Starts' is numeric and handle division by zero
            starts_numeric = pd.to_numeric(result['Starts'], errors='coerce').fillna(0)
            wins_numeric = pd.to_numeric(result['Wins'], errors='coerce').fillna(0)
            result['Win%'] = np.where(starts_numeric > 0, (wins_numeric / starts_numeric) * 100, pd.NA)
            result['Win%'] = result['Win%'].round(1)

        if 'M/L Odds' in result.columns:
            return result.sort_values('M/L Odds', ascending=True)
        return result


    def get_pace_scenario(self, race_num: int) -> pd.DataFrame:
        """Get pace scenario analysis view."""
        self._load_past_data()

        # Base columns from current data
        base_columns = {
            'program_number_if_available': 'Number',
            'horse_name': 'Name',
            'morn_line_odds_if_available': 'M/L Odds',
            'bris_run_style_designation': 'Run Style',
            'quirin_style_speed_points': 'Speed Points'
        }

        race_current = self.current_df[self.current_df['race'] == race_num].copy()
        race_data = race_current # Initialize race_data with current data

        # If we have past data, calculate pace metrics
        if self.past_df is not None:
            race_past_for_metrics = self.past_df[self.past_df['race'] == race_num] # Use a different variable name

            if not race_past_for_metrics.empty:
                # Calculate average best 2 recent pace figures
                pace_metrics = race_past_for_metrics.groupby(['race', 'post_position', 'horse_name']).apply(
                    lambda group: pd.Series({
                        'avg_e1_pace': group['pp_e1_pace'].nlargest(2).mean() if not group['pp_e1_pace'].dropna().empty else pd.NA,
                        'avg_turn_time': group['pp_turn_time'].nlargest(2).mean() if not group['pp_turn_time'].dropna().empty else pd.NA,
                        'avg_e2_pace': group['pp_e2_pace'].nlargest(2).mean() if not group['pp_e2_pace'].dropna().empty else pd.NA,
                        'avg_late_pace': group['pp_bris_late_pace'].nlargest(2).mean() if not group['pp_bris_late_pace'].dropna().empty else pd.NA,
                        'avg_combined_pace': group['pp_combined_pace'].nlargest(2).mean() if not group['pp_combined_pace'].dropna().empty else pd.NA
                    })
                ).reset_index()

                # Merge with current data
                if not pace_metrics.empty:
                    race_data = race_current.merge(
                        pace_metrics,
                        on=['race', 'post_position', 'horse_name'],
                        how='left'
                    )
        
        # Select columns that exist
        all_columns = {
            **base_columns,
            'avg_e1_pace': 'Avg E1',
            'avg_turn_time': 'Turn Time',
            'avg_e2_pace': 'Avg E2',
            'avg_late_pace': 'Avg LP',
            'avg_combined_pace': 'Avg Combined'
        }

        existing_cols_in_data = [col for col in all_columns.keys() if col in race_data.columns]
        rename_map = {col: all_columns[col] for col in existing_cols_in_data}

        result = race_data[existing_cols_in_data].rename(columns=rename_map)

        if 'M/L Odds' in result.columns:
            return result.sort_values('M/L Odds', ascending=True)
        return result


    def get_all_race_numbers(self) -> List[int]:
        """Get sorted list of all race numbers."""
        if 'race' in self.current_df.columns:
            return sorted(self.current_df['race'].unique())
        return []

    def get_horses_in_race(self, race_num: int) -> List[Tuple[int, str]]:
        """Get list of (post_position, horse_name) tuples for a race."""
        race_data = self.current_df[self.current_df['race'] == race_num]
        if not race_data.empty and 'post_position' in race_data.columns and 'horse_name' in race_data.columns:
            horses = race_data[['post_position', 'horse_name']].values.tolist()
            return sorted(horses, key=lambda x: x[0])  # Sort by post position
        return []