import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from config.settings import (
    PROCESSED_DATA_DIR,
    PAST_STARTS_LONG,
    PARSED_RACE_DATA,
)
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class HorseRacingFeatureEngineer:
    """
    Comprehensive feature engineering for horse racing predictive modeling
    with limited historical data (typically 10 races per horse maximum).
    """
    
    def __init__(self, processed_data_path: Path):
        self.processed_data_path = processed_data_path
        self.past_starts_path = processed_data_path / PAST_STARTS_LONG.name
        self.current_race_path = processed_data_path / PARSED_RACE_DATA.name
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load past starts and current race data."""
        print("Loading race data...")
        past_starts = pd.read_parquet(self.past_starts_path)
        current_race = pd.read_parquet(self.current_race_path)
        
        print(f"Past starts: {len(past_starts)} records")
        print(f"Current race: {len(current_race)} horses")
        
        return past_starts, current_race
    
    def calculate_recent_speed_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate recent speed figure features."""
        print("Calculating recent speed features...")
        
        # Sort by date within each horse group (newest first)
        df_sorted = df.sort_values(['race', 'horse_name', 'pp_race_date'], ascending=[True, True, False])
        
        def speed_features(group):
            # Ensure we have speed ratings
            speeds = group['pp_bris_speed_rating'].dropna()
            if len(speeds) == 0:
                return pd.Series({
                    'last_3_speed_avg': np.nan,
                    'last_speed': np.nan,
                    'best_speed_l10': np.nan,
                    'speed_consistency': np.nan,
                    'speed_trend': np.nan,
                    'speed_trajectory': 'unknown'
                })
            
            # Last 3 speed ratings average
            last_3_avg = speeds.head(3).mean() if len(speeds) >= 3 else speeds.mean()
            
            # Most recent speed
            last_speed = speeds.iloc[0] if len(speeds) > 0 else np.nan
            
            # Best speed in last 10 races
            best_speed = speeds.max()
            
            # Speed consistency (coefficient of variation)
            speed_consistency = speeds.std() / speeds.mean() if speeds.mean() > 0 else np.nan
            
            # Speed trend using linear regression on last 5 races
            if len(speeds) >= 3:
                recent_speeds = speeds.head(5).values
                x = np.arange(len(recent_speeds))
                slope, _, r_value, _, _ = stats.linregress(x, recent_speeds)
                speed_trend = slope  # Positive = improving, Negative = declining
                
                # Categorize trajectory
                if abs(r_value) > 0.5:  # Strong correlation
                    if slope > 1:
                        trajectory = 'improving'
                    elif slope < -1:
                        trajectory = 'declining'
                    else:
                        trajectory = 'stable'
                else:
                    trajectory = 'inconsistent'
            else:
                speed_trend = 0
                trajectory = 'insufficient_data'
            
            return pd.Series({
                'last_3_speed_avg': last_3_avg,
                'last_speed': last_speed,
                'best_speed_l10': best_speed,
                'speed_consistency': speed_consistency,
                'speed_trend': speed_trend,
                'speed_trajectory': trajectory
            })
        
        speed_features_df = df_sorted.groupby(['race', 'horse_name']).apply(speed_features).reset_index()
        return speed_features_df
    
    def calculate_distance_surface_features(self, past_df: pd.DataFrame, current_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate distance and surface specific features."""
        print("Calculating distance/surface specific features...")
        
        # Get today's race conditions for each horse
        race_conditions = current_df[['race', 'post_position', 'horse_name', 'distance_in_yards', 'surface']].copy()
        
        def distance_surface_features(row):
            race_num = row['race']
            horse = row['horse_name']
            today_distance = row['distance_in_yards']
            today_surface = row['surface']
            
            # Get this horse's past performances
            horse_past = past_df[(past_df['race'] == race_num) & (past_df['horse_name'] == horse)].copy()
            
            if len(horse_past) == 0:
                return pd.Series({
                    'best_speed_today_distance': np.nan,
                    'best_speed_today_surface': np.nan,
                    'distance_experience': 0,
                    'surface_experience': 0,
                    'distance_win_rate': np.nan,
                    'surface_win_rate': np.nan,
                    'distance_suitability': 0,
                    'surface_suitability': 0
                })
            
            # Distance-specific features with custom tolerance rules
            if today_distance <= 1650:
                # Short distances: 0.5 furlong tolerance (110 yards)
                distance_mask = abs(horse_past['pp_distance'] - today_distance) <= 110
            elif 1760 <= today_distance <= 1870:
                # Mid-range distances: only 1 furlong higher tolerance
                distance_mask = (horse_past['pp_distance'] >= today_distance) & (horse_past['pp_distance'] <= today_distance + 220)
            else:
                # Long distances (>1870): 1 furlong higher or lower tolerance
                distance_mask = abs(horse_past['pp_distance'] - today_distance) <= 220
            
            distance_races = horse_past[distance_mask]
            
            best_speed_distance = distance_races['pp_bris_speed_rating'].max() if len(distance_races) > 0 else np.nan
            distance_experience = len(distance_races)
            distance_wins = len(distance_races[distance_races['pp_finish_pos'] == 1])
            distance_win_rate = distance_wins / distance_experience if distance_experience > 0 else np.nan
            
            # Surface-specific features
            surface_mask = horse_past['pp_surface'] == today_surface
            surface_races = horse_past[surface_mask]
            
            best_speed_surface = surface_races['pp_bris_speed_rating'].max() if len(surface_races) > 0 else np.nan
            surface_experience = len(surface_races)
            surface_wins = len(surface_races[surface_races['pp_finish_pos'] == 1])
            surface_win_rate = surface_wins / surface_experience if surface_experience > 0 else np.nan
            
            # Suitability scores (0-100 scale)
            distance_suitability = min(100, distance_experience * 20 + (distance_win_rate * 50 if not pd.isna(distance_win_rate) else 0))
            surface_suitability = min(100, surface_experience * 15 + (surface_win_rate * 40 if not pd.isna(surface_win_rate) else 0))
            
            return pd.Series({
                'best_speed_today_distance': best_speed_distance,
                'best_speed_today_surface': best_speed_surface,
                'distance_experience': distance_experience,
                'surface_experience': surface_experience,
                'distance_win_rate': distance_win_rate,
                'surface_win_rate': surface_win_rate,
                'distance_suitability': distance_suitability,
                'surface_suitability': surface_suitability
            })
        
        features_df = race_conditions.apply(distance_surface_features, axis=1)
        result_df = pd.concat([race_conditions[['race', 'horse_name']], features_df], axis=1)
        
        return result_df
    
    def calculate_pace_features(self, df: pd.DataFrame, current_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate pace-related features converted to lengths."""
        print("Calculating pace features...")
        
        # Get race distance info to determine sprint vs route
        race_info = current_df[['race', 'distance_in_yards']].drop_duplicates()
        
        df_sorted = df.sort_values(['race', 'horse_name', 'pp_race_date'], ascending=[True, True, False])
        
        def pace_features(group):
            race_num = group['race'].iloc[0]
            today_distance = race_info[race_info['race'] == race_num]['distance_in_yards'].iloc[0]
            is_sprint = today_distance <= 1540  # Sprint = 1540 yards or less
            
            # Convert pace figures to lengths (2 points = 1 length)
            e1_lengths = group['pp_e1_pace'] / 2.0
            e2_lengths = group['pp_e2_pace'] / 2.0
            lp_lengths = group['pp_bris_late_pace'] / 2.0
            
            # Recent pace averages (last 3 races)
            recent_e1 = e1_lengths.head(3).mean()
            recent_e2 = e2_lengths.head(3).mean()
            recent_lp = lp_lengths.head(3).mean()
            
            # Best pace figures
            best_e1 = e1_lengths.max()
            best_e2 = e2_lengths.max()
            best_lp = lp_lengths.max()
            
            # Pace versatility (ability to rate - lower CV is better)
            e2_consistency = e2_lengths.std() / e2_lengths.mean() if e2_lengths.mean() > 0 else np.nan
            
            # Early pace pressure indicator (average E2 position in field)
            # Lower E2 pace = more pressure (faster early pace)
            avg_e2_pace = group['pp_e2_pace'].mean()
            
            # Late pace kick (improvement from E2 to Late Pace)
            pace_kick = group['pp_bris_late_pace'] - group['pp_e2_pace']
            avg_pace_kick = pace_kick.mean()
            
            # Pace style classification based on E2 focus
            if not pd.isna(recent_e2):
                if recent_e2 < 40:  # Very fast early pace
                    pace_style = 'speed'
                elif recent_e2 < 50:
                    pace_style = 'presser'
                elif recent_e2 < 60:
                    pace_style = 'stalker'
                else:
                    pace_style = 'closer'
            else:
                pace_style = 'unknown'
            
            return pd.Series({
                'recent_e1_lengths': recent_e1,
                'recent_e2_lengths': recent_e2,
                'recent_lp_lengths': recent_lp,
                'best_e1_lengths': best_e1,
                'best_e2_lengths': best_e2,
                'best_lp_lengths': best_lp,
                'e2_consistency': e2_consistency,
                'avg_e2_pace': avg_e2_pace,
                'avg_pace_kick': avg_pace_kick,
                'pace_style': pace_style,
                'is_sprint_race': is_sprint
            })
        
        pace_features_df = df_sorted.groupby(['race', 'horse_name']).apply(pace_features).reset_index()
        return pace_features_df
    
    def calculate_class_features(self, df: pd.DataFrame, current_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate class-related features using purse as proxy."""
        print("Calculating class features...")
        
        # Get today's purse for each race
        today_purse = current_df[['race', 'purse']].drop_duplicates()
        
        df_sorted = df.sort_values(['race', 'horse_name', 'pp_race_date'], ascending=[True, True, False])
        
        def class_features(group):
            race_num = group['race'].iloc[0]
            today_purse_value = today_purse[today_purse['race'] == race_num]['purse'].iloc[0]
            
            past_purses = group['pp_purse'].dropna()
            
            if len(past_purses) == 0:
                return pd.Series({
                    'avg_class_level': np.nan,
                    'best_class_level': np.nan,
                    'class_differential': np.nan,
                    'class_rise_drop': 'unknown',
                    'high_class_experience': 0,
                    'class_consistency': np.nan,
                    'last_3_class_avg': np.nan
                })
            
            # Average class level (purse)
            avg_class = past_purses.mean()
            best_class = past_purses.max()
            last_3_class = past_purses.head(3).mean()
            
            # Class differential (today vs average)
            class_diff = today_purse_value - avg_class
            
            # Classify class move
            if class_diff > avg_class * 0.2:  # 20% increase
                class_move = 'major_rise'
            elif class_diff > avg_class * 0.1:  # 10% increase
                class_move = 'minor_rise'
            elif class_diff < -avg_class * 0.2:  # 20% decrease
                class_move = 'major_drop'
            elif class_diff < -avg_class * 0.1:  # 10% decrease
                class_move = 'minor_drop'
            else:
                class_move = 'similar'
            
            # High class experience (races in top 25% of purses)
            high_class_threshold = np.percentile(past_purses, 75)
            high_class_exp = len(past_purses[past_purses >= high_class_threshold])
            
            # Class consistency
            class_consistency = past_purses.std() / past_purses.mean() if past_purses.mean() > 0 else np.nan
            
            return pd.Series({
                'avg_class_level': avg_class,
                'best_class_level': best_class,
                'class_differential': class_diff,
                'class_rise_drop': class_move,
                'high_class_experience': high_class_exp,
                'class_consistency': class_consistency,
                'last_3_class_avg': last_3_class
            })
        
        class_features_df = df_sorted.groupby(['race', 'horse_name']).apply(class_features).reset_index()
        return class_features_df
    
    def calculate_condition_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate track condition and context features."""
        print("Calculating track condition features...")
        
        df_sorted = df.sort_values(['race', 'horse_name', 'pp_race_date'], ascending=[True, True, False])
        
        def condition_features(group):
            conditions = group['pp_track_condition'].dropna()
            
            if len(conditions) == 0:
                return pd.Series({
                    'fast_track_experience': 0,
                    'off_track_experience': 0,
                    'condition_versatility': 0,
                    'recent_condition_performance': np.nan
                })
            
            # Track condition experience
            fast_exp = len(conditions[conditions.isin(['FT', 'Fast'])])
            off_exp = len(conditions[~conditions.isin(['FT', 'Fast'])])
            
            # Condition versatility (has run on different surfaces)
            unique_conditions = len(conditions.unique())
            versatility = min(100, unique_conditions * 25)  # 0-100 scale
            
            # Recent condition performance (speed rating on last similar condition)
            last_condition = conditions.iloc[0] if len(conditions) > 0 else None
            recent_performance = np.nan
            
            if last_condition:
                same_condition_races = group[group['pp_track_condition'] == last_condition]
                if len(same_condition_races) > 0:
                    recent_performance = same_condition_races['pp_bris_speed_rating'].iloc[0]
            
            return pd.Series({
                'fast_track_experience': fast_exp,
                'off_track_experience': off_exp,
                'condition_versatility': versatility,
                'recent_condition_performance': recent_performance
            })
        
        condition_features_df = df_sorted.groupby(['race', 'horse_name']).apply(condition_features).reset_index()
        return condition_features_df
    
    def create_composite_features(self, features_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create composite features by combining individual feature sets."""
        print("Creating composite features...")
        
        # Start with base features
        base_cols = ['race', 'horse_name']
        final_df = features_dict['speed'][base_cols].copy()
        
        # Merge all feature sets
        for feature_name, feature_df in features_dict.items():
            if feature_name != 'speed':  # Skip base since we started with it
                final_df = final_df.merge(
                    feature_df, 
                    on=['race', 'horse_name'], 
                    how='left'
                )
        
        # Add speed features
        speed_features = features_dict['speed'].drop(columns=['race', 'horse_name'])
        final_df = pd.concat([final_df, speed_features], axis=1)
        
        # Create composite scores
        final_df['speed_class_combo'] = (
            final_df['last_3_speed_avg'].fillna(0) * 0.7 + 
            final_df['class_differential'].fillna(0) * 0.3
        )
        
        final_df['pace_suitability'] = (
            final_df['distance_suitability'].fillna(0) * 0.4 +
            final_df['surface_suitability'].fillna(0) * 0.3 +
            final_df['e2_consistency'].fillna(1) * -20 + 50  # Lower consistency = higher score
        )
        
        final_df['overall_rating'] = (
            final_df['last_3_speed_avg'].fillna(0) * 0.3 +
            final_df['pace_suitability'].fillna(0) * 0.25 +
            final_df['distance_suitability'].fillna(0) * 0.2 +
            final_df['class_differential'].fillna(0) * 0.15 +
            final_df['condition_versatility'].fillna(0) * 0.1
        )
        
        return final_df
    
    def engineer_all_features(self) -> pd.DataFrame:
        """Main function to engineer all features."""
        print("=== HORSE RACING FEATURE ENGINEERING ===")
        print("=" * 50)
        
        # Load data
        past_df, current_df = self.load_data()
        
        # Calculate all feature sets
        features = {}
        
        features['speed'] = self.calculate_recent_speed_features(past_df)
        features['distance_surface'] = self.calculate_distance_surface_features(past_df, current_df)
        features['pace'] = self.calculate_pace_features(past_df, current_df)
        features['class'] = self.calculate_class_features(past_df, current_df)
        features['condition'] = self.calculate_condition_features(past_df)
        
        # Combine all features
        final_features = self.create_composite_features(features)
        
        print(f"\nFeature engineering complete!")
        print(f"Final dataset shape: {final_features.shape}")
        print(f"Features created: {final_features.shape[1] - 2}")  # Subtract race and horse_name
        
        return final_features

def main():
    """Example usage of the feature engineering system."""
    # Initialize feature engineer
    processed_data_path = PROCESSED_DATA_DIR
    engineer = HorseRacingFeatureEngineer(processed_data_path)
    
    # Engineer all features
    features_df = engineer.engineer_all_features()
    
    # Save results
    output_path = processed_data_path / "racing_features_for_modeling.parquet"
    features_df.to_parquet(output_path, index=False)
    print(f"\nFeatures saved to: {output_path}")
    
    # Display sample results
    print("\nSample of engineered features:")
    print("=" * 50)
    
    # Show key columns for first few horses
    display_cols = [
        'race', 'horse_name', 'last_3_speed_avg', 'speed_trajectory',
        'best_speed_today_distance', 'distance_suitability', 'pace_style',
        'class_rise_drop', 'overall_rating'
    ]
    
    available_cols = [col for col in display_cols if col in features_df.columns]
    print(features_df[available_cols].head(10).to_string(index=False))
    
    # Summary statistics
    print(f"\nFeature Summary:")
    print("-" * 30)
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    print(f"Numeric features: {len(numeric_cols)}")
    print(f"Categorical features: {len(features_df.columns) - len(numeric_cols) - 2}")
    
    return features_df

if __name__ == "__main__":
    features_df = main()