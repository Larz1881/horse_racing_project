# src/transformers/advanced_fitness_metrics.py
"""
Advanced Form & Fitness Metrics Module

Calculates sophisticated fitness indicators:
1. Recovery Rate Index (RRI)
2. Form Momentum Score
3. Cardiovascular Fitness Proxy
4. Sectional Improvement Index
5. Energy Distribution Profile
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Dict, Tuple, Optional
from config.settings import (
    PROCESSED_DATA_DIR,
    CURRENT_RACE_INFO,
    PAST_STARTS_LONG,
    WORKOUTS_LONG,
)
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedFitnessMetrics:
    """Calculate advanced fitness metrics for horses"""
    
    def __init__(self, current_race_path: str, past_starts_path: str, workouts_path: str = None):
        """
        Initialize with paths to parquet files
        
        Args:
            current_race_path: Path to current_race_info.parquet
            past_starts_path: Path to past_starts_long_format.parquet
            workouts_path: Path to workouts_long_format.parquet (optional)
        """
        self.current_df = pd.read_parquet(current_race_path)
        self.past_df = pd.read_parquet(past_starts_path)
        
        # Load workouts if provided
        self.workouts_df = pd.read_parquet(workouts_path) if workouts_path else None
        
        # Ensure proper date formatting
        self.past_df['pp_race_date'] = pd.to_datetime(self.past_df['pp_race_date'])
        
        # Sort past performances properly (most recent first)
        self.past_df = self.past_df.sort_values(
            ['race', 'horse_name', 'pp_race_date'], 
            ascending=[True, True, False]
        )
        
        # Calculate horse age
        current_year = datetime.now().year
        self.current_df['age'] = current_year - self.current_df['year_of_birth']
        
    def calculate_recovery_rate_index(self) -> pd.DataFrame:
        """
        Calculate Recovery Rate Index (RRI)
        Analyzes days between races vs performance decline
        
        Returns:
            DataFrame with RRI metrics per horse
        """
        logger.info("Calculating Recovery Rate Index...")
        
        # Get last 5 races for each horse
        recent_races = self.past_df.groupby(['race', 'horse_name']).head(5)
        
        # Calculate performance changes with days off
        rri_data = []
        
        for (race, horse), group in recent_races.groupby(['race', 'horse_name']):
            if len(group) < 2:
                continue
                
            # Get days off and speed ratings
            days_off = group['pp_days_since_prev'].values[:-1]
            speed_ratings = group['pp_bris_speed_rating'].values
            
            # Calculate speed rating changes
            rating_changes = np.diff(speed_ratings)
            
            # Filter out invalid values
            valid_mask = (days_off > 0) & (~np.isnan(days_off)) & (~np.isnan(rating_changes))
            
            if not any(valid_mask):
                continue
            
            days_off_valid = days_off[valid_mask]
            rating_changes_valid = rating_changes[valid_mask]
            
            # Calculate optimal recovery (where performance improved most)
            if len(days_off_valid) > 0:
                # Find optimal days off
                if len(rating_changes_valid) > 0:
                    best_improvement_idx = np.argmax(rating_changes_valid)
                    optimal_days = days_off_valid[best_improvement_idx]
                    best_improvement = rating_changes_valid[best_improvement_idx]
                else:
                    optimal_days = None
                    best_improvement = 0
                
                avg_recovery_impact = np.mean(rating_changes_valid / days_off_valid) if len(days_off_valid) > 0 else 0
                
                # Get current days since last race
                current_days_off = self.current_df[
                    (self.current_df['race'] == race) & 
                    (self.current_df['horse_name'] == horse)
                ]['of_days_since_last_race'].values
                
                current_days_off = current_days_off[0] if len(current_days_off) > 0 else None
                
                # Calculate recovery score (0-100)
                if optimal_days and current_days_off and optimal_days > 0:
                    # Perfect recovery = 100, decreases with distance from optimal
                    recovery_score = 100 * np.exp(-abs(current_days_off - optimal_days) / optimal_days)
                else:
                    recovery_score = 50  # neutral score
                
                # Age-adjusted recovery (older horses may need more time)
                age = self.current_df[
                    (self.current_df['race'] == race) & 
                    (self.current_df['horse_name'] == horse)
                ]['age'].values
                
                if len(age) > 0 and age[0] > 5:
                    # Older horses get bonus for adequate rest
                    if current_days_off and current_days_off > 21:
                        recovery_score = min(100, recovery_score * 1.1)
                
                rri_data.append({
                    'race': race,
                    'horse_name': horse,
                    'optimal_recovery_days': optimal_days,
                    'current_days_off': current_days_off,
                    'avg_recovery_impact': avg_recovery_impact,
                    'best_improvement_with_rest': best_improvement,
                    'recovery_score': recovery_score
                })
        
        return pd.DataFrame(rri_data)
    
    def calculate_form_momentum(self) -> pd.DataFrame:
        """
        Calculate Form Momentum Score
        Weights recent performances more heavily
        
        Returns:
            DataFrame with form momentum metrics
        """
        logger.info("Calculating Form Momentum Score...")
        
        momentum_data = []
        
        for (race, horse), group in self.past_df.groupby(['race', 'horse_name']):
            # Get last 6 races
            recent = group.head(6).copy()
            
            if len(recent) < 3:
                continue
            
            # Calculate days ago for each race
            most_recent_date = recent['pp_race_date'].iloc[0]
            recent['days_ago'] = (most_recent_date - recent['pp_race_date']).dt.days
            
            # Exponential decay weights (more recent = higher weight)
            recent['weight'] = np.exp(-recent['days_ago'] / 30)  # 30-day half-life
            
            # Normalize weights
            recent['weight'] = recent['weight'] / recent['weight'].sum()
            
            # 1. Speed rating trend
            speed_ratings = recent['pp_bris_speed_rating'].values
            valid_speed = ~np.isnan(speed_ratings)
            
            if sum(valid_speed) >= 2:
                x = np.arange(len(speed_ratings))[valid_speed]
                y = speed_ratings[valid_speed]
                weights = recent['weight'].values[valid_speed]
                
                # Weighted linear regression for trend
                if len(x) > 1:
                    coef = np.polyfit(x, y, 1, w=weights)
                    speed_trend = -coef[0]  # Negative because recent races have higher index
                else:
                    speed_trend = 0
            else:
                speed_trend = 0
            
            # 2. Beaten lengths improvement
            beaten_lengths = recent['pp_finish_lengths_behind'].fillna(0).values
            if len(beaten_lengths) >= 2:
                # Improvement = reduction in beaten lengths
                beaten_trend = -np.average(np.diff(beaten_lengths), weights=recent['weight'].values[:-1])
            else:
                beaten_trend = 0
            
            # 3. Late pace improvement
            late_pace = recent['pp_bris_late_pace'].values
            valid_late = ~np.isnan(late_pace)
            
            if sum(valid_late) >= 2:
                late_pace_trend = np.average(
                    np.diff(late_pace[valid_late]), 
                    weights=recent['weight'].values[valid_late][:-1]
                )
            else:
                late_pace_trend = 0
            
            # 4. Position improvement (gaining ground)
            start_to_finish = recent['pp_pos_gain_start_to_finish'].values
            valid_gain = ~np.isnan(start_to_finish)
            
            if sum(valid_gain) >= 2:
                position_gain_trend = np.average(
                    start_to_finish[valid_gain],
                    weights=recent['weight'].values[valid_gain]
                )
            else:
                position_gain_trend = 0
            
            # Composite momentum score (-100 to 100)
            momentum_score = (
                speed_trend * 10 +  # Scale speed trend
                beaten_trend * 5 +  # Scale beaten lengths improvement
                late_pace_trend * 2 + # Scale late pace improvement
                position_gain_trend * 3  # Scale position gains
            )
            momentum_score = np.clip(momentum_score, -100, 100)
            
            momentum_data.append({
                'race': race,
                'horse_name': horse,
                'speed_trend': speed_trend,
                'beaten_lengths_trend': beaten_trend,
                'late_pace_trend': late_pace_trend,
                'position_gain_trend': position_gain_trend,
                'form_momentum_score': momentum_score,
                'races_analyzed': len(recent)
            })
        
        return pd.DataFrame(momentum_data)
    
    def calculate_cardiovascular_fitness(self) -> pd.DataFrame:
        """
        Calculate Cardiovascular Fitness Proxy
        Based on finish speed sustainability and deceleration rates
        
        Returns:
            DataFrame with cardiovascular fitness metrics
        """
        logger.info("Calculating Cardiovascular Fitness Proxy...")
        
        cardio_data = []
        
        for (race, horse), group in self.past_df.groupby(['race', 'horse_name']):
            recent = group.head(5)
            
            cardio_metrics = []
            
            for idx, row in recent.iterrows():
                # Calculate deceleration in final fraction
                if pd.notna(row['pp_bris_late_pace']) and pd.notna(row['pp_bris_6f_pace']):
                    # Higher late pace relative to early = better fitness
                    pace_sustainability = row['pp_bris_late_pace'] - row['pp_bris_6f_pace']
                else:
                    pace_sustainability = None
                
                # Use the engineered turn time if available
                if pd.notna(row.get('pp_turn_time', None)):
                    turn_efficiency = row['pp_turn_time']
                else:
                    turn_efficiency = None
                
                # Final fraction deceleration (for routes)
                if pd.notna(row['pp_split_10f_finish_secs']) and row['pp_split_10f_finish_secs'] > 0:
                    # Compare final fraction to average of middle fractions
                    middle_fractions = [
                        row.get('pp_split_4f_6f_secs', np.nan), 
                        row.get('pp_split_6f_8f_secs', np.nan)
                    ]
                    valid_middle = [f for f in middle_fractions if pd.notna(f) and f > 0]
                    
                    if valid_middle:
                        avg_middle_pace = np.mean(valid_middle)
                        final_fraction_ratio = row['pp_split_10f_finish_secs'] / avg_middle_pace
                        deceleration = (final_fraction_ratio - 1) * 100  # Percentage slowdown
                    else:
                        deceleration = None
                else:
                    deceleration = None
                
                # Stretch run efficiency
                if pd.notna(row['pp_stretch_lengths_behind']) and pd.notna(row['pp_second_call_lengths_behind']):
                    # Positive = gained ground, negative = lost ground
                    stretch_efficiency = row['pp_second_call_lengths_behind'] - row['pp_stretch_lengths_behind']
                else:
                    stretch_efficiency = None
                
                if any(x is not None for x in [pace_sustainability, deceleration, stretch_efficiency]):
                    cardio_metrics.append({
                        'pace_sustainability': pace_sustainability,
                        'final_deceleration': deceleration,
                        'turn_efficiency': turn_efficiency,
                        'stretch_efficiency': stretch_efficiency
                    })
            
            if cardio_metrics:
                # Average metrics - convert None to np.nan for proper handling
                avg_sustainability = np.nanmean([m['pace_sustainability'] if m['pace_sustainability'] is not None else np.nan for m in cardio_metrics])
                avg_deceleration = np.nanmean([m['final_deceleration'] if m['final_deceleration'] is not None else np.nan for m in cardio_metrics])
                avg_stretch_eff = np.nanmean([m['stretch_efficiency'] if m['stretch_efficiency'] is not None else np.nan for m in cardio_metrics])
                
                # Fitness score (0-100, higher is better)
                fitness_score = 50  # Base score
                
                if not np.isnan(avg_sustainability):
                    fitness_score += avg_sustainability * 2  # Positive is good
                
                if not np.isnan(avg_deceleration):
                    fitness_score -= avg_deceleration * 0.5  # Less deceleration is better
                
                if not np.isnan(avg_stretch_eff):
                    fitness_score += avg_stretch_eff * 3  # Gaining ground late is good
                
                fitness_score = np.clip(fitness_score, 0, 100)
                
                cardio_data.append({
                    'race': race,
                    'horse_name': horse,
                    'avg_pace_sustainability': avg_sustainability,
                    'avg_final_deceleration': avg_deceleration,
                    'avg_stretch_efficiency': avg_stretch_eff,
                    'cardiovascular_fitness_score': fitness_score
                })
        
        return pd.DataFrame(cardio_data)
    
    def calculate_sectional_improvement(self) -> pd.DataFrame:
        """
        Calculate Sectional Improvement Index
        Tracks improvement in 2f sectional times
        
        Returns:
            DataFrame with sectional improvement metrics
        """
        logger.info("Calculating Sectional Improvement Index...")
        
        sectional_data = []
        
        for (race, horse), group in self.past_df.groupby(['race', 'horse_name']):
            recent = group.head(6)
            
            # Collect all sectional times
            sectionals = {
                '0-2f': recent['pp_split_0_2f_secs'].values,
                '2f-4f': recent['pp_split_2f_4f_secs'].values,
                '4f-6f': recent['pp_split_4f_6f_secs'].values,
                '6f-8f': recent['pp_split_6f_8f_secs'].values,
            }
            
            improvements = {}
            
            for section, times in sectionals.items():
                # Filter valid times
                valid_times = times[~np.isnan(times) & (times > 0)]
                
                if len(valid_times) >= 2:
                    # Calculate improvement (negative diff = improvement)
                    time_diffs = np.diff(valid_times)
                    avg_improvement = -np.mean(time_diffs)  # Positive = getting faster
                    
                    # Best recent vs average
                    if len(valid_times) >= 3:
                        best_recent = np.min(valid_times[:2])
                        avg_older = np.mean(valid_times[2:])
                        improvement_pct = (avg_older - best_recent) / avg_older * 100
                    else:
                        improvement_pct = avg_improvement
                    
                    improvements[f'{section}_improvement'] = improvement_pct
            
            # Also check engineered pace features
            if 'avg_best2_recent_pp_e1_pace' in recent.columns:
                # Compare current form E1 pace to historical
                current_e1 = recent['avg_best2_recent_pp_e1_pace'].iloc[0]
                historical_e1 = recent['pp_e1_pace'].iloc[1:].mean()
                
                if pd.notna(current_e1) and pd.notna(historical_e1):
                    e1_improvement = ((current_e1 - historical_e1) / historical_e1) * 100
                    improvements['e1_pace_improvement'] = e1_improvement
            
            if improvements:
                # Overall sectional improvement score
                overall_improvement = np.mean(list(improvements.values()))
                
                sectional_data.append({
                    'race': race,
                    'horse_name': horse,
                    **improvements,
                    'sectional_improvement_score': overall_improvement
                })
        
        return pd.DataFrame(sectional_data)
    
    def calculate_energy_distribution(self) -> pd.DataFrame:
        """
        Calculate Energy Distribution Profile
        Analyzes how horses distribute effort throughout race
        
        Returns:
            DataFrame with energy distribution metrics
        """
        logger.info("Calculating Energy Distribution Profile...")
        
        energy_data = []
        
        for (race, horse), group in self.past_df.groupby(['race', 'horse_name']):
            recent = group.head(5)
            
            energy_profiles = []
            
            for idx, row in recent.iterrows():
                # Get positions at each call
                positions = {
                    'start': row['pp_start_call_pos'],
                    'first': row['pp_first_call_pos'],
                    'second': row['pp_second_call_pos'],
                    'stretch': row['pp_stretch_pos'],
                    'finish': row['pp_finish_pos']
                }
                
                # Filter out invalid positions
                valid_positions = {}
                for k, v in positions.items():
                    if pd.notna(v) and str(v).replace('.', '').isdigit():
                        valid_positions[k] = float(v)
                
                if len(valid_positions) < 3:
                    continue
                
                # Calculate position changes (negative = moving forward)
                pos_list = list(valid_positions.values())
                pos_changes = np.diff(pos_list)
                
                # Energy distribution phases
                total_movement = np.sum(np.abs(pos_changes))
                
                if total_movement > 0:
                    # Calculate percentage of energy used in each phase
                    energy_phases = np.abs(pos_changes) / total_movement * 100
                    
                    # Categorize energy usage
                    if len(energy_phases) >= 2:
                        early_energy = np.sum(energy_phases[:len(energy_phases)//2])
                        late_energy = np.sum(energy_phases[len(energy_phases)//2:])
                    else:
                        early_energy = energy_phases[0] if len(energy_phases) > 0 else 50
                        late_energy = 100 - early_energy
                    
                    # Use position gains if available
                    efficiency_score = 0
                    if pd.notna(row['pp_pos_gain_start_to_finish']):
                        # Positive gains with less movement = efficient
                        efficiency_score = row['pp_pos_gain_start_to_finish'] * 10
                    
                    energy_profiles.append({
                        'early_energy_pct': early_energy,
                        'late_energy_pct': late_energy,
                        'energy_efficiency': efficiency_score,
                        'total_movement': total_movement
                    })
            
            if energy_profiles:
                # Average energy distribution
                avg_early = np.mean([p['early_energy_pct'] for p in energy_profiles])
                avg_late = np.mean([p['late_energy_pct'] for p in energy_profiles])
                avg_efficiency = np.mean([p['energy_efficiency'] for p in energy_profiles])
                avg_movement = np.mean([p['total_movement'] for p in energy_profiles])
                
                # Classify energy distribution type
                if avg_early > 60:
                    energy_type = 'front_loaded'
                elif avg_late > 60:
                    energy_type = 'closer'
                else:
                    energy_type = 'even_paced'
                
                # Check against BRIS run style
                bris_style = self.current_df[
                    (self.current_df['race'] == race) & 
                    (self.current_df['horse_name'] == horse)
                ]['bris_run_style_designation'].values
                
                if len(bris_style) > 0:
                    style_match = self._match_energy_to_style(energy_type, bris_style[0])
                else:
                    style_match = True
                
                energy_data.append({
                    'race': race,
                    'horse_name': horse,
                    'avg_early_energy_pct': avg_early,
                    'avg_late_energy_pct': avg_late,
                    'energy_efficiency_score': avg_efficiency + (10 if style_match else 0),
                    'energy_distribution_type': energy_type,
                    'avg_position_changes': avg_movement,
                    'style_consistency': style_match
                })
        
        return pd.DataFrame(energy_data)
    
    def _match_energy_to_style(self, energy_type: str, bris_style: str) -> bool:
        """Check if energy distribution matches BRIS run style"""
        if pd.isna(bris_style):
            return True
        
        bris_style = str(bris_style).upper()
        
        # Map BRIS styles to energy types
        front_styles = ['E', 'E/P', 'P']
        closer_styles = ['S', 'SS']
        
        if energy_type == 'front_loaded' and any(s in bris_style for s in front_styles):
            return True
        elif energy_type == 'closer' and any(s in bris_style for s in closer_styles):
            return True
        elif energy_type == 'even_paced' and bris_style in ['P/S', 'PS']:
            return True
        
        return False
    
    def calculate_workout_fitness(self) -> pd.DataFrame:
        """
        Calculate workout-based fitness indicators
        
        Returns:
            DataFrame with workout fitness metrics
        """
        if self.workouts_df is None:
            logger.warning("No workout data available")
            return pd.DataFrame()
        
        logger.info("Calculating Workout Fitness Indicators...")
        
        workout_data = []
        
        for (race, horse), group in self.workouts_df.groupby(['race', 'horse_name']):
            # Get recent workouts
            recent_works = group.sort_values('work_date', ascending=False).head(6)
            
            if len(recent_works) == 0:
                continue
            
            # Calculate workout metrics
            total_works = len(recent_works)
            
            # Bullet works (rank = 1)
            bullet_works = (recent_works['work_rank'] == 1).sum()
            bullet_pct = (bullet_works / total_works) * 100 if total_works > 0 else 0
            
            # Average workout time per furlong
            work_times = []
            for idx, row in recent_works.iterrows():
                if pd.notna(row['work_time']) and pd.notna(row['work_distance']) and row['work_distance'] > 0:
                    # Convert distance to furlongs
                    furlongs = row['work_distance'] / 220  # 220 yards per furlong
                    time_per_furlong = abs(row['work_time']) / furlongs  # abs for bullet works
                    work_times.append(time_per_furlong)
            
            avg_time_per_furlong = np.mean(work_times) if work_times else None
            
            # Workout frequency (days between works)
            if len(recent_works) > 1:
                work_dates = pd.to_datetime(recent_works['work_date'])
                work_intervals = work_dates.diff().dt.days.dropna()
                avg_days_between = work_intervals.mean()
            else:
                avg_days_between = None
            
            # Workout pattern score
            workout_score = 50  # Base
            if bullet_pct > 30:
                workout_score += 20
            if avg_time_per_furlong and avg_time_per_furlong < 12:  # Fast works
                workout_score += 15
            if avg_days_between and 4 <= avg_days_between <= 7:  # Regular pattern
                workout_score += 15
            
            workout_data.append({
                'race': race,
                'horse_name': horse,
                'recent_workout_count': total_works,
                'bullet_work_pct': bullet_pct,
                'avg_work_time_per_furlong': avg_time_per_furlong,
                'avg_days_between_works': avg_days_between,
                'workout_fitness_score': workout_score
            })
        
        return pd.DataFrame(workout_data)
    
    def calculate_all_metrics(self) -> pd.DataFrame:
        """
        Calculate all fitness metrics and combine into single DataFrame
        
        Returns:
            DataFrame with all fitness metrics for each horse
        """
        logger.info("Calculating all advanced fitness metrics...")
        
        # Calculate individual components
        rri_df = self.calculate_recovery_rate_index()
        momentum_df = self.calculate_form_momentum()
        cardio_df = self.calculate_cardiovascular_fitness()
        sectional_df = self.calculate_sectional_improvement()
        energy_df = self.calculate_energy_distribution()
        
        # Start with basic horse info
        fitness_df = self.current_df[['race', 'horse_name', 'age', 'track', 
                                      'post_position', 'morn_line_odds_if_available']].copy()
        
        # Add current form indicators
        form_columns = [
            'bris_prime_power_rating',
            'best_bris_speed_life',
            'best_bris_speed_fast_track',
            'best_bris_speed_turf',
            'best_bris_speed_distance',
            'of_days_since_last_race',
            'bris_run_style_designation',
            'quirin_style_speed_points',
            'lifetime_win_pct',
            'current_year_win_pct'
        ]
        
        fitness_df = fitness_df.merge(
            self.current_df[['race', 'horse_name'] + form_columns],
            on=['race', 'horse_name'],
            how='left'
        )
        
        # Merge calculated metrics
        for df in [rri_df, momentum_df, cardio_df, sectional_df, energy_df]:
            if not df.empty:
                fitness_df = fitness_df.merge(df, on=['race', 'horse_name'], how='left')
        
        # Add workout metrics if available
        if self.workouts_df is not None:
            workout_df = self.calculate_workout_fitness()
            if not workout_df.empty:
                fitness_df = fitness_df.merge(workout_df, on=['race', 'horse_name'], how='left')
        
        # Calculate composite fitness score
        score_columns = [
            'recovery_score',
            'form_momentum_score',
            'cardiovascular_fitness_score',
            'sectional_improvement_score',
            'energy_efficiency_score'
        ]
        
        if 'workout_fitness_score' in fitness_df.columns:
            score_columns.append('workout_fitness_score')
        
        # Fill NaN with neutral score (50) and calculate mean
        fitness_df['composite_fitness_score'] = fitness_df[score_columns].fillna(50).mean(axis=1)
        
        # Add fitness categories
        fitness_df['fitness_category'] = pd.cut(
            fitness_df['composite_fitness_score'],
            bins=[0, 40, 60, 80, 100],
            labels=['Poor', 'Fair', 'Good', 'Excellent']
        )
        
        # Add fitness rank within race
        fitness_df['fitness_rank'] = fitness_df.groupby('race')['composite_fitness_score'].rank(
            ascending=False, method='min'
        )
        
        # Add improvement flag
        fitness_df['improving'] = (fitness_df['form_momentum_score'] > 10) & \
                                 (fitness_df['sectional_improvement_score'] > 0)
        
        logger.info(f"Calculated fitness metrics for {len(fitness_df)} horses")
        
        return fitness_df


def main():
    """Main function to run fitness calculations"""
    
    # Set up paths
    base_path = PROCESSED_DATA_DIR
    current_race_path = CURRENT_RACE_INFO
    past_starts_path = PAST_STARTS_LONG
    workouts_path = WORKOUTS_LONG
    
    # Check if files exist
    if not current_race_path.exists():
        logger.error(f"File not found: {current_race_path}")
        return
    
    if not past_starts_path.exists():
        logger.error(f"File not found: {past_starts_path}")
        return
    
    # Initialize calculator
    calculator = AdvancedFitnessMetrics(
        str(current_race_path), 
        str(past_starts_path),
        str(workouts_path) if workouts_path.exists() else None
    )
    
    # Calculate all metrics
    fitness_df = calculator.calculate_all_metrics()
    
    # Save results
    output_path = base_path / 'advanced_fitness_metrics.parquet'
    fitness_df.to_parquet(output_path, index=False)
    logger.info(f"Saved fitness metrics to {output_path}")
    
    # Display summary
    print("\n" + "="*60)
    print("FITNESS METRICS SUMMARY")
    print("="*60)
    print(f"Total horses analyzed: {len(fitness_df)}")
    
    print(f"\nFitness Categories:")
    print(fitness_df['fitness_category'].value_counts().sort_index())
    
    print(f"\nTop 10 Horses by Composite Fitness Score:")
    top_horses = fitness_df.nlargest(10, 'composite_fitness_score')[
        ['race', 'horse_name', 'post_position', 'composite_fitness_score', 
         'fitness_category', 'improving', 'form_momentum_score']
    ]
    print(top_horses.to_string(index=False))
    
    print(f"\nImproving Horses:")
    improving = fitness_df[fitness_df['improving'] == True]
    print(f"Count: {len(improving)}")
    if len(improving) > 0:
        print(improving[['race', 'horse_name', 'form_momentum_score', 
                        'sectional_improvement_score']].head(10).to_string(index=False))
    
    # Save summary report
    summary_path = base_path / 'fitness_metrics_summary.csv'
    summary_df = fitness_df[[
        'race', 'horse_name', 'post_position', 'age', 'composite_fitness_score',
        'fitness_category', 'fitness_rank', 'improving', 'recovery_score',
        'form_momentum_score', 'cardiovascular_fitness_score',
        'sectional_improvement_score', 'energy_efficiency_score'
    ]].sort_values(['race', 'fitness_rank'])
    
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary report to {summary_path}")


if __name__ == '__main__':
    main()