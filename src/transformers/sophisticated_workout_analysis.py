# src/transformers/sophisticated_workout_analysis.py
"""
Sophisticated Workout Analysis Module

Analyzes workout patterns to identify:
1. Trainer-specific workout patterns for winning horses
2. Workout quality scoring
3. Work-to-race performance translation
4. Trainer intent signals
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SophisticatedWorkoutAnalysis:
    """Analyze workout patterns for performance prediction"""
    
    def __init__(self, workouts_path: str, current_race_path: str, past_starts_path: str):
        """
        Initialize with paths to parquet files
        
        Args:
            workouts_path: Path to workouts_long_format.parquet
            current_race_path: Path to current_race_info.parquet
            past_starts_path: Path to past_starts_long_format.parquet
        """
        self.workouts_df = pd.read_parquet(workouts_path)
        self.current_df = pd.read_parquet(current_race_path)
        self.past_df = pd.read_parquet(past_starts_path)
        
        # Ensure proper date formatting
        self.workouts_df['work_date'] = pd.to_datetime(self.workouts_df['work_date'])
        self.past_df['pp_race_date'] = pd.to_datetime(self.past_df['pp_race_date'])
        
        # Add trainer info to workouts
        self._add_trainer_info()
        
    def _add_trainer_info(self):
        """Add trainer information to workout data"""
        trainer_info = self.current_df[['race', 'horse_name', 'today_s_trainer']].copy()
        self.workouts_df = self.workouts_df.merge(
            trainer_info,
            on=['race', 'horse_name'],
            how='left'
        )
    
    def analyze_trainer_workout_patterns(self) -> pd.DataFrame:
        """
        Identify trainer-specific workout patterns for winning horses
        
        Returns:
            DataFrame with trainer pattern analysis
        """
        logger.info("Analyzing trainer-specific workout patterns...")
        
        # First, identify winning performances from past starts
        winners = self.past_df[self.past_df['pp_finish_pos'] == 1].copy()
        
        if len(winners) == 0:
            logger.warning("No winning performances found in past data")
            return pd.DataFrame()
        
        # For each winner, look at their workout pattern before the win
        trainer_patterns = []
        
        for idx, win_row in winners.iterrows():
            horse = win_row['horse_name']
            race_date = win_row['pp_race_date']
            
            # Get workouts for this horse before the winning race
            horse_works = self.workouts_df[
                (self.workouts_df['horse_name'] == horse) &
                (self.workouts_df['work_date'] < race_date) &
                (self.workouts_df['work_date'] >= race_date - timedelta(days=60))  # 60 days before
            ].sort_values('work_date', ascending=False)
            
            if len(horse_works) == 0:
                continue
            
            trainer = horse_works['today_s_trainer'].iloc[0] if 'today_s_trainer' in horse_works.columns else 'Unknown'
            
            # Analyze workout pattern
            pattern_data = {
                'trainer': trainer,
                'days_to_race': (race_date - horse_works['work_date'].iloc[0]).days if len(horse_works) > 0 else None,
                'num_works_30_days': len(horse_works[horse_works['work_date'] >= race_date - timedelta(days=30)]),
                'num_works_14_days': len(horse_works[horse_works['work_date'] >= race_date - timedelta(days=14)]),
                'num_works_7_days': len(horse_works[horse_works['work_date'] >= race_date - timedelta(days=7)]),
                'had_bullet_work': (horse_works['work_rank'] == 1).any(),
                'bullet_days_before': None,
                'avg_work_distance': horse_works['work_distance'].mean(),
                'longest_work': horse_works['work_distance'].max(),
                'had_gate_work': horse_works['work_description'].str.contains('g', na=False).any(),
                'work_pattern_type': None
            }
            
            # Days before race of bullet work
            bullet_works = horse_works[horse_works['work_rank'] == 1]
            if len(bullet_works) > 0:
                pattern_data['bullet_days_before'] = (race_date - bullet_works['work_date'].iloc[0]).days
            
            # Classify workout pattern
            if pattern_data['num_works_7_days'] >= 1:
                pattern_data['work_pattern_type'] = 'maintenance'
            elif pattern_data['num_works_14_days'] >= 2:
                pattern_data['work_pattern_type'] = 'sharpening'
            elif pattern_data['num_works_30_days'] >= 3:
                pattern_data['work_pattern_type'] = 'building'
            else:
                pattern_data['work_pattern_type'] = 'light'
            
            trainer_patterns.append(pattern_data)
        
        # Aggregate by trainer
        patterns_df = pd.DataFrame(trainer_patterns)
        
        if len(patterns_df) == 0:
            return pd.DataFrame()
        
        trainer_summary = patterns_df.groupby('trainer').agg({
            'days_to_race': ['mean', 'std'],
            'num_works_30_days': 'mean',
            'num_works_14_days': 'mean',
            'had_bullet_work': ['sum', 'mean'],  # count and percentage
            'bullet_days_before': 'mean',
            'avg_work_distance': 'mean',
            'had_gate_work': 'mean',
            'work_pattern_type': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown'
        }).round(2)
        
        trainer_summary.columns = ['_'.join(col).strip() for col in trainer_summary.columns.values]
        trainer_summary = trainer_summary.reset_index()
        trainer_summary['win_count'] = patterns_df.groupby('trainer').size().values
        
        return trainer_summary
    
    def calculate_workout_quality_score(self) -> pd.DataFrame:
        """
        Calculate workout quality scoring for each horse
        
        Returns:
            DataFrame with workout quality scores
        """
        logger.info("Calculating workout quality scores...")
        
        workout_scores = []
        
        for (race, horse), group in self.workouts_df.groupby(['race', 'horse_name']):
            # Sort by date (most recent first)
            works = group.sort_values('work_date', ascending=False)
            
            if len(works) == 0:
                continue
            
            # Get race date from current entries
            race_date = datetime.now()  # Default to today
            
            # Calculate various workout metrics
            total_works = len(works)
            recent_works = works.head(5)  # Last 5 workouts
            
            # 1. Bullet work score
            bullet_count = (recent_works['work_rank'] == 1).sum()
            bullet_score = (bullet_count / len(recent_works)) * 30 if len(recent_works) > 0 else 0
            
            # 2. Work time quality (compare to track standard)
            time_scores = []
            for idx, work in recent_works.iterrows():
                if pd.notna(work['work_time']) and pd.notna(work['work_distance']):
                    # Convert to time per furlong
                    furlongs = work['work_distance'] / 220
                    if furlongs > 0:
                        time_per_furlong = abs(work['work_time']) / furlongs
                        
                        # Score based on time (faster = better)
                        if time_per_furlong < 12:  # Sub-12 second furlongs
                            time_score = 25
                        elif time_per_furlong < 12.5:
                            time_score = 20
                        elif time_per_furlong < 13:
                            time_score = 15
                        else:
                            time_score = 10
                        
                        # Bonus for being ranked high
                        if pd.notna(work['work_rank']) and work['work_rank'] <= 3:
                            time_score += 5
                        
                        time_scores.append(time_score)
            
            avg_time_score = np.mean(time_scores) if time_scores else 15
            
            # 3. Work frequency score
            if len(works) >= 2:
                work_dates = pd.to_datetime(works['work_date'])
                intervals = work_dates.diff().abs().dt.days.dropna()
                avg_interval = intervals.mean()
                
                # Optimal is 5-7 days between works
                if 5 <= avg_interval <= 7:
                    frequency_score = 20
                elif 4 <= avg_interval <= 10:
                    frequency_score = 15
                else:
                    frequency_score = 10
            else:
                frequency_score = 10
            
            # 4. Distance variety score
            unique_distances = works['work_distance'].nunique()
            if unique_distances >= 3:
                variety_score = 10
            elif unique_distances >= 2:
                variety_score = 7
            else:
                variety_score = 5
            
            # 5. Gate work score
            gate_works = works['work_description'].str.contains('g', na=False).sum()
            gate_score = min(10, gate_works * 5)
            
            # 6. Surface consistency
            work_surfaces = works['work_surface_type'].value_counts()
            if len(work_surfaces) > 0:
                dominant_surface_pct = work_surfaces.iloc[0] / len(works)
                surface_score = dominant_surface_pct * 10
            else:
                surface_score = 5
            
            # Calculate composite score
            total_score = (
                bullet_score +
                avg_time_score +
                frequency_score +
                variety_score +
                gate_score +
                surface_score
            )
            
            # Normalize to 0-100 scale
            total_score = min(100, total_score)
            
            # Determine workout quality category
            if total_score >= 80:
                quality_category = 'Excellent'
            elif total_score >= 65:
                quality_category = 'Good'
            elif total_score >= 50:
                quality_category = 'Fair'
            else:
                quality_category = 'Poor'
            
            workout_scores.append({
                'race': race,
                'horse_name': horse,
                'total_workouts': total_works,
                'bullet_work_count': bullet_count,
                'avg_time_score': avg_time_score,
                'frequency_score': frequency_score,
                'variety_score': variety_score,
                'gate_work_score': gate_score,
                'surface_consistency_score': surface_score,
                'workout_quality_score': total_score,
                'workout_quality_category': quality_category
            })
        
        return pd.DataFrame(workout_scores)
    
    def analyze_work_to_race_translation(self) -> pd.DataFrame:
        """
        Analyze how workout patterns translate to race performance
        
        Returns:
            DataFrame with work-to-race correlations
        """
        logger.info("Analyzing work-to-race translation...")
        
        translation_data = []
        
        # For each horse's past race, analyze pre-race workouts
        for (race, horse), past_races in self.past_df.groupby(['race', 'horse_name']):
            for idx, race_row in past_races.iterrows():
                race_date = race_row['pp_race_date']
                
                # Get workouts before this race
                pre_race_works = self.workouts_df[
                    (self.workouts_df['horse_name'] == horse) &
                    (self.workouts_df['work_date'] < race_date) &
                    (self.workouts_df['work_date'] >= race_date - timedelta(days=45))
                ].sort_values('work_date', ascending=False)
                
                if len(pre_race_works) == 0:
                    continue
                
                # Workout pattern metrics
                work_metrics = {
                    'race': race,
                    'horse_name': horse,
                    'race_date': race_date,
                    'finish_position': race_row['pp_finish_pos'],
                    'bris_speed': race_row['pp_bris_speed_rating'],
                    'days_since_last_work': (race_date - pre_race_works['work_date'].iloc[0]).days,
                    'num_works_30d': len(pre_race_works[pre_race_works['work_date'] >= race_date - timedelta(days=30)]),
                    'num_works_14d': len(pre_race_works[pre_race_works['work_date'] >= race_date - timedelta(days=14)]),
                    'had_bullet_14d': (pre_race_works[pre_race_works['work_date'] >= race_date - timedelta(days=14)]['work_rank'] == 1).any(),
                    'avg_work_distance': pre_race_works['work_distance'].mean(),
                    'longest_work_distance': pre_race_works['work_distance'].max(),
                    'race_success': 1 if race_row['pp_finish_pos'] <= 3 else 0
                }
                
                # Calculate workout intensity score
                intensity_score = 0
                for _, work in pre_race_works.iterrows():
                    if pd.notna(work['work_time']) and pd.notna(work['work_distance']) and work['work_distance'] > 0:
                        time_per_furlong = abs(work['work_time']) / (work['work_distance'] / 220)
                        if time_per_furlong < 12:
                            intensity_score += 3
                        elif time_per_furlong < 12.5:
                            intensity_score += 2
                        else:
                            intensity_score += 1
                
                work_metrics['workout_intensity_score'] = intensity_score
                
                translation_data.append(work_metrics)
        
        translation_df = pd.DataFrame(translation_data)
        
        if len(translation_df) == 0:
            return pd.DataFrame()
        
        # Analyze correlations
        correlation_summary = []
        
        # Group by workout patterns and calculate success rates
        pattern_groups = [
            ('days_since_last_work', [(0, 4), (5, 7), (8, 14), (15, 30)]),
            ('num_works_30d', [(0, 2), (3, 4), (5, 6), (7, 10)]),
            ('workout_intensity_score', [(0, 5), (6, 10), (11, 15), (16, 30)])
        ]
        
        for metric, bins in pattern_groups:
            for bin_start, bin_end in bins:
                mask = (translation_df[metric] >= bin_start) & (translation_df[metric] <= bin_end)
                subset = translation_df[mask]
                
                if len(subset) > 0:
                    correlation_summary.append({
                        'metric': metric,
                        'range': f"{bin_start}-{bin_end}",
                        'count': len(subset),
                        'success_rate': subset['race_success'].mean() * 100,
                        'avg_finish_pos': subset['finish_position'].mean(),
                        'avg_speed_rating': subset['bris_speed'].mean()
                    })
        
        return pd.DataFrame(correlation_summary)
    
    def identify_trainer_intent_signals(self) -> pd.DataFrame:
        """
        Identify trainer intent signals from workout patterns
        
        Returns:
            DataFrame with trainer intent analysis
        """
        logger.info("Identifying trainer intent signals...")
        
        intent_signals = []
        
        for (race, horse), works in self.workouts_df.groupby(['race', 'horse_name']):
            if len(works) == 0:
                continue
            
            # Sort by date
            works = works.sort_values('work_date', ascending=False)
            trainer = works['today_s_trainer'].iloc[0] if 'today_s_trainer' in works.columns else 'Unknown'
            
            # Get current race info
            current_info = self.current_df[
                (self.current_df['race'] == race) & 
                (self.current_df['horse_name'] == horse)
            ]
            
            if len(current_info) == 0:
                continue
            
            current_info = current_info.iloc[0]
            
            # Analyze workout patterns for intent
            intent_data = {
                'race': race,
                'horse_name': horse,
                'trainer': trainer,
                'post_position': current_info['post_position'],
                'morning_line_odds': current_info['morn_line_odds_if_available']
            }
            
            # 1. Workout frequency change
            if len(works) >= 4:
                recent_dates = pd.to_datetime(works['work_date'].iloc[:2])
                older_dates = pd.to_datetime(works['work_date'].iloc[2:4])
                
                recent_interval = abs((recent_dates.iloc[0] - recent_dates.iloc[1]).days) if len(recent_dates) > 1 else None
                older_interval = abs((older_dates.iloc[0] - older_dates.iloc[1]).days) if len(older_dates) > 1 else None
                
                if recent_interval and older_interval:
                    if recent_interval < older_interval - 2:
                        intent_data['frequency_signal'] = 'increasing'
                    elif recent_interval > older_interval + 2:
                        intent_data['frequency_signal'] = 'decreasing'
                    else:
                        intent_data['frequency_signal'] = 'stable'
                else:
                    intent_data['frequency_signal'] = 'unknown'
            else:
                intent_data['frequency_signal'] = 'insufficient_data'
            
            # 2. Distance progression
            recent_distances = works['work_distance'].iloc[:3].values
            if len(recent_distances) >= 2:
                if all(recent_distances[i] >= recent_distances[i+1] for i in range(len(recent_distances)-1)):
                    intent_data['distance_pattern'] = 'lengthening'
                elif all(recent_distances[i] <= recent_distances[i+1] for i in range(len(recent_distances)-1)):
                    intent_data['distance_pattern'] = 'shortening'
                else:
                    intent_data['distance_pattern'] = 'mixed'
            else:
                intent_data['distance_pattern'] = 'unknown'
            
            # 3. Bullet work timing
            bullet_works = works[works['work_rank'] == 1]
            if len(bullet_works) > 0:
                days_since_bullet = (datetime.now() - pd.to_datetime(bullet_works['work_date'].iloc[0])).days
                intent_data['days_since_bullet'] = days_since_bullet
                
                if days_since_bullet <= 7:
                    intent_data['bullet_timing'] = 'recent_sharp'
                elif days_since_bullet <= 14:
                    intent_data['bullet_timing'] = 'maintaining_form'
                else:
                    intent_data['bullet_timing'] = 'past_peak'
            else:
                intent_data['days_since_bullet'] = None
                intent_data['bullet_timing'] = 'no_bullets'
            
            # 4. Equipment/surface changes
            surface_changes = works['work_surface_type'].nunique()
            intent_data['surface_experimentation'] = surface_changes > 1
            
            # 5. Gate work presence
            gate_works = works['work_description'].str.contains('g', na=False).sum()
            intent_data['gate_work_count'] = gate_works
            
            # Overall intent classification
            signals = []
            
            if intent_data['frequency_signal'] == 'increasing' and intent_data['bullet_timing'] in ['recent_sharp', 'maintaining_form']:
                signals.append('targeting_race')
            
            if intent_data['distance_pattern'] == 'lengthening':
                signals.append('building_stamina')
            elif intent_data['distance_pattern'] == 'shortening':
                signals.append('sharpening_speed')
            
            if gate_works >= 2:
                signals.append('improving_break')
            
            if surface_changes > 1:
                signals.append('surface_uncertain')
            
            intent_data['trainer_intent_signals'] = ', '.join(signals) if signals else 'maintenance'
            
            # Confidence score based on data quality
            confidence = 50
            if len(works) >= 4:
                confidence += 20
            if intent_data['frequency_signal'] != 'unknown':
                confidence += 15
            if intent_data['bullet_timing'] != 'no_bullets':
                confidence += 15
            
            intent_data['intent_confidence'] = min(100, confidence)
            
            intent_signals.append(intent_data)
        
        return pd.DataFrame(intent_signals)
    
    def calculate_all_workout_metrics(self) -> pd.DataFrame:
        """
        Calculate all workout metrics and combine into single DataFrame
        
        Returns:
            DataFrame with comprehensive workout analysis
        """
        logger.info("Calculating all workout metrics...")
        
        # Get base horse info
        workout_analysis = self.current_df[['race', 'horse_name', 'post_position', 
                                           'today_s_trainer', 'morn_line_odds_if_available']].copy()
        
        # Calculate individual components
        quality_scores = self.calculate_workout_quality_score()
        intent_signals = self.identify_trainer_intent_signals()
        
        # Merge quality scores
        if not quality_scores.empty:
            workout_analysis = workout_analysis.merge(
                quality_scores,
                on=['race', 'horse_name'],
                how='left'
            )
        
        # Merge intent signals
        if not intent_signals.empty:
            intent_cols = ['race', 'horse_name', 'frequency_signal', 'distance_pattern',
                          'bullet_timing', 'days_since_bullet', 'gate_work_count',
                          'trainer_intent_signals', 'intent_confidence']
            workout_analysis = workout_analysis.merge(
                intent_signals[intent_cols],
                on=['race', 'horse_name'],
                how='left'
            )
        
        # Add workout summary statistics
        workout_stats = []
        for (race, horse), works in self.workouts_df.groupby(['race', 'horse_name']):
            stats = {
                'race': race,
                'horse_name': horse,
                'total_workouts': len(works),
                'works_last_30d': len(works[works['work_date'] >= datetime.now() - timedelta(days=30)]),
                'works_last_14d': len(works[works['work_date'] >= datetime.now() - timedelta(days=14)]),
                'avg_work_distance': works['work_distance'].mean(),
                'max_work_distance': works['work_distance'].max(),
                'pct_fast_works': 0
            }
            
            # Calculate percentage of fast works
            fast_works = 0
            for _, work in works.iterrows():
                if pd.notna(work['work_time']) and pd.notna(work['work_distance']) and work['work_distance'] > 0:
                    time_per_furlong = abs(work['work_time']) / (work['work_distance'] / 220)
                    if time_per_furlong < 12.5:
                        fast_works += 1
            
            stats['pct_fast_works'] = (fast_works / len(works) * 100) if len(works) > 0 else 0
            workout_stats.append(stats)
        
        if workout_stats:
            stats_df = pd.DataFrame(workout_stats)
            workout_analysis = workout_analysis.merge(
                stats_df,
                on=['race', 'horse_name'],
                how='left'
            )
        
        # Add composite workout readiness score
        score_components = []
        if 'workout_quality_score' in workout_analysis.columns:
            score_components.append(workout_analysis['workout_quality_score'].fillna(50))
        
        if 'intent_confidence' in workout_analysis.columns:
            score_components.append(workout_analysis['intent_confidence'].fillna(50))
        
        if 'pct_fast_works' in workout_analysis.columns:
            # Scale percentage to 0-100 score
            score_components.append(workout_analysis['pct_fast_works'].fillna(0) * 2)
        
        if score_components:
            workout_analysis['workout_readiness_score'] = np.mean(score_components, axis=0)
        else:
            workout_analysis['workout_readiness_score'] = 50
        
        # Add readiness category
        workout_analysis['workout_readiness_category'] = pd.cut(
            workout_analysis['workout_readiness_score'],
            bins=[0, 40, 60, 80, 100],
            labels=['Poor', 'Fair', 'Good', 'Excellent']
        )
        
        # Add ranking within race
        workout_analysis['workout_rank'] = workout_analysis.groupby('race')['workout_readiness_score'].rank(
            ascending=False,
            method='min'
        )
        
        logger.info(f"Analyzed workouts for {len(workout_analysis)} horses")
        
        return workout_analysis
    
    def generate_trainer_pattern_report(self) -> pd.DataFrame:
        """
        Generate detailed trainer pattern report
        
        Returns:
            DataFrame with trainer-specific winning patterns
        """
        logger.info("Generating trainer pattern report...")
        
        trainer_patterns = self.analyze_trainer_workout_patterns()
        
        if trainer_patterns.empty:
            logger.warning("No trainer patterns found")
            return pd.DataFrame()
        
        # Add success metrics
        trainer_success = []
        
        for trainer in trainer_patterns['trainer'].unique():
            # Get all horses for this trainer
            trainer_horses = self.current_df[self.current_df['today_s_trainer'] == trainer]['horse_name'].unique()
            
            # Calculate historical success rate
            trainer_past = self.past_df[self.past_df['horse_name'].isin(trainer_horses)]
            
            if len(trainer_past) > 0:
                win_rate = (trainer_past['pp_finish_pos'] == 1).mean() * 100
                itm_rate = (trainer_past['pp_finish_pos'] <= 3).mean() * 100
                avg_odds = trainer_past['pp_odds'].mean()
                
                trainer_success.append({
                    'trainer': trainer,
                    'historical_win_rate': win_rate,
                    'historical_itm_rate': itm_rate,
                    'avg_winning_odds': avg_odds
                })
        
        if trainer_success:
            success_df = pd.DataFrame(trainer_success)
            trainer_patterns = trainer_patterns.merge(success_df, on='trainer', how='left')
        
        return trainer_patterns


def main():
    """Main function to run workout analysis"""
    
    # Set up paths
    base_path = Path('data/processed')
    workouts_path = base_path / 'workouts_long_format.parquet'
    current_race_path = base_path / 'current_race_info.parquet'
    past_starts_path = base_path / 'past_starts_long_format.parquet'
    
    # Check if files exist
    for path in [workouts_path, current_race_path, past_starts_path]:
        if not path.exists():
            logger.error(f"File not found: {path}")
            return
    
    # Initialize analyzer
    analyzer = SophisticatedWorkoutAnalysis(
        str(workouts_path),
        str(current_race_path),
        str(past_starts_path)
    )
    
    # Calculate all metrics
    workout_analysis = analyzer.calculate_all_workout_metrics()
    
    # Generate trainer pattern report
    trainer_report = analyzer.generate_trainer_pattern_report()
    
    # Analyze work-to-race translation
    translation_analysis = analyzer.analyze_work_to_race_translation()
    
    # Save results
    output_path = base_path / 'sophisticated_workout_analysis.parquet'
    workout_analysis.to_parquet(output_path, index=False)
    logger.info(f"Saved workout analysis to {output_path}")
    
    if not trainer_report.empty:
        trainer_path = base_path / 'trainer_workout_patterns.csv'
        trainer_report.to_csv(trainer_path, index=False)
        logger.info(f"Saved trainer patterns to {trainer_path}")
    
    if not translation_analysis.empty:
        translation_path = base_path / 'workout_race_translation.csv'
        translation_analysis.to_csv(translation_path, index=False)
        logger.info(f"Saved translation analysis to {translation_path}")
    
    # Display summary
    print("\n" + "="*60)
    print("WORKOUT ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total horses analyzed: {len(workout_analysis)}")
    
    print(f"\nWorkout Readiness Categories:")
    if 'workout_readiness_category' in workout_analysis.columns:
        print(workout_analysis['workout_readiness_category'].value_counts().sort_index())
    
    print(f"\nTop 10 Horses by Workout Readiness:")
    if 'workout_readiness_score' in workout_analysis.columns:
        top_horses = workout_analysis.nlargest(10, 'workout_readiness_score')[
            ['race', 'horse_name', 'post_position', 'workout_readiness_score', 
             'workout_quality_score', 'trainer_intent_signals']
        ]
        print(top_horses.to_string(index=False))
    
    print(f"\nTrainer Intent Signals Distribution:")
    if 'trainer_intent_signals' in workout_analysis.columns:
        intent_counts = workout_analysis['trainer_intent_signals'].value_counts()
        print(intent_counts.head(10))
    
    # Print trainer patterns if available
    if not trainer_report.empty:
        print(f"\n" + "="*60)
        print("TOP TRAINER PATTERNS FOR WINNERS")
        print("="*60)
        print(trainer_report.head(10).to_string(index=False))
    
    # Print translation insights if available
    if not translation_analysis.empty:
        print(f"\n" + "="*60)
        print("WORKOUT-TO-RACE TRANSLATION INSIGHTS")
        print("="*60)
        # Sort by success rate
        best_patterns = translation_analysis.nlargest(10, 'success_rate')
        print(best_patterns.to_string(index=False))
    
    # Save detailed summary report
    summary_path = base_path / 'workout_analysis_summary.csv'
    summary_cols = [
        'race', 'horse_name', 'post_position', 'today_s_trainer',
        'workout_readiness_score', 'workout_readiness_category',
        'workout_quality_score', 'trainer_intent_signals',
        'bullet_timing', 'works_last_14d', 'pct_fast_works'
    ]
    
    available_cols = [col for col in summary_cols if col in workout_analysis.columns]
    summary_df = workout_analysis[available_cols].sort_values(['race', 'workout_rank'])
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved workout summary to {summary_path}")


if __name__ == '__main__':
    main()