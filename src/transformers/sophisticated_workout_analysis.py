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
            
            # 1. Bullet work score (0-50)
            bullet_count = (recent_works['work_rank'] == 1).sum()
            bullet_score = (bullet_count / 5) * 50 if len(recent_works) > 0 else 0
            
            # 2. Work time quality (0-25)
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
            
            # 3. Work frequency score (0-10)
            if len(works) >= 2:
                work_dates = pd.to_datetime(works['work_date'])
                intervals = work_dates.diff().abs().dt.days.dropna()
                avg_interval = intervals.mean()
                
                # Optimal is 5-7 days between works
                if 5 <= avg_interval <= 7:
                    frequency_score = 10
                elif 4 <= avg_interval <= 10:
                    frequency_score = 7
                else:
                    frequency_score = 5
            else:
                frequency_score = 5
            
            # 4. Distance variety score (0-5)
            unique_distances = works['work_distance'].nunique()
            if unique_distances >= 3:
                variety_score = 5
            elif unique_distances >= 2:
                variety_score = 3
            else:
                variety_score = 1
            
            # 5. Gate work score (0-5)
            gate_works = works['work_description'].str.contains('g', na=False).sum()
            gate_score = min(5, gate_works)
            
            # 6. Surface consistency (0-5)
            work_surfaces = works['work_surface_type'].value_counts()
            if len(work_surfaces) > 0:
                dominant_surface_pct = work_surfaces.iloc[0] / len(works)
                surface_score = dominant_surface_pct * 5
            else:
                surface_score = 2.5
            
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
        
        # Calculate quality scores
        quality_scores = self.calculate_workout_quality_score()
        
        # Merge quality scores
        if not quality_scores.empty:
            workout_analysis = workout_analysis.merge(
                quality_scores,
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
        
        if 'pct_fast_works' in workout_analysis.columns:
            # Use raw percentage
            score_components.append(workout_analysis['pct_fast_works'].fillna(0))
        
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
    
    # Save results
    output_path = base_path / 'sophisticated_workout_analysis.parquet'
    workout_analysis.to_parquet(output_path, index=False)
    logger.info(f"Saved workout analysis to {output_path}")
    
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
             'workout_quality_score']
        ]
        print(top_horses.to_string(index=False))
    
    # Save detailed summary report
    summary_path = base_path / 'workout_analysis_summary.csv'
    summary_cols = [
        'race', 'horse_name', 'post_position', 'today_s_trainer',
        'workout_readiness_score', 'workout_readiness_category',
        'workout_quality_score', 'bullet_work_count', 'works_last_14d', 
        'pct_fast_works', 'workout_rank'
    ]
    
    available_cols = [col for col in summary_cols if col in workout_analysis.columns]
    
    # Check if workout_rank is available for sorting
    if 'workout_rank' in workout_analysis.columns:
        summary_df = workout_analysis[available_cols].sort_values(['race', 'workout_rank'])
    else:
        # Fallback to sorting by workout_readiness_score if workout_rank doesn't exist
        summary_df = workout_analysis[available_cols].sort_values(['race', 'workout_readiness_score'], ascending=[True, False])
    
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved workout summary to {summary_path}")


if __name__ == '__main__':
    main()