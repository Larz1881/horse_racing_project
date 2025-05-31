# src/transformers/form_cycle_detector.py
"""
Form Cycle & Improvement Detection Module

Identifies horses on upward/downward trajectories through:
1. Beaten Lengths Trajectory Analysis
2. Position Call Analytics
3. Form Cycle Pattern Recognition
4. Fractional Time Evolution
5. Field Size Adjusted Performance
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FormCycleDetector:
    """Detect and analyze form cycles and improvement patterns"""
    
    # Beaten lengths to seconds conversion factors by distance
    LENGTHS_PER_SECOND = {
        'sprint': 5.5,    # 6f and under
        'route': 5.0,     # Over 6f
        'default': 5.25   # Average
    }
    
    # Form cycle states
    CYCLE_STATES = {
        'IMPROVING': 'Upward trajectory',
        'PEAKING': 'At or near peak form',
        'BOUNCING': 'Regression after peak effort',
        'RECOVERING': 'Rebounding from poor effort',
        'DECLINING': 'Downward trajectory',
        'FRESHENING': 'Returning from layoff',
        'STABLE': 'Consistent form',
        'ERRATIC': 'Inconsistent pattern'
    }
    
    def __init__(self, current_race_path: str, past_starts_path: str):
        """
        Initialize with paths to parquet files
        
        Args:
            current_race_path: Path to current_race_info.parquet
            past_starts_path: Path to past_starts_long_format.parquet
        """
        self.current_df = pd.read_parquet(current_race_path)
        self.past_df = pd.read_parquet(past_starts_path)
        
        # Ensure proper data types
        self.past_df['pp_race_date'] = pd.to_datetime(self.past_df['pp_race_date'])
        
        # Pre-calculate some useful metrics
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare data with additional calculated fields"""
        # Add distance category
        self.past_df['distance_category'] = self.past_df['pp_distance'].apply(
            lambda x: 'sprint' if x <= 1320 else 'route'  # 1320 yards = 6 furlongs
        )
        
        # Sort by horse and date
        self.past_df = self.past_df.sort_values(
            ['race', 'horse_name', 'pp_race_date'],
            ascending=[True, True, False]
        )
    
    def analyze_beaten_lengths_trajectory(self) -> pd.DataFrame:
        """
        Analyze beaten lengths trajectory and convert to time
        
        Returns:
            DataFrame with beaten lengths analysis
        """
        logger.info("Analyzing beaten lengths trajectory...")
        
        trajectory_data = []
        
        for (race, horse), group in self.past_df.groupby(['race', 'horse_name']):
            if len(group) < 3:
                continue
            
            # Get recent performances
            recent = group.head(10)
            
            analysis = {
                'race': race,
                'horse_name': horse
            }
            
            # Extract beaten lengths data
            beaten_lengths = recent['pp_finish_lengths_behind'].values
            finish_positions = recent['pp_finish_pos'].values
            distances = recent['pp_distance'].values
            field_sizes = recent['pp_num_entrants'].values
            
            # Convert beaten lengths to time
            beaten_times = []
            for i, (bl, dist) in enumerate(zip(beaten_lengths, distances)):
                if pd.notna(bl) and pd.notna(dist):
                    # Determine conversion factor
                    if dist <= 1320:  # Sprint
                        factor = self.LENGTHS_PER_SECOND['sprint']
                    else:  # Route
                        factor = self.LENGTHS_PER_SECOND['route']
                    
                    beaten_time = bl / factor
                    beaten_times.append(beaten_time)
                else:
                    beaten_times.append(None)
            
            beaten_times = np.array(beaten_times)
            
            # Trajectory analysis
            valid_times = [t for t in beaten_times if t is not None]
            if len(valid_times) >= 3:
                # Calculate improvement (negative = getting closer)
                time_diffs = np.diff(valid_times)
                analysis['avg_improvement_per_race'] = -np.mean(time_diffs)
                
                # Trend analysis
                x = np.arange(len(valid_times))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, valid_times)
                
                analysis['beaten_time_trend'] = -slope  # Negative slope = improvement
                analysis['trend_r_squared'] = r_value ** 2
                analysis['trend_significant'] = p_value < 0.05
                
                # Recent form vs older
                if len(valid_times) >= 6:
                    recent_avg = np.mean(valid_times[:3])
                    older_avg = np.mean(valid_times[3:6])
                    analysis['recent_vs_older_improvement'] = older_avg - recent_avg
                else:
                    analysis['recent_vs_older_improvement'] = 0
            else:
                analysis['avg_improvement_per_race'] = 0
                analysis['beaten_time_trend'] = 0
                analysis['trend_r_squared'] = 0
                analysis['trend_significant'] = False
            
            # Identify unlucky performances
            unlucky_races = []
            trouble_keywords = ['blocked', 'checked', 'bumped', 'steadied', 'interfered',
                              'wide', 'boxed', 'trapped', 'no room', 'shut off']
            
            for idx, row in recent.iterrows():
                trip_comment = str(row.get('pp_trip_comment', '')).lower()
                
                # Check for trouble
                had_trouble = any(keyword in trip_comment for keyword in trouble_keywords)
                
                # Check for significant position loss
                start_pos = row.get('pp_start_call_pos', 99)
                finish_pos = row.get('pp_finish_pos', 99)
                
                if pd.notna(start_pos) and pd.notna(finish_pos):
                    # Convert string positions to numeric
                    try:
                        start_pos = float(str(start_pos))
                        finish_pos = float(str(finish_pos))
                        position_loss = finish_pos - start_pos > 3
                    except:
                        position_loss = False
                else:
                    position_loss = False
                
                if had_trouble or position_loss:
                    unlucky_races.append({
                        'date': row['pp_race_date'],
                        'trouble': had_trouble,
                        'position_loss': position_loss
                    })
            
            analysis['unlucky_race_count'] = len(unlucky_races)
            analysis['pct_races_with_trouble'] = (len(unlucky_races) / len(recent)) * 100
            
            # Ground gained/lost metrics
            position_changes = []
            for idx, row in recent.iterrows():
                # Use the pre-calculated position gain columns
                total_gain = row.get('pp_pos_gain_start_to_finish', 0)
                if pd.notna(total_gain):
                    position_changes.append(total_gain)
            
            if position_changes:
                analysis['avg_position_gain'] = np.mean(position_changes)
                analysis['consistent_gainer'] = sum(pc > 0 for pc in position_changes) > len(position_changes) / 2
            else:
                analysis['avg_position_gain'] = 0
                analysis['consistent_gainer'] = False
            
            # Best and worst efforts
            if valid_times:
                analysis['best_beaten_time'] = min(valid_times)
                analysis['worst_beaten_time'] = max(valid_times)
                analysis['beaten_time_volatility'] = np.std(valid_times)
            
            trajectory_data.append(analysis)
        
        return pd.DataFrame(trajectory_data)
    
    def analyze_position_calls(self) -> pd.DataFrame:
        """
        Analyze progression through race via position calls
        
        Returns:
            DataFrame with position call analytics
        """
        logger.info("Analyzing position call patterns...")
        
        position_data = []
        
        for (race, horse), group in self.past_df.groupby(['race', 'horse_name']):
            if len(group) < 3:
                continue
            
            recent = group.head(10)
            
            analysis = {
                'race': race,
                'horse_name': horse
            }
            
            # Collect position data
            early_positions = []
            mid_positions = []
            late_positions = []
            finish_positions = []
            
            for idx, row in recent.iterrows():
                # Get positions (handle various formats)
                early = self._parse_position(row.get('pp_first_call_pos'))
                mid = self._parse_position(row.get('pp_second_call_pos'))
                late = self._parse_position(row.get('pp_stretch_pos'))
                finish = self._parse_position(row.get('pp_finish_pos'))
                
                if early is not None:
                    early_positions.append(early)
                if mid is not None:
                    mid_positions.append(mid)
                if late is not None:
                    late_positions.append(late)
                if finish is not None:
                    finish_positions.append(finish)
            
            # Early position vs finish correlation
            if len(early_positions) >= 3 and len(finish_positions) >= 3:
                min_len = min(len(early_positions), len(finish_positions))
                correlation, p_value = stats.pearsonr(
                    early_positions[:min_len],
                    finish_positions[:min_len]
                )
                analysis['early_finish_correlation'] = correlation
                analysis['correlation_significant'] = p_value < 0.05
                
                # Average improvement from early to finish
                improvements = [e - f for e, f in zip(early_positions[:min_len], 
                                                     finish_positions[:min_len])]
                analysis['avg_early_to_finish_gain'] = np.mean(improvements)
            else:
                analysis['early_finish_correlation'] = 0
                analysis['avg_early_to_finish_gain'] = 0
            
            # Optimal position patterns by distance
            sprint_races = recent[recent['distance_category'] == 'sprint']
            route_races = recent[recent['distance_category'] == 'route']
            
            # Sprint pattern
            if len(sprint_races) >= 2:
                sprint_early = [self._parse_position(p) for p in sprint_races['pp_first_call_pos']]
                sprint_finish = [self._parse_position(p) for p in sprint_races['pp_finish_pos']]
                
                valid_sprint = [(e, f) for e, f in zip(sprint_early, sprint_finish) 
                               if e is not None and f is not None]
                
                if valid_sprint:
                    best_sprint_finish = min(f for _, f in valid_sprint)
                    best_sprint_early = [e for e, f in valid_sprint if f == best_sprint_finish][0]
                    analysis['optimal_sprint_early_position'] = best_sprint_early
                else:
                    analysis['optimal_sprint_early_position'] = None
            
            # Route pattern
            if len(route_races) >= 2:
                route_early = [self._parse_position(p) for p in route_races['pp_first_call_pos']]
                route_finish = [self._parse_position(p) for p in route_races['pp_finish_pos']]
                
                valid_route = [(e, f) for e, f in zip(route_early, route_finish) 
                              if e is not None and f is not None]
                
                if valid_route:
                    best_route_finish = min(f for _, f in valid_route)
                    best_route_early = [e for e, f in valid_route if f == best_route_finish][0]
                    analysis['optimal_route_early_position'] = best_route_early
                else:
                    analysis['optimal_route_early_position'] = None
            
            # Late position gain patterns
            late_gains = []
            for idx, row in recent.iterrows():
                mid_pos = self._parse_position(row.get('pp_second_call_pos'))
                finish_pos = self._parse_position(row.get('pp_finish_pos'))
                
                if mid_pos is not None and finish_pos is not None:
                    gain = mid_pos - finish_pos
                    late_gains.append(gain)
            
            if late_gains:
                analysis['avg_late_gain'] = np.mean(late_gains)
                analysis['strong_late_kick'] = analysis['avg_late_gain'] > 2
                analysis['late_gain_consistency'] = (sum(g > 0 for g in late_gains) / 
                                                   len(late_gains)) * 100
            else:
                analysis['avg_late_gain'] = 0
                analysis['strong_late_kick'] = False
                analysis['late_gain_consistency'] = 0
            
            # Traffic trouble indicators
            wide_trips = 0
            position_losses = 0
            
            for idx, row in recent.iterrows():
                # Check for wide trips (position + lengths behind > threshold)
                if pd.notna(row.get('pp_first_call_pos')) and pd.notna(row.get('pp_first_call_lengths_behind')):
                    try:
                        pos = float(str(row['pp_first_call_pos']))
                        lengths = float(row['pp_first_call_lengths_behind'])
                        if pos > 4 and lengths < 3:  # Wide but close = traffic
                            wide_trips += 1
                    except:
                        pass
                
                # Check for position loss at any call
                positions = [
                    self._parse_position(row.get('pp_start_call_pos')),
                    self._parse_position(row.get('pp_first_call_pos')),
                    self._parse_position(row.get('pp_second_call_pos')),
                    self._parse_position(row.get('pp_stretch_pos'))
                ]
                
                valid_positions = [p for p in positions if p is not None]
                if len(valid_positions) >= 2:
                    # Check for any backward movement
                    for i in range(1, len(valid_positions)):
                        if valid_positions[i] > valid_positions[i-1] + 1:
                            position_losses += 1
                            break
            
            analysis['wide_trip_count'] = wide_trips
            analysis['position_loss_count'] = position_losses
            analysis['traffic_trouble_rate'] = ((wide_trips + position_losses) / 
                                              (len(recent) * 2)) * 100
            
            position_data.append(analysis)
        
        return pd.DataFrame(position_data)
    
    def _parse_position(self, position) -> Optional[float]:
        """Parse position which might be string or numeric"""
        if pd.isna(position):
            return None
        
        # Convert to string and clean
        pos_str = str(position).strip()
        
        # Remove any non-numeric characters except decimal
        cleaned = ''.join(c for c in pos_str if c.isdigit() or c == '.')
        
        try:
            return float(cleaned) if cleaned else None
        except:
            return None
    
    def detect_form_cycle_patterns(self) -> pd.DataFrame:
        """
        Detect form cycle patterns (bounce, recovery, freshening, etc.)
        
        Returns:
            DataFrame with form cycle pattern analysis
        """
        logger.info("Detecting form cycle patterns...")
        
        pattern_data = []
        
        for (race, horse), group in self.past_df.groupby(['race', 'horse_name']):
            if len(group) < 3:
                continue
            
            recent = group.head(15)  # Need more races for pattern detection
            
            analysis = {
                'race': race,
                'horse_name': horse
            }
            
            # Get performance metrics
            speed_ratings = recent['pp_bris_speed_rating'].values
            days_between = recent['pp_days_since_prev'].values
            finish_positions = recent['pp_finish_pos'].values
            
            # Filter valid data
            valid_speeds = [s for s in speed_ratings if pd.notna(s)]
            
            if len(valid_speeds) >= 3:
                # Detect bounce pattern (regression after career best)
                analysis.update(self._detect_bounce_pattern(valid_speeds))
                
                # Detect recovery pattern
                analysis.update(self._detect_recovery_pattern(valid_speeds, days_between))
                
                # Detect freshening pattern
                analysis.update(self._detect_freshening_pattern(
                    valid_speeds, days_between, recent
                ))
                
                # Peak performance prediction
                analysis.update(self._predict_peak_performance(
                    valid_speeds, days_between
                ))
                
                # Overall form cycle state
                analysis['form_cycle_state'] = self._determine_cycle_state(analysis)
            else:
                analysis['form_cycle_state'] = 'INSUFFICIENT_DATA'
            
            # Add current days since last race
            current_horse = self.current_df[
                (self.current_df['race'] == race) & 
                (self.current_df['horse_name'] == horse)
            ]
            
            if len(current_horse) > 0:
                analysis['current_days_off'] = current_horse.iloc[0].get('of_days_since_last_race', 0)
            
            pattern_data.append(analysis)
        
        return pd.DataFrame(pattern_data)
    
    def _detect_bounce_pattern(self, speed_ratings: List[float]) -> Dict:
        """Detect bounce pattern after peak efforts"""
        result = {}
        
        if len(speed_ratings) < 4:
            result['bounce_risk'] = 'Unknown'
            return result
        
        # Find peaks in performance
        peaks, properties = find_peaks(speed_ratings, prominence=3)
        
        if len(peaks) > 0:
            # Check most recent peak
            last_peak_idx = peaks[-1]
            
            # Was last race a peak?
            if last_peak_idx == 0:  # Most recent race was peak
                peak_value = speed_ratings[0]
                career_best = max(speed_ratings)
                
                if peak_value >= career_best - 2:
                    result['bounce_risk'] = 'High'
                    result['last_race_career_best'] = True
                    result['peak_improvement'] = peak_value - np.mean(speed_ratings[1:4])
                else:
                    result['bounce_risk'] = 'Moderate'
                    result['last_race_career_best'] = False
            else:
                # Check if bounced after previous peak
                if last_peak_idx < len(speed_ratings) - 1:
                    post_peak = speed_ratings[last_peak_idx + 1]
                    peak_value = speed_ratings[last_peak_idx]
                    bounce_magnitude = peak_value - post_peak
                    
                    if bounce_magnitude > 5:
                        result['previous_bounce_magnitude'] = bounce_magnitude
                        result['bounce_risk'] = 'Historical_bouncer'
                    else:
                        result['bounce_risk'] = 'Low'
                else:
                    result['bounce_risk'] = 'Low'
        else:
            result['bounce_risk'] = 'Low'
        
        # Calculate bounce tendency
        if len(speed_ratings) >= 6:
            improvements = np.diff(speed_ratings)
            big_improvements = [i for i in improvements if i > 5]
            
            if big_improvements:
                # Check what happened after big improvements
                bounce_count = 0
                for i in range(len(improvements) - 1):
                    if improvements[i] > 5 and improvements[i + 1] < -3:
                        bounce_count += 1
                
                result['bounce_tendency'] = (bounce_count / len(big_improvements)) * 100
            else:
                result['bounce_tendency'] = 0
        
        return result
    
    def _detect_recovery_pattern(self, speed_ratings: List[float], 
                                days_between: np.ndarray) -> Dict:
        """Detect recovery from poor performances"""
        result = {}
        
        if len(speed_ratings) < 4:
            return result
        
        # Look for poor efforts followed by improvement
        recovery_patterns = []
        
        for i in range(len(speed_ratings) - 2):
            current = speed_ratings[i]
            previous = speed_ratings[i + 1]
            
            # Poor effort detection (5+ points below average)
            if i + 3 < len(speed_ratings):
                typical_level = np.mean(speed_ratings[i + 2:i + 5])
            else:
                typical_level = np.mean(speed_ratings)
            
            if previous < typical_level - 5:
                # Check if recovered
                recovery = current - previous
                if recovery > 3:
                    days_to_recover = days_between[i] if i < len(days_between) else None
                    recovery_patterns.append({
                        'recovery_points': recovery,
                        'days_to_recover': days_to_recover
                    })
        
        if recovery_patterns:
            avg_recovery = np.mean([r['recovery_points'] for r in recovery_patterns])
            avg_days = np.mean([r['days_to_recover'] for r in recovery_patterns 
                               if r['days_to_recover'] is not None])
            
            result['recovery_ability'] = 'Strong' if avg_recovery > 5 else 'Moderate'
            result['avg_recovery_points'] = avg_recovery
            result['avg_recovery_days'] = avg_days if not np.isnan(avg_days) else None
            
            # Check if currently recovering
            if speed_ratings[1] < typical_level - 5:
                result['currently_recovering'] = True
                result['expected_recovery'] = avg_recovery
            else:
                result['currently_recovering'] = False
        else:
            result['recovery_ability'] = 'Unproven'
            result['currently_recovering'] = False
        
        return result
    
    def _detect_freshening_pattern(self, speed_ratings: List[float],
                                  days_between: np.ndarray,
                                  recent_df: pd.DataFrame) -> Dict:
        """Detect freshening patterns after layoffs"""
        result = {}
        
        # Find layoffs (30+ days)
        layoff_performances = []
        
        for i in range(len(days_between)):
            if pd.notna(days_between[i]) and days_between[i] >= 30:
                if i < len(speed_ratings):
                    layoff_performances.append({
                        'days_off': days_between[i],
                        'return_rating': speed_ratings[i],
                        'previous_rating': speed_ratings[i + 1] if i + 1 < len(speed_ratings) else None
                    })
        
        if layoff_performances:
            # Analyze layoff success
            successful_returns = 0
            total_returns = 0
            
            for perf in layoff_performances:
                if perf['previous_rating'] is not None:
                    total_returns += 1
                    if perf['return_rating'] >= perf['previous_rating'] - 2:
                        successful_returns += 1
            
            if total_returns > 0:
                result['freshening_success_rate'] = (successful_returns / total_returns) * 100
                result['handles_layoffs_well'] = result['freshening_success_rate'] > 50
            else:
                result['freshening_success_rate'] = 50
                result['handles_layoffs_well'] = False
            
            # Optimal layoff duration
            if len(layoff_performances) >= 2:
                # Find best performing layoff duration
                best_perf = max(layoff_performances, 
                              key=lambda x: x['return_rating'] if x['return_rating'] is not None else 0)
                result['optimal_layoff_days'] = best_perf['days_off']
        else:
            result['freshening_success_rate'] = 50
            result['handles_layoffs_well'] = False
        
        # Check if currently freshening
        if len(days_between) > 0 and pd.notna(days_between[0]) and days_between[0] >= 30:
            result['currently_freshening'] = True
            result['freshening_days'] = days_between[0]
        else:
            result['currently_freshening'] = False
        
        return result
    
    def _predict_peak_performance(self, speed_ratings: List[float],
                                 days_between: np.ndarray) -> Dict:
        """Predict when horse might peak"""
        result = {}
        
        if len(speed_ratings) < 5:
            result['races_to_peak'] = 'Unknown'
            return result
        
        # Analyze improvement trajectory
        recent_ratings = speed_ratings[:5]
        older_ratings = speed_ratings[5:10] if len(speed_ratings) >= 10 else speed_ratings[5:]
        
        if older_ratings:
            recent_avg = np.mean(recent_ratings)
            older_avg = np.mean(older_ratings)
            improvement_rate = recent_avg - older_avg
            
            # Check if still improving
            recent_trend = np.polyfit(range(len(recent_ratings)), recent_ratings, 1)[0]
            
            if recent_trend > 0.5:  # Still improving
                # Estimate races to peak based on historical patterns
                current_level = recent_ratings[0]
                historical_peak = max(speed_ratings)
                
                if current_level < historical_peak - 3:
                    result['races_to_peak'] = '2-3'
                    result['improvement_trajectory'] = 'Ascending'
                else:
                    result['races_to_peak'] = '0-1'
                    result['improvement_trajectory'] = 'Near_peak'
            elif recent_trend < -0.5:  # Declining
                result['races_to_peak'] = 'Past_peak'
                result['improvement_trajectory'] = 'Declining'
            else:
                result['races_to_peak'] = 'At_peak'
                result['improvement_trajectory'] = 'Stable'
            
            result['recent_trend_slope'] = recent_trend
        else:
            result['races_to_peak'] = 'Developing'
            result['improvement_trajectory'] = 'Unknown'
        
        return result
    
    def _determine_cycle_state(self, analysis: Dict) -> str:
        """Determine overall form cycle state"""
        # Priority order for state determination
        
        if analysis.get('currently_freshening'):
            return 'FRESHENING'
        
        if analysis.get('bounce_risk') == 'High':
            return 'BOUNCING'
        
        if analysis.get('currently_recovering'):
            return 'RECOVERING'
        
        trajectory = analysis.get('improvement_trajectory', '')
        if trajectory == 'Ascending':
            return 'IMPROVING'
        elif trajectory == 'Near_peak' or trajectory == 'At_peak':
            return 'PEAKING'
        elif trajectory == 'Declining':
            return 'DECLINING'
        
        # Check for erratic pattern
        if analysis.get('bounce_tendency', 0) > 50:
            return 'ERRATIC'
        
        return 'STABLE'
    
    def analyze_fractional_evolution(self) -> pd.DataFrame:
        """
        Track improvement in fractional times
        
        Returns:
            DataFrame with fractional time analysis
        """
        logger.info("Analyzing fractional time evolution...")
        
        fractional_data = []
        
        for (race, horse), group in self.past_df.groupby(['race', 'horse_name']):
            if len(group) < 3:
                continue
            
            recent = group.head(10)
            
            analysis = {
                'race': race,
                'horse_name': horse
            }
            
            # Collect fractional times
            fractions = {
                '2f': recent['pp_frac_2f'].values,
                '4f': recent['pp_frac_4f'].values,
                '6f': recent['pp_frac_6f'].values,
                'final': recent['pp_final_time'].values
            }
            
            # Also get splits
            splits = {
                '0-2f': recent['pp_split_0_2f_secs'].values,
                '2f-4f': recent['pp_split_2f_4f_secs'].values,
                '4f-6f': recent['pp_split_4f_6f_secs'].values,
                '6f-8f': recent['pp_split_6f_8f_secs'].values
            }
            
            # Analyze improvements in each fraction
            fraction_improvements = {}
            
            for frac_name, times in fractions.items():
                valid_times = [t for t in times if pd.notna(t) and t > 0]
                
                if len(valid_times) >= 3:
                    # Recent vs older comparison
                    if len(valid_times) >= 6:
                        recent_avg = np.mean(valid_times[:3])
                        older_avg = np.mean(valid_times[3:6])
                        improvement = older_avg - recent_avg  # Positive = faster
                        fraction_improvements[f'{frac_name}_improvement'] = improvement
                    
                    # Trend analysis
                    x = np.arange(len(valid_times))
                    slope, _, _, _, _ = stats.linregress(x, valid_times)
                    fraction_improvements[f'{frac_name}_trend'] = -slope  # Negative slope = improvement
            
            analysis.update(fraction_improvements)
            
            # Split analysis for pace sustainability
            sustainability_scores = []
            
            for idx, row in recent.iterrows():
                early_split = row.get('pp_split_0_2f_secs')
                late_split = row.get('pp_split_6f_8f_secs')
                
                if pd.notna(early_split) and pd.notna(late_split) and early_split > 0:
                    # Ratio of late to early (lower = better sustainability)
                    sustainability = late_split / early_split
                    sustainability_scores.append(sustainability)
            
            if sustainability_scores:
                analysis['avg_pace_sustainability'] = np.mean(sustainability_scores)
                analysis['improving_sustainability'] = False
                
                # Check if sustainability improving
                if len(sustainability_scores) >= 4:
                    recent_sust = np.mean(sustainability_scores[:2])
                    older_sust = np.mean(sustainability_scores[2:4])
                    analysis['improving_sustainability'] = recent_sust < older_sust
            
            # Final time progression
            final_times = [t for t in fractions['final'] if pd.notna(t) and t > 0]
            
            if len(final_times) >= 3:
                # Calculate progression curve
                analysis['final_time_progression'] = self._calculate_progression_curve(final_times)
                
                # Best recent vs average
                best_recent = min(final_times[:3]) if len(final_times) >= 3 else min(final_times)
                avg_time = np.mean(final_times)
                analysis['best_vs_average'] = ((avg_time - best_recent) / avg_time) * 100
            
            # Efficiency gains
            analysis['efficiency_score'] = self._calculate_efficiency_gains(recent)
            
            fractional_data.append(analysis)
        
        return pd.DataFrame(fractional_data)
    
    def _calculate_progression_curve(self, times: List[float]) -> str:
        """Determine the shape of progression curve"""
        if len(times) < 3:
            return 'Unknown'
        
        # Fit polynomial to see curve shape
        x = np.arange(len(times))
        
        # Try quadratic fit
        try:
            coeffs = np.polyfit(x, times, 2)
            
            # Positive quadratic term = U-shaped (improving after decline)
            # Negative quadratic term = inverse U (peaking then declining)
            if coeffs[0] > 0.01:
                return 'U_shaped_recovery'
            elif coeffs[0] < -0.01:
                return 'Peaked_declining'
            else:
                # Mostly linear, check slope
                if coeffs[1] < -0.1:
                    return 'Steady_improvement'
                elif coeffs[1] > 0.1:
                    return 'Steady_decline'
                else:
                    return 'Plateau'
        except:
            return 'Unknown'
    
    def _calculate_efficiency_gains(self, recent_df: pd.DataFrame) -> float:
        """Calculate efficiency improvements"""
        efficiency_scores = []
        
        for idx, row in recent_df.iterrows():
            # Energy efficiency: speed rating per unit of pace
            speed = row.get('pp_bris_speed_rating')
            e1_pace = row.get('pp_e1_pace')
            
            if pd.notna(speed) and pd.notna(e1_pace) and e1_pace > 0:
                efficiency = speed / e1_pace * 100
                efficiency_scores.append(efficiency)
        
        if len(efficiency_scores) >= 3:
            # Compare recent to older
            recent_eff = np.mean(efficiency_scores[:2])
            older_eff = np.mean(efficiency_scores[2:])
            
            improvement = ((recent_eff - older_eff) / older_eff) * 100
            return improvement
        
        return 0
    
    def calculate_field_adjusted_performance(self) -> pd.DataFrame:
        """
        Adjust performance for field size and quality
        
        Returns:
            DataFrame with field-adjusted metrics
        """
        logger.info("Calculating field-adjusted performance...")
        
        adjusted_data = []
        
        for (race, horse), group in self.past_df.groupby(['race', 'horse_name']):
            if len(group) < 3:
                continue
            
            recent = group.head(10)
            
            analysis = {
                'race': race,
                'horse_name': horse
            }
            
            # Collect performance data
            performances = []
            
            for idx, row in recent.iterrows():
                finish_pos = row.get('pp_finish_pos')
                field_size = row.get('pp_num_entrants')
                speed_rating = row.get('pp_bris_speed_rating')
                race_type = row.get('pp_race_type')
                purse = row.get('pp_purse')
                
                if all(pd.notna(x) for x in [finish_pos, field_size, speed_rating]):
                    # Calculate horses beaten percentage
                    beaten_pct = ((field_size - finish_pos) / (field_size - 1)) * 100 if field_size > 1 else 0
                    
                    # Estimate field quality
                    quality_score = 50  # Base
                    
                    # Adjust for race type
                    if race_type in ['G1', 'G2', 'G3']:
                        quality_score += 30
                    elif race_type in ['N', 'A']:
                        quality_score += 15
                    elif race_type in ['C']:
                        quality_score -= 10
                    
                    # Adjust for purse
                    if pd.notna(purse):
                        if purse > 100000:
                            quality_score += 10
                        elif purse < 20000:
                            quality_score -= 10
                    
                    # Adjust for field size
                    if field_size >= 10:
                        quality_score += 5
                    elif field_size <= 5:
                        quality_score -= 10
                    
                    performances.append({
                        'finish_pos': finish_pos,
                        'field_size': field_size,
                        'beaten_pct': beaten_pct,
                        'speed_rating': speed_rating,
                        'field_quality': quality_score,
                        'quality_adjusted_rating': speed_rating * (quality_score / 100)
                    })
            
            if performances:
                # Calculate averages
                analysis['avg_beaten_pct'] = np.mean([p['beaten_pct'] for p in performances])
                analysis['avg_field_size'] = np.mean([p['field_size'] for p in performances])
                analysis['avg_field_quality'] = np.mean([p['field_quality'] for p in performances])
                
                # Performance vs expectations
                # Expected finish based on consistent random performance
                expected_finish_pct = 50  # Middle of field
                actual_finish_pct = 100 - analysis['avg_beaten_pct']
                analysis['performance_vs_expected'] = expected_finish_pct - actual_finish_pct
                
                # Quality-adjusted speed rating
                analysis['quality_adjusted_speed'] = np.mean([p['quality_adjusted_rating'] 
                                                             for p in performances])
                
                # Consistency in different field sizes
                small_fields = [p for p in performances if p['field_size'] <= 7]
                large_fields = [p for p in performances if p['field_size'] >= 10]
                
                if small_fields:
                    analysis['small_field_beaten_pct'] = np.mean([p['beaten_pct'] 
                                                                  for p in small_fields])
                if large_fields:
                    analysis['large_field_beaten_pct'] = np.mean([p['beaten_pct'] 
                                                                  for p in large_fields])
                
                # Improvement against quality
                if len(performances) >= 6:
                    recent_quality = np.mean([p['field_quality'] for p in performances[:3]])
                    older_quality = np.mean([p['field_quality'] for p in performances[3:6]])
                    
                    recent_performance = np.mean([p['beaten_pct'] for p in performances[:3]])
                    older_performance = np.mean([p['beaten_pct'] for p in performances[3:6]])
                    
                    # If facing better competition but maintaining performance = improvement
                    quality_change = recent_quality - older_quality
                    performance_change = recent_performance - older_performance
                    
                    analysis['quality_improvement_score'] = performance_change + (quality_change * 0.5)
                else:
                    analysis['quality_improvement_score'] = 0
            
            adjusted_data.append(analysis)
        
        return pd.DataFrame(adjusted_data)
    
    def generate_form_cycle_report(self) -> pd.DataFrame:
        """
        Generate comprehensive form cycle report
        
        Returns:
            DataFrame with complete form cycle analysis
        """
        logger.info("Generating comprehensive form cycle report...")
        
        # Calculate all components
        trajectory = self.analyze_beaten_lengths_trajectory()
        position_calls = self.analyze_position_calls()
        cycle_patterns = self.detect_form_cycle_patterns()
        fractionals = self.analyze_fractional_evolution()
        field_adjusted = self.calculate_field_adjusted_performance()
        
        # Start with base info
        form_report = self.current_df[['race', 'horse_name', 'post_position',
                                       'morn_line_odds_if_available']].copy()
        
        # Merge all analyses
        for df in [trajectory, position_calls, cycle_patterns, fractionals, field_adjusted]:
            if not df.empty:
                merge_cols = [col for col in df.columns if col not in ['race', 'horse_name']]
                form_report = form_report.merge(
                    df[['race', 'horse_name'] + merge_cols],
                    on=['race', 'horse_name'],
                    how='left'
                )
        
        # Calculate composite form score
        form_report['composite_form_score'] = form_report.apply(
            self._calculate_composite_form_score, axis=1
        )
        
        # Identify form edges
        form_report['form_edge'] = form_report.apply(
            self._identify_form_edge, axis=1
        )
        
        # Rank within race
        form_report['form_rank'] = form_report.groupby('race')['composite_form_score'].rank(
            ascending=False, method='min'
        )
        
        return form_report
    
    def _calculate_composite_form_score(self, row) -> float:
        """Calculate overall form score"""
        score = 50  # Base
        
        # Beaten lengths trajectory (20%)
        if pd.notna(row.get('beaten_time_trend')):
            # Positive trend = improvement
            trend_score = 50 + (row['beaten_time_trend'] * 10)
            score += (np.clip(trend_score, 0, 100) - 50) * 0.2
        
        # Position gains (15%)
        if pd.notna(row.get('avg_position_gain')):
            gain_score = 50 + (row['avg_position_gain'] * 5)
            score += (np.clip(gain_score, 0, 100) - 50) * 0.15
        
        # Form cycle state (25%)
        state_scores = {
            'IMPROVING': 80,
            'PEAKING': 85,
            'RECOVERING': 60,
            'FRESHENING': 65,
            'STABLE': 50,
            'BOUNCING': 30,
            'DECLINING': 20,
            'ERRATIC': 40
        }
        state = row.get('form_cycle_state', 'STABLE')
        score += (state_scores.get(state, 50) - 50) * 0.25
        
        # Fractional improvements (15%)
        if pd.notna(row.get('final_time_improvement')):
            time_score = 50 + (row['final_time_improvement'] * 5)
            score += (np.clip(time_score, 0, 100) - 50) * 0.15
        
        # Field-adjusted performance (15%)
        if pd.notna(row.get('quality_improvement_score')):
            quality_score = 50 + row['quality_improvement_score']
            score += (np.clip(quality_score, 0, 100) - 50) * 0.15
        
        # Late kick bonus (10%)
        if row.get('strong_late_kick'):
            score += 5
        
        return np.clip(score, 0, 100)
    
    def _identify_form_edge(self, row) -> str:
        """Identify specific form advantages"""
        edges = []
        
        # Peak form
        if row.get('form_cycle_state') == 'PEAKING':
            edges.append('Peak_form')
        
        # Improving trajectory
        if row.get('beaten_time_trend', 0) > 0.5:
            edges.append('Sharp_improvement')
        
        # Bounce candidate
        if row.get('bounce_risk') == 'High':
            edges.append('Bounce_risk')
        
        # Traffic troubles
        if row.get('pct_races_with_trouble', 0) > 30:
            edges.append('Often_troubled')
        
        # Strong closer
        if row.get('strong_late_kick') and row.get('late_gain_consistency', 0) > 70:
            edges.append('Consistent_closer')
        
        # Fresh horse
        if row.get('currently_freshening') and row.get('handles_layoffs_well'):
            edges.append('Fresh_and_ready')
        
        return ', '.join(edges) if edges else 'None'


def main():
    """Main function to run form cycle detection"""
    
    # Set up paths
    base_path = Path('data/processed')
    current_race_path = base_path / 'current_race_info.parquet'
    past_starts_path = base_path / 'past_starts_long_format.parquet'
    
    # Check if files exist
    for path in [current_race_path, past_starts_path]:
        if not path.exists():
            logger.error(f"File not found: {path}")
            return
    
    # Initialize detector
    detector = FormCycleDetector(
        str(current_race_path),
        str(past_starts_path)
    )
    
    # Generate comprehensive report
    form_report = detector.generate_form_cycle_report()
    
    # Save main report
    output_path = base_path / 'form_cycle_analysis.parquet'
    form_report.to_parquet(output_path, index=False)
    logger.info(f"Saved form cycle analysis to {output_path}")
    
    # Generate component reports
    trajectory = detector.analyze_beaten_lengths_trajectory()
    if not trajectory.empty:
        trajectory_path = base_path / 'beaten_lengths_trajectory.csv'
        trajectory.to_csv(trajectory_path, index=False)
    
    patterns = detector.detect_form_cycle_patterns()
    if not patterns.empty:
        patterns_path = base_path / 'form_cycle_patterns.csv'
        patterns.to_csv(patterns_path, index=False)
    
    # Display summary
    print("\n" + "="*70)
    print("FORM CYCLE ANALYSIS SUMMARY")
    print("="*70)
    
    print(f"\nTotal horses analyzed: {len(form_report)}")
    
    print("\nFORM CYCLE STATES:")
    if 'form_cycle_state' in patterns.columns:
        state_counts = patterns['form_cycle_state'].value_counts()
        for state, count in state_counts.items():
            desc = FormCycleDetector.CYCLE_STATES.get(state, state)
            print(f"  {state}: {count} horses - {desc}")
    
    print("\nTOP IMPROVING HORSES:")
    improving = form_report[form_report['form_cycle_state'] == 'IMPROVING'].nlargest(
        10, 'composite_form_score'
    )
    if len(improving) > 0:
        print(improving[['race', 'horse_name', 'composite_form_score', 
                        'beaten_time_trend', 'form_edge']].to_string(index=False))
    
    print("\nBOUNCE CANDIDATES:")
    bounce_risks = patterns[patterns['bounce_risk'] == 'High']
    if len(bounce_risks) > 0:
        print(bounce_risks[['race', 'horse_name', 'bounce_risk', 
                           'last_race_career_best']].to_string(index=False))
    
    print("\nFRESH HORSES:")
    fresh = patterns[patterns['currently_freshening'] == True]
    if len(fresh) > 0:
        print(fresh[['race', 'horse_name', 'freshening_days', 
                    'handles_layoffs_well']].to_string(index=False))
    
    # Save summary report
    summary_path = base_path / 'form_cycle_summary.csv'
    summary_cols = ['race', 'horse_name', 'post_position', 'composite_form_score',
                   'form_rank', 'form_cycle_state', 'form_edge', 'beaten_time_trend',
                   'avg_position_gain', 'strong_late_kick', 'bounce_risk']
    
    available_cols = [col for col in summary_cols if col in form_report.columns]
    summary_df = form_report[available_cols].sort_values(['race', 'form_rank'])
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved form cycle summary to {summary_path}")


if __name__ == '__main__':
    main()