# src/transformers/advanced_pace_projection.py
"""
Advanced Pace Scenario Projection Module

Projects race pace scenarios using:
1. Multi-Factor Pace Model
2. Dynamic Pace Pressure Calculator
3. Energy Cost Modeling
4. Pace Shape Classification
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedPaceProjection:
    """Project and analyze race pace scenarios"""
    
    # BRIS Run Style mappings
    RUN_STYLE_CATEGORIES = {
        'E': 'Early',
        'E/P': 'Early/Presser', 
        'P': 'Presser',
        'P/S': 'Presser/Stalker',
        'S': 'Stalker',
        'SS': 'Sustained',
        'U': 'Unknown'
    }
    
    # Expected position ranges by style
    STYLE_POSITION_RANGES = {
        'E': {'early': (1, 2), 'mid': (1, 3), 'late': (1, 5)},
        'E/P': {'early': (1, 3), 'mid': (2, 4), 'late': (2, 6)},
        'P': {'early': (2, 5), 'mid': (2, 5), 'late': (2, 6)},
        'P/S': {'early': (3, 6), 'mid': (3, 5), 'late': (2, 5)},
        'S': {'early': (4, 8), 'mid': (3, 6), 'late': (1, 4)},
        'SS': {'early': (6, 12), 'mid': (5, 10), 'late': (1, 5)}
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
        
        # Calculate race-level information
        self._prepare_race_data()
    
    def _prepare_race_data(self):
        """Prepare race-level aggregated data"""
        # Get unique races
        self.races = self.current_df['race'].unique()
        
        # Store race compositions
        self.race_compositions = {}
        for race in self.races:
            race_horses = self.current_df[self.current_df['race'] == race]
            self.race_compositions[race] = {
                'num_horses': len(race_horses),
                'horses': race_horses['horse_name'].tolist(),
                'run_styles': race_horses['bris_run_style_designation'].tolist(),
                'speed_points': race_horses['quirin_style_speed_points'].tolist(),
                'post_positions': race_horses['post_position'].tolist()
            }
    
    def calculate_multi_factor_pace_model(self) -> pd.DataFrame:
        """
        Calculate multi-factor pace projections for each horse
        
        Returns:
            DataFrame with pace projections
        """
        logger.info("Calculating multi-factor pace model...")
        
        pace_projections = []
        
        for race in self.races:
            race_data = self.current_df[self.current_df['race'] == race]
            
            for idx, horse_row in race_data.iterrows():
                horse = horse_row['horse_name']
                
                # Get horse's past performances
                horse_past = self.past_df[
                    (self.past_df['race'] == race) & 
                    (self.past_df['horse_name'] == horse)
                ].head(10)
                
                if len(horse_past) == 0:
                    continue
                
                # 1. Base run style component
                run_style = horse_row['bris_run_style_designation']
                speed_points = horse_row['quirin_style_speed_points']
                
                # 2. Historical pace figures
                pace_metrics = {
                    'avg_e1_pace': horse_past['pp_e1_pace'].mean(),
                    'best_e1_pace': horse_past['pp_e1_pace'].max(),
                    'avg_e2_pace': horse_past['pp_e2_pace'].mean(),
                    'best_e2_pace': horse_past['pp_e2_pace'].max(),
                    'avg_late_pace': horse_past['pp_bris_late_pace'].mean(),
                    'best_late_pace': horse_past['pp_bris_late_pace'].max()
                }
                
                # 3. Position consistency analysis
                position_data = self._analyze_position_consistency(horse_past)
                
                # 4. Post position adjustment
                post_adjustment = self._calculate_post_position_adjustment(
                    horse_row['post_position'],
                    len(race_data),
                    run_style
                )
                
                # 5. Recent form adjustment
                recent_form = self._calculate_recent_form_factor(horse_past.head(3))
                
                # Calculate projected positions
                projected_positions = self._project_positions(
                    run_style,
                    speed_points,
                    pace_metrics,
                    position_data,
                    post_adjustment,
                    recent_form,
                    len(race_data)
                )
                
                pace_projections.append({
                    'race': race,
                    'horse_name': horse,
                    'run_style': run_style,
                    'speed_points': speed_points,
                    'post_position': horse_row['post_position'],
                    **pace_metrics,
                    **position_data,
                    'post_adjustment': post_adjustment,
                    'recent_form_factor': recent_form,
                    **projected_positions
                })
        
        return pd.DataFrame(pace_projections)
    
    def _analyze_position_consistency(self, past_performances: pd.DataFrame) -> Dict:
        """Analyze how consistently a horse runs to its style"""
        position_stats = {}
        
        # Early position consistency
        early_positions = past_performances['pp_first_call_pos'].dropna()
        if len(early_positions) >= 3:
            position_stats['avg_early_position'] = early_positions.mean()
            position_stats['std_early_position'] = early_positions.std()
            position_stats['early_consistency'] = 1 / (1 + early_positions.std())
        else:
            position_stats['avg_early_position'] = 5
            position_stats['std_early_position'] = 2
            position_stats['early_consistency'] = 0.5
        
        # Late position consistency
        finish_positions = past_performances['pp_finish_pos'].dropna()
        if len(finish_positions) >= 3:
            position_stats['avg_finish_position'] = finish_positions.mean()
            position_stats['position_improvement'] = (
                position_stats['avg_early_position'] - 
                position_stats['avg_finish_position']
            )
        else:
            position_stats['avg_finish_position'] = 5
            position_stats['position_improvement'] = 0
        
        return position_stats
    
    def _calculate_post_position_adjustment(self, post: int, field_size: int, 
                                          run_style: str) -> float:
        """Calculate position adjustment based on post position"""
        # Inside posts favor early speed
        # Outside posts can be disadvantageous for early speed
        
        if pd.isna(run_style):
            return 0
        
        # Normalize post position (0 to 1)
        normalized_post = (post - 1) / max(1, field_size - 1)
        
        # Early runners prefer inside
        if run_style in ['E', 'E/P']:
            adjustment = -normalized_post * 0.5  # Negative = harder from outside
        # Closers might prefer outside to avoid traffic
        elif run_style in ['S', 'SS']:
            adjustment = normalized_post * 0.2  # Slight positive for outside
        else:
            adjustment = 0
        
        return adjustment
    
    def _calculate_recent_form_factor(self, recent_races: pd.DataFrame) -> float:
        """Calculate form factor based on recent performances"""
        if len(recent_races) == 0:
            return 1.0
        
        # Look at speed ratings trend
        speed_ratings = recent_races['pp_bris_speed_rating'].dropna()
        if len(speed_ratings) >= 2:
            # Calculate trend
            x = np.arange(len(speed_ratings))
            slope, _, _, _, _ = stats.linregress(x, speed_ratings)
            
            # Positive slope = improving
            form_factor = 1.0 + (slope / 10)  # Scale adjustment
        else:
            form_factor = 1.0
        
        return np.clip(form_factor, 0.8, 1.2)
    
    def _project_positions(self, run_style: str, speed_points: float,
                          pace_metrics: Dict, position_data: Dict,
                          post_adj: float, form_factor: float,
                          field_size: int) -> Dict:
        """Project positions at each call"""
        projections = {}
        
        # Base projections from run style
        if pd.isna(run_style) or run_style not in self.STYLE_POSITION_RANGES:
            style_ranges = self.STYLE_POSITION_RANGES['U']
        else:
            style_ranges = self.STYLE_POSITION_RANGES.get(run_style, self.STYLE_POSITION_RANGES['U'])
        
        # Early position (1st call)
        early_base = np.mean(style_ranges['early'])
        
        # Adjust for speed points
        if pd.notna(speed_points):
            early_base -= (speed_points - 4) * 0.3  # More points = closer to lead
        
        # Adjust for early pace ability
        if pd.notna(pace_metrics['avg_e1_pace']):
            # Higher E1 pace = faster early = lower position number
            pace_adjustment = (pace_metrics['avg_e1_pace'] - 95) / 10
            early_base -= pace_adjustment
        
        # Apply adjustments
        early_base += post_adj
        early_base *= form_factor
        
        # Consistency adjustment
        if 'early_consistency' in position_data:
            # Higher consistency = more likely to hit projection
            consistency_factor = position_data['early_consistency']
        else:
            consistency_factor = 0.7
        
        projections['projected_1st_call_position'] = max(1, min(field_size, round(early_base)))
        projections['position_confidence_early'] = consistency_factor * 100
        
        # Mid-race position (2nd call)
        mid_base = np.mean(style_ranges['mid'])
        
        # Adjust for E2 pace
        if pd.notna(pace_metrics['avg_e2_pace']):
            pace_adjustment = (pace_metrics['avg_e2_pace'] - 95) / 10
            mid_base -= pace_adjustment
        
        mid_base *= form_factor
        projections['projected_2nd_call_position'] = max(1, min(field_size, round(mid_base)))
        
        # Late position (stretch)
        late_base = np.mean(style_ranges['late'])
        
        # Adjust for late pace and position improvement history
        if pd.notna(pace_metrics['avg_late_pace']):
            pace_adjustment = (pace_metrics['avg_late_pace'] - 95) / 10
            late_base -= pace_adjustment
        
        if 'position_improvement' in position_data:
            late_base -= position_data['position_improvement'] * 0.3
        
        late_base *= form_factor
        projections['projected_stretch_position'] = max(1, min(field_size, round(late_base)))
        projections['projected_finish_position'] = projections['projected_stretch_position']
        
        return projections
    
    def calculate_pace_pressure(self) -> pd.DataFrame:
        """
        Calculate dynamic pace pressure for each race
        
        Returns:
            DataFrame with pace pressure analysis
        """
        logger.info("Calculating pace pressure...")
        
        # First get pace projections
        pace_projections = self.calculate_multi_factor_pace_model()
        
        pace_pressure_data = []
        
        for race in self.races:
            race_projections = pace_projections[pace_projections['race'] == race]
            
            if len(race_projections) == 0:
                continue
            
            # Count horses vying for early positions
            early_speed_horses = 0
            pace_setters = []
            pressers = []
            
            for idx, horse in race_projections.iterrows():
                if horse['projected_1st_call_position'] <= 2:
                    early_speed_horses += 1
                    pace_setters.append(horse['horse_name'])
                elif horse['projected_1st_call_position'] <= 4:
                    pressers.append(horse['horse_name'])
            
            # Calculate pace pressure metrics
            field_size = len(race_projections)
            
            # Raw pressure score
            pressure_score = (early_speed_horses / field_size) * 100
            
            # Adjusted for speed points concentration
            speed_points = race_projections['speed_points'].fillna(0)
            high_speed_points = (speed_points >= 6).sum()
            
            if high_speed_points >= 3:
                pressure_score *= 1.5  # Multiple high speed point horses
            elif high_speed_points >= 2:
                pressure_score *= 1.2
            
            # E1 pace concentration
            e1_paces = race_projections['avg_e1_pace'].dropna()
            if len(e1_paces) >= 3:
                # High E1 pace horses competing
                fast_e1_count = (e1_paces >= e1_paces.quantile(0.75)).sum()
                if fast_e1_count >= 2:
                    pressure_score *= 1.3
            
            # Classify pace scenario
            if pressure_score >= 150:
                pace_scenario = 'Hot'
                pace_description = 'Extremely contentious early pace likely'
            elif pressure_score >= 100:
                pace_scenario = 'Contested'
                pace_description = 'Multiple horses will vie for lead'
            elif pressure_score >= 50:
                pace_scenario = 'Honest'
                pace_description = 'Normal pace pressure expected'
            elif early_speed_horses == 1:
                pace_scenario = 'Lone Speed'
                pace_description = 'Single pacesetter likely to control'
            else:
                pace_scenario = 'Slow'
                pace_description = 'Lack of early pace, tactical race'
            
            # Identify pace beneficiaries
            beneficiaries = []
            if pace_scenario in ['Hot', 'Contested']:
                # Closers benefit from hot pace
                closers = race_projections[
                    race_projections['run_style'].isin(['S', 'SS', 'P/S'])
                ]['horse_name'].tolist()
                beneficiaries = closers
            elif pace_scenario == 'Lone Speed':
                # Lone speed benefits
                if pace_setters:
                    beneficiaries = pace_setters[:1]
            
            pace_pressure_data.append({
                'race': race,
                'field_size': field_size,
                'early_speed_count': early_speed_horses,
                'high_speed_point_count': high_speed_points,
                'pace_pressure_score': pressure_score,
                'pace_scenario': pace_scenario,
                'pace_description': pace_description,
                'pace_setters': ', '.join(pace_setters),
                'pressers': ', '.join(pressers),
                'likely_beneficiaries': ', '.join(beneficiaries),
                'projected_early_fractions': self._project_fractions(race_projections, pace_scenario)
            })
        
        return pd.DataFrame(pace_pressure_data)
    
    def _project_fractions(self, race_projections: pd.DataFrame, 
                          pace_scenario: str) -> str:
        """Project likely fractional times based on pace scenario"""
        # Get average E1 pace for likely leaders
        leaders = race_projections.nsmallest(2, 'projected_1st_call_position')
        
        if len(leaders) == 0:
            return "Unable to project"
        
        avg_e1 = leaders['avg_e1_pace'].mean()
        
        if pd.isna(avg_e1):
            return "Insufficient data"
        
        # Adjust for pace pressure
        if pace_scenario == 'Hot':
            avg_e1 += 3  # Faster due to pressure
        elif pace_scenario == 'Contested':
            avg_e1 += 1.5
        elif pace_scenario == 'Slow':
            avg_e1 -= 2  # Slower pace
        
        # Convert to time estimate (rough approximation)
        # This would need track-specific par times for accuracy
        if avg_e1 >= 100:
            quarter_estimate = "22.2 - 22.4"
            half_estimate = "45.0 - 45.4"
        elif avg_e1 >= 97:
            quarter_estimate = "22.4 - 22.8"
            half_estimate = "45.4 - 46.0"
        elif avg_e1 >= 94:
            quarter_estimate = "22.8 - 23.2"
            half_estimate = "46.0 - 46.8"
        else:
            quarter_estimate = "23.2+"
            half_estimate = "47.0+"
        
        return f"1/4: {quarter_estimate}, 1/2: {half_estimate}"
    
    def calculate_energy_cost_model(self) -> pd.DataFrame:
        """
        Model energy expenditure and sustainability
        
        Returns:
            DataFrame with energy cost analysis
        """
        logger.info("Calculating energy cost model...")
        
        pace_projections = self.calculate_multi_factor_pace_model()
        pace_pressure = self.calculate_pace_pressure()
        
        energy_analysis = []
        
        for race in self.races:
            race_proj = pace_projections[pace_projections['race'] == race]
            race_pressure = pace_pressure[pace_pressure['race'] == race]
            
            if len(race_proj) == 0 or len(race_pressure) == 0:
                continue
            
            pace_scenario = race_pressure.iloc[0]['pace_scenario']
            
            for idx, horse in race_proj.iterrows():
                # Calculate energy expenditure by phase
                energy_data = {
                    'race': race,
                    'horse_name': horse['horse_name'],
                    'run_style': horse['run_style']
                }
                
                # Early energy cost
                if horse['projected_1st_call_position'] <= 2:
                    # On or near lead
                    if pace_scenario in ['Hot', 'Contested']:
                        energy_data['early_energy_cost'] = 35  # High cost due to pressure
                    else:
                        energy_data['early_energy_cost'] = 25  # Comfortable lead
                elif horse['projected_1st_call_position'] <= 4:
                    # Pressing
                    energy_data['early_energy_cost'] = 20
                else:
                    # Settling back
                    energy_data['early_energy_cost'] = 15
                
                # Middle race energy
                position_change = (
                    horse['projected_2nd_call_position'] - 
                    horse['projected_1st_call_position']
                )
                
                if abs(position_change) > 2:
                    energy_data['middle_energy_cost'] = 25  # Making move
                else:
                    energy_data['middle_energy_cost'] = 20  # Maintaining
                
                # Late energy requirements
                late_move = (
                    horse['projected_2nd_call_position'] - 
                    horse['projected_stretch_position']
                )
                
                energy_data['late_energy_required'] = max(20, late_move * 5)
                
                # Total energy budget (100 units)
                energy_data['total_energy_used'] = (
                    energy_data['early_energy_cost'] +
                    energy_data['middle_energy_cost'] +
                    energy_data['late_energy_required']
                )
                
                # Energy reserve
                energy_data['energy_reserve'] = 100 - energy_data['total_energy_used']
                
                # Sustainability analysis
                if pd.notna(horse['avg_late_pace']) and pd.notna(horse['avg_e1_pace']):
                    # Horses with good late pace can sustain better
                    pace_differential = horse['avg_late_pace'] - horse['avg_e1_pace']
                    energy_data['sustainability_factor'] = 50 + pace_differential
                else:
                    energy_data['sustainability_factor'] = 50
                
                # Fade risk assessment
                if energy_data['energy_reserve'] < 0:
                    energy_data['fade_risk'] = 'High'
                    energy_data['projected_fade_point'] = '1/8 pole'
                elif energy_data['energy_reserve'] < 10:
                    energy_data['fade_risk'] = 'Moderate'
                    energy_data['projected_fade_point'] = '1/16 pole'
                elif energy_data['early_energy_cost'] > 30 and pace_scenario == 'Hot':
                    energy_data['fade_risk'] = 'Moderate'
                    energy_data['projected_fade_point'] = '3/16 pole'
                else:
                    energy_data['fade_risk'] = 'Low'
                    energy_data['projected_fade_point'] = 'None'
                
                # Optimal energy distribution for style
                energy_data['energy_efficiency'] = self._calculate_energy_efficiency(
                    horse['run_style'],
                    energy_data['early_energy_cost'],
                    energy_data['middle_energy_cost'],
                    energy_data['late_energy_required']
                )
                
                energy_analysis.append(energy_data)
        
        return pd.DataFrame(energy_analysis)
    
    def _calculate_energy_efficiency(self, run_style: str, early: float, 
                                   middle: float, late: float) -> float:
        """Calculate how efficiently horse uses energy for its style"""
        if pd.isna(run_style):
            return 50
        
        # Ideal energy distribution by style
        ideal_distributions = {
            'E': {'early': 35, 'middle': 35, 'late': 30},
            'E/P': {'early': 30, 'middle': 35, 'late': 35},
            'P': {'early': 25, 'middle': 40, 'late': 35},
            'P/S': {'early': 20, 'middle': 40, 'late': 40},
            'S': {'early': 15, 'middle': 35, 'late': 50},
            'SS': {'early': 10, 'middle': 30, 'late': 60}
        }
        
        ideal = ideal_distributions.get(run_style, {'early': 25, 'middle': 35, 'late': 40})
        
        # Calculate deviation from ideal
        actual_total = early + middle + late
        if actual_total == 0:
            return 50
        
        actual_dist = {
            'early': (early / actual_total) * 100,
            'middle': (middle / actual_total) * 100,
            'late': (late / actual_total) * 100
        }
        
        # Mean absolute deviation
        deviation = np.mean([
            abs(actual_dist[phase] - ideal[phase]) 
            for phase in ['early', 'middle', 'late']
        ])
        
        # Convert to efficiency score (0-100)
        efficiency = max(0, 100 - deviation * 2)
        
        return efficiency
    
    def classify_pace_shapes(self) -> pd.DataFrame:
        """
        Classify races into pace shape categories
        
        Returns:
            DataFrame with pace shape classifications
        """
        logger.info("Classifying pace shapes...")
        
        pace_pressure = self.calculate_pace_pressure()
        energy_model = self.calculate_energy_cost_model()
        
        pace_shapes = []
        
        for race in self.races:
            race_pressure = pace_pressure[pace_pressure['race'] == race]
            race_energy = energy_model[energy_model['race'] == race]
            
            if len(race_pressure) == 0:
                continue
            
            pressure_data = race_pressure.iloc[0]
            
            # Analyze energy distribution of field
            early_types = race_energy[race_energy['run_style'].isin(['E', 'E/P'])]
            late_types = race_energy[race_energy['run_style'].isin(['S', 'SS'])]
            
            avg_early_energy = race_energy['early_energy_cost'].mean()
            avg_late_energy = race_energy['late_energy_required'].mean()
            
            # High fade risk horses
            high_fade_count = (race_energy['fade_risk'] == 'High').sum()
            
            # Classify pace shape
            if pressure_data['pace_scenario'] in ['Hot', 'Contested']:
                if high_fade_count >= 2:
                    pace_shape = 'Fast-Fast-Collapse'
                    shape_description = 'Blazing early fractions leading to late collapse'
                else:
                    pace_shape = 'Fast-Fast-Moderate'
                    shape_description = 'Quick early pace maintained reasonably'
            elif pressure_data['pace_scenario'] == 'Honest':
                if avg_late_energy > avg_early_energy:
                    pace_shape = 'Moderate-Fast-Fast'
                    shape_description = 'Building pace with strong late move'
                else:
                    pace_shape = 'Even-Even-Even'
                    shape_description = 'Steady, evenly run race throughout'
            elif pressure_data['pace_scenario'] == 'Slow':
                pace_shape = 'Slow-Slow-Sprint'
                shape_description = 'Tactical early pace, sprint home'
            else:  # Lone Speed
                if len(late_types) >= 3:
                    pace_shape = 'Moderate-Even-Fast'
                    shape_description = 'Controlled pace with late runners closing'
                else:
                    pace_shape = 'Wire-to-Wire'
                    shape_description = 'Front-runner likely to control throughout'
            
            # Identify horses suited to pace shape
            suited_horses = []
            
            if 'Collapse' in pace_shape or 'Sprint' in pace_shape:
                # Closers benefit
                suited = race_energy[
                    (race_energy['run_style'].isin(['S', 'SS', 'P/S'])) &
                    (race_energy['sustainability_factor'] > 50)
                ]['horse_name'].tolist()
                suited_horses.extend(suited)
            elif 'Wire' in pace_shape:
                # Front runner benefits
                suited = race_energy[
                    (race_energy['run_style'].isin(['E', 'E/P'])) &
                    (race_energy['fade_risk'] == 'Low')
                ]['horse_name'].tolist()
                suited_horses.extend(suited)
            else:
                # Tactical race suits versatile types
                suited = race_energy[
                    race_energy['energy_efficiency'] > 70
                ]['horse_name'].tolist()
                suited_horses.extend(suited)
            
            pace_shapes.append({
                'race': race,
                'pace_scenario': pressure_data['pace_scenario'],
                'pace_shape': pace_shape,
                'shape_description': shape_description,
                'avg_early_energy': avg_early_energy,
                'avg_late_energy': avg_late_energy,
                'high_fade_risk_count': high_fade_count,
                'horses_suited_to_shape': ', '.join(suited_horses[:5]),  # Top 5
                'recommended_bet_types': self._recommend_bet_types(pace_shape, race_energy)
            })
        
        return pd.DataFrame(pace_shapes)
    
    def _recommend_bet_types(self, pace_shape: str, race_energy: pd.DataFrame) -> str:
        """Recommend bet types based on pace shape"""
        recommendations = []
        
        if 'Collapse' in pace_shape:
            recommendations.append("Closers in exactas/trifectas")
            recommendations.append("Fade early speed")
        elif 'Wire' in pace_shape:
            recommendations.append("Front-runner win bets")
            recommendations.append("Key speed horse on top")
        elif 'Sprint' in pace_shape:
            recommendations.append("Late runners for place/show")
            recommendations.append("Exacta boxes with closers")
        elif 'Even' in pace_shape:
            recommendations.append("Focus on class/form")
            recommendations.append("Wide exacta coverage")
        
        # Check for standout efficiency
        high_efficiency = race_energy[race_energy['energy_efficiency'] > 80]
        if len(high_efficiency) > 0:
            top_horse = high_efficiency.iloc[0]['horse_name']
            recommendations.append(f"Consider {top_horse} (high efficiency)")
        
        return '; '.join(recommendations[:3])
    
    def generate_pace_report(self) -> pd.DataFrame:
        """
        Generate comprehensive pace analysis report
        
        Returns:
            DataFrame with complete pace analysis
        """
        logger.info("Generating comprehensive pace report...")
        
        # Get all components
        pace_projections = self.calculate_multi_factor_pace_model()
        pace_pressure = self.calculate_pace_pressure()
        energy_model = self.calculate_energy_cost_model()
        pace_shapes = self.classify_pace_shapes()
        
        # Combine into comprehensive report
        pace_report = pace_projections.merge(
            energy_model[['race', 'horse_name', 'total_energy_used', 
                         'energy_reserve', 'fade_risk', 'energy_efficiency']],
            on=['race', 'horse_name'],
            how='left'
        )
        
        # Add race-level information
        for race in self.races:
            race_pressure_info = pace_pressure[pace_pressure['race'] == race]
            race_shape_info = pace_shapes[pace_shapes['race'] == race]
            
            if len(race_pressure_info) > 0:
                pressure_info = race_pressure_info.iloc[0]
                pace_report.loc[pace_report['race'] == race, 'pace_scenario'] = pressure_info['pace_scenario']
                pace_report.loc[pace_report['race'] == race, 'pace_pressure_score'] = pressure_info['pace_pressure_score']
            
            if len(race_shape_info) > 0:
                shape_info = race_shape_info.iloc[0]
                pace_report.loc[pace_report['race'] == race, 'pace_shape'] = shape_info['pace_shape']
        
        # Add pace advantage scores
        pace_report['pace_advantage_score'] = pace_report.apply(
            self._calculate_pace_advantage, axis=1
        )
        
        # Rank within race
        pace_report['pace_rank'] = pace_report.groupby('race')['pace_advantage_score'].rank(
            ascending=False, method='min'
        )
        
        return pace_report
    
    def _calculate_pace_advantage(self, row) -> float:
        """Calculate overall pace advantage score"""
        score = 50  # Base score
        
        # Energy efficiency bonus
        if pd.notna(row.get('energy_efficiency')):
            score += (row['energy_efficiency'] - 50) * 0.3
        
        # Low fade risk bonus
        if row.get('fade_risk') == 'Low':
            score += 10
        elif row.get('fade_risk') == 'High':
            score -= 10
        
        # Pace scenario fit
        if row.get('pace_scenario') == 'Hot' and row.get('run_style') in ['S', 'SS']:
            score += 15  # Closers in hot pace
        elif row.get('pace_scenario') == 'Lone Speed' and row.get('projected_1st_call_position') == 1:
            score += 20  # Lone leader
        
        # Energy reserve bonus
        if pd.notna(row.get('energy_reserve')):
            score += row['energy_reserve'] * 0.2
        
        return np.clip(score, 0, 100)


def main():
    """Main function to run pace analysis"""
    
    # Set up paths
    base_path = Path('data/processed')
    current_race_path = base_path / 'current_race_info.parquet'
    past_starts_path = base_path / 'past_starts_long_format.parquet'
    
    # Check if files exist
    for path in [current_race_path, past_starts_path]:
        if not path.exists():
            logger.error(f"File not found: {path}")
            return
    
    # Initialize analyzer
    analyzer = AdvancedPaceProjection(
        str(current_race_path),
        str(past_starts_path)
    )
    
    # Generate comprehensive pace report
    pace_report = analyzer.generate_pace_report()
    
    # Get individual analysis components
    pace_pressure = analyzer.calculate_pace_pressure()
    pace_shapes = analyzer.classify_pace_shapes()
    
    # Save results
    output_path = base_path / 'advanced_pace_analysis.parquet'
    pace_report.to_parquet(output_path, index=False)
    logger.info(f"Saved pace analysis to {output_path}")
    
    # Save summary reports
    pressure_path = base_path / 'pace_pressure_summary.csv'
    pace_pressure.to_csv(pressure_path, index=False)
    
    shapes_path = base_path / 'pace_shapes_summary.csv' 
    pace_shapes.to_csv(shapes_path, index=False)
    
    # Display summary
    print("\n" + "="*70)
    print("PACE ANALYSIS SUMMARY")
    print("="*70)
    
    print("\nPACE SCENARIOS BY RACE:")
    for idx, row in pace_pressure.iterrows():
        print(f"\nRace {row['race']}:")
        print(f"  Scenario: {row['pace_scenario']}")
        print(f"  Description: {row['pace_description']}")
        print(f"  Pressure Score: {row['pace_pressure_score']:.1f}")
        print(f"  Likely Leaders: {row['pace_setters']}")
        if row['likely_beneficiaries']:
            print(f"  Likely Beneficiaries: {row['likely_beneficiaries']}")
    
    print("\n" + "="*70)
    print("PACE SHAPES BY RACE:")
    for idx, row in pace_shapes.iterrows():
        print(f"\nRace {row['race']}:")
        print(f"  Shape: {row['pace_shape']}")
        print(f"  Description: {row['shape_description']}")
        print(f"  Suited Horses: {row['horses_suited_to_shape']}")
        print(f"  Betting Strategy: {row['recommended_bet_types']}")
    
    print("\n" + "="*70)
    print("TOP PACE ADVANTAGE HORSES:")
    top_horses = pace_report.nlargest(10, 'pace_advantage_score')[
        ['race', 'horse_name', 'run_style', 'pace_advantage_score',
         'projected_1st_call_position', 'fade_risk', 'energy_efficiency']
    ]
    print(top_horses.to_string(index=False))
    
    # Save detailed summary
    summary_path = base_path / 'pace_analysis_summary.csv'
    summary_cols = [
        'race', 'horse_name', 'post_position', 'run_style', 'speed_points',
        'pace_advantage_score', 'pace_rank', 'projected_1st_call_position',
        'fade_risk', 'energy_efficiency', 'pace_scenario', 'pace_shape'
    ]
    
    available_cols = [col for col in summary_cols if col in pace_report.columns]
    summary_df = pace_report[available_cols].sort_values(['race', 'pace_rank'])
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved pace summary to {summary_path}")


if __name__ == '__main__':
    main()