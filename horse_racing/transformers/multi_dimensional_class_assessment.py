# src/transformers/multi_dimensional_class_assessment.py
"""
Multi-Dimensional Class Assessment Module

Creates sophisticated class ratings using:
1. Earnings-Based Class Metrics
2. Race Classification Hierarchy
3. Hidden Class Indicators
4. Competitive Level Index
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from config.settings import PROCESSED_DATA_DIR, CURRENT_RACE_INFO, PAST_STARTS_LONG
from scipy import stats
import re
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiDimensionalClassAssessment:
    """Comprehensive class rating system"""
    
    # Race type hierarchy (higher number = higher class)
    RACE_TYPE_HIERARCHY = {
        'M': 1,      # Maiden claiming
        'S': 2,      # Maiden special weight
        'MO': 2.5,   # Maiden optional claiming
        'C': 3,      # Claiming
        'CO': 4,     # Optional claiming
        'R': 5,      # Starter allowance
        'T': 5,      # Starter handicap
        'A': 6,      # Allowance
        'AO': 6.5,   # Allowance optional claiming
        'N': 7,      # Non-graded stakes
        'NO': 7.5,   # Optional claiming stakes
        'G3': 8,     # Grade 3
        'G2': 9,     # Grade 2
        'G1': 10     # Grade 1
    }
    
    # Classification patterns for parsing
    CLASSIFICATION_PATTERNS = {
        'claiming': r'[Cc]lm?\s*(\d+)(?:000)?',
        'allowance': r'[Aa]lw?\s*(\d+)(?:000)?',
        'stakes': r'[Ss]tk',
        'maiden': r'[Mm]dn|[Mm]aiden',
        'handicap': r'[Hh]cp|[Hh]andicap',
        'optional': r'[Oo]pt|[Oo]ptional'
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
        
        # Pre-calculate some aggregates
        self._prepare_earnings_data()
    
    def _prepare_earnings_data(self):
        """Pre-calculate earnings aggregates"""
        # Calculate average purses by race type
        self.avg_purse_by_type = self.past_df.groupby('pp_race_type')['pp_purse'].agg(['mean', 'median', 'std'])
        
        # Calculate percentiles for various metrics
        self.earnings_percentiles = {
            'lifetime_per_start': self.current_df['lifetime_earnings_per_start'].quantile([0.25, 0.5, 0.75, 0.9]),
            'current_year_per_start': self.current_df['current_year_earnings_per_start'].quantile([0.25, 0.5, 0.75, 0.9])
        }
    
    def calculate_earnings_based_metrics(self) -> pd.DataFrame:
        """
        Calculate sophisticated earnings-based class metrics
        
        Returns:
            DataFrame with earnings analysis
        """
        logger.info("Calculating earnings-based class metrics...")
        
        earnings_metrics = []
        
        for idx, horse in self.current_df.iterrows():
            metrics = {
                'race': horse['race'],
                'horse_name': horse['horse_name']
            }
            
            # 1. Earnings per start analysis
            lifetime_eps = horse.get('lifetime_earnings_per_start', 0)
            current_eps = horse.get('current_year_earnings_per_start', 0)
            previous_eps = horse.get('previous_year_earnings_per_start', 0)
            
            # Calculate earnings velocity (improvement rate)
            if previous_eps > 0:
                earnings_velocity = ((current_eps - previous_eps) / previous_eps) * 100
            else:
                earnings_velocity = 0 if current_eps == 0 else 100
            
            metrics['lifetime_earnings_per_start'] = lifetime_eps
            metrics['current_year_eps'] = current_eps
            metrics['earnings_velocity'] = earnings_velocity
            
            # 2. Surface/Distance specific earnings
            metrics['dirt_eps'] = self._calculate_surface_eps(horse, 'dirt')
            metrics['turf_eps'] = self._calculate_surface_eps(horse, 'turf')
            metrics['distance_eps'] = horse.get('distance_earnings_per_start', 0)
            metrics['track_eps'] = horse.get('track_earnings_per_start', 0)
            
            # 3. Earnings percentile rankings
            metrics['lifetime_eps_percentile'] = self._get_percentile_rank(
                lifetime_eps, 
                self.earnings_percentiles['lifetime_per_start']
            )
            
            metrics['current_eps_percentile'] = self._get_percentile_rank(
                current_eps,
                self.earnings_percentiles['current_year_per_start']
            )
            
            # 4. Consistency metrics
            lifetime_starts = horse.get('starts_pos_97', 0)
            lifetime_itm = horse.get('lifetime_itm_pct', 0)
            
            # High earners with high ITM% = consistent class
            if lifetime_starts >= 5:
                consistency_score = (lifetime_itm / 100) * metrics['lifetime_eps_percentile']
                metrics['earnings_consistency_score'] = consistency_score
            else:
                metrics['earnings_consistency_score'] = 50
            
            # 5. Recent earnings trend from past performances
            horse_past = self.past_df[
                (self.past_df['race'] == horse['race']) & 
                (self.past_df['horse_name'] == horse['horse_name'])
            ].head(10)
            
            if len(horse_past) >= 3:
                recent_purses = horse_past['pp_purse'].head(5)
                older_purses = horse_past['pp_purse'].tail(5)
                
                avg_recent = recent_purses.mean()
                avg_older = older_purses.mean() if len(older_purses) > 0 else avg_recent
                
                if avg_older > 0:
                    purse_trend = ((avg_recent - avg_older) / avg_older) * 100
                else:
                    purse_trend = 0
                
                metrics['purse_trend'] = purse_trend
                metrics['avg_recent_purse'] = avg_recent
            else:
                metrics['purse_trend'] = 0
                metrics['avg_recent_purse'] = horse.get('purse', 0)
            
            # 6. Composite earnings class score
            metrics['earnings_class_score'] = self._calculate_earnings_class_score(metrics)
            
            earnings_metrics.append(metrics)
        
        return pd.DataFrame(earnings_metrics)
    
    def _calculate_surface_eps(self, horse: pd.Series, surface: str) -> float:
        """Calculate earnings per start for specific surface"""
        if surface == 'dirt':
            earnings = horse.get('earnings_fast_dirt', 0)
            starts = horse.get('starts_fast_dirt', 0)
        elif surface == 'turf':
            earnings = horse.get('turf_earnings_per_start', 0)
            return earnings  # Already per start
        else:
            return 0
        
        if starts > 0:
            return earnings / starts
        return 0
    
    def _get_percentile_rank(self, value: float, percentiles: pd.Series) -> float:
        """Get percentile rank for a value"""
        if value <= percentiles[0.25]:
            return 25
        elif value <= percentiles[0.5]:
            return 50
        elif value <= percentiles[0.75]:
            return 75
        elif value <= percentiles[0.9]:
            return 90
        else:
            return 95
    
    def _calculate_earnings_class_score(self, metrics: Dict) -> float:
        """Calculate composite earnings class score"""
        # Weight different components
        score = 0
        
        # Lifetime earnings percentile (30%)
        score += metrics.get('lifetime_eps_percentile', 50) * 0.3
        
        # Current form (25%)
        score += metrics.get('current_eps_percentile', 50) * 0.25
        
        # Consistency (20%)
        score += metrics.get('earnings_consistency_score', 50) * 0.2
        
        # Earnings velocity (15%)
        velocity = metrics.get('earnings_velocity', 0)
        velocity_score = 50 + np.clip(velocity / 10, -20, 20)  # Cap at ±20 points
        score += velocity_score * 0.15
        
        # Surface specialization (10%)
        best_surface_eps = max(
            metrics.get('dirt_eps', 0),
            metrics.get('turf_eps', 0),
            metrics.get('distance_eps', 0)
        )
        if best_surface_eps > 0:
            surface_score = min(100, (best_surface_eps / 1000) * 10)  # Scale to 0-100
        else:
            surface_score = 50
        score += surface_score * 0.1
        
        return np.clip(score, 0, 100)
    
    def build_race_classification_hierarchy(self) -> pd.DataFrame:
        """
        Build proper race classification hierarchy
        
        Returns:
            DataFrame with classification analysis
        """
        logger.info("Building race classification hierarchy...")
        
        classification_data = []
        
        for idx, horse in self.current_df.iterrows():
            class_info = {
                'race': horse['race'],
                'horse_name': horse['horse_name'],
                'current_race_type': horse.get('race_type', 'Unknown'),
                'current_classification': horse.get('today_s_race_classification', ''),
                'current_purse': horse.get('purse', 0)
            }
            
            # Parse current race level
            current_level = self._parse_race_level(
                class_info['current_race_type'],
                class_info['current_classification'],
                class_info['current_purse']
            )
            class_info.update(current_level)
            
            # Analyze past race levels
            horse_past = self.past_df[
                (self.past_df['race'] == horse['race']) & 
                (self.past_df['horse_name'] == horse['horse_name'])
            ].head(10)
            
            if len(horse_past) > 0:
                past_levels = []
                class_movements = []
                
                for idx_past, past_race in horse_past.iterrows():
                    past_level = self._parse_race_level(
                        past_race.get('pp_race_type', ''),
                        past_race.get('pp_race_classification', ''),
                        past_race.get('pp_purse', 0)
                    )
                    past_levels.append(past_level['numeric_class_level'])
                
                # Calculate class metrics
                if past_levels:
                    class_info['avg_past_class_level'] = np.mean(past_levels)
                    class_info['highest_class_level'] = max(past_levels)
                    class_info['lowest_class_level'] = min(past_levels)
                    class_info['class_volatility'] = np.std(past_levels)
                    
                    # Class movement from last race
                    if len(past_levels) >= 1:
                        last_level = past_levels[0]
                        class_move = current_level['numeric_class_level'] - last_level
                        class_info['class_move_magnitude'] = abs(class_move)
                        
                        if class_move > 0.5:
                            class_info['class_move_direction'] = 'Up'
                        elif class_move < -0.5:
                            class_info['class_move_direction'] = 'Down'
                        else:
                            class_info['class_move_direction'] = 'Lateral'
                    
                    # Success at different levels
                    class_info['class_success_pattern'] = self._analyze_class_success(
                        horse_past, past_levels
                    )
            else:
                # First time starter or limited data
                class_info['avg_past_class_level'] = current_level['numeric_class_level']
                class_info['class_volatility'] = 0
                class_info['class_move_direction'] = 'Unknown'
                class_info['class_success_pattern'] = 'Unproven'
            
            # Class suitability score
            class_info['class_suitability_score'] = self._calculate_class_suitability(
                class_info, current_level['numeric_class_level']
            )
            
            classification_data.append(class_info)
        
        return pd.DataFrame(classification_data)
    
    def _parse_race_level(self, race_type: str, classification: str, purse: float) -> Dict:
        """Parse race type and classification into numeric level"""
        result = {
            'race_type_code': race_type,
            'numeric_class_level': 5,  # Default middle level
            'class_category': 'Unknown'
        }
        
        # Get base level from race type
        if pd.notna(race_type) and race_type in self.RACE_TYPE_HIERARCHY:
            result['numeric_class_level'] = self.RACE_TYPE_HIERARCHY[race_type]
            
            # Categorize
            if race_type in ['G1', 'G2', 'G3']:
                result['class_category'] = 'Stakes'
            elif race_type in ['N', 'NO']:
                result['class_category'] = 'Stakes'
            elif race_type in ['A', 'AO', 'R', 'T']:
                result['class_category'] = 'Allowance'
            elif race_type in ['C', 'CO']:
                result['class_category'] = 'Claiming'
            elif race_type in ['S', 'M', 'MO']:
                result['class_category'] = 'Maiden'
        
        # Refine with classification string
        if pd.notna(classification) and isinstance(classification, str):
            class_lower = classification.lower()
            
            # Extract claiming price
            claiming_match = re.search(self.CLASSIFICATION_PATTERNS['claiming'], class_lower)
            if claiming_match:
                claiming_price = float(claiming_match.group(1))
                if claiming_price < 1000:  # Likely in thousands
                    claiming_price *= 1000
                
                # Adjust level based on claiming price
                if claiming_price >= 50000:
                    result['numeric_class_level'] += 0.5
                elif claiming_price <= 10000:
                    result['numeric_class_level'] -= 0.5
                
                result['claiming_price'] = claiming_price
            
            # Check for conditions
            if 'n1x' in class_lower or 'n2l' in class_lower:
                result['restricted_conditions'] = True
                result['numeric_class_level'] -= 0.2  # Slightly easier
            elif 'n3l' in class_lower or 'n4l' in class_lower:
                result['restricted_conditions'] = True
                result['numeric_class_level'] += 0.2  # Slightly harder
        
        # Adjust for purse size
        if pd.notna(purse) and purse > 0:
            if result['class_category'] == 'Stakes':
                if purse >= 500000:
                    result['numeric_class_level'] += 0.3
                elif purse >= 200000:
                    result['numeric_class_level'] += 0.1
            elif result['class_category'] == 'Allowance':
                if purse >= 100000:
                    result['numeric_class_level'] += 0.2
                elif purse <= 30000:
                    result['numeric_class_level'] -= 0.2
        
        return result
    
    def _analyze_class_success(self, past_races: pd.DataFrame, 
                              class_levels: List[float]) -> str:
        """Analyze success patterns at different class levels"""
        if len(past_races) < 3:
            return 'Insufficient_data'
        
        # Group performances by class level
        high_class = []
        mid_class = []
        low_class = []
        
        median_level = np.median(class_levels)
        
        for (level, (idx, race)) in zip(class_levels, past_races.iterrows()):
            finish = race.get('pp_finish_pos', 99)
            if level > median_level + 0.5:
                high_class.append(finish)
            elif level < median_level - 0.5:
                low_class.append(finish)
            else:
                mid_class.append(finish)
        
        # Analyze patterns
        patterns = []
        
        if high_class:
            avg_high = np.mean([f for f in high_class if f < 99])
            if avg_high <= 3:
                patterns.append('Handles_class_well')
            elif avg_high >= 6:
                patterns.append('Struggles_up_in_class')
        
        if low_class:
            avg_low = np.mean([f for f in low_class if f < 99])
            if avg_low <= 2:
                patterns.append('Dominates_when_dropped')
        
        if patterns:
            return '_'.join(patterns)
        else:
            return 'Mixed_results'
    
    def _calculate_class_suitability(self, class_info: Dict, 
                                   current_level: float) -> float:
        """Calculate how suitable current class level is"""
        score = 50  # Base score
        
        # Compare to average past level
        avg_level = class_info.get('avg_past_class_level', current_level)
        level_diff = current_level - avg_level
        
        if abs(level_diff) < 0.5:
            score += 10  # At appropriate level
        elif level_diff > 1:
            score -= 20  # Significant class rise
        elif level_diff < -1:
            score += 15  # Significant class drop
        
        # Class volatility penalty
        volatility = class_info.get('class_volatility', 0)
        if volatility > 2:
            score -= 10  # Inconsistent class levels
        elif volatility < 0.5:
            score += 10  # Consistent class placement
        
        # Success pattern bonus
        pattern = class_info.get('class_success_pattern', '')
        if 'Handles_class_well' in pattern:
            score += 15
        elif 'Dominates_when_dropped' in pattern and class_info.get('class_move_direction') == 'Down':
            score += 20
        elif 'Struggles_up_in_class' in pattern and class_info.get('class_move_direction') == 'Up':
            score -= 15
        
        return np.clip(score, 0, 100)
    
    def calculate_hidden_class_indicators(self) -> pd.DataFrame:
        """
        Calculate hidden class indicators beyond purse values
        
        Returns:
            DataFrame with hidden class indicators
        """
        logger.info("Calculating hidden class indicators...")
        
        hidden_indicators = []
        
        for idx, horse in self.current_df.iterrows():
            indicators = {
                'race': horse['race'],
                'horse_name': horse['horse_name']
            }
            
            # 1. Breeding class indicators
            sire_fee = horse.get('sire_stud_fee_current', 0)
            if pd.notna(sire_fee) and sire_fee > 0:
                # Log scale for stud fees
                indicators['sire_fee_class'] = np.log10(sire_fee + 1) * 10
            else:
                indicators['sire_fee_class'] = 30  # Default middle value
            
            # 2. Auction indicators
            auction_price = horse.get('auction_price', 0)
            if pd.notna(auction_price) and auction_price > 0:
                indicators['auction_price_class'] = np.log10(auction_price + 1) * 10
                
                # Depreciation analysis (value retained)
                horse_age = datetime.now().year - horse.get('year_of_birth', datetime.now().year - 3)
                if horse_age > 0:
                    annual_depreciation = (1 - 0.2) ** horse_age  # 20% annual depreciation
                    current_value = auction_price * annual_depreciation
                    indicators['estimated_current_value'] = current_value
                else:
                    indicators['estimated_current_value'] = auction_price
            else:
                indicators['auction_price_class'] = 30
                indicators['estimated_current_value'] = 50000  # Default
            
            # 3. Pedigree ratings
            pedigree_scores = {
                'dirt': horse.get('bris_dirt_pedigree_rating', ''),
                'turf': horse.get('bris_turf_pedigree_rating', ''),
                'distance': horse.get('bris_dist_pedigree_rating', ''),
                'mud': horse.get('bris_mud_pedigree_rating', '')
            }
            
            # Parse pedigree ratings (format: "115*")
            pedigree_values = []
            for surface, rating in pedigree_scores.items():
                if pd.notna(rating) and isinstance(rating, str):
                    # Extract numeric part
                    numeric_match = re.search(r'(\d+)', str(rating))
                    if numeric_match:
                        value = int(numeric_match.group(1))
                        pedigree_values.append(value)
                        indicators[f'pedigree_{surface}_rating'] = value
            
            if pedigree_values:
                indicators['avg_pedigree_rating'] = np.mean(pedigree_values)
                indicators['best_pedigree_rating'] = max(pedigree_values)
            else:
                indicators['avg_pedigree_rating'] = 100
                indicators['best_pedigree_rating'] = 100
            
            # 4. Previous race quality (beaten favorites, stakes horses)
            horse_past = self.past_df[
                (self.past_df['race'] == horse['race']) & 
                (self.past_df['horse_name'] == horse['horse_name'])
            ].head(10)
            
            quality_points = 0
            if len(horse_past) > 0:
                for idx_past, past in horse_past.iterrows():
                    # Check if raced in stakes
                    if past.get('pp_race_type') in ['G1', 'G2', 'G3', 'N']:
                        quality_points += 10
                    
                    # Check finish position in quality races
                    if past.get('pp_purse', 0) > 100000 and past.get('pp_finish_pos', 99) <= 3:
                        quality_points += 5
                    
                    # Check if beat the favorite
                    if past.get('pp_favorite_indicator') == 0 and past.get('pp_finish_pos') == 1:
                        quality_points += 3
            
            indicators['quality_competition_score'] = min(100, quality_points * 2)
            
            # 5. Trainer/Owner quality indicators
            # Use trainer stats as proxy for stable quality
            trainer_win_pct = horse.get('win_1', 0)  # Trainer win % from key stats
            if pd.notna(trainer_win_pct) and trainer_win_pct > 0:
                indicators['trainer_quality_score'] = min(100, trainer_win_pct * 5)
            else:
                indicators['trainer_quality_score'] = 50
            
            # 6. Calculate composite hidden class score
            indicators['hidden_class_score'] = self._calculate_hidden_class_score(indicators)
            
            hidden_indicators.append(indicators)
        
        return pd.DataFrame(hidden_indicators)
    
    def _calculate_hidden_class_score(self, indicators: Dict) -> float:
        """Calculate composite hidden class score"""
        components = []
        weights = []
        
        # Sire fee (25% weight)
        if 'sire_fee_class' in indicators:
            components.append(indicators['sire_fee_class'])
            weights.append(0.25)
        
        # Auction/current value (20% weight)
        if 'auction_price_class' in indicators:
            components.append(indicators['auction_price_class'])
            weights.append(0.20)
        
        # Pedigree ratings (25% weight)
        if 'best_pedigree_rating' in indicators:
            # Normalize to 0-100 scale
            pedigree_score = (indicators['best_pedigree_rating'] / 130) * 100
            components.append(pedigree_score)
            weights.append(0.25)
        
        # Competition quality (20% weight)
        if 'quality_competition_score' in indicators:
            components.append(indicators['quality_competition_score'])
            weights.append(0.20)
        
        # Trainer quality (10% weight)
        if 'trainer_quality_score' in indicators:
            components.append(indicators['trainer_quality_score'])
            weights.append(0.10)
        
        if components:
            # Weighted average
            total_weight = sum(weights)
            weighted_sum = sum(c * w for c, w in zip(components, weights))
            score = weighted_sum / total_weight
        else:
            score = 50
        
        return np.clip(score, 0, 100)
    
    def generate_comprehensive_class_report(self) -> pd.DataFrame:
        """
        Generate comprehensive class assessment report
        
        Returns:
            DataFrame with complete class analysis
        """
        logger.info("Generating comprehensive class report...")
        
        # Calculate all components
        earnings_metrics = self.calculate_earnings_based_metrics()
        classification = self.build_race_classification_hierarchy()
        hidden_indicators = self.calculate_hidden_class_indicators()
        
        # Start with base horse info
        class_report = self.current_df[[
            'race', 'horse_name', 'post_position', 'today_s_trainer',
            'morn_line_odds_if_available', 'purse'
        ]].copy()
        
        # Merge all components
        for df in [earnings_metrics, classification, hidden_indicators]:
            if not df.empty:
                merge_cols = [col for col in df.columns if col not in ['race', 'horse_name']]
                class_report = class_report.merge(
                    df[['race', 'horse_name'] + merge_cols],
                    on=['race', 'horse_name'],
                    how='left'
                )
        
        # Calculate overall class rating
        class_report['overall_class_rating'] = class_report.apply(
            self._calculate_overall_class_rating, axis=1
        )
        
        # Add class categories
        class_report['class_category'] = pd.cut(
            class_report['overall_class_rating'],
            bins=[0, 40, 60, 75, 90, 100],
            labels=['Low', 'Claiming', 'Allowance', 'Stakes', 'Elite']
        )
        
        # Rank within race
        class_report['class_rank'] = class_report.groupby('race')['overall_class_rating'].rank(
            ascending=False, method='min'
        )
        
        # Identify class standouts
        class_report['class_edge'] = class_report.apply(
            self._identify_class_edge, axis=1
        )
        
        return class_report
    
    def _calculate_overall_class_rating(self, row) -> float:
        """Calculate comprehensive class rating"""
        components = []
        weights = []
        
        # Earnings class (45% - increased from 30%)
        if pd.notna(row.get('earnings_class_score')):
            components.append(row['earnings_class_score'])
            weights.append(0.45)
        
        # Classification suitability (25% - increased from 20%)
        if pd.notna(row.get('class_suitability_score')):
            components.append(row['class_suitability_score'])
            weights.append(0.25)
        
        # Hidden class indicators (30% - increased from 25%)
        if pd.notna(row.get('hidden_class_score')):
            components.append(row['hidden_class_score'])
            weights.append(0.30)
        
        if components:
            total_weight = sum(weights)
            weighted_sum = sum(c * w for c, w in zip(components, weights))
            rating = weighted_sum / total_weight
        else:
            rating = 50
        
        return np.clip(rating, 0, 100)
    
    def _identify_class_edge(self, row) -> str:
        """Identify specific class advantages"""
        edges = []
        
        # Dropping in class
        if row.get('class_move_direction') == 'Down' and row.get('class_move_magnitude', 0) > 1:
            edges.append('Significant_drop')
        
        # Hidden class
        if row.get('hidden_class_score', 0) > row.get('earnings_class_score', 0) + 15:
            edges.append('Hidden_quality')
        
        # Proven at level
        if row.get('class_suitability_score', 0) > 80:
            edges.append('Proven_class')
        
        # Rising star
        if row.get('earnings_velocity', 0) > 50:
            edges.append('Rising_star')
        
        # Faces easier
        if row.get('numeric_class_level', 5) < row.get('avg_past_class_level', 5) - 1:
            edges.append('Faces_easier')
        
        return ', '.join(edges) if edges else 'None'


def main():
    """Main function to run class assessment"""
    
    # Set up paths
    base_path = PROCESSED_DATA_DIR
    current_race_path = CURRENT_RACE_INFO
    past_starts_path = PAST_STARTS_LONG
    
    # Check if files exist
    for path in [current_race_path, past_starts_path]:
        if not path.exists():
            logger.error(f"File not found: {path}")
            return
    
    # Initialize analyzer
    analyzer = MultiDimensionalClassAssessment(
        str(current_race_path),
        str(past_starts_path)
    )
    
    # Generate comprehensive report
    class_report = analyzer.generate_comprehensive_class_report()
    
    # Save main report
    output_path = base_path / 'multi_dimensional_class_assessment.parquet'
    class_report.to_parquet(output_path, index=False)
    logger.info(f"Saved class assessment to {output_path}")
    
    # Generate summary reports
    
    # 1. Class movements summary
    movements = class_report[['race', 'horse_name', 'class_move_direction', 
                             'class_move_magnitude', 'class_suitability_score']]
    movements_path = base_path / 'class_movements_summary.csv'
    movements.to_csv(movements_path, index=False)
    
    # 2. Hidden class indicators
    hidden = class_report[['race', 'horse_name', 'earnings_class_score',
                          'hidden_class_score', 'overall_class_rating']]
    hidden['hidden_vs_earnings_diff'] = hidden['hidden_class_score'] - hidden['earnings_class_score']
    hidden_sorted = hidden.sort_values('hidden_vs_earnings_diff', ascending=False)
    hidden_path = base_path / 'hidden_class_indicators.csv'
    hidden_sorted.to_csv(hidden_path, index=False)
    
    # Display summary
    print("\n" + "="*70)
    print("CLASS ASSESSMENT SUMMARY")
    print("="*70)
    
    print(f"\nTotal horses analyzed: {len(class_report)}")
    
    print("\nCLASS CATEGORIES:")
    if 'class_category' in class_report.columns:
        print(class_report['class_category'].value_counts().sort_index())
    
    print("\nCLASS MOVEMENTS:")
    if 'class_move_direction' in class_report.columns:
        print(class_report['class_move_direction'].value_counts())
    
    print("\nTOP 10 HORSES BY OVERALL CLASS:")
    top_class = class_report.nlargest(10, 'overall_class_rating')[
        ['race', 'horse_name', 'overall_class_rating', 'class_category',
         'class_edge', 'earnings_class_score', 'hidden_class_score']
    ]
    print(top_class.to_string(index=False))
    
    print("\nHIDDEN CLASS STANDOUTS (Hidden > Earnings):")
    hidden_standouts = class_report[
        class_report['hidden_class_score'] > class_report['earnings_class_score'] + 10
    ].nlargest(10, 'hidden_class_score')[
        ['race', 'horse_name', 'hidden_class_score', 'earnings_class_score',
         'overall_class_rating']
    ]
    if len(hidden_standouts) > 0:
        print(hidden_standouts.to_string(index=False))
    
    print("\nCLASS DROPPERS:")
    droppers = class_report[
        class_report['class_move_direction'] == 'Down'
    ].nlargest(10, 'class_move_magnitude')[
        ['race', 'horse_name', 'class_move_magnitude', 'class_suitability_score',
         'overall_class_rating']
    ]
    if len(droppers) > 0:
        print(droppers.to_string(index=False))
    
    # Save detailed summary
    summary_path = base_path / 'class_assessment_summary.csv'
    summary_cols = [
        'race', 'horse_name', 'post_position', 'overall_class_rating',
        'class_rank', 'class_category', 'class_edge', 'earnings_class_score',
        'hidden_class_score', 'competitive_level_index', 'class_move_direction'
    ]
    
    available_cols = [col for col in summary_cols if col in class_report.columns]
    summary_df = class_report[available_cols].sort_values(['race', 'class_rank'])
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved class summary to {summary_path}")


if __name__ == '__main__':
    main()