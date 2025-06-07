#!/usr/bin/env python
"""
Simple Pace Analysis and Visualization
Analyzes pace metrics from past performances and visualizes in Plotly Dash
"""

import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
from datetime import datetime
import logging
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings

from config.settings import PROCESSED_DATA_DIR, PAST_STARTS_LONG, CURRENT_RACE_INFO

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# File paths from centralized settings
PAST_STARTS_FILE = PAST_STARTS_LONG
CURRENT_RACE_FILE = CURRENT_RACE_INFO

# Column mappings
PAST_COLUMNS = {
    'e1': 'pp_e1_pace',
    'e2': 'pp_e2_pace',
    'turn_time': 'pp_turn_time',
    'late_pace': 'pp_bris_late_pace',
    'race_date': 'pp_race_date',
    'shape_1c': 'pp_bris_shape_1st_call',
    'shape_2c': 'pp_bris_shape_2nd_call',
    'distance': 'pp_distance',
    'surface': 'pp_surface'
}

CURRENT_COLUMNS = {
    'race': 'race',
    'program_number': 'program_number_if_available',
    'post_position': 'post_position',
    'horse_name': 'horse_name',
    'ml_odds': 'morn_line_odds_if_available',
    'run_style': 'bris_run_style_designation',
    'speed_points': 'quirin_style_speed_points'
}


class ClusteringBestRaceAnalyzer:
    """
    Uses clustering to identify best races and analyze patterns
    """
    
    def __init__(self):
        self.factors_to_analyze = [
            'pp_days_since_prev',
            'pp_track_code',
            'pp_track_condition',
            'pp_distance',
            'pp_surface',
            'pp_num_entrants',
            'pp_post_position',
            'pp_purse',
            'pp_first_call_pos',
            'pp_second_call_pos',
            'pp_first_call_lengths_leader',
            'pp_first_call_lengths_behind',
            'pp_second_call_lengths_leader',
            'pp_second_call_lengths_behind',
            'pp_bris_shape_1st_call',
            'pp_bris_shape_2nd_call',
            'pp_distance_type'
        ]
    
    def identify_best_races(self, speed_ratings):
        """
        Uses clustering to find best races group
        
        Returns: boolean mask of best races
        """
        if len(speed_ratings) < 3:
            # Too few races - just take top 30%
            threshold = np.percentile(speed_ratings, 70)
            return speed_ratings >= threshold
        
        # Reshape for clustering
        X = speed_ratings.reshape(-1, 1)
        
        # Try DBSCAN first - better for finding natural groups
        # Adaptive epsilon based on rating spread
        eps = np.std(speed_ratings) * 0.3
        db = DBSCAN(eps=eps, min_samples=2).fit(X)
        
        if len(set(db.labels_)) > 1 and -1 not in db.labels_:
            # Found clear clusters without noise
            cluster_means = []
            for label in set(db.labels_):
                cluster_ratings = speed_ratings[db.labels_ == label]
                cluster_means.append((label, np.mean(cluster_ratings)))
            
            # Get the best cluster
            best_cluster = max(cluster_means, key=lambda x: x[1])[0]
            best_mask = db.labels_ == best_cluster
        else:
            # DBSCAN didn't find good clusters, try K-means
            # Determine optimal number of clusters (2 or 3)
            if len(speed_ratings) >= 6:
                # Try both 2 and 3 clusters, pick better separation
                inertias = []
                for k in [2, 3]:
                    km = KMeans(n_clusters=k, random_state=42).fit(X)
                    inertias.append(km.inertia_)
                
                # Use elbow method - if 3 clusters significantly better, use 3
                if inertias[1] < inertias[0] * 0.6:
                    n_clusters = 3
                else:
                    n_clusters = 2
            else:
                n_clusters = 2
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
            
            # Find cluster with highest mean
            cluster_means = []
            for i in range(n_clusters):
                cluster_ratings = speed_ratings[kmeans.labels_ == i]
                cluster_means.append((i, np.mean(cluster_ratings)))
            
            best_cluster = max(cluster_means, key=lambda x: x[1])[0]
            best_mask = kmeans.labels_ == best_cluster
        
        # Sanity check - ensure we're not selecting too many or too few
        pct_selected = np.sum(best_mask) / len(speed_ratings)
        if pct_selected > 0.5:  # More than 50% selected
            # Be more selective
            threshold = np.percentile(speed_ratings, 75)
            best_mask = speed_ratings >= threshold
        elif pct_selected < 0.1:  # Less than 10% selected
            # Be less selective
            threshold = np.percentile(speed_ratings, 70)
            best_mask = speed_ratings >= threshold
        
        return best_mask
    
    def analyze_best_race_patterns(self, past_starts_df, horse_name, speed_column='pp_speed_rating'):
        """
        Analyzes patterns in a horse's best races
        
        Returns detailed analysis of factors in best vs regular races
        """
        # Filter for specific horse
        horse_data = past_starts_df[
            past_starts_df['horse_name'] == horse_name
        ].copy()
        
        if len(horse_data) == 0 or speed_column not in horse_data.columns:
            return None
        
        # Sort by date
        if 'pp_race_date' in horse_data.columns:
            horse_data = horse_data.sort_values('pp_race_date')
        
        # Get speed ratings and identify best races
        valid_mask = horse_data[speed_column].notna()
        ratings = horse_data.loc[valid_mask, speed_column].values
        
        if len(ratings) < 2:
            return None
        
        # Identify best races using clustering
        best_races_mask = self.identify_best_races(ratings)
        
        # Add best race flag to horse data
        horse_data.loc[valid_mask, 'is_best_race'] = best_races_mask
        
        # Analyze patterns
        best_races = horse_data[horse_data['is_best_race'] == True]
        regular_races = horse_data[horse_data['is_best_race'] == False]
        
        patterns = {
            'horse_name': horse_name,
            'total_races': len(horse_data),
            'best_races_count': len(best_races),
            'best_avg_speed': best_races[speed_column].mean() if len(best_races) > 0 else np.nan,
            'regular_avg_speed': regular_races[speed_column].mean() if len(regular_races) > 0 else np.nan,
            'factors': {}
        }
        
        # Analyze each factor
        for factor in self.factors_to_analyze:
            if factor not in horse_data.columns:
                continue
            
            factor_analysis = {}
            
            # Numeric factors
            if pd.api.types.is_numeric_dtype(horse_data[factor]):
                if len(best_races) > 0:
                    best_values = best_races[factor].dropna()
                    regular_values = regular_races[factor].dropna() if len(regular_races) > 0 else pd.Series()
                    
                    if len(best_values) > 0:
                        factor_analysis['best_mean'] = best_values.mean()
                        factor_analysis['best_std'] = best_values.std()
                        factor_analysis['best_median'] = best_values.median()
                        
                        if len(regular_values) > 0:
                            factor_analysis['regular_mean'] = regular_values.mean()
                            factor_analysis['difference'] = factor_analysis['best_mean'] - factor_analysis['regular_mean']
                            factor_analysis['pct_change'] = (
                                (factor_analysis['best_mean'] - factor_analysis['regular_mean']) / 
                                factor_analysis['regular_mean'] * 100
                                if factor_analysis['regular_mean'] != 0 else np.nan
                            )
                            
                            # Statistical significance test
                            if len(best_values) >= 3 and len(regular_values) >= 3:
                                if best_values.std() == 0 and regular_values.std() == 0:
                                    p_value = 1.0
                                else:
                                    with warnings.catch_warnings():
                                        warnings.simplefilter("ignore", RuntimeWarning)
                                        _, p_value = stats.ttest_ind(
                                            best_values,
                                            regular_values,
                                            equal_var=False,
                                            nan_policy="omit",
                                        )

                                factor_analysis['significant'] = p_value < 0.05
                                factor_analysis['p_value'] = p_value
            
            # Categorical factors
            else:
                if len(best_races) > 0:
                    best_dist = best_races[factor].value_counts(normalize=True).to_dict()
                    regular_dist = regular_races[factor].value_counts(normalize=True).to_dict() if len(regular_races) > 0 else {}
                    
                    factor_analysis['best_distribution'] = best_dist
                    factor_analysis['regular_distribution'] = regular_dist
                    factor_analysis['best_mode'] = best_races[factor].mode()[0] if len(best_races[factor].mode()) > 0 else None
                    
                    # Find categories that appear more in best races
                    overrepresented = {}
                    for category in best_dist:
                        best_pct = best_dist.get(category, 0)
                        regular_pct = regular_dist.get(category, 0)
                        if best_pct > regular_pct * 1.2:  # 20% more common
                            overrepresented[category] = {
                                'best_pct': best_pct,
                                'regular_pct': regular_pct,
                                'ratio': best_pct / regular_pct if regular_pct > 0 else np.inf
                            }
                    
                    if overrepresented:
                        factor_analysis['overrepresented_in_best'] = overrepresented
            
            if factor_analysis:
                patterns['factors'][factor] = factor_analysis
        
        # Add best races data for reference
        patterns['best_races_data'] = best_races
        patterns['clustering_details'] = {
            'method': 'DBSCAN/KMeans',
            'num_clusters_found': len(set(best_races_mask)) if len(best_races_mask) > 0 else 0
        }
        
        return horse_data


class PaceAnalyzer:
    """Main class for pace analysis calculations"""
    
    def __init__(self):
        self.past_starts_df = None
        self.current_race_df = None
        self.pace_results = None
        self.best_race_analyzer = ClusteringBestRaceAnalyzer()
        self.best_race_report = None
        
    def load_data(self):
        """Load data from parquet files"""
        try:
            logger.info("Loading past starts data...")
            self.past_starts_df = pd.read_parquet(PAST_STARTS_FILE)
            
            logger.info("Loading current race data...")
            self.current_race_df = pd.read_parquet(CURRENT_RACE_FILE)
            
            logger.info(f"Loaded {len(self.past_starts_df)} past performance records")
            logger.info(f"Loaded {len(self.current_race_df)} current race entries")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def validate_columns(self):
        """Validate that required columns exist"""
        missing_past = []
        missing_current = []
        
        # Check past starts columns
        for key, col in PAST_COLUMNS.items():
            if col not in self.past_starts_df.columns:
                missing_past.append(col)
                
        # Check current race columns
        for key, col in CURRENT_COLUMNS.items():
            if col not in self.current_race_df.columns:
                missing_current.append(col)
                
        if missing_past:
            logger.warning(f"Missing columns in past_starts: {missing_past}")
        if missing_current:
            logger.warning(f"Missing columns in current_race: {missing_current}")
            
        return len(missing_past) == 0 and len(missing_current) == 0
    
    def calculate_pace_metrics(self):
        """
        Calculate all pace metrics for current race entries
        """
        logger.info("Calculating pace metrics...")
        
        # Start with current race data
        results = self.current_race_df.copy()
        
        # Log unique races
        unique_races = results['race'].unique()
        logger.info(f"Calculating metrics for races: {unique_races}")
        
        # Get today's distance and surface for filtering
        # Assuming these are consistent within a race
        if 'distance_in_yards' in results.columns:
            today_distance = results['distance_in_yards'].iloc[0]
        else:
            today_distance = None
            
        if 'surface' in results.columns:
            today_surface = results['surface'].iloc[0]
        else:
            today_surface = None
        
        # Initialize new columns
        results['e1_consistency'] = np.nan
        results['lp_consistency'] = np.nan
        results['average_e1'] = np.nan
        results['first_call_mean'] = np.nan
        results['average_lp'] = np.nan
        
        # Calculate metrics for each horse
        for idx, row in results.iterrows():
            race_num = row['race']
            horse_name = row['horse_name']
            
            # Get past performance data for this horse
            horse_pp = self.past_starts_df[
                (self.past_starts_df['race'] == race_num) & 
                (self.past_starts_df['horse_name'] == horse_name)
            ].copy()
            
            if len(horse_pp) > 0:
                # Sort by race date (most recent first)
                if 'pp_race_date' in horse_pp.columns:
                    horse_pp = horse_pp.sort_values('pp_race_date', ascending=False)
                
                # E1 Consistency - SD of last 5 pp_e1_pace
                if 'pp_e1_pace' in horse_pp.columns:
                    last_5_e1 = horse_pp['pp_e1_pace'].dropna().head(5)
                    if len(last_5_e1) >= 2:  # Need at least 2 for SD
                        results.at[idx, 'e1_consistency'] = last_5_e1.std()
                
                # LP Consistency - SD of last 5 pp_bris_late_pace
                if 'pp_bris_late_pace' in horse_pp.columns:
                    last_5_lp = horse_pp['pp_bris_late_pace'].dropna().head(5)
                    if len(last_5_lp) >= 2:
                        results.at[idx, 'lp_consistency'] = last_5_lp.std()
                
                # Filter for same distance and surface
                if today_distance is not None and today_surface is not None:
                    same_conditions = horse_pp[
                        (horse_pp['pp_distance'] == today_distance) & 
                        (horse_pp['pp_surface'] == today_surface)
                    ]
                    
                    # Average E1 for same distance/surface
                    if len(same_conditions) > 0 and 'pp_e1_pace' in same_conditions.columns:
                        results.at[idx, 'average_e1'] = same_conditions['pp_e1_pace'].dropna().mean()
                    if len(same_conditions) > 0 and 'pp_bris_late_pace' in same_conditions.columns:
                        results.at[idx, 'average_lp'] = same_conditions['pp_bris_late_pace'].dropna().mean()
                
                # First Call Mean - average of 3 highest pp_e1_pace
                if 'pp_e1_pace' in horse_pp.columns:
                    top_3_e1 = horse_pp['pp_e1_pace'].dropna().nlargest(3)
                    if len(top_3_e1) > 0:
                        results.at[idx, 'first_call_mean'] = top_3_e1.mean()
                
                # Add energy distribution metrics to past performances
                if all(col in horse_pp.columns for col in ['pp_e1_pace', 'pp_e2_pace', 'pp_bris_late_pace']):
                    # Only calculate where all values are non-null
                    valid_rows = horse_pp[['pp_e1_pace', 'pp_e2_pace', 'pp_bris_late_pace']].notna().all(axis=1)
                    
                    if valid_rows.any():
                        horse_pp.loc[valid_rows, 'energy_distribution'] = (
                            (horse_pp.loc[valid_rows, 'pp_e1_pace'] + horse_pp.loc[valid_rows, 'pp_e2_pace']) / 
                            (2 * horse_pp.loc[valid_rows, 'pp_bris_late_pace'])
                        )
                        horse_pp.loc[valid_rows, 'accel_pct'] = (
                            (horse_pp.loc[valid_rows, 'pp_e2_pace'] - horse_pp.loc[valid_rows, 'pp_e1_pace']) / 
                            horse_pp.loc[valid_rows, 'pp_e1_pace'] * 100
                        )
                        horse_pp.loc[valid_rows, 'finish_pct'] = (
                            (horse_pp.loc[valid_rows, 'pp_bris_late_pace'] - horse_pp.loc[valid_rows, 'pp_e2_pace']) / 
                            horse_pp.loc[valid_rows, 'pp_e2_pace'] * 100
                        )
                        
                        # Store latest values
                        latest_valid = horse_pp[valid_rows].iloc[0]
                        results.at[idx, 'latest_energy_dist'] = latest_valid['energy_distribution']
                        results.at[idx, 'latest_accel_pct'] = latest_valid['accel_pct']
                        results.at[idx, 'latest_finish_pct'] = latest_valid['finish_pct']
        
        # Calculate race-level metrics PER RACE
        for race_num in results['race'].unique():
            race_mask = results['race'] == race_num
            
            # Lead-Abundance Flag for this specific race
            if 'bris_run_style_designation' in results.columns and 'quirin_style_speed_points' in results.columns:
                run_styles = results.loc[race_mask, 'bris_run_style_designation']
                run_styles = run_styles.fillna('').astype(str).str.upper()

                sp = pd.to_numeric(
                    results.loc[race_mask, 'quirin_style_speed_points'], errors='coerce'
                ).fillna(0)

                lead_mask = run_styles.isin(['E', 'E/P']) & (sp >= 6)
                lead_horses_in_race = results.loc[race_mask][lead_mask]
                results.loc[race_mask, 'lead_abundance_flag'] = len(lead_horses_in_race)
            else:
                results.loc[race_mask, 'lead_abundance_flag'] = 0
            
            # Pace Vol - SD of all pp_e1_pace for horses in THIS race
            race_horses = results[race_mask]['horse_name'].tolist()
            race_pp = self.past_starts_df[
                (self.past_starts_df['race'] == race_num) & 
                (self.past_starts_df['horse_name'].isin(race_horses))
            ]
            
            if 'pp_e1_pace' in race_pp.columns:
                race_e1_values = race_pp['pp_e1_pace'].dropna()
                if len(race_e1_values) >= 2:
                    results.loc[race_mask, 'pace_vol'] = np.std(race_e1_values)
                else:
                    results.loc[race_mask, 'pace_vol'] = np.nan
            else:
                results.loc[race_mask, 'pace_vol'] = np.nan
        
        # ARFIMA regression (simplified version - trend analysis)
        results['e1_trend'] = np.nan
        results['lp_trend'] = np.nan
        
        for idx, row in results.iterrows():
            race_num = row['race']
            horse_name = row['horse_name']
            
            horse_pp = self.past_starts_df[
                (self.past_starts_df['race'] == race_num) & 
                (self.past_starts_df['horse_name'] == horse_name)
            ]
            
            if 'pp_race_date' in horse_pp.columns:
                horse_pp = horse_pp.sort_values('pp_race_date', ascending=True)
            
            if len(horse_pp) >= 3:
                # Simple linear trend for E1 and LP
                x = np.arange(len(horse_pp))
                
                # E1 trend
                if 'pp_e1_pace' in horse_pp.columns:
                    e1_valid = horse_pp['pp_e1_pace'].notna()
                    if e1_valid.sum() >= 3:
                        e1_coef = np.polyfit(x[e1_valid], horse_pp.loc[e1_valid, 'pp_e1_pace'], 1)[0]
                        results.at[idx, 'e1_trend'] = e1_coef
                
                # LP trend
                if 'pp_bris_late_pace' in horse_pp.columns:
                    lp_valid = horse_pp['pp_bris_late_pace'].notna()
                    if lp_valid.sum() >= 3:
                        lp_coef = np.polyfit(x[lp_valid], horse_pp.loc[lp_valid, 'pp_bris_late_pace'], 1)[0]
                        results.at[idx, 'lp_trend'] = lp_coef
        
        self.pace_results = results
        return results
    
    def analyze_best_races(self):
        """Analyze best race patterns for all horses"""
        logger.info("Analyzing best race patterns...")
        
        # Check if we need to use a different speed column name
        speed_column = 'pp_speed_rating'
        if speed_column not in self.past_starts_df.columns:
            # Try alternative names
            for alt_name in ['pp_bris_speed_rating', 'pp_speed_figure', 'pp_final_time', 'pp_beyer_speed']:
                if alt_name in self.past_starts_df.columns:
                    speed_column = alt_name
                    logger.info(f"Using {alt_name} for speed ratings")
                    break
            else:
                # If no speed rating column found, log available columns
                logger.warning(f"No speed rating column found. Available columns: {self.past_starts_df.columns.tolist()[:20]}...")
                speed_column = None
        
        # Generate report for all horses
        report = {
            'race_reports': {},
            'overall_patterns': None,
            'consistent_factors': []
        }
        
        if speed_column is None:
            logger.error("Cannot perform best race analysis without speed rating column")
            return report
        
        all_patterns = []
        
        # Analyze each race
        for race_num in self.current_race_df['race'].unique():
            race_horses = self.current_race_df[
                self.current_race_df['race'] == race_num
            ]['horse_name'].unique()
            
            race_patterns = []
            
            for horse_name in race_horses:
                pattern = self.best_race_analyzer.analyze_best_race_patterns(
                    self.past_starts_df, 
                    horse_name,
                    speed_column=speed_column
                )
                
                if pattern:
                    race_patterns.append(pattern)
                    all_patterns.append(pattern)
            
            report['race_reports'][race_num] = {
                'horses_analyzed': len(race_patterns),
                'patterns': race_patterns
            }
        
        # Summarize patterns across all horses
        if all_patterns:
            factor_summary = {}
            
            for factor in self.best_race_analyzer.factors_to_analyze:
                improvements = []
                significant_count = 0
                
                for pattern in all_patterns:
                    if factor in pattern['factors']:
                        factor_info = pattern['factors'][factor]
                        if 'difference' in factor_info:
                            improvements.append(factor_info['difference'])
                            if factor_info.get('significant', False):
                                significant_count += 1
                
                if improvements:
                    avg_improvement = np.mean(improvements)
                    positive_pct = sum(1 for x in improvements if x > 0) / len(improvements)
                    
                    # Identify consistent factors
                    if (positive_pct > 0.7 or positive_pct < 0.3) and abs(avg_improvement) > 0.1:
                        report['consistent_factors'].append({
                            'factor': factor,
                            'direction': 'higher' if positive_pct > 0.7 else 'lower',
                            'avg_change': avg_improvement,
                            'consistency': positive_pct,
                            'horses_affected': len(improvements)
                        })
        
        # Sort consistent factors by impact
        report['consistent_factors'].sort(
            key=lambda x: abs(x['avg_change']) * x['consistency'], 
            reverse=True
        )
        
        logger.info(f"Best race analysis complete. Found {len(all_patterns)} horses with patterns")
        logger.info(f"Consistent factors: {len(report['consistent_factors'])}")
        
        self.best_race_report = report
        return report
    
    def get_race_list(self):
        """Get list of unique races"""
        if self.current_race_df is not None:
            races = self.current_race_df['race'].unique()
            logger.info(f"Found races: {races}")
            return sorted(races)
        return []

    def export_best_race_analysis_to_csv(self, output_dir):
        import os
        os.makedirs(output_dir, exist_ok=True)

        speed_column = 'pp_speed_rating'
        if speed_column not in self.past_starts_df.columns:
            for alt_name in ['pp_bris_speed_rating', 'pp_speed_figure', 'pp_final_time', 'pp_beyer_speed']:
                if alt_name in self.past_starts_df.columns:
                    speed_column = alt_name
                    break
            else:
                raise ValueError("No speed rating column found.")

        horses = self.past_starts_df['horse_name'].unique()

        for horse in horses:
            horse_df = self.best_race_analyzer.analyze_best_race_patterns(
                self.past_starts_df,
                horse,
                speed_column=speed_column
            )

            if horse_df is None or horse_df.empty:
                continue

            # Save detailed race data with best race flag
            file_detail = os.path.join(output_dir, f"{horse}_best_races_detail.csv")
            horse_df.to_csv(file_detail, index=False)

            # Save speed ratings and best race mask only
            summary_df = horse_df[[speed_column, 'is_best_race']]
            file_summary = os.path.join(output_dir, f"{horse}_best_race_summary.csv")
            summary_df.to_csv(file_summary, index=False)

def create_horse_best_race_card(pattern):
    """Create a card showing a horse's best race patterns"""
    
    # Find top factors
    top_factors = []
    for factor, analysis in pattern['factors'].items():
        if 'difference' in analysis and abs(analysis['difference']) > 0.1:
            top_factors.append({
                'name': factor.replace('pp_', '').replace('_', ' ').title(),
                'change': analysis['difference'],
                'significant': analysis.get('significant', False)
            })
    
    top_factors.sort(key=lambda x: abs(x['change']), reverse=True)
    
    return dbc.Card([
        dbc.CardHeader(html.H6(pattern['horse_name'], className="mb-0")),
        dbc.CardBody([
            html.P([
                f"Best Races: {pattern['best_races_count']} of {pattern['total_races']} ",
                f"(Avg Speed: {pattern['best_avg_speed']:.0f} vs {pattern['regular_avg_speed']:.0f})"
            ], className="text-muted small"),
            
            html.H6("Key Patterns in Best Races:", className="mt-2"),
            html.Ul([
                html.Li([
                    html.Strong(f"{factor['name']}: "),
                    html.Span(
                        f"{factor['change']:+.1f}",
                        className="text-success" if factor['change'] > 0 else "text-danger"
                    ),
                    html.Span(" *", className="text-warning") if factor['significant'] else ""
                ], className="small")
                for factor in top_factors[:4]  # Top 4 factors
            ], className="mb-0")
        ])
    ], className="mb-2", style={'backgroundColor': 'rgb(40, 40, 40)'})

