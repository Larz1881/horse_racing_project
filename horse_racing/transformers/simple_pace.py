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
                                t_stat, p_value = stats.ttest_ind(best_values, regular_values)
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
        
        return patterns


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
                lead_horses_in_race = results[
                    race_mask &
                    ((results['bris_run_style_designation'] == 'E') | 
                     (results['bris_run_style_designation'] == 'E/P')) & 
                    (results['quirin_style_speed_points'] >= 6)
                ]
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


# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "Simple Pace Analysis"

# Add custom CSS for dropdown visibility
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* Ensure body and html allow scrolling */
            html, body {
                overflow-y: auto !important;
                overflow-x: hidden !important;
                height: auto !important;
                min-height: 100% !important;
                margin: 0;
                padding: 0;
            }
            
            /* Main app container */
            #react-entry-point, #_dash-app-content {
                min-height: 100vh !important;
                height: auto !important;
                overflow: visible !important;
            }
            
            /* Fix dropdown text visibility */
            .dash-dropdown .Select-value {
                color: white !important;
            }
            .dash-dropdown .Select-menu {
                background-color: rgb(50, 50, 50) !important;
            }
            .dash-dropdown .Select-option {
                color: white !important;
                background-color: rgb(50, 50, 50) !important;
            }
            .dash-dropdown .Select-option--is-focused {
                background-color: rgb(70, 70, 70) !important;
            }
            .dash-dropdown .Select-option--is-selected {
                background-color: rgb(80, 80, 80) !important;
            }
            
            /* Ensure buttons have good contrast */
            .btn-success {
                background-color: #28a745 !important;
                border-color: #28a745 !important;
                color: white !important;
            }
            .btn-success:hover {
                background-color: #218838 !important;
                border-color: #1e7e34 !important;
            }
            .btn-primary {
                background-color: #007bff !important;
                border-color: #007bff !important;
                color: white !important;
            }
            .btn-primary:hover {
                background-color: #0056b3 !important;
                border-color: #004085 !important;
            }
            
            /* Ensure plotly graphs don't block scrolling */
            .js-plotly-plot, .plotly {
                position: relative !important;
                overflow: visible !important;
            }
            
            /* Container styles */
            .container-fluid {
                overflow: visible !important;
                min-height: 100vh !important;
                padding-bottom: 100px !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Initialize analyzer
analyzer = PaceAnalyzer()

# Layout with explicit sections
app.layout = html.Div([
    dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H1("Simple Pace Analysis", className="text-center mb-4"),
                html.Hr()
            ], width=12)
        ]),
        
        # Controls
        dbc.Row([
            dbc.Col([
                html.Label("Select Race:", className="fw-bold"),
                dcc.Dropdown(
                    id='race-dropdown',
                    options=[],
                    value=None,
                    className="mb-3",
                    style={
                        'backgroundColor': 'rgb(50, 50, 50)',
                        'color': 'white'
                    }
                )
            ], width=4),
            
            dbc.Col([
                dbc.Button(
                    "Load Data", 
                    id="load-data-btn", 
                    color="primary", 
                    size="lg",
                    className="mt-4",
                    style={
                        'backgroundColor': '#007bff',
                        'borderColor': '#007bff',
                        'color': 'white',
                        'fontWeight': 'bold'
                    }
                )
            ], width=2),
            
            dbc.Col([
                html.Div(id="data-status", className="mt-4")
            ], width=3),
            
            dbc.Col([
                html.A(
                    "Go to Best Race Analysis ↓",
                    href="#best-race-header",
                    className="btn btn-info mt-4",
                    style={'color': 'white', 'textDecoration': 'none'}
                )
            ], width=3)
        ]),
        
        html.Hr(),

        dbc.Tabs([
            dbc.Tab(label="Pace Results", tab_id="results-tab", children=[
                dbc.Row([
                    dbc.Col([
                        html.H3("Pace Analysis Results", className="mb-3"),
                        html.Div(id="pace-table-container")
                    ], width=12)
                ]),

                html.Hr(),

                dbc.Row([
                    dbc.Col([
                        html.H3("Best Race Analysis", className="mb-3", id="best-race-header"),
                        dbc.Button(
                            "Analyze Best Races",
                            id="analyze-best-races-btn",
                            color="success",
                            className="mb-3",
                            style={
                                'backgroundColor': '#28a745',
                                'borderColor': '#28a745',
                                'color': 'white',
                                'fontWeight': 'bold',
                                'padding': '10px 20px',
                                'fontSize': '16px'
                            }
                        ),
                        html.Div(
                            id="best-race-container",
                            children=[
                                html.P(
                                    "Click 'Analyze Best Races' to identify patterns in each horse's best performances.",
                                    className="text-muted"
                                )
                            ],
                            style={'minHeight': '200px', 'paddingBottom': '100px'}
                        )
                    ], width=12)
                ])
            ]),

            dbc.Tab(label="Pace Visualizations", tab_id="viz-tab", children=[
                dbc.Row([
                    dbc.Col([
                        html.H3("Visualizations", className="mb-3"),
                        dcc.Graph(id="pace-visualization")
                    ], width=12)
                ])
            ])
        ], id="main-tabs", active_tab="results-tab"),

        html.Hr(),
        
        # Debug section to test scrolling
        dbc.Row([
            dbc.Col([
                html.Hr(),
                html.H5("Debug: If you can see this, scrolling works!", className="text-warning"),
                html.P("This is a test section at the bottom of the page.", className="text-muted"),
                html.Div(style={'height': '200px', 'backgroundColor': 'rgba(255,255,255,0.1)', 'border': '1px dashed white'})
            ], width=12)
        ])
        
    ], fluid=True)
], style={
    'minHeight': '100vh',
    'paddingBottom': '100px',
    'overflow': 'visible',
    'position': 'relative'
})


# Callbacks
@app.callback(
    [Output('race-dropdown', 'options'),
     Output('race-dropdown', 'value'),
     Output('data-status', 'children')],
    Input('load-data-btn', 'n_clicks'),
    prevent_initial_call=True
)
def load_data(n_clicks):
    """Load data and populate race dropdown"""
    if analyzer.load_data():
        if analyzer.validate_columns():
            races = analyzer.get_race_list()
            logger.info(f"Creating dropdown options for races: {races}")
            options = [{'label': f'Race {r}', 'value': r} for r in races]
            value = races[0] if races else None
            
            status = dbc.Alert(
                f"Data loaded successfully! Found {len(races)} race{'s' if len(races) != 1 else ''}.",
                color="success",
                dismissable=True
            )
            
            logger.info(f"Dropdown options created: {len(options)} options")
            return options, value, status
        else:
            status = dbc.Alert(
                "Data loaded but some columns are missing. Check logs.",
                color="warning",
                dismissable=True
            )
            return [], None, status
    else:
        status = dbc.Alert(
            "Failed to load data. Check file paths.",
            color="danger",
            dismissable=True
        )
        return [], None, status


@app.callback(
    [Output('pace-table-container', 'children'),
     Output('pace-visualization', 'figure')],
    Input('race-dropdown', 'value'),
    prevent_initial_call=True
)
def update_display(selected_race):
    """Update table and visualization based on selected race"""
    if selected_race is None or analyzer.current_race_df is None:
        return html.Div("No race selected"), go.Figure()
    
    # Calculate pace metrics
    results_df = analyzer.calculate_pace_metrics()
    
    # Filter for selected race
    race_df = results_df[results_df['race'] == selected_race].copy()
    
    if race_df.empty:
        return html.Div("No data for selected race"), go.Figure()
    
    # Round numeric columns for display
    numeric_cols = ['e1_consistency', 'lp_consistency', 'average_e1', 'first_call_mean', 
                   'average_lp', 'pace_vol', 'e1_trend', 'lp_trend', 
                   'latest_energy_dist', 'latest_accel_pct', 'latest_finish_pct']
    
    for col in numeric_cols:
        if col in race_df.columns:
            race_df[col] = race_df[col].round(2)
    
    # Create display dataframe with key columns
    display_columns = [
        'post_position', 'horse_name', 'bris_run_style_designation', 
        'quirin_style_speed_points', 'e1_consistency', 'lp_consistency',
        'average_e1', 'first_call_mean', 'average_lp', 'lead_abundance_flag',
        'pace_vol', 'e1_trend', 'lp_trend', 'latest_energy_dist',
        'latest_accel_pct', 'latest_finish_pct'
    ]
    
    # Only include columns that exist
    display_columns = [col for col in display_columns if col in race_df.columns]
    
    display_df = race_df[display_columns].copy()
    
    # Rename columns for better display
    column_rename = {
        'post_position': 'PP',
        'horse_name': 'Horse',
        'bris_run_style_designation': 'Style',
        'quirin_style_speed_points': 'SP',
        'e1_consistency': 'E1 SD',
        'lp_consistency': 'LP SD',
        'average_e1': 'Avg E1',
        'first_call_mean': 'Top3 E1',
        'average_lp': 'Avg LP',
        'lead_abundance_flag': 'Lead Count',
        'pace_vol': 'Pace Vol',
        'e1_trend': 'E1 Trend',
        'lp_trend': 'LP Trend',
        'latest_energy_dist': 'Energy Dist',
        'latest_accel_pct': 'Accel %',
        'latest_finish_pct': 'Finish %'
    }
    
    display_df = display_df.rename(columns=column_rename)
    
    # Create data table
    table = dash_table.DataTable(
        data=display_df.to_dict('records'),
        columns=[{"name": col, "id": col} for col in display_df.columns],
        style_cell={
            'textAlign': 'left',
            'backgroundColor': 'rgb(30, 30, 30)',
            'color': 'white',
            'fontSize': '12px',
            'padding': '5px'
        },
        style_header={
            'backgroundColor': 'rgb(50, 50, 50)',
            'fontWeight': 'bold',
            'fontSize': '13px'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(40, 40, 40)'
            },
            # Highlight lead horses
            {
                'if': {
                    'filter_query': '{Style} = "E" || {Style} = "E/P"',
                    'column_id': 'Style'
                },
                'backgroundColor': 'rgb(70, 30, 30)',
                'color': 'white',
            },
            # Highlight high speed points
            {
                'if': {
                    'filter_query': '{SP} >= 6',
                    'column_id': 'SP'
                },
                'color': 'rgb(255, 200, 100)',
                'fontWeight': 'bold'
            },
            # Highlight improving E1 trend
            {
                'if': {
                    'filter_query': '{E1 Trend} < 0',
                    'column_id': 'E1 Trend'
                },
                'color': 'rgb(100, 255, 100)'
            },
            # Highlight low consistency (good)
            {
                'if': {
                    'filter_query': '{E1 SD} < 2',
                    'column_id': 'E1 SD'
                },
                'color': 'rgb(100, 255, 100)'
            }
        ],
        sort_action="native",
        page_size=20
    )
    
    # Create visualizations
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            'E1 Pace Analysis', 'Late Pace Analysis',
            'Energy Distribution Profile', 'Pace Trends'
        ),
        vertical_spacing=0.1
    )
    
    # Sort by post position for cleaner display
    race_df = race_df.sort_values('post_position')
    
    # 1. E1 Pace Analysis
    fig.add_trace(
        go.Bar(
            x=race_df['post_position'],
            y=race_df['average_e1'],
            name='Avg E1',
            text=race_df['horse_name'],
            textposition='outside',
            marker_color='lightblue'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=race_df['post_position'],
            y=race_df['e1_consistency'],
            name='E1 Consistency',
            mode='lines+markers',
            line=dict(color='red', width=2),
            yaxis='y2'
        ),
        row=1, col=1
    )
    
    # 2. Late Pace Analysis
    fig.add_trace(
        go.Bar(
            x=race_df['post_position'],
            y=race_df['average_lp'],
            name='Avg LP',
            text=race_df['horse_name'],
            textposition='outside',
            marker_color='lightgreen'
        ),
        row=2, col=1
    )
    
    # 3. Energy Distribution
    if 'latest_energy_dist' in race_df.columns:
        # Handle NaN values in marker size
        energy_dist_size = race_df['latest_energy_dist'].fillna(0) * 20
        energy_dist_size = energy_dist_size.clip(lower=5)  # Minimum size of 5
        
        fig.add_trace(
            go.Scatter(
                x=race_df['latest_accel_pct'].fillna(0),
                y=race_df['latest_finish_pct'].fillna(0),
                mode='markers+text',
                text=race_df['horse_name'],
                textposition='top center',
                marker=dict(
                    size=energy_dist_size,
                    color=race_df['quirin_style_speed_points'].fillna(0),
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Speed Points")
                ),
                name='Energy Profile'
            ),
            row=3, col=1
        )
    
    # 4. Pace Trends
    if 'e1_trend' in race_df.columns and 'lp_trend' in race_df.columns:
        # Filter out rows with NaN values in trends
        valid_trends = race_df[['e1_trend', 'lp_trend', 'horse_name', 'post_position']].dropna()
        
        if len(valid_trends) > 0:
            fig.add_trace(
                go.Scatter(
                    x=valid_trends['e1_trend'],
                    y=valid_trends['lp_trend'],
                    mode='markers+text',
                    text=valid_trends['horse_name'],
                    textposition='top center',
                    marker=dict(
                        size=12,
                        color=valid_trends['post_position'],
                        colorscale='Rainbow'
                    ),
                    name='Pace Trends'
                ),
                row=4, col=1
            )
    
    # Update layout
    fig.update_layout(
        title=f"Race {selected_race} - Comprehensive Pace Analysis",
        template="plotly_dark",
        height=1600,
        showlegend=False,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    # Update axes
    fig.update_xaxes(title_text="Post Position", row=1, col=1)
    fig.update_xaxes(title_text="Post Position", row=2, col=1)
    fig.update_xaxes(title_text="Acceleration %", row=3, col=1)
    fig.update_xaxes(title_text="E1 Trend", row=4, col=1)

    fig.update_yaxes(title_text="E1 Pace", row=1, col=1)
    fig.update_yaxes(title_text="Late Pace", row=2, col=1)
    fig.update_yaxes(title_text="Finish %", row=3, col=1)
    fig.update_yaxes(title_text="LP Trend", row=4, col=1)
    
    # Add summary info
    lead_count = race_df['lead_abundance_flag'].iloc[0] if 'lead_abundance_flag' in race_df.columns else 0
    pace_vol = race_df['pace_vol'].iloc[0] if 'pace_vol' in race_df.columns else 'N/A'
    
    summary_div = html.Div([
        html.H5(f"Race Summary:", className="mt-3"),
        html.P(f"Lead Abundance: {lead_count} horses with E/E-P style and 6+ speed points"),
        html.P(f"Pace Volatility: {pace_vol}")
    ])
    
    return [summary_div, table], fig


@app.callback(
    Output('best-race-container', 'children'),
    Input('analyze-best-races-btn', 'n_clicks'),
    State('race-dropdown', 'value'),
    prevent_initial_call=False
)
def analyze_best_races(n_clicks, selected_race):
    """Analyze and display best race patterns"""
    # Show initial message
    if n_clicks is None:
        return html.P(
            "Click 'Analyze Best Races' to identify patterns in each horse's best performances.",
            className="text-muted"
        )
    
    if analyzer.past_starts_df is None:
        return html.Div("Please load data first", className="text-warning")
    
    # Run best race analysis
    report = analyzer.analyze_best_races()
    
    if not report or not report['race_reports']:
        return html.Div("No best race patterns found", className="text-warning")
    
    # Create display components
    components = []
    
    # Add a test message to verify content is showing
    components.append(html.H5("Analysis Complete!", className="text-success mb-3"))
    
    # Overall insights
    if report['consistent_factors']:
        insights_div = html.Div([
            html.H5("Key Factors in Best Performances:", className="text-info mb-3"),
            html.Ul([
                html.Li([
                    html.Strong(f"{factor['factor'].replace('pp_', '').replace('_', ' ').title()}: "),
                    f"{factor['direction']} by {abs(factor['avg_change']):.1f} on average ",
                    f"({factor['consistency']:.0%} of horses)"
                ])
                for factor in report['consistent_factors'][:5]  # Top 5 factors
            ])
        ])
        components.append(insights_div)
    
    # Race-specific analysis
    if selected_race and selected_race in report['race_reports']:
        race_data = report['race_reports'][selected_race]
        
        if race_data['patterns']:
            race_div = html.Div([
                html.Hr(),
                html.H5(f"Race {selected_race} - Best Race Analysis", className="mb-3"),
                html.Div([
                    create_horse_best_race_card(pattern)
                    for pattern in race_data['patterns']
                ])
            ])
            components.append(race_div)
    
    # Add bottom padding to ensure visibility
    components.append(html.Div(style={'height': '100px'}))
    
    return html.Div(components)


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


if __name__ == "__main__":
    app.run(debug=True, port=8051)