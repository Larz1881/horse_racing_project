# src/transformers/integrated_analytics_system.py
"""
Integrated Analytics System

Combines all advanced metrics into a unified analytical framework:
1. Composite Fitness Score with time-to-peak predictions
2. Class-Adjusted Speed Figures
3. Pace Impact Predictions
4. Machine Learning Enhancements
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntegratedAnalyticsSystem:
    """Unified system combining all analytical components"""
    
    def __init__(self, base_path: str = 'data/processed'):
        """
        Initialize with path to processed data
        
        Args:
            base_path: Base directory containing all processed files
        """
        self.base_path = Path(base_path)
        
        # Load all component analyses
        self._load_all_data()
        
        # Initialize ML models
        self.models = {}
        self.scalers = {}
        
    def _load_all_data(self):
        """Load all previously calculated metrics"""
        logger.info("Loading all analytical components...")
        
        # Core data
        self.current_df = pd.read_parquet(self.base_path / 'current_race_info.parquet')
        self.past_df = pd.read_parquet(self.base_path / 'past_starts_long_format.parquet')
        
        # Component analyses (check if exist)
        components = {
            'fitness': 'advanced_fitness_metrics.parquet',
            'workout': 'sophisticated_workout_analysis.parquet',
            'pace': 'advanced_pace_analysis.parquet',
            'class': 'multi_dimensional_class_assessment.parquet',
            'form': 'form_cycle_analysis.parquet'
        }
        
        self.component_data = {}
        for name, filename in components.items():
            filepath = self.base_path / filename
            if filepath.exists():
                self.component_data[name] = pd.read_parquet(filepath)
                logger.info(f"Loaded {name} data: {len(self.component_data[name])} records")
            else:
                logger.warning(f"{name} data not found at {filepath}")
                self.component_data[name] = pd.DataFrame()
    
    def calculate_composite_fitness_score(self) -> pd.DataFrame:
        """
        Calculate weighted composite fitness score with peak predictions
        
        Returns:
            DataFrame with comprehensive fitness assessment
        """
        logger.info("Calculating composite fitness scores...")
        
        # Start with base data
        composite_df = self.current_df[['race', 'horse_name', 'post_position',
                                        'morn_line_odds_if_available']].copy()
        
        # Merge all component scores
        score_columns = {}
        
        # Fitness metrics
        if not self.component_data['fitness'].empty:
            fitness_cols = ['race', 'horse_name', 'composite_fitness_score', 'recovery_score',
                           'form_momentum_score', 'cardiovascular_fitness_score',
                           'sectional_improvement_score', 'energy_efficiency_score']
            available_cols = [col for col in fitness_cols if col in self.component_data['fitness'].columns]
            composite_df = composite_df.merge(
                self.component_data['fitness'][available_cols],
                on=['race', 'horse_name'],
                how='left',
                suffixes=('', '_fitness')
            )
            score_columns['fitness'] = 'composite_fitness_score'
        
        # Workout readiness
        if not self.component_data['workout'].empty:
            workout_cols = ['race', 'horse_name', 'workout_readiness_score',
                           'workout_quality_score', 'trainer_intent_signals']
            available_cols = [col for col in workout_cols if col in self.component_data['workout'].columns]
            composite_df = composite_df.merge(
                self.component_data['workout'][available_cols],
                on=['race', 'horse_name'],
                how='left'
            )
            score_columns['workout'] = 'workout_readiness_score'
        
        # Pace advantage
        if not self.component_data['pace'].empty:
            pace_cols = ['race', 'horse_name', 'pace_advantage_score', 'fade_risk',
                        'energy_efficiency', 'pace_scenario', 'pace_shape']
            available_cols = [col for col in pace_cols if col in self.component_data['pace'].columns]
            composite_df = composite_df.merge(
                self.component_data['pace'][available_cols],
                on=['race', 'horse_name'],
                how='left',
                suffixes=('', '_pace')
            )
            score_columns['pace'] = 'pace_advantage_score'
        
        # Class rating
        if not self.component_data['class'].empty:
            class_cols = ['race', 'horse_name', 'overall_class_rating', 'class_edge',
                         'earnings_class_score', 'hidden_class_score']
            available_cols = [col for col in class_cols if col in self.component_data['class'].columns]
            composite_df = composite_df.merge(
                self.component_data['class'][available_cols],
                on=['race', 'horse_name'],
                how='left'
            )
            score_columns['class'] = 'overall_class_rating'
        
        # Form cycle
        if not self.component_data['form'].empty:
            form_cols = ['race', 'horse_name', 'composite_form_score', 'form_cycle_state',
                        'form_edge', 'beaten_time_trend']
            available_cols = [col for col in form_cols if col in self.component_data['form'].columns]
            composite_df = composite_df.merge(
                self.component_data['form'][available_cols],
                on=['race', 'horse_name'],
                how='left'
            )
            score_columns['form'] = 'composite_form_score'
        
        # Calculate integrated fitness score
        weights = {
            'fitness': 0.25,
            'workout': 0.15,
            'pace': 0.20,
            'class': 0.20,
            'form': 0.20
        }
        
        composite_df['integrated_fitness_score'] = 0
        total_weight = 0
        
        for component, score_col in score_columns.items():
            if score_col in composite_df.columns:
                composite_df['integrated_fitness_score'] += (
                    composite_df[score_col].fillna(50) * weights[component]
                )
                total_weight += weights[component]
        
        # Normalize by actual weights used
        if total_weight > 0:
            composite_df['integrated_fitness_score'] /= total_weight
        else:
            composite_df['integrated_fitness_score'] = 50
        
        # Time-to-peak predictions
        composite_df['time_to_peak'] = composite_df.apply(
            self._predict_time_to_peak, axis=1
        )
        
        # Flag horses entering optimal form
        composite_df['entering_optimal_form'] = (
            (composite_df['integrated_fitness_score'] > 70) &
            (composite_df.get('form_cycle_state', '') == 'IMPROVING') &
            (composite_df.get('form_momentum_score', 0) > 60)
        )
        
        # Add confidence score
        composite_df['fitness_confidence'] = self._calculate_confidence_score(composite_df)
        
        return composite_df
    
    def _predict_time_to_peak(self, row) -> str:
        """Predict when horse will reach peak form"""
        # Use form cycle state as primary indicator
        form_state = row.get('form_cycle_state', 'UNKNOWN')
        
        if form_state == 'PEAKING':
            return '0-1 races'
        elif form_state == 'IMPROVING':
            # Check momentum
            momentum = row.get('form_momentum_score', 50)
            if momentum > 70:
                return '1-2 races'
            else:
                return '2-3 races'
        elif form_state == 'RECOVERING':
            return '2-4 races'
        elif form_state == 'FRESHENING':
            # Check workout readiness
            workout_ready = row.get('workout_readiness_score', 50)
            if workout_ready > 70:
                return '0-1 races'
            else:
                return '1-2 races'
        elif form_state in ['DECLINING', 'BOUNCING']:
            return '4+ races'
        else:
            return 'Unknown'
    
    def _calculate_confidence_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate confidence in fitness assessment"""
        confidence = pd.Series(50, index=df.index)
        
        # More data points = higher confidence
        components_present = 0
        for col in ['composite_fitness_score', 'workout_readiness_score',
                   'pace_advantage_score', 'overall_class_rating', 'composite_form_score']:
            if col in df.columns:
                components_present += df[col].notna().astype(int)
        
        confidence += components_present * 5
        
        # Consistency across metrics
        if 'form_cycle_state' in df.columns:
            confidence += (df['form_cycle_state'] == 'STABLE').astype(int) * 10
            confidence += (df['form_cycle_state'] == 'IMPROVING').astype(int) * 15
        
        return confidence.clip(0, 100)
    
    def calculate_class_adjusted_speed_figures(self) -> pd.DataFrame:
        """
        Adjust speed figures based on class context
        
        Returns:
            DataFrame with class-adjusted ratings
        """
        logger.info("Calculating class-adjusted speed figures...")
        
        # Get recent performances with class info
        recent_perfs = []
        
        for (race, horse), group in self.past_df.groupby(['race', 'horse_name']):
            recent = group.head(10)
            
            for idx, row in recent.iterrows():
                perf = {
                    'race': race,
                    'horse_name': horse,
                    'pp_race_date': row['pp_race_date'],
                    'pp_bris_speed_rating': row.get('pp_bris_speed_rating'),
                    'pp_race_type': row.get('pp_race_type'),
                    'pp_purse': row.get('pp_purse'),
                    'pp_num_entrants': row.get('pp_num_entrants'),
                    'pp_finish_pos': row.get('pp_finish_pos')
                }
                recent_perfs.append(perf)
        
        perfs_df = pd.DataFrame(recent_perfs)
        
        # Calculate class levels for each race
        perfs_df['class_level'] = perfs_df.apply(self._calculate_race_class_level, axis=1)
        
        # Adjust speed ratings by class
        perfs_df['class_adjusted_speed'] = perfs_df.apply(
            lambda row: self._adjust_speed_for_class(
                row['pp_bris_speed_rating'],
                row['class_level']
            ),
            axis=1
        )
        
        # Aggregate to horse level
        adjusted_ratings = []
        
        for (race, horse), group in perfs_df.groupby(['race', 'horse_name']):
            valid_speeds = group['class_adjusted_speed'].dropna()
            
            if len(valid_speeds) > 0:
                rating_data = {
                    'race': race,
                    'horse_name': horse,
                    'avg_class_adjusted_speed': valid_speeds.mean(),
                    'best_class_adjusted_speed': valid_speeds.max(),
                    'recent_class_adjusted_speed': valid_speeds.iloc[0] if len(valid_speeds) > 0 else None,
                    'speed_consistency': 100 - (valid_speeds.std() * 2) if len(valid_speeds) > 1 else 75
                }
                
                # Pound-for-pound rating (performance relative to class)
                avg_class = group['class_level'].mean()
                avg_speed = group['pp_bris_speed_rating'].mean()
                
                if pd.notna(avg_speed) and pd.notna(avg_class):
                    # Higher speed in lower class = higher p4p rating
                    rating_data['pound_for_pound_rating'] = avg_speed / (avg_class / 5)
                else:
                    rating_data['pound_for_pound_rating'] = 80
                
                # Project class movement success
                if not self.component_data['class'].empty:
                    class_data = self.component_data['class'][
                        (self.component_data['class']['race'] == race) &
                        (self.component_data['class']['horse_name'] == horse)
                    ]
                    
                    if len(class_data) > 0:
                        class_move = class_data.iloc[0].get('class_move_direction', 'Unknown')
                        
                        if class_move == 'Up':
                            # Check if speed figures support move up
                            if rating_data['pound_for_pound_rating'] > 85:
                                rating_data['class_move_projection'] = 'Likely_success'
                            else:
                                rating_data['class_move_projection'] = 'Questionable'
                        elif class_move == 'Down':
                            rating_data['class_move_projection'] = 'Should_dominate'
                        else:
                            rating_data['class_move_projection'] = 'Appropriate_level'
                
                adjusted_ratings.append(rating_data)
        
        return pd.DataFrame(adjusted_ratings)
    
    def _calculate_race_class_level(self, row) -> float:
        """Calculate numeric class level for a race"""
        # Use hierarchy from class assessment module
        race_type_values = {
            'M': 1, 'S': 2, 'MO': 2.5, 'C': 3, 'CO': 4,
            'R': 5, 'T': 5, 'A': 6, 'AO': 6.5,
            'N': 7, 'NO': 7.5, 'G3': 8, 'G2': 9, 'G1': 10
        }
        
        base_level = race_type_values.get(row.get('pp_race_type', ''), 5)
        
        # Adjust for purse
        purse = row.get('pp_purse', 0)
        if pd.notna(purse):
            if purse > 100000:
                base_level += 0.5
            elif purse > 500000:
                base_level += 1
            elif purse < 20000:
                base_level -= 0.5
        
        return base_level
    
    def _adjust_speed_for_class(self, speed: float, class_level: float) -> float:
        """Adjust speed figure based on class context"""
        if pd.isna(speed) or pd.isna(class_level):
            return speed
        
        # Base adjustment: higher class = bonus to speed figure
        class_adjustment = (class_level - 5) * 2  # ±2 points per class level
        
        return speed + class_adjustment
    
    def calculate_pace_impact_predictions(self) -> pd.DataFrame:
        """
        Calculate individual pace advantages and race flow predictions
        
        Returns:
            DataFrame with pace impact analysis
        """
        logger.info("Calculating pace impact predictions...")
        
        if self.component_data['pace'].empty:
            logger.warning("No pace data available")
            return pd.DataFrame()
        
        pace_df = self.component_data['pace'].copy()
        
        # Add additional pace impact metrics
        pace_impact = []
        
        for race in pace_df['race'].unique():
            race_horses = pace_df[pace_df['race'] == race]
            
            if len(race_horses) == 0:
                continue
            
            # Get pace scenario
            pace_scenario = race_horses.iloc[0].get('pace_scenario', 'Unknown')
            pace_shape = race_horses.iloc[0].get('pace_shape', 'Unknown')
            
            for idx, horse in race_horses.iterrows():
                impact_data = {
                    'race': race,
                    'horse_name': horse['horse_name'],
                    'pace_scenario': pace_scenario,
                    'pace_shape': pace_shape
                }
                
                # Individual advantages based on scenario
                run_style = horse.get('run_style', '')
                early_position = horse.get('projected_1st_call_position', 5)
                
                # Calculate scenario-specific advantages
                if pace_scenario == 'Hot':
                    if run_style in ['S', 'SS']:
                        impact_data['pace_advantage'] = 'Major_beneficiary'
                        impact_data['advantage_score'] = 85
                    elif run_style in ['E', 'E/P']:
                        impact_data['pace_advantage'] = 'Likely_victim'
                        impact_data['advantage_score'] = 20
                    else:
                        impact_data['pace_advantage'] = 'Neutral'
                        impact_data['advantage_score'] = 50
                
                elif pace_scenario == 'Lone Speed':
                    if early_position == 1:
                        impact_data['pace_advantage'] = 'Controls_race'
                        impact_data['advantage_score'] = 90
                    else:
                        impact_data['pace_advantage'] = 'Must_close_into_slow_pace'
                        impact_data['advantage_score'] = 30
                
                else:  # Honest or moderate pace
                    impact_data['pace_advantage'] = 'Even_chance'
                    impact_data['advantage_score'] = 50
                
                # Energy reserve impact
                energy_reserve = horse.get('energy_reserve', 0)
                if energy_reserve < -10:
                    impact_data['energy_concern'] = 'High_fade_risk'
                    impact_data['advantage_score'] -= 15
                elif energy_reserve > 20:
                    impact_data['energy_concern'] = 'Strong_reserve'
                    impact_data['advantage_score'] += 10
                else:
                    impact_data['energy_concern'] = 'Adequate'
                
                # Race flow position
                if early_position <= 3:
                    impact_data['race_flow_position'] = 'Pace_setter'
                elif early_position <= 6:
                    impact_data['race_flow_position'] = 'Stalker'
                else:
                    impact_data['race_flow_position'] = 'Closer'
                
                # Tactical advantage
                if horse.get('energy_efficiency', 0) > 80:
                    impact_data['tactical_advantage'] = 'Versatile'
                elif run_style in ['E'] and pace_scenario != 'Lone Speed':
                    impact_data['tactical_advantage'] = 'One_dimensional'
                else:
                    impact_data['tactical_advantage'] = 'Style_dependent'
                
                pace_impact.append(impact_data)
        
        return pd.DataFrame(pace_impact)
    
    def train_ml_models(self) -> Dict:
        """
        Train machine learning models for advanced predictions
        
        Returns:
            Dictionary of trained models and performance metrics
        """
        logger.info("Training machine learning models...")
        
        results = {}
        
        # 1. Form Cycle State Classifier
        form_model_results = self._train_form_cycle_classifier()
        if form_model_results:
            results['form_cycle_classifier'] = form_model_results
        
        # 2. Workout Pattern Clustering
        workout_clusters = self._perform_workout_clustering()
        if workout_clusters is not None:
            results['workout_clusters'] = workout_clusters
        
        # 3. Class Trajectory Predictor
        class_model_results = self._train_class_trajectory_model()
        if class_model_results:
            results['class_trajectory_model'] = class_model_results
        
        # 4. Pace Scenario Optimizer
        pace_optimizer_results = self._train_pace_optimizer()
        if pace_optimizer_results:
            results['pace_optimizer'] = pace_optimizer_results
        
        # 5. Integrated Performance Predictor
        performance_model = self._train_performance_predictor()
        if performance_model:
            results['performance_predictor'] = performance_model
        
        return results
    
    def _train_form_cycle_classifier(self) -> Optional[Dict]:
        """Train classifier to predict form cycle states"""
        if self.component_data['form'].empty or self.past_df.empty:
            logger.warning("Insufficient data for form cycle classifier")
            return None
        
        logger.info("Training form cycle state classifier...")
        
        # Prepare features
        features = []
        labels = []
        
        for idx, row in self.component_data['form'].iterrows():
            if pd.isna(row.get('form_cycle_state')):
                continue
            
            # Extract features
            feature_vec = [
                row.get('beaten_time_trend', 0),
                row.get('avg_position_gain', 0),
                row.get('form_momentum_score', 50),
                row.get('cardiovascular_fitness_score', 50),
                row.get('sectional_improvement_score', 0),
                row.get('current_days_off', 0),
                row.get('bounce_risk', '') == 'High',
                row.get('currently_freshening', False)
            ]
            
            features.append(feature_vec)
            labels.append(row['form_cycle_state'])
        
        if len(features) < 50:
            logger.warning("Insufficient samples for form cycle classifier")
            return None
        
        # Convert to arrays
        X = np.array(features)
        y = np.array(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        
        # Feature importance
        feature_names = [
            'beaten_time_trend', 'avg_position_gain', 'form_momentum_score',
            'cardiovascular_fitness', 'sectional_improvement', 'days_off',
            'bounce_risk', 'freshening'
        ]
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Store model
        self.models['form_cycle_classifier'] = model
        self.scalers['form_cycle_classifier'] = scaler
        
        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'feature_importance': feature_importance,
            'model': model,
            'scaler': scaler
        }
    
    def _perform_workout_clustering(self) -> Optional[pd.DataFrame]:
        """Cluster horses based on workout patterns"""
        if self.component_data['workout'].empty:
            logger.warning("No workout data for clustering")
            return None
        
        logger.info("Performing workout pattern clustering...")
        
        # Prepare features
        workout_features = []
        horse_ids = []
        
        workout_df = self.component_data['workout']
        
        for idx, row in workout_df.iterrows():
            features = [
                row.get('workout_quality_score', 50),
                row.get('bullet_work_count', 0),
                row.get('works_last_14d', 0),
                row.get('pct_fast_works', 0),
                row.get('gate_work_count', 0),
                row.get('frequency_signal', '') == 'increasing',
                row.get('bullet_timing', '') == 'recent_sharp'
            ]
            
            workout_features.append(features)
            horse_ids.append((row['race'], row['horse_name']))
        
        if len(workout_features) < 10:
            logger.warning("Insufficient samples for workout clustering")
            return None
        
        # Scale features
        X = np.array(workout_features)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering
        n_clusters = min(5, len(X) // 10)  # Adaptive number of clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Analyze clusters
        cluster_df = pd.DataFrame({
            'race': [h[0] for h in horse_ids],
            'horse_name': [h[1] for h in horse_ids],
            'workout_cluster': clusters
        })
        
        # Characterize clusters
        cluster_profiles = []
        feature_names = [
            'quality_score', 'bullet_count', 'recent_works',
            'fast_work_pct', 'gate_works', 'increasing_freq', 'recent_sharp'
        ]
        
        for cluster_id in range(n_clusters):
            cluster_mask = clusters == cluster_id
            cluster_features = X[cluster_mask].mean(axis=0)
            
            profile = {
                'cluster_id': cluster_id,
                'size': cluster_mask.sum(),
                'avg_quality': cluster_features[0],
                'profile': self._interpret_workout_cluster(cluster_features)
            }
            cluster_profiles.append(profile)
        
        # Store clustering model
        self.models['workout_clustering'] = kmeans
        self.scalers['workout_clustering'] = scaler
        
        # Add cluster profiles to results
        cluster_df = cluster_df.merge(
            pd.DataFrame(cluster_profiles),
            left_on='workout_cluster',
            right_on='cluster_id',
            how='left'
        )
        
        return cluster_df
    
    def _interpret_workout_cluster(self, features: np.ndarray) -> str:
        """Interpret workout cluster characteristics"""
        quality, bullets, recent, fast_pct, gates, increasing, sharp = features
        
        if quality > 70 and bullets > 0.5 and sharp > 0.5:
            return "Peak_preparation"
        elif recent > 2 and increasing > 0.5:
            return "Heavy_training"
        elif fast_pct > 50 and gates > 1:
            return "Speed_sharpening"
        elif quality < 40 and recent < 1:
            return "Light_maintenance"
        else:
            return "Standard_preparation"
    
    def _train_class_trajectory_model(self) -> Optional[Dict]:
        """Train model to predict class movement success"""
        if self.component_data['class'].empty:
            logger.warning("No class data for trajectory model")
            return None
        
        logger.info("Training class trajectory predictor...")
        
        # Prepare training data from historical performances
        training_data = []
        
        for (race, horse), group in self.past_df.groupby(['race', 'horse_name']):
            if len(group) < 5:
                continue
            
            # Look at class changes and outcomes
            for i in range(len(group) - 1):
                current = group.iloc[i]
                previous = group.iloc[i + 1]
                
                # Calculate class change
                current_class = self._calculate_race_class_level(current)
                previous_class = self._calculate_race_class_level(previous)
                class_change = current_class - previous_class
                
                # Features
                features = [
                    class_change,  # Magnitude of class move
                    current.get('pp_bris_speed_rating', 80),
                    previous.get('pp_bris_speed_rating', 80),
                    current.get('pp_finish_pos', 5),
                    current.get('pp_num_entrants', 8),
                    current.get('pp_odds', 10)
                ]
                
                # Target: success in new class (top 3 finish)
                success = current.get('pp_finish_pos', 99) <= 3
                
                training_data.append({
                    'features': features,
                    'success': success,
                    'class_change': class_change
                })
        
        if len(training_data) < 50:
            logger.warning("Insufficient data for class trajectory model")
            return None
        
        # Convert to arrays
        X = np.array([d['features'] for d in training_data])
        y = np.array([d['success'] for d in training_data])
        
        # Split and scale
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=4,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        
        # Store model
        self.models['class_trajectory'] = model
        self.scalers['class_trajectory'] = scaler
        
        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'model': model,
            'scaler': scaler
        }
    
    def _train_pace_optimizer(self) -> Optional[Dict]:
        """Train model to optimize pace positioning"""
        if self.component_data['pace'].empty:
            logger.warning("No pace data for optimizer")
            return None
        
        logger.info("Training pace scenario optimizer...")
        
        # Use historical data to learn optimal positioning
        training_data = []
        
        for (race, horse), group in self.past_df.groupby(['race', 'horse_name']):
            for idx, row in group.iterrows():
                # Features about race pace
                features = [
                    row.get('pp_e1_pace', 95),
                    row.get('pp_e2_pace', 95),
                    row.get('pp_bris_late_pace', 95),
                    row.get('pp_first_call_pos', 5),
                    row.get('pp_num_entrants', 8),
                    row.get('pp_pos_gain_start_to_finish', 0)
                ]
                
                # Target: optimal result
                finish = row.get('pp_finish_pos', 99)
                success_score = max(0, (10 - finish) / 9) * 100  # 0-100 scale
                
                training_data.append({
                    'features': features,
                    'success': success_score
                })
        
        if len(training_data) < 100:
            logger.warning("Insufficient data for pace optimizer")
            return None
        
        # Convert to arrays
        X = np.array([d['features'] for d in training_data])
        y = np.array([d['success'] for d in training_data])
        
        # Split and scale
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train regression model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, model.predict(X_train_scaled)))
        test_rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test_scaled)))
        
        # Store model
        self.models['pace_optimizer'] = model
        self.scalers['pace_optimizer'] = scaler
        
        return {
            'train_r2': train_score,
            'test_r2': test_score,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'model': model,
            'scaler': scaler
        }
    
    def _train_performance_predictor(self) -> Optional[Dict]:
        """Train integrated model to predict race performance"""
        logger.info("Training integrated performance predictor...")
        
        # Combine all available features
        integrated_df = self.calculate_composite_fitness_score()
        
        if integrated_df.empty:
            logger.warning("No integrated data for performance predictor")
            return None
        
        # Prepare features
        feature_cols = [
            'integrated_fitness_score', 'composite_fitness_score',
            'workout_readiness_score', 'pace_advantage_score',
            'overall_class_rating', 'composite_form_score'
        ]
        
        available_features = [col for col in feature_cols if col in integrated_df.columns]
        
        if len(available_features) < 3:
            logger.warning("Insufficient features for performance predictor")
            return None
        
        # For training, we need historical outcomes
        # This is a placeholder - in production, you'd have actual race results
        # For now, we'll create synthetic targets based on our scoring
        
        X = integrated_df[available_features].fillna(50).values
        
        # Synthetic target: combination of all scores with some noise
        y = integrated_df['integrated_fitness_score'].values + np.random.normal(0, 5, len(integrated_df))
        y = np.clip(y, 0, 100)
        
        if len(X) < 50:
            logger.warning("Insufficient samples for performance predictor")
            return None
        
        # Split and scale
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=150,
            max_depth=6,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': available_features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Store model
        self.models['performance_predictor'] = model
        self.scalers['performance_predictor'] = scaler
        
        return {
            'train_r2': train_score,
            'test_r2': test_score,
            'feature_importance': feature_importance,
            'model': model,
            'scaler': scaler
        }
    
    def generate_integrated_report(self) -> pd.DataFrame:
        """
        Generate comprehensive integrated analysis report
        
        Returns:
            DataFrame with all integrated metrics and predictions
        """
        logger.info("Generating integrated analysis report...")
        
        # Start with composite fitness
        report = self.calculate_composite_fitness_score()
        
        # Add class-adjusted speeds
        class_adjusted = self.calculate_class_adjusted_speed_figures()
        if not class_adjusted.empty:
            report = report.merge(
                class_adjusted[['race', 'horse_name', 'avg_class_adjusted_speed',
                               'pound_for_pound_rating', 'class_move_projection']],
                on=['race', 'horse_name'],
                how='left'
            )
        
        # Add pace impact
        pace_impact = self.calculate_pace_impact_predictions()
        if not pace_impact.empty:
            report = report.merge(
                pace_impact[['race', 'horse_name', 'pace_advantage',
                            'advantage_score', 'race_flow_position']],
                on=['race', 'horse_name'],
                how='left'
            )
        
        # Apply ML predictions if models are trained
        if 'performance_predictor' in self.models:
            feature_cols = [col for col in report.columns 
                           if col.endswith('_score') or col.endswith('_rating')]
            
            if feature_cols:
                X = report[feature_cols].fillna(50).values
                X_scaled = self.scalers['performance_predictor'].transform(X)
                
                report['ml_performance_prediction'] = self.models['performance_predictor'].predict(X_scaled)
                report['ml_performance_prediction'] = report['ml_performance_prediction'].clip(0, 100)
        
        # Calculate final rankings
        ranking_factors = ['integrated_fitness_score']
        
        if 'ml_performance_prediction' in report.columns:
            ranking_factors.append('ml_performance_prediction')
        
        if 'pound_for_pound_rating' in report.columns:
            ranking_factors.append('pound_for_pound_rating')
        
        # Weight the factors
        report['final_rating'] = report[ranking_factors].fillna(50).mean(axis=1)
        
        # Rank within race
        report['overall_rank'] = report.groupby('race')['final_rating'].rank(
            ascending=False, method='min'
        )
        
        # Add key insights
        report['key_angles'] = report.apply(self._identify_key_angles, axis=1)
        
        # Add confidence rating
        report['prediction_confidence'] = report['fitness_confidence']
        
        return report
    
    def _identify_key_angles(self, row) -> str:
        """Identify key betting angles for each horse"""
        angles = []
        
        # Top-rated horse
        if row.get('overall_rank') == 1:
            angles.append('Top_rated')
        
        # Class dropper with form
        if (row.get('class_move_projection') == 'Should_dominate' and 
            row.get('form_cycle_state') in ['IMPROVING', 'PEAKING']):
            angles.append('Class_drop_ready')
        
        # Pace setup
        if row.get('pace_advantage') == 'Major_beneficiary':
            angles.append('Pace_setup')
        
        # Fresh and working well
        if (row.get('form_cycle_state') == 'FRESHENING' and 
            row.get('workout_readiness_score', 0) > 70):
            angles.append('Fresh_and_working')
        
        # Hidden value
        if (row.get('morn_line_odds_if_available', 0) > 10 and 
            row.get('overall_rank', 99) <= 3):
            angles.append('Value_play')
        
        # Peak form
        if row.get('entering_optimal_form'):
            angles.append('Entering_peak')
        
        return ', '.join(angles) if angles else 'None'
    
    def save_models(self, model_dir: str = 'models'):
        """Save trained models to disk"""
        model_path = Path(model_dir)
        model_path.mkdir(exist_ok=True)
        
        for name, model in self.models.items():
            joblib.dump(model, model_path / f'{name}.pkl')
            
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, model_path / f'{name}_scaler.pkl')
        
        logger.info(f"Saved {len(self.models)} models to {model_path}")
    
    def load_models(self, model_dir: str = 'models'):
        """Load trained models from disk"""
        model_path = Path(model_dir)
        
        if not model_path.exists():
            logger.warning(f"Model directory {model_path} not found")
            return
        
        # Load models
        for model_file in model_path.glob('*.pkl'):
            if '_scaler' in model_file.stem:
                name = model_file.stem.replace('_scaler', '')
                self.scalers[name] = joblib.load(model_file)
            else:
                name = model_file.stem
                self.models[name] = joblib.load(model_file)
        
        logger.info(f"Loaded {len(self.models)} models from {model_path}")


def main():
    """Main function to run integrated analytics"""
    
    # Initialize system
    analytics = IntegratedAnalyticsSystem()
    
    # Train ML models
    logger.info("Training machine learning models...")
    ml_results = analytics.train_ml_models()
    
    # Display ML results
    print("\n" + "="*70)
    print("MACHINE LEARNING MODEL RESULTS")
    print("="*70)
    
    for model_name, results in ml_results.items():
        print(f"\n{model_name.upper()}:")
        if isinstance(results, dict):
            for key, value in results.items():
                if key not in ['model', 'scaler', 'feature_importance']:
                    print(f"  {key}: {value:.3f}")
            
            if 'feature_importance' in results and isinstance(results['feature_importance'], pd.DataFrame):
                print("\n  Feature Importance:")
                print(results['feature_importance'].head().to_string(index=False))
    
    # Generate integrated report
    integrated_report = analytics.generate_integrated_report()
    
    # Save results
    output_path = Path('data/processed/integrated_analytics_report.parquet')
    integrated_report.to_parquet(output_path, index=False)
    logger.info(f"Saved integrated report to {output_path}")
    
    # Save models
    analytics.save_models()
    
    # Display summary
    print("\n" + "="*70)
    print("INTEGRATED ANALYTICS SUMMARY")
    print("="*70)
    print(f"Total horses analyzed: {len(integrated_report)}")
    
    print("\nTOP 10 HORSES BY FINAL RATING:")
    top_horses = integrated_report.nlargest(10, 'final_rating')[
        ['race', 'horse_name', 'post_position', 'final_rating', 
         'overall_rank', 'key_angles', 'prediction_confidence']
    ]
    print(top_horses.to_string(index=False))
    
    print("\nKEY ANGLES DISTRIBUTION:")
    all_angles = []
    for angles in integrated_report['key_angles']:
        if angles != 'None':
            all_angles.extend(angles.split(', '))
    
    if all_angles:
        angle_counts = pd.Series(all_angles).value_counts()
        print(angle_counts.head(10))
    
    print("\nRACE-BY-RACE TOP 3:")
    for race in sorted(integrated_report['race'].unique()):
        race_horses = integrated_report[integrated_report['race'] == race].nsmallest(3, 'overall_rank')
        print(f"\nRace {race}:")
        for idx, horse in race_horses.iterrows():
            print(f"  {int(horse['overall_rank'])}. {horse['horse_name']} "
                  f"(Post {int(horse['post_position'])}) - "
                  f"Rating: {horse['final_rating']:.1f} - "
                  f"{horse['key_angles']}")
    
    # Save summary CSV
    summary_path = Path('data/processed/integrated_summary.csv')
    summary_cols = [
        'race', 'horse_name', 'post_position', 'morn_line_odds_if_available',
        'final_rating', 'overall_rank', 'integrated_fitness_score',
        'form_cycle_state', 'pace_advantage', 'class_move_projection',
        'key_angles', 'prediction_confidence'
    ]
    
    available_cols = [col for col in summary_cols if col in integrated_report.columns]
    summary_df = integrated_report[available_cols].sort_values(['race', 'overall_rank'])
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved integrated summary to {summary_path}")


if __name__ == '__main__':
    main()