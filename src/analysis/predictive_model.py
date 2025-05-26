"""
Predictive model system for horse racing that integrates with existing feature engineering.
Location: src/analysis/predictive_model.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
import logging
from datetime import datetime
import pickle

from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from scipy import stats

# Dummy class for holding preprocessing info (pickle can serialize this!)
class PreprocessingHolder:
    pass


def preprocess_features(self, X: pd.DataFrame, model_name: str, 
                          preprocessing_info: Dict = None, fit: bool = False) -> Tuple[Any, Dict]:
    """Centralized preprocessing to ensure consistency between train and test."""
    X_processed = X.copy()
    
    # Initialize preprocessing info if fitting
    if fit:
        preprocessing_info = {
            'encoders': {},
            'categorical_cols': [],
            'numeric_cols': [],
            'categorical_dropped': [],
            'original_columns': X.columns.tolist(),
            'final_columns': []
        }
    
    # Identify column types
    categorical_cols = X_processed.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = X_processed.select_dtypes(include=[np.number]).columns.tolist()
    
    if fit:
        preprocessing_info['categorical_cols'] = categorical_cols
        preprocessing_info['numeric_cols'] = numeric_cols
        if model_name == 'lgb_regularized':
            logger.info(f"Preprocessing (fit) - Model: {model_name}, Categorical: {categorical_cols}, Numeric: {len(numeric_cols)}")
    else:
        if model_name == 'lgb_regularized':
            logger.info(f"Preprocessing (transform) - Model: {model_name}, Input categorical: {categorical_cols}")
            logger.info(f"Preprocessing (transform) - Expected categorical: {preprocessing_info.get('categorical_cols', [])}")
            logger.info(f"Preprocessing (transform) - Available encoders: {list(preprocessing_info.get('encoders', {}).keys())}")
    
    # Handle categorical features
    if categorical_cols:
        for col in categorical_cols:
            if model_name in ['lgb_regularized', 'rf_constrained']:
                # Tree models: use label encoding
                X_processed[col] = X_processed[col].fillna('unknown')
                
                if fit:
                    # Fit encoder
                    le = LabelEncoder()
                    X_processed[col] = le.fit_transform(X_processed[col])
                    preprocessing_info['encoders'][col] = le
                else:
                    # Transform using existing encoder
                    if col in preprocessing_info.get('encoders', {}):
                        le = preprocessing_info['encoders'][col]
                        # Handle unseen categories
                        known_classes = set(le.classes_)
                        X_processed[col] = X_processed[col].apply(
                            lambda x: x if x in known_classes else 'unknown'
                        )
                        X_processed[col] = le.transform(X_processed[col])
                    elif col in preprocessing_info.get('categorical_cols', []):
                        # This categorical was present during training but no encoder
                        # This shouldn't happen for tree models
                        logger.warning(f"Categorical column {col} was in training but has no encoder")
                        X_processed = X_processed.drop(columns=[col])
                    else:
                        # This is a new categorical column not seen during training
                        logger.warning(f"New categorical column {col} not seen during training - dropping")
                        X_processed = X_processed.drop(columns=[col])
                        
            elif model_name == 'domain_rules':
                # Domain model handles categoricals internally
                pass
            else:
                # Other models: drop categorical
                X_processed = X_processed.drop(columns=[col])
                if fit:
                    preprocessing_info['categorical_dropped'].append(col)
    
    # Handle missing values in numeric features
    numeric_cols_current = X_processed.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols_current:
        if model_name in ['lgb_regularized', 'rf_constrained']:
            X_processed[numeric_cols_current] = X_processed[numeric_cols_current].fillna(-999)
        else:
            # Use median imputation
            X_processed[numeric_cols_current] = X_processed[numeric_cols_current].fillna(
                X_processed[numeric_cols_current].median()
            )
    
    if fit:
        preprocessing_info['final_columns'] = X_processed.columns.tolist()
        if model_name == 'lgb_regularized':
            logger.info(f"Preprocessing (fit) - Final columns: {len(X_processed.columns)}")
    else:
        if model_name == 'lgb_regularized':
            logger.info(f"Preprocessing (transform) - Final columns: {len(X_processed.columns)}")
            expected_cols = set(preprocessing_info.get('final_columns', []))
            actual_cols = set(X_processed.columns)
            if expected_cols != actual_cols:
                logger.warning(f"Column mismatch! Missing: {expected_cols - actual_cols}, Extra: {actual_cols - expected_cols}")
    
    return X_processed, preprocessing_info

import lightgbm as lgb
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import your feature engineering module
from src.transformers.feature_engineering import HorseRacingFeatureEngineer
from config.settings import PROCESSED_DATA_DIR

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IntegratedHorseRacingModel:
    """
    Predictive model that integrates with existing feature engineering pipeline.
    Designed specifically for small datasets (max 10 races per horse).
    """
    
    def __init__(self, processed_data_dir: Path = None):
        self.processed_data_dir = processed_data_dir or PROCESSED_DATA_DIR
        self.models = {}
        self.scalers = {}
        self.selected_features = {}
        self.model_performance = {}
        self.ensemble_weights = {}
        
        # Feature sets based on your engineering
        self.feature_groups = {
            'core': [
                'last_3_speed_avg', 'speed_trajectory', 'best_speed_l10',
                'distance_suitability', 'surface_suitability',
                'class_differential', 'overall_rating'
            ],
            'pace': [
                'recent_e2_lengths', 'pace_style', 'avg_pace_kick',
                'pace_suitability', 'e2_consistency'
            ],
            'form': [
                'speed_trend', 'speed_consistency', 
                'best_speed_today_distance', 'best_speed_today_surface'
            ],
            'class': [
                'class_rise_drop', 'high_class_experience',
                'avg_class_level', 'class_consistency'
            ],
            'condition': [
                'condition_versatility', 'fast_track_experience',
                'recent_condition_performance'
            ]
        }
        
    def load_engineered_features(self) -> pd.DataFrame:
        """Load features created by your feature engineering pipeline."""
        features_path = self.processed_data_dir / "racing_features_for_modeling.parquet"
        
        if not features_path.exists():
            logger.info("Features not found. Running feature engineering...")
            engineer = HorseRacingFeatureEngineer(self.processed_data_dir)
            features_df = engineer.engineer_all_features()
        else:
            logger.info(f"Loading features from {features_path}")
            features_df = pd.read_parquet(features_path)
            
        return features_df
    
    def merge_with_results(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Merge features with actual race results for training."""
        # Load current race data with results
        current_race_path = self.processed_data_dir / "current_race_info.parquet"
        
        if current_race_path.exists():
            current_df = pd.read_parquet(current_race_path)
            
            # You'll need to add actual finish positions to your data
            # This is a placeholder - replace with actual results when available
            if 'finish_position' not in current_df.columns:
                logger.warning("No finish positions found. Using simulated data for demo.")
                # Simulate realistic finish positions for demo
                np.random.seed(42)
                current_df['finish_position'] = current_df.groupby('race')['post_position'].transform(
                    lambda x: np.random.permutation(np.arange(1, len(x) + 1))
                )
            
            # Merge with features
            result_df = features_df.merge(
                current_df[['race', 'horse_name', 'post_position', 'finish_position', 
                           'morn_line_odds_if_available']],
                on=['race', 'horse_name'],
                how='left'
            )
            
            # Create target variables
            result_df['target_win'] = (result_df['finish_position'] == 1).astype(int)
            result_df['target_top3'] = (result_df['finish_position'] <= 3).astype(int)
            
            return result_df
        else:
            raise FileNotFoundError(f"Current race info not found at {current_race_path}")
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       max_features: int = 15) -> List[str]:
        """Select most informative features for small dataset."""
        logger.info(f"Selecting top {max_features} features...")
        
        # Separate numeric and categorical columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # For feature selection, only use numeric columns
        X_numeric = X[numeric_cols].copy()
        
        # Handle missing values
        # Option 1: Drop features with too many missing values (>50%)
        missing_pct = X_numeric.isnull().sum() / len(X_numeric)
        numeric_cols_to_keep = missing_pct[missing_pct < 0.5].index.tolist()
        X_numeric = X_numeric[numeric_cols_to_keep]
        
        # Option 2: Simple imputation for remaining missing values
        # Use median for numeric features (robust to outliers)
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X_numeric)
        
        # Use mutual information for feature selection
        selector = SelectKBest(mutual_info_classif, k=min(max_features, X_numeric.shape[1]))
        selector.fit(X_imputed, y)
        
        # Get selected feature names
        selected_mask = selector.get_support()
        selected_features = [col for col, selected in zip(X_numeric.columns, selected_mask) if selected]
        
        # Always include core features if available (and not too many missing)
        core_features = [f for f in self.feature_groups['core'] 
                        if f in X.columns and f in numeric_cols_to_keep]
        
        # Add important categorical features that we want to keep
        important_categorical = ['pace_style', 'speed_trajectory', 'class_rise_drop']
        categorical_to_keep = [f for f in important_categorical 
                              if f in categorical_cols and f in X.columns]
        
        # Combine all selected features
        all_selected = list(set(selected_features + core_features + categorical_to_keep))
        final_features = all_selected[:max_features + len(categorical_to_keep)]
        
        logger.info(f"Numeric features with <50% missing: {len(numeric_cols_to_keep)}")
        logger.info(f"Categorical features included: {categorical_to_keep}")
        logger.info(f"Total selected features: {len(final_features)}")
        logger.info(f"Selected features: {final_features}")
        return final_features
    
    def preprocess_features(self, X: pd.DataFrame, model_name: str, 
                          preprocessing_info: Dict = None, fit: bool = False) -> Tuple[Any, Dict]:
        """Centralized preprocessing to ensure consistency between train and test."""
        X_processed = X.copy()
        
        # Initialize preprocessing info if fitting
        if fit:
            preprocessing_info = {
                'encoders': {},
                'categorical_cols': [],
                'numeric_cols': [],
                'categorical_dropped': [],
                'original_columns': X.columns.tolist(),
                'final_columns': []
            }
        
        # Identify column types
        categorical_cols = X_processed.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = X_processed.select_dtypes(include=[np.number]).columns.tolist()
        
        if fit:
            preprocessing_info['categorical_cols'] = categorical_cols
            preprocessing_info['numeric_cols'] = numeric_cols
        
        # Handle categorical features
        if categorical_cols:
            for col in categorical_cols:
                if model_name in ['lgb_regularized', 'rf_constrained']:
                    # Tree models: use label encoding
                    X_processed[col] = X_processed[col].fillna('unknown')
                    
                    if fit:
                        # Always ensure 'unknown' is in the data before fitting
                        fit_values = X_processed[col].tolist()
                        if 'unknown' not in fit_values:
                            fit_values.append('unknown')
                        le = LabelEncoder()
                        le.fit(fit_values)
                        X_processed[col] = le.transform(X_processed[col])
                        preprocessing_info['encoders'][col] = le
                    else:
                        # Transform using existing encoder
                        if col in preprocessing_info.get('encoders', {}):
                            le = preprocessing_info['encoders'][col]
                            # Handle unseen categories
                            known_classes = set(le.classes_)
                            X_processed[col] = X_processed[col].apply(
                                lambda x: x if x in known_classes else 'unknown'
                            )
                            X_processed[col] = le.transform(X_processed[col])
                        else:
                            # If no encoder exists, drop the column
                            X_processed = X_processed.drop(columns=[col])
                            
                elif model_name == 'domain_rules':
                    # Domain model handles categoricals internally
                    pass
                else:
                    # Other models: drop categorical
                    X_processed = X_processed.drop(columns=[col])
                    if fit:
                        preprocessing_info['categorical_dropped'].append(col)
        
        # Handle missing values in numeric features
        numeric_cols_current = X_processed.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols_current:
            if model_name in ['lgb_regularized', 'rf_constrained']:
                X_processed[numeric_cols_current] = X_processed[numeric_cols_current].fillna(-999)
            else:
                # Use median imputation
                X_processed[numeric_cols_current] = X_processed[numeric_cols_current].fillna(
                    X_processed[numeric_cols_current].median()
                )
        
        if fit:
            preprocessing_info['final_columns'] = X_processed.columns.tolist()
        
        return X_processed, preprocessing_info
    
    def create_base_models(self) -> Dict[str, Any]:
        """Create base models optimized for small datasets."""
        models = {}
        
        # 1. L1 Regularized Logistic Regression (feature selection built-in)
        models['logistic_l1'] = {
            'model': LogisticRegression(
                penalty='l1', C=0.2, solver='liblinear',
                class_weight='balanced', max_iter=1000, random_state=42
            ),
            'features': 'selected',  # Use selected features
            'scale': True
        }
        
        # 2. L2 Regularized Logistic Regression
        models['logistic_l2'] = {
            'model': LogisticRegression(
                penalty='l2', C=0.5, solver='lbfgs',
                class_weight='balanced', max_iter=1000, random_state=42
            ),
            'features': 'selected',
            'scale': True
        }
        
        # 3. LightGBM with heavy regularization
        models['lgb_regularized'] = {
            'model': lgb.LGBMClassifier(
                n_estimators=50,
                num_leaves=8,  # Very shallow
                max_depth=3,
                learning_rate=0.05,
                feature_fraction=0.7,
                bagging_fraction=0.7,
                bagging_freq=5,
                lambda_l1=1.0,
                lambda_l2=2.0,
                min_data_in_leaf=20,
                min_gain_to_split=0.1,
                class_weight='balanced',
                random_state=42,
                verbosity=-1
            ),
            'features': 'selected',  # Changed from 'all' to 'selected' to match other models
            'scale': False
        }
        
        # 4. Random Forest (constrained)
        models['rf_constrained'] = {
            'model': RandomForestClassifier(
                n_estimators=100,
                max_depth=3,
                min_samples_split=25,
                min_samples_leaf=15,
                max_features='sqrt',
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'features': 'selected',
            'scale': False
        }
        
        # 5. Domain-based model
        models['domain_rules'] = {
            'model': DomainExpertModel(),
            'features': 'domain',
            'scale': False
        }
        
        return models
    
    def train_single_model(self, model_dict: Dict, X_train: pd.DataFrame, 
                          y_train: pd.Series, model_name: str) -> Tuple[Any, Any, List[str]]:
        """Train a single model with appropriate preprocessing."""
        model = model_dict['model']
        feature_type = model_dict['features']
        needs_scaling = model_dict['scale']
        
        # Select appropriate features
        if feature_type == 'selected':
            features = self.selected_features.get('main', self.selected_features.get('all', []))
        elif feature_type == 'domain':
            # Use composite and interpretable features for domain model
            features = ['overall_rating', 'distance_suitability', 'speed_trajectory',
                       'class_rise_drop', 'pace_suitability']
            features = [f for f in features if f in X_train.columns]
        else:  # 'all'
            features = [col for col in X_train.columns if col not in ['race', 'horse_name']]
        
        # Select features
        X_subset = X_train[features].copy()
        
        # Store original feature list
        original_features = X_subset.columns.tolist()
        
        # Apply preprocessing
        X_processed, preprocessing_info = self.preprocess_features(
            X_subset, model_name, fit=True
        )
        
        # Scale if needed
        scaler = None
        X_scaled = X_processed  # Default to no scaling
        
        if needs_scaling and model_name not in ['lgb_regularized', 'rf_constrained', 'domain_rules']:
            scaler = RobustScaler()  # Better for small data with outliers
            X_scaled = scaler.fit_transform(X_processed)
        elif model_name not in ['domain_rules']:
            # For non-domain models that don't need scaling, convert to array
            X_scaled = X_processed.values
        
        # Train model
        try:
            model.fit(X_scaled, y_train)
        except Exception as e:
            logger.error(f"Error training {model_name}")
            logger.error(f"X_scaled shape: {X_scaled.shape if hasattr(X_scaled, 'shape') else 'DataFrame'}")
            logger.error(f"Preprocessing info: {preprocessing_info}")
            raise
        
        # Store all preprocessing info
        if scaler is None:
           scaler = PreprocessingHolder()
        
        scaler._preprocessing = preprocessing_info
        scaler._original_features = original_features
        
        # Return the original features list (what we expect as input)
        return model, scaler, original_features
    
    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series,
                      scaler: Any = None, features: List[str] = None) -> Dict[str, float]:
        """Evaluate model performance with multiple metrics."""
        # Check model type
        model_type = type(model).__name__
        
        # Select same features as used in training
        if features is not None:
            available_features = [f for f in features if f in X_test.columns]
            X_test = X_test[available_features].copy()
            
            # Log missing features
            missing_features = [f for f in features if f not in X_test.columns]
            if missing_features:
                logger.warning(f"Missing features in test set: {missing_features}")
        
        # Get preprocessing info and model name
        preprocessing_info = scaler._preprocessing if scaler and hasattr(scaler, '_preprocessing') else {}
        
        # Determine model name from type
        if 'LGBM' in model_type:
            model_name = 'lgb_regularized'
        elif 'RandomForest' in model_type:
            model_name = 'rf_constrained'
        elif model_type == 'DomainExpertModel':
            model_name = 'domain_rules'
        elif 'Logistic' in model_type:
            model_name = 'logistic_l1'  # Could be l1 or l2, doesn't matter for preprocessing
        else:
            model_name = 'other'
        
        # Log preprocessing details for debugging
        if model_name == 'lgb_regularized':
            logger.info(f"Evaluating {model_name} - Input features: {len(X_test.columns)}")
            logger.info(f"Evaluating {model_name} - Expected final features: {len(preprocessing_info.get('final_columns', []))}")
        
        # Apply same preprocessing as training
        X_processed, _ = self.preprocess_features(
            X_test, model_name, preprocessing_info=preprocessing_info, fit=False
        )
        
        if model_name == 'lgb_regularized':
            logger.info(f"After preprocessing - Features: {len(X_processed.columns)}")
        
        # Apply scaling if needed
        if scaler is not None and hasattr(scaler, 'transform') and model_name not in ['domain_rules']:
            X_scaled = scaler.transform(X_processed)
        else:
            # For DomainExpertModel, keep as DataFrame; for others, convert to array
            X_scaled = X_processed if model_name == 'domain_rules' else X_processed.values
        
        # Get predictions
        try:
            if model_name == 'lgb_regularized':
                logger.info(f"Predicting with {model_name} - Input shape: {X_scaled.shape}")
            y_pred_proba = model.predict_proba(X_scaled)[:, 1]
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            logger.error(f"Model type: {model_type}")
            logger.error(f"Model name: {model_name}")
            logger.error(f"Expected features from preprocessing: {preprocessing_info.get('final_columns', 'Unknown')}")
            logger.error(f"Expected feature count: {len(preprocessing_info.get('final_columns', []))}")
            logger.error(f"Input shape: {X_scaled.shape if hasattr(X_scaled, 'shape') else 'DataFrame'}")
            if hasattr(X_processed, 'columns'):
                logger.error(f"Processed columns ({len(X_processed.columns)}): {X_processed.columns.tolist()}")
            raise
        
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'log_loss': log_loss(y_test, y_pred_proba),
            'brier_score': brier_score_loss(y_test, y_pred_proba),
            'accuracy': np.mean(y_pred == y_test),
            'win_rate': np.mean(y_test),  # Base rate for comparison
            'pred_win_rate': np.mean(y_pred)
        }
        
        return metrics
    
    def train_all_models(self, train_df: pd.DataFrame, 
                        target_col: str = 'target_win') -> Dict[str, Any]:
        """Train all models with cross-validation."""
        logger.info("Training all models...")
        
        # Prepare features and target
        feature_cols = [col for col in train_df.columns 
                       if col not in ['race', 'horse_name', 'post_position', 
                                     'finish_position', 'target_win', 'target_top3',
                                     'morn_line_odds_if_available']]
        
        X = train_df[feature_cols]
        y = train_df[target_col]
        races = train_df['race']
        
        # Feature selection on full dataset
        self.selected_features['all'] = self.select_features(X, y, max_features=15)
        
        # Create models
        base_models = self.create_base_models()
        
        # Log feature selection results
        logger.info(f"Feature selection complete. Selected {len(self.selected_features['all'])} features.")
        logger.info(f"Selected features include {sum(1 for f in self.selected_features['all'] if f in ['pace_style', 'speed_trajectory', 'class_rise_drop'])} categorical features")
        
        # Cross-validation setup
        gkf = GroupKFold(n_splits=3)
        
        # Train and evaluate each model
        for model_name, model_dict in base_models.items():
            logger.info(f"Training {model_name}...")
            
            cv_scores = []
            trained_models = []
            
            for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=races)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Train model
                model, scaler, used_features = self.train_single_model(
                    model_dict, X_train, y_train, model_name
                )
                
                # Debug logging for problematic model
                if fold == 0 and model_name == 'lgb_regularized':  # Log only for first fold of problematic model
                    logger.info(f"{model_name} - Original features: {len(used_features)} - {used_features[:5]}...")  # Show first 5
                    if hasattr(scaler, '_preprocessing'):
                        prep_info = scaler._preprocessing
                        logger.info(f"{model_name} - Categorical cols: {prep_info.get('categorical_cols', [])}")
                        logger.info(f"{model_name} - Encoders: {list(prep_info.get('encoders', {}).keys())}")
                        logger.info(f"{model_name} - Final feature count after preprocessing: {len(prep_info.get('final_columns', []))}")
                
                # Evaluate
                scores = self.evaluate_model(model, X_val, y_val, scaler, used_features)
                cv_scores.append(scores)
                
                # Store trained model with all preprocessing info
                trained_models.append({
                    'model': model,
                    'scaler': scaler,
                    'features': used_features  # Original features before preprocessing
                })
            
            # Average CV scores
            avg_scores = {
                metric: np.mean([s[metric] for s in cv_scores])
                for metric in cv_scores[0].keys()
            }
            
            # Store results
            self.models[model_name] = trained_models
            self.model_performance[model_name] = avg_scores
            
            logger.info(f"{model_name} - ROC-AUC: {avg_scores['roc_auc']:.4f}, "
                       f"Log Loss: {avg_scores['log_loss']:.4f}")
        
        # Calculate ensemble weights based on performance
        self._calculate_ensemble_weights()
        
        return self.models
    
    def _calculate_ensemble_weights(self):
        """Calculate optimal ensemble weights based on log loss."""
        log_losses = {name: perf['log_loss'] 
                     for name, perf in self.model_performance.items()}
        
        # Convert to weights (lower log loss = higher weight)
        # Use softmax-like transformation
        scores = np.array(list(log_losses.values()))
        weights = np.exp(-scores * 5)  # Scale factor
        weights = weights / weights.sum()
        
        self.ensemble_weights = dict(zip(log_losses.keys(), weights))
        
        logger.info("Ensemble weights:")
        for model, weight in sorted(self.ensemble_weights.items(), 
                                   key=lambda x: x[1], reverse=True):
            logger.info(f"  {model}: {weight:.3f}")
    
    def predict_race(self, race_df: pd.DataFrame, 
                    method: str = 'weighted_average') -> pd.DataFrame:
        """Generate predictions for a single race."""
        # Prepare features
        feature_cols = [col for col in race_df.columns 
                       if col not in ['race', 'horse_name', 'post_position', 
                                     'finish_position', 'target_win', 'target_top3',
                                     'morn_line_odds_if_available']]
        
        X_race = race_df[feature_cols]
        
        # Get predictions from each model
        all_predictions = []
        
        for model_name, model_list in self.models.items():
            model_preds = []
            
            # Average predictions across CV folds
            for fold_data in model_list:
                model = fold_data['model']
                scaler = fold_data['scaler']
                features = fold_data['features']
                
                # Prepare features - use the same features as training
                if features:
                    available_features = [f for f in features if f in X_race.columns]
                    X_subset = X_race[available_features].copy()
                    
                    # Get preprocessing info
                    preprocessing_info = scaler._preprocessing if scaler and hasattr(scaler, '_preprocessing') else {}
                    
                    # Determine model name from type
                    model_type = type(model).__name__
                    if 'LGBM' in model_type:
                        model_name_pred = 'lgb_regularized'
                    elif 'RandomForest' in model_type:
                        model_name_pred = 'rf_constrained'
                    elif model_type == 'DomainExpertModel':
                        model_name_pred = 'domain_rules'
                    elif 'Logistic' in model_type:
                        model_name_pred = 'logistic_l1'
                    else:
                        model_name_pred = 'other'
                    
                    # Apply preprocessing
                    X_subset, _ = self.preprocess_features(
                        X_subset, model_name_pred, 
                        preprocessing_info=preprocessing_info, 
                        fit=False
                    )
                else:
                    # Fallback if features not stored
                    if model_name in ['logistic_l1', 'logistic_l2', 'rf_constrained']:
                        X_subset = X_race[self.selected_features.get('all', feature_cols)]
                    elif model_name == 'domain_rules':
                        domain_features = ['overall_rating', 'distance_suitability', 
                                         'speed_trajectory', 'class_rise_drop', 'pace_suitability']
                        X_subset = X_race[[f for f in domain_features if f in X_race.columns]]
                    else:
                        X_subset = X_race
                
                # Scale if needed
                if scaler is not None and hasattr(scaler, 'transform') and model_name_pred not in ['domain_rules']:
                    X_scaled = scaler.transform(X_subset)
                else:
                    # For DomainExpertModel, keep as DataFrame; for others, convert to array
                    X_scaled = X_subset if model_name_pred == 'domain_rules' else X_subset.values
                
                # Predict
                try:
                    pred = model.predict_proba(X_scaled)[:, 1]
                    model_preds.append(pred)
                except Exception as e:
                    logger.warning(f"Error predicting with {model_name}: {e}")
                    continue
            
            if model_preds:
                # Average across folds
                avg_pred = np.mean(model_preds, axis=0)
                all_predictions.append({
                    'model': model_name,
                    'predictions': avg_pred,
                    'weight': self.ensemble_weights.get(model_name, 0)
                })
        
        # Combine predictions
        if method == 'weighted_average':
            final_pred = np.zeros(len(race_df))
            total_weight = 0
            
            for pred_data in all_predictions:
                weight = pred_data['weight']
                final_pred += weight * pred_data['predictions']
                total_weight += weight
            
            final_pred = final_pred / total_weight if total_weight > 0 else final_pred
            
        elif method == 'rank_average':
            # Average ranks instead of probabilities
            ranks = []
            for pred_data in all_predictions:
                rank = pd.Series(pred_data['predictions']).rank(ascending=False)
                ranks.append(rank)
            
            avg_rank = np.mean(ranks, axis=0)
            # Convert back to probability-like scores
            final_pred = 1 - (avg_rank - 1) / (len(race_df) - 1)
        
        else:  # simple average
            final_pred = np.mean([p['predictions'] for p in all_predictions], axis=0)
        
        # Create results DataFrame
        results = race_df[['post_position', 'horse_name', 'morn_line_odds_if_available']].copy()
        results['win_probability'] = final_pred
        results['implied_odds'] = 1 / final_pred
        results['rank'] = results['win_probability'].rank(ascending=False)
        
        # Add value flag
        results['overlay'] = (results['morn_line_odds_if_available'] / results['implied_odds']) - 1
        results['bet_value'] = results['overlay'] > 0.2  # 20% overlay threshold
        
        # Sort by probability
        results = results.sort_values('win_probability', ascending=False)
        
        return results
    
    def save_model(self, filepath: Path = None):
        """Save trained models and configuration."""
        if filepath is None:
            filepath = self.processed_data_dir / f"horse_model_{datetime.now():%Y%m%d_%H%M%S}.pkl"
        
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'selected_features': self.selected_features,
            'model_performance': self.model_performance,
            'ensemble_weights': self.ensemble_weights,
            'feature_groups': self.feature_groups
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: Path):
        """Load trained models and configuration."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.selected_features = model_data['selected_features']
        self.model_performance = model_data['model_performance']
        self.ensemble_weights = model_data['ensemble_weights']
        self.feature_groups = model_data.get('feature_groups', self.feature_groups)
        
        logger.info(f"Model loaded from {filepath}")


class DomainExpertModel:
    """Rule-based model incorporating domain expertise."""
    
    def __init__(self):
        self.feature_weights = {
            'overall_rating': 0.30,
            'distance_suitability': 0.25,
            'speed_trajectory': 0.20,
            'class_rise_drop': 0.15,
            'pace_suitability': 0.10
        }
        self.feature_names = list(self.feature_weights.keys())
        self.classes_ = np.array([0, 1])  # For sklearn compatibility
    
    def fit(self, X, y):
        """Store feature names if X is a DataFrame."""
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        elif hasattr(X, 'shape') and hasattr(self, 'feature_names'):
            # If X is array and we already have feature names, keep them
            pass
        return self
    
    def predict_proba(self, X):
        """Generate probabilities based on weighted rules."""
        # Initialize scores
        scores = np.zeros(len(X))
        
        # Convert to DataFrame if numpy array
        if isinstance(X, np.ndarray):
            # Use stored feature names
            X_df = pd.DataFrame(X, columns=self.feature_names)
        elif isinstance(X, pd.DataFrame):
            X_df = X
        else:
            X_df = pd.DataFrame(X)
        
        # Overall rating (normalized to 0-1)
        if 'overall_rating' in X_df.columns:
            rating_values = pd.to_numeric(X_df['overall_rating'], errors='coerce').fillna(50)
            rating_score = rating_values / 100
            scores += self.feature_weights.get('overall_rating', 0) * rating_score
        
        # Distance suitability
        if 'distance_suitability' in X_df.columns:
            dist_values = pd.to_numeric(X_df['distance_suitability'], errors='coerce').fillna(50)
            dist_score = dist_values / 100
            scores += self.feature_weights.get('distance_suitability', 0) * dist_score
        
        # Speed trajectory bonus
        if 'speed_trajectory' in X_df.columns:
            trajectory_map = {
                'improving': 1.0,
                'stable': 0.7,
                'inconsistent': 0.5,
                'declining': 0.3,
                'unknown': 0.5,
                'insufficient_data': 0.5
            }
            # Handle missing or invalid values
            trajectory_score = X_df['speed_trajectory'].map(trajectory_map).fillna(0.5)
            scores += self.feature_weights.get('speed_trajectory', 0) * trajectory_score
        
        # Class drop advantage
        if 'class_rise_drop' in X_df.columns:
            class_map = {
                'major_drop': 1.0,
                'minor_drop': 0.8,
                'similar': 0.6,
                'minor_rise': 0.4,
                'major_rise': 0.2,
                'unknown': 0.5
            }
            class_score = X_df['class_rise_drop'].map(class_map).fillna(0.5)
            scores += self.feature_weights.get('class_rise_drop', 0) * class_score
        
        # Pace suitability
        if 'pace_suitability' in X_df.columns:
            pace_values = pd.to_numeric(X_df['pace_suitability'], errors='coerce').fillna(50)
            pace_score = pace_values / 100
            scores += self.feature_weights.get('pace_suitability', 0) * pace_score
        
        # Normalize scores to probabilities
        scores = np.clip(scores, 0, 1)
        
        # Add some noise to avoid identical predictions
        np.random.seed(42)  # For reproducibility
        scores += np.random.normal(0, 0.01, len(scores))
        scores = np.clip(scores, 0.01, 0.99)
        
        # Return in sklearn format [prob_negative, prob_positive]
        return np.column_stack([1 - scores, scores])
    
    def predict(self, X):
        """Return binary predictions."""
        probs = self.predict_proba(X)
        return (probs[:, 1] > 0.5).astype(int)


def run_training_pipeline():
    """Run the complete training pipeline."""
    logger.info("🏇 Starting Horse Racing Model Training Pipeline")
    logger.info("=" * 60)
    
    # Initialize model
    model = IntegratedHorseRacingModel()
    
    # Load engineered features
    features_df = model.load_engineered_features()
    logger.info(f"Loaded features: {features_df.shape}")
    
    # Merge with results
    train_df = model.merge_with_results(features_df)
    logger.info(f"Training data shape: {train_df.shape}")
    logger.info(f"Win rate: {train_df['target_win'].mean():.3f}")
    
    # Train models
    model.train_all_models(train_df, target_col='target_win')
    
    # Display model performance
    logger.info("\n📊 Model Performance Summary:")
    logger.info("-" * 40)
    for model_name, perf in sorted(model.model_performance.items(), 
                                  key=lambda x: x[1]['roc_auc'], reverse=True):
        logger.info(f"{model_name:20s} ROC-AUC: {perf['roc_auc']:.4f} "
                   f"Log Loss: {perf['log_loss']:.4f}")
    
    # Save model
    model.save_model()
    
    return model


def run_prediction_pipeline(race_num: int = None):
    """Run predictions for a specific race."""
    logger.info("🏇 Starting Horse Racing Prediction Pipeline")
    logger.info("=" * 60)
    
    # Initialize model
    model = IntegratedHorseRacingModel()
    
    # Load latest saved model
    model_files = list(PROCESSED_DATA_DIR.glob("horse_model_*.pkl"))
    if model_files:
        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
        model.load_model(latest_model)
        logger.info(f"Loaded model: {latest_model.name}")
    else:
        logger.error("No saved model found. Please run training first.")
        return None
    
    # Load features for prediction
    features_df = model.load_engineered_features()
    features_df = model.merge_with_results(features_df)
    
    # Select race
    if race_num is None:
        race_num = features_df['race'].max()
    
    race_df = features_df[features_df['race'] == race_num].copy()
    logger.info(f"Predicting Race {race_num} with {len(race_df)} horses")
    
    # Generate predictions
    predictions = model.predict_race(race_df, method='weighted_average')
    
    # Display results
    logger.info("\n🎯 RACE PREDICTIONS")
    logger.info("=" * 80)
    display_cols = ['rank', 'post_position', 'horse_name', 'win_probability', 
                   'implied_odds', 'morn_line_odds_if_available', 'overlay', 'bet_value']
    
    print(predictions[display_cols].to_string(index=False, float_format='%.3f'))
    
    # Betting recommendations
    value_bets = predictions[predictions['bet_value']]
    if not value_bets.empty:
        logger.info("\n💰 VALUE BETS (20%+ overlay):")
        logger.info("-" * 40)
        for _, horse in value_bets.iterrows():
            logger.info(f"#{horse['post_position']} {horse['horse_name']} - "
                       f"Win Prob: {horse['win_probability']:.1%} "
                       f"(Overlay: {horse['overlay']:.1%})")
    
    return predictions


# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Horse Racing Predictive Model')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--predict', action='store_true', help='Make predictions')
    parser.add_argument('--race', type=int, help='Race number to predict')
    
    args = parser.parse_args()
    
    if args.train:
        model = run_training_pipeline()
    elif args.predict:
        predictions = run_prediction_pipeline(args.race)
    else:
        # Default: train then predict
        model = run_training_pipeline()
        predictions = run_prediction_pipeline()