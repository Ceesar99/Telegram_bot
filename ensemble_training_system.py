#!/usr/bin/env python3
"""
🎯 ENSEMBLE TRAINING SYSTEM - PRODUCTION READY
Advanced ensemble model training with XGBoost, LightGBM, CatBoost, and meta-learning
Designed for >80% accuracy in financial market prediction
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna
import logging
import json
import pickle
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import joblib
import warnings
warnings.filterwarnings('ignore')

from config import DATABASE_CONFIG, TIMEZONE
from enhanced_lstm_trainer import EnhancedLSTMTrainer, TrainingConfig
from enhanced_feature_engine import EnhancedFeatureEngine, FeatureConfig
from real_market_data_collector import RealMarketDataCollector
from advanced_data_validator import MarketDataValidator

@dataclass
class EnsembleConfig:
    """Ensemble training configuration"""
    use_xgboost: bool = True
    use_lightgbm: bool = True
    use_catboost: bool = True
    use_random_forest: bool = True
    use_lstm: bool = True
    use_stacking: bool = True
    use_voting: bool = True
    meta_learner: str = 'logistic'  # 'logistic', 'xgboost', 'neural'
    cv_folds: int = 5
    optimize_hyperparams: bool = True
    n_trials: int = 100
    ensemble_weights: Dict[str, float] = field(default_factory=dict)

@dataclass
class ModelPerformance:
    """Individual model performance metrics"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: float
    training_time: float
    prediction_time: float
    feature_importance: Dict[str, float] = field(default_factory=dict)

@dataclass
class EnsembleResults:
    """Ensemble training results"""
    individual_performances: List[ModelPerformance]
    ensemble_accuracy: float
    ensemble_precision: float
    ensemble_recall: float
    ensemble_f1: float
    ensemble_auc: float
    best_model_name: str
    ensemble_weights: Dict[str, float]
    meta_model: Any
    trained_models: Dict[str, Any]
    feature_importance: Dict[str, float]
    training_time: float

class XGBoostOptimizer:
    """XGBoost hyperparameter optimizer"""
    
    def __init__(self):
        self.logger = logging.getLogger('XGBoostOptimizer')
    
    def optimize(self, X_train: np.ndarray, y_train: np.ndarray, 
                X_val: np.ndarray, y_val: np.ndarray, n_trials: int = 50) -> Dict[str, Any]:
        """Optimize XGBoost hyperparameters"""
        
        def objective(trial):
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear']),
                'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
                'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'random_state': 42,
                'verbosity': 0
            }
            
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            
            return accuracy
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        return study.best_params

class LightGBMOptimizer:
    """LightGBM hyperparameter optimizer"""
    
    def __init__(self):
        self.logger = logging.getLogger('LightGBMOptimizer')
    
    def optimize(self, X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray, n_trials: int = 50) -> Dict[str, Any]:
        """Optimize LightGBM hyperparameters"""
        
        def objective(trial):
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'random_state': 42,
                'verbosity': -1
            }
            
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(10)])
            
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            
            return accuracy
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        return study.best_params

class CatBoostOptimizer:
    """CatBoost hyperparameter optimizer"""
    
    def __init__(self):
        self.logger = logging.getLogger('CatBoostOptimizer')
    
    def optimize(self, X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray, n_trials: int = 50) -> Dict[str, Any]:
        """Optimize CatBoost hyperparameters"""
        
        def objective(trial):
            params = {
                'objective': 'Logloss',
                'eval_metric': 'AUC',
                'iterations': trial.suggest_int('iterations', 100, 1000),
                'depth': trial.suggest_int('depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
                'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
                'border_count': trial.suggest_int('border_count', 1, 255),
                'random_state': 42,
                'verbose': False
            }
            
            model = cb.CatBoostClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            
            return accuracy
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        return study.best_params

class MetaLearner:
    """Meta-learner for ensemble combination"""
    
    def __init__(self, learner_type: str = 'logistic'):
        self.learner_type = learner_type
        self.model = None
        self.logger = logging.getLogger('MetaLearner')
    
    def build_meta_model(self, input_dim: int) -> Any:
        """Build meta-learning model"""
        
        if self.learner_type == 'logistic':
            self.model = LogisticRegression(
                C=1.0,
                solver='liblinear',
                random_state=42,
                max_iter=1000
            )
        
        elif self.learner_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
                verbosity=0
            )
        
        elif self.learner_type == 'neural':
            self.model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            
            self.model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
        
        return self.model
    
    def train(self, meta_features: np.ndarray, targets: np.ndarray):
        """Train meta-learner"""
        
        self.build_meta_model(meta_features.shape[1])
        
        if self.learner_type == 'neural':
            self.model.fit(
                meta_features, targets,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                verbose=0
            )
        else:
            self.model.fit(meta_features, targets)
    
    def predict(self, meta_features: np.ndarray) -> np.ndarray:
        """Make predictions with meta-learner"""
        
        if self.learner_type == 'neural':
            return (self.model.predict(meta_features) > 0.5).astype(int).flatten()
        else:
            return self.model.predict(meta_features)
    
    def predict_proba(self, meta_features: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        
        if self.learner_type == 'neural':
            return self.model.predict(meta_features).flatten()
        else:
            return self.model.predict_proba(meta_features)[:, 1]

class EnsembleTrainingSystem:
    """Comprehensive ensemble training system"""
    
    def __init__(self, config: EnsembleConfig = None):
        self.config = config or EnsembleConfig()
        self.logger = logging.getLogger('EnsembleTrainingSystem')
        
        # Initialize optimizers
        self.xgb_optimizer = XGBoostOptimizer()
        self.lgb_optimizer = LightGBMOptimizer()
        self.cb_optimizer = CatBoostOptimizer()
        
        # Data components
        self.data_collector = RealMarketDataCollector()
        self.feature_engineer = EnhancedFeatureEngine()
        self.data_validator = MarketDataValidator()
        self.lstm_trainer = EnhancedLSTMTrainer()
        
        # Models storage
        self.trained_models = {}
        self.model_performances = {}
        self.ensemble_model = None
        self.meta_learner = MetaLearner(self.config.meta_learner)
        
        # Ensure directories exist
        os.makedirs('/workspace/models/ensemble', exist_ok=True)
        os.makedirs('/workspace/logs/ensemble', exist_ok=True)
    
    def prepare_ensemble_data(self, 
                            symbols: List[str] = None,
                            timeframes: List[str] = None,
                            start_date: datetime = None,
                            end_date: datetime = None) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Prepare data for ensemble training"""
        
        self.logger.info("Preparing ensemble training data...")
        
        # Use LSTM trainer's data preparation (already comprehensive)
        X_sequences, y_sequences, features_df = self.lstm_trainer.prepare_training_data(
            symbols=symbols,
            timeframes=timeframes,
            start_date=start_date,
            end_date=end_date
        )
        
        # For tree-based models, we need flattened features
        # Take the last sequence step for each sample
        X_flat = X_sequences[:, -1, :]  # Shape: (samples, features)
        
        self.logger.info(f"Prepared ensemble data: {X_flat.shape} flat features, {y_sequences.shape} targets")
        
        return X_flat, y_sequences, features_df
    
    def train_individual_models(self, X_train: np.ndarray, y_train: np.ndarray,
                              X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Train individual models in the ensemble"""
        
        models = {}
        performances = []
        
        # XGBoost
        if self.config.use_xgboost:
            self.logger.info("Training XGBoost model...")
            start_time = datetime.now()
            
            if self.config.optimize_hyperparams:
                best_params = self.xgb_optimizer.optimize(X_train, y_train, X_val, y_val, 
                                                        self.config.n_trials // 4)
            else:
                best_params = {
                    'n_estimators': 500,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42
                }
            
            xgb_model = xgb.XGBClassifier(**best_params)
            xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
            # Evaluate
            y_pred = xgb_model.predict(X_val)
            y_pred_proba = xgb_model.predict_proba(X_val)[:, 1]
            
            performance = ModelPerformance(
                model_name='XGBoost',
                accuracy=accuracy_score(y_val, y_pred),
                precision=precision_score(y_val, y_pred, zero_division=0),
                recall=recall_score(y_val, y_pred, zero_division=0),
                f1_score=f1_score(y_val, y_pred, zero_division=0),
                auc_score=roc_auc_score(y_val, y_pred_proba),
                training_time=(datetime.now() - start_time).total_seconds(),
                prediction_time=0.0,
                feature_importance=dict(zip(range(len(xgb_model.feature_importances_)), 
                                          xgb_model.feature_importances_))
            )
            
            models['xgboost'] = xgb_model
            performances.append(performance)
            self.logger.info(f"XGBoost accuracy: {performance.accuracy:.4f}")
        
        # LightGBM
        if self.config.use_lightgbm:
            self.logger.info("Training LightGBM model...")
            start_time = datetime.now()
            
            if self.config.optimize_hyperparams:
                best_params = self.lgb_optimizer.optimize(X_train, y_train, X_val, y_val,
                                                        self.config.n_trials // 4)
            else:
                best_params = {
                    'n_estimators': 500,
                    'num_leaves': 31,
                    'learning_rate': 0.1,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'random_state': 42,
                    'verbosity': -1
                }
            
            lgb_model = lgb.LGBMClassifier(**best_params)
            lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                         callbacks=[lgb.early_stopping(50)])
            
            # Evaluate
            y_pred = lgb_model.predict(X_val)
            y_pred_proba = lgb_model.predict_proba(X_val)[:, 1]
            
            performance = ModelPerformance(
                model_name='LightGBM',
                accuracy=accuracy_score(y_val, y_pred),
                precision=precision_score(y_val, y_pred, zero_division=0),
                recall=recall_score(y_val, y_pred, zero_division=0),
                f1_score=f1_score(y_val, y_pred, zero_division=0),
                auc_score=roc_auc_score(y_val, y_pred_proba),
                training_time=(datetime.now() - start_time).total_seconds(),
                prediction_time=0.0,
                feature_importance=dict(zip(range(len(lgb_model.feature_importances_)), 
                                          lgb_model.feature_importances_))
            )
            
            models['lightgbm'] = lgb_model
            performances.append(performance)
            self.logger.info(f"LightGBM accuracy: {performance.accuracy:.4f}")
        
        # CatBoost
        if self.config.use_catboost:
            self.logger.info("Training CatBoost model...")
            start_time = datetime.now()
            
            if self.config.optimize_hyperparams:
                best_params = self.cb_optimizer.optimize(X_train, y_train, X_val, y_val,
                                                       self.config.n_trials // 4)
            else:
                best_params = {
                    'iterations': 500,
                    'depth': 6,
                    'learning_rate': 0.1,
                    'random_state': 42,
                    'verbose': False
                }
            
            cb_model = cb.CatBoostClassifier(**best_params)
            cb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
            # Evaluate
            y_pred = cb_model.predict(X_val)
            y_pred_proba = cb_model.predict_proba(X_val)[:, 1]
            
            performance = ModelPerformance(
                model_name='CatBoost',
                accuracy=accuracy_score(y_val, y_pred),
                precision=precision_score(y_val, y_pred, zero_division=0),
                recall=recall_score(y_val, y_pred, zero_division=0),
                f1_score=f1_score(y_val, y_pred, zero_division=0),
                auc_score=roc_auc_score(y_val, y_pred_proba),
                training_time=(datetime.now() - start_time).total_seconds(),
                prediction_time=0.0
            )
            
            models['catboost'] = cb_model
            performances.append(performance)
            self.logger.info(f"CatBoost accuracy: {performance.accuracy:.4f}")
        
        # Random Forest
        if self.config.use_random_forest:
            self.logger.info("Training Random Forest model...")
            start_time = datetime.now()
            
            rf_model = RandomForestClassifier(
                n_estimators=500,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = rf_model.predict(X_val)
            y_pred_proba = rf_model.predict_proba(X_val)[:, 1]
            
            performance = ModelPerformance(
                model_name='RandomForest',
                accuracy=accuracy_score(y_val, y_pred),
                precision=precision_score(y_val, y_pred, zero_division=0),
                recall=recall_score(y_val, y_pred, zero_division=0),
                f1_score=f1_score(y_val, y_pred, zero_division=0),
                auc_score=roc_auc_score(y_val, y_pred_proba),
                training_time=(datetime.now() - start_time).total_seconds(),
                prediction_time=0.0,
                feature_importance=dict(zip(range(len(rf_model.feature_importances_)), 
                                          rf_model.feature_importances_))
            )
            
            models['random_forest'] = rf_model
            performances.append(performance)
            self.logger.info(f"Random Forest accuracy: {performance.accuracy:.4f}")
        
        self.trained_models = models
        return models, performances
    
    def train_lstm_model(self, X_sequences: np.ndarray, y: np.ndarray) -> Tuple[Any, ModelPerformance]:
        """Train LSTM model for ensemble"""
        
        if not self.config.use_lstm:
            return None, None
        
        self.logger.info("Training LSTM model for ensemble...")
        start_time = datetime.now()
        
        # Split data
        split_idx = int(len(X_sequences) * 0.8)
        X_train, X_val = X_sequences[:split_idx], X_sequences[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Train LSTM
        results = self.lstm_trainer.train_model(X_train, y_train, 'ensemble', 'multi', use_optuna=False)
        
        # Evaluate on validation set
        y_pred_proba = results.model.predict(X_val)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        performance = ModelPerformance(
            model_name='LSTM',
            accuracy=accuracy_score(y_val, y_pred),
            precision=precision_score(y_val, y_pred, zero_division=0),
            recall=recall_score(y_val, y_pred, zero_division=0),
            f1_score=f1_score(y_val, y_pred, zero_division=0),
            auc_score=roc_auc_score(y_val, y_pred_proba.flatten()),
            training_time=results.training_time,
            prediction_time=0.0
        )
        
        self.logger.info(f"LSTM accuracy: {performance.accuracy:.4f}")
        
        return results.model, performance
    
    def create_meta_features(self, models: Dict[str, Any], lstm_model: Any,
                           X_flat: np.ndarray, X_sequences: np.ndarray) -> np.ndarray:
        """Create meta-features from base model predictions"""
        
        meta_features = []
        
        # Tree-based model predictions
        for name, model in models.items():
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_flat)[:, 1]
            else:
                proba = model.predict(X_flat)
            meta_features.append(proba)
        
        # LSTM predictions
        if lstm_model is not None:
            lstm_proba = lstm_model.predict(X_sequences).flatten()
            meta_features.append(lstm_proba)
        
        return np.column_stack(meta_features)
    
    def train_ensemble(self, X_flat: np.ndarray, X_sequences: np.ndarray, y: np.ndarray) -> EnsembleResults:
        """Train complete ensemble system"""
        
        self.logger.info("Starting ensemble training...")
        overall_start_time = datetime.now()
        
        # Split data
        split_idx = int(len(X_flat) * 0.8)
        X_flat_train, X_flat_val = X_flat[:split_idx], X_flat[split_idx:]
        X_seq_train, X_seq_val = X_sequences[:split_idx], X_sequences[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Train individual models
        models, performances = self.train_individual_models(X_flat_train, y_train, X_flat_val, y_val)
        
        # Train LSTM model
        lstm_model, lstm_performance = self.train_lstm_model(X_sequences, y)
        if lstm_performance:
            performances.append(lstm_performance)
        
        # Create meta-features for meta-learning
        meta_features_train = self.create_meta_features(models, lstm_model, X_flat_train, X_seq_train)
        meta_features_val = self.create_meta_features(models, lstm_model, X_flat_val, X_seq_val)
        
        # Train meta-learner
        self.logger.info("Training meta-learner...")
        self.meta_learner.train(meta_features_train, y_train)
        
        # Evaluate ensemble
        ensemble_pred = self.meta_learner.predict(meta_features_val)
        ensemble_proba = self.meta_learner.predict_proba(meta_features_val)
        
        ensemble_accuracy = accuracy_score(y_val, ensemble_pred)
        ensemble_precision = precision_score(y_val, ensemble_pred, zero_division=0)
        ensemble_recall = recall_score(y_val, ensemble_pred, zero_division=0)
        ensemble_f1 = f1_score(y_val, ensemble_pred, zero_division=0)
        ensemble_auc = roc_auc_score(y_val, ensemble_proba)
        
        # Find best individual model
        best_model = max(performances, key=lambda x: x.accuracy)
        
        # Calculate ensemble weights based on performance
        total_accuracy = sum(p.accuracy for p in performances)
        ensemble_weights = {p.model_name: p.accuracy / total_accuracy for p in performances}
        
        # Calculate feature importance
        feature_importance = {}
        for perf in performances:
            if perf.feature_importance:
                for feat, imp in perf.feature_importance.items():
                    if feat not in feature_importance:
                        feature_importance[feat] = 0
                    feature_importance[feat] += imp * ensemble_weights.get(perf.model_name, 0)
        
        training_time = (datetime.now() - overall_start_time).total_seconds()
        
        results = EnsembleResults(
            individual_performances=performances,
            ensemble_accuracy=ensemble_accuracy,
            ensemble_precision=ensemble_precision,
            ensemble_recall=ensemble_recall,
            ensemble_f1=ensemble_f1,
            ensemble_auc=ensemble_auc,
            best_model_name=best_model.model_name,
            ensemble_weights=ensemble_weights,
            meta_model=self.meta_learner,
            trained_models=models,
            feature_importance=feature_importance,
            training_time=training_time
        )
        
        # Save models
        self.save_ensemble_models(models, lstm_model, self.meta_learner)
        
        self.logger.info(f"Ensemble training completed in {training_time:.0f}s")
        self.logger.info(f"Ensemble accuracy: {ensemble_accuracy:.4f}")
        self.logger.info(f"Best individual model: {best_model.model_name} ({best_model.accuracy:.4f})")
        
        return results
    
    def save_ensemble_models(self, models: Dict[str, Any], lstm_model: Any, meta_learner: MetaLearner):
        """Save all ensemble models"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save tree-based models
        for name, model in models.items():
            model_path = f'/workspace/models/ensemble/{name}_{timestamp}.pkl'
            joblib.dump(model, model_path)
            self.logger.info(f"Saved {name} model to {model_path}")
        
        # Save LSTM model
        if lstm_model:
            lstm_path = f'/workspace/models/ensemble/lstm_{timestamp}.h5'
            lstm_model.save(lstm_path)
            self.logger.info(f"Saved LSTM model to {lstm_path}")
        
        # Save meta-learner
        meta_path = f'/workspace/models/ensemble/meta_learner_{timestamp}.pkl'
        joblib.dump(meta_learner, meta_path)
        self.logger.info(f"Saved meta-learner to {meta_path}")
    
    def save_ensemble_results(self, results: EnsembleResults):
        """Save ensemble training results"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results_dict = {
            'timestamp': timestamp,
            'ensemble_accuracy': results.ensemble_accuracy,
            'ensemble_precision': results.ensemble_precision,
            'ensemble_recall': results.ensemble_recall,
            'ensemble_f1': results.ensemble_f1,
            'ensemble_auc': results.ensemble_auc,
            'best_model_name': results.best_model_name,
            'ensemble_weights': results.ensemble_weights,
            'training_time': results.training_time,
            'individual_performances': [
                {
                    'model_name': p.model_name,
                    'accuracy': p.accuracy,
                    'precision': p.precision,
                    'recall': p.recall,
                    'f1_score': p.f1_score,
                    'auc_score': p.auc_score,
                    'training_time': p.training_time
                }
                for p in results.individual_performances
            ],
            'feature_importance': results.feature_importance
        }
        
        results_path = f'/workspace/models/ensemble/ensemble_results_{timestamp}.json'
        
        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        self.logger.info(f"Ensemble results saved to {results_path}")

async def main():
    """Main ensemble training function"""
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('/workspace/logs/ensemble/ensemble_training.log'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger('EnsembleTrainingSystem')
    logger.info("Starting ensemble training system")
    
    try:
        # Initialize ensemble system
        config = EnsembleConfig(
            use_xgboost=True,
            use_lightgbm=True,
            use_catboost=True,
            use_random_forest=True,
            use_lstm=True,
            optimize_hyperparams=True,
            n_trials=50,
            meta_learner='logistic'
        )
        
        ensemble_system = EnsembleTrainingSystem(config)
        
        # Prepare data
        logger.info("Preparing ensemble training data...")
        X_flat, y, features_df = ensemble_system.prepare_ensemble_data(
            symbols=["EUR/USD", "GBP/USD", "USD/JPY"],
            timeframes=["1h"]
        )
        
        # Prepare sequences for LSTM
        X_sequences, _, _ = ensemble_system.lstm_trainer.prepare_training_data(
            symbols=["EUR/USD", "GBP/USD", "USD/JPY"],
            timeframes=["1h"]
        )
        
        logger.info(f"Data prepared: {X_flat.shape} flat features, {X_sequences.shape} sequences")
        
        # Train ensemble
        logger.info("Training ensemble models...")
        results = ensemble_system.train_ensemble(X_flat, X_sequences, y)
        
        # Save results
        ensemble_system.save_ensemble_results(results)
        
        # Print results
        logger.info("=== ENSEMBLE TRAINING RESULTS ===")
        logger.info(f"Ensemble Accuracy: {results.ensemble_accuracy:.4f}")
        logger.info(f"Ensemble Precision: {results.ensemble_precision:.4f}")
        logger.info(f"Ensemble Recall: {results.ensemble_recall:.4f}")
        logger.info(f"Ensemble F1-Score: {results.ensemble_f1:.4f}")
        logger.info(f"Ensemble AUC: {results.ensemble_auc:.4f}")
        logger.info(f"Best Individual Model: {results.best_model_name}")
        
        logger.info("\nIndividual Model Performances:")
        for perf in results.individual_performances:
            logger.info(f"  {perf.model_name}: {perf.accuracy:.4f} accuracy")
        
        logger.info(f"\nEnsemble Weights:")
        for model, weight in results.ensemble_weights.items():
            logger.info(f"  {model}: {weight:.3f}")
        
        # Check if accuracy target is met
        if results.ensemble_accuracy >= 0.80:
            logger.info("🎉 SUCCESS: Ensemble achieved >80% accuracy target!")
        else:
            logger.warning(f"⚠️  Ensemble accuracy {results.ensemble_accuracy:.2%} below 80% target")
        
    except Exception as e:
        logger.error(f"Ensemble training failed: {e}")
        raise

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())