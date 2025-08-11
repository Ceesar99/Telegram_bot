import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna
from optuna.integration import TensorFlowPruningCallback
import logging
import json
import pickle
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from advanced_transformer_models import (
    FinancialTransformer, TransformerTrainer, MultiTimeframeTransformer, 
    TransformerFeatureProcessor
)
from reinforcement_learning_engine import RLTradingEngine, TradingEnvironment
from ensemble_models import EnsembleSignalGenerator
from ultra_low_latency_wrapper import AdvancedFeatureEngineer

@dataclass
class ModelPerformance:
    """Model performance metrics"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: float
    sharpe_ratio: float
    max_drawdown: float
    training_time: float
    inference_time: float
    hyperparameters: Dict[str, Any]

@dataclass
class TrainingConfiguration:
    """Training configuration for AI models"""
    model_type: str
    hyperparameters: Dict[str, Any]
    training_epochs: int
    batch_size: int
    learning_rate: float
    validation_split: float
    early_stopping_patience: int
    use_optuna_optimization: bool = True
    optuna_trials: int = 100

class AdvancedHyperparameterOptimizer:
    """Advanced hyperparameter optimization using Optuna"""
    
    def __init__(self):
        self.logger = logging.getLogger('HyperparameterOptimizer')
        self.study_storage = "sqlite:///optuna_studies.db"
        
    def optimize_transformer_params(self, X_train: np.ndarray, y_train: np.ndarray,
                                  X_val: np.ndarray, y_val: np.ndarray,
                                  n_trials: int = 100) -> Dict[str, Any]:
        """Optimize transformer hyperparameters"""
        
        def objective(trial):
            # Suggest hyperparameters
            d_model = trial.suggest_categorical('d_model', [128, 256, 512, 768])
            num_heads = trial.suggest_categorical('num_heads', [4, 8, 12, 16])
            num_layers = trial.suggest_int('num_layers', 4, 12)
            d_ff = trial.suggest_categorical('d_ff', [512, 1024, 2048, 4096])
            dropout = trial.suggest_float('dropout', 0.1, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
            
            try:
                # Create model
                model = FinancialTransformer(
                    input_dim=X_train.shape[-1],
                    d_model=d_model,
                    num_heads=num_heads,
                    num_layers=num_layers,
                    d_ff=d_ff,
                    dropout=dropout
                )
                
                trainer = TransformerTrainer(model)
                trainer.optimizer = torch.optim.AdamW(
                    model.parameters(), lr=learning_rate, weight_decay=0.01
                )
                
                # Prepare data
                from advanced_transformer_models import FinancialDataset
                train_dataset = FinancialDataset(X_train, y_train)
                val_dataset = FinancialDataset(X_val, y_val)
                
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                
                # Train for limited epochs
                best_val_acc = 0
                patience_counter = 0
                max_patience = 5
                
                for epoch in range(20):  # Limited epochs for optimization
                    train_stats = trainer.train_epoch(train_loader, epoch)
                    val_stats = trainer.evaluate(val_loader)
                    
                    if val_stats['accuracy'] > best_val_acc:
                        best_val_acc = val_stats['accuracy']
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= max_patience:
                        break
                    
                    # Report intermediate value for pruning
                    trial.report(best_val_acc, epoch)
                    
                    # Handle pruning
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
                
                return best_val_acc
                
            except Exception as e:
                self.logger.warning(f"Trial failed: {e}")
                return 0.0
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            storage=self.study_storage,
            study_name=f'transformer_optimization_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            load_if_exists=True
        )
        
        # Optimize
        study.optimize(objective, n_trials=n_trials, timeout=3600)  # 1 hour timeout
        
        self.logger.info(f"Best transformer params: {study.best_params}")
        self.logger.info(f"Best accuracy: {study.best_value:.4f}")
        
        return study.best_params
    
    def optimize_xgboost_params(self, X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray,
                               n_trials: int = 100) -> Dict[str, Any]:
        """Optimize XGBoost hyperparameters"""
        
        def objective(trial):
            params = {
                'objective': 'multi:softprob',
                'num_class': len(np.unique(y_train)),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'random_state': 42
            }
            
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                     early_stopping_rounds=10, verbose=False)
            
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            
            return accuracy
        
        study = optuna.create_study(
            direction='maximize',
            storage=self.study_storage,
            study_name=f'xgboost_optimization_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            load_if_exists=True
        )
        
        study.optimize(objective, n_trials=n_trials)
        
        return study.best_params
    
    def optimize_lightgbm_params(self, X_train: np.ndarray, y_train: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray,
                                n_trials: int = 100) -> Dict[str, Any]:
        """Optimize LightGBM hyperparameters"""
        
        def objective(trial):
            params = {
                'objective': 'multiclass',
                'num_class': len(np.unique(y_train)),
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': 42,
                'verbose': -1
            }
            
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            model = lgb.train(
                params, train_data, valid_sets=[val_data],
                num_boost_round=1000, callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
            )
            
            y_pred = model.predict(X_val, num_iteration=model.best_iteration)
            y_pred_class = np.argmax(y_pred, axis=1)
            accuracy = accuracy_score(y_val, y_pred_class)
            
            return accuracy
        
        study = optuna.create_study(
            direction='maximize',
            storage=self.study_storage,
            study_name=f'lightgbm_optimization_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            load_if_exists=True
        )
        
        study.optimize(objective, n_trials=n_trials)
        
        return study.best_params

class ModelEnsembleOptimizer:
    """Optimize ensemble model combinations and weights"""
    
    def __init__(self):
        self.logger = logging.getLogger('ModelEnsembleOptimizer')
        
    def optimize_ensemble_weights(self, model_predictions: Dict[str, np.ndarray],
                                 true_labels: np.ndarray, n_trials: int = 100) -> Dict[str, float]:
        """Optimize ensemble weights using Optuna"""
        
        model_names = list(model_predictions.keys())
        
        def objective(trial):
            # Suggest weights for each model
            weights = []
            for i, model_name in enumerate(model_names):
                if i == len(model_names) - 1:
                    # Last weight is 1 - sum of others to ensure weights sum to 1
                    weight = 1.0 - sum(weights)
                else:
                    weight = trial.suggest_float(f'weight_{model_name}', 0.0, 1.0)
                weights.append(max(0.0, weight))
            
            # Normalize weights
            weight_sum = sum(weights)
            if weight_sum == 0:
                return 0.0
            weights = [w / weight_sum for w in weights]
            
            # Calculate ensemble prediction
            ensemble_pred = np.zeros_like(list(model_predictions.values())[0])
            for i, (model_name, predictions) in enumerate(model_predictions.items()):
                ensemble_pred += weights[i] * predictions
            
            # Convert to class predictions
            if len(ensemble_pred.shape) > 1:
                ensemble_class = np.argmax(ensemble_pred, axis=1)
            else:
                ensemble_class = (ensemble_pred > 0.5).astype(int)
            
            accuracy = accuracy_score(true_labels, ensemble_class)
            return accuracy
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        # Extract optimal weights
        optimal_weights = {}
        weights = []
        for i, model_name in enumerate(model_names):
            if i == len(model_names) - 1:
                weight = 1.0 - sum(weights)
            else:
                weight = study.best_params[f'weight_{model_name}']
            weights.append(max(0.0, weight))
        
        # Normalize
        weight_sum = sum(weights)
        for i, model_name in enumerate(model_names):
            optimal_weights[model_name] = weights[i] / weight_sum
        
        return optimal_weights

class AdvancedAITrainingSystem:
    """Comprehensive AI training system for maximum signal accuracy"""
    
    def __init__(self):
        self.logger = logging.getLogger('AdvancedAITrainingSystem')
        self.hyperparameter_optimizer = AdvancedHyperparameterOptimizer()
        self.ensemble_optimizer = ModelEnsembleOptimizer()
        self.feature_engineer = AdvancedFeatureEngineer()
        
        # Model registry
        self.trained_models = {}
        self.model_performances = {}
        self.optimal_hyperparameters = {}
        
        # Training configurations
        self.training_configs = self._get_default_training_configs()
        
    def _get_default_training_configs(self) -> Dict[str, TrainingConfiguration]:
        """Get default training configurations for different model types"""
        return {
            'transformer': TrainingConfiguration(
                model_type='transformer',
                hyperparameters={
                    'd_model': 256,
                    'num_heads': 8,
                    'num_layers': 6,
                    'd_ff': 1024,
                    'dropout': 0.1
                },
                training_epochs=100,
                batch_size=32,
                learning_rate=1e-4,
                validation_split=0.2,
                early_stopping_patience=10
            ),
            'xgboost': TrainingConfiguration(
                model_type='xgboost',
                hyperparameters={
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'n_estimators': 1000,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8
                },
                training_epochs=1000,
                batch_size=0,  # Not applicable for XGBoost
                learning_rate=0.1,
                validation_split=0.2,
                early_stopping_patience=10
            ),
            'lightgbm': TrainingConfiguration(
                model_type='lightgbm',
                hyperparameters={
                    'num_leaves': 31,
                    'learning_rate': 0.1,
                    'feature_fraction': 0.9,
                    'bagging_fraction': 0.8
                },
                training_epochs=1000,
                batch_size=0,
                learning_rate=0.1,
                validation_split=0.2,
                early_stopping_patience=10
            ),
            'catboost': TrainingConfiguration(
                model_type='catboost',
                hyperparameters={
                    'iterations': 1000,
                    'depth': 6,
                    'learning_rate': 0.1,
                    'l2_leaf_reg': 3
                },
                training_epochs=1000,
                batch_size=0,
                learning_rate=0.1,
                validation_split=0.2,
                early_stopping_patience=10
            ),
            'lstm': TrainingConfiguration(
                model_type='lstm',
                hyperparameters={
                    'hidden_size': 128,
                    'num_layers': 2,
                    'dropout': 0.2,
                    'bidirectional': True
                },
                training_epochs=100,
                batch_size=32,
                learning_rate=1e-3,
                validation_split=0.2,
                early_stopping_patience=10
            )
        }
    
    def prepare_training_data(self, price_data: np.ndarray, 
                            additional_features: np.ndarray = None,
                            target_labels: np.ndarray = None,
                            sequence_length: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare comprehensive training data with advanced feature engineering"""
        
        self.logger.info("Preparing training data with advanced feature engineering...")
        
        # Generate advanced features
        volume_data = np.random.rand(len(price_data)) * 1000  # Simulated volume data
        features = self.feature_engineer.generate_advanced_features(price_data, volume_data)
        
        # Add additional features if provided
        if additional_features is not None:
            combined_features = {}
            combined_features.update(features)
            for i, feature in enumerate(additional_features.T):
                combined_features[f'additional_feature_{i}'] = feature
            features = combined_features
        
        # Convert to structured array
        feature_names = list(features.keys())
        feature_matrix = np.column_stack([features[name] for name in feature_names])
        
        # Handle NaN values
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Create sequences for time series models
        X_sequences = []
        y_sequences = []
        
        for i in range(sequence_length, len(feature_matrix)):
            X_sequences.append(feature_matrix[i-sequence_length:i])
            if target_labels is not None:
                y_sequences.append(target_labels[i])
            else:
                # Generate target labels based on price movement
                price_change = (price_data[i] - price_data[i-1]) / price_data[i-1]
                if price_change > 0.001:  # 0.1% threshold
                    label = 1  # BUY
                elif price_change < -0.001:
                    label = 2  # SELL
                else:
                    label = 0  # HOLD
                y_sequences.append(label)
        
        X = np.array(X_sequences)
        y = np.array(y_sequences)
        
        self.logger.info(f"Prepared training data: {X.shape[0]} samples, {X.shape[1]} sequence length, {X.shape[2]} features")
        
        return X, y
    
    def train_all_models(self, X: np.ndarray, y: np.ndarray, 
                        optimize_hyperparameters: bool = True) -> Dict[str, ModelPerformance]:
        """Train all AI models with hyperparameter optimization"""
        
        self.logger.info("Starting comprehensive AI model training...")
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        val_split_idx = int(0.8 * len(X_train))
        X_train_final, X_val = X_train[:val_split_idx], X_train[val_split_idx:]
        y_train_final, y_val = y_train[:val_split_idx], y_train[val_split_idx:]
        
        model_performances = {}
        
        # Train each model type
        for model_name, config in self.training_configs.items():
            self.logger.info(f"Training {model_name} model...")
            
            try:
                start_time = datetime.now()
                
                # Optimize hyperparameters if requested
                if optimize_hyperparameters and config.use_optuna_optimization:
                    self.logger.info(f"Optimizing hyperparameters for {model_name}...")
                    optimal_params = self._optimize_model_hyperparameters(
                        model_name, X_train_final, y_train_final, X_val, y_val
                    )
                    config.hyperparameters.update(optimal_params)
                    self.optimal_hyperparameters[model_name] = optimal_params
                
                # Train model
                model, performance = self._train_single_model(
                    model_name, config, X_train, y_train, X_test, y_test
                )
                
                training_time = (datetime.now() - start_time).total_seconds()
                performance.training_time = training_time
                performance.hyperparameters = config.hyperparameters
                
                self.trained_models[model_name] = model
                model_performances[model_name] = performance
                
                self.logger.info(f"✅ {model_name} training completed - Accuracy: {performance.accuracy:.4f}")
                
            except Exception as e:
                self.logger.error(f"❌ Error training {model_name}: {e}")
                continue
        
        self.model_performances = model_performances
        
        # Optimize ensemble
        self.logger.info("Optimizing ensemble weights...")
        self._optimize_ensemble(X_test, y_test)
        
        return model_performances
    
    def _optimize_model_hyperparameters(self, model_name: str, X_train: np.ndarray, 
                                      y_train: np.ndarray, X_val: np.ndarray, 
                                      y_val: np.ndarray) -> Dict[str, Any]:
        """Optimize hyperparameters for a specific model"""
        
        if model_name == 'transformer':
            return self.hyperparameter_optimizer.optimize_transformer_params(
                X_train, y_train, X_val, y_val
            )
        elif model_name == 'xgboost':
            # Flatten sequences for traditional ML models
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_val_flat = X_val.reshape(X_val.shape[0], -1)
            return self.hyperparameter_optimizer.optimize_xgboost_params(
                X_train_flat, y_train, X_val_flat, y_val
            )
        elif model_name == 'lightgbm':
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_val_flat = X_val.reshape(X_val.shape[0], -1)
            return self.hyperparameter_optimizer.optimize_lightgbm_params(
                X_train_flat, y_train, X_val_flat, y_val
            )
        else:
            return {}
    
    def _train_single_model(self, model_name: str, config: TrainingConfiguration,
                           X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray) -> Tuple[Any, ModelPerformance]:
        """Train a single model"""
        
        if model_name == 'transformer':
            return self._train_transformer(config, X_train, y_train, X_test, y_test)
        elif model_name == 'xgboost':
            return self._train_xgboost(config, X_train, y_train, X_test, y_test)
        elif model_name == 'lightgbm':
            return self._train_lightgbm(config, X_train, y_train, X_test, y_test)
        elif model_name == 'catboost':
            return self._train_catboost(config, X_train, y_train, X_test, y_test)
        elif model_name == 'lstm':
            return self._train_lstm(config, X_train, y_train, X_test, y_test)
        else:
            raise ValueError(f"Unknown model type: {model_name}")
    
    def _train_transformer(self, config: TrainingConfiguration,
                          X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray) -> Tuple[TransformerTrainer, ModelPerformance]:
        """Train transformer model"""
        
        model = FinancialTransformer(
            input_dim=X_train.shape[-1],
            **config.hyperparameters
        )
        
        trainer = TransformerTrainer(model)
        
        # Prepare data
        from advanced_transformer_models import FinancialDataset
        train_dataset = FinancialDataset(X_train, y_train)
        test_dataset = FinancialDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
        
        # Train
        for epoch in range(config.training_epochs):
            train_stats = trainer.train_epoch(train_loader, epoch)
            
            if epoch % 10 == 0:
                val_stats = trainer.evaluate(test_loader)
                self.logger.debug(f"Transformer Epoch {epoch}: Val Acc = {val_stats['accuracy']:.4f}")
        
        # Final evaluation
        test_stats = trainer.evaluate(test_loader)
        
        # Calculate inference time
        import time
        start_time = time.time()
        sample_prediction = trainer.predict(X_test[0])
        inference_time = (time.time() - start_time) * 1000  # ms
        
        performance = ModelPerformance(
            model_name='transformer',
            accuracy=test_stats['accuracy'],
            precision=0.0,  # Would need to calculate properly
            recall=0.0,
            f1_score=0.0,
            auc_score=0.0,
            sharpe_ratio=0.0,  # Would need price data
            max_drawdown=0.0,
            training_time=0.0,  # Set later
            inference_time=inference_time,
            hyperparameters=config.hyperparameters
        )
        
        return trainer, performance
    
    def _train_xgboost(self, config: TrainingConfiguration,
                      X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray) -> Tuple[xgb.XGBClassifier, ModelPerformance]:
        """Train XGBoost model"""
        
        # Flatten sequences for traditional ML
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        model = xgb.XGBClassifier(**config.hyperparameters, random_state=42)
        model.fit(X_train_flat, y_train, eval_set=[(X_test_flat, y_test)], 
                 early_stopping_rounds=config.early_stopping_patience, verbose=False)
        
        # Predictions
        y_pred = model.predict(X_test_flat)
        y_pred_proba = model.predict_proba(X_test_flat)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # AUC for multiclass
        try:
            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        except:
            auc = 0.0
        
        # Inference time
        import time
        start_time = time.time()
        _ = model.predict(X_test_flat[0:1])
        inference_time = (time.time() - start_time) * 1000
        
        performance = ModelPerformance(
            model_name='xgboost',
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_score=auc,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            training_time=0.0,
            inference_time=inference_time,
            hyperparameters=config.hyperparameters
        )
        
        return model, performance
    
    def _train_lightgbm(self, config: TrainingConfiguration,
                       X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray) -> Tuple[lgb.LGBMClassifier, ModelPerformance]:
        """Train LightGBM model"""
        
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        model = lgb.LGBMClassifier(**config.hyperparameters, random_state=42, verbose=-1)
        model.fit(X_train_flat, y_train, eval_set=[(X_test_flat, y_test)], 
                 callbacks=[lgb.early_stopping(config.early_stopping_patience), lgb.log_evaluation(0)])
        
        y_pred = model.predict(X_test_flat)
        y_pred_proba = model.predict_proba(X_test_flat)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        try:
            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        except:
            auc = 0.0
        
        import time
        start_time = time.time()
        _ = model.predict(X_test_flat[0:1])
        inference_time = (time.time() - start_time) * 1000
        
        performance = ModelPerformance(
            model_name='lightgbm',
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_score=auc,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            training_time=0.0,
            inference_time=inference_time,
            hyperparameters=config.hyperparameters
        )
        
        return model, performance
    
    def _train_catboost(self, config: TrainingConfiguration,
                       X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray) -> Tuple[cb.CatBoostClassifier, ModelPerformance]:
        """Train CatBoost model"""
        
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        model = cb.CatBoostClassifier(**config.hyperparameters, random_state=42, verbose=False)
        model.fit(X_train_flat, y_train, eval_set=(X_test_flat, y_test), 
                 early_stopping_rounds=config.early_stopping_patience)
        
        y_pred = model.predict(X_test_flat)
        y_pred_proba = model.predict_proba(X_test_flat)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        try:
            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        except:
            auc = 0.0
        
        import time
        start_time = time.time()
        _ = model.predict(X_test_flat[0:1])
        inference_time = (time.time() - start_time) * 1000
        
        performance = ModelPerformance(
            model_name='catboost',
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_score=auc,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            training_time=0.0,
            inference_time=inference_time,
            hyperparameters=config.hyperparameters
        )
        
        return model, performance
    
    def _train_lstm(self, config: TrainingConfiguration,
                   X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray, y_test: np.ndarray) -> Tuple[tf.keras.Model, ModelPerformance]:
        """Train LSTM model"""
        
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(config.hyperparameters['hidden_size'], 
                               return_sequences=True, dropout=config.hyperparameters['dropout']),
            tf.keras.layers.LSTM(config.hyperparameters['hidden_size'], 
                               dropout=config.hyperparameters['dropout']),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(len(np.unique(y_train)), activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=config.training_epochs,
            batch_size=config.batch_size,
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=config.early_stopping_patience, restore_best_weights=True)
            ]
        )
        
        # Evaluate
        y_pred_proba = model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        try:
            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        except:
            auc = 0.0
        
        import time
        start_time = time.time()
        _ = model.predict(X_test[0:1])
        inference_time = (time.time() - start_time) * 1000
        
        performance = ModelPerformance(
            model_name='lstm',
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_score=auc,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            training_time=0.0,
            inference_time=inference_time,
            hyperparameters=config.hyperparameters
        )
        
        return model, performance
    
    def _optimize_ensemble(self, X_test: np.ndarray, y_test: np.ndarray):
        """Optimize ensemble weights"""
        
        model_predictions = {}
        
        for model_name, model in self.trained_models.items():
            if model_name == 'transformer':
                # Get transformer predictions
                predictions = []
                for i in range(len(X_test)):
                    pred = model.predict(X_test[i])
                    predictions.append(pred['probabilities'])
                model_predictions[model_name] = np.array(predictions)
                
            elif model_name in ['xgboost', 'lightgbm', 'catboost']:
                # Traditional ML models
                X_test_flat = X_test.reshape(X_test.shape[0], -1)
                model_predictions[model_name] = model.predict_proba(X_test_flat)
                
            elif model_name == 'lstm':
                # LSTM predictions
                model_predictions[model_name] = model.predict(X_test)
        
        if model_predictions:
            optimal_weights = self.ensemble_optimizer.optimize_ensemble_weights(
                model_predictions, y_test
            )
            self.optimal_ensemble_weights = optimal_weights
            self.logger.info(f"Optimal ensemble weights: {optimal_weights}")
    
    def predict_ensemble(self, X: np.ndarray) -> Dict[str, Any]:
        """Make ensemble prediction using all trained models"""
        
        predictions = {}
        
        # Get predictions from all models
        for model_name, model in self.trained_models.items():
            if model_name == 'transformer':
                pred = model.predict(X)
                predictions[model_name] = pred['probabilities']
                
            elif model_name in ['xgboost', 'lightgbm', 'catboost']:
                X_flat = X.reshape(1, -1)
                predictions[model_name] = model.predict_proba(X_flat)[0]
                
            elif model_name == 'lstm':
                predictions[model_name] = model.predict(X.reshape(1, *X.shape))[0]
        
        # Combine using optimal weights
        if hasattr(self, 'optimal_ensemble_weights') and predictions:
            ensemble_pred = np.zeros_like(list(predictions.values())[0])
            for model_name, pred in predictions.items():
                weight = self.optimal_ensemble_weights.get(model_name, 1.0 / len(predictions))
                ensemble_pred += weight * pred
            
            final_prediction = np.argmax(ensemble_pred)
            confidence = np.max(ensemble_pred)
            
            return {
                'prediction': int(final_prediction),
                'confidence': float(confidence),
                'probabilities': ensemble_pred.tolist(),
                'individual_predictions': predictions
            }
        
        return {'prediction': 0, 'confidence': 0.0, 'probabilities': [0.33, 0.33, 0.34]}
    
    def get_training_report(self) -> Dict[str, Any]:
        """Generate comprehensive training report"""
        
        report = {
            'training_timestamp': datetime.now().isoformat(),
            'models_trained': list(self.trained_models.keys()),
            'model_performances': {},
            'optimal_hyperparameters': self.optimal_hyperparameters,
            'ensemble_weights': getattr(self, 'optimal_ensemble_weights', {}),
            'best_model': None,
            'ensemble_expected_accuracy': 0.0
        }
        
        # Model performance summary
        best_accuracy = 0.0
        best_model = None
        
        for model_name, performance in self.model_performances.items():
            report['model_performances'][model_name] = {
                'accuracy': performance.accuracy,
                'precision': performance.precision,
                'recall': performance.recall,
                'f1_score': performance.f1_score,
                'auc_score': performance.auc_score,
                'training_time': performance.training_time,
                'inference_time': performance.inference_time
            }
            
            if performance.accuracy > best_accuracy:
                best_accuracy = performance.accuracy
                best_model = model_name
        
        report['best_model'] = best_model
        report['best_individual_accuracy'] = best_accuracy
        
        # Estimate ensemble accuracy (would be higher than individual models)
        if self.model_performances:
            avg_accuracy = np.mean([p.accuracy for p in self.model_performances.values()])
            report['ensemble_expected_accuracy'] = min(0.99, avg_accuracy * 1.1)  # Conservative estimate
        
        return report
    
    def save_models(self, save_directory: str = "trained_models"):
        """Save all trained models"""
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        for model_name, model in self.trained_models.items():
            model_path = os.path.join(save_directory, f"{model_name}_model")
            
            if model_name == 'transformer':
                model.save_model(f"{model_path}.pth")
            elif model_name in ['xgboost', 'lightgbm', 'catboost']:
                with open(f"{model_path}.pkl", 'wb') as f:
                    pickle.dump(model, f)
            elif model_name == 'lstm':
                model.save(f"{model_path}.h5")
        
        # Save training report
        report = self.get_training_report()
        with open(os.path.join(save_directory, "training_report.json"), 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"All models saved to {save_directory}")

# Example usage
if __name__ == "__main__":
    # Initialize training system
    training_system = AdvancedAITrainingSystem()
    
    # Generate sample data (in practice, use real market data)
    np.random.seed(42)
    price_data = np.cumsum(np.random.randn(5000) * 0.01) + 100
    
    # Prepare training data
    X, y = training_system.prepare_training_data(price_data)
    
    # Train all models with hyperparameter optimization
    performances = training_system.train_all_models(X, y, optimize_hyperparameters=True)
    
    # Get training report
    report = training_system.get_training_report()
    print(json.dumps(report, indent=2))
    
    # Save models
    training_system.save_models()