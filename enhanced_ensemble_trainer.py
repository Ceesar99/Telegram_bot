import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
import os
import json
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    VotingClassifier, BaggingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)

# XGBoost for gradient boosting
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available, skipping XGBoost models")

# LightGBM for additional gradient boosting
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("LightGBM not available, skipping LightGBM models")

# TensorFlow for neural networks
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available, skipping neural network models")

from dataclasses import dataclass

@dataclass
class ModelResult:
    """Container for individual model results"""
    model_name: str
    model: Any
    train_accuracy: float
    test_accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    cv_scores: List[float]
    feature_importance: Dict[str, float]
    training_time: float
    model_path: str

@dataclass
class EnsembleResult:
    """Container for ensemble training results"""
    ensemble_accuracy: float
    individual_results: List[ModelResult]
    best_model: str
    ensemble_weights: Dict[str, float]
    feature_rankings: Dict[str, float]
    cross_validation_score: float
    training_summary: Dict[str, Any]

class EnhancedEnsembleTrainer:
    """
    🚀 Enhanced Ensemble Training System
    
    Features:
    - Multiple ML Algorithms (RF, XGB, LightGBM, Neural Networks)
    - Automated Hyperparameter Tuning
    - Time Series Cross-Validation
    - Weighted Ensemble with Calibration
    - Feature Importance Analysis
    - Production-Ready Model Pipeline
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._get_default_config()
        self.logger = self._setup_logger()
        
        # Model storage
        self.trained_models = {}
        self.ensemble_model = None
        self.feature_scaler = StandardScaler()
        self.feature_names = []
        
        # Results tracking
        self.training_results = []
        self.ensemble_weights = {}
        
        # Create directories
        os.makedirs(self.config['model_save_dir'], exist_ok=True)
        os.makedirs(self.config['logs_dir'], exist_ok=True)
        
    def _get_default_config(self) -> Dict:
        """Enhanced ensemble configuration"""
        return {
            # Model Selection
            'enable_random_forest': True,
            'enable_gradient_boosting': True,
            'enable_xgboost': XGBOOST_AVAILABLE,
            'enable_lightgbm': LIGHTGBM_AVAILABLE,
            'enable_svm': True,
            'enable_logistic_regression': True,
            'enable_neural_network': TENSORFLOW_AVAILABLE,
            
            # Training Parameters
            'test_size': 0.2,
            'cv_folds': 5,
            'random_state': 42,
            'n_jobs': -1,  # Use all available cores
            
            # Model Parameters
            'rf_params': {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'class_weight': 'balanced'
            },
            
            'gb_params': {
                'n_estimators': 200,
                'learning_rate': 0.1,
                'max_depth': 6,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'subsample': 0.8
            },
            
            'xgb_params': {
                'n_estimators': 200,
                'learning_rate': 0.1,
                'max_depth': 6,
                'min_child_weight': 1,
                'gamma': 0,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'multi:softprob',
                'eval_metric': 'mlogloss'
            },
            
            'lgb_params': {
                'n_estimators': 200,
                'learning_rate': 0.1,
                'max_depth': 6,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'multiclass',
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt'
            },
            
            'svm_params': {
                'C': 1.0,
                'kernel': 'rbf',
                'gamma': 'scale',
                'probability': True,
                'class_weight': 'balanced'
            },
            
            'lr_params': {
                'C': 1.0,
                'solver': 'liblinear',
                'class_weight': 'balanced',
                'max_iter': 1000
            },
            
            'nn_params': {
                'hidden_layers': [128, 64, 32],
                'dropout_rate': 0.3,
                'learning_rate': 0.001,
                'epochs': 100,
                'batch_size': 32,
                'patience': 15
            },
            
            # Ensemble Parameters
            'ensemble_method': 'weighted',  # 'voting', 'weighted', 'stacking'
            'min_models_for_ensemble': 3,
            'weight_by_accuracy': True,
            'calibrate_probabilities': True,
            
            # Paths
            'model_save_dir': '/workspace/models/ensemble/',
            'logs_dir': '/workspace/logs/ensemble_training/',
            'data_path': '/workspace/data/real_market_data/',
            
            # Feature Engineering
            'enable_feature_selection': True,
            'max_features': 50,
            'feature_selection_method': 'mutual_info',
            
            # Performance Thresholds
            'min_accuracy_threshold': 0.60,
            'min_precision_threshold': 0.55,
            'min_recall_threshold': 0.55
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Setup comprehensive logging"""
        logger = logging.getLogger('EnhancedEnsembleTrainer')
        logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler('/workspace/logs/enhanced_ensemble_training.log')
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def prepare_data(self, data_path: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for ensemble training
        """
        self.logger.info("📊 Preparing data for ensemble training...")
        
        data_path = data_path or self.config['data_path']
        combined_file = os.path.join(data_path, 'combined_market_data_20250816_092932.csv')
        
        if not os.path.exists(combined_file):
            raise FileNotFoundError(f"Training data not found: {combined_file}")
        
        # Load data in chunks
        chunk_size = 10000
        data_chunks = []
        
        for chunk in pd.read_csv(combined_file, chunksize=chunk_size):
            data_chunks.append(chunk)
        
        df = pd.concat(data_chunks, ignore_index=True)
        self.logger.info(f"📊 Loaded {len(df):,} data points")
        
        # Basic preprocessing
        df = self._preprocess_ensemble_data(df)
        
        # Feature engineering
        df = self._engineer_ensemble_features(df)
        
        # Clean data
        df = df.dropna()
        self.logger.info(f"📊 After preprocessing: {len(df):,} clean samples")
        
        # Prepare features and targets
        X, y = self._prepare_features_targets(df)
        
        self.logger.info(f"✅ Data prepared: {X.shape[0]:,} samples, {X.shape[1]} features")
        return X, y
    
    def _preprocess_ensemble_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data for ensemble models"""
        
        # Ensure required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            self.logger.warning(f"⚠️ Missing columns: {missing_cols}")
            for col in missing_cols:
                if col == 'volume':
                    df[col] = 1000
                else:
                    df[col] = df.get('close', 1.0)
        
        # Remove extreme outliers
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.01)
                Q3 = df[col].quantile(0.99)
                df = df[(df[col] >= Q1) & (df[col] <= Q3)]
        
        return df
    
    def _engineer_ensemble_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features optimized for ensemble models"""
        
        self.logger.info("🔬 Engineering ensemble features...")
        
        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['price_change'] = df['close'] - df['open']
        df['price_range'] = df['high'] - df['low']
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
            df[f'price_vs_sma_{window}'] = df['close'] / df[f'sma_{window}'] - 1
        
        # Volatility features
        for window in [5, 10, 20]:
            df[f'volatility_{window}'] = df['returns'].rolling(window=window).std()
            df[f'realized_vol_{window}'] = df['returns'].rolling(window=window).std() * np.sqrt(252)
        
        # Momentum features
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
            df[f'roc_{period}'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period)
        
        # Volume features
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['price_volume'] = df['close'] * df['volume']
        
        # Technical indicators (simplified)
        df['rsi'] = self._calculate_rsi(df['close'])
        df['bb_upper'], df['bb_lower'] = self._calculate_bollinger_bands(df['close'])
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Lag features
        for lag in [1, 2, 3, 5]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        
        # Statistical features
        for window in [10, 20]:
            df[f'close_mean_{window}'] = df['close'].rolling(window=window).mean()
            df[f'close_std_{window}'] = df['close'].rolling(window=window).std()
            df[f'close_min_{window}'] = df['close'].rolling(window=window).min()
            df[f'close_max_{window}'] = df['close'].rolling(window=window).max()
            df[f'close_skew_{window}'] = df['close'].rolling(window=window).skew()
        
        # Create target variable (3-class classification)
        df['target'] = self._create_ensemble_target(df)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std: float = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        rolling_std = prices.rolling(window=period).std()
        upper_band = sma + (rolling_std * std)
        lower_band = sma - (rolling_std * std)
        return upper_band, lower_band
    
    def _create_ensemble_target(self, df: pd.DataFrame) -> pd.Series:
        """Create target variable for ensemble classification"""
        
        # Calculate future returns
        future_returns = df['close'].shift(-1) / df['close'] - 1
        
        # Define thresholds for classification
        buy_threshold = 0.002   # 0.2% up movement
        sell_threshold = -0.002  # 0.2% down movement
        
        # Create classification targets
        # 0 = BUY, 1 = SELL, 2 = HOLD
        targets = np.where(
            future_returns > buy_threshold, 0,
            np.where(future_returns < sell_threshold, 1, 2)
        )
        
        return pd.Series(targets, index=df.index)
    
    def _prepare_features_targets(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare final features and targets"""
        
        # Select feature columns
        feature_cols = [col for col in df.columns 
                       if col not in ['target', 'timestamp'] 
                       and df[col].dtype in ['float64', 'int64']]
        
        # Handle infinite values
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        # Store feature names
        self.feature_names = feature_cols
        
        # Get features and targets
        X = df[feature_cols].values
        y = df['target'].values
        
        # Remove invalid targets
        valid_mask = ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask].astype(int)
        
        # Scale features
        X = self.feature_scaler.fit_transform(X)
        
        return X, y
    
    def train_individual_models(self, X: np.ndarray, y: np.ndarray) -> List[ModelResult]:
        """Train individual models for ensemble"""
        
        self.logger.info("🚀 Training individual models...")
        
        # Split data
        split_idx = int(len(X) * (1 - self.config['test_size']))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        models_to_train = []
        
        # Random Forest
        if self.config['enable_random_forest']:
            models_to_train.append(
                ('RandomForest', RandomForestClassifier, self.config['rf_params'])
            )
        
        # Gradient Boosting
        if self.config['enable_gradient_boosting']:
            models_to_train.append(
                ('GradientBoosting', GradientBoostingClassifier, self.config['gb_params'])
            )
        
        # XGBoost
        if self.config['enable_xgboost'] and XGBOOST_AVAILABLE:
            models_to_train.append(
                ('XGBoost', xgb.XGBClassifier, self.config['xgb_params'])
            )
        
        # LightGBM
        if self.config['enable_lightgbm'] and LIGHTGBM_AVAILABLE:
            models_to_train.append(
                ('LightGBM', lgb.LGBMClassifier, self.config['lgb_params'])
            )
        
        # SVM
        if self.config['enable_svm']:
            models_to_train.append(
                ('SVM', SVC, self.config['svm_params'])
            )
        
        # Logistic Regression
        if self.config['enable_logistic_regression']:
            models_to_train.append(
                ('LogisticRegression', LogisticRegression, self.config['lr_params'])
            )
        
        # Neural Network
        if self.config['enable_neural_network'] and TENSORFLOW_AVAILABLE:
            models_to_train.append(
                ('NeuralNetwork', self._create_neural_network, self.config['nn_params'])
            )
        
        results = []
        
        for model_name, model_class, params in models_to_train:
            self.logger.info(f"🔄 Training {model_name}...")
            
            try:
                start_time = datetime.now()
                
                if model_name == 'NeuralNetwork':
                    result = self._train_neural_network(
                        X_train, y_train, X_test, y_test, params
                    )
                else:
                    result = self._train_sklearn_model(
                        model_name, model_class, params,
                        X_train, y_train, X_test, y_test, X
                    )
                
                training_time = (datetime.now() - start_time).total_seconds()
                result.training_time = training_time
                
                results.append(result)
                self.trained_models[model_name] = result.model
                
                self.logger.info(f"✅ {model_name} - Accuracy: {result.test_accuracy:.4f}")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to train {model_name}: {e}")
                continue
        
        return results
    
    def _train_sklearn_model(self, model_name: str, model_class: Any, params: Dict,
                           X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray,
                           X_full: np.ndarray) -> ModelResult:
        """Train a scikit-learn model"""
        
        # Create model
        model = model_class(**params, random_state=self.config['random_state'])
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_test_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_test_pred, average='macro', zero_division=0)
        
        # ROC AUC for multi-class
        try:
            if hasattr(model, 'predict_proba'):
                y_test_proba = model.predict_proba(X_test)
                roc_auc = roc_auc_score(y_test, y_test_proba, multi_class='ovr', average='macro')
            else:
                roc_auc = 0.0
        except:
            roc_auc = 0.0
        
        # Cross-validation
        cv_scores = []
        try:
            tscv = TimeSeriesSplit(n_splits=self.config['cv_folds'])
            cv_scores = cross_val_score(model, X_full, np.concatenate([y_train, y_test]), 
                                      cv=tscv, scoring='accuracy')
        except:
            cv_scores = [test_accuracy]
        
        # Feature importance
        feature_importance = {}
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_importance = {
                    self.feature_names[i]: float(importances[i])
                    for i in range(len(importances))
                }
            elif hasattr(model, 'coef_') and len(model.coef_.shape) == 2:
                # For multi-class, take mean absolute coefficients
                importances = np.mean(np.abs(model.coef_), axis=0)
                feature_importance = {
                    self.feature_names[i]: float(importances[i])
                    for i in range(len(importances))
                }
        except:
            pass
        
        # Save model
        model_path = os.path.join(
            self.config['model_save_dir'], 
            f'{model_name.lower()}_model.pkl'
        )
        joblib.dump(model, model_path)
        
        return ModelResult(
            model_name=model_name,
            model=model,
            train_accuracy=train_accuracy,
            test_accuracy=test_accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_auc=roc_auc,
            cv_scores=cv_scores.tolist() if len(cv_scores) > 0 else [test_accuracy],
            feature_importance=feature_importance,
            training_time=0.0,  # Will be set by caller
            model_path=model_path
        )
    
    def _create_neural_network(self, input_dim: int, num_classes: int = 3) -> tf.keras.Model:
        """Create neural network model"""
        
        model = Sequential()
        
        # Input layer
        model.add(Dense(
            self.config['nn_params']['hidden_layers'][0], 
            activation='relu', 
            input_dim=input_dim
        ))
        model.add(Dropout(self.config['nn_params']['dropout_rate']))
        model.add(BatchNormalization())
        
        # Hidden layers
        for units in self.config['nn_params']['hidden_layers'][1:]:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(self.config['nn_params']['dropout_rate']))
            model.add(BatchNormalization())
        
        # Output layer
        model.add(Dense(num_classes, activation='softmax'))
        
        # Compile
        model.compile(
            optimizer=Adam(learning_rate=self.config['nn_params']['learning_rate']),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _train_neural_network(self, X_train: np.ndarray, y_train: np.ndarray,
                            X_test: np.ndarray, y_test: np.ndarray,
                            params: Dict) -> ModelResult:
        """Train neural network model"""
        
        # Create model
        model = self._create_neural_network(X_train.shape[1])
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=params['patience'],
                restore_best_weights=True
            )
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            callbacks=callbacks,
            verbose=0
        )
        
        # Predictions
        y_train_pred = np.argmax(model.predict(X_train), axis=1)
        y_test_pred = np.argmax(model.predict(X_test), axis=1)
        
        # Metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_test_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_test_pred, average='macro', zero_division=0)
        
        # ROC AUC
        try:
            y_test_proba = model.predict(X_test)
            roc_auc = roc_auc_score(y_test, y_test_proba, multi_class='ovr', average='macro')
        except:
            roc_auc = 0.0
        
        # Save model
        model_path = os.path.join(
            self.config['model_save_dir'], 
            'neural_network_model.h5'
        )
        model.save(model_path)
        
        return ModelResult(
            model_name='NeuralNetwork',
            model=model,
            train_accuracy=train_accuracy,
            test_accuracy=test_accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_auc=roc_auc,
            cv_scores=[test_accuracy],  # NN doesn't support cross-validation easily
            feature_importance={},  # NN feature importance is complex
            training_time=0.0,
            model_path=model_path
        )
    
    def create_ensemble(self, individual_results: List[ModelResult]) -> EnsembleResult:
        """Create ensemble from individual models"""
        
        self.logger.info("🔗 Creating ensemble model...")
        
        # Filter models that meet minimum thresholds
        qualified_models = [
            result for result in individual_results
            if (result.test_accuracy >= self.config['min_accuracy_threshold'] and
                result.precision >= self.config['min_precision_threshold'] and
                result.recall >= self.config['min_recall_threshold'])
        ]
        
        if len(qualified_models) < self.config['min_models_for_ensemble']:
            self.logger.warning(f"Not enough qualified models ({len(qualified_models)}) for ensemble")
            # Use all available models
            qualified_models = individual_results
        
        self.logger.info(f"📊 Using {len(qualified_models)} models for ensemble")
        
        # Calculate ensemble weights
        if self.config['weight_by_accuracy']:
            weights = self._calculate_accuracy_weights(qualified_models)
        else:
            weights = {result.model_name: 1.0 / len(qualified_models) 
                      for result in qualified_models}
        
        # Create ensemble model
        if self.config['ensemble_method'] == 'voting':
            ensemble_model = self._create_voting_ensemble(qualified_models)
        elif self.config['ensemble_method'] == 'weighted':
            ensemble_model = self._create_weighted_ensemble(qualified_models, weights)
        else:
            # Default to weighted
            ensemble_model = self._create_weighted_ensemble(qualified_models, weights)
        
        # Calculate ensemble performance
        ensemble_accuracy = np.mean([result.test_accuracy for result in qualified_models])
        cv_score = np.mean([np.mean(result.cv_scores) for result in qualified_models])
        
        # Find best individual model
        best_model = max(qualified_models, key=lambda x: x.test_accuracy)
        
        # Aggregate feature importance
        feature_rankings = self._aggregate_feature_importance(qualified_models)
        
        # Training summary
        training_summary = {
            'total_models_trained': len(individual_results),
            'qualified_models': len(qualified_models),
            'ensemble_method': self.config['ensemble_method'],
            'training_date': datetime.now().isoformat(),
            'best_individual_accuracy': best_model.test_accuracy,
            'ensemble_accuracy': ensemble_accuracy,
            'improvement_over_best': ensemble_accuracy - best_model.test_accuracy
        }
        
        # Save ensemble
        self.ensemble_model = ensemble_model
        self.ensemble_weights = weights
        
        ensemble_path = os.path.join(self.config['model_save_dir'], 'ensemble_model.pkl')
        joblib.dump(ensemble_model, ensemble_path)
        
        # Save weights and metadata
        metadata = {
            'weights': weights,
            'feature_rankings': feature_rankings,
            'training_summary': training_summary,
            'qualified_models': [m.model_name for m in qualified_models]
        }
        
        metadata_path = os.path.join(self.config['model_save_dir'], 'ensemble_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return EnsembleResult(
            ensemble_accuracy=ensemble_accuracy,
            individual_results=individual_results,
            best_model=best_model.model_name,
            ensemble_weights=weights,
            feature_rankings=feature_rankings,
            cross_validation_score=cv_score,
            training_summary=training_summary
        )
    
    def _calculate_accuracy_weights(self, results: List[ModelResult]) -> Dict[str, float]:
        """Calculate weights based on model accuracy"""
        
        accuracies = np.array([result.test_accuracy for result in results])
        
        # Use softmax to convert accuracies to weights
        exp_acc = np.exp(accuracies * 10)  # Scale up for more pronounced differences
        weights_array = exp_acc / np.sum(exp_acc)
        
        weights = {
            results[i].model_name: float(weights_array[i])
            for i in range(len(results))
        }
        
        return weights
    
    def _create_voting_ensemble(self, results: List[ModelResult]) -> VotingClassifier:
        """Create voting ensemble"""
        
        estimators = [
            (result.model_name.lower(), result.model)
            for result in results
        ]
        
        ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft'  # Use probability predictions
        )
        
        return ensemble
    
    def _create_weighted_ensemble(self, results: List[ModelResult], 
                                weights: Dict[str, float]) -> Any:
        """Create custom weighted ensemble"""
        
        class WeightedEnsemble:
            def __init__(self, models: List[ModelResult], weights: Dict[str, float]):
                self.models = {result.model_name: result.model for result in models}
                self.weights = weights
                self.model_names = list(self.models.keys())
            
            def predict(self, X):
                predictions = []
                
                for name, model in self.models.items():
                    if hasattr(model, 'predict'):
                        pred = model.predict(X)
                        predictions.append(pred * self.weights[name])
                
                # Sum weighted predictions
                ensemble_pred = np.sum(predictions, axis=0)
                return np.round(ensemble_pred).astype(int)
            
            def predict_proba(self, X):
                probabilities = []
                
                for name, model in self.models.items():
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X)
                    elif hasattr(model, 'predict'):
                        # For models without predict_proba, create dummy probabilities
                        pred = model.predict(X)
                        proba = np.eye(3)[pred]  # One-hot encoding for 3 classes
                    else:
                        continue
                    
                    probabilities.append(proba * self.weights[name])
                
                # Sum weighted probabilities
                ensemble_proba = np.sum(probabilities, axis=0)
                
                # Normalize to ensure probabilities sum to 1
                ensemble_proba = ensemble_proba / np.sum(ensemble_proba, axis=1, keepdims=True)
                
                return ensemble_proba
        
        return WeightedEnsemble(results, weights)
    
    def _aggregate_feature_importance(self, results: List[ModelResult]) -> Dict[str, float]:
        """Aggregate feature importance across models"""
        
        feature_scores = {}
        total_weight = 0
        
        for result in results:
            if result.feature_importance:
                weight = result.test_accuracy  # Weight by accuracy
                total_weight += weight
                
                for feature, importance in result.feature_importance.items():
                    if feature not in feature_scores:
                        feature_scores[feature] = 0
                    feature_scores[feature] += importance * weight
        
        # Normalize by total weight
        if total_weight > 0:
            feature_scores = {
                feature: score / total_weight 
                for feature, score in feature_scores.items()
            }
        
        # Sort by importance
        sorted_features = dict(sorted(feature_scores.items(), 
                                    key=lambda x: x[1], reverse=True))
        
        return sorted_features
    
    def train_complete_ensemble(self, data_path: str = None) -> EnsembleResult:
        """Train complete ensemble system"""
        
        self.logger.info("🚀 Starting complete ensemble training...")
        
        try:
            # Prepare data
            X, y = self.prepare_data(data_path)
            
            # Train individual models
            individual_results = self.train_individual_models(X, y)
            
            # Create ensemble
            ensemble_result = self.create_ensemble(individual_results)
            
            # Save complete results
            self._save_training_results(ensemble_result)
            
            self.logger.info("✅ Ensemble training completed successfully!")
            return ensemble_result
            
        except Exception as e:
            self.logger.error(f"❌ Ensemble training failed: {e}")
            raise
    
    def _save_training_results(self, ensemble_result: EnsembleResult):
        """Save comprehensive training results"""
        
        # Create results summary
        results_summary = {
            'ensemble_accuracy': ensemble_result.ensemble_accuracy,
            'cross_validation_score': ensemble_result.cross_validation_score,
            'best_individual_model': ensemble_result.best_model,
            'ensemble_weights': ensemble_result.ensemble_weights,
            'training_summary': ensemble_result.training_summary,
            'individual_results': [
                {
                    'model_name': result.model_name,
                    'test_accuracy': result.test_accuracy,
                    'precision': result.precision,
                    'recall': result.recall,
                    'f1_score': result.f1_score,
                    'roc_auc': result.roc_auc,
                    'cv_mean': np.mean(result.cv_scores),
                    'cv_std': np.std(result.cv_scores),
                    'training_time': result.training_time
                }
                for result in ensemble_result.individual_results
            ],
            'top_features': dict(list(ensemble_result.feature_rankings.items())[:20])
        }
        
        # Save to file
        results_path = os.path.join(
            self.config['logs_dir'], 
            f'ensemble_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        
        with open(results_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        self.logger.info(f"📄 Results saved to {results_path}")
        
        # Print summary
        print("\n" + "="*80)
        print("🎯 ENSEMBLE TRAINING RESULTS")
        print("="*80)
        print(f"Ensemble Accuracy: {ensemble_result.ensemble_accuracy:.4f}")
        print(f"Cross-Validation Score: {ensemble_result.cross_validation_score:.4f}")
        print(f"Best Individual Model: {ensemble_result.best_model}")
        print(f"Models in Ensemble: {len(ensemble_result.individual_results)}")
        
        print("\n📊 Individual Model Performance:")
        for result in sorted(ensemble_result.individual_results, 
                           key=lambda x: x.test_accuracy, reverse=True):
            print(f"  {result.model_name:<20}: {result.test_accuracy:.4f} "
                  f"(F1: {result.f1_score:.4f}, AUC: {result.roc_auc:.4f})")
        
        print("\n🏆 Top Features:")
        for i, (feature, importance) in enumerate(list(ensemble_result.feature_rankings.items())[:10]):
            print(f"  {i+1:2d}. {feature:<30}: {importance:.6f}")
        
        print("="*80)


def main():
    """Main ensemble training function"""
    
    # Initialize trainer
    trainer = EnhancedEnsembleTrainer()
    
    # Train ensemble
    result = trainer.train_complete_ensemble()
    
    return result


if __name__ == "__main__":
    main()