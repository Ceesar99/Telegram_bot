#!/usr/bin/env python3
"""
🧠 ENHANCED LSTM TRAINER - PRODUCTION READY
Advanced LSTM model training with real market data, hyperparameter optimization, and validation
Designed to achieve >80% accuracy for live trading deployment
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization, Input, 
    MultiHeadAttention, LayerNormalization, Add, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, 
    TensorBoard, CSVLogger
)
from tensorflow.keras.regularizers import l1_l2
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import logging
import os
import json
import talib
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

class EnhancedLSTMTrainer:
    """
    🚀 Enhanced LSTM Trainer with Overfitting Prevention
    
    Features:
    - Proper Time Series Cross-Validation
    - Advanced Regularization Techniques  
    - Temperature Scaling for Calibration
    - Comprehensive Feature Engineering
    - Production-Ready Architecture
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._get_default_config()
        self.logger = self._setup_logger()
        
        # Model components
        self.model = None
        self.feature_scaler = RobustScaler()  # More robust to outliers
        self.target_scaler = StandardScaler()
        self.temperature = 1.0  # For calibration
        
        # Training state
        self.is_trained = False
        self.training_history = {}
        self.validation_scores = []
        self.feature_importance = {}
        
        # Create directories
        os.makedirs(self.config['model_save_dir'], exist_ok=True)
        os.makedirs(self.config['logs_dir'], exist_ok=True)
        
    def _get_default_config(self) -> Dict:
        """Enhanced configuration for overfitting prevention"""
        return {
            # Model Architecture
            'sequence_length': 60,
            'lstm_units': [128, 64, 32],  # Reduced complexity
            'dense_units': [64, 32],
            'dropout_rate': 0.3,  # Increased dropout
            'recurrent_dropout': 0.2,
            'l1_reg': 0.01,
            'l2_reg': 0.01,
            
            # Training Parameters
            'batch_size': 64,  # Larger batch size
            'epochs': 150,  # More epochs with early stopping
            'learning_rate': 0.001,
            'patience': 20,  # Early stopping patience
            'min_delta': 0.0001,
            
            # Cross-Validation
            'n_splits': 5,
            'test_size': 0.2,
            'validation_size': 0.2,
            
            # Data Parameters
            'features_count': 35,  # Expanded features
            'target_classes': 3,  # BUY, SELL, HOLD
            'min_samples': 50000,  # Minimum training samples
            
            # Paths
            'model_save_dir': '/workspace/models/enhanced/',
            'logs_dir': '/workspace/logs/enhanced_training/',
            'data_path': '/workspace/data/real_market_data/',
            
            # Feature Engineering
            'enable_technical_indicators': True,
            'enable_volatility_features': True,
            'enable_momentum_features': True,
            'enable_pattern_features': True,
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Setup comprehensive logging"""
        logger = logging.getLogger('EnhancedLSTMTrainer')
        logger.setLevel(logging.INFO)
        
        # File handler
        os.makedirs('/workspace/logs', exist_ok=True)
        fh = logging.FileHandler('/workspace/logs/enhanced_lstm_training.log')
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
    
    def load_and_prepare_data(self, data_path: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and prepare training data with advanced preprocessing
        """
        self.logger.info("🔄 Loading and preparing training data...")
        
        data_path = data_path or self.config['data_path']
        
        # Load combined market data
        combined_file = os.path.join(data_path, 'combined_market_data_20250816_092932.csv')
        
        if not os.path.exists(combined_file):
            raise FileNotFoundError(f"Training data not found: {combined_file}")
        
        # Load data in chunks to handle large files
        chunk_size = 10000
        data_chunks = []
        
        for chunk in pd.read_csv(combined_file, chunksize=chunk_size):
            data_chunks.append(chunk)
        
        df = pd.concat(data_chunks, ignore_index=True)
        self.logger.info(f"📊 Loaded {len(df):,} data points")
        
        # Data preprocessing
        df = self._preprocess_data(df)
        
        # Feature engineering
        df = self._engineer_features(df)
        
        # Remove NaN values
        df = df.dropna()
        self.logger.info(f"📊 After preprocessing: {len(df):,} clean samples")
        
        if len(df) < self.config['min_samples']:
            self.logger.warning(f"⚠️ Insufficient data: {len(df)} < {self.config['min_samples']}")
        
        # Prepare sequences
        X, y = self._create_sequences(df)
        
        self.logger.info(f"✅ Prepared {X.shape[0]:,} sequences with {X.shape[2]} features")
        return X, y
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced data preprocessing"""
        self.logger.info("🔧 Preprocessing data...")
        
        # Ensure required columns
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            self.logger.warning(f"⚠️ Missing columns: {missing_cols}")
            # Add default values for missing columns
            for col in missing_cols:
                if col == 'volume':
                    df[col] = 1000  # Default volume
                elif col == 'timestamp':
                    df[col] = pd.date_range(start='2020-01-01', periods=len(df), freq='1H')
        
        # Convert timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
        
        # Data validation
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                # Remove extreme outliers (beyond 5 standard deviations)
                mean_val = df[col].mean()
                std_val = df[col].std()
                lower_bound = mean_val - 5 * std_val
                upper_bound = mean_val + 5 * std_val
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Comprehensive feature engineering to improve model performance
        """
        self.logger.info("🔬 Engineering advanced features...")
        
        # Price-based features
        df['price_change'] = df['close'].pct_change()
        df['price_volatility'] = df['price_change'].rolling(window=20).std()
        df['price_momentum'] = df['close'] / df['close'].shift(10) - 1
        
        # Technical Indicators
        if self.config['enable_technical_indicators']:
            df = self._add_technical_indicators(df)
        
        # Volatility Features
        if self.config['enable_volatility_features']:
            df = self._add_volatility_features(df)
        
        # Momentum Features
        if self.config['enable_momentum_features']:
            df = self._add_momentum_features(df)
        
        # Pattern Features
        if self.config['enable_pattern_features']:
            df = self._add_pattern_features(df)
        
        # Create target variable (next period direction)
        df['target'] = self._create_target_variable(df)
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators"""
        
        # RSI
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        df['rsi_signal'] = np.where(df['rsi'] > 70, -1, np.where(df['rsi'] < 30, 1, 0))
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(df['close'])
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_histogram'] = macd_hist
        df['macd_crossover'] = np.where(df['macd'] > df['macd_signal'], 1, -1)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(df['close'])
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        df['bb_squeeze'] = (bb_upper - bb_lower) / bb_middle
        
        # Stochastic
        slowk, slowd = talib.STOCH(df['high'], df['low'], df['close'])
        df['stoch_k'] = slowk
        df['stoch_d'] = slowd
        
        # ADX
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'])
        
        # ATR
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'])
        df['atr_normalized'] = df['atr'] / df['close']
        
        # Williams %R
        df['williams_r'] = talib.WILLR(df['high'], df['low'], df['close'])
        
        # CCI
        df['cci'] = talib.CCI(df['high'], df['low'], df['close'])
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based features"""
        
        # Rolling volatility (multiple windows)
        for window in [5, 10, 20, 50]:
            df[f'vol_{window}'] = df['price_change'].rolling(window=window).std()
        
        # Volatility ratios
        df['vol_ratio_5_20'] = df['vol_5'] / df['vol_20']
        df['vol_ratio_10_50'] = df['vol_10'] / df['vol_50']
        
        # High-Low volatility
        df['hl_volatility'] = (df['high'] - df['low']) / df['close']
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based features"""
        
        # Price momentum (multiple periods)
        for period in [3, 5, 10, 20]:
            df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
        
        # Rate of change
        df['roc_5'] = talib.ROC(df['close'], timeperiod=5)
        df['roc_10'] = talib.ROC(df['close'], timeperiod=10)
        
        return df
    
    def _add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add pattern recognition features"""
        
        # Moving averages
        df['sma_10'] = talib.SMA(df['close'], timeperiod=10)
        df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
        df['sma_50'] = talib.SMA(df['close'], timeperiod=50)
        
        # MA crossovers
        df['ma_cross_10_20'] = np.where(df['sma_10'] > df['sma_20'], 1, -1)
        df['ma_cross_20_50'] = np.where(df['sma_20'] > df['sma_50'], 1, -1)
        
        # Price position relative to MAs
        df['price_vs_sma_10'] = (df['close'] - df['sma_10']) / df['sma_10']
        df['price_vs_sma_20'] = (df['close'] - df['sma_20']) / df['sma_20']
        
        return df
    
    def _create_target_variable(self, df: pd.DataFrame) -> np.ndarray:
        """Create target variable for classification"""
        
        # Calculate future returns
        future_returns = df['close'].shift(-1) / df['close'] - 1
        
        # Define thresholds (can be optimized)
        buy_threshold = 0.001  # 0.1% up movement
        sell_threshold = -0.001  # 0.1% down movement
        
        # Create classification targets
        targets = np.where(
            future_returns > buy_threshold, 0,  # BUY
            np.where(future_returns < sell_threshold, 1, 2)  # SELL, HOLD
        )
        
        return targets
    
    def _create_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        
        # Select feature columns (exclude target and non-numeric)
        feature_cols = [col for col in df.columns if col not in ['target', 'timestamp'] and df[col].dtype in ['float64', 'int64']]
        
        # Handle inf and -inf values
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        # Update feature count
        self.config['features_count'] = len(feature_cols)
        self.logger.info(f"📊 Using {len(feature_cols)} features: {feature_cols[:10]}...")
        
        # Scale features
        features_scaled = self.feature_scaler.fit_transform(df[feature_cols])
        targets = df['target'].values
        
        # Create sequences
        sequence_length = self.config['sequence_length']
        X, y = [], []
        
        for i in range(sequence_length, len(features_scaled)):
            X.append(features_scaled[i-sequence_length:i])
            y.append(targets[i])
        
        X = np.array(X)
        y = np.array(y)
        
        # Remove samples with NaN targets
        valid_indices = ~np.isnan(y)
        X = X[valid_indices]
        y = y[valid_indices].astype(int)
        
        return X, y
    
    def build_enhanced_model(self, input_shape: Tuple[int, int]) -> Model:
        """
        Build enhanced LSTM model with overfitting prevention
        """
        self.logger.info("🏗️ Building enhanced LSTM architecture...")
        
        # Input layer
        inputs = Input(shape=input_shape)
        
        # First LSTM layer with regularization
        x = LSTM(
            self.config['lstm_units'][0],
            return_sequences=True,
            dropout=self.config['dropout_rate'],
            recurrent_dropout=self.config['recurrent_dropout'],
            kernel_regularizer=l1_l2(self.config['l1_reg'], self.config['l2_reg'])
        )(inputs)
        x = BatchNormalization()(x)
        
        # Second LSTM layer
        x = LSTM(
            self.config['lstm_units'][1],
            return_sequences=True,
            dropout=self.config['dropout_rate'],
            recurrent_dropout=self.config['recurrent_dropout'],
            kernel_regularizer=l1_l2(self.config['l1_reg'], self.config['l2_reg'])
        )(x)
        x = BatchNormalization()(x)
        
        # Attention mechanism for better feature selection
        attention = MultiHeadAttention(
            num_heads=4,
            key_dim=32,
            dropout=self.config['dropout_rate']
        )(x, x)
        x = LayerNormalization()(attention)
        x = Add()([x, attention])  # Residual connection
        
        # Third LSTM layer
        x = LSTM(
            self.config['lstm_units'][2],
            dropout=self.config['dropout_rate'],
            recurrent_dropout=self.config['recurrent_dropout'],
            kernel_regularizer=l1_l2(self.config['l1_reg'], self.config['l2_reg'])
        )(x)
        x = BatchNormalization()(x)
        
        # Dense layers with dropout
        for units in self.config['dense_units']:
            x = Dense(
                units,
                activation='relu',
                kernel_regularizer=l1_l2(self.config['l1_reg'], self.config['l2_reg'])
            )(x)
            x = Dropout(self.config['dropout_rate'])(x)
            x = BatchNormalization()(x)
        
        # Output layer
        outputs = Dense(
            self.config['target_classes'],
            activation='softmax'
        )(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile with appropriate loss and metrics
        model.compile(
            optimizer=Adam(learning_rate=self.config['learning_rate']),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.logger.info("✅ Enhanced model architecture built successfully")
        return model
    
    def train_with_cross_validation(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Train model with proper time series cross-validation
        """
        self.logger.info("🚀 Starting enhanced training with cross-validation...")
        
        # Time Series Split
        tscv = TimeSeriesSplit(n_splits=self.config['n_splits'])
        cv_scores = []
        fold_histories = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            self.logger.info(f"🔄 Training fold {fold + 1}/{self.config['n_splits']}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Build model for this fold
            model = self.build_enhanced_model((X.shape[1], X.shape[2]))
            
            # Callbacks
            callbacks = self._get_callbacks(fold)
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.config['epochs'],
                batch_size=self.config['batch_size'],
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate fold
            val_score = model.evaluate(X_val, y_val, verbose=0)
            cv_scores.append(val_score[1])  # accuracy
            fold_histories.append(history.history)
            
            self.logger.info(f"✅ Fold {fold + 1} validation accuracy: {val_score[1]:.4f}")
        
        # Calculate average performance
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        
        self.logger.info(f"🎯 Cross-validation results:")
        self.logger.info(f"   Mean accuracy: {mean_score:.4f} ± {std_score:.4f}")
        self.logger.info(f"   Individual scores: {[f'{score:.4f}' for score in cv_scores]}")
        
        # Train final model on all data
        self.logger.info("🎯 Training final model on complete dataset...")
        
        # Split data for final training
        split_idx = int(len(X) * (1 - self.config['test_size']))
        X_train_final, X_test = X[:split_idx], X[split_idx:]
        y_train_final, y_test = y[:split_idx], y[split_idx:]
        
        # Build final model
        self.model = self.build_enhanced_model((X.shape[1], X.shape[2]))
        
        # Final training
        callbacks = self._get_callbacks('final')
        final_history = self.model.fit(
            X_train_final, y_train_final,
            validation_data=(X_test, y_test),
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        # Final evaluation
        final_score = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Temperature calibration
        self.logger.info("🌡️ Applying temperature calibration...")
        val_probs = self.model.predict(X_test)
        self.temperature = self._calibrate_temperature(val_probs, y_test)
        
        # Mark as trained
        self.is_trained = True
        
        # Save training results
        training_results = {
            'cross_validation': {
                'mean_accuracy': float(mean_score),
                'std_accuracy': float(std_score),
                'individual_scores': [float(score) for score in cv_scores]
            },
            'final_model': {
                'test_accuracy': float(final_score[1]),
                'test_loss': float(final_score[0])
            },
            'temperature': float(self.temperature),
            'config': self.config,
            'training_date': datetime.now().isoformat()
        }
        
        # Save results
        with open(os.path.join(self.config['logs_dir'], 'training_results.json'), 'w') as f:
            json.dump(training_results, f, indent=2)
        
        self.logger.info("✅ Enhanced training completed successfully!")
        return training_results
    
    def _get_callbacks(self, fold_name: str) -> List:
        """Get training callbacks"""
        callbacks = []
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=self.config['patience'],
            min_delta=self.config['min_delta'],
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Learning rate reduction
        lr_reducer = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(lr_reducer)
        
        # Model checkpoint
        checkpoint_path = os.path.join(
            self.config['model_save_dir'],
            f'best_model_fold_{fold_name}.h5'
        )
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # CSV Logger
        csv_logger = CSVLogger(
            os.path.join(self.config['logs_dir'], f'training_log_fold_{fold_name}.csv')
        )
        callbacks.append(csv_logger)
        
        return callbacks
    
    def _calibrate_temperature(self, probabilities: np.ndarray, y_true: np.ndarray) -> float:
        """Calibrate model confidence using temperature scaling"""
        
        def temperature_scale(logits: np.ndarray, temperature: float) -> np.ndarray:
            """Apply temperature scaling to logits"""
            return tf.nn.softmax(logits / temperature).numpy()
        
        # Convert probabilities to logits
        eps = 1e-12
        logits = np.log(np.clip(probabilities, eps, 1.0))
        
        best_temperature = 1.0
        best_loss = float('inf')
        
        # Grid search for best temperature
        for temp in np.linspace(0.1, 5.0, 50):
            scaled_probs = temperature_scale(logits, temp)
            
            # Calculate negative log-likelihood
            loss = 0
            for i in range(len(y_true)):
                loss -= np.log(scaled_probs[i, y_true[i]] + eps)
            loss /= len(y_true)
            
            if loss < best_loss:
                best_loss = loss
                best_temperature = temp
        
        self.logger.info(f"🌡️ Optimal temperature: {best_temperature:.3f}")
        return best_temperature
    
    def save_model(self, model_name: str = 'enhanced_lstm_model') -> str:
        """Save trained model and components"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Save model
        model_path = os.path.join(self.config['model_save_dir'], f'{model_name}.h5')
        self.model.save(model_path)
        
        # Save scaler
        scaler_path = os.path.join(self.config['model_save_dir'], f'{model_name}_scaler.pkl')
        joblib.dump(self.feature_scaler, scaler_path)
        
        # Save configuration and metadata
        metadata = {
            'model_path': model_path,
            'scaler_path': scaler_path,
            'temperature': self.temperature,
            'config': self.config,
            'features_count': self.config['features_count'],
            'save_date': datetime.now().isoformat()
        }
        
        metadata_path = os.path.join(self.config['model_save_dir'], f'{model_name}_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"✅ Model saved successfully: {model_path}")
        return model_path
    
    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with calibrated confidence"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Get raw predictions
        raw_probs = self.model.predict(X)
        
        # Apply temperature scaling
        eps = 1e-12
        logits = np.log(np.clip(raw_probs, eps, 1.0))
        calibrated_probs = tf.nn.softmax(logits / self.temperature).numpy()
        
        # Get predictions and confidence
        predictions = np.argmax(calibrated_probs, axis=1)
        confidence = np.max(calibrated_probs, axis=1)
        
        return predictions, confidence


def main():
    """Main training function"""
    logger = logging.getLogger('EnhancedLSTMTrainer')
    
    try:
        # Initialize trainer
        trainer = EnhancedLSTMTrainer()
        
        # Load and prepare data
        X, y = trainer.load_and_prepare_data()
        
        # Train with cross-validation
        results = trainer.train_with_cross_validation(X, y)
        
        # Save model
        model_path = trainer.save_model('production_enhanced_lstm')
        
        print("\n" + "="*80)
        print("🎉 ENHANCED LSTM TRAINING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"📊 Cross-validation accuracy: {results['cross_validation']['mean_accuracy']:.4f} ± {results['cross_validation']['std_accuracy']:.4f}")
        print(f"🎯 Final test accuracy: {results['final_model']['test_accuracy']:.4f}")
        print(f"🌡️ Temperature calibration: {results['temperature']:.3f}")
        print(f"💾 Model saved: {model_path}")
        print("="*80)
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()