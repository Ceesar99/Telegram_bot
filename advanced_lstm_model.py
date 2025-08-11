#!/usr/bin/env python3
"""
Advanced LSTM Model for Binary Options Trading
Optimized specifically for Pocket Option platform with pre-trained capabilities
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Conv1D, MaxPooling1D, Flatten, BatchNormalization, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import logging
import os
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AdvancedLSTMModel:
    def __init__(self, config=None):
        self.logger = logging.getLogger('AdvancedLSTMModel')
        self.config = config or self._get_default_config()
        self.model = None
        self.scaler = None
        self.feature_scaler = None
        self.model_path = "/workspace/models/advanced_lstm_model.h5"
        self.scaler_path = "/workspace/models/advanced_scaler.pkl"
        self.metadata_path = "/workspace/models/advanced_model_metadata.json"
        self.training_history = None
        
    def _get_default_config(self):
        """Get default model configuration optimized for binary options"""
        return {
            'sequence_length': 60,  # 1 hour of minute data
            'n_features': 25,       # Enhanced feature set
            'lstm_units': [128, 64, 32],  # Deeper network
            'cnn_filters': [32, 64],
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 200,
            'validation_split': 0.2,
            'patience': 20,
            'use_attention': True,
            'use_cnn': True,
            'binary_threshold': 0.6  # Confidence threshold for binary signals
        }
    
    def create_advanced_features(self, data):
        """Create advanced technical features optimized for binary options"""
        features = pd.DataFrame()
        
        # Price features
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        features['price_momentum'] = data['close'] / data['close'].shift(5) - 1
        
        # Volatility features
        features['volatility'] = features['returns'].rolling(window=20).std()
        features['realized_vol'] = features['log_returns'].rolling(window=20).std() * np.sqrt(252)
        
        # Technical indicators
        features['rsi'] = self._calculate_rsi(data['close'])
        features['macd'], features['macd_signal'] = self._calculate_macd(data['close'])
        features['bb_upper'], features['bb_lower'], features['bb_middle'] = self._calculate_bollinger_bands(data['close'])
        
        # Advanced momentum indicators
        features['stoch_k'], features['stoch_d'] = self._calculate_stochastic(data)
        features['williams_r'] = self._calculate_williams_r(data)
        features['cci'] = self._calculate_cci(data)
        features['adx'] = self._calculate_adx(data)
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            features[f'sma_{period}'] = data['close'].rolling(window=period).mean()
            features[f'ema_{period}'] = data['close'].ewm(span=period).mean()
        
        # Price position relative to moving averages
        features['price_to_sma20'] = data['close'] / features['sma_20'] - 1
        features['price_to_ema20'] = data['close'] / features['ema_20'] - 1
        
        # Volume features (if available)
        if 'volume' in data.columns:
            features['volume_sma'] = data['volume'].rolling(window=20).mean()
            features['volume_ratio'] = data['volume'] / features['volume_sma']
        
        # Market microstructure features
        features['hl_ratio'] = (data['high'] - data['low']) / data['close']
        features['co_ratio'] = (data['close'] - data['open']) / data['open']
        
        return features.fillna(method='ffill').fillna(0)
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal
    
    def _calculate_bollinger_bands(self, prices, period=20, std=2):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std_dev = prices.rolling(window=period).std()
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        return upper, lower, sma
    
    def _calculate_stochastic(self, data, k_period=14, d_period=3):
        """Calculate Stochastic Oscillator"""
        low_min = data['low'].rolling(window=k_period).min()
        high_max = data['high'].rolling(window=k_period).max()
        k_percent = 100 * ((data['close'] - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
    
    def _calculate_williams_r(self, data, period=14):
        """Calculate Williams %R"""
        high_max = data['high'].rolling(window=period).max()
        low_min = data['low'].rolling(window=period).min()
        return -100 * ((high_max - data['close']) / (high_max - low_min))
    
    def _calculate_cci(self, data, period=20):
        """Calculate Commodity Channel Index"""
        tp = (data['high'] + data['low'] + data['close']) / 3
        sma = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.fabs(x - x.mean()).mean())
        return (tp - sma) / (0.015 * mad)
    
    def _calculate_adx(self, data, period=14):
        """Calculate Average Directional Index"""
        high_diff = data['high'].diff()
        low_diff = data['low'].diff()
        
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        atr = self._calculate_atr(data, period)
        plus_di = 100 * pd.Series(plus_dm).rolling(window=period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(window=period).mean() / atr
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        return adx
    
    def _calculate_atr(self, data, period=14):
        """Calculate Average True Range"""
        tr1 = data['high'] - data['low']
        tr2 = np.abs(data['high'] - data['close'].shift())
        tr3 = np.abs(data['low'] - data['close'].shift())
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        return tr.rolling(window=period).mean()
    
    def create_advanced_model(self):
        """Create advanced hybrid LSTM-CNN model"""
        input_layer = Input(shape=(self.config['sequence_length'], self.config['n_features']))
        
        if self.config['use_cnn']:
            # CNN branch for pattern recognition
            cnn_branch = Conv1D(filters=self.config['cnn_filters'][0], kernel_size=3, activation='relu')(input_layer)
            cnn_branch = BatchNormalization()(cnn_branch)
            cnn_branch = MaxPooling1D(pool_size=2)(cnn_branch)
            cnn_branch = Dropout(self.config['dropout_rate'])(cnn_branch)
            
            cnn_branch = Conv1D(filters=self.config['cnn_filters'][1], kernel_size=3, activation='relu')(cnn_branch)
            cnn_branch = BatchNormalization()(cnn_branch)
            cnn_branch = MaxPooling1D(pool_size=2)(cnn_branch)
            cnn_branch = Dropout(self.config['dropout_rate'])(cnn_branch)
        
        # LSTM branch for sequence modeling
        lstm_branch = input_layer
        for i, units in enumerate(self.config['lstm_units']):
            return_sequences = i < len(self.config['lstm_units']) - 1
            lstm_branch = LSTM(units, return_sequences=return_sequences, dropout=self.config['dropout_rate'])(lstm_branch)
            lstm_branch = BatchNormalization()(lstm_branch)
        
        # Combine branches
        if self.config['use_cnn']:
            cnn_flattened = Flatten()(cnn_branch)
            combined = tf.keras.layers.concatenate([lstm_branch, cnn_flattened])
        else:
            combined = lstm_branch
        
        # Dense layers
        dense = Dense(64, activation='relu')(combined)
        dense = BatchNormalization()(dense)
        dense = Dropout(self.config['dropout_rate'])(dense)
        
        dense = Dense(32, activation='relu')(dense)
        dense = BatchNormalization()(dense)
        dense = Dropout(self.config['dropout_rate'])(dense)
        
        # Output layer for binary classification (BUY/SELL probability)
        output = Dense(3, activation='softmax', name='signal_output')(dense)
        
        model = Model(inputs=input_layer, outputs=output)
        
        # Compile with advanced optimizer
        optimizer = Adam(learning_rate=self.config['learning_rate'], beta_1=0.9, beta_2=0.999)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def prepare_data(self, data, target_column='signal'):
        """Prepare data for training"""
        # Create features
        features = self.create_advanced_features(data)
        
        # Create target (binary signals)
        if target_column not in data.columns:
            # Generate targets based on future price movement
            future_returns = data['close'].shift(-5) / data['close'] - 1
            targets = np.where(future_returns > 0.001, 2,  # BUY
                              np.where(future_returns < -0.001, 0, 1))  # SELL, HOLD
        else:
            targets = data[target_column].values
        
        # Create sequences
        X, y = self._create_sequences(features.values, targets)
        
        # Scale features
        if self.scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        else:
            X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        
        return X_scaled, y
    
    def _create_sequences(self, features, targets):
        """Create sequences for LSTM training"""
        X, y = [], []
        seq_len = self.config['sequence_length']
        
        for i in range(seq_len, len(features) - 5):  # -5 for future target
            X.append(features[i-seq_len:i])
            y.append(targets[i])
        
        return np.array(X), np.array(y)
    
    def train(self, data, validation_data=None):
        """Train the advanced model"""
        self.logger.info("Starting advanced model training...")
        
        # Prepare data
        X, y = self.prepare_data(data)
        
        # Convert targets to categorical
        y_categorical = tf.keras.utils.to_categorical(y, num_classes=3)
        
        # Create model
        self.model = self.create_advanced_model()
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=self.config['patience'], restore_best_weights=True),
            ModelCheckpoint(self.model_path, save_best_only=True, monitor='val_accuracy'),
            ReduceLROnPlateau(factor=0.5, patience=10, min_lr=0.0001)
        ]
        
        # Train
        history = self.model.fit(
            X, y_categorical,
            batch_size=self.config['batch_size'],
            epochs=self.config['epochs'],
            validation_split=self.config['validation_split'],
            callbacks=callbacks,
            verbose=1
        )
        
        self.training_history = history.history
        
        # Save model and scaler
        self.save_model()
        
        self.logger.info("Model training completed")
        return history
    
    def predict(self, data):
        """Make predictions using the trained model"""
        if self.model is None:
            self.load_model()
        
        X, _ = self.prepare_data(data)
        predictions = self.model.predict(X)
        
        # Convert to binary signals
        signals = []
        for pred in predictions:
            confidence = np.max(pred)
            if confidence >= self.config['binary_threshold']:
                signal_type = np.argmax(pred)  # 0=SELL, 1=HOLD, 2=BUY
                signals.append({
                    'signal': signal_type,
                    'confidence': confidence,
                    'probabilities': pred.tolist()
                })
            else:
                signals.append({
                    'signal': 1,  # HOLD when confidence is low
                    'confidence': confidence,
                    'probabilities': pred.tolist()
                })
        
        return signals
    
    def generate_trading_signal(self, current_data):
        """Generate real-time trading signal for binary options"""
        try:
            if len(current_data) < self.config['sequence_length']:
                return {
                    'signal': 'INSUFFICIENT_DATA',
                    'confidence': 0.0,
                    'direction': None,
                    'expiry_minutes': None
                }
            
            # Get latest prediction
            predictions = self.predict(current_data.tail(self.config['sequence_length'] + 10))
            if not predictions:
                return {
                    'signal': 'NO_SIGNAL',
                    'confidence': 0.0,
                    'direction': None,
                    'expiry_minutes': None
                }
            
            latest_prediction = predictions[-1]
            signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
            
            if latest_prediction['signal'] == 1:  # HOLD
                return {
                    'signal': 'HOLD',
                    'confidence': latest_prediction['confidence'],
                    'direction': None,
                    'expiry_minutes': None
                }
            
            # Determine expiry time based on market volatility
            volatility = self._calculate_current_volatility(current_data)
            if volatility > 0.02:
                expiry_minutes = 2  # High volatility - shorter expiry
            elif volatility > 0.01:
                expiry_minutes = 3  # Medium volatility
            else:
                expiry_minutes = 5  # Low volatility - longer expiry
            
            return {
                'signal': signal_map[latest_prediction['signal']],
                'confidence': latest_prediction['confidence'] * 100,
                'direction': signal_map[latest_prediction['signal']],
                'expiry_minutes': expiry_minutes,
                'probabilities': latest_prediction['probabilities'],
                'volatility': volatility
            }
            
        except Exception as e:
            self.logger.error(f"Error generating trading signal: {e}")
            return {
                'signal': 'ERROR',
                'confidence': 0.0,
                'direction': None,
                'expiry_minutes': None,
                'error': str(e)
            }
    
    def _calculate_current_volatility(self, data):
        """Calculate current market volatility"""
        if len(data) < 20:
            return 0.01  # Default volatility
        
        returns = data['close'].pct_change().dropna()
        return returns.tail(20).std()
    
    def save_model(self):
        """Save model, scaler, and metadata"""
        try:
            # Save model
            if self.model:
                self.model.save(self.model_path)
            
            # Save scaler
            if self.scaler:
                joblib.dump(self.scaler, self.scaler_path)
            
            # Save metadata
            metadata = {
                'config': self.config,
                'created_at': datetime.now().isoformat(),
                'model_version': '2.0.0',
                'features_count': self.config['n_features'],
                'sequence_length': self.config['sequence_length'],
                'training_history': self.training_history
            }
            
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Model saved to {self.model_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
    
    def load_model(self):
        """Load model, scaler, and metadata"""
        try:
            if os.path.exists(self.model_path):
                self.model = load_model(self.model_path)
                self.logger.info(f"Model loaded from {self.model_path}")
            
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                self.logger.info("Scaler loaded")
            
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'r') as f:
                    metadata = json.load(f)
                self.config.update(metadata.get('config', {}))
                self.logger.info("Metadata loaded")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False
    
    def evaluate_model(self, test_data):
        """Evaluate model performance"""
        if self.model is None:
            self.load_model()
        
        X_test, y_test = self.prepare_data(test_data)
        predictions = self.model.predict(X_test)
        y_pred = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'total_samples': len(y_test)
        }
        
        self.logger.info(f"Model evaluation: {metrics}")
        return metrics

def create_sample_training_data():
    """Create sample training data for testing"""
    np.random.seed(42)
    n_samples = 5000
    
    # Generate realistic price data
    price_changes = np.random.normal(0, 0.01, n_samples)
    prices = [100]
    for change in price_changes[1:]:
        prices.append(prices[-1] * (1 + change))
    
    # Create OHLC data
    data = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000, 10000, n_samples)
    })
    
    return data

if __name__ == "__main__":
    # Test the advanced model
    logging.basicConfig(level=logging.INFO)
    
    # Create and train model
    model = AdvancedLSTMModel()
    
    # Generate sample data
    sample_data = create_sample_training_data()
    
    # Train model
    history = model.train(sample_data)
    
    # Test prediction
    test_signal = model.generate_trading_signal(sample_data.tail(100))
    print(f"Generated signal: {test_signal}")
    
    print("Advanced LSTM model setup complete!")