#!/usr/bin/env python3
"""
Binary Options AI Model - Optimized for Pocket Option
Pre-trained LSTM model specifically designed for binary options trading
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import logging
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class BinaryOptionsAIModel:
    def __init__(self):
        self.logger = logging.getLogger('BinaryOptionsAI')
        self.model = None
        self.scaler = None
        self.sequence_length = 60
        self.n_features = 20
        self.model_path = "/workspace/models/binary_options_model.h5"
        self.scaler_path = "/workspace/models/binary_scaler.pkl"
        self.metadata_path = "/workspace/models/binary_model_metadata.json"
        
        # Model configuration
        self.config = {
            'sequence_length': 60,
            'n_features': 20,
            'lstm_units': [100, 50],
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 50,
            'patience': 10
        }
    
    def create_features(self, data):
        """Create technical features for binary options trading"""
        df = data.copy()
        features = pd.DataFrame(index=df.index)
        
        # Price features
        features['price'] = df['close']
        features['price_change'] = df['close'].pct_change()
        features['price_momentum'] = df['close'] / df['close'].shift(5) - 1
        
        # Simple moving averages
        for period in [5, 10, 20]:
            features[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            features[f'price_vs_sma_{period}'] = df['close'] / features[f'sma_{period}'] - 1
        
        # Exponential moving averages
        for period in [5, 10, 20]:
            features[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            features[f'price_vs_ema_{period}'] = df['close'] / features[f'ema_{period}'] - 1
        
        # RSI
        features['rsi'] = self.calculate_rsi(df['close'])
        
        # MACD
        features['macd'] = self.calculate_macd(df['close'])
        
        # Bollinger Bands
        bb_middle = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        bb_upper = bb_middle + (bb_std * 2)
        bb_lower = bb_middle - (bb_std * 2)
        features['bb_upper'] = bb_upper
        features['bb_lower'] = bb_lower
        features['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # Volatility
        features['volatility'] = df['close'].rolling(window=20).std()
        
        # Volume features (if available)
        if 'volume' in df.columns:
            features['volume'] = df['volume']
            features['volume_sma'] = df['volume'].rolling(window=20).mean()
        else:
            features['volume'] = 1.0  # Placeholder
            features['volume_sma'] = 1.0
        
        # Market structure
        features['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        features['open_close_ratio'] = (df['close'] - df['open']) / df['open']
        
        # Ensure we have exactly n_features
        feature_columns = list(features.columns)[:self.n_features]
        features = features[feature_columns]
        
        # Fill missing values
        features = features.fillna(method='ffill').fillna(0)
        
        return features
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def calculate_macd(self, prices, fast=12, slow=26):
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd.fillna(0)
    
    def create_sequences(self, features, targets=None):
        """Create sequences for LSTM input"""
        X = []
        y = []
        
        for i in range(self.sequence_length, len(features)):
            X.append(features[i-self.sequence_length:i].values)
            if targets is not None:
                y.append(targets[i])
        
        X = np.array(X)
        if targets is not None:
            y = np.array(y)
            return X, y
        
        return X
    
    def create_targets(self, data, lookahead=5, threshold=0.0005):
        """Create binary targets based on future price movement"""
        future_prices = data['close'].shift(-lookahead)
        current_prices = data['close']
        
        # Calculate future returns
        future_returns = (future_prices - current_prices) / current_prices
        
        # Create binary targets
        # 0 = SELL (price will go down)
        # 1 = HOLD (price will stay roughly same)  
        # 2 = BUY (price will go up)
        targets = np.where(future_returns > threshold, 2,
                          np.where(future_returns < -threshold, 0, 1))
        
        return targets
    
    def build_model(self):
        """Build the LSTM model for binary options"""
        model = Sequential([
            LSTM(self.config['lstm_units'][0], 
                 return_sequences=True, 
                 input_shape=(self.sequence_length, self.n_features)),
            Dropout(self.config['dropout_rate']),
            BatchNormalization(),
            
            LSTM(self.config['lstm_units'][1], 
                 return_sequences=False),
            Dropout(self.config['dropout_rate']),
            BatchNormalization(),
            
            Dense(50, activation='relu'),
            Dropout(self.config['dropout_rate']),
            
            Dense(25, activation='relu'),
            Dropout(self.config['dropout_rate']),
            
            Dense(3, activation='softmax')  # 3 classes: SELL, HOLD, BUY
        ])
        
        optimizer = Adam(learning_rate=self.config['learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, data):
        """Train the model with market data"""
        self.logger.info("Starting model training...")
        
        # Create features and targets
        features = self.create_features(data)
        targets = self.create_targets(data)
        
        # Create sequences
        X, y = self.create_sequences(features, targets)
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        
        # Split data
        split_idx = int(len(X_scaled) * 0.8)
        X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Build model
        self.model = self.build_model()
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=self.config['patience'], restore_best_weights=True),
            ModelCheckpoint(self.model_path, save_best_only=True, monitor='val_accuracy')
        ]
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            batch_size=self.config['batch_size'],
            epochs=self.config['epochs'],
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        val_predictions = self.model.predict(X_val)
        val_pred_classes = np.argmax(val_predictions, axis=1)
        accuracy = accuracy_score(y_val, val_pred_classes)
        
        self.logger.info(f"Training completed. Validation accuracy: {accuracy:.4f}")
        
        # Save model
        self.save_model()
        
        return history
    
    def predict_signal(self, recent_data):
        """Predict trading signal for binary options"""
        try:
            if self.model is None:
                self.load_model()
            
            if len(recent_data) < self.sequence_length:
                return {
                    'signal': 'INSUFFICIENT_DATA',
                    'confidence': 0.0,
                    'direction': None,
                    'expiry_minutes': None
                }
            
            # Create features
            features = self.create_features(recent_data)
            
            # Get last sequence
            last_sequence = features.tail(self.sequence_length).values
            
            # Scale
            last_sequence_scaled = self.scaler.transform(last_sequence)
            
            # Reshape for prediction
            X = last_sequence_scaled.reshape(1, self.sequence_length, self.n_features)
            
            # Predict
            prediction = self.model.predict(X, verbose=0)[0]
            
            # Get signal
            signal_class = np.argmax(prediction)
            confidence = np.max(prediction)
            
            signal_map = {0: 'PUT', 1: 'HOLD', 2: 'CALL'}
            
            # Determine expiry based on confidence and volatility
            if confidence > 0.8:
                expiry_minutes = 2  # High confidence - shorter expiry
            elif confidence > 0.6:
                expiry_minutes = 3  # Medium confidence
            else:
                expiry_minutes = 5  # Lower confidence - longer expiry
            
            result = {
                'signal': signal_map[signal_class],
                'confidence': float(confidence * 100),
                'direction': signal_map[signal_class],
                'expiry_minutes': expiry_minutes,
                'probabilities': {
                    'PUT': float(prediction[0]),
                    'HOLD': float(prediction[1]),
                    'CALL': float(prediction[2])
                }
            }
            
            self.logger.info(f"Generated signal: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error predicting signal: {e}")
            return {
                'signal': 'ERROR',
                'confidence': 0.0,
                'direction': None,
                'expiry_minutes': None,
                'error': str(e)
            }
    
    def save_model(self):
        """Save model, scaler and metadata"""
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
                'model_version': '1.0.0',
                'model_type': 'Binary Options LSTM',
                'features_count': self.n_features,
                'sequence_length': self.sequence_length
            }
            
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Model saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
    
    def load_model(self):
        """Load pre-trained model"""
        try:
            if os.path.exists(self.model_path):
                self.model = load_model(self.model_path)
                self.logger.info("Model loaded successfully")
            
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                self.logger.info("Scaler loaded successfully")
            
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'r') as f:
                    metadata = json.load(f)
                self.config.update(metadata.get('config', {}))
                self.logger.info("Metadata loaded successfully")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False
    
    def is_model_trained(self):
        """Check if model is trained and available"""
        return (os.path.exists(self.model_path) and 
                os.path.exists(self.scaler_path) and 
                os.path.exists(self.metadata_path))

def create_realistic_market_data(n_samples=10000):
    """Create realistic market data for training"""
    np.random.seed(42)
    
    # Generate realistic price movements
    base_price = 1.2000  # EUR/USD starting price
    returns = np.random.normal(0, 0.001, n_samples)  # Small random returns
    
    # Add some trending behavior
    trend = np.sin(np.arange(n_samples) * 2 * np.pi / 1000) * 0.0005
    returns += trend
    
    # Add volatility clustering
    volatility = np.abs(np.random.normal(0, 0.0005, n_samples))
    returns *= (1 + volatility)
    
    # Calculate prices
    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Create OHLC data
    data = []
    for i, price in enumerate(prices):
        high = price * (1 + abs(np.random.normal(0, 0.0002)))
        low = price * (1 - abs(np.random.normal(0, 0.0002)))
        open_price = prices[i-1] if i > 0 else price
        
        data.append({
            'timestamp': pd.Timestamp.now() - pd.Timedelta(minutes=n_samples-i),
            'open': open_price,
            'high': high,
            'low': low,
            'close': price,
            'volume': np.random.randint(100, 1000)
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    return df

def main():
    """Main function to train and test the model"""
    logging.basicConfig(level=logging.INFO)
    
    # Create AI model
    ai_model = BinaryOptionsAIModel()
    
    # Check if model already exists
    if ai_model.is_model_trained():
        print("✅ Pre-trained model found, loading...")
        ai_model.load_model()
    else:
        print("🔄 Training new model...")
        
        # Create training data
        training_data = create_realistic_market_data(10000)
        
        # Train model
        history = ai_model.train(training_data)
        print("✅ Model training completed!")
    
    # Test prediction
    test_data = create_realistic_market_data(100)
    signal = ai_model.predict_signal(test_data)
    
    print(f"\n🎯 Test Signal Generated:")
    print(f"Direction: {signal['direction']}")
    print(f"Confidence: {signal['confidence']:.1f}%")
    print(f"Expiry: {signal['expiry_minutes']} minutes")
    print(f"Probabilities: {signal.get('probabilities', {})}")
    
    return ai_model

if __name__ == "__main__":
    model = main()