#!/usr/bin/env python3
"""
🎯 PRODUCTION MODEL TRAINER - 85%+ ACCURACY TARGET
Comprehensive training system for achieving production-ready model performance
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import talib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import asyncio
import warnings
warnings.filterwarnings('ignore')

from enhanced_data_collector import RealTimeDataCollector
from production_config import MODEL_CONFIG, TRADING_CONFIG, SIGNAL_CONFIG

class ProductionModelTrainer:
    """Advanced model trainer for achieving 85%+ accuracy"""
    
    def __init__(self):
        self.logger = logging.getLogger('ProductionModelTrainer')
        self.data_collector = RealTimeDataCollector()
        self.models = {}
        self.scalers = {}
        self.accuracy_targets = {
            'lstm': 85.0,
            'ensemble': 90.0,
            'transformer': 87.0
        }
        
    async def collect_training_data(self, symbols: List[str], days: int = 180) -> pd.DataFrame:
        """Collect 6 months of high-quality training data"""
        
        self.logger.info(f"Collecting {days} days of training data for {len(symbols)} symbols")
        
        all_data = []
        
        for symbol in symbols:
            try:
                # Get historical data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
                # Collect data from multiple timeframes
                for timeframe in ['1m', '5m', '15m']:
                    data = await self.data_collector.get_real_time_data(symbol, timeframe)
                    
                    if data is not None and len(data) > 100:
                        # Add metadata
                        data['symbol'] = symbol
                        data['timeframe'] = timeframe
                        data['timestamp'] = data.index
                        
                        all_data.append(data)
                        
                        self.logger.info(f"Collected {len(data)} records for {symbol} {timeframe}")
                    
                    # Rate limiting
                    await asyncio.sleep(1)
                    
            except Exception as e:
                self.logger.error(f"Error collecting data for {symbol}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No training data collected")
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        combined_data = combined_data.sort_values('timestamp')
        
        self.logger.info(f"Total training data collected: {len(combined_data)} records")
        return combined_data
    
    def engineer_advanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create advanced features for better model performance"""
        
        self.logger.info("Engineering advanced features for model training")
        
        # Group by symbol and timeframe for feature engineering
        processed_groups = []
        
        for (symbol, timeframe), group in data.groupby(['symbol', 'timeframe']):
            if len(group) < 100:
                continue
                
            group = group.sort_values('timestamp').copy()
            
            # Basic price features
            group['returns'] = group['close'].pct_change()
            group['log_returns'] = np.log(group['close'] / group['close'].shift(1))
            group['price_momentum'] = group['close'] / group['close'].shift(10) - 1
            
            # Volatility features
            group['volatility'] = group['returns'].rolling(20).std()
            group['volatility_ratio'] = group['volatility'] / group['volatility'].rolling(50).mean()
            
            # Technical indicators
            close_prices = group['close'].values
            high_prices = group['high'].values
            low_prices = group['low'].values
            volume = group.get('volume', pd.Series([1000] * len(group))).values
            
            # Trend indicators
            group['rsi'] = talib.RSI(close_prices, timeperiod=14)
            group['rsi_slope'] = group['rsi'].diff(5)
            
            # MACD family
            macd, macd_signal, macd_hist = talib.MACD(close_prices)
            group['macd'] = macd
            group['macd_signal'] = macd_signal
            group['macd_histogram'] = macd_hist
            group['macd_crossover'] = np.where(macd > macd_signal, 1, -1)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close_prices)
            group['bb_position'] = (close_prices - bb_lower) / (bb_upper - bb_lower)
            group['bb_width'] = (bb_upper - bb_lower) / bb_middle
            
            # Stochastic oscillators
            stoch_k, stoch_d = talib.STOCH(high_prices, low_prices, close_prices)
            group['stoch_k'] = stoch_k
            group['stoch_d'] = stoch_d
            group['stoch_divergence'] = stoch_k - stoch_d
            
            # Williams %R
            group['williams_r'] = talib.WILLR(high_prices, low_prices, close_prices)
            
            # Commodity Channel Index
            group['cci'] = talib.CCI(high_prices, low_prices, close_prices)
            
            # Average Directional Index
            group['adx'] = talib.ADX(high_prices, low_prices, close_prices)
            group['di_plus'] = talib.PLUS_DI(high_prices, low_prices, close_prices)
            group['di_minus'] = talib.MINUS_DI(high_prices, low_prices, close_prices)
            
            # Average True Range
            group['atr'] = talib.ATR(high_prices, low_prices, close_prices)
            group['atr_ratio'] = group['atr'] / close_prices
            
            # Moving averages and crossovers
            for period in [9, 21, 50]:
                ema_col = f'ema_{period}'
                group[ema_col] = talib.EMA(close_prices, timeperiod=period)
                group[f'{ema_col}_slope'] = group[ema_col].diff(5)
                group[f'price_above_{ema_col}'] = np.where(close_prices > group[ema_col], 1, 0)
            
            # EMA crossovers
            group['ema_9_21_cross'] = np.where(group['ema_9'] > group['ema_21'], 1, -1)
            group['ema_21_50_cross'] = np.where(group['ema_21'] > group['ema_50'], 1, -1)
            
            # Volume indicators (if available)
            if volume.sum() > 0:
                group['volume_sma'] = talib.SMA(volume, timeperiod=20)
                group['volume_ratio'] = volume / group['volume_sma']
                group['obv'] = talib.OBV(close_prices, volume)
                group['obv_slope'] = group['obv'].diff(5)
            else:
                group['volume_ratio'] = 1.0
                group['obv'] = 0.0
                group['obv_slope'] = 0.0
            
            # Support and resistance
            group['resistance'] = group['high'].rolling(20).max()
            group['support'] = group['low'].rolling(20).min()
            group['price_position'] = (close_prices - group['support']) / (group['resistance'] - group['support'])
            
            # Market structure
            group['higher_high'] = (group['high'] > group['high'].shift(1)).astype(int)
            group['lower_low'] = (group['low'] < group['low'].shift(1)).astype(int)
            group['trend_strength'] = group['higher_high'].rolling(10).sum() - group['lower_low'].rolling(10).sum()
            
            # Time-based features
            group['hour'] = pd.to_datetime(group['timestamp']).dt.hour
            group['day_of_week'] = pd.to_datetime(group['timestamp']).dt.dayofweek
            group['is_market_open'] = np.where(
                (group['hour'] >= 9) & (group['hour'] <= 16) & (group['day_of_week'] < 5), 1, 0
            )
            
            # Multi-timeframe features (if we have different timeframes)
            group['timeframe_encoded'] = {'1m': 1, '5m': 5, '15m': 15}.get(timeframe, 1)
            
            processed_groups.append(group)
        
        # Combine all processed groups
        result = pd.concat(processed_groups, ignore_index=True)
        result = result.fillna(method='ffill').fillna(0)
        
        self.logger.info(f"Feature engineering complete: {len(result.columns)} features created")
        return result
    
    def create_advanced_labels(self, data: pd.DataFrame, 
                              lookahead_minutes: int = 2, 
                              profit_threshold: float = 0.0008) -> np.ndarray:
        """Create advanced labels with better signal quality"""
        
        labels = []
        
        # Group by symbol for label creation
        for (symbol, timeframe), group in data.groupby(['symbol', 'timeframe']):
            group_labels = []
            
            prices = group['close'].values
            
            for i in range(len(prices)):
                if i + lookahead_minutes >= len(prices):
                    group_labels.append(2)  # HOLD
                    continue
                
                current_price = prices[i]
                future_price = prices[i + lookahead_minutes]
                
                # Calculate price change with spread consideration
                price_change = (future_price - current_price) / current_price
                
                # Dynamic threshold based on volatility
                recent_volatility = group['volatility'].iloc[max(0, i-20):i+1].mean()
                dynamic_threshold = max(profit_threshold, recent_volatility * 0.5)
                
                # Consider market conditions
                market_condition = self._assess_market_condition(group, i)
                
                # Adjust threshold based on market condition
                if market_condition == 'trending':
                    dynamic_threshold *= 0.8  # Lower threshold in trending markets
                elif market_condition == 'volatile':
                    dynamic_threshold *= 1.5  # Higher threshold in volatile markets
                
                # Generate labels
                if price_change > dynamic_threshold:
                    group_labels.append(0)  # BUY
                elif price_change < -dynamic_threshold:
                    group_labels.append(1)  # SELL
                else:
                    group_labels.append(2)  # HOLD
            
            labels.extend(group_labels)
        
        return np.array(labels)
    
    def _assess_market_condition(self, data: pd.DataFrame, index: int) -> str:
        """Assess current market condition"""
        
        if index < 20:
            return 'neutral'
        
        # Get recent data
        recent_data = data.iloc[max(0, index-20):index+1]
        
        # Check trend strength
        adx = recent_data['adx'].iloc[-1] if 'adx' in recent_data.columns else 20
        volatility = recent_data['volatility'].iloc[-1] if 'volatility' in recent_data.columns else 0.01
        
        if adx > 25:
            return 'trending'
        elif volatility > 0.02:
            return 'volatile'
        else:
            return 'ranging'
    
    def build_advanced_lstm_model(self, input_shape: Tuple[int, int]) -> Model:
        """Build advanced LSTM model with attention mechanism"""
        
        # Input layer
        inputs = Input(shape=input_shape)
        
        # First LSTM layer with attention
        lstm1 = LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.1)(inputs)
        lstm1 = BatchNormalization()(lstm1)
        
        # Second LSTM layer
        lstm2 = LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.1)(lstm1)
        lstm2 = BatchNormalization()(lstm2)
        
        # Attention mechanism
        attention = tf.keras.layers.MultiHeadAttention(
            num_heads=4, key_dim=16, dropout=0.1
        )(lstm2, lstm2)
        attention = tf.keras.layers.LayerNormalization()(attention)
        
        # Combine LSTM and attention
        combined = tf.keras.layers.Add()([lstm2, attention])
        
        # Final LSTM layer
        lstm3 = LSTM(32, return_sequences=False, dropout=0.2)(combined)
        lstm3 = BatchNormalization()(lstm3)
        
        # Dense layers with regularization
        dense1 = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(lstm3)
        dense1 = Dropout(0.3)(dense1)
        dense1 = BatchNormalization()(dense1)
        
        dense2 = Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(dense1)
        dense2 = Dropout(0.2)(dense2)
        
        # Output layer
        outputs = Dense(3, activation='softmax')(dense2)  # BUY, SELL, HOLD
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile with advanced optimizer
        optimizer = Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            clipnorm=1.0
        )
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    async def train_production_models(self) -> Dict[str, float]:
        """Train all models to production standards"""
        
        self.logger.info("Starting production model training...")
        
        # Collect training data
        symbols = TRADING_CONFIG['pairs'][:10]  # Start with top 10 pairs
        training_data = await self.collect_training_data(symbols, days=180)
        
        # Engineer features
        feature_data = self.engineer_advanced_features(training_data)
        
        # Create labels
        labels = self.create_advanced_labels(feature_data)
        
        # Prepare features for training
        feature_columns = [
            'returns', 'log_returns', 'price_momentum', 'volatility', 'volatility_ratio',
            'rsi', 'rsi_slope', 'macd', 'macd_signal', 'macd_histogram', 'macd_crossover',
            'bb_position', 'bb_width', 'stoch_k', 'stoch_d', 'stoch_divergence',
            'williams_r', 'cci', 'adx', 'di_plus', 'di_minus', 'atr_ratio',
            'ema_9', 'ema_21', 'ema_50', 'ema_9_slope', 'ema_21_slope', 'ema_50_slope',
            'price_above_ema_9', 'price_above_ema_21', 'price_above_ema_50',
            'ema_9_21_cross', 'ema_21_50_cross', 'volume_ratio', 'obv_slope',
            'price_position', 'trend_strength', 'hour', 'day_of_week', 'is_market_open'
        ]
        
        # Ensure all feature columns exist
        for col in feature_columns:
            if col not in feature_data.columns:
                feature_data[col] = 0
        
        X = feature_data[feature_columns].fillna(0).values
        y = labels
        
        # Remove samples with insufficient lookahead
        valid_indices = ~np.isnan(y)
        X = X[valid_indices]
        y = y[valid_indices]
        
        self.logger.info(f"Training data shape: X={X.shape}, y={y.shape}")
        self.logger.info(f"Label distribution: {np.bincount(y.astype(int))}")
        
        # Scale features
        scaler = RobustScaler()  # More robust to outliers than StandardScaler
        X_scaled = scaler.fit_transform(X)
        self.scalers['production_scaler'] = scaler
        
        # Create sequences for LSTM
        sequence_length = 60
        X_sequences, y_sequences = self._create_sequences(X_scaled, y, sequence_length)
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        best_accuracy = 0
        best_model = None
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_sequences)):
            self.logger.info(f"Training fold {fold + 1}/5...")
            
            X_train, X_val = X_sequences[train_idx], X_sequences[val_idx]
            y_train, y_val = y_sequences[train_idx], y_sequences[val_idx]
            
            # Build model
            model = self.build_advanced_lstm_model((sequence_length, X.shape[1]))
            
            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_accuracy',
                    patience=20,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_accuracy',
                    factor=0.5,
                    patience=10,
                    min_lr=1e-7,
                    verbose=1
                ),
                ModelCheckpoint(
                    f'/workspace/models/production_lstm_fold_{fold}.h5',
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                )
            ]
            
            # Calculate class weights for imbalanced data
            class_weights = self._calculate_class_weights(y_train)
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=32,
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=1
            )
            
            # Evaluate
            val_accuracy = max(history.history['val_accuracy'])
            self.logger.info(f"Fold {fold + 1} validation accuracy: {val_accuracy:.4f}")
            
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_model = model
        
        # Save best model
        if best_model is not None:
            best_model.save('/workspace/models/production_lstm_optimized.h5')
            self.models['lstm'] = best_model
            
            import joblib
            joblib.dump(scaler, '/workspace/models/production_scaler.pkl')
            
            self.logger.info(f"Best model saved with accuracy: {best_accuracy:.4f}")
        
        return {'lstm_accuracy': best_accuracy}
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        
        sequences = []
        targets = []
        
        for i in range(sequence_length, len(X)):
            sequences.append(X[i-sequence_length:i])
            targets.append(y[i])
        
        return np.array(sequences), np.array(targets)
    
    def _calculate_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """Calculate class weights for imbalanced data"""
        
        classes, counts = np.unique(y, return_counts=True)
        total = len(y)
        
        weights = {}
        for cls, count in zip(classes, counts):
            weights[int(cls)] = total / (len(classes) * count)
        
        return weights
    
    async def validate_model_performance(self) -> Dict[str, float]:
        """Validate model performance on out-of-sample data"""
        
        self.logger.info("Validating model performance...")
        
        # Collect fresh validation data (last 30 days)
        symbols = TRADING_CONFIG['pairs'][:5]
        validation_data = await self.collect_training_data(symbols, days=30)
        
        if validation_data.empty:
            self.logger.error("No validation data available")
            return {}
        
        # Process validation data same way as training
        feature_data = self.engineer_advanced_features(validation_data)
        labels = self.create_advanced_labels(feature_data)
        
        # Prepare features
        feature_columns = [
            'returns', 'log_returns', 'price_momentum', 'volatility', 'volatility_ratio',
            'rsi', 'rsi_slope', 'macd', 'macd_signal', 'macd_histogram', 'macd_crossover',
            'bb_position', 'bb_width', 'stoch_k', 'stoch_d', 'stoch_divergence',
            'williams_r', 'cci', 'adx', 'di_plus', 'di_minus', 'atr_ratio',
            'ema_9', 'ema_21', 'ema_50', 'ema_9_slope', 'ema_21_slope', 'ema_50_slope',
            'price_above_ema_9', 'price_above_ema_21', 'price_above_ema_50',
            'ema_9_21_cross', 'ema_21_50_cross', 'volume_ratio', 'obv_slope',
            'price_position', 'trend_strength', 'hour', 'day_of_week', 'is_market_open'
        ]
        
        for col in feature_columns:
            if col not in feature_data.columns:
                feature_data[col] = 0
        
        X = feature_data[feature_columns].fillna(0).values
        y = labels
        
        # Scale features using saved scaler
        if 'production_scaler' in self.scalers:
            X_scaled = self.scalers['production_scaler'].transform(X)
        else:
            self.logger.error("No scaler found for validation")
            return {}
        
        # Create sequences
        sequence_length = 60
        X_sequences, y_sequences = self._create_sequences(X_scaled, y, sequence_length)
        
        # Load and evaluate model
        try:
            model = tf.keras.models.load_model('/workspace/models/production_lstm_optimized.h5')
            
            # Make predictions
            predictions = model.predict(X_sequences)
            predicted_classes = np.argmax(predictions, axis=1)
            
            # Calculate metrics
            accuracy = accuracy_score(y_sequences, predicted_classes)
            
            # Calculate per-class accuracy
            buy_accuracy = accuracy_score(
                y_sequences[y_sequences == 0],
                predicted_classes[y_sequences == 0]
            ) if np.sum(y_sequences == 0) > 0 else 0
            
            sell_accuracy = accuracy_score(
                y_sequences[y_sequences == 1],
                predicted_classes[y_sequences == 1]
            ) if np.sum(y_sequences == 1) > 0 else 0
            
            self.logger.info(f"Validation Results:")
            self.logger.info(f"Overall Accuracy: {accuracy:.4f}")
            self.logger.info(f"BUY Signal Accuracy: {buy_accuracy:.4f}")
            self.logger.info(f"SELL Signal Accuracy: {sell_accuracy:.4f}")
            
            return {
                'overall_accuracy': accuracy,
                'buy_accuracy': buy_accuracy,
                'sell_accuracy': sell_accuracy,
                'production_ready': accuracy >= 0.85
            }
            
        except Exception as e:
            self.logger.error(f"Error loading model for validation: {e}")
            return {}

# Training execution script
async def main():
    """Main training execution"""
    
    trainer = ProductionModelTrainer()
    
    # Train models
    training_results = await trainer.train_production_models()
    
    # Validate performance
    validation_results = await trainer.validate_model_performance()
    
    print("Training Results:", training_results)
    print("Validation Results:", validation_results)
    
    if validation_results.get('production_ready', False):
        print("✅ Model is ready for production deployment!")
    else:
        print("⚠️ Model needs further training to reach production standards")

if __name__ == "__main__":
    asyncio.run(main())