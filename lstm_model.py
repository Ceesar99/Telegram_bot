import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import logging
from datetime import datetime, timedelta
import talib
from config import LSTM_CONFIG, TECHNICAL_INDICATORS, DATABASE_CONFIG

class LSTMTradingModel:
    def __init__(self):
        self.model = None
        self.price_scaler = MinMaxScaler()
        self.feature_scaler = StandardScaler()
        self.sequence_length = LSTM_CONFIG["sequence_length"]
        self.features_count = LSTM_CONFIG["features"]
        self.is_trained = False
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        logger = logging.getLogger('LSTM_Model')
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler('/workspace/logs/lstm_model.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def calculate_technical_indicators(self, data):
        """Calculate comprehensive technical indicators for LSTM input"""
        df = data.copy()
        
        # Price-based indicators
        df['price_change'] = df['close'].pct_change()
        df['price_volatility'] = df['price_change'].rolling(window=20).std()
        
        # RSI
        df['rsi'] = talib.RSI(df['close'], timeperiod=TECHNICAL_INDICATORS['RSI']['period'])
        df['rsi_signal'] = np.where(df['rsi'] > TECHNICAL_INDICATORS['RSI']['overbought'], -1,
                                   np.where(df['rsi'] < TECHNICAL_INDICATORS['RSI']['oversold'], 1, 0))
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(df['close'], 
                                                  fastperiod=TECHNICAL_INDICATORS['MACD']['fast'],
                                                  slowperiod=TECHNICAL_INDICATORS['MACD']['slow'],
                                                  signalperiod=TECHNICAL_INDICATORS['MACD']['signal'])
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_histogram'] = macd_hist
        df['macd_crossover'] = np.where(df['macd'] > df['macd_signal'], 1, -1)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(df['close'], 
                                                     timeperiod=TECHNICAL_INDICATORS['Bollinger_Bands']['period'],
                                                     nbdevup=TECHNICAL_INDICATORS['Bollinger_Bands']['std'],
                                                     nbdevdn=TECHNICAL_INDICATORS['Bollinger_Bands']['std'])
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        df['bb_squeeze'] = (bb_upper - bb_lower) / bb_middle
        
        # Stochastic
        slowk, slowd = talib.STOCH(df['high'], df['low'], df['close'],
                                   fastk_period=TECHNICAL_INDICATORS['Stochastic']['k_period'],
                                   slowk_period=TECHNICAL_INDICATORS['Stochastic']['d_period'],
                                   slowd_period=TECHNICAL_INDICATORS['Stochastic']['d_period'])
        df['stoch_k'] = slowk
        df['stoch_d'] = slowd
        df['stoch_signal'] = np.where(df['stoch_k'] > 80, -1, np.where(df['stoch_k'] < 20, 1, 0))
        
        # Williams %R
        df['williams_r'] = talib.WILLR(df['high'], df['low'], df['close'], 
                                      timeperiod=TECHNICAL_INDICATORS['Williams_R']['period'])
        
        # CCI
        df['cci'] = talib.CCI(df['high'], df['low'], df['close'], 
                             timeperiod=TECHNICAL_INDICATORS['CCI']['period'])
        
        # ADX
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], 
                             timeperiod=TECHNICAL_INDICATORS['ADX']['period'])
        
        # ATR
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], 
                             timeperiod=TECHNICAL_INDICATORS['ATR']['period'])
        df['atr_normalized'] = df['atr'] / df['close']
        
        # Moving Averages
        for period in TECHNICAL_INDICATORS['EMA']['periods']:
            df[f'ema_{period}'] = talib.EMA(df['close'], timeperiod=period)
            df[f'ema_{period}_signal'] = np.where(df['close'] > df[f'ema_{period}'], 1, -1)
        
        for period in TECHNICAL_INDICATORS['SMA']['periods']:
            df[f'sma_{period}'] = talib.SMA(df['close'], timeperiod=period)
            df[f'sma_{period}_signal'] = np.where(df['close'] > df[f'sma_{period}'], 1, -1)
        
        # Volume-based indicators (if volume data available)
        if 'volume' in df.columns:
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            df['obv'] = talib.OBV(df['close'], df['volume'])
        else:
            df['volume_ratio'] = 1
            df['obv'] = 0
        
        # Support and Resistance levels
        df['resistance'] = df['high'].rolling(window=20).max()
        df['support'] = df['low'].rolling(window=20).min()
        df['price_position'] = (df['close'] - df['support']) / (df['resistance'] - df['support'])
        
        # Market structure
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        df['trend_strength'] = df['higher_high'].rolling(window=10).sum() - df['lower_low'].rolling(window=10).sum()
        
        return df.fillna(0)
    
    def prepare_features(self, data):
        """Prepare feature matrix for LSTM model"""
        feature_columns = [
            'price_change', 'price_volatility', 'rsi', 'rsi_signal',
            'macd', 'macd_signal', 'macd_histogram', 'macd_crossover',
            'bb_position', 'bb_squeeze', 'stoch_k', 'stoch_d', 'stoch_signal',
            'williams_r', 'cci', 'adx', 'atr_normalized',
            'ema_9_signal', 'ema_21_signal', 'sma_10_signal', 'sma_20_signal',
            'volume_ratio', 'price_position', 'trend_strength'
        ]
        
        return data[feature_columns].values
    
    def create_sequences(self, data, target):
        """Create sequences for LSTM training"""
        sequences = []
        targets = []
        
        for i in range(self.sequence_length, len(data)):
            sequences.append(data[i-self.sequence_length:i])
            targets.append(target[i])
        
        return np.array(sequences), np.array(targets)
    
    def build_model(self):
        """Build advanced LSTM model architecture"""
        # Input layer
        input_layer = Input(shape=(self.sequence_length, self.features_count))
        
        # LSTM layers with batch normalization and dropout
        lstm1 = LSTM(LSTM_CONFIG["lstm_units"][0], return_sequences=True)(input_layer)
        lstm1 = BatchNormalization()(lstm1)
        lstm1 = Dropout(LSTM_CONFIG["dropout_rate"])(lstm1)
        
        lstm2 = LSTM(LSTM_CONFIG["lstm_units"][1], return_sequences=True)(lstm1)
        lstm2 = BatchNormalization()(lstm2)
        lstm2 = Dropout(LSTM_CONFIG["dropout_rate"])(lstm2)
        
        lstm3 = LSTM(LSTM_CONFIG["lstm_units"][2], return_sequences=False)(lstm2)
        lstm3 = BatchNormalization()(lstm3)
        lstm3 = Dropout(LSTM_CONFIG["dropout_rate"])(lstm3)
        
        # Dense layers for classification
        dense1 = Dense(64, activation='relu')(lstm3)
        dense1 = BatchNormalization()(dense1)
        dense1 = Dropout(0.3)(dense1)
        
        dense2 = Dense(32, activation='relu')(dense1)
        dense2 = BatchNormalization()(dense2)
        dense2 = Dropout(0.2)(dense2)
        
        # Output layer for binary classification (BUY/SELL)
        output = Dense(3, activation='softmax')(dense2)  # BUY, SELL, HOLD
        
        model = Model(inputs=input_layer, outputs=output)
        
        # Compile model with custom optimizer
        optimizer = Adam(learning_rate=LSTM_CONFIG["learning_rate"])
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def generate_labels(self, data, lookahead_minutes=2):
        """Generate labels for binary options trading"""
        labels = []
        
        for i in range(len(data)):
            if i + lookahead_minutes >= len(data):
                labels.append(2)  # HOLD for insufficient data
                continue
            
            current_price = data['close'].iloc[i]
            future_price = data['close'].iloc[i + lookahead_minutes]
            
            # Calculate percentage change
            price_change = (future_price - current_price) / current_price * 100
            
            # Define threshold for signal generation (0.01% minimum movement)
            threshold = 0.01
            
            if price_change > threshold:
                labels.append(0)  # BUY signal
            elif price_change < -threshold:
                labels.append(1)  # SELL signal
            else:
                labels.append(2)  # HOLD signal
        
        return np.array(labels)
    
    def train_model(self, data, validation_split=0.2, epochs=None):
        """Train the LSTM model with comprehensive data preparation"""
        self.logger.info("Starting model training...")
        
        # Calculate technical indicators
        processed_data = self.calculate_technical_indicators(data)
        
        # Prepare features
        features = self.prepare_features(processed_data)
        
        # Generate labels
        labels = self.generate_labels(processed_data)
        
        # Scale features
        features_scaled = self.feature_scaler.fit_transform(features)
        
        # Create sequences
        X, y = self.create_sequences(features_scaled, labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # Build model
        self.build_model()
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=20, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=10),
            ModelCheckpoint(
                f"{DATABASE_CONFIG['models_dir']}/best_model.h5",
                save_best_only=True,
                monitor='val_accuracy'
            )
        ]
        
        # Train model
        epochs = epochs or LSTM_CONFIG["epochs"]
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=LSTM_CONFIG["batch_size"],
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        train_accuracy = self.model.evaluate(X_train, y_train, verbose=0)[1]
        test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)[1]
        
        self.logger.info(f"Training accuracy: {train_accuracy:.4f}")
        self.logger.info(f"Test accuracy: {test_accuracy:.4f}")
        
        # Save scalers
        joblib.dump(self.feature_scaler, f"{DATABASE_CONFIG['models_dir']}/feature_scaler.pkl")
        
        self.is_trained = True
        return history
    
    def predict_signal(self, data):
        """Predict trading signal for new data"""
        if not self.is_trained:
            self.logger.error("Model not trained. Please train the model first.")
            return None
        
        # Process data
        processed_data = self.calculate_technical_indicators(data)
        features = self.prepare_features(processed_data)
        
        # Scale features
        features_scaled = self.feature_scaler.transform(features)
        
        # Get last sequence
        if len(features_scaled) < self.sequence_length:
            self.logger.error("Insufficient data for prediction")
            return None
        
        last_sequence = features_scaled[-self.sequence_length:].reshape(1, self.sequence_length, -1)
        
        # Make prediction
        prediction = self.model.predict(last_sequence, verbose=0)
        confidence = np.max(prediction[0]) * 100
        signal_class = np.argmax(prediction[0])
        
        # Map to signal
        signal_map = {0: 'BUY', 1: 'SELL', 2: 'HOLD'}
        signal = signal_map[signal_class]
        
        return {
            'signal': signal,
            'confidence': confidence,
            'probabilities': {
                'BUY': prediction[0][0] * 100,
                'SELL': prediction[0][1] * 100,
                'HOLD': prediction[0][2] * 100
            }
        }
    
    def save_model(self, filepath=None):
        """Save the trained model"""
        if filepath is None:
            filepath = f"{DATABASE_CONFIG['models_dir']}/lstm_trading_model.h5"
        
        self.model.save(filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath=None):
        """Load a pre-trained model"""
        if filepath is None:
            # Try to find any available trained model
            models_dir = DATABASE_CONFIG['models_dir']
            import os
            try:
                model_files = [f for f in os.listdir(models_dir) if f.endswith('.h5')]
                if model_files:
                    filepath = f"{models_dir}/{model_files[0]}"
                else:
                    filepath = f"{models_dir}/lstm_trading_model.h5"
            except:
                filepath = f"{models_dir}/lstm_trading_model.h5"
        
        try:
            self.model = tf.keras.models.load_model(filepath)
            try:
                self.feature_scaler = joblib.load(f"{DATABASE_CONFIG['models_dir']}/feature_scaler.pkl")
            except:
                # If scaler not found, create a default one
                from sklearn.preprocessing import StandardScaler
                self.feature_scaler = StandardScaler()
            self.is_trained = True
            self.logger.info(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    def get_model_performance(self, data):
        """Evaluate model performance on test data"""
        processed_data = self.calculate_technical_indicators(data)
        features = self.prepare_features(processed_data)
        labels = self.generate_labels(processed_data)
        
        features_scaled = self.feature_scaler.transform(features)
        X, y = self.create_sequences(features_scaled, labels)
        
        predictions = self.model.predict(X)
        predicted_classes = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(y, predicted_classes)
        report = classification_report(y, predicted_classes, target_names=['BUY', 'SELL', 'HOLD'])
        
        return {
            'accuracy': accuracy * 100,
            'classification_report': report,
            'confusion_matrix': confusion_matrix(y, predicted_classes)
        }