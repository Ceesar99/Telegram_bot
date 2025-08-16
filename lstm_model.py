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
import os
import json

class LSTMTradingModel:
	def __init__(self):
		self.model = None
		self.price_scaler = MinMaxScaler()
		self.feature_scaler = StandardScaler()
		self.sequence_length = LSTM_CONFIG["sequence_length"]
		# Fix: Update features count to match actual features
		self.features_count = 24  # Updated to match actual feature count
		self.is_trained = False
		self.logger = self._setup_logger()
		self.calibration_temperature = 1.0
		self.feature_version = "lstm_features_v1"
		
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
		# Define all possible feature columns
		all_feature_columns = [
			'price_change', 'price_volatility', 'rsi', 'rsi_signal',
			'macd', 'macd_signal', 'macd_histogram', 'macd_crossover',
			'bb_position', 'bb_squeeze', 'stoch_k', 'stoch_d', 'stoch_signal',
			'williams_r', 'cci', 'adx', 'atr_normalized',
			'ema_9_signal', 'ema_21_signal', 'sma_10_signal', 'sma_20_signal',
			'volume_ratio', 'price_position', 'trend_strength'
		]
		
		# Ensure all columns exist, fill with 0 if missing
		for col in all_feature_columns:
			if col not in data.columns:
				data[col] = 0
		
		# Select only the required features in the correct order
		feature_data = data[all_feature_columns].fillna(0)
		
		# Ensure we have exactly 24 features
		if feature_data.shape[1] != 24:
			self.logger.error(f"Feature count mismatch: expected 24, got {feature_data.shape[1]}")
			# Pad or truncate to exactly 24 features
			if feature_data.shape[1] < 24:
				padding = np.zeros((feature_data.shape[0], 24 - feature_data.shape[1]))
				feature_data = np.hstack([feature_data, padding])
			else:
				feature_data = feature_data.iloc[:, :24]
		
		return feature_data.values
	
	def create_sequences(self, data, target):
		"""Create sequences for LSTM training"""
		sequences = []
		targets = []
		
		for i in range(self.sequence_length, len(data)):
			sequences.append(data[i-self.sequence_length:i])
			targets.append(target[i])
		
		sequences = np.array(sequences)
		targets = np.array(targets)
		
		# Ensure consistent shapes
		if len(sequences.shape) != 3:
			self.logger.error(f"Invalid sequence shape: {sequences.shape}")
			return None, None
			
		if sequences.shape[2] != self.features_count:
			self.logger.error(f"Feature count mismatch: expected {self.features_count}, got {sequences.shape[2]}")
			return None, None
		
		self.logger.info(f"Created sequences: {sequences.shape}, targets: {targets.shape}")
		return sequences, targets
	
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
	
	def generate_labels(self, data, lookahead_minutes=2, threshold_pct=0.05, spread_pct=0.01):
		"""Generate labels using profit targets including spread/slippage buffers
		threshold_pct and spread_pct are percentages (e.g., 0.05 = 0.05%)
		"""
		labels = []
		
		for i in range(len(data)):
			if i + lookahead_minutes >= len(data):
				labels.append(2)  # HOLD for insufficient data
				continue
			
			current_price = data['close'].iloc[i]
			future_price = data['close'].iloc[i + lookahead_minutes]
			
			price_change = (future_price - current_price) / current_price * 100
			# Include spread/slippage buffer
			effective_threshold = threshold_pct + spread_pct
			
			if price_change > effective_threshold:
				labels.append(0)  # BUY signal
			elif price_change < -effective_threshold:
				labels.append(1)  # SELL signal
			else:
				labels.append(2)  # HOLD signal
		
		return np.array(labels)
	
	def _compute_class_weights(self, y: np.ndarray) -> dict:
		"""Compute inverse-frequency class weights for balancing"""
		classes, counts = np.unique(y, return_counts=True)
		total = y.shape[0]
		weights = {int(c): float(total / (len(classes) * cnt)) for c, cnt in zip(classes, counts)}
		return weights
	
	def _fit_temperature(self, probs: np.ndarray, y_true: np.ndarray) -> float:
		"""Simple temperature scaling fit using grid search on NLL"""
		eps = 1e-12
		def nll_for_T(T: float) -> float:
			# Adjust probabilities using pseudo-logits
			logits = np.log(np.clip(probs, eps, 1.0))
			adj = logits / max(T, eps)
			# softmax
			exp_adj = np.exp(adj - adj.max(axis=1, keepdims=True))
			p_adj = exp_adj / np.sum(exp_adj, axis=1, keepdims=True)
			# NLL
			rows = np.arange(len(y_true))
			return -float(np.mean(np.log(np.clip(p_adj[rows, y_true], eps, 1.0))))
		
		best_T = 1.0
		best_loss = float('inf')
		for T in np.linspace(0.5, 3.0, 26):
			loss = nll_for_T(T)
			if loss < best_loss:
				best_loss = loss
				best_T = float(T)
		return best_T
	
	def train_model(self, data, validation_split=0.2, epochs=None):
		"""Train the LSTM model with comprehensive data preparation"""
		self.logger.info("Starting model training...")
		
		try:
			# Ensure models dir exists
			os.makedirs(DATABASE_CONFIG['models_dir'], exist_ok=True)
			
			# Calculate technical indicators
			processed_data = self.calculate_technical_indicators(data)
			
			# Prepare features
			features = self.prepare_features(processed_data)
			
			# Generate labels
			labels = self.generate_labels(processed_data)
			
			# Validate data shapes
			if features is None or labels is None:
				self.logger.error("Failed to prepare features or labels")
				return None
				
			if len(features) != len(labels):
				self.logger.error(f"Feature/label length mismatch: {len(features)} vs {len(labels)}")
				return None
			
			# Time-ordered split index to avoid leakage
			n = len(features)
			split_idx = int(n * (1 - validation_split))
			if split_idx <= self.sequence_length:
				self.logger.error("Insufficient data after split for sequence creation")
				return None
			
			# Fit scaler on training window only
			self.feature_scaler.fit(features[:split_idx])
			features_scaled = self.feature_scaler.transform(features)
			
			# Create sequences on fully scaled features
			X_all, y_all = self.create_sequences(features_scaled, labels)
			if X_all is None or y_all is None:
				self.logger.error("Failed to create sequences")
				return None
			
			# Split sequences chronologically
			split_idx_seq = split_idx - self.sequence_length
			X_train, y_train = X_all[:split_idx_seq], y_all[:split_idx_seq]
			X_test, y_test = X_all[split_idx_seq:], y_all[split_idx_seq:]
			
			# Guardrails: NaN checks and minimum sizes
			if np.isnan(X_train).any() or np.isnan(X_test).any():
				self.logger.error("NaNs detected in training or validation data")
				return None
			if X_train.shape[0] < 100 or X_test.shape[0] < 50:
				self.logger.error("Insufficient sequence data for training/validation")
				return None
			
			self.logger.info(f"Training data shapes - X_train: {X_train.shape}, X_val: {X_test.shape}")
			
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
			
			# Class weights for imbalance
			class_weight = self._compute_class_weights(y_train)
			
			# Train model
			epochs = epochs or LSTM_CONFIG["epochs"]
			self.logger.info(f"Starting training with {epochs} epochs")
			
			history = self.model.fit(
				X_train, y_train,
				validation_data=(X_test, y_test),
				epochs=epochs,
				batch_size=LSTM_CONFIG["batch_size"],
				callbacks=callbacks,
				verbose=1,
				class_weight=class_weight
			)
			
			# Evaluate model
			train_accuracy = self.model.evaluate(X_train, y_train, verbose=0)[1]
			test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)[1]
			
			self.logger.info(f"Training accuracy: {train_accuracy:.4f}")
			self.logger.info(f"Test accuracy: {test_accuracy:.4f}")
			
			# Probability calibration (temperature scaling on validation)
			val_probs = self.model.predict(X_test, verbose=0)
			self.calibration_temperature = self._fit_temperature(val_probs, y_test.astype(int))
			self.logger.info(f"Calibrated temperature: {self.calibration_temperature:.3f}")
			
			# Save scalers and manifest
			joblib.dump(self.feature_scaler, f"{DATABASE_CONFIG['models_dir']}/feature_scaler.pkl")
			manifest = {
				"saved_at": datetime.utcnow().isoformat(),
				"sequence_length": int(self.sequence_length),
				"features_count": int(self.features_count),
				"feature_version": self.feature_version,
				"calibration_temperature": float(self.calibration_temperature)
			}
			with open(f"{DATABASE_CONFIG['models_dir']}/lstm_manifest.json", 'w') as f:
				json.dump(manifest, f)
			
			self.is_trained = True
			return history
			
		except Exception as e:
			self.logger.error(f"Training failed with error: {e}")
			import traceback
			self.logger.error(traceback.format_exc())
			return None
	
	def _apply_temperature(self, probs: np.ndarray) -> np.ndarray:
		"""Apply stored temperature to probability vector(s)"""
		eps = 1e-12
		logits = np.log(np.clip(probs, eps, 1.0))
		adj = logits / max(self.calibration_temperature, eps)
		exp_adj = np.exp(adj - adj.max(axis=1, keepdims=True))
		return exp_adj / np.sum(exp_adj, axis=1, keepdims=True)
	
	def predict_signal(self, data):
		"""Predict trading signal for new data"""
		if not self.is_trained:
			self.logger.error("Model not trained. Please train the model first.")
			return None
		
		# Process data
		processed_data = self.calculate_technical_indicators(data)
		features = self.prepare_features(processed_data)
		
		# Scale features with guardrails
		try:
			features_scaled = self.feature_scaler.transform(features)
		except Exception as e:
			self.logger.error(f"Feature scaling failed: {e}")
			return None
		
		if np.isnan(features_scaled).any():
			self.logger.error("NaNs detected in features during prediction")
			return None
		
		# Get last sequence
		if len(features_scaled) < self.sequence_length:
			self.logger.error("Insufficient data for prediction")
			return None
		
		last_sequence = features_scaled[-self.sequence_length:].reshape(1, self.sequence_length, -1)
		
		# Make prediction
		prediction = self.model.predict(last_sequence, verbose=0)
		# Apply temperature calibration
		prediction = self._apply_temperature(prediction)
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
			},
			'feature_version': self.feature_version
		}
	
	def save_model(self, filepath=None):
		"""Save the trained model"""
		if filepath is None:
			filepath = f"{DATABASE_CONFIG['models_dir']}/lstm_trading_model.h5"
		
		# Ensure directory exists
		os.makedirs(os.path.dirname(filepath), exist_ok=True)
		
		self.model.save(filepath)
		# Also persist calibration and manifest (already saved during training)
		self.logger.info(f"Model saved to {filepath}")
	
	def load_model(self, filepath=None):
		"""Load a pre-trained model"""
		if filepath is None:
			filepath = f"{DATABASE_CONFIG['models_dir']}/best_model.h5"
		
		try:
			self.model = tf.keras.models.load_model(filepath)
			self.feature_scaler = joblib.load(f"{DATABASE_CONFIG['models_dir']}/feature_scaler.pkl")
			# Load manifest if present
			manifest_path = f"{DATABASE_CONFIG['models_dir']}/lstm_manifest.json"
			if os.path.exists(manifest_path):
				with open(manifest_path, 'r') as f:
					manifest = json.load(f)
					self.calibration_temperature = float(manifest.get('calibration_temperature', 1.0))
					self.feature_version = manifest.get('feature_version', self.feature_version)
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
		# Apply calibration
		predictions = self._apply_temperature(predictions)
		predicted_classes = np.argmax(predictions, axis=1)
		
		accuracy = accuracy_score(y, predicted_classes)
		report = classification_report(y, predicted_classes, target_names=['BUY', 'SELL', 'HOLD'])
		
		return {
			'accuracy': accuracy * 100,
			'classification_report': report,
			'confusion_matrix': confusion_matrix(y, predicted_classes)
		}