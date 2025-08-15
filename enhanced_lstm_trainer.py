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
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Attention, MultiHeadAttention
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Concatenate, LayerNormalization
from tensorflow.keras.optimizers import Adam, RMSprop, AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.regularizers import l1_l2
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import optuna
from optuna.integration import TensorFlowPruningCallback
import logging
import json
import pickle
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

from config import DATABASE_CONFIG, TIMEZONE, LSTM_CONFIG
from real_market_data_collector import RealMarketDataCollector
from enhanced_feature_engine import EnhancedFeatureEngine, FeatureConfig
from advanced_data_validator import MarketDataValidator

@dataclass
class TrainingConfig:
    """LSTM training configuration"""
    sequence_length: int = 60
    batch_size: int = 128
    epochs: int = 200
    learning_rate: float = 0.001
    dropout_rate: float = 0.3
    l1_reg: float = 0.01
    l2_reg: float = 0.01
    validation_split: float = 0.2
    early_stopping_patience: int = 20
    reduce_lr_patience: int = 10
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-7
    use_attention: bool = True
    use_conv: bool = True
    use_multihead_attention: bool = True
    lstm_units: List[int] = field(default_factory=lambda: [128, 64, 32])
    dense_units: List[int] = field(default_factory=lambda: [64, 32])

@dataclass
class TrainingResults:
    """Training results container"""
    model: tf.keras.Model
    history: Dict[str, List[float]]
    best_accuracy: float
    best_val_accuracy: float
    training_time: float
    final_loss: float
    final_val_loss: float
    hyperparameters: Dict[str, Any]
    feature_importance: Dict[str, float]
    model_path: str

class EnhancedLSTMTrainer:
    """Enhanced LSTM training system with optimization"""
    
    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.logger = logging.getLogger('EnhancedLSTMTrainer')
        self.data_collector = RealMarketDataCollector()
        self.feature_engineer = EnhancedFeatureEngine()
        self.data_validator = MarketDataValidator()
        
        # Scalers for different data types
        self.price_scaler = RobustScaler()
        self.feature_scaler = StandardScaler()
        self.target_scaler = None
        
        # Model storage
        self.models = {}
        self.training_history = {}
        
        # Ensure directories exist
        os.makedirs('/workspace/models', exist_ok=True)
        os.makedirs('/workspace/logs/tensorboard', exist_ok=True)
        
    def prepare_training_data(self, 
                            symbols: List[str] = None,
                            timeframes: List[str] = None,
                            start_date: datetime = None,
                            end_date: datetime = None) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Prepare comprehensive training data"""
        
        self.logger.info("Starting training data preparation...")
        
        # Set defaults
        if symbols is None:
            symbols = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD"]
        
        if timeframes is None:
            timeframes = ["1h", "4h"]
        
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365*2)  # 2 years
        
        if end_date is None:
            end_date = datetime.now() - timedelta(days=30)  # Leave recent data for testing
        
        all_features = []
        all_targets = []
        
        # Collect and process data for each symbol/timeframe combination
        for symbol in symbols:
            for timeframe in timeframes:
                try:
                    self.logger.info(f"Processing {symbol} {timeframe}")
                    
                    # Check if data exists, collect if not
                    data = self.data_collector.get_collected_data(symbol, timeframe, start_date, end_date)
                    
                    if data is None or len(data) < 100:
                        self.logger.info(f"Collecting data for {symbol} {timeframe}")
                        # Collect data
                        stats = await self.data_collector.collect_historical_data(
                            symbols=[symbol],
                            timeframes=[timeframe],
                            start_date=start_date,
                            end_date=end_date
                        )
                        
                        if stats.completed_symbols == 0:
                            self.logger.warning(f"Failed to collect data for {symbol} {timeframe}")
                            continue
                        
                        data = self.data_collector.get_collected_data(symbol, timeframe, start_date, end_date)
                    
                    if data is None or len(data) < 100:
                        self.logger.warning(f"Insufficient data for {symbol} {timeframe}: {len(data) if data is not None else 0} records")
                        continue
                    
                    # Validate data quality
                    validation_report = self.data_validator.validate_dataset(data, symbol, timeframe)
                    
                    if validation_report.data_quality_score < 0.7:
                        self.logger.warning(f"Poor data quality for {symbol} {timeframe}: {validation_report.data_quality_score:.2%}")
                        # Clean the data
                        data = self.data_validator.clean_dataset(data, validation_report, aggressive_cleaning=True)
                    
                    if len(data) < 100:
                        self.logger.warning(f"Insufficient data after cleaning for {symbol} {timeframe}")
                        continue
                    
                    # Create targets (next period direction)
                    data = data.sort_index()
                    targets = (data['close'].shift(-1) > data['close']).astype(int)
                    
                    # Engineer features
                    feature_config = FeatureConfig(
                        statistical_features=True,
                        regime_detection=True,
                        feature_selection=True,
                        max_features=80
                    )
                    
                    features = self.feature_engineer.engineer_features(
                        data,
                        symbol=symbol,
                        timeframe=timeframe,
                        target=targets
                    )
                    
                    # Align features and targets
                    common_idx = features.index.intersection(targets.index)
                    features = features.loc[common_idx]
                    targets = targets.loc[common_idx]
                    
                    # Remove non-feature columns
                    exclude_cols = ['timestamp', 'symbol', 'timeframe', 'source']
                    feature_cols = [col for col in features.columns if col not in exclude_cols]
                    features = features[feature_cols]
                    
                    # Handle missing values
                    features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
                    features = features.replace([np.inf, -np.inf], 0)
                    
                    # Remove rows with NaN targets
                    valid_idx = ~targets.isna()
                    features = features[valid_idx]
                    targets = targets[valid_idx]
                    
                    if len(features) < 100:
                        self.logger.warning(f"Insufficient valid data for {symbol} {timeframe}")
                        continue
                    
                    self.logger.info(f"Processed {symbol} {timeframe}: {len(features)} samples, {len(feature_cols)} features")
                    
                    # Add to collection
                    all_features.append(features)
                    all_targets.append(targets)
                    
                except Exception as e:
                    self.logger.error(f"Error processing {symbol} {timeframe}: {e}")
                    continue
        
        if not all_features:
            raise ValueError("No valid training data collected")
        
        # Combine all data
        combined_features = pd.concat(all_features, ignore_index=True)
        combined_targets = pd.concat(all_targets, ignore_index=True)
        
        self.logger.info(f"Combined training data: {len(combined_features)} samples, {len(combined_features.columns)} features")
        
        # Scale features
        scaled_features = self.feature_scaler.fit_transform(combined_features)
        scaled_features_df = pd.DataFrame(scaled_features, columns=combined_features.columns)
        
        # Create sequences for LSTM
        X, y = self._create_sequences(scaled_features_df, combined_targets)
        
        self.logger.info(f"Created sequences: {X.shape} features, {y.shape} targets")
        
        return X, y, combined_features
    
    def _create_sequences(self, features: pd.DataFrame, targets: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        sequence_length = self.config.sequence_length
        
        X, y = [], []
        
        for i in range(sequence_length, len(features)):
            # Feature sequence
            X.append(features.iloc[i-sequence_length:i].values)
            # Target (next period)
            y.append(targets.iloc[i])
        
        return np.array(X), np.array(y)
    
    def build_enhanced_lstm_model(self, input_shape: Tuple[int, int], trial: optuna.Trial = None) -> tf.keras.Model:
        """Build enhanced LSTM model with attention and optimization"""
        
        # Hyperparameters (from trial or config)
        if trial:
            lstm_units_1 = trial.suggest_int('lstm_units_1', 64, 256)
            lstm_units_2 = trial.suggest_int('lstm_units_2', 32, 128)
            lstm_units_3 = trial.suggest_int('lstm_units_3', 16, 64)
            dense_units_1 = trial.suggest_int('dense_units_1', 32, 128)
            dense_units_2 = trial.suggest_int('dense_units_2', 16, 64)
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
            l1_reg = trial.suggest_float('l1_reg', 1e-5, 1e-2, log=True)
            l2_reg = trial.suggest_float('l2_reg', 1e-5, 1e-2, log=True)
            use_attention = trial.suggest_categorical('use_attention', [True, False])
            use_conv = trial.suggest_categorical('use_conv', [True, False])
        else:
            lstm_units_1, lstm_units_2, lstm_units_3 = self.config.lstm_units
            dense_units_1, dense_units_2 = self.config.dense_units
            dropout_rate = self.config.dropout_rate
            learning_rate = self.config.learning_rate
            l1_reg = self.config.l1_reg
            l2_reg = self.config.l2_reg
            use_attention = self.config.use_attention
            use_conv = self.config.use_conv
        
        # Input layer
        inputs = Input(shape=input_shape)
        
        # Convolutional layer for pattern recognition (optional)
        if use_conv:
            conv = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(inputs)
            conv = MaxPooling1D(pool_size=2)(conv)
            conv = Dropout(dropout_rate * 0.5)(conv)
            lstm_input = conv
        else:
            lstm_input = inputs
        
        # LSTM layers with residual connections
        lstm1 = LSTM(lstm_units_1, return_sequences=True, dropout=dropout_rate, 
                    recurrent_dropout=dropout_rate * 0.5,
                    kernel_regularizer=l1_l2(l1_reg, l2_reg))(lstm_input)
        lstm1 = BatchNormalization()(lstm1)
        
        lstm2 = LSTM(lstm_units_2, return_sequences=True, dropout=dropout_rate,
                    recurrent_dropout=dropout_rate * 0.5,
                    kernel_regularizer=l1_l2(l1_reg, l2_reg))(lstm1)
        lstm2 = BatchNormalization()(lstm2)
        
        # Attention mechanism
        if use_attention:
            if self.config.use_multihead_attention:
                attention = MultiHeadAttention(num_heads=4, key_dim=lstm_units_2//4)(lstm2, lstm2)
                attention = LayerNormalization()(attention)
                lstm2 = lstm2 + attention  # Residual connection
            else:
                attention = Attention()([lstm2, lstm2])
                lstm2 = Concatenate()([lstm2, attention])
        
        lstm3 = LSTM(lstm_units_3, return_sequences=False, dropout=dropout_rate,
                    kernel_regularizer=l1_l2(l1_reg, l2_reg))(lstm2)
        lstm3 = BatchNormalization()(lstm3)
        
        # Dense layers
        dense1 = Dense(dense_units_1, activation='relu',
                      kernel_regularizer=l1_l2(l1_reg, l2_reg))(lstm3)
        dense1 = Dropout(dropout_rate)(dense1)
        dense1 = BatchNormalization()(dense1)
        
        dense2 = Dense(dense_units_2, activation='relu',
                      kernel_regularizer=l1_l2(l1_reg, l2_reg))(dense1)
        dense2 = Dropout(dropout_rate)(dense2)
        dense2 = BatchNormalization()(dense2)
        
        # Output layer (binary classification)
        outputs = Dense(1, activation='sigmoid', name='prediction')(dense2)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        optimizer = AdamW(learning_rate=learning_rate, weight_decay=l2_reg)
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall', 'AUC']
        )
        
        return model
    
    def train_model(self, X: np.ndarray, y: np.ndarray, 
                   symbol: str = 'multi', timeframe: str = 'multi',
                   use_optuna: bool = True) -> TrainingResults:
        """Train enhanced LSTM model with optimization"""
        
        self.logger.info(f"Starting model training for {symbol} {timeframe}")
        
        if use_optuna:
            return self._train_with_optuna(X, y, symbol, timeframe)
        else:
            return self._train_single_model(X, y, symbol, timeframe)
    
    def _train_with_optuna(self, X: np.ndarray, y: np.ndarray,
                          symbol: str, timeframe: str) -> TrainingResults:
        """Train model with Optuna hyperparameter optimization"""
        
        self.logger.info("Starting Optuna hyperparameter optimization")
        
        def objective(trial):
            try:
                # Build model with trial hyperparameters
                model = self.build_enhanced_lstm_model(input_shape=(X.shape[1], X.shape[2]), trial=trial)
                
                # Split data
                split_idx = int(len(X) * (1 - self.config.validation_split))
                X_train, X_val = X[:split_idx], X[split_idx:]
                y_train, y_val = y[:split_idx], y[split_idx:]
                
                # Callbacks
                callbacks = [
                    EarlyStopping(
                        monitor='val_accuracy',
                        patience=10,
                        restore_best_weights=True,
                        mode='max'
                    ),
                    ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=5,
                        min_lr=1e-7
                    ),
                    TensorFlowPruningCallback(trial, 'val_accuracy')
                ]
                
                # Train model
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=50,  # Reduced for optimization
                    batch_size=trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
                    callbacks=callbacks,
                    verbose=0
                )
                
                # Return best validation accuracy
                best_val_accuracy = max(history.history['val_accuracy'])
                return best_val_accuracy
                
            except Exception as e:
                self.logger.error(f"Trial failed: {e}")
                return 0.0
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            study_name=f'lstm_optimization_{symbol}_{timeframe}',
            storage=f'sqlite:///optuna_study_{symbol}_{timeframe}.db',
            load_if_exists=True
        )
        
        # Optimize
        study.optimize(objective, n_trials=50, timeout=7200)  # 2 hours max
        
        self.logger.info(f"Best trial accuracy: {study.best_value:.4f}")
        self.logger.info(f"Best parameters: {study.best_params}")
        
        # Train final model with best parameters
        best_trial = study.best_trial
        final_model = self.build_enhanced_lstm_model(
            input_shape=(X.shape[1], X.shape[2]), 
            trial=best_trial
        )
        
        # Train final model
        results = self._train_single_model(X, y, symbol, timeframe, model=final_model)
        results.hyperparameters = best_trial.params
        
        return results
    
    def _train_single_model(self, X: np.ndarray, y: np.ndarray,
                           symbol: str, timeframe: str,
                           model: tf.keras.Model = None) -> TrainingResults:
        """Train a single model"""
        
        start_time = datetime.now()
        
        # Build model if not provided
        if model is None:
            model = self.build_enhanced_lstm_model(input_shape=(X.shape[1], X.shape[2]))
        
        # Split data with time series split
        tscv = TimeSeriesSplit(n_splits=3)
        best_accuracy = 0.0
        best_model = None
        best_history = None
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            self.logger.info(f"Training fold {fold + 1}/3")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Model checkpoint path
            model_path = f'/workspace/models/lstm_{symbol}_{timeframe}_fold_{fold}.h5'
            
            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_accuracy',
                    patience=self.config.early_stopping_patience,
                    restore_best_weights=True,
                    mode='max'
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=self.config.reduce_lr_factor,
                    patience=self.config.reduce_lr_patience,
                    min_lr=self.config.min_lr
                ),
                ModelCheckpoint(
                    model_path,
                    monitor='val_accuracy',
                    save_best_only=True,
                    mode='max'
                ),
                TensorBoard(
                    log_dir=f'/workspace/logs/tensorboard/{symbol}_{timeframe}_fold_{fold}',
                    histogram_freq=1
                )
            ]
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate
            val_accuracy = max(history.history['val_accuracy'])
            
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_model = tf.keras.models.load_model(model_path)
                best_history = history.history
        
        # Final evaluation
        y_pred = (best_model.predict(X) > 0.5).astype(int)
        final_accuracy = accuracy_score(y, y_pred)
        
        # Calculate training time
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Save best model
        final_model_path = f'/workspace/models/lstm_final_{symbol}_{timeframe}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.h5'
        best_model.save(final_model_path)
        
        # Save scaler
        scaler_path = f'/workspace/models/scaler_{symbol}_{timeframe}.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.feature_scaler, f)
        
        self.logger.info(f"Training completed: {final_accuracy:.4f} accuracy in {training_time:.0f}s")
        
        return TrainingResults(
            model=best_model,
            history=best_history,
            best_accuracy=final_accuracy,
            best_val_accuracy=best_accuracy,
            training_time=training_time,
            final_loss=best_history['loss'][-1],
            final_val_loss=best_history['val_loss'][-1],
            hyperparameters=self.config.__dict__,
            feature_importance={},
            model_path=final_model_path
        )
    
    def validate_model(self, model: tf.keras.Model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Comprehensive model validation"""
        
        # Predictions
        y_pred_prob = model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positive_rate': recall,
            'false_positive_rate': 1 - report['0']['recall'] if '0' in report else 0,
            'positive_predictive_value': precision,
            'negative_predictive_value': report['0']['precision'] if '0' in report else 0
        }
        
        self.logger.info(f"Validation metrics: {metrics}")
        
        return metrics
    
    def save_training_results(self, results: TrainingResults, symbol: str, timeframe: str):
        """Save training results to file"""
        
        results_dict = {
            'symbol': symbol,
            'timeframe': timeframe,
            'best_accuracy': results.best_accuracy,
            'best_val_accuracy': results.best_val_accuracy,
            'training_time': results.training_time,
            'final_loss': results.final_loss,
            'final_val_loss': results.final_val_loss,
            'hyperparameters': results.hyperparameters,
            'model_path': results.model_path,
            'training_date': datetime.now().isoformat(),
            'history': results.history
        }
        
        results_path = f'/workspace/models/training_results_{symbol}_{timeframe}.json'
        
        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        self.logger.info(f"Training results saved to {results_path}")

async def main():
    """Main training function"""
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('/workspace/logs/lstm_training.log'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger('EnhancedLSTMTrainer')
    logger.info("Starting enhanced LSTM training")
    
    try:
        # Initialize trainer
        config = TrainingConfig(
            sequence_length=60,
            batch_size=128,
            epochs=100,
            learning_rate=0.001,
            dropout_rate=0.3,
            use_attention=True,
            use_conv=True,
            use_multihead_attention=True
        )
        
        trainer = EnhancedLSTMTrainer(config)
        
        # Prepare training data
        logger.info("Preparing training data...")
        X, y, features_df = trainer.prepare_training_data(
            symbols=["EUR/USD", "GBP/USD", "USD/JPY"],
            timeframes=["1h"]
        )
        
        logger.info(f"Training data prepared: {X.shape} samples")
        
        # Split into train/test
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train model
        logger.info("Training model...")
        results = trainer.train_model(X_train, y_train, 'multi', '1h', use_optuna=True)
        
        # Validate model
        logger.info("Validating model...")
        validation_metrics = trainer.validate_model(results.model, X_test, y_test)
        
        # Save results
        trainer.save_training_results(results, 'multi', '1h')
        
        logger.info(f"Training completed successfully!")
        logger.info(f"Final accuracy: {validation_metrics['accuracy']:.4f}")
        logger.info(f"Final precision: {validation_metrics['precision']:.4f}")
        logger.info(f"Final recall: {validation_metrics['recall']:.4f}")
        logger.info(f"Final F1-score: {validation_metrics['f1_score']:.4f}")
        
        # Check if accuracy target is met
        if validation_metrics['accuracy'] >= 0.80:
            logger.info("🎉 SUCCESS: Model achieved >80% accuracy target!")
        else:
            logger.warning(f"⚠️  Model accuracy {validation_metrics['accuracy']:.2%} below 80% target")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())