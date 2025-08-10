import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import joblib
import json
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization, Input, 
    MultiHeadAttention, LayerNormalization, Add
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import optuna
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from config import DATABASE_CONFIG, LSTM_CONFIG
from advanced_features import AdvancedFeatureEngine

@dataclass
class ModelPrediction:
    """Container for model predictions"""
    model_name: str
    prediction: int  # 0=BUY, 1=SELL, 2=HOLD
    confidence: float
    probabilities: np.ndarray
    features_used: int
    processing_time: float

@dataclass
class EnsemblePrediction:
    """Container for ensemble predictions"""
    final_prediction: int
    final_confidence: float
    individual_predictions: List[ModelPrediction]
    meta_features: Dict[str, float]
    consensus_level: float
    processing_time: float

class LSTMTrendModel:
    """Advanced LSTM model for trend prediction"""
    
    def __init__(self, config: Dict = None):
        self.config = config or LSTM_CONFIG
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.logger = logging.getLogger('LSTMTrendModel')
        
    def build_model(self, input_shape: Tuple[int, int]):
        """Build advanced LSTM architecture"""
        
        # Input layer
        inputs = Input(shape=input_shape)
        
        # Multi-layer LSTM with attention
        lstm1 = LSTM(128, return_sequences=True, dropout=0.2)(inputs)
        lstm1 = BatchNormalization()(lstm1)
        
        lstm2 = LSTM(64, return_sequences=True, dropout=0.2)(lstm1)
        lstm2 = BatchNormalization()(lstm2)
        
        # Attention mechanism
        attention = MultiHeadAttention(num_heads=4, key_dim=16)(lstm2, lstm2)
        attention = LayerNormalization()(attention)
        
        # Combine LSTM and attention
        combined = Add()([lstm2, attention])
        
        # Final LSTM layer
        lstm3 = LSTM(32, return_sequences=False, dropout=0.2)(combined)
        lstm3 = BatchNormalization()(lstm3)
        
        # Dense layers
        dense1 = Dense(64, activation='relu')(lstm3)
        dense1 = Dropout(0.3)(dense1)
        dense1 = BatchNormalization()(dense1)
        
        dense2 = Dense(32, activation='relu')(dense1)
        dense2 = Dropout(0.2)(dense2)
        
        # Output layer
        outputs = Dense(3, activation='softmax')(dense2)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, X: np.ndarray, y: np.ndarray, validation_data: Tuple = None):
        """Train the LSTM model"""
        try:
            if self.model is None:
                self.build_model((X.shape[1], X.shape[2]))
            
            callbacks = [
                EarlyStopping(patience=15, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-6)
            ]
            
            history = self.model.fit(
                X, y,
                validation_data=validation_data,
                epochs=100,
                batch_size=32,
                callbacks=callbacks,
                verbose=0
            )
            
            self.is_trained = True
            return history
            
        except Exception as e:
            self.logger.error(f"Error training LSTM model: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> ModelPrediction:
        """Make prediction with timing"""
        start_time = datetime.now()
        
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        try:
            probabilities = self.model.predict(X, verbose=0)[0]
            prediction = np.argmax(probabilities)
            confidence = float(np.max(probabilities))
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ModelPrediction(
                model_name="LSTM_Trend",
                prediction=prediction,
                confidence=confidence,
                probabilities=probabilities,
                features_used=X.shape[-1],
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Error in LSTM prediction: {e}")
            raise

class XGBoostFeatureModel:
    """XGBoost model optimized for feature-based predictions"""
    
    def __init__(self):
        self.model = None
        self.scaler = RobustScaler()
        self.is_trained = False
        self.logger = logging.getLogger('XGBoostFeatureModel')
        self.best_params = None
        
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray, n_trials: int = 50):
        """Optimize hyperparameters using Optuna"""
        
        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
                'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1.0),
                'random_state': 42
            }
            
            model = xgb.XGBClassifier(**params)
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
            
            return scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        self.best_params = study.best_params
        self.logger.info(f"Best XGBoost params: {self.best_params}")
        
        return self.best_params
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train XGBoost model"""
        try:
            # Optimize hyperparameters if not done
            if self.best_params is None:
                self.optimize_hyperparameters(X, y)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model = xgb.XGBClassifier(**self.best_params)
            self.model.fit(X_scaled, y)
            
            self.is_trained = True
            
        except Exception as e:
            self.logger.error(f"Error training XGBoost model: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> ModelPrediction:
        """Make prediction"""
        start_time = datetime.now()
        
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        try:
            X_scaled = self.scaler.transform(X)
            probabilities = self.model.predict_proba(X_scaled)[0]
            prediction = np.argmax(probabilities)
            confidence = float(np.max(probabilities))
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ModelPrediction(
                model_name="XGBoost_Features",
                prediction=prediction,
                confidence=confidence,
                probabilities=probabilities,
                features_used=X.shape[-1],
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Error in XGBoost prediction: {e}")
            raise

class TransformerModel:
    """Transformer model for sequence analysis"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.logger = logging.getLogger('TransformerModel')
    
    def build_transformer_block(self, inputs, head_size, num_heads, ff_dim, dropout=0.1):
        """Build transformer block"""
        
        # Multi-head attention
        attention = MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=head_size,
            dropout=dropout
        )(inputs, inputs)
        
        attention = Dropout(dropout)(attention)
        attention = LayerNormalization(epsilon=1e-6)(attention)
        
        # Residual connection
        res = Add()([inputs, attention])
        
        # Feed forward network
        ff = Dense(ff_dim, activation='relu')(res)
        ff = Dropout(dropout)(ff)
        ff = Dense(inputs.shape[-1])(ff)
        ff = Dropout(dropout)(ff)
        ff = LayerNormalization(epsilon=1e-6)(ff)
        
        # Residual connection
        return Add()([res, ff])
    
    def build_model(self, input_shape: Tuple[int, int]):
        """Build transformer model"""
        
        inputs = Input(shape=input_shape)
        
        # Positional encoding (simplified)
        x = inputs
        
        # Multiple transformer blocks
        x = self.build_transformer_block(x, head_size=64, num_heads=4, ff_dim=128)
        x = self.build_transformer_block(x, head_size=64, num_heads=4, ff_dim=128)
        
        # Global average pooling
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # Classification head
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(3, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, X: np.ndarray, y: np.ndarray, validation_data: Tuple = None):
        """Train transformer model"""
        try:
            if self.model is None:
                self.build_model((X.shape[1], X.shape[2]))
            
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=5)
            ]
            
            history = self.model.fit(
                X, y,
                validation_data=validation_data,
                epochs=50,
                batch_size=32,
                callbacks=callbacks,
                verbose=0
            )
            
            self.is_trained = True
            return history
            
        except Exception as e:
            self.logger.error(f"Error training Transformer model: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> ModelPrediction:
        """Make prediction"""
        start_time = datetime.now()
        
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        try:
            probabilities = self.model.predict(X, verbose=0)[0]
            prediction = np.argmax(probabilities)
            confidence = float(np.max(probabilities))
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ModelPrediction(
                model_name="Transformer",
                prediction=prediction,
                confidence=confidence,
                probabilities=probabilities,
                features_used=X.shape[-1],
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Error in Transformer prediction: {e}")
            raise

class RandomForestRegimeModel:
    """Random Forest model for market regime detection"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.logger = logging.getLogger('RandomForestRegimeModel')
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train Random Forest model"""
        try:
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Optimize parameters
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
            
            self.model.fit(X_scaled, y)
            self.is_trained = True
            
        except Exception as e:
            self.logger.error(f"Error training Random Forest model: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> ModelPrediction:
        """Make prediction"""
        start_time = datetime.now()
        
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        try:
            X_scaled = self.scaler.transform(X)
            probabilities = self.model.predict_proba(X_scaled)[0]
            prediction = np.argmax(probabilities)
            confidence = float(np.max(probabilities))
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ModelPrediction(
                model_name="RandomForest_Regime",
                prediction=prediction,
                confidence=confidence,
                probabilities=probabilities,
                features_used=X.shape[-1],
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Error in Random Forest prediction: {e}")
            raise

class SVMRegimeModel:
    """SVM model for regime classification"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.logger = logging.getLogger('SVMRegimeModel')
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train SVM model"""
        try:
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Use RBF kernel with probability estimates
            self.model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            )
            
            self.model.fit(X_scaled, y)
            self.is_trained = True
            
        except Exception as e:
            self.logger.error(f"Error training SVM model: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> ModelPrediction:
        """Make prediction"""
        start_time = datetime.now()
        
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        try:
            X_scaled = self.scaler.transform(X)
            probabilities = self.model.predict_proba(X_scaled)[0]
            prediction = np.argmax(probabilities)
            confidence = float(np.max(probabilities))
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ModelPrediction(
                model_name="SVM_Regime",
                prediction=prediction,
                confidence=confidence,
                probabilities=probabilities,
                features_used=X.shape[-1],
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Error in SVM prediction: {e}")
            raise

class MetaLearnerModel:
    """Meta-learner that combines predictions from base models"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.logger = logging.getLogger('MetaLearnerModel')
        self.feature_importance = {}
    
    def extract_meta_features(self, predictions: List[ModelPrediction]) -> np.ndarray:
        """Extract meta-features from base model predictions"""
        features = []
        
        # Individual model confidences
        for pred in predictions:
            features.append(pred.confidence)
        
        # Model agreement features
        pred_classes = [pred.prediction for pred in predictions]
        
        # Consensus level
        consensus = max(pred_classes.count(0), pred_classes.count(1), pred_classes.count(2)) / len(pred_classes)
        features.append(consensus)
        
        # Confidence spread
        confidences = [pred.confidence for pred in predictions]
        features.append(np.std(confidences))
        features.append(np.mean(confidences))
        features.append(np.max(confidences) - np.min(confidences))
        
        # Processing time features
        times = [pred.processing_time for pred in predictions]
        features.append(np.mean(times))
        
        # Probability distributions
        for pred in predictions:
            features.extend(pred.probabilities)
        
        return np.array(features).reshape(1, -1)
    
    def train(self, meta_features: np.ndarray, y: np.ndarray):
        """Train meta-learner"""
        try:
            # Scale features
            X_scaled = self.scaler.fit_transform(meta_features)
            
            # Use XGBoost as meta-learner
            self.model = xgb.XGBClassifier(
                max_depth=6,
                learning_rate=0.1,
                n_estimators=100,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            
            self.model.fit(X_scaled, y)
            self.is_trained = True
            
            # Store feature importance
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = dict(enumerate(self.model.feature_importances_))
            
        except Exception as e:
            self.logger.error(f"Error training meta-learner: {e}")
            raise
    
    def predict(self, meta_features: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """Make meta-prediction"""
        if not self.is_trained:
            raise ValueError("Meta-learner not trained")
        
        try:
            X_scaled = self.scaler.transform(meta_features)
            probabilities = self.model.predict_proba(X_scaled)[0]
            prediction = np.argmax(probabilities)
            confidence = float(np.max(probabilities))
            
            return prediction, confidence, probabilities
            
        except Exception as e:
            self.logger.error(f"Error in meta-learner prediction: {e}")
            raise

class EnsembleSignalGenerator:
    """Main ensemble model that combines all base models"""
    
    def __init__(self):
        self.logger = logging.getLogger('EnsembleSignalGenerator')
        
        # Initialize base models
        self.models = {
            'lstm_trend': LSTMTrendModel(),
            'xgboost_features': XGBoostFeatureModel(),
            'transformer': TransformerModel(),
            'random_forest': RandomForestRegimeModel(),
            'svm_regime': SVMRegimeModel()
        }
        
        # Meta-learner
        self.meta_learner = MetaLearnerModel()
        
        # Feature engine
        self.feature_engine = AdvancedFeatureEngine()
        
        self.is_trained = False
        self.training_history = {}
    
    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for different model types"""
        
        # Generate advanced features
        processed_data = self.feature_engine.generate_all_features(data)
        
        # Prepare sequence data for LSTM/Transformer
        sequence_length = 60
        feature_columns = [col for col in processed_data.columns if col not in ['target', 'timestamp']]
        
        sequences = []
        flat_features = []
        targets = []
        
        for i in range(sequence_length, len(processed_data)):
            # Sequence data for LSTM/Transformer
            seq = processed_data[feature_columns].iloc[i-sequence_length:i].values
            sequences.append(seq)
            
            # Flat features for tree-based models
            flat_feat = processed_data[feature_columns].iloc[i].values
            flat_features.append(flat_feat)
            
            # Target
            targets.append(processed_data['target'].iloc[i])
        
        sequence_data = np.array(sequences)
        flat_data = np.array(flat_features)
        targets = np.array(targets)
        
        return sequence_data, flat_data, targets
    
    def train_ensemble(self, data: pd.DataFrame, validation_split: float = 0.2):
        """Train all models in the ensemble"""
        try:
            self.logger.info("Starting ensemble training...")
            
            # Prepare data
            sequence_data, flat_data, targets = self.prepare_data(data)
            
            # Split data
            split_idx = int(len(sequence_data) * (1 - validation_split))
            
            seq_train, seq_val = sequence_data[:split_idx], sequence_data[split_idx:]
            flat_train, flat_val = flat_data[:split_idx], flat_data[split_idx:]
            y_train, y_val = targets[:split_idx], targets[split_idx:]
            
            # Train sequence models (LSTM, Transformer)
            self.logger.info("Training LSTM model...")
            lstm_history = self.models['lstm_trend'].train(
                seq_train, y_train, 
                validation_data=(seq_val, y_val)
            )
            self.training_history['lstm_trend'] = lstm_history
            
            self.logger.info("Training Transformer model...")
            transformer_history = self.models['transformer'].train(
                seq_train, y_train,
                validation_data=(seq_val, y_val)
            )
            self.training_history['transformer'] = transformer_history
            
            # Train feature-based models
            self.logger.info("Training XGBoost model...")
            self.models['xgboost_features'].train(flat_train, y_train)
            
            self.logger.info("Training Random Forest model...")
            self.models['random_forest'].train(flat_train, y_train)
            
            self.logger.info("Training SVM model...")
            self.models['svm_regime'].train(flat_train, y_train)
            
            # Generate meta-training data
            self.logger.info("Training meta-learner...")
            meta_features_list = []
            meta_targets = []
            
            for i in range(len(seq_val)):
                try:
                    # Get predictions from all base models
                    predictions = []
                    
                    # Sequence models
                    lstm_pred = self.models['lstm_trend'].predict(seq_val[i:i+1])
                    predictions.append(lstm_pred)
                    
                    transformer_pred = self.models['transformer'].predict(seq_val[i:i+1])
                    predictions.append(transformer_pred)
                    
                    # Feature models
                    xgb_pred = self.models['xgboost_features'].predict(flat_val[i:i+1])
                    predictions.append(xgb_pred)
                    
                    rf_pred = self.models['random_forest'].predict(flat_val[i:i+1])
                    predictions.append(rf_pred)
                    
                    svm_pred = self.models['svm_regime'].predict(flat_val[i:i+1])
                    predictions.append(svm_pred)
                    
                    # Extract meta-features
                    meta_features = self.meta_learner.extract_meta_features(predictions)
                    meta_features_list.append(meta_features[0])
                    meta_targets.append(y_val[i])
                    
                except Exception as e:
                    self.logger.warning(f"Error generating meta-features for sample {i}: {e}")
                    continue
            
            if meta_features_list:
                meta_X = np.array(meta_features_list)
                meta_y = np.array(meta_targets)
                self.meta_learner.train(meta_X, meta_y)
            
            self.is_trained = True
            self.logger.info("Ensemble training completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error training ensemble: {e}")
            raise
    
    def predict(self, data: pd.DataFrame) -> EnsemblePrediction:
        """Generate ensemble prediction"""
        start_time = datetime.now()
        
        if not self.is_trained:
            raise ValueError("Ensemble not trained")
        
        try:
            # Prepare data
            sequence_data, flat_data, _ = self.prepare_data(data)
            
            # Get last sample for prediction
            seq_sample = sequence_data[-1:] if len(sequence_data) > 0 else None
            flat_sample = flat_data[-1:] if len(flat_data) > 0 else None
            
            if seq_sample is None or flat_sample is None:
                raise ValueError("Insufficient data for prediction")
            
            # Get predictions from all base models
            predictions = []
            
            try:
                lstm_pred = self.models['lstm_trend'].predict(seq_sample)
                predictions.append(lstm_pred)
            except Exception as e:
                self.logger.warning(f"LSTM prediction failed: {e}")
            
            try:
                transformer_pred = self.models['transformer'].predict(seq_sample)
                predictions.append(transformer_pred)
            except Exception as e:
                self.logger.warning(f"Transformer prediction failed: {e}")
            
            try:
                xgb_pred = self.models['xgboost_features'].predict(flat_sample)
                predictions.append(xgb_pred)
            except Exception as e:
                self.logger.warning(f"XGBoost prediction failed: {e}")
            
            try:
                rf_pred = self.models['random_forest'].predict(flat_sample)
                predictions.append(rf_pred)
            except Exception as e:
                self.logger.warning(f"Random Forest prediction failed: {e}")
            
            try:
                svm_pred = self.models['svm_regime'].predict(flat_sample)
                predictions.append(svm_pred)
            except Exception as e:
                self.logger.warning(f"SVM prediction failed: {e}")
            
            if not predictions:
                raise ValueError("No valid predictions from base models")
            
            # Calculate consensus
            pred_classes = [pred.prediction for pred in predictions]
            consensus_level = max(pred_classes.count(0), pred_classes.count(1), pred_classes.count(2)) / len(pred_classes)
            
            # Meta-learner prediction
            try:
                meta_features = self.meta_learner.extract_meta_features(predictions)
                final_prediction, final_confidence, _ = self.meta_learner.predict(meta_features)
            except Exception as e:
                self.logger.warning(f"Meta-learner failed, using voting: {e}")
                # Fallback to weighted voting
                final_prediction = max(set(pred_classes), key=pred_classes.count)
                confidences = [pred.confidence for pred in predictions]
                final_confidence = np.mean(confidences)
            
            # Extract meta-features for reporting
            meta_features_dict = {
                'consensus_level': consensus_level,
                'avg_confidence': np.mean([p.confidence for p in predictions]),
                'confidence_std': np.std([p.confidence for p in predictions]),
                'model_count': len(predictions)
            }
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return EnsemblePrediction(
                final_prediction=final_prediction,
                final_confidence=final_confidence,
                individual_predictions=predictions,
                meta_features=meta_features_dict,
                consensus_level=consensus_level,
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Error in ensemble prediction: {e}")
            raise
    
    def get_model_performance(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """Evaluate individual model performance"""
        try:
            sequence_data, flat_data, targets = self.prepare_data(data)
            
            performance = {}
            
            # Evaluate each model
            for model_name, model in self.models.items():
                if not model.is_trained:
                    continue
                
                try:
                    predictions = []
                    
                    if model_name in ['lstm_trend', 'transformer']:
                        # Sequence models
                        for i in range(len(sequence_data)):
                            pred = model.predict(sequence_data[i:i+1])
                            predictions.append(pred.prediction)
                    else:
                        # Feature models
                        for i in range(len(flat_data)):
                            pred = model.predict(flat_data[i:i+1])
                            predictions.append(pred.prediction)
                    
                    if predictions:
                        accuracy = accuracy_score(targets[:len(predictions)], predictions)
                        precision = precision_score(targets[:len(predictions)], predictions, average='weighted')
                        recall = recall_score(targets[:len(predictions)], predictions, average='weighted')
                        f1 = f1_score(targets[:len(predictions)], predictions, average='weighted')
                        
                        performance[model_name] = {
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall,
                            'f1_score': f1,
                            'predictions_count': len(predictions)
                        }
                    
                except Exception as e:
                    self.logger.warning(f"Error evaluating {model_name}: {e}")
                    performance[model_name] = {'error': str(e)}
            
            return performance
            
        except Exception as e:
            self.logger.error(f"Error evaluating model performance: {e}")
            return {}
    
    def save_ensemble(self, filepath: str):
        """Save trained ensemble"""
        try:
            # Save each model
            for model_name, model in self.models.items():
                if hasattr(model, 'model') and model.model is not None:
                    if hasattr(model.model, 'save'):
                        # TensorFlow model
                        model.model.save(f"{filepath}_{model_name}.h5")
                    else:
                        # Scikit-learn/XGBoost model
                        joblib.dump(model, f"{filepath}_{model_name}.pkl")
            
            # Save meta-learner
            if self.meta_learner.is_trained:
                joblib.dump(self.meta_learner, f"{filepath}_meta_learner.pkl")
            
            # Save training history
            with open(f"{filepath}_training_history.json", 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                history_json = {}
                for key, value in self.training_history.items():
                    if hasattr(value, 'history'):
                        history_json[key] = {k: [float(x) for x in v] for k, v in value.history.items()}
                json.dump(history_json, f)
            
            self.logger.info(f"Ensemble saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving ensemble: {e}")
            raise
    
    def load_ensemble(self, filepath: str):
        """Load trained ensemble"""
        try:
            # Load each model (implementation depends on specific requirements)
            # This would need to be implemented based on the saved model formats
            
            self.logger.info(f"Ensemble loaded from {filepath}")
            self.is_trained = True
            
        except Exception as e:
            self.logger.error(f"Error loading ensemble: {e}")
            raise