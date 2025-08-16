#!/usr/bin/env python3
"""
🧠 ENSEMBLE MODELS TRAINING WITH REAL MARKET DATA
Train XGBoost, Random Forest, and SVM models with real market data
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import joblib
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb

# Add project path
sys.path.append('/workspace')
from advanced_features import AdvancedFeatureEngine

class RealDataEnsembleTrainer:
    """Train ensemble models with real market data"""
    
    def __init__(self):
        self.setup_logging()
        self.feature_engine = AdvancedFeatureEngine()
        self.models = {}
        self.scalers = {}
        self.performance_metrics = {}
        
    def setup_logging(self):
        """Setup logging"""
        log_dir = "/workspace/logs"
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{log_dir}/ensemble_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('EnsembleTrainer')
        
    def load_and_prepare_data(self, data_file):
        """Load and prepare real market data"""
        self.logger.info(f"📊 Loading real market data from {data_file}")
        
        # Load data
        data = pd.read_csv(data_file)
        self.logger.info(f"✅ Loaded {len(data):,} records")
        
        # Convert datetime
        if 'datetime' in data.columns:
            data['datetime'] = pd.to_datetime(data['datetime'])
        else:
            data['datetime'] = pd.to_datetime(data['date'])
            
        # Sort by datetime
        data = data.sort_values(['symbol', 'datetime']).reset_index(drop=True)
        
        # Focus on main symbols for training
        main_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
        data = data[data['symbol'].isin(main_symbols)].copy()
        
        self.logger.info(f"📈 Using {len(data):,} records from {len(data['symbol'].unique())} symbols")
        return data
        
    def engineer_features(self, data):
        """Engineer comprehensive features"""
        self.logger.info("🔧 Engineering advanced features...")
        
        all_features = []
        
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol].copy()
            
            if len(symbol_data) < 50:  # Skip if too few records
                continue
                
            # Calculate returns
            symbol_data['returns'] = symbol_data['close'].pct_change()
            symbol_data['future_returns'] = symbol_data['returns'].shift(-1)  # Target variable
            
            # Technical indicators using AdvancedFeatureEngine
            try:
                features = self.feature_engine.generate_all_features(symbol_data, symbol)
                if features is not None and not features.empty:
                    features['symbol'] = symbol
                    all_features.append(features)
                    self.logger.info(f"✅ {symbol}: {len(features)} records with {len(features.columns)} features")
            except Exception as e:
                self.logger.warning(f"⚠️ {symbol}: Feature engineering failed - {e}")
                # Create simple features as backup
                simple_features = self._create_simple_features(symbol_data)
                if simple_features is not None and not simple_features.empty:
                    simple_features['symbol'] = symbol
                    all_features.append(simple_features)
                    self.logger.info(f"✅ {symbol}: {len(simple_features)} records with basic features")
                continue
        
        if not all_features:
            raise ValueError("No features could be engineered from the data")
            
        # Combine all features
        combined_features = pd.concat(all_features, ignore_index=True)
        
        # Remove rows with NaN targets
        combined_features = combined_features.dropna(subset=['future_returns'])
        
        self.logger.info(f"🎯 Final dataset: {len(combined_features):,} records with {len(combined_features.columns)} features")
        return combined_features
        
    def _create_simple_features(self, data):
        """Create simple technical features as backup"""
        try:
            df = data.copy()
            
            # Basic price features
            df['price_change'] = df['close'].pct_change()
            df['high_low_spread'] = (df['high'] - df['low']) / df['close']
            df['open_close_spread'] = (df['close'] - df['open']) / df['open']
            
            # Simple moving averages
            for period in [5, 10, 20]:
                df[f'sma_{period}'] = df['close'].rolling(period).mean()
                df[f'price_to_sma_{period}'] = df['close'] / df[f'sma_{period}']
                
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Volume features
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Volatility
            df['volatility'] = df['close'].pct_change().rolling(20).std()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error creating simple features: {e}")
            return pd.DataFrame()
        
    def prepare_training_data(self, features):
        """Prepare data for training"""
        self.logger.info("📋 Preparing training data...")
        
        # Define feature columns (exclude target and metadata)
        exclude_cols = ['future_returns', 'symbol', 'datetime', 'returns']
        feature_cols = [col for col in features.columns if col not in exclude_cols]
        
        X = features[feature_cols].copy()
        y = features['future_returns'].copy()
        
        # Remove infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.mean())
        
        # Remove extreme outliers
        y_std = y.std()
        y_mean = y.mean()
        outlier_mask = np.abs(y - y_mean) < 3 * y_std
        X = X[outlier_mask]
        y = y[outlier_mask]
        
        self.logger.info(f"📊 Training data shape: X={X.shape}, y={y.shape}")
        self.logger.info(f"📈 Target statistics: mean={y.mean():.6f}, std={y.std():.6f}")
        
        return X, y, feature_cols
        
    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """Train XGBoost model"""
        self.logger.info("🚀 Training XGBoost model...")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # XGBoost parameters optimized for trading
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 300,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Train model
        model = xgb.XGBRegressor(**params)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        test_r2 = r2_score(y_test, test_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        
        self.models['xgboost'] = model
        self.scalers['xgboost'] = scaler
        self.performance_metrics['xgboost'] = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'test_mae': test_mae
        }
        
        self.logger.info(f"✅ XGBoost - Test RMSE: {test_rmse:.6f}, R²: {test_r2:.4f}")
        return model
        
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """Train Random Forest model"""
        self.logger.info("🌲 Training Random Forest model...")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Random Forest parameters
        params = {
            'n_estimators': 200,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Train model
        model = RandomForestRegressor(**params)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        test_r2 = r2_score(y_test, test_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        
        self.models['random_forest'] = model
        self.scalers['random_forest'] = scaler
        self.performance_metrics['random_forest'] = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'test_mae': test_mae
        }
        
        self.logger.info(f"✅ Random Forest - Test RMSE: {test_rmse:.6f}, R²: {test_r2:.4f}")
        return model
        
    def train_svm(self, X_train, y_train, X_test, y_test):
        """Train SVM model"""
        self.logger.info("🎯 Training SVM model...")
        
        # Scale features (critical for SVM)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # SVM parameters
        params = {
            'kernel': 'rbf',
            'C': 1.0,
            'gamma': 'scale',
            'epsilon': 0.01
        }
        
        # Train model
        model = SVR(**params)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        test_r2 = r2_score(y_test, test_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        
        self.models['svm'] = model
        self.scalers['svm'] = scaler
        self.performance_metrics['svm'] = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'test_mae': test_mae
        }
        
        self.logger.info(f"✅ SVM - Test RMSE: {test_rmse:.6f}, R²: {test_r2:.4f}")
        return model
        
    def save_models(self):
        """Save all trained models"""
        self.logger.info("💾 Saving trained models...")
        
        model_dir = "/workspace/models/production/ensemble"
        os.makedirs(model_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for model_name in self.models:
            # Save model
            model_file = f"{model_dir}/{model_name}_real_data_{timestamp}.joblib"
            joblib.dump(self.models[model_name], model_file)
            
            # Save scaler
            scaler_file = f"{model_dir}/{model_name}_scaler_{timestamp}.joblib"
            joblib.dump(self.scalers[model_name], scaler_file)
            
            self.logger.info(f"✅ Saved {model_name} model and scaler")
        
        # Save performance metrics
        metrics_file = f"{model_dir}/performance_metrics_{timestamp}.joblib"
        joblib.dump(self.performance_metrics, metrics_file)
        
        self.logger.info(f"📊 Saved performance metrics to {metrics_file}")
        
    def train_all_models(self, data_file):
        """Train all ensemble models"""
        start_time = datetime.now()
        self.logger.info("🚀 Starting ensemble models training with real data...")
        
        # Load and prepare data
        data = self.load_and_prepare_data(data_file)
        features = self.engineer_features(data)
        X, y, feature_cols = self.prepare_training_data(features)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        
        self.logger.info(f"📊 Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
        
        # Train models
        self.train_xgboost(X_train, y_train, X_test, y_test)
        self.train_random_forest(X_train, y_train, X_test, y_test)
        self.train_svm(X_train, y_train, X_test, y_test)
        
        # Save models
        self.save_models()
        
        # Print final results
        training_time = datetime.now() - start_time
        self.logger.info(f"🎉 ENSEMBLE TRAINING COMPLETED in {training_time}")
        
        print("\n" + "="*80)
        print("🏆 ENSEMBLE MODELS PERFORMANCE SUMMARY")
        print("="*80)
        
        for model_name, metrics in self.performance_metrics.items():
            print(f"\n📊 {model_name.upper()}:")
            print(f"   Test RMSE: {metrics['test_rmse']:.6f}")
            print(f"   Test R²:   {metrics['test_r2']:.4f}")
            print(f"   Test MAE:  {metrics['test_mae']:.6f}")
            
        print(f"\n⏱️  Total Training Time: {training_time}")
        print("💾 Models saved to: /workspace/models/production/ensemble/")

if __name__ == "__main__":
    trainer = RealDataEnsembleTrainer()
    
    # Use the real market data we collected
    data_files = [
        "/workspace/data/real_training_data/market_data_7day.csv",
        "/workspace/data/real_training_data/real_market_data_20250816_094716.csv"
    ]
    
    # Find existing data file
    data_file = None
    for file in data_files:
        if os.path.exists(file):
            data_file = file
            break
    
    if data_file:
        trainer.train_all_models(data_file)
    else:
        print("❌ No real market data file found!")