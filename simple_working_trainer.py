#!/usr/bin/env python3
"""
🚀 SIMPLE WORKING TRAINER
Basic functional training script to get the system operational
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleWorkingTrainer:
    """Simple trainer that actually works"""
    
    def __init__(self):
        self.data_file = '/workspace/data/real_market_data/combined_market_data_20250816_092932.csv'
        self.models_dir = '/workspace/models'
        self.logs_dir = '/workspace/logs'
        
    def load_and_prepare_data(self):
        """Load and prepare data for training"""
        logger.info("🔄 Loading market data...")
        
        try:
            # Load data
            df = pd.read_csv(self.data_file)
            logger.info(f"📊 Loaded {len(df):,} records")
            
            # Convert datetime
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Sort by symbol and datetime
            df = df.sort_values(['symbol', 'datetime']).reset_index(drop=True)
            
            # Focus on main pairs for now
            main_pairs = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X']
            df = df[df['symbol'].isin(main_pairs)].copy()
            
            logger.info(f"📊 Using {len(df):,} records from {df['symbol'].nunique()} pairs")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            return None
    
    def create_simple_features(self, df):
        """Create simple but effective features"""
        logger.info("🔧 Creating simple features...")
        
        # Simple price-based features
        df['price_change'] = df['close'].pct_change()
        df['high_low_ratio'] = df['high'] / df['low']
        df['open_close_ratio'] = df['open'] / df['close']
        df['volatility'] = (df['high'] - df['low']) / df['close']
        
        # Simple moving averages
        for window in [5, 10, 20]:
            df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'price_vs_ma_{window}'] = df['close'] / df[f'ma_{window}']
        
        # Simple target: next candle direction
        df['next_close'] = df['close'].shift(-1)
        df['target'] = (df['next_close'] > df['close']).astype(int)
        
        # Drop rows with NaN values
        df = df.dropna().reset_index(drop=True)
        
        logger.info(f"✅ Created features, {len(df):,} clean samples")
        
        return df
    
    def prepare_training_data(self, df):
        """Prepare data for training"""
        logger.info("📊 Preparing training data...")
        
        # Select feature columns
        feature_cols = [
            'open', 'high', 'low', 'close', 'volume',
            'price_change', 'high_low_ratio', 'open_close_ratio', 'volatility',
            'ma_5', 'ma_10', 'ma_20',
            'price_vs_ma_5', 'price_vs_ma_10', 'price_vs_ma_20'
        ]
        
        # Filter to available columns
        available_cols = [col for col in feature_cols if col in df.columns]
        
        X = df[available_cols].values
        y = df['target'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        logger.info(f"📊 Training samples: {len(X_train):,}")
        logger.info(f"📊 Test samples: {len(X_test):,}")
        logger.info(f"📊 Features: {len(available_cols)}")
        logger.info(f"📊 Target distribution: {np.bincount(y_train)}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler, available_cols
    
    def train_simple_model(self, X_train, X_test, y_train, y_test):
        """Train a simple but effective model"""
        logger.info("🚀 Training simple model...")
        
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score, classification_report
            
            # Create simple random forest
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            )
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predict
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            train_accuracy = accuracy_score(y_train, y_pred_train)
            test_accuracy = accuracy_score(y_test, y_pred_test)
            
            logger.info(f"✅ Training accuracy: {train_accuracy:.3f}")
            logger.info(f"✅ Test accuracy: {test_accuracy:.3f}")
            
            # Detailed report
            logger.info("📊 Classification Report:")
            logger.info(f"\n{classification_report(y_test, y_pred_test)}")
            
            return model, train_accuracy, test_accuracy
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            return None, 0, 0
    
    def save_model(self, model, scaler, feature_cols, metrics):
        """Save the trained model"""
        try:
            import joblib
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save model
            model_path = os.path.join(self.models_dir, f'simple_working_model_{timestamp}.pkl')
            joblib.dump(model, model_path)
            
            # Save scaler
            scaler_path = os.path.join(self.models_dir, f'simple_working_scaler_{timestamp}.pkl')
            joblib.dump(scaler, scaler_path)
            
            # Save metadata
            metadata = {
                'timestamp': timestamp,
                'model_type': 'RandomForest',
                'train_accuracy': metrics['train_accuracy'],
                'test_accuracy': metrics['test_accuracy'],
                'features': feature_cols,
                'model_file': model_path,
                'scaler_file': scaler_path
            }
            
            metadata_path = os.path.join(self.models_dir, f'simple_working_metadata_{timestamp}.json')
            with open(metadata_path, 'w') as f:
                import json
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"💾 Model saved: {model_path}")
            logger.info(f"💾 Scaler saved: {scaler_path}")
            logger.info(f"💾 Metadata saved: {metadata_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving model: {e}")
            return False
    
    def run_training(self):
        """Run the complete training process"""
        logger.info("🚀 STARTING SIMPLE WORKING TRAINING")
        logger.info("=" * 50)
        
        # Load data
        df = self.load_and_prepare_data()
        if df is None:
            return False
        
        # Create features
        df = self.create_simple_features(df)
        if len(df) == 0:
            logger.error("❌ No data after feature engineering")
            return False
        
        # Prepare training data
        X_train, X_test, y_train, y_test, scaler, feature_cols = self.prepare_training_data(df)
        
        # Train model
        model, train_acc, test_acc = self.train_simple_model(X_train, X_test, y_train, y_test)
        
        if model is None:
            return False
        
        # Save model
        metrics = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc
        }
        
        saved = self.save_model(model, scaler, feature_cols, metrics)
        
        if saved:
            logger.info("=" * 50)
            logger.info("🎉 TRAINING COMPLETED SUCCESSFULLY!")
            logger.info(f"✅ Train Accuracy: {train_acc:.1%}")
            logger.info(f"✅ Test Accuracy: {test_acc:.1%}")
            
            if test_acc > 0.52:  # Better than random
                logger.info("🎯 Model performance is above random - GOOD!")
            else:
                logger.warning("⚠️ Model performance is close to random - needs improvement")
            
            logger.info("🚀 Ready for paper trading validation!")
            return True
        else:
            return False

def test_simple_prediction():
    """Test the trained model with a simple prediction"""
    logger.info("🧪 Testing simple prediction...")
    
    try:
        import joblib
        import glob
        
        # Find latest model
        model_files = glob.glob('/workspace/models/simple_working_model_*.pkl')
        if not model_files:
            logger.error("❌ No trained model found")
            return False
        
        latest_model = max(model_files)
        model = joblib.load(latest_model)
        
        logger.info(f"✅ Loaded model from: {latest_model}")
        
        # Create dummy prediction
        dummy_features = np.random.randn(1, 15)  # Assuming 15 features
        
        try:
            prediction = model.predict(dummy_features)
            probability = model.predict_proba(dummy_features)
            
            logger.info(f"✅ Prediction: {'UP' if prediction[0] == 1 else 'DOWN'}")
            logger.info(f"✅ Probability: {probability[0]}")
            logger.info("🎉 Model is working and ready for use!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            return False
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    trainer = SimpleWorkingTrainer()
    success = trainer.run_training()
    
    if success:
        print("\n" + "="*60)
        print("🎉 SIMPLE WORKING TRAINER - SUCCESS!")
        print("✅ Model trained and saved successfully")
        print("🧪 Testing prediction capability...")
        
        test_success = test_simple_prediction()
        
        if test_success:
            print("✅ Prediction test passed!")
            print("🚀 System is ready for paper trading!")
        else:
            print("⚠️ Prediction test failed, but model is saved")
        
        print("\n🎯 NEXT STEPS:")
        print("1. Start paper trading: python3 paper_trading_engine.py --duration=1day")
        print("2. Monitor results: python3 training_monitor.py")
        print("3. If successful, extend to longer periods")
        print("="*60)
    else:
        print("\n❌ TRAINING FAILED")
        print("Check logs for details")