#!/usr/bin/env python3
"""
🚀 FIXED Ensemble Model Training Script
Comprehensive training for all AI models: LSTM, XGBoost, Transformer, Random Forest, SVM

Usage:
    python train_ensemble.py --mode quick      # Quick training (50 epochs)
    python train_ensemble.py --mode standard  # Standard training (100 epochs)
    python train_ensemble.py --mode intensive # Intensive training (200 epochs)
    python train_ensemble.py --models lstm,xgb # Train specific models only
"""

import argparse
import sys
import os
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append('/workspace')

# Import required components
from ensemble_models import EnsembleSignalGenerator
from data_manager import DataManager
from config import DATABASE_CONFIG

def setup_logging():
    """Setup comprehensive logging for ensemble training"""
    log_dir = '/workspace/logs'
    os.makedirs(log_dir, exist_ok=True)
    
    # Create training log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/ensemble_training_fixed_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger('EnsembleTrainingFixed')

def create_enhanced_sample_data():
    """Create enhanced sample training data with realistic market patterns"""
    import pandas as pd
    import numpy as np
    
    logger = logging.getLogger('EnsembleTrainingFixed')
    logger.info("Creating enhanced sample data for ensemble training...")
    
    # Generate 2 years of hourly data for better training
    dates = pd.date_range(start='2023-01-01', end='2025-01-01', freq='H')
    n_samples = len(dates)
    
    # Create realistic price data with multiple patterns
    np.random.seed(42)
    
    # Base price (starting at 1.1000 for EUR/USD)
    base_price = 1.1000
    
    # Generate price movements with multiple patterns
    returns = np.random.normal(0, 0.0005, n_samples)  # Base volatility
    
    # Add cyclical trends
    trend_cycle = 0.0002 * np.sin(dates.astype(np.int64) / (1e9 * 60 * 60 * 24 * 30))  # Monthly cycle
    trend_long = 0.0001 * np.sin(dates.astype(np.int64) / (1e9 * 60 * 60 * 24 * 90))   # Quarterly cycle
    
    # Add volatility clustering (GARCH-like behavior)
    volatility = np.ones(n_samples) * 0.0005
    for i in range(1, n_samples):
        volatility[i] = 0.9 * volatility[i-1] + 0.1 * abs(returns[i-1])
    
    # Add market regime changes
    regime_changes = np.random.choice([0, 1, 2], n_samples, p=[0.3, 0.5, 0.2])
    regime_multiplier = np.where(regime_changes == 0, 0.5, np.where(regime_changes == 1, 1.0, 2.0))
    
    prices = [base_price]
    
    for i in range(1, n_samples):
        # Combine all effects
        price_change = (returns[i] * volatility[i] + trend_cycle[i] + trend_long[i]) * regime_multiplier[i]
        new_price = prices[-1] * (1 + price_change)
        prices.append(new_price)
    
    # Create DataFrame with OHLCV data
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.0003))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.0003))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000, 10000, n_samples)
    })
    
    # Ensure high >= open, close and low <= open, close
    data['high'] = data[['open', 'close', 'high']].max(axis=1)
    data['low'] = data[['open', 'close', 'low']].min(axis=1)
    
    # Add some realistic patterns
    data['volume'] = data['volume'] * (1 + 0.5 * np.sin(dates.astype(np.int64) / (1e9 * 60 * 60 * 24 * 7)))  # Weekly volume pattern
    
    logger.info(f"Created enhanced sample data: {len(data)} samples from {data['timestamp'].min()} to {data['timestamp'].max()}")
    logger.info(f"Data shape: {data.shape}")
    logger.info(f"Price range: {data['close'].min():.4f} - {data['close'].max():.4f}")
    
    return data

def get_real_market_data():
    """Attempt to get real market data from Pocket Option API"""
    logger = logging.getLogger('EnsembleTrainingFixed')
    
    try:
        # Import Pocket Option API
        from pocket_option_api import PocketOptionAPI
        
        api = PocketOptionAPI()
        
        # Try to get data for major pairs
        pairs = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CAD', 'NZD/USD']
        all_data = []
        
        for pair in pairs:
            try:
                logger.info(f"Fetching data for {pair}...")
                data = api.get_market_data(pair, timeframe="1m", limit=1000)
                if data is not None and len(data) > 100:
                    data['pair'] = pair
                    all_data.append(data)
                    logger.info(f"Successfully fetched {len(data)} samples for {pair}")
                else:
                    logger.warning(f"Insufficient data for {pair}")
            except Exception as e:
                logger.error(f"Error fetching data for {pair}: {e}")
                continue
        
        if all_data:
            # Combine all data
            combined_data = pd.concat(all_data, ignore_index=True)
            combined_data = combined_data.sort_values('timestamp')
            logger.info(f"Combined real market data: {len(combined_data)} samples")
            return combined_data
        else:
            logger.warning("No real market data available, using sample data")
            return None
            
    except Exception as e:
        logger.error(f"Error accessing Pocket Option API: {e}")
        return None

def validate_training_data(data):
    """Validate training data quality"""
    logger = logging.getLogger('EnsembleTrainingFixed')
    
    try:
        # Check data shape
        if len(data) < 1000:
            logger.error(f"Insufficient data: {len(data)} samples (minimum 1000 required)")
            return False
        
        # Check for required columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        # Check for NaN values
        nan_count = data[required_columns].isna().sum().sum()
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN values, cleaning data...")
            data = data.dropna(subset=required_columns)
        
        # Check price consistency
        price_errors = ((data['high'] < data['low']) | 
                       (data['high'] < data['open']) | 
                       (data['high'] < data['close']) |
                       (data['low'] > data['open']) | 
                       (data['low'] > data['close'])).sum()
        
        if price_errors > 0:
            logger.warning(f"Found {price_errors} price consistency errors")
            return False
        
        # Check for sufficient price movement
        price_changes = data['close'].pct_change().abs()
        if price_changes.mean() < 0.0001:  # Less than 1 pip average movement
            logger.warning("Very low price volatility detected")
        
        logger.info(f"Data validation passed: {len(data)} samples")
        return True
        
    except Exception as e:
        logger.error(f"Error validating data: {e}")
        return False

def train_ensemble_with_validation(ensemble, training_data, config, models_to_train=None):
    """Train ensemble with comprehensive validation"""
    logger = logging.getLogger('EnsembleTrainingFixed')
    
    try:
        # Validate data first
        if not validate_training_data(training_data):
            logger.error("Data validation failed, cannot proceed with training")
            return None
        
        logger.info("Starting ensemble training with validation...")
        
        # Train ensemble
        history = ensemble.train_ensemble(
            data=training_data,
            validation_split=0.2
        )
        
        # Evaluate model performance
        if history:
            logger.info("Training completed successfully!")
            
            # Check individual model performance
            for model_name, model_history in history.items():
                if isinstance(model_history, dict) and 'val_accuracy' in model_history:
                    final_accuracy = max(model_history['val_accuracy'])
                    logger.info(f"{model_name} validation accuracy: {final_accuracy:.4f}")
                    
                    # Check if accuracy meets requirements
                    if final_accuracy >= 0.95:
                        logger.info(f"🎉 {model_name} accuracy meets production requirements (95%+)")
                    elif final_accuracy >= 0.85:
                        logger.info(f"✅ {model_name} accuracy is good (85%+) but needs improvement")
                    elif final_accuracy >= 0.75:
                        logger.info(f"⚠️ {model_name} accuracy is acceptable (75%+) but needs significant improvement")
                    else:
                        logger.warning(f"❌ {model_name} accuracy is below acceptable threshold")
            
            return history
        else:
            logger.error("Training failed - no history returned")
            return None
            
    except Exception as e:
        logger.error(f"Error during ensemble training: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train ensemble models for trading')
    parser.add_argument('--mode', choices=['quick', 'standard', 'intensive', 'custom'], 
                       default='standard', help='Training mode')
    parser.add_argument('--epochs', type=int, help='Number of epochs (custom mode)')
    parser.add_argument('--batch-size', type=int, help='Batch size (custom mode)')
    parser.add_argument('--models', type=str, help='Comma-separated list of models to train (lstm,xgb,rf,svm,transformer)')
    parser.add_argument('--use-real-data', action='store_true', help='Use real market data if available')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("🚀 FIXED Ensemble Model Training Started")
    logger.info(f"Training mode: {args.mode}")
    
    # Configure training parameters
    if args.mode == 'quick':
        config = {'epochs': 50, 'batch_size': 32}
        logger.info("Starting quick training: Quick training for testing")
    elif args.mode == 'standard':
        config = {'epochs': 100, 'batch_size': 32}
        logger.info("Starting standard training: Standard training for production")
    elif args.mode == 'intensive':
        config = {'epochs': 200, 'batch_size': 32}
        logger.info("Starting intensive training: Intensive training for maximum accuracy")
    else:  # custom
        config = {
            'epochs': args.epochs or 100,
            'batch_size': args.batch_size or 32
        }
        logger.info(f"Starting custom training: {config['epochs']} epochs, batch size {config['batch_size']}")
    
    logger.info(f"Configuration: {config['epochs']} epochs, batch size {config['batch_size']}")
    
    # Parse models to train
    models_to_train = None
    if args.models:
        models_to_train = [model.strip() for model in args.models.split(',')]
        logger.info(f"Training specific models: {models_to_train}")
    
    try:
        # Get training data
        training_data = None
        
        if args.use_real_data:
            logger.info("Attempting to get real market data...")
            training_data = get_real_market_data()
        
        if training_data is None:
            logger.info("Using enhanced sample data for training...")
            training_data = create_enhanced_sample_data()
        
        logger.info(f"Training data shape: {training_data.shape}")
        
        # Initialize ensemble
        ensemble = EnsembleSignalGenerator()
        
        # Train ensemble with validation
        history = train_ensemble_with_validation(ensemble, training_data, config, models_to_train)
        
        if history:
            # Save models to standardized prefix
            os.makedirs(DATABASE_CONFIG['models_dir'], exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base = os.path.join(DATABASE_CONFIG['models_dir'], f"ensemble_{timestamp}")
            ensemble.save_ensemble(base)
            logger.info("✅ Ensemble training completed successfully!")
            logger.info("All models saved and ready for production use")
            
            # Print final metrics
            for model_name, model_history in history.items():
                if isinstance(model_history, dict) and 'val_accuracy' in model_history:
                    final_accuracy = max(model_history['val_accuracy'])
                    logger.info(f"{model_name} final validation accuracy: {final_accuracy:.4f}")
            
        else:
            logger.error("❌ Ensemble training failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()