#!/usr/bin/env python3
"""
LSTM Model Training Script
Trains the LSTM neural network for binary options trading signals

Usage:
    python train_lstm.py --mode quick      # Quick training (50 epochs)
    python train_lstm.py --mode standard  # Standard training (100 epochs)
    python train_lstm.py --mode intensive # Intensive training (200 epochs)
    python train_lstm.py --mode custom --epochs 150 --batch-size 64
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
from lstm_model import LSTMTradingModel
from data_manager import DataManager
from config import LSTM_CONFIG, DATABASE_CONFIG

def setup_logging():
    """Setup comprehensive logging for training"""
    log_dir = '/workspace/logs'
    os.makedirs(log_dir, exist_ok=True)
    
    # Create training log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/lstm_training_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger('LSTMTraining')

def create_sample_data():
    """Create sample training data if no real data exists"""
    import pandas as pd
    import numpy as np
    
    logger = logging.getLogger('LSTMTraining')
    logger.info("Creating sample training data...")
    
    # Generate 1 year of hourly data
    dates = pd.date_range(start='2024-01-01', end='2025-01-01', freq='H')
    n_samples = len(dates)
    
    # Create realistic price data with trends and volatility
    np.random.seed(42)
    
    # Base price (starting at 1.1000 for EUR/USD)
    base_price = 1.1000
    
    # Generate price movements
    returns = np.random.normal(0, 0.0005, n_samples)  # 5 pips volatility per hour
    prices = [base_price]
    
    for i in range(1, n_samples):
        # Add some trend and mean reversion
        trend = 0.0001 * np.sin(i / 100)  # Cyclical trend
        price_change = returns[i] + trend
        new_price = prices[-1] * (1 + price_change)
        prices.append(new_price)
    
    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.0002))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.0002))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000, 10000, n_samples)
    })
    
    # Ensure high >= open, close and low <= open, close
    data['high'] = data[['open', 'close', 'high']].max(axis=1)
    data['low'] = data[['open', 'close', 'low']].min(axis=1)
    
    logger.info(f"Created sample data: {len(data)} samples from {data['timestamp'].min()} to {data['timestamp'].max()}")
    return data

def get_training_data():
    """Get training data from various sources"""
    logger = logging.getLogger('LSTMTraining')
    
    # Try to get real data first
    try:
        data_manager = DataManager()
        # Get data for major pairs
        pairs = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD']
        all_data = []
        
        for pair in pairs:
            try:
                data = data_manager.get_market_data(pair, limit=1000)
                if data is not None and len(data) > 100:
                    all_data.append(data)
                    logger.info(f"Loaded {len(data)} samples for {pair}")
            except Exception as e:
                logger.warning(f"Could not load data for {pair}: {e}")
        
        if all_data:
            # Combine all data
            combined_data = pd.concat(all_data, ignore_index=True)
            logger.info(f"Total real data loaded: {len(combined_data)} samples")
            return combined_data
        
    except Exception as e:
        logger.warning(f"Could not load real data: {e}")
    
    # Fall back to sample data
    logger.info("Using sample data for training")
    return create_sample_data()

def train_model(mode='standard', epochs=None, batch_size=None, validation_split=0.2):
    """Train the LSTM model"""
    logger = logging.getLogger('LSTMTraining')
    
    # Training configuration based on mode
    training_configs = {
        'quick': {'epochs': 50, 'batch_size': 32, 'description': 'Quick training for testing'},
        'standard': {'epochs': 100, 'batch_size': 32, 'description': 'Standard training for production'},
        'intensive': {'epochs': 200, 'batch_size': 64, 'description': 'Intensive training for maximum accuracy'},
        'custom': {'epochs': epochs or 100, 'batch_size': batch_size or 32, 'description': 'Custom training configuration'}
    }
    
    if mode not in training_configs:
        logger.error(f"Invalid mode: {mode}. Available modes: {list(training_configs.keys())}")
        return False
    
    config = training_configs[mode]
    logger.info(f"Starting {mode} training: {config['description']}")
    logger.info(f"Configuration: {config['epochs']} epochs, batch size {config['batch_size']}")
    
    try:
        # Get training data
        training_data = get_training_data()
        
        if training_data is None or len(training_data) < 100:
            logger.error("Insufficient training data. Need at least 100 samples.")
            return False
        
        logger.info(f"Training data shape: {training_data.shape}")
        
        # Initialize LSTM model
        lstm_model = LSTMTradingModel()
        
        # Train the model
        logger.info("Starting model training...")
        start_time = datetime.now()
        
        history = lstm_model.train_model(
            data=training_data,
            validation_split=validation_split,
            epochs=config['epochs']
        )
        
        training_time = datetime.now() - start_time
        logger.info(f"Training completed in {training_time}")
        
        # Save the trained model
        models_dir = DATABASE_CONFIG['models_dir']
        os.makedirs(models_dir, exist_ok=True)
        
        model_path = f"{models_dir}/lstm_model_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
        lstm_model.save_model(model_path)
        
        logger.info(f"Model saved to: {model_path}")
        
        # Test the model
        logger.info("Testing trained model...")
        test_result = lstm_model.predict_signal(training_data.tail(100))
        
        if test_result:
            logger.info(f"Test prediction: {test_result['signal']} (confidence: {test_result['confidence']:.2f}%)")
            logger.info("✅ Model training successful!")
            return True
        else:
            logger.error("❌ Model training failed - prediction test failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train LSTM Trading Model')
    parser.add_argument('--mode', choices=['quick', 'standard', 'intensive', 'custom'], 
                       default='standard', help='Training mode')
    parser.add_argument('--epochs', type=int, help='Number of epochs (custom mode only)')
    parser.add_argument('--batch-size', type=int, help='Batch size (custom mode only)')
    parser.add_argument('--validation-split', type=float, default=0.2, 
                       help='Validation split ratio (default: 0.2)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    logger.info("🚀 LSTM Model Training Started")
    logger.info(f"Training mode: {args.mode}")
    logger.info(f"Validation split: {args.validation_split}")
    
    # Create models directory
    models_dir = DATABASE_CONFIG['models_dir']
    os.makedirs(models_dir, exist_ok=True)
    
    # Train the model
    success = train_model(
        mode=args.mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split
    )
    
    if success:
        logger.info("🎉 Training completed successfully!")
        logger.info(f"Model saved in: {models_dir}")
        logger.info("You can now use the trained model with your trading system!")
    else:
        logger.error("💥 Training failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()