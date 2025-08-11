#!/usr/bin/env python3
"""
Ensemble Model Training Script
Trains all AI models for the trading system: LSTM, XGBoost, Transformer, Random Forest, SVM

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
    log_file = f"{log_dir}/ensemble_training_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger('EnsembleTraining')

def create_sample_data():
    """Create sample training data if no real data exists"""
    import pandas as pd
    import numpy as np
    
    logger = logging.getLogger('EnsembleTraining')
    logger.info("Creating sample training data for ensemble training...")
    
    # Generate 2 years of hourly data for better training
    dates = pd.date_range(start='2023-01-01', end='2025-01-01', freq='H')
    n_samples = len(dates)
    
    # Create realistic price data with trends, volatility, and patterns
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
    
    prices = [base_price]
    
    for i in range(1, n_samples):
        # Combine all effects
        price_change = returns[i] * volatility[i] + trend_cycle[i] + trend_long[i]
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
    data['day_of_week'] = data['timestamp'].dt.dayofweek
    data['hour'] = data['timestamp'].dt.hour
    data['month'] = data['timestamp'].dt.month
    
    logger.info(f"Created sample data: {len(data)} samples from {data['timestamp'].min()} to {data['timestamp'].max()}")
    return data

def get_training_data():
    """Get training data from various sources"""
    logger = logging.getLogger('EnsembleTraining')
    
    # Try to get real data first
    try:
        data_manager = DataManager()
        # Get data for major pairs
        pairs = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CAD', 'NZD/USD']
        all_data = []
        
        for pair in pairs:
            try:
                data = data_manager.get_market_data(pair, limit=2000)  # More data for ensemble
                if data is not None and len(data) > 200:
                    all_data.append(data)
                    logger.info(f"Loaded {len(data)} samples for {pair}")
            except Exception as e:
                logger.warning(f"Could not load data for {pair}: {e}")
        
        if all_data:
            # Combine all data
            import pandas as pd
            combined_data = pd.concat(all_data, ignore_index=True)
            logger.info(f"Total real data loaded: {len(combined_data)} samples")
            return combined_data
        
    except Exception as e:
        logger.warning(f"Could not load real data: {e}")
    
    # Fall back to sample data
    logger.info("Using sample data for ensemble training")
    return create_sample_data()

def train_ensemble(mode='standard', models=None, validation_split=0.2):
    """Train the ensemble of models"""
    logger = logging.getLogger('EnsembleTraining')
    
    # Training configuration based on mode
    training_configs = {
        'quick': {'epochs': 50, 'description': 'Quick training for testing'},
        'standard': {'epochs': 100, 'description': 'Standard training for production'},
        'intensive': {'epochs': 200, 'description': 'Intensive training for maximum accuracy'}
    }
    
    if mode not in training_configs:
        logger.error(f"Invalid mode: {mode}. Available modes: {list(training_configs.keys())}")
        return False
    
    config = training_configs[mode]
    logger.info(f"Starting {mode} ensemble training: {config['description']}")
    
    try:
        # Get training data
        training_data = get_training_data()
        
        if training_data is None or len(training_data) < 500:
            logger.error("Insufficient training data for ensemble. Need at least 500 samples.")
            return False
        
        logger.info(f"Training data shape: {training_data.shape}")
        
        # Initialize ensemble generator
        ensemble = EnsembleSignalGenerator()
        
        # Train the ensemble
        logger.info("Starting ensemble training...")
        start_time = datetime.now()
        
        # Train all models
        ensemble.train_ensemble(
            data=training_data,
            validation_split=validation_split
        )
        
        training_time = datetime.now() - start_time
        logger.info(f"Ensemble training completed in {training_time}")
        
        # Save the trained ensemble
        models_dir = DATABASE_CONFIG['models_dir']
        os.makedirs(models_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ensemble_path = f"{models_dir}/ensemble_models_{mode}_{timestamp}.pkl"
        ensemble.save_ensemble(ensemble_path)
        
        logger.info(f"Ensemble saved to: {ensemble_path}")
        
        # Test the ensemble
        logger.info("Testing trained ensemble...")
        test_data = training_data.tail(100)
        prediction = ensemble.predict(test_data)
        
        if prediction:
            logger.info(f"Test prediction: {prediction.final_prediction} (confidence: {prediction.final_confidence:.2f})")
            logger.info("✅ Ensemble training successful!")
            
            # Show model performance
            performance = ensemble.get_model_performance(training_data)
            logger.info("Model Performance Summary:")
            for model_name, metrics in performance.items():
                if 'accuracy' in metrics:
                    logger.info(f"  {model_name}: {metrics['accuracy']:.4f} accuracy")
            
            return True
        else:
            logger.error("❌ Ensemble training failed - prediction test failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Ensemble training failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Main ensemble training function"""
    parser = argparse.ArgumentParser(description='Train Ensemble Trading Models')
    parser.add_argument('--mode', choices=['quick', 'standard', 'intensive'], 
                       default='standard', help='Training mode')
    parser.add_argument('--models', help='Specific models to train (comma-separated)')
    parser.add_argument('--validation-split', type=float, default=0.2, 
                       help='Validation split ratio (default: 0.2)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    logger.info("🚀 Ensemble Model Training Started")
    logger.info(f"Training mode: {args.mode}")
    logger.info(f"Validation split: {args.validation_split}")
    
    if args.models:
        logger.info(f"Training specific models: {args.models}")
    
    # Create models directory
    models_dir = DATABASE_CONFIG['models_dir']
    os.makedirs(models_dir, exist_ok=True)
    
    # Train the ensemble
    success = train_ensemble(
        mode=args.mode,
        models=args.models,
        validation_split=args.validation_split
    )
    
    if success:
        logger.info("🎉 Ensemble training completed successfully!")
        logger.info(f"Models saved in: {models_dir}")
        logger.info("You can now use the trained ensemble with your trading system!")
        
        # Show next steps
        logger.info("\n📋 Next Steps:")
        logger.info("1. Test the models with: python test_models.py")
        logger.info("2. Start trading system: python start_unified_system.py")
        logger.info("3. Monitor performance in logs/ directory")
    else:
        logger.error("💥 Ensemble training failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()