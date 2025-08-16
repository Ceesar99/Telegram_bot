#!/usr/bin/env python3
"""
🚀 QUICK PRODUCTION TRAINING SCRIPT
Trains all models with production settings
"""

import os
import sys
import logging
import asyncio
from datetime import datetime

# Add workspace to path
sys.path.append('/workspace')

def setup_logging():
    """Setup training logging"""
    log_file = f"/workspace/logs/training/production_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('ProductionTraining')

def train_lstm_production():
    """Train LSTM model for production"""
    logger = logging.getLogger('ProductionTraining')
    logger.info("🧠 Training LSTM model for production...")
    
    try:
        # Import and train LSTM
        from lstm_model import LSTMTradingModel
        
        model = LSTMTradingModel()
        
        # Create enhanced training data (until real data is available)
        import pandas as pd
        import numpy as np
        
        # Generate more comprehensive training data
        dates = pd.date_range(start='2021-01-01', end='2025-01-01', freq='H')
        n_samples = len(dates)
        
        np.random.seed(42)
        base_price = 1.1000
        
        # More realistic price generation
        returns = np.random.normal(0, 0.0008, n_samples)
        prices = [base_price]
        
        for i in range(1, n_samples):
            new_price = prices[-1] * (1 + returns[i])
            prices.append(new_price)
        
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.0005))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.0005))) for p in prices],
            'close': prices,
            'volume': np.random.randint(1000, 50000, n_samples)
        })
        
        # Ensure realistic OHLC relationships
        data['high'] = data[['open', 'close', 'high']].max(axis=1)
        data['low'] = data[['open', 'close', 'low']].min(axis=1)
        
        logger.info(f"Created training data: {len(data)} samples")
        
        # Train with production settings
        history = model.train_model(
            data=data,
            validation_split=0.2,
            epochs=100  # Reduced for quick setup, increase for production
        )
        
        if history:
            logger.info("✅ LSTM training completed successfully")
            
            # Save with production naming
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model.save_model(f"/workspace/models/production/lstm_production_{timestamp}.h5")
            
            return True
        else:
            logger.error("❌ LSTM training failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error training LSTM: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Main training function"""
    logger = setup_logging()
    logger.info("🚀 STARTING PRODUCTION MODEL TRAINING")
    logger.info("=" * 50)
    
    training_results = {
        'lstm': train_lstm_production()
    }
    
    logger.info("=" * 50)
    logger.info("📊 TRAINING SUMMARY")
    
    for model_type, result in training_results.items():
        status = "✅ SUCCESS" if result else "❌ FAILED"
        logger.info(f"{model_type.upper()}: {status}")
    
    return all(training_results.values())

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
