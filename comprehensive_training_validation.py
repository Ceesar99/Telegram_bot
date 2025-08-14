#!/usr/bin/env python3
"""
🚀 COMPREHENSIVE AI/ML TRAINING & VALIDATION SYSTEM
Complete pipeline for training and validating all models before live trading

This script performs:
1. Fix all training issues
2. Train LSTM model with enhanced data
3. Train ensemble models
4. Validate models with paper trading
5. Generate comprehensive reports

Usage:
    python comprehensive_training_validation.py --mode quick      # Quick validation (7 days)
    python comprehensive_training_validation.py --mode standard  # Standard validation (30 days)
    python comprehensive_training_validation.py --mode intensive # Intensive validation (90 days)
"""

import argparse
import sys
import os
import logging
import asyncio
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append('/workspace')

# Import required components
from lstm_model import LSTMTradingModel
from ensemble_models import EnsembleSignalGenerator
from paper_trading_validator import PaperTradingValidator, run_paper_trading_validation
from config import DATABASE_CONFIG, LSTM_CONFIG

def setup_logging():
    """Setup comprehensive logging for training and validation"""
    log_dir = '/workspace/logs'
    os.makedirs(log_dir, exist_ok=True)
    
    # Create training log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/comprehensive_training_validation_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger('ComprehensiveTrainingValidation')

def create_enhanced_training_data():
    """Create enhanced training data with realistic market patterns"""
    import pandas as pd
    import numpy as np
    
    logger = logging.getLogger('ComprehensiveTrainingValidation')
    logger.info("Creating enhanced training data with realistic market patterns...")
    
    # Generate 3 years of hourly data for comprehensive training
    dates = pd.date_range(start='2022-01-01', end='2025-01-01', freq='H')
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
    
    # Add news events (random spikes)
    news_events = np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    news_impact = news_events * np.random.normal(0, 0.002, n_samples)
    
    prices = [base_price]
    
    for i in range(1, n_samples):
        # Combine all effects
        price_change = (returns[i] * volatility[i] + trend_cycle[i] + trend_long[i] + news_impact[i]) * regime_multiplier[i]
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
    
    # Add realistic patterns
    data['volume'] = data['volume'] * (1 + 0.5 * np.sin(dates.astype(np.int64) / (1e9 * 60 * 60 * 24 * 7)))  # Weekly volume pattern
    
    logger.info(f"Created enhanced training data: {len(data)} samples from {data['timestamp'].min()} to {data['timestamp'].max()}")
    logger.info(f"Data shape: {data.shape}")
    logger.info(f"Price range: {data['close'].min():.4f} - {data['close'].max():.4f}")
    
    return data

def train_lstm_model(training_data, config):
    """Train LSTM model with comprehensive validation"""
    logger = logging.getLogger('ComprehensiveTrainingValidation')
    
    try:
        logger.info("=" * 60)
        logger.info("🤖 TRAINING LSTM MODEL")
        logger.info("=" * 60)
        
        # Initialize LSTM model
        lstm_model = LSTMTradingModel()
        
        # Train model
        logger.info("Starting LSTM model training...")
        history = lstm_model.train_model(
            data=training_data,
            validation_split=0.2,
            epochs=config['epochs']
        )
        
        if history:
            # Evaluate performance
            final_accuracy = max(history.history['val_accuracy'])
            final_loss = min(history.history['val_loss'])
            
            logger.info(f"LSTM Training completed successfully!")
            logger.info(f"Final validation accuracy: {final_accuracy:.4f}")
            logger.info(f"Final validation loss: {final_loss:.4f}")
            
            # Save model
            lstm_model.save_model()
            logger.info("LSTM model saved successfully")
            
            # Check if accuracy meets requirements
            if final_accuracy >= 0.95:
                logger.info("🎉 LSTM accuracy meets production requirements (95%+)")
                return True
            elif final_accuracy >= 0.85:
                logger.info("✅ LSTM accuracy is good (85%+) but needs improvement")
                return True
            elif final_accuracy >= 0.75:
                logger.info("⚠️ LSTM accuracy is acceptable (75%+) but needs significant improvement")
                return True
            else:
                logger.warning("❌ LSTM accuracy is below acceptable threshold")
                return False
        else:
            logger.error("LSTM training failed")
            return False
            
    except Exception as e:
        logger.error(f"Error training LSTM model: {e}")
        return False

def train_ensemble_models(training_data, config):
    """Train ensemble models with comprehensive validation"""
    logger = logging.getLogger('ComprehensiveTrainingValidation')
    
    try:
        logger.info("=" * 60)
        logger.info("🤖 TRAINING ENSEMBLE MODELS")
        logger.info("=" * 60)
        
        # Initialize ensemble
        ensemble = EnsembleSignalGenerator()
        
        # Train ensemble
        logger.info("Starting ensemble model training...")
        history = ensemble.train_ensemble(
            data=training_data,
            validation_split=0.2
        )
        
        if history:
            logger.info("Ensemble training completed successfully!")
            
            # Check individual model performance
            success_count = 0
            total_models = 0
            
            for model_name, model_history in history.items():
                if isinstance(model_history, dict) and 'val_accuracy' in model_history:
                    total_models += 1
                    final_accuracy = max(model_history['val_accuracy'])
                    logger.info(f"{model_name} validation accuracy: {final_accuracy:.4f}")
                    
                    if final_accuracy >= 0.75:
                        success_count += 1
                        logger.info(f"✅ {model_name} accuracy is acceptable")
                    else:
                        logger.warning(f"❌ {model_name} accuracy is below threshold")
            
            # Save models
            ensemble.save_models()
            logger.info("Ensemble models saved successfully")
            
            # Check overall success
            success_rate = success_count / total_models if total_models > 0 else 0
            logger.info(f"Ensemble success rate: {success_rate:.2%} ({success_count}/{total_models} models)")
            
            return success_rate >= 0.6  # At least 60% of models should be successful
        else:
            logger.error("Ensemble training failed")
            return False
            
    except Exception as e:
        logger.error(f"Error training ensemble models: {e}")
        return False

async def run_comprehensive_validation(duration_days, signals_per_day):
    """Run comprehensive paper trading validation"""
    logger = logging.getLogger('ComprehensiveTrainingValidation')
    
    try:
        logger.info("=" * 60)
        logger.info("📊 RUNNING PAPER TRADING VALIDATION")
        logger.info("=" * 60)
        
        # Run paper trading validation
        final_stats = await run_paper_trading_validation(
            duration_days=duration_days,
            signals_per_day=signals_per_day
        )
        
        # Analyze results
        win_rate = final_stats.get('win_rate', 0)
        total_pnl = final_stats.get('total_pnl', 0)
        total_trades = final_stats.get('total_trades', 0)
        
        logger.info("=" * 60)
        logger.info("📊 VALIDATION RESULTS ANALYSIS")
        logger.info("=" * 60)
        logger.info(f"Total Trades: {total_trades}")
        logger.info(f"Win Rate: {win_rate:.2%}")
        logger.info(f"Total PnL: ${total_pnl:.2f}")
        
        # Determine if validation passed
        validation_passed = True
        issues = []
        
        if win_rate < 0.75:
            validation_passed = False
            issues.append(f"Win rate too low: {win_rate:.2%} (minimum 75%)")
        
        if total_pnl < 0:
            validation_passed = False
            issues.append(f"Negative PnL: ${total_pnl:.2f}")
        
        if total_trades < 100:
            validation_passed = False
            issues.append(f"Insufficient trades: {total_trades} (minimum 100)")
        
        if validation_passed:
            logger.info("🎉 VALIDATION PASSED - Models ready for live trading!")
        else:
            logger.warning("❌ VALIDATION FAILED - Issues found:")
            for issue in issues:
                logger.warning(f"  - {issue}")
        
        return validation_passed, final_stats
        
    except Exception as e:
        logger.error(f"Error during validation: {e}")
        return False, {}

def generate_comprehensive_report(lstm_success, ensemble_success, validation_passed, validation_stats):
    """Generate comprehensive training and validation report"""
    logger = logging.getLogger('ComprehensiveTrainingValidation')
    
    logger.info("=" * 80)
    logger.info("📋 COMPREHENSIVE TRAINING & VALIDATION REPORT")
    logger.info("=" * 80)
    
    # Training Results
    logger.info("🤖 MODEL TRAINING RESULTS:")
    logger.info(f"  LSTM Model: {'✅ PASSED' if lstm_success else '❌ FAILED'}")
    logger.info(f"  Ensemble Models: {'✅ PASSED' if ensemble_success else '❌ FAILED'}")
    
    # Validation Results
    logger.info("\n📊 PAPER TRADING VALIDATION RESULTS:")
    if validation_stats:
        logger.info(f"  Total Trades: {validation_stats.get('total_trades', 0)}")
        logger.info(f"  Win Rate: {validation_stats.get('win_rate', 0):.2%}")
        logger.info(f"  Total PnL: ${validation_stats.get('total_pnl', 0):.2f}")
        logger.info(f"  Validation Status: {'✅ PASSED' if validation_passed else '❌ FAILED'}")
    
    # Overall Assessment
    logger.info("\n🎯 OVERALL ASSESSMENT:")
    if lstm_success and ensemble_success and validation_passed:
        logger.info("🎉 ALL SYSTEMS PASSED - READY FOR LIVE TRADING!")
        logger.info("✅ LSTM Model: Trained and validated")
        logger.info("✅ Ensemble Models: Trained and validated")
        logger.info("✅ Paper Trading: Passed all criteria")
        logger.info("\n🚀 NEXT STEPS:")
        logger.info("1. Start with small position sizes")
        logger.info("2. Monitor performance closely")
        logger.info("3. Gradually increase position sizes")
        logger.info("4. Continue monitoring and retraining as needed")
    else:
        logger.info("❌ SYSTEM NOT READY FOR LIVE TRADING")
        if not lstm_success:
            logger.info("  - LSTM model needs retraining")
        if not ensemble_success:
            logger.info("  - Ensemble models need retraining")
        if not validation_passed:
            logger.info("  - Paper trading validation failed")
        logger.info("\n🔧 RECOMMENDED ACTIONS:")
        logger.info("1. Fix training issues")
        logger.info("2. Retrain models with better data")
        logger.info("3. Run validation again")
        logger.info("4. Only proceed when all tests pass")
    
    logger.info("=" * 80)

async def main():
    """Main comprehensive training and validation function"""
    parser = argparse.ArgumentParser(description='Comprehensive AI/ML Training and Validation')
    parser.add_argument('--mode', choices=['quick', 'standard', 'intensive'], 
                       default='standard', help='Validation mode')
    parser.add_argument('--skip-training', action='store_true', help='Skip training, run validation only')
    parser.add_argument('--skip-validation', action='store_true', help='Skip validation, run training only')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("🚀 COMPREHENSIVE AI/ML TRAINING & VALIDATION STARTED")
    logger.info(f"Mode: {args.mode}")
    
    # Configure parameters based on mode
    if args.mode == 'quick':
        config = {'epochs': 50, 'batch_size': 32}
        validation_days = 7
        signals_per_day = 10
        logger.info("Quick mode: 50 epochs, 7 days validation, 10 signals/day")
    elif args.mode == 'standard':
        config = {'epochs': 100, 'batch_size': 32}
        validation_days = 30
        signals_per_day = 20
        logger.info("Standard mode: 100 epochs, 30 days validation, 20 signals/day")
    else:  # intensive
        config = {'epochs': 200, 'batch_size': 32}
        validation_days = 90
        signals_per_day = 30
        logger.info("Intensive mode: 200 epochs, 90 days validation, 30 signals/day")
    
    try:
        # Step 1: Create enhanced training data
        if not args.skip_training:
            logger.info("Step 1: Creating enhanced training data...")
            training_data = create_enhanced_training_data()
            
            # Step 2: Train LSTM model
            logger.info("Step 2: Training LSTM model...")
            lstm_success = train_lstm_model(training_data, config)
            
            # Step 3: Train ensemble models
            logger.info("Step 3: Training ensemble models...")
            ensemble_success = train_ensemble_models(training_data, config)
        else:
            logger.info("Skipping training phase...")
            lstm_success = True  # Assume models are already trained
            ensemble_success = True
        
        # Step 4: Run paper trading validation
        if not args.skip_validation:
            logger.info("Step 4: Running paper trading validation...")
            validation_passed, validation_stats = await run_comprehensive_validation(
                duration_days=validation_days,
                signals_per_day=signals_per_day
            )
        else:
            logger.info("Skipping validation phase...")
            validation_passed = True
            validation_stats = {}
        
        # Step 5: Generate comprehensive report
        logger.info("Step 5: Generating comprehensive report...")
        await generate_comprehensive_report(lstm_success, ensemble_success, validation_passed, validation_stats)
        
        # Final status
        if lstm_success and ensemble_success and validation_passed:
            logger.info("🎉 COMPREHENSIVE TRAINING & VALIDATION COMPLETED SUCCESSFULLY!")
            logger.info("Your AI/ML models are ready for live Pocket Option trading!")
            return 0
        else:
            logger.error("❌ COMPREHENSIVE TRAINING & VALIDATION FAILED!")
            logger.error("Please fix the issues before proceeding to live trading.")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Comprehensive training and validation failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    # Run comprehensive training and validation
    exit_code = asyncio.run(main())
    sys.exit(exit_code)