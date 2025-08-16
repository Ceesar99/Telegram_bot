#!/usr/bin/env python3
"""
🤖 Production Model Training Script
Comprehensive training of all AI/ML models with real market data
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/workspace/logs/model_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_and_prepare_data():
    """Load and prepare market data for training"""
    logger.info("📊 Loading market data...")
    
    data = pd.read_csv('/workspace/data/real_market_data/combined_market_data_20250816_092932.csv')
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.sort_values('datetime').reset_index(drop=True)
    
    logger.info(f"✅ Loaded {len(data)} records from {data['symbol'].nunique()} symbols")
    
    # Use EURUSD for primary training (most liquid pair)
    eurusd_data = data[data['symbol'] == 'EURUSD_X'].copy()
    
    if len(eurusd_data) > 1000:
        training_data = eurusd_data
        logger.info(f"🎯 Using EURUSD data: {len(training_data)} records")
    else:
        training_data = data
        logger.info(f"🎯 Using all currency data: {len(training_data)} records")
    
    return training_data

def train_lstm_model(data):
    """Train LSTM model with optimized parameters"""
    logger.info("🧠 Training LSTM Model...")
    
    try:
        from lstm_model import LSTMTradingModel
        
        model = LSTMTradingModel()
        
        # Use smaller dataset for faster training
        sample_size = min(10000, len(data))
        training_sample = data.tail(sample_size).copy()
        
        logger.info(f"Training on {len(training_sample)} records")
        
        # Train with reduced epochs for faster training
        history = model.train_model(training_sample, validation_split=0.2, epochs=20)
        
        if history is not None:
            # Save model
            model.save_model('/workspace/models/production_lstm_trained.h5')
            
            # Get performance metrics
            validation_data = training_sample.tail(int(len(training_sample) * 0.2))
            performance = model.get_model_performance(validation_data)
            
            results = {
                'model': 'LSTM',
                'status': 'SUCCESS',
                'accuracy': performance.get('accuracy', 0),
                'training_samples': len(training_sample),
                'validation_samples': len(validation_data),
                'epochs_completed': 20,
                'model_file': '/workspace/models/production_lstm_trained.h5'
            }
            
            logger.info(f"✅ LSTM Training Complete - Accuracy: {results['accuracy']:.2f}%")
            return results
            
        else:
            raise Exception("Training returned None")
            
    except Exception as e:
        logger.error(f"❌ LSTM Training Failed: {e}")
        return {
            'model': 'LSTM',
            'status': 'FAILED',
            'error': str(e)
        }

def train_ensemble_model(data):
    """Train ensemble model system"""
    logger.info("🎭 Training Ensemble Models...")
    
    try:
        from ensemble_models import EnsembleSignalGenerator
        
        ensemble = EnsembleSignalGenerator()
        
        # Use sample for faster training
        sample_size = min(5000, len(data))
        training_sample = data.tail(sample_size).copy()
        
        logger.info(f"Training ensemble on {len(training_sample)} records")
        
        # Train ensemble with reduced complexity
        ensemble.train_ensemble(training_sample, validation_split=0.2)
        
        # Save ensemble
        ensemble.save_models()
        
        # Get performance metrics
        performance = ensemble.get_model_performance(training_sample)
        
        results = {
            'model': 'Ensemble',
            'status': 'SUCCESS',
            'individual_models': len(ensemble.models),
            'training_samples': len(training_sample),
            'performance': performance
        }
        
        logger.info(f"✅ Ensemble Training Complete - {results['individual_models']} models trained")
        return results
        
    except Exception as e:
        logger.error(f"❌ Ensemble Training Failed: {e}")
        return {
            'model': 'Ensemble',
            'status': 'FAILED', 
            'error': str(e)
        }

def run_basic_backtesting(data):
    """Run basic backtesting validation"""
    logger.info("📈 Running Basic Backtesting...")
    
    try:
        from paper_trading_validator import PaperTradingValidator
        
        validator = PaperTradingValidator(initial_balance=10000)
        session_id = validator.start_session("backtest_validation")
        
        # Generate mock signals for backtesting
        test_signals = []
        test_data = data.tail(100).copy()  # Last 100 records
        
        for i, row in test_data.iterrows():
            # Simple signal generation based on price movement
            if i > 0:
                prev_close = test_data.iloc[i-1]['close']
                current_close = row['close']
                price_change = (current_close - prev_close) / prev_close * 100
                
                if price_change > 0.1:
                    direction = 'BUY'
                    accuracy = min(95, 85 + abs(price_change) * 10)
                elif price_change < -0.1:
                    direction = 'SELL'
                    accuracy = min(95, 85 + abs(price_change) * 10)
                else:
                    continue
                
                signal = {
                    'pair': 'EURUSD',
                    'direction': direction,
                    'entry_price': current_close,
                    'accuracy': accuracy,
                    'time_expiry': (datetime.now() + timedelta(minutes=2)).strftime('%H:%M:%S'),
                    'recommended_duration': 2
                }
                
                trade = validator.execute_paper_trade(signal)
                if trade:
                    validator.simulate_trade_result(trade)
                    test_signals.append(signal)
        
        final_stats = validator.end_session()
        
        results = {
            'backtesting': 'SUCCESS',
            'total_trades': final_stats.get('total_trades', 0),
            'win_rate': final_stats.get('win_rate', 0),
            'total_pnl': final_stats.get('total_pnl', 0),
            'final_balance': validator.current_balance
        }
        
        logger.info(f"✅ Backtesting Complete - Win Rate: {results['win_rate']:.2%}, PnL: ${results['total_pnl']:.2f}")
        return results
        
    except Exception as e:
        logger.error(f"❌ Backtesting Failed: {e}")
        return {
            'backtesting': 'FAILED',
            'error': str(e)
        }

def establish_benchmarks():
    """Establish baseline performance benchmarks"""
    logger.info("📊 Establishing Performance Benchmarks...")
    
    benchmarks = {
        'target_accuracy': 85.0,  # Minimum accuracy target
        'target_win_rate': 0.60,  # 60% win rate target
        'max_drawdown': 0.15,     # Maximum 15% drawdown
        'min_sharpe_ratio': 1.5,  # Minimum Sharpe ratio
        'latency_target': 100,    # Maximum 100ms per prediction
        'daily_signals': 20,      # Maximum 20 signals per day
        'risk_per_trade': 0.02,   # 2% risk per trade
        'established_date': datetime.now().isoformat()
    }
    
    # Save benchmarks
    with open('/workspace/models/performance_benchmarks.json', 'w') as f:
        json.dump(benchmarks, f, indent=2)
    
    logger.info("✅ Performance benchmarks established")
    return benchmarks

def main():
    """Main training orchestration"""
    logger.info("🚀 Starting Production Model Training")
    logger.info("=" * 60)
    
    # Ensure directories exist
    os.makedirs('/workspace/logs', exist_ok=True)
    os.makedirs('/workspace/models', exist_ok=True)
    
    training_results = {
        'training_date': datetime.now().isoformat(),
        'results': {}
    }
    
    try:
        # Load data
        data = load_and_prepare_data()
        
        # Train LSTM model
        lstm_results = train_lstm_model(data)
        training_results['results']['lstm'] = lstm_results
        
        # Train ensemble model (if LSTM successful)
        if lstm_results['status'] == 'SUCCESS':
            ensemble_results = train_ensemble_model(data)
            training_results['results']['ensemble'] = ensemble_results
        else:
            logger.warning("⚠️ Skipping ensemble training due to LSTM failure")
        
        # Run backtesting
        backtest_results = run_basic_backtesting(data)
        training_results['results']['backtesting'] = backtest_results
        
        # Establish benchmarks
        benchmarks = establish_benchmarks()
        training_results['benchmarks'] = benchmarks
        
        # Save complete results
        with open('/workspace/models/training_results.json', 'w') as f:
            json.dump(training_results, f, indent=2)
        
        logger.info("🎉 Training Complete! Results saved to training_results.json")
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 TRAINING SUMMARY")
        print("=" * 60)
        
        for model_name, result in training_results['results'].items():
            status = result.get('status', 'Unknown')
            print(f"{model_name.upper()}: {status}")
            
            if model_name == 'lstm' and status == 'SUCCESS':
                print(f"  - Accuracy: {result.get('accuracy', 0):.2f}%")
                print(f"  - Training samples: {result.get('training_samples', 0)}")
                
            elif model_name == 'backtesting' and status == 'SUCCESS':
                print(f"  - Win rate: {result.get('win_rate', 0):.2%}")
                print(f"  - Total PnL: ${result.get('total_pnl', 0):.2f}")
        
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        training_results['status'] = 'FAILED'
        training_results['error'] = str(e)

if __name__ == "__main__":
    main()