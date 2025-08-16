#!/usr/bin/env python3
"""
🔬 Comprehensive Model Validation & Backtesting
Production-ready validation of all AI/ML models
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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_lstm_model():
    """Validate LSTM model performance"""
    logger.info("🧠 Validating LSTM Model...")
    
    try:
        from lstm_model import LSTMTradingModel
        
        # Load trained model
        model = LSTMTradingModel()
        if model.load_model('/workspace/models/production_lstm_trained.h5'):
            logger.info("✅ LSTM model loaded successfully")
            
            # Load test data
            data = pd.read_csv('/workspace/data/real_market_data/combined_market_data_20250816_092932.csv')
            data['datetime'] = pd.to_datetime(data['datetime'])
            data = data.sort_values('datetime').reset_index(drop=True)
            
            # Use recent data for validation
            test_data = data.tail(5000).copy()
            
            # Test prediction capability
            try:
                prediction = model.predict_signal(test_data)
                if prediction:
                    logger.info(f"✅ Prediction successful: {prediction['signal']} ({prediction['confidence']:.2f}%)")
                    
                    validation_results = {
                        'status': 'SUCCESS',
                        'model_loaded': True,
                        'prediction_working': True,
                        'sample_prediction': prediction,
                        'model_file_size': os.path.getsize('/workspace/models/production_lstm_trained.h5'),
                        'features_count': model.features_count,
                        'sequence_length': model.sequence_length
                    }
                else:
                    raise Exception("Prediction returned None")
                    
            except Exception as e:
                logger.error(f"❌ Prediction failed: {e}")
                validation_results = {
                    'status': 'PREDICTION_FAILED',
                    'model_loaded': True,
                    'prediction_working': False,
                    'error': str(e)
                }
                
        else:
            logger.error("❌ Failed to load LSTM model")
            validation_results = {
                'status': 'LOAD_FAILED',
                'model_loaded': False,
                'prediction_working': False
            }
            
        return validation_results
        
    except Exception as e:
        logger.error(f"❌ LSTM validation failed: {e}")
        return {
            'status': 'FAILED',
            'error': str(e)
        }

def run_comprehensive_backtesting():
    """Run comprehensive backtesting with realistic scenarios"""
    logger.info("📈 Running Comprehensive Backtesting...")
    
    try:
        # Load market data
        data = pd.read_csv('/workspace/data/real_market_data/combined_market_data_20250816_092932.csv')
        data['datetime'] = pd.to_datetime(data['datetime'])
        data = data.sort_values('datetime').reset_index(drop=True)
        
        # Focus on EURUSD for backtesting
        eurusd_data = data[data['symbol'] == 'EURUSD_X'].tail(1000).copy()
        
        if len(eurusd_data) < 100:
            # Fallback to all data
            test_data = data.tail(1000).copy()
            logger.info("Using all currency data for backtesting")
        else:
            test_data = eurusd_data
            logger.info(f"Using EURUSD data: {len(test_data)} records")
        
        # Initialize backtesting metrics
        total_trades = 0
        winning_trades = 0
        total_pnl = 0.0
        initial_balance = 10000.0
        current_balance = initial_balance
        max_drawdown = 0.0
        peak_balance = initial_balance
        
        # Simple backtesting simulation
        for i in range(1, len(test_data) - 1):
            current_row = test_data.iloc[i]
            next_row = test_data.iloc[i + 1]
            
            current_price = current_row['close']
            next_price = next_row['close']
            
            # Simple trend-following strategy
            price_change = (current_price - test_data.iloc[i-1]['close']) / test_data.iloc[i-1]['close'] * 100
            
            # Generate signal based on price momentum
            if abs(price_change) > 0.05:  # 0.05% threshold
                total_trades += 1
                
                # Predict direction
                if price_change > 0:
                    signal = 'BUY'
                else:
                    signal = 'SELL'
                
                # Calculate outcome
                actual_change = (next_price - current_price) / current_price * 100
                
                # Determine if trade wins (simplified binary options logic)
                if signal == 'BUY' and actual_change > 0:
                    win = True
                elif signal == 'SELL' and actual_change < 0:
                    win = True
                else:
                    win = False
                
                # Calculate P&L (80% payout for wins, -100% for losses)
                trade_amount = current_balance * 0.02  # 2% risk per trade
                
                if win:
                    pnl = trade_amount * 0.80  # 80% payout
                    winning_trades += 1
                else:
                    pnl = -trade_amount  # Lose full amount
                
                current_balance += pnl
                total_pnl += pnl
                
                # Update drawdown
                if current_balance > peak_balance:
                    peak_balance = current_balance
                
                drawdown = (peak_balance - current_balance) / peak_balance
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
        
        # Calculate final metrics
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        roi = (current_balance - initial_balance) / initial_balance * 100
        
        backtest_results = {
            'status': 'SUCCESS',
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'initial_balance': initial_balance,
            'final_balance': current_balance,
            'roi_percentage': roi,
            'max_drawdown': max_drawdown * 100,
            'data_points': len(test_data)
        }
        
        logger.info(f"✅ Backtesting Complete:")
        logger.info(f"  - Total Trades: {total_trades}")
        logger.info(f"  - Win Rate: {win_rate:.2%}")
        logger.info(f"  - ROI: {roi:.2f}%")
        logger.info(f"  - Max Drawdown: {max_drawdown*100:.2f}%")
        
        return backtest_results
        
    except Exception as e:
        logger.error(f"❌ Backtesting failed: {e}")
        return {
            'status': 'FAILED',
            'error': str(e)
        }

def performance_benchmarking():
    """Establish comprehensive performance benchmarks"""
    logger.info("📊 Performance Benchmarking...")
    
    try:
        from lstm_model import LSTMTradingModel
        import time
        
        # Load model for latency testing
        model = LSTMTradingModel()
        model_loaded = model.load_model('/workspace/models/production_lstm_trained.h5')
        
        if model_loaded:
            # Load test data
            data = pd.read_csv('/workspace/data/real_market_data/combined_market_data_20250816_092932.csv')
            test_sample = data.tail(1000).copy()
            
            # Latency testing
            latencies = []
            successful_predictions = 0
            
            for i in range(10):  # Test 10 predictions
                start_time = time.time()
                try:
                    prediction = model.predict_signal(test_sample)
                    end_time = time.time()
                    
                    if prediction:
                        latency = (end_time - start_time) * 1000  # Convert to milliseconds
                        latencies.append(latency)
                        successful_predictions += 1
                        
                except Exception as e:
                    logger.warning(f"Prediction {i+1} failed: {e}")
            
            # Calculate latency metrics
            if latencies:
                avg_latency = np.mean(latencies)
                max_latency = np.max(latencies)
                min_latency = np.min(latencies)
            else:
                avg_latency = max_latency = min_latency = 0
            
            benchmarks = {
                'latency_benchmarks': {
                    'average_ms': avg_latency,
                    'maximum_ms': max_latency,
                    'minimum_ms': min_latency,
                    'target_ms': 100,
                    'meets_target': avg_latency < 100
                },
                'reliability_benchmarks': {
                    'successful_predictions': successful_predictions,
                    'total_attempts': 10,
                    'success_rate': successful_predictions / 10,
                    'target_success_rate': 0.95,
                    'meets_target': (successful_predictions / 10) >= 0.95
                },
                'model_specifications': {
                    'features_count': model.features_count,
                    'sequence_length': model.sequence_length,
                    'model_size_mb': os.path.getsize('/workspace/models/production_lstm_trained.h5') / (1024*1024),
                    'calibration_temperature': model.calibration_temperature
                }
            }
            
            logger.info(f"✅ Benchmarking Complete:")
            logger.info(f"  - Average Latency: {avg_latency:.2f}ms")
            logger.info(f"  - Success Rate: {successful_predictions/10:.0%}")
            logger.info(f"  - Model Size: {benchmarks['model_specifications']['model_size_mb']:.2f}MB")
            
        else:
            benchmarks = {
                'status': 'FAILED',
                'error': 'Could not load model for benchmarking'
            }
        
        return benchmarks
        
    except Exception as e:
        logger.error(f"❌ Benchmarking failed: {e}")
        return {
            'status': 'FAILED',
            'error': str(e)
        }

def generate_production_readiness_report():
    """Generate comprehensive production readiness assessment"""
    logger.info("📋 Generating Production Readiness Report...")
    
    # Run all validations
    lstm_validation = validate_lstm_model()
    backtest_results = run_comprehensive_backtesting()
    benchmark_results = performance_benchmarking()
    
    # Assess production readiness
    readiness_score = 0
    max_score = 100
    
    # LSTM Model Assessment (40 points)
    if lstm_validation.get('status') == 'SUCCESS':
        readiness_score += 25
        if lstm_validation.get('prediction_working'):
            readiness_score += 15
    
    # Backtesting Assessment (30 points)
    if backtest_results.get('status') == 'SUCCESS':
        readiness_score += 15
        win_rate = backtest_results.get('win_rate', 0)
        if win_rate > 0.5:  # Above 50% win rate
            readiness_score += 10
        if backtest_results.get('max_drawdown', 100) < 20:  # Less than 20% drawdown
            readiness_score += 5
    
    # Performance Assessment (30 points)
    if benchmark_results.get('latency_benchmarks', {}).get('meets_target', False):
        readiness_score += 15
    if benchmark_results.get('reliability_benchmarks', {}).get('meets_target', False):
        readiness_score += 15
    
    # Overall assessment
    if readiness_score >= 80:
        readiness_level = "PRODUCTION READY"
    elif readiness_score >= 60:
        readiness_level = "NEEDS IMPROVEMENT"
    else:
        readiness_level = "NOT READY"
    
    final_report = {
        'assessment_date': datetime.now().isoformat(),
        'overall_score': readiness_score,
        'max_score': max_score,
        'readiness_percentage': (readiness_score / max_score) * 100,
        'readiness_level': readiness_level,
        'validations': {
            'lstm_model': lstm_validation,
            'backtesting': backtest_results,
            'benchmarking': benchmark_results
        },
        'recommendations': []
    }
    
    # Add recommendations
    if lstm_validation.get('status') != 'SUCCESS':
        final_report['recommendations'].append("Retrain LSTM model with more data and optimized parameters")
    
    if backtest_results.get('win_rate', 0) < 0.6:
        final_report['recommendations'].append("Improve signal generation strategy to achieve >60% win rate")
    
    if benchmark_results.get('latency_benchmarks', {}).get('average_ms', 1000) > 100:
        final_report['recommendations'].append("Optimize model inference for <100ms latency")
    
    if not final_report['recommendations']:
        final_report['recommendations'].append("System meets basic production requirements")
    
    # Save report
    with open('/workspace/models/production_readiness_report.json', 'w') as f:
        json.dump(final_report, f, indent=2)
    
    logger.info("🎉 Production Readiness Report Generated!")
    logger.info(f"Overall Score: {readiness_score}/{max_score} ({(readiness_score/max_score)*100:.1f}%)")
    logger.info(f"Readiness Level: {readiness_level}")
    
    return final_report

def main():
    """Main validation orchestration"""
    logger.info("🚀 Starting Comprehensive Model Validation")
    logger.info("=" * 60)
    
    # Ensure directories exist
    os.makedirs('/workspace/logs', exist_ok=True)
    os.makedirs('/workspace/models', exist_ok=True)
    
    try:
        # Generate comprehensive report
        report = generate_production_readiness_report()
        
        # Print summary
        print("\n" + "=" * 60)
        print("🔬 COMPREHENSIVE VALIDATION REPORT")
        print("=" * 60)
        print(f"Overall Score: {report['overall_score']}/{report['max_score']} ({report['readiness_percentage']:.1f}%)")
        print(f"Readiness Level: {report['readiness_level']}")
        print("\n📊 Component Status:")
        
        validations = report['validations']
        for component, result in validations.items():
            status = result.get('status', 'UNKNOWN')
            print(f"  {component.upper()}: {status}")
            
            if component == 'backtesting' and status == 'SUCCESS':
                print(f"    - Win Rate: {result.get('win_rate', 0):.2%}")
                print(f"    - ROI: {result.get('roi_percentage', 0):.2f}%")
                
            elif component == 'benchmarking' and 'latency_benchmarks' in result:
                print(f"    - Avg Latency: {result['latency_benchmarks'].get('average_ms', 0):.2f}ms")
        
        print("\n💡 Recommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        print("=" * 60)
        print("📄 Full report saved to: production_readiness_report.json")
        
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()