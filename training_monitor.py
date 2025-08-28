#!/usr/bin/env python3
"""
📊 TRADING SYSTEM TRAINING MONITOR
Real-time monitoring of model training and paper trading progress
"""

import os
import time
import pandas as pd
import json
from datetime import datetime, timedelta
import subprocess
import sys

class TradingSystemMonitor:
    """Monitor training progress and system status"""
    
    def __init__(self):
        self.logs_dir = '/workspace/logs'
        self.models_dir = '/workspace/models'
        self.data_dir = '/workspace/data'
        
    def check_background_processes(self):
        """Check if training processes are running"""
        print("🔍 CHECKING BACKGROUND PROCESSES")
        print("=" * 40)
        
        try:
            # Check for running python processes
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            processes = result.stdout
            
            training_processes = []
            for line in processes.split('\n'):
                if 'python3' in line and any(keyword in line for keyword in 
                    ['ensemble_trainer', 'paper_trading', 'enhanced_ensemble']):
                    training_processes.append(line.strip())
            
            if training_processes:
                print("✅ Found active training processes:")
                for proc in training_processes:
                    print(f"   • {proc}")
            else:
                print("⚠️ No active training processes found")
                
            return len(training_processes)
            
        except Exception as e:
            print(f"❌ Error checking processes: {e}")
            return 0
    
    def check_training_logs(self):
        """Check training progress from logs"""
        print("\n📋 TRAINING PROGRESS")
        print("=" * 25)
        
        log_files = [
            'enhanced_ensemble_training.log',
            'training.log',
            'ensemble_training.log',
            'lstm_training.log'
        ]
        
        latest_training_info = {}
        
        for log_file in log_files:
            log_path = os.path.join(self.logs_dir, log_file)
            if os.path.exists(log_path):
                try:
                    # Get file size and last modified
                    size = os.path.getsize(log_path)
                    mtime = datetime.fromtimestamp(os.path.getmtime(log_path))
                    
                    print(f"📄 {log_file}")
                    print(f"   Size: {size:,} bytes")
                    print(f"   Modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    # Read last few lines
                    if size > 0:
                        with open(log_path, 'r') as f:
                            lines = f.readlines()
                            if lines:
                                print(f"   Last entry: {lines[-1].strip()}")
                                latest_training_info[log_file] = {
                                    'last_line': lines[-1].strip(),
                                    'total_lines': len(lines),
                                    'size': size,
                                    'modified': mtime
                                }
                    print()
                    
                except Exception as e:
                    print(f"   ❌ Error reading {log_file}: {e}")
        
        return latest_training_info
    
    def check_paper_trading_status(self):
        """Check paper trading progress"""
        print("💰 PAPER TRADING STATUS") 
        print("=" * 28)
        
        paper_trading_log = os.path.join(self.logs_dir, 'paper_trading.log')
        
        if os.path.exists(paper_trading_log):
            try:
                size = os.path.getsize(paper_trading_log)
                mtime = datetime.fromtimestamp(os.path.getmtime(paper_trading_log))
                
                print(f"📊 Paper Trading Log: {size:,} bytes")
                print(f"🕒 Last Updated: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Count recent trades
                with open(paper_trading_log, 'r') as f:
                    lines = f.readlines()
                    
                recent_trades = 0
                win_count = 0
                total_trades = 0
                
                # Look for trade patterns in last 100 lines
                for line in lines[-100:]:
                    if 'Trade' in line and 'WIN' in line:
                        win_count += 1
                        total_trades += 1
                    elif 'Trade' in line and ('LOSS' in line or 'LOSE' in line):
                        total_trades += 1
                
                if total_trades > 0:
                    win_rate = (win_count / total_trades) * 100
                    print(f"🎯 Recent Win Rate: {win_rate:.1f}% ({win_count}/{total_trades})")
                else:
                    print("⚠️ No recent trading activity detected")
                
                # Show last few lines
                if lines:
                    print("📝 Recent Activity:")
                    for line in lines[-3:]:
                        print(f"   {line.strip()}")
                
            except Exception as e:
                print(f"❌ Error reading paper trading log: {e}")
        else:
            print("❌ Paper trading log not found")
    
    def check_model_files(self):
        """Check model file status"""
        print("\n🧠 MODEL FILES STATUS")
        print("=" * 25)
        
        model_files = [
            'production_lstm_trained.h5',
            'feature_scaler.pkl',
            'best_model.h5'
        ]
        
        for model_file in model_files:
            model_path = os.path.join(self.models_dir, model_file)
            if os.path.exists(model_path):
                size = os.path.getsize(model_path)
                mtime = datetime.fromtimestamp(os.path.getmtime(model_path))
                print(f"✅ {model_file}: {size:,} bytes (Modified: {mtime.strftime('%H:%M:%S')})")
            else:
                print(f"❌ Missing: {model_file}")
    
    def check_data_status(self):
        """Check data availability and quality"""
        print("\n📊 DATA STATUS")
        print("=" * 15)
        
        # Check market data
        market_data_dir = os.path.join(self.data_dir, 'real_market_data')
        if os.path.exists(market_data_dir):
            csv_files = [f for f in os.listdir(market_data_dir) if f.endswith('.csv')]
            total_size = sum(os.path.getsize(os.path.join(market_data_dir, f)) for f in csv_files)
            print(f"📈 Market Data: {len(csv_files)} files ({total_size/(1024*1024):.1f} MB)")
            
            # Check main combined file
            main_file = os.path.join(market_data_dir, 'combined_market_data_20250816_092932.csv')
            if os.path.exists(main_file):
                try:
                    df = pd.read_csv(main_file, nrows=5)  # Just peek at structure
                    print(f"✅ Main dataset columns: {list(df.columns)}")
                    print(f"✅ Sample symbols: {df['symbol'].unique()[:3] if 'symbol' in df.columns else 'No symbol column'}")
                except Exception as e:
                    print(f"⚠️ Issue reading main dataset: {e}")
        else:
            print("❌ Market data directory not found")
    
    def diagnose_training_issues(self):
        """Diagnose common training issues"""
        print("\n🔧 ISSUE DIAGNOSIS")
        print("=" * 20)
        
        # Check for common errors in logs
        log_files = ['enhanced_ensemble_training.log', 'training.log']
        
        for log_file in log_files:
            log_path = os.path.join(self.logs_dir, log_file)
            if os.path.exists(log_path):
                try:
                    with open(log_path, 'r') as f:
                        content = f.read()
                        
                    if 'Found array with 0 sample(s)' in content:
                        print("⚠️ ISSUE: Data preprocessing removing all samples")
                        print("   💡 Solution: Check feature engineering and data cleaning logic")
                    
                    if 'CUDA' in content and 'failed' in content:
                        print("⚠️ ISSUE: CUDA/GPU problems detected")
                        print("   💡 Solution: Training will use CPU (slower but functional)")
                    
                    if 'Memory' in content or 'OOM' in content:
                        print("⚠️ ISSUE: Memory issues detected")
                        print("   💡 Solution: Reduce batch size or model complexity")
                    
                    if 'ImportError' in content or 'ModuleNotFoundError' in content:
                        print("⚠️ ISSUE: Missing dependencies detected")
                        print("   💡 Solution: Check requirements.txt installation")
                        
                except Exception as e:
                    print(f"❌ Error checking {log_file}: {e}")
    
    def provide_recommendations(self):
        """Provide actionable recommendations"""
        print("\n💡 RECOMMENDATIONS")
        print("=" * 20)
        
        print("🚀 IMMEDIATE ACTIONS:")
        print("1. Fix data preprocessing issue:")
        print("   python3 -c \"import pandas as pd; df=pd.read_csv('/workspace/data/real_market_data/combined_market_data_20250816_092932.csv'); print(f'Rows: {len(df)}, Cols: {len(df.columns)}')\"")
        
        print("\n2. Start simplified training:")
        print("   python3 enhanced_lstm_trainer.py")
        
        print("\n3. Test paper trading with shorter duration:")
        print("   python3 paper_trading_engine.py --duration=1day --test-mode")
        
        print("\n📋 MONITORING COMMANDS:")
        print("• Watch logs: tail -f /workspace/logs/enhanced_ensemble_training.log")
        print("• Check processes: ps aux | grep python3")
        print("• Monitor this script: python3 training_monitor.py")
        
        print("\n🎯 SUCCESS CRITERIA:")
        print("• Training completes without errors")
        print("• Model accuracy > 60%")
        print("• Paper trading shows positive signals") 
        print("• Win rate > 55% (initially, building to 65%+)")
    
    def run_continuous_monitoring(self, interval=30):
        """Run continuous monitoring"""
        print("🚀 STARTING CONTINUOUS MONITORING")
        print(f"⏰ Update interval: {interval} seconds")
        print("=" * 50)
        
        try:
            while True:
                print(f"\n📊 SYSTEM STATUS UPDATE - {datetime.now().strftime('%H:%M:%S')}")
                print("=" * 60)
                
                # Quick status checks
                active_processes = self.check_background_processes()
                self.check_training_logs()
                self.check_paper_trading_status()
                
                if active_processes == 0:
                    print("\n⚠️ No active processes - checking for issues...")
                    self.diagnose_training_issues()
                    self.provide_recommendations()
                    break
                
                print(f"\n⏰ Next update in {interval} seconds... (Ctrl+C to stop)")
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Monitoring stopped by user")
    
    def generate_full_status_report(self):
        """Generate comprehensive status report"""
        print("🔍 ULTIMATE TRADING SYSTEM - STATUS REPORT")
        print("=" * 50)
        print(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Run all checks
        self.check_background_processes()
        self.check_training_logs()
        self.check_paper_trading_status()
        self.check_model_files()
        self.check_data_status()
        self.diagnose_training_issues()
        self.provide_recommendations()
        
        print("\n" + "=" * 50)
        print("📊 REPORT COMPLETE")

if __name__ == "__main__":
    monitor = TradingSystemMonitor()
    
    if len(sys.argv) > 1 and sys.argv[1] == '--continuous':
        monitor.run_continuous_monitoring()
    else:
        monitor.generate_full_status_report()