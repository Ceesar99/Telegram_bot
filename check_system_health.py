#!/usr/bin/env python3
"""
System Health Check Script
Monitors all trading system components and reports status
"""

import sys
import os
import asyncio
import logging
from datetime import datetime
import psutil

# Add workspace to Python path
sys.path.insert(0, '/workspace')

def check_system_resources():
    """Check system resource usage"""
    print("🔍 **SYSTEM RESOURCES**")
    print("=" * 50)
    
    # CPU Usage
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"CPU Usage: {cpu_percent}%")
    
    # Memory Usage
    memory = psutil.virtual_memory()
    print(f"Memory Usage: {memory.percent}% ({memory.used // (1024**3)}GB / {memory.total // (1024**3)}GB)")
    
    # Disk Usage
    disk = psutil.disk_usage('/workspace')
    print(f"Disk Usage: {disk.percent}% ({disk.used // (1024**3)}GB / {disk.total // (1024**3)}GB)")
    
    # Network
    network = psutil.net_io_counters()
    print(f"Network: {network.bytes_sent // (1024**2)}MB sent, {network.bytes_recv // (1024**2)}MB received")
    
    print()

def check_python_environment():
    """Check Python environment and packages"""
    print("🐍 **PYTHON ENVIRONMENT**")
    print("=" * 50)
    
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    print(f"Python Path: {sys.path[0]}")
    
    # Check required packages with correct import names
    required_packages = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'), 
        ('tensorflow', 'tensorflow'),
        ('requests', 'requests'),
        ('matplotlib', 'matplotlib'),
        ('pytz', 'pytz'),
        ('websocket-client', 'websocket'),
        ('python-telegram-bot', 'telegram'),
        ('psutil', 'psutil'),
        ('scikit-learn', 'sklearn')
    ]
    
    print("\nRequired Packages:")
    for package_name, import_name in required_packages:
        try:
            module = __import__(import_name)
            version = getattr(module, '__version__', 'Unknown')
            print(f"  ✅ {package_name}: {version}")
        except ImportError:
            print(f"  ❌ {package_name}: Not installed")
    
    print()

def check_file_structure():
    """Check workspace file structure"""
    print("📁 **WORKSPACE STRUCTURE**")
    print("=" * 50)
    
    required_dirs = [
        '/workspace/logs',
        '/workspace/data', 
        '/workspace/models',
        '/workspace/backup'
    ]
    
    required_files = [
        '/workspace/config.py',
        '/workspace/signal_engine.py',
        '/workspace/telegram_bot.py',
        '/workspace/pocket_option_api.py',
        '/workspace/data_manager.py',
        '/workspace/start_system.py'
    ]
    
    print("Required Directories:")
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"  ✅ {directory}")
        else:
            print(f"  ❌ {directory} - Missing")
    
    print("\nRequired Files:")
    for file_path in required_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"  ✅ {file_path} ({size} bytes)")
        else:
            print(f"  ❌ {file_path} - Missing")
    
    print()

def check_log_files():
    """Check log files and recent activity"""
    print("📋 **LOG FILES STATUS**")
    print("=" * 50)
    
    log_dir = '/workspace/logs'
    if not os.path.exists(log_dir):
        print("❌ Logs directory not found")
        return
    
    log_files = [
        'unified_system.log',
        'signal_engine.log', 
        'pocket_option_api.log',
        'telegram_bot.log',
        'system_manager.log'
    ]
    
    for log_file in log_files:
        log_path = os.path.join(log_dir, log_file)
        if os.path.exists(log_path):
            size = os.path.getsize(log_path)
            mtime = datetime.fromtimestamp(os.path.getmtime(log_path))
            age = datetime.now() - mtime
            
            if age.total_seconds() < 3600:  # Less than 1 hour
                status = "🟢 Recent"
            elif age.total_seconds() < 86400:  # Less than 1 day
                status = "🟡 Old"
            else:
                status = "🔴 Very Old"
                
            print(f"  {status} {log_file}: {size} bytes, Last modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print(f"  ❌ {log_file}: Not found")
    
    print()

async def check_component_health():
    """Check individual component health"""
    print("🧩 **COMPONENT HEALTH**")
    print("=" * 50)
    
    try:
        # Test config import
        import config
        print("✅ Config: Loaded successfully")
        
        # Test data manager
        try:
            from data_manager import DataManager
            data_manager = DataManager()
            symbols = data_manager.get_available_symbols()
            print(f"✅ Data Manager: {len(symbols)} symbols available")
        except Exception as e:
            print(f"❌ Data Manager: {e}")
        
        # Test performance tracker
        try:
            from performance_tracker import PerformanceTracker
            perf_tracker = PerformanceTracker()
            db_ok = perf_tracker.test_connection()
            print(f"✅ Performance Tracker: Database {'connected' if db_ok else 'disconnected'}")
        except Exception as e:
            print(f"❌ Performance Tracker: {e}")
        
        # Test signal engine
        try:
            from signal_engine import SignalEngine
            signal_engine = SignalEngine()
            print(f"✅ Signal Engine: Model {'loaded' if signal_engine.is_model_loaded() else 'not loaded'}")
            print(f"✅ Signal Engine: Data {'connected' if signal_engine.is_data_connected() else 'disconnected'}")
        except Exception as e:
            print(f"❌ Signal Engine: {e}")
        
        # Test PocketOption API
        try:
            from pocket_option_api import PocketOptionAPI
            api = PocketOptionAPI()
            print("✅ PocketOption API: Initialized")
        except Exception as e:
            print(f"❌ PocketOption API: {e}")
        
        # Test Telegram Bot
        try:
            from telegram_bot import TradingBot
            bot = TradingBot()
            print("✅ Telegram Bot: Initialized")
        except Exception as e:
            print(f"❌ Telegram Bot: {e}")
            
    except Exception as e:
        print(f"❌ Component health check failed: {e}")
    
    print()

def check_running_processes():
    """Check for running trading system processes"""
    print("🔄 **RUNNING PROCESSES**")
    print("=" * 50)
    
    trading_processes = []
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
            if any(keyword in cmdline.lower() for keyword in ['python', 'trading', 'bot', 'signal']):
                trading_processes.append({
                    'pid': proc.info['pid'],
                    'name': proc.info['name'],
                    'cmdline': cmdline[:100] + '...' if len(cmdline) > 100 else cmdline
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    if trading_processes:
        print("Found trading system processes:")
        for proc in trading_processes:
            print(f"  🔄 PID {proc['pid']}: {proc['name']} - {proc['cmdline']}")
    else:
        print("❌ No trading system processes found")
    
    print()

def generate_health_report():
    """Generate comprehensive health report"""
    print("🏥 **SYSTEM HEALTH REPORT**")
    print("=" * 60)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    check_system_resources()
    check_python_environment()
    check_file_structure()
    check_log_files()
    check_running_processes()
    
    # Async component check
    print("Checking component health (this may take a moment)...")
    asyncio.run(check_component_health())
    
    print("🏁 **HEALTH CHECK COMPLETE**")
    print("=" * 60)

if __name__ == "__main__":
    try:
        generate_health_report()
    except KeyboardInterrupt:
        print("\n🛑 Health check interrupted by user")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        sys.exit(1)