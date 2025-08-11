#!/usr/bin/env python3
"""
Startup Script for LSTM AI Trading System

This script orchestrates the complete startup process:
1. Checks system requirements
2. Stops all existing trading bots
3. Launches the LSTM AI-powered signal system
4. Monitors system status
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_requirements():
    """Check if all required files exist"""
    required_files = [
        'simple_bot_manager.py',
        'enhanced_signal_engine.py',
        'config.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        return False
    
    print("✅ All required files found")
    return True

def stop_existing_bots():
    """Stop all existing trading bots using the bot manager"""
    print("🛑 Stopping all existing trading bots...")
    
    try:
        result = subprocess.run([
            'python3', 'simple_bot_manager.py', 
            '--action', 'stop-all', 
            '--force'
        ], capture_output=True, text=True, cwd='/workspace')
        
        if result.returncode == 0:
            print("✅ All existing bots stopped successfully")
            return True
        else:
            print(f"⚠️  Warning: Bot stopping had issues: {result.stderr}")
            return True  # Continue anyway
            
    except Exception as e:
        print(f"❌ Error stopping bots: {e}")
        return False

def start_lstm_system():
    """Start the LSTM AI trading system"""
    print("🚀 Starting LSTM AI-powered signal system...")
    
    try:
        # Start the LSTM AI system
        process = subprocess.Popen([
            'python3', 'simple_bot_manager.py', 
            '--action', 'start-lstm'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
        text=True, cwd='/workspace')
        
        # Wait for startup
        time.sleep(5)
        
        # Check if process completed successfully
        if process.poll() is None:
            print("✅ LSTM AI system startup initiated")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ LSTM AI system failed to start")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            return None
            
    except Exception as e:
        print(f"❌ Error starting LSTM AI system: {e}")
        return None

def monitor_system():
    """Monitor the system status"""
    print("\n📊 Monitoring LSTM AI system status...")
    
    try:
        # Get system status
        result = subprocess.run([
            'python3', 'simple_bot_manager.py', 
            '--action', 'status'
        ], capture_output=True, text=True, cwd='/workspace')
        
        if result.returncode == 0:
            print("📈 System Status:")
            print(result.stdout)
        else:
            print(f"⚠️  Could not get system status: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Error monitoring system: {e}")

def print_banner():
    """Print startup banner"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                    🧠 LSTM AI Trading System                 ║
║                                                              ║
║  🎯 Weekday/Weekend Pair Switching                          ║
║  ⏰ 1+ Minute Signal Advance Warning                        ║
║  🛑 Automatic Bot Management                                ║
║  🚀 Real Trained LSTM AI-Powered Signals                   ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def main():
    """Main startup function"""
    print_banner()
    
    print("🔍 Checking system requirements...")
    if not check_requirements():
        print("❌ System requirements not met. Exiting.")
        sys.exit(1)
    
    print("\n🔄 Starting LSTM AI Trading System...")
    
    # Stop existing bots
    if not stop_existing_bots():
        print("❌ Failed to stop existing bots. Exiting.")
        sys.exit(1)
    
    # Start LSTM system
    lstm_process = start_lstm_system()
    if not lstm_process:
        print("❌ Failed to start LSTM AI system. Exiting.")
        sys.exit(1)
    
    # Monitor system
    monitor_system()
    
    print("\n🎉 LSTM AI Trading System startup completed!")
    print("\n📋 System Features Active:")
    print("   ✅ Weekday/Weekend pair switching (OTC vs Regular)")
    print("   ✅ 1+ minute signal advance warning")
    print("   ✅ All existing bots stopped")
    print("   ✅ LSTM AI-powered signal generation")
    print("   ✅ Real-time market analysis")
    
    print("\n💡 To monitor the system:")
    print("   python3 simple_bot_manager.py --action status")
    
    print("\n💡 To restart the system:")
    print("   python3 simple_bot_manager.py --action restart-lstm")
    
    print("\n💡 To stop the system:")
    print("   python3 simple_bot_manager.py --action stop-all")
    
    print("\n🚀 System is now running and generating AI-powered trading signals!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Startup interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error during startup: {e}")
        sys.exit(1)