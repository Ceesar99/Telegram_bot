#!/usr/bin/env python3
"""
Unified Trading System Startup Script

This script provides a user-friendly interface to start the unified trading system
in different modes: original, institutional, or hybrid.

Usage:
    python3 start_unified_system.py [mode]
    
Modes:
    original      - Run only the original binary options trading bot
    institutional - Run only the institutional-grade trading system
    hybrid        - Run both systems simultaneously (default)
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def print_banner():
    """Print startup banner"""
    print("=" * 70)
    print("🚀 UNIFIED TRADING SYSTEM - STARTUP")
    print("=" * 70)
    print("Original Bot + Institutional Grade System")
    print("=" * 70)
    print()

def print_mode_info():
    """Print information about available modes"""
    print("📋 Available Modes:")
    print("  1. original      - Original binary options trading bot")
    print("  2. institutional - Institutional-grade trading system")
    print("  3. hybrid        - Both systems running simultaneously")
    print()

def check_system_requirements():
    """Check if system meets requirements"""
    print("🔍 Checking system requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    
    print(f"✅ Python version: {sys.version.split()[0]}")
    
    # Check required directories
    required_dirs = [
        '/workspace/logs',
        '/workspace/data',
        '/workspace/models',
        '/workspace/backup'
    ]
    
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✅ Directory: {directory}")
        else:
            print(f"❌ Directory: {directory} - Creating...")
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"✅ Directory: {directory} - Created")
    
    # Check core files
    required_files = [
        'unified_trading_system.py',
        'config.py',
        'telegram_bot.py',
        'signal_engine.py'
    ]
    
    for file in required_files:
        if os.path.exists(f'/workspace/{file}'):
            print(f"✅ File: {file}")
        else:
            print(f"❌ File: {file} - Missing")
            return False
    
    print("✅ System requirements check passed")
    return True

def check_dependencies():
    """Check if core dependencies are available"""
    print("\n📦 Checking dependencies...")
    
    try:
        import telegram
        print("✅ python-telegram-bot")
    except ImportError:
        print("❌ python-telegram-bot - Missing")
        return False
    
    try:
        import requests
        print("✅ requests")
    except ImportError:
        print("❌ requests - Missing")
        return False
    
    try:
        import websocket
        print("✅ websocket-client")
    except ImportError:
        print("❌ websocket-client - Missing")
        return False
    
    try:
        import aiohttp
        print("✅ aiohttp")
    except ImportError:
        print("❌ aiohttp - Missing")
        return False
    
    print("✅ Core dependencies available")
    return True

def check_configuration():
    """Check if configuration is properly set"""
    print("\n⚙️  Checking configuration...")
    
    try:
        from config import TELEGRAM_BOT_TOKEN, TELEGRAM_USER_ID, POCKET_OPTION_SSID
        
        issues = []
        
        if TELEGRAM_BOT_TOKEN and TELEGRAM_BOT_TOKEN != "YOUR_BOT_TOKEN":
            print("✅ Telegram Bot Token: Configured")
        else:
            issues.append("Telegram Bot Token not configured")
        
        if TELEGRAM_USER_ID and TELEGRAM_USER_ID != "YOUR_USER_ID":
            print("✅ Telegram User ID: Configured")
        else:
            issues.append("Telegram User ID not configured")
        
        if POCKET_OPTION_SSID and len(POCKET_OPTION_SSID) > 10:
            print("✅ Pocket Option SSID: Configured")
        else:
            issues.append("Pocket Option SSID not configured")
        
        if issues:
            print("⚠️  Configuration issues found:")
            for issue in issues:
                print(f"   - {issue}")
            return False
        
        print("✅ Configuration check passed")
        return True
        
    except ImportError as e:
        print(f"❌ Configuration import error: {e}")
        return False

def get_user_mode():
    """Get user's preferred mode"""
    print("\n🎯 Select System Mode:")
    print("1. original      - Original binary options trading bot")
    print("2. institutional - Institutional-grade trading system")
    print("3. hybrid        - Both systems running simultaneously")
    print("4. exit          - Exit startup")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == "1":
                return "original"
            elif choice == "2":
                return "institutional"
            elif choice == "3":
                return "hybrid"
            elif choice == "4":
                return "exit"
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
        except KeyboardInterrupt:
            print("\n\n⚠️  Startup interrupted by user")
            return "exit"
        except EOFError:
            print("\n\n⚠️  Startup interrupted")
            return "exit"

def start_system(mode):
    """Start the unified trading system"""
    print(f"\n🚀 Starting Unified Trading System in {mode} mode...")
    print("=" * 50)
    
    try:
        # Run the unified system
        cmd = [sys.executable, "unified_trading_system.py", mode]
        
        print(f"Command: {' '.join(cmd)}")
        print("=" * 50)
        
        # Start the process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Stream output in real-time
        print("📊 System Output:")
        print("-" * 30)
        
        for line in process.stdout:
            print(line.rstrip())
        
        # Wait for process to complete
        process.wait()
        
        if process.returncode == 0:
            print("\n✅ System completed successfully")
        else:
            print(f"\n❌ System exited with code {process.returncode}")
            
    except KeyboardInterrupt:
        print("\n\n⚠️  System startup interrupted by user")
        if 'process' in locals():
            process.terminate()
            process.wait()
    except Exception as e:
        print(f"\n❌ Error starting system: {e}")

def print_startup_instructions():
    """Print startup instructions"""
    print("\n📚 Startup Instructions:")
    print("1. Ensure all dependencies are installed")
    print("2. Configure your Telegram bot token and user ID in config.py")
    print("3. Set your Pocket Option SSID in config.py")
    print("4. Choose your preferred system mode")
    print("5. The system will start and begin trading")
    print()
    print("📱 Telegram Commands:")
    print("   /start     - Start the bot")
    print("   /signal    - Get trading signal")
    print("   /status    - Check system status")
    print("   /help      - Show available commands")
    print()

def main():
    """Main startup function"""
    print_banner()
    
    # Check system requirements
    if not check_system_requirements():
        print("\n❌ System requirements not met. Please fix the issues above.")
        return 1
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Core dependencies missing. Please install them first.")
        print("   Run: pip3 install -r requirements_core.txt")
        return 1
    
    # Check configuration
    if not check_configuration():
        print("\n❌ Configuration issues found. Please fix them before starting.")
        return 1
    
    # Print startup instructions
    print_startup_instructions()
    
    # Get user mode
    mode = get_user_mode()
    
    if mode == "exit":
        print("\n👋 Goodbye!")
        return 0
    
    # Start the system
    start_system(mode)
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Startup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)