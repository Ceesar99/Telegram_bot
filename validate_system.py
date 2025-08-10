#!/usr/bin/env python3
"""
Trading Bot System Validation Script

This script performs comprehensive validation of all system components
to ensure everything is properly configured and ready to run.
"""

import os
import sys
import importlib
from pathlib import Path

def print_header():
    """Print validation header"""
    print("="*70)
    print("🔍 BINARY OPTIONS TRADING BOT - SYSTEM VALIDATION")
    print("="*70)
    print()

def print_section(title):
    """Print section header"""
    print(f"\n📋 {title}")
    print("-" * (len(title) + 4))

def check_python_version():
    """Check Python version compatibility"""
    print_section("Python Version Check")
    
    if sys.version_info >= (3, 8):
        print(f"✅ Python {sys.version.split()[0]} - Compatible")
        return True
    else:
        print(f"❌ Python {sys.version.split()[0]} - Requires 3.8+")
        return False

def check_directories():
    """Check if all required directories exist"""
    print_section("Directory Structure")
    
    required_dirs = [
        '/workspace/logs',
        '/workspace/data',
        '/workspace/models',
        '/workspace/backup'
    ]
    
    all_good = True
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✅ {directory}")
        else:
            print(f"❌ {directory} - Missing")
            all_good = False
    
    return all_good

def check_core_files():
    """Check if all core files exist"""
    print_section("Core Files")
    
    required_files = [
        'config.py',
        'main.py',
        'start_bot.py',
        'telegram_bot.py',
        'signal_engine.py',
        'lstm_model.py',
        'pocket_option_api.py',
        'risk_manager.py',
        'performance_tracker.py',
        'requirements.txt',
        'README.md'
    ]
    
    all_good = True
    for file in required_files:
        if os.path.exists(f'/workspace/{file}'):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - Missing")
            all_good = False
    
    return all_good

def check_dependencies():
    """Check if key dependencies can be imported"""
    print_section("Key Dependencies")
    
    dependencies = [
        ('pandas', 'Data manipulation'),
        ('numpy', 'Numerical computing'),
        ('tensorflow', 'Machine learning'),
        ('requests', 'HTTP requests'),
        ('websocket', 'WebSocket client'),
        ('matplotlib', 'Plotting'),
        ('sqlite3', 'Database'),
        ('json', 'JSON handling'),
        ('datetime', 'Date/time operations'),
        ('asyncio', 'Async operations'),
        ('logging', 'Logging system')
    ]
    
    missing = []
    for module, description in dependencies:
        try:
            importlib.import_module(module)
            print(f"✅ {module} - {description}")
        except ImportError:
            print(f"❌ {module} - {description} (MISSING)")
            missing.append(module)
    
    return missing

def check_configuration():
    """Check configuration settings"""
    print_section("Configuration Validation")
    
    try:
        # Try to import config
        config_path = '/workspace/config.py'
        if os.path.exists(config_path):
            print("✅ Config file exists")
            
            # Read config content
            with open(config_path, 'r') as f:
                content = f.read()
            
            # Check for required settings
            required_settings = [
                ('TELEGRAM_BOT_TOKEN', 'Telegram bot token'),
                ('TELEGRAM_USER_ID', 'Telegram user ID'),
                ('POCKET_OPTION_SSID', 'Pocket Option session'),
                ('CURRENCY_PAIRS', 'Currency pairs list'),
                ('SIGNAL_CONFIG', 'Signal configuration'),
                ('RISK_MANAGEMENT', 'Risk management settings')
            ]
            
            for setting, description in required_settings:
                if setting in content:
                    print(f"✅ {setting} - {description}")
                else:
                    print(f"❌ {setting} - {description} (Missing)")
            
            return True
        else:
            print("❌ Config file missing")
            return False
            
    except Exception as e:
        print(f"❌ Config validation error: {e}")
        return False

def check_telegram_config():
    """Check Telegram configuration"""
    print_section("Telegram Configuration")
    
    try:
        with open('/workspace/config.py', 'r') as f:
            content = f.read()
        
        # Check bot token format
        if 'TELEGRAM_BOT_TOKEN = "8226952507:AAG' in content:
            print("✅ Telegram Bot Token - Configured")
        else:
            print("⚠️  Telegram Bot Token - Format may be incorrect")
        
        # Check user ID
        if 'TELEGRAM_USER_ID = "8093708320"' in content:
            print("✅ Telegram User ID - Configured")
        else:
            print("⚠️  Telegram User ID - May need updating")
        
        return True
        
    except Exception as e:
        print(f"❌ Telegram config check failed: {e}")
        return False

def check_pocket_option_config():
    """Check Pocket Option configuration"""
    print_section("Pocket Option Configuration")
    
    try:
        with open('/workspace/config.py', 'r') as f:
            content = f.read()
        
        if 'POCKET_OPTION_SSID = ' in content and len(content.split('POCKET_OPTION_SSID = ')[1].split('\n')[0]) > 20:
            print("✅ Pocket Option SSID - Configured")
        else:
            print("❌ Pocket Option SSID - Not properly configured")
        
        return True
        
    except Exception as e:
        print(f"❌ Pocket Option config check failed: {e}")
        return False

def check_ai_model_config():
    """Check AI model configuration"""
    print_section("AI Model Configuration")
    
    try:
        with open('/workspace/config.py', 'r') as f:
            content = f.read()
        
        ai_configs = [
            ('LSTM_CONFIG', 'LSTM neural network settings'),
            ('TECHNICAL_INDICATORS', 'Technical analysis indicators'),
            ('SIGNAL_CONFIG', 'Signal generation parameters')
        ]
        
        for config, description in ai_configs:
            if config in content:
                print(f"✅ {config} - {description}")
            else:
                print(f"❌ {config} - {description} (Missing)")
        
        return True
        
    except Exception as e:
        print(f"❌ AI model config check failed: {e}")
        return False

def generate_status_report(checks):
    """Generate final status report"""
    print_section("SYSTEM STATUS REPORT")
    
    total_checks = len(checks)
    passed_checks = sum(checks.values())
    
    print(f"Total Checks: {total_checks}")
    print(f"Passed: {passed_checks}")
    print(f"Failed: {total_checks - passed_checks}")
    print(f"Success Rate: {(passed_checks/total_checks)*100:.1f}%")
    
    if passed_checks == total_checks:
        print("\n🎉 ALL CHECKS PASSED! System is ready to run.")
        print("\n🚀 To start the bot:")
        print("   python3 start_bot.py")
        return True
    else:
        print(f"\n⚠️  {total_checks - passed_checks} checks failed. See details above.")
        print("\n🔧 To install missing dependencies:")
        print("   pip3 install -r requirements.txt")
        return False

def check_telegram_commands():
    """Check if Telegram bot commands are properly defined"""
    print_section("Telegram Commands Validation")
    
    try:
        with open('/workspace/telegram_bot.py', 'r') as f:
            content = f.read()
        
        expected_commands = [
            'start', 'signal', 'auto_on', 'auto_off', 'pairs',
            'market_status', 'analyze', 'stats', 'status',
            'settings', 'help', 'performance'
        ]
        
        defined_commands = []
        for cmd in expected_commands:
            if f'def {cmd}(' in content or f'CommandHandler("{cmd}"' in content:
                print(f"✅ /{cmd}")
                defined_commands.append(cmd)
            else:
                print(f"❌ /{cmd} - Not found")
        
        print(f"\nTotal commands defined: {len(defined_commands)}/{len(expected_commands)}")
        return len(defined_commands) >= len(expected_commands) * 0.8  # At least 80% commands
        
    except Exception as e:
        print(f"❌ Command validation failed: {e}")
        return False

def main():
    """Main validation routine"""
    print_header()
    
    # Run all checks
    checks = {
        'Python Version': check_python_version(),
        'Directories': check_directories(),
        'Core Files': check_core_files(),
        'Configuration': check_configuration(),
        'Telegram Config': check_telegram_config(),
        'Pocket Option Config': check_pocket_option_config(),
        'AI Model Config': check_ai_model_config(),
        'Telegram Commands': check_telegram_commands()
    }
    
    # Check dependencies (special handling)
    missing_deps = check_dependencies()
    checks['Dependencies'] = len(missing_deps) == 0
    
    # Generate final report
    system_ready = generate_status_report(checks)
    
    # Additional recommendations
    print_section("RECOMMENDATIONS")
    
    if missing_deps:
        print(f"📦 Install missing dependencies: pip3 install {' '.join(missing_deps)}")
    
    if not system_ready:
        print("🔧 Fix the issues above before starting the bot")
    
    print("📚 Read README.md for detailed usage instructions")
    print("📱 Test the bot with /start command after launch")
    print("🎯 Use /signal to get your first trading signal")
    
    print("\n" + "="*70)
    
    return system_ready

if __name__ == "__main__":
    main()