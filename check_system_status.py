#!/usr/bin/env python3
"""
🔍 ULTIMATE TRADING SYSTEM - STATUS CHECKER
Real-time System Monitoring and Health Check
"""

import os
import sys
import time
import psutil
import requests
from datetime import datetime
import json

def check_system_status():
    """Check the status of the Ultimate Trading System"""
    print("🔍 ULTIMATE TRADING SYSTEM - STATUS CHECK")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    # Check if Python processes are running
    python_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] in ['python', 'python3'] and proc.info['cmdline']:
                cmdline = ' '.join(proc.info['cmdline'])
                if 'ultimate' in cmdline.lower() or 'telegram' in cmdline.lower():
                    python_processes.append({
                        'pid': proc.info['pid'],
                        'command': cmdline
                    })
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    print(f"🤖 Python Processes Found: {len(python_processes)}")
    for proc in python_processes:
        print(f"   PID {proc['pid']}: {proc['command'][:80]}...")
    
    # Check log files
    log_dir = "/workspace/logs"
    if os.path.exists(log_dir):
        log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
        print(f"📋 Log Files: {len(log_files)}")
        
        # Check recent log activity
        recent_logs = []
        for log_file in log_files:
            log_path = os.path.join(log_dir, log_file)
            try:
                stat = os.stat(log_path)
                if time.time() - stat.st_mtime < 300:  # Modified in last 5 minutes
                    recent_logs.append(log_file)
            except:
                pass
        
        print(f"📊 Recently Active Logs: {len(recent_logs)}")
        for log in recent_logs:
            print(f"   ✅ {log}")
    
    # System Resources
    memory = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=1)
    disk = psutil.disk_usage('/workspace')
    
    print(f"💾 Memory Usage: {memory.percent:.1f}% ({memory.used//1024//1024} MB used)")
    print(f"⚡ CPU Usage: {cpu:.1f}%")
    print(f"💿 Disk Usage: {disk.percent:.1f}% ({disk.free//1024//1024//1024} GB free)")
    
    # Check configuration
    config_file = "/workspace/config.py"
    if os.path.exists(config_file):
        print("⚙️ Configuration: ✅ FOUND")
    else:
        print("⚙️ Configuration: ❌ MISSING")
    
    # Check key files
    key_files = [
        "ultimate_telegram_bot.py",
        "ultimate_trading_system.py", 
        "ultimate_universal_launcher.py"
    ]
    
    print("📁 Key System Files:")
    for file in key_files:
        file_path = f"/workspace/{file}"
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"   ✅ {file} ({size//1024} KB)")
        else:
            print(f"   ❌ {file} - MISSING")
    
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    if python_processes:
        print("🏆 ULTIMATE TRADING SYSTEM STATUS: OPERATIONAL")
        print("🚀 System is running and ready to process commands!")
    else:
        print("⚠️ ULTIMATE TRADING SYSTEM STATUS: NOT RUNNING")
        print("💡 Use: python3 ultimate_telegram_bot.py to start")
    
    print(f"🕐 Status Check Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    check_system_status()