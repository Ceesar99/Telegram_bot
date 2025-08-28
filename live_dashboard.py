#!/usr/bin/env python3
"""
Real-time Trading System Dashboard
"""

import time
import os
import json
import glob
from datetime import datetime

def display_status():
    print("\033[2J\033[H")  # Clear screen
    print("🔴 LIVE" + " " * 50 + f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print("🎯 ULTIMATE TRADING SYSTEM - LIVE DASHBOARD")
    print("=" * 80)
    
    # Check for running processes
    import subprocess
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        active_processes = []
        for line in result.stdout.split('\n'):
            if 'python3' in line and any(keyword in line for keyword in 
                ['paper_trader', 'paper_trading', 'training_monitor']):
                active_processes.append(line.strip())
        
        print(f"🔄 ACTIVE PROCESSES: {len(active_processes)}")
        for proc in active_processes[:3]:  # Show first 3
            print(f"   • {proc[:60]}...")
    except:
        print("🔄 ACTIVE PROCESSES: Unable to check")
    
    print()
    
    # Check latest reports
    report_files = glob.glob('/workspace/logs/paper_trading_report_*.json')
    if report_files:
        latest_report = max(report_files, key=os.path.getmtime)
        try:
            with open(latest_report, 'r') as f:
                report = json.load(f)
            
            print("📊 LATEST PAPER TRADING SESSION")
            print("-" * 40)
            print(f"Session ID: {report.get('session_id', 'Unknown')}")
            print(f"Total Trades: {report.get('total_trades', 0)}")
            print(f"Win Rate: {report.get('win_rate', 0):.1f}%")
            print(f"PnL: ${report.get('total_pnl', 0):.2f}")
            print(f"ROI: {report.get('roi', 0):.2f}%")
        except:
            print("📊 LATEST PAPER TRADING SESSION: Unable to read")
    else:
        print("📊 LATEST PAPER TRADING SESSION: No reports found")
    
    print()
    
    # Check log files
    log_files = glob.glob('/workspace/logs/*.log')
    if log_files:
        print("📋 RECENT LOG ACTIVITY")
        print("-" * 30)
        
        latest_logs = sorted(log_files, key=os.path.getmtime, reverse=True)[:3]
        for log_file in latest_logs:
            mtime = datetime.fromtimestamp(os.path.getmtime(log_file))
            size = os.path.getsize(log_file)
            name = os.path.basename(log_file)
            print(f"{name[:25]:25} {size:>8,} bytes {mtime.strftime('%H:%M:%S')}")
    
    print()
    print("🎯 MONITORING STATUS: ACTIVE")
    print("Press Ctrl+C to exit...")

def main():
    try:
        while True:
            display_status()
            time.sleep(10)  # Update every 10 seconds
    except KeyboardInterrupt:
        print("\n\n🛑 Monitoring stopped")

if __name__ == "__main__":
    main()
