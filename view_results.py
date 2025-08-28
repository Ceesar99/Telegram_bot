#!/usr/bin/env python3
import json
import glob
import os
from datetime import datetime

def view_latest_results():
    # Find latest report
    reports = glob.glob('/workspace/logs/paper_trading_report_*.json')
    if not reports:
        print("No paper trading reports found")
        return
    
    latest_report = max(reports, key=os.path.getmtime)
    
    with open(latest_report, 'r') as f:
        data = json.load(f)
    
    print("🎉 LATEST PAPER TRADING RESULTS")
    print("=" * 40)
    print(f"📅 Session ID: {data.get('session_id', 'Unknown')}")
    print(f"⏰ Duration: {data.get('duration_days', 0)} days")
    print(f"📊 Total Trades: {data.get('total_trades', 0)}")
    print(f"✅ Winning Trades: {data.get('winning_trades', 0)}")
    print(f"❌ Losing Trades: {data.get('losing_trades', 0)}")
    print(f"🎯 Win Rate: {data.get('win_rate', 0):.1f}%")
    print(f"💰 Total PnL: ${data.get('total_pnl', 0):.2f}")
    print(f"💰 Final Balance: ${data.get('final_balance', 0):.2f}")
    print(f"📈 ROI: {data.get('roi', 0):.2f}%")
    
    # Show recent trades
    trades = data.get('trades', [])
    if trades:
        print(f"\n📋 RECENT TRADES (Last 5):")
        for trade in trades[-5:]:
            print(f"   Trade {trade['id']}: {trade['result']} {trade['direction']} "
                  f"${trade['profit']:.2f} (Confidence: {trade['confidence']:.1f}%)")

if __name__ == "__main__":
    view_latest_results()