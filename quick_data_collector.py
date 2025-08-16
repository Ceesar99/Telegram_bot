#!/usr/bin/env python3
"""
🚀 QUICK REAL MARKET DATA COLLECTOR - PRODUCTION READY
Fast collection of real market data for AI/ML model training
"""

import os
import pandas as pd
import yfinance as yf
import requests
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv('/workspace/.env')

def collect_yfinance_data(symbols, period="30d", interval="1m"):
    """Collect data using Yahoo Finance (always works)"""
    print("🚀 Collecting real market data using Yahoo Finance...")
    
    all_data = []
    
    for symbol in symbols:
        try:
            print(f"📊 Downloading {symbol}...")
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if not data.empty:
                data.reset_index(inplace=True)
                data['symbol'] = symbol
                data['source'] = 'yahoo_finance'
                data.columns = [col.lower() for col in data.columns]
                all_data.append(data)
                print(f"✅ {symbol}: {len(data)} records collected")
            else:
                print(f"❌ {symbol}: No data available")
                
        except Exception as e:
            print(f"❌ {symbol}: Error - {e}")
        
        time.sleep(0.5)  # Rate limiting
    
    return all_data

def collect_twelve_data_simple(symbol, api_key):
    """Simple Twelve Data collection"""
    url = "https://api.twelvedata.com/time_series"
    params = {
        'symbol': symbol,
        'interval': '1min',
        'apikey': api_key,
        'outputsize': 5000
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if 'values' in data:
            df = pd.DataFrame(data['values'])
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime')
            df['symbol'] = symbol
            df['source'] = 'twelve_data'
            
            # Convert to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
    except:
        pass
    
    return pd.DataFrame()

if __name__ == "__main__":
    # Major trading symbols
    symbols = [
        'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCAD=X',  # Forex
        'AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA',                      # Stocks
        'BTC-USD', 'ETH-USD'                                           # Crypto
    ]
    
    # Collect data
    all_datasets = collect_yfinance_data(symbols)
    
    # Try Twelve Data for some additional symbols
    twelve_key = os.getenv('TWELVE_DATA_API_KEY')
    if twelve_key:
        print("\n🌐 Collecting additional data from Twelve Data...")
        for symbol in ['AAPL', 'EURUSD', 'BTCUSD']:
            twelve_data = collect_twelve_data_simple(symbol, twelve_key)
            if not twelve_data.empty:
                all_datasets.append(twelve_data)
                print(f"✅ Twelve Data {symbol}: {len(twelve_data)} records")
            time.sleep(8)  # Rate limiting
    
    # Combine all data
    if all_datasets:
        combined_data = pd.concat(all_datasets, ignore_index=True)
        
        # Save to directory
        output_dir = "/workspace/data/real_training_data"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save combined data
        output_file = f"{output_dir}/real_market_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        combined_data.to_csv(output_file, index=False)
        
        # Save individual symbol files
        for symbol in combined_data['symbol'].unique():
            symbol_data = combined_data[combined_data['symbol'] == symbol]
            symbol_file = f"{output_dir}/{symbol.replace('=X', '').replace('-', '_')}_data.csv"
            symbol_data.to_csv(symbol_file, index=False)
        
        print(f"\n🎉 DATA COLLECTION COMPLETE!")
        print(f"📊 Total Records: {len(combined_data):,}")
        print(f"📈 Symbols: {len(combined_data['symbol'].unique())}")
        print(f"💾 Saved to: {output_file}")
        print(f"📁 Individual files: {output_dir}/")
        
        # Print summary by symbol
        print("\n📊 DATA SUMMARY BY SYMBOL:")
        summary = combined_data.groupby('symbol').size().sort_values(ascending=False)
        for symbol, count in summary.items():
            print(f"  {symbol}: {count:,} records")
    
    else:
        print("❌ No data collected!")