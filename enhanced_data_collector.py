#!/usr/bin/env python3
"""
🌐 ENHANCED MARKET DATA COLLECTOR - PRODUCTION READY
Uses multiple premium APIs to collect comprehensive real market data

APIs Integrated:
- Alpha Vantage: Stocks, Forex, Crypto, Technical Indicators
- Finnhub: Real-time prices, company fundamentals, news
- Twelve Data: High-frequency data, multiple timeframes
- Polygon.io: Professional-grade data, options, derivatives
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import sqlite3
from dotenv import load_dotenv
import json
import asyncio
import aiohttp

# Load environment variables
load_dotenv('/workspace/.env')

class EnhancedMarketDataCollector:
    """Professional multi-API market data collector"""
    
    def __init__(self):
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.finnhub_key = os.getenv('FINNHUB_API_KEY')
        self.twelve_data_key = os.getenv('TWELVE_DATA_API_KEY')
        self.polygon_key = os.getenv('POLYGON_API_KEY')
        
        self.setup_logging()
        self.setup_database()
        
        # Rate limiting
        self.last_alpha_call = 0
        self.last_finnhub_call = 0
        self.last_twelve_call = 0
        self.last_polygon_call = 0
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_dir = "/workspace/logs"
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{log_dir}/enhanced_data_collector_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('EnhancedDataCollector')
        
    def setup_database(self):
        """Setup SQLite database for storing market data"""
        db_dir = "/workspace/data/database"
        os.makedirs(db_dir, exist_ok=True)
        
        self.db_path = f"{db_dir}/enhanced_market_data.db"
        self.conn = sqlite3.connect(self.db_path)
        
        # Create tables
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                datetime TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                source TEXT,
                timeframe TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS technical_indicators (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                datetime TEXT NOT NULL,
                indicator_name TEXT,
                value REAL,
                source TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.conn.commit()
        
    def respect_rate_limit(self, api_name: str):
        """Respect API rate limits"""
        current_time = time.time()
        
        if api_name == 'alpha_vantage':
            if current_time - self.last_alpha_call < 12:  # 5 calls/minute = 12 seconds between calls
                time.sleep(12 - (current_time - self.last_alpha_call))
            self.last_alpha_call = time.time()
            
        elif api_name == 'finnhub':
            if current_time - self.last_finnhub_call < 1:  # 60 calls/minute
                time.sleep(1 - (current_time - self.last_finnhub_call))
            self.last_finnhub_call = time.time()
            
        elif api_name == 'twelve_data':
            if current_time - self.last_twelve_call < 7.5:  # 8 calls/minute
                time.sleep(7.5 - (current_time - self.last_twelve_call))
            self.last_twelve_call = time.time()
            
        elif api_name == 'polygon':
            if current_time - self.last_polygon_call < 0.1:  # Professional tier
                time.sleep(0.1 - (current_time - self.last_polygon_call))
            self.last_polygon_call = time.time()
    
    def collect_alpha_vantage_data(self, symbol: str, timeframe: str = '1min') -> pd.DataFrame:
        """Collect data from Alpha Vantage API"""
        self.respect_rate_limit('alpha_vantage')
        
        url = f"https://www.alphavantage.co/query"
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': symbol,
            'interval': timeframe,
            'apikey': self.alpha_vantage_key,
            'outputsize': 'full'
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            if 'Time Series (1min)' in data:
                df = pd.DataFrame.from_dict(data['Time Series (1min)'], orient='index')
                df.columns = ['open', 'high', 'low', 'close', 'volume']
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                df.reset_index(inplace=True)
                df.rename(columns={'index': 'datetime'}, inplace=True)
                df['symbol'] = symbol
                df['source'] = 'alpha_vantage'
                df['timeframe'] = timeframe
                
                self.logger.info(f"Alpha Vantage: Collected {len(df)} records for {symbol}")
                return df
                
        except Exception as e:
            self.logger.error(f"Alpha Vantage error for {symbol}: {e}")
            
        return pd.DataFrame()
    
    def collect_finnhub_data(self, symbol: str) -> pd.DataFrame:
        """Collect data from Finnhub API"""
        self.respect_rate_limit('finnhub')
        
        # Get current timestamp and 30 days ago
        end_time = int(datetime.now().timestamp())
        start_time = int((datetime.now() - timedelta(days=30)).timestamp())
        
        url = f"https://finnhub.io/api/v1/stock/candle"
        params = {
            'symbol': symbol,
            'resolution': '1',  # 1 minute
            'from': start_time,
            'to': end_time,
            'token': self.finnhub_key
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            if data.get('s') == 'ok':
                df = pd.DataFrame({
                    'datetime': pd.to_datetime(data['t'], unit='s'),
                    'open': data['o'],
                    'high': data['h'],
                    'low': data['l'],
                    'close': data['c'],
                    'volume': data['v']
                })
                df['symbol'] = symbol
                df['source'] = 'finnhub'
                df['timeframe'] = '1min'
                
                self.logger.info(f"Finnhub: Collected {len(df)} records for {symbol}")
                return df
                
        except Exception as e:
            self.logger.error(f"Finnhub error for {symbol}: {e}")
            
        return pd.DataFrame()
    
    def collect_twelve_data(self, symbol: str, timeframe: str = '1min') -> pd.DataFrame:
        """Collect data from Twelve Data API"""
        self.respect_rate_limit('twelve_data')
        
        url = f"https://api.twelvedata.com/time_series"
        params = {
            'symbol': symbol,
            'interval': timeframe,
            'apikey': self.twelve_data_key,
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
                df['timeframe'] = timeframe
                
                # Convert string values to float
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                self.logger.info(f"Twelve Data: Collected {len(df)} records for {symbol}")
                return df
                
        except Exception as e:
            self.logger.error(f"Twelve Data error for {symbol}: {e}")
            
        return pd.DataFrame()
    
    def collect_polygon_data(self, symbol: str) -> pd.DataFrame:
        """Collect data from Polygon.io API"""
        self.respect_rate_limit('polygon')
        
        # Get date range
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/{start_date}/{end_date}"
        params = {
            'apikey': self.polygon_key,
            'limit': 50000
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            if data.get('results'):
                df = pd.DataFrame(data['results'])
                df['datetime'] = pd.to_datetime(df['t'], unit='ms')
                df = df.rename(columns={
                    'o': 'open',
                    'h': 'high',
                    'l': 'low',
                    'c': 'close',
                    'v': 'volume'
                })
                df['symbol'] = symbol
                df['source'] = 'polygon'
                df['timeframe'] = '1min'
                
                self.logger.info(f"Polygon.io: Collected {len(df)} records for {symbol}")
                return df
                
        except Exception as e:
            self.logger.error(f"Polygon.io error for {symbol}: {e}")
            
        return pd.DataFrame()
    
    def save_to_database(self, df: pd.DataFrame, table_name: str = 'market_data'):
        """Save data to SQLite database"""
        if not df.empty:
            df.to_sql(table_name, self.conn, if_exists='append', index=False)
            self.conn.commit()
            self.logger.info(f"Saved {len(df)} records to {table_name} table")
    
    def collect_comprehensive_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Collect comprehensive data from all APIs"""
        self.logger.info("🚀 Starting comprehensive data collection from all APIs...")
        
        all_data = {}
        
        for symbol in symbols:
            self.logger.info(f"📊 Collecting data for {symbol}...")
            
            # Collect from all sources
            alpha_data = self.collect_alpha_vantage_data(symbol)
            finnhub_data = self.collect_finnhub_data(symbol)
            twelve_data = self.collect_twelve_data(symbol)
            polygon_data = self.collect_polygon_data(symbol)
            
            # Combine all data
            combined_data = []
            for data, source in [(alpha_data, 'alpha_vantage'), (finnhub_data, 'finnhub'), 
                               (twelve_data, 'twelve_data'), (polygon_data, 'polygon')]:
                if not data.empty:
                    combined_data.append(data)
                    self.save_to_database(data)
            
            if combined_data:
                symbol_data = pd.concat(combined_data, ignore_index=True)
                all_data[symbol] = symbol_data
                
                # Save combined data to CSV
                output_dir = "/workspace/data/enhanced_real_data"
                os.makedirs(output_dir, exist_ok=True)
                symbol_data.to_csv(f"{output_dir}/{symbol}_enhanced_data.csv", index=False)
                
                self.logger.info(f"✅ {symbol}: Total {len(symbol_data)} records collected and saved")
            
            # Small delay between symbols
            time.sleep(2)
        
        self.logger.info("🎉 Comprehensive data collection completed!")
        return all_data

if __name__ == "__main__":
    collector = EnhancedMarketDataCollector()
    
    # Major trading symbols
    symbols = [
        'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD',  # Forex
        'AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA',           # Stocks
        'BTC-USD', 'ETH-USD', 'BTCUSD'                      # Crypto
    ]
    
    # Collect comprehensive data
    all_data = collector.collect_comprehensive_data(symbols)
    
    # Print summary
    total_records = sum(len(df) for df in all_data.values())
    print(f"\n🎉 ENHANCED DATA COLLECTION COMPLETE")
    print(f"📊 Total Records Collected: {total_records:,}")
    print(f"📈 Symbols Processed: {len(all_data)}")
    print(f"💾 Data saved to: /workspace/data/enhanced_real_data/")
    print(f"🗄️ Database: {collector.db_path}")