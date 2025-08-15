#!/usr/bin/env python3
"""
📊 REAL MARKET DATA COLLECTOR - PRODUCTION READY
Comprehensive historical and real-time data collection for AI model training
Supports multiple timeframes, data validation, and storage optimization
"""

import asyncio
import aiohttp
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import sqlite3
import h5py
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import talib
import warnings
warnings.filterwarnings('ignore')

from config import CURRENCY_PAIRS, OTC_PAIRS, DATABASE_CONFIG, TIMEZONE
from redundant_data_manager import RedundantDataManager, MarketDataPoint

@dataclass
class HistoricalDataRequest:
    """Historical data request specification"""
    symbol: str
    timeframe: str  # '1m', '5m', '15m', '1h', '4h', '1d'
    start_date: datetime
    end_date: datetime
    source: str = 'auto'
    priority: int = 1

@dataclass
class DataCollectionStats:
    """Data collection statistics"""
    total_symbols: int = 0
    completed_symbols: int = 0
    failed_symbols: int = 0
    total_records: int = 0
    data_quality_score: float = 0.0
    completion_percentage: float = 0.0
    avg_records_per_symbol: float = 0.0

class HistoricalDataProvider:
    """Base class for historical data providers"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f'DataProvider_{name}')
        self.rate_limit_delay = 1.0  # seconds between requests
        self.last_request_time = 0
        
    async def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
        
    async def get_historical_data(self, request: HistoricalDataRequest) -> Optional[pd.DataFrame]:
        """Get historical data - to be implemented by subclasses"""
        raise NotImplementedError

class YahooFinanceHistoricalProvider(HistoricalDataProvider):
    """Yahoo Finance historical data provider"""
    
    def __init__(self):
        super().__init__('yahoo_historical')
        self.rate_limit_delay = 0.5  # Yahoo Finance is generous with rate limits
        
    async def get_historical_data(self, request: HistoricalDataRequest) -> Optional[pd.DataFrame]:
        """Get historical data from Yahoo Finance"""
        try:
            await self._rate_limit()
            
            # Convert symbol format for Yahoo Finance
            if '/' in request.symbol:
                yf_symbol = request.symbol.replace('/', '') + '=X'
            else:
                yf_symbol = request.symbol
            
            # Convert timeframe
            interval_map = {
                '1m': '1m',
                '5m': '5m',
                '15m': '15m',
                '1h': '1h',
                '4h': '4h',
                '1d': '1d'
            }
            
            interval = interval_map.get(request.timeframe, '1h')
            
            # Yahoo Finance limits for minute data
            if interval in ['1m', '5m', '15m']:
                # Limit to last 7 days for minute data
                max_days = 7
                if (request.end_date - request.start_date).days > max_days:
                    self.logger.warning(f"Limiting {request.symbol} {interval} data to last {max_days} days")
                    request.start_date = request.end_date - timedelta(days=max_days)
            
            ticker = yf.Ticker(yf_symbol)
            
            # Get data
            data = ticker.history(
                start=request.start_date,
                end=request.end_date,
                interval=interval,
                auto_adjust=True,
                prepost=False
            )
            
            if data.empty:
                self.logger.warning(f"No data returned for {request.symbol}")
                return None
            
            # Standardize column names
            data.columns = [col.lower() for col in data.columns]
            
            # Add symbol column
            data['symbol'] = request.symbol
            data['timeframe'] = request.timeframe
            data['source'] = self.name
            
            # Reset index to make timestamp a column
            data.reset_index(inplace=True)
            if 'date' in data.columns:
                data.rename(columns={'date': 'timestamp'}, inplace=True)
            elif 'datetime' in data.columns:
                data.rename(columns={'datetime': 'timestamp'}, inplace=True)
            
            self.logger.info(f"Retrieved {len(data)} records for {request.symbol} {request.timeframe}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error getting historical data for {request.symbol}: {e}")
            return None

class AlphaVantageHistoricalProvider(HistoricalDataProvider):
    """Alpha Vantage historical data provider"""
    
    def __init__(self, api_key: str):
        super().__init__('alphavantage_historical')
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.rate_limit_delay = 12.0  # 5 requests per minute
        
    async def get_historical_data(self, request: HistoricalDataRequest) -> Optional[pd.DataFrame]:
        """Get historical data from Alpha Vantage"""
        try:
            await self._rate_limit()
            
            # Convert symbol format
            if '/' in request.symbol:
                from_currency = request.symbol[:3]
                to_currency = request.symbol[4:7]
            else:
                self.logger.warning(f"Invalid symbol format for Alpha Vantage: {request.symbol}")
                return None
            
            # Determine function based on timeframe
            function_map = {
                '1m': 'FX_INTRADAY',
                '5m': 'FX_INTRADAY',
                '15m': 'FX_INTRADAY',
                '1h': 'FX_INTRADAY',
                '1d': 'FX_DAILY'
            }
            
            function = function_map.get(request.timeframe, 'FX_DAILY')
            
            params = {
                'function': function,
                'from_symbol': from_currency,
                'to_symbol': to_currency,
                'apikey': self.api_key,
                'outputsize': 'full'
            }
            
            if function == 'FX_INTRADAY':
                params['interval'] = request.timeframe
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status != 200:
                        self.logger.error(f"Alpha Vantage API error: {response.status}")
                        return None
                    
                    data = await response.json()
                    
                    # Find the time series data key
                    time_series_key = None
                    for key in data.keys():
                        if 'Time Series' in key:
                            time_series_key = key
                            break
                    
                    if not time_series_key:
                        self.logger.error(f"No time series data in response for {request.symbol}")
                        return None
                    
                    time_series = data[time_series_key]
                    
                    # Convert to DataFrame
                    df_data = []
                    for timestamp, values in time_series.items():
                        row = {
                            'timestamp': pd.to_datetime(timestamp),
                            'open': float(values.get('1. open', 0)),
                            'high': float(values.get('2. high', 0)),
                            'low': float(values.get('3. low', 0)),
                            'close': float(values.get('4. close', 0)),
                            'volume': 0,  # FX data doesn't have volume
                            'symbol': request.symbol,
                            'timeframe': request.timeframe,
                            'source': self.name
                        }
                        df_data.append(row)
                    
                    if not df_data:
                        return None
                    
                    df = pd.DataFrame(df_data)
                    df = df.sort_values('timestamp')
                    
                    # Filter by date range
                    df = df[
                        (df['timestamp'] >= request.start_date) & 
                        (df['timestamp'] <= request.end_date)
                    ]
                    
                    self.logger.info(f"Retrieved {len(df)} records for {request.symbol} {request.timeframe}")
                    return df
                    
        except Exception as e:
            self.logger.error(f"Alpha Vantage error for {request.symbol}: {e}")
            return None

class RealMarketDataCollector:
    """Comprehensive real market data collection system"""
    
    def __init__(self, storage_path: str = "/workspace/data"):
        self.storage_path = storage_path
        self.logger = logging.getLogger('RealMarketDataCollector')
        self.providers = {}
        self.collection_stats = DataCollectionStats()
        
        # Ensure storage directory exists
        os.makedirs(storage_path, exist_ok=True)
        
        # Initialize database
        self._initialize_database()
        
        # Setup data providers
        self._setup_providers()
        
        # Initialize data validator
        self.redundant_manager = RedundantDataManager()
        
    def _initialize_database(self):
        """Initialize data storage database"""
        try:
            db_path = os.path.join(self.storage_path, 'market_data.db')
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Historical data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS historical_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL,
                    source TEXT NOT NULL,
                    quality_score REAL DEFAULT 1.0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timeframe, timestamp)
                )
            ''')
            
            # Collection metadata table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS collection_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    total_records INTEGER,
                    data_quality REAL,
                    collection_date TEXT DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'completed',
                    UNIQUE(symbol, timeframe, start_date, end_date)
                )
            ''')
            
            # Create indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_historical_symbol_timeframe ON historical_data(symbol, timeframe)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_historical_timestamp ON historical_data(timestamp)')
            
            conn.commit()
            conn.close()
            
            self.logger.info("Database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Database initialization error: {e}")
    
    def _setup_providers(self):
        """Setup historical data providers"""
        try:
            # Yahoo Finance (primary)
            self.providers['yahoo'] = YahooFinanceHistoricalProvider()
            
            # Alpha Vantage (if API key available)
            alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY', 'demo')
            if alpha_vantage_key and alpha_vantage_key != 'demo':
                self.providers['alphavantage'] = AlphaVantageHistoricalProvider(alpha_vantage_key)
            
            self.logger.info(f"Initialized {len(self.providers)} data providers")
            
        except Exception as e:
            self.logger.error(f"Provider setup error: {e}")
    
    async def collect_historical_data(self, 
                                    symbols: List[str] = None,
                                    timeframes: List[str] = None,
                                    start_date: datetime = None,
                                    end_date: datetime = None,
                                    max_concurrent: int = 5) -> DataCollectionStats:
        """Collect comprehensive historical data"""
        
        # Set defaults
        if symbols is None:
            symbols = CURRENCY_PAIRS[:10]  # Start with major pairs
        
        if timeframes is None:
            timeframes = ['1h', '4h', '1d']  # Focus on reliable timeframes
        
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365*2)  # 2 years
        
        if end_date is None:
            end_date = datetime.now()
        
        # Create collection requests
        requests = []
        for symbol in symbols:
            for timeframe in timeframes:
                request = HistoricalDataRequest(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date
                )
                requests.append(request)
        
        self.collection_stats.total_symbols = len(requests)
        
        self.logger.info(f"Starting data collection for {len(requests)} requests")
        
        # Process requests with concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        tasks = [self._collect_single_request(request, semaphore) for request in requests]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Collection failed for {requests[i].symbol}: {result}")
                self.collection_stats.failed_symbols += 1
            elif result:
                self.collection_stats.completed_symbols += 1
                self.collection_stats.total_records += len(result)
            else:
                self.collection_stats.failed_symbols += 1
        
        # Calculate final statistics
        if self.collection_stats.total_symbols > 0:
            self.collection_stats.completion_percentage = (
                self.collection_stats.completed_symbols / self.collection_stats.total_symbols * 100
            )
        
        if self.collection_stats.completed_symbols > 0:
            self.collection_stats.avg_records_per_symbol = (
                self.collection_stats.total_records / self.collection_stats.completed_symbols
            )
        
        self.logger.info(f"Data collection completed: {self.collection_stats}")
        return self.collection_stats
    
    async def _collect_single_request(self, request: HistoricalDataRequest, semaphore: asyncio.Semaphore) -> Optional[pd.DataFrame]:
        """Collect data for a single request"""
        async with semaphore:
            try:
                # Check if data already exists
                if self._data_exists(request):
                    self.logger.info(f"Data already exists for {request.symbol} {request.timeframe}")
                    return await self._load_existing_data(request)
                
                # Try each provider in order
                for provider_name, provider in self.providers.items():
                    try:
                        self.logger.info(f"Collecting {request.symbol} {request.timeframe} from {provider_name}")
                        
                        data = await provider.get_historical_data(request)
                        
                        if data is not None and not data.empty:
                            # Validate and store data
                            validated_data = self._validate_historical_data(data)
                            if validated_data is not None:
                                await self._store_data(validated_data, request)
                                return validated_data
                        
                    except Exception as e:
                        self.logger.error(f"Error with provider {provider_name} for {request.symbol}: {e}")
                        continue
                
                self.logger.error(f"All providers failed for {request.symbol} {request.timeframe}")
                return None
                
            except Exception as e:
                self.logger.error(f"Collection error for {request.symbol}: {e}")
                return None
    
    def _data_exists(self, request: HistoricalDataRequest) -> bool:
        """Check if data already exists for the request"""
        try:
            db_path = os.path.join(self.storage_path, 'market_data.db')
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT COUNT(*) FROM collection_metadata 
                WHERE symbol = ? AND timeframe = ? AND start_date <= ? AND end_date >= ?
            ''', (
                request.symbol,
                request.timeframe,
                request.start_date.isoformat(),
                request.end_date.isoformat()
            ))
            
            count = cursor.fetchone()[0]
            conn.close()
            
            return count > 0
            
        except Exception as e:
            self.logger.error(f"Error checking data existence: {e}")
            return False
    
    async def _load_existing_data(self, request: HistoricalDataRequest) -> Optional[pd.DataFrame]:
        """Load existing data from database"""
        try:
            db_path = os.path.join(self.storage_path, 'market_data.db')
            conn = sqlite3.connect(db_path)
            
            query = '''
                SELECT timestamp, open, high, low, close, volume, symbol, timeframe, source
                FROM historical_data 
                WHERE symbol = ? AND timeframe = ? AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            '''
            
            df = pd.read_sql_query(query, conn, params=(
                request.symbol,
                request.timeframe,
                request.start_date.isoformat(),
                request.end_date.isoformat()
            ))
            
            conn.close()
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error loading existing data: {e}")
            return None
    
    def _validate_historical_data(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Validate historical data quality"""
        try:
            if data.empty:
                return None
            
            # Basic validation
            required_columns = ['timestamp', 'open', 'high', 'low', 'close']
            if not all(col in data.columns for col in required_columns):
                self.logger.error("Missing required columns in data")
                return None
            
            # Remove rows with invalid data
            original_len = len(data)
            
            # Remove rows where high < low
            data = data[data['high'] >= data['low']]
            
            # Remove rows with zero or negative prices
            for col in ['open', 'high', 'low', 'close']:
                data = data[data[col] > 0]
            
            # Remove rows where close is outside OHLC range
            data = data[
                (data['close'] >= data['low']) & 
                (data['close'] <= data['high'])
            ]
            
            # Remove duplicates
            data = data.drop_duplicates(subset=['timestamp'])
            
            # Sort by timestamp
            data = data.sort_values('timestamp')
            
            # Calculate data quality score
            quality_score = len(data) / original_len if original_len > 0 else 0
            data['quality_score'] = quality_score
            
            removed_count = original_len - len(data)
            if removed_count > 0:
                self.logger.warning(f"Removed {removed_count} invalid records ({quality_score:.2%} quality)")
            
            if quality_score < 0.5:
                self.logger.error(f"Data quality too low: {quality_score:.2%}")
                return None
            
            return data
            
        except Exception as e:
            self.logger.error(f"Data validation error: {e}")
            return None
    
    async def _store_data(self, data: pd.DataFrame, request: HistoricalDataRequest):
        """Store validated data to database"""
        try:
            db_path = os.path.join(self.storage_path, 'market_data.db')
            conn = sqlite3.connect(db_path)
            
            # Store historical data
            data_to_store = data.copy()
            data_to_store['timestamp'] = data_to_store['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            data_to_store.to_sql('historical_data', conn, if_exists='append', index=False)
            
            # Store collection metadata
            metadata = {
                'symbol': request.symbol,
                'timeframe': request.timeframe,
                'start_date': request.start_date.isoformat(),
                'end_date': request.end_date.isoformat(),
                'total_records': len(data),
                'data_quality': data['quality_score'].iloc[0] if not data.empty else 0,
                'status': 'completed'
            }
            
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO collection_metadata 
                (symbol, timeframe, start_date, end_date, total_records, data_quality, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                metadata['symbol'],
                metadata['timeframe'], 
                metadata['start_date'],
                metadata['end_date'],
                metadata['total_records'],
                metadata['data_quality'],
                metadata['status']
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Stored {len(data)} records for {request.symbol} {request.timeframe}")
            
        except Exception as e:
            self.logger.error(f"Data storage error: {e}")
    
    def get_collected_data(self, 
                          symbol: str, 
                          timeframe: str,
                          start_date: datetime = None,
                          end_date: datetime = None) -> Optional[pd.DataFrame]:
        """Retrieve collected data"""
        try:
            db_path = os.path.join(self.storage_path, 'market_data.db')
            conn = sqlite3.connect(db_path)
            
            query = '''
                SELECT timestamp, open, high, low, close, volume
                FROM historical_data 
                WHERE symbol = ? AND timeframe = ?
            '''
            params = [symbol, timeframe]
            
            if start_date:
                query += ' AND timestamp >= ?'
                params.append(start_date.isoformat())
            
            if end_date:
                query += ' AND timestamp <= ?'
                params.append(end_date.isoformat())
            
            query += ' ORDER BY timestamp'
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            
            return df if not df.empty else None
            
        except Exception as e:
            self.logger.error(f"Error retrieving data: {e}")
            return None
    
    def export_to_hdf5(self, output_path: str = None):
        """Export collected data to HDF5 format for ML training"""
        try:
            if output_path is None:
                output_path = os.path.join(self.storage_path, 'training_data.h5')
            
            db_path = os.path.join(self.storage_path, 'market_data.db')
            conn = sqlite3.connect(db_path)
            
            # Get all unique symbol-timeframe combinations
            cursor = conn.cursor()
            cursor.execute('''
                SELECT DISTINCT symbol, timeframe FROM historical_data
                ORDER BY symbol, timeframe
            ''')
            combinations = cursor.fetchall()
            
            with h5py.File(output_path, 'w') as h5file:
                for symbol, timeframe in combinations:
                    # Get data
                    df = pd.read_sql_query('''
                        SELECT timestamp, open, high, low, close, volume
                        FROM historical_data 
                        WHERE symbol = ? AND timeframe = ?
                        ORDER BY timestamp
                    ''', conn, params=(symbol, timeframe))
                    
                    if not df.empty:
                        # Create group
                        group_name = f"{symbol.replace('/', '_')}_{timeframe}"
                        group = h5file.create_group(group_name)
                        
                        # Store data
                        group.create_dataset('timestamp', data=df['timestamp'].astype(str))
                        group.create_dataset('open', data=df['open'].values)
                        group.create_dataset('high', data=df['high'].values)
                        group.create_dataset('low', data=df['low'].values)
                        group.create_dataset('close', data=df['close'].values)
                        group.create_dataset('volume', data=df['volume'].values)
                        
                        # Store metadata
                        group.attrs['symbol'] = symbol
                        group.attrs['timeframe'] = timeframe
                        group.attrs['records'] = len(df)
            
            conn.close()
            self.logger.info(f"Data exported to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Export error: {e}")
    
    def get_collection_summary(self) -> Dict[str, Any]:
        """Get collection summary statistics"""
        try:
            db_path = os.path.join(self.storage_path, 'market_data.db')
            conn = sqlite3.connect(db_path)
            
            # Get overall statistics
            cursor = conn.cursor()
            cursor.execute('''
                SELECT 
                    COUNT(DISTINCT symbol) as unique_symbols,
                    COUNT(DISTINCT timeframe) as unique_timeframes,
                    COUNT(*) as total_records,
                    MIN(timestamp) as earliest_date,
                    MAX(timestamp) as latest_date,
                    AVG(quality_score) as avg_quality
                FROM historical_data
            ''')
            
            stats = cursor.fetchone()
            
            # Get per-symbol statistics
            cursor.execute('''
                SELECT symbol, timeframe, COUNT(*) as records, AVG(quality_score) as quality
                FROM historical_data
                GROUP BY symbol, timeframe
                ORDER BY symbol, timeframe
            ''')
            
            symbol_stats = cursor.fetchall()
            conn.close()
            
            return {
                'overall': {
                    'unique_symbols': stats[0],
                    'unique_timeframes': stats[1],
                    'total_records': stats[2],
                    'earliest_date': stats[3],
                    'latest_date': stats[4],
                    'avg_quality': stats[5]
                },
                'by_symbol': [
                    {
                        'symbol': row[0],
                        'timeframe': row[1],
                        'records': row[2],
                        'quality': row[3]
                    }
                    for row in symbol_stats
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting collection summary: {e}")
            return {}

# Example usage and testing
async def test_data_collector():
    """Test the real market data collector"""
    collector = RealMarketDataCollector()
    
    # Test with a small subset
    test_symbols = ["EUR/USD", "GBP/USD"]
    test_timeframes = ["1h", "1d"]
    
    print("Starting data collection test...")
    
    stats = await collector.collect_historical_data(
        symbols=test_symbols,
        timeframes=test_timeframes,
        start_date=datetime.now() - timedelta(days=30),
        end_date=datetime.now()
    )
    
    print(f"Collection completed:")
    print(f"  Total symbols: {stats.total_symbols}")
    print(f"  Completed: {stats.completed_symbols}")
    print(f"  Failed: {stats.failed_symbols}")
    print(f"  Total records: {stats.total_records}")
    print(f"  Completion: {stats.completion_percentage:.1f}%")
    
    # Test data retrieval
    print("\nTesting data retrieval...")
    data = collector.get_collected_data("EUR/USD", "1h")
    if data is not None:
        print(f"Retrieved {len(data)} records for EUR/USD 1h")
        print(data.head())
    
    # Export to HDF5
    print("\nExporting to HDF5...")
    collector.export_to_hdf5()
    
    # Get summary
    print("\nCollection summary:")
    summary = collector.get_collection_summary()
    print(json.dumps(summary, indent=2, default=str))

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(test_data_collector())