#!/usr/bin/env python3
"""
🌐 REDUNDANT DATA MANAGER - PRODUCTION READY
Multi-source data feeds with automatic failover and quality validation
Implements institutional-grade data reliability for live trading
"""

import asyncio
import aiohttp
import yfinance as yf
import requests
import websocket
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
import queue
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import warnings
warnings.filterwarnings('ignore')

from config import CURRENCY_PAIRS, DATABASE_CONFIG, TIMEZONE

class DataSourceStatus(Enum):
    ACTIVE = "active"
    DEGRADED = "degraded"
    FAILED = "failed"
    MAINTENANCE = "maintenance"

@dataclass
class MarketDataPoint:
    """Standardized market data point"""
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    open: float
    high: float
    low: float
    close: float
    volume: float
    source: str
    quality_score: float = 1.0
    latency_ms: float = 0.0

@dataclass
class DataSourceConfig:
    """Data source configuration"""
    name: str
    priority: int
    api_key: str
    base_url: str
    rate_limit: int
    websocket_url: Optional[str] = None
    symbols_supported: List[str] = field(default_factory=list)
    active: bool = True

class DataQualityValidator:
    """Validates data quality and detects anomalies"""
    
    def __init__(self):
        self.logger = logging.getLogger('DataQualityValidator')
        self.price_history = {}
        self.volume_history = {}
        
    def validate_tick(self, data: MarketDataPoint) -> tuple[bool, float, str]:
        """Validate individual tick data"""
        issues = []
        quality_score = 1.0
        
        # Basic sanity checks
        if data.bid <= 0 or data.ask <= 0:
            issues.append("Invalid bid/ask prices")
            quality_score -= 0.5
            
        if data.bid > data.ask:
            issues.append("Bid > Ask (crossed market)")
            quality_score -= 0.3
            
        if data.high < data.low:
            issues.append("High < Low")
            quality_score -= 0.5
            
        if not (data.low <= data.close <= data.high):
            issues.append("Close price outside range")
            quality_score -= 0.3
            
        # Historical validation
        if data.symbol in self.price_history:
            last_price = self.price_history[data.symbol]
            price_change = abs(data.close - last_price) / last_price
            
            # Check for unrealistic price movements (>10% in 1 minute)
            if price_change > 0.10:
                issues.append(f"Extreme price movement: {price_change:.2%}")
                quality_score -= 0.4
                
        # Update history
        self.price_history[data.symbol] = data.close
        
        is_valid = quality_score > 0.5
        return is_valid, max(0.0, quality_score), "; ".join(issues)

class AlphaVantageProvider:
    """Alpha Vantage data provider"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.rate_limit = 5  # requests per minute
        self.last_request = 0
        self.logger = logging.getLogger('AlphaVantage')
        
    async def get_real_time_quote(self, symbol: str) -> Optional[MarketDataPoint]:
        """Get real-time quote from Alpha Vantage"""
        try:
            # Rate limiting
            current_time = time.time()
            if current_time - self.last_request < 12:  # 5 requests per minute
                await asyncio.sleep(12 - (current_time - self.last_request))
            
            self.last_request = time.time()
            
            # Convert symbol format
            av_symbol = symbol.replace('/', '')
            
            params = {
                'function': 'CURRENCY_EXCHANGE_RATE',
                'from_currency': av_symbol[:3],
                'to_currency': av_symbol[3:],
                'apikey': self.api_key
            }
            
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                async with session.get(self.base_url, params=params) as response:
                    latency = (time.time() - start_time) * 1000
                    
                    if response.status != 200:
                        return None
                        
                    data = await response.json()
                    
                    if 'Realtime Currency Exchange Rate' not in data:
                        return None
                    
                    rate_data = data['Realtime Currency Exchange Rate']
                    rate = float(rate_data['5. Exchange Rate'])
                    
                    return MarketDataPoint(
                        symbol=symbol,
                        timestamp=datetime.now(TIMEZONE),
                        bid=rate * 0.9999,  # Approximate bid
                        ask=rate * 1.0001,  # Approximate ask
                        open=rate,
                        high=rate,
                        low=rate,
                        close=rate,
                        volume=0,
                        source='alphavantage',
                        latency_ms=latency
                    )
                    
        except Exception as e:
            self.logger.error(f"Alpha Vantage error for {symbol}: {e}")
            return None

class YahooFinanceProvider:
    """Yahoo Finance data provider"""
    
    def __init__(self):
        self.logger = logging.getLogger('YahooFinance')
        
    async def get_real_time_quote(self, symbol: str) -> Optional[MarketDataPoint]:
        """Get real-time quote from Yahoo Finance"""
        try:
            start_time = time.time()
            
            # Convert symbol format for Yahoo
            yf_symbol = symbol.replace('/', '') + '=X'
            
            ticker = yf.Ticker(yf_symbol)
            info = ticker.info
            
            if not info or 'regularMarketPrice' not in info:
                return None
                
            latency = (time.time() - start_time) * 1000
            price = info['regularMarketPrice']
            
            return MarketDataPoint(
                symbol=symbol,
                timestamp=datetime.now(TIMEZONE),
                bid=price * 0.9999,
                ask=price * 1.0001,
                open=info.get('regularMarketOpen', price),
                high=info.get('regularMarketDayHigh', price),
                low=info.get('regularMarketDayLow', price),
                close=price,
                volume=info.get('regularMarketVolume', 0),
                source='yahoo',
                latency_ms=latency
            )
            
        except Exception as e:
            self.logger.error(f"Yahoo Finance error for {symbol}: {e}")
            return None

class FXAPIProvider:
    """Free FX API provider as backup"""
    
    def __init__(self):
        self.base_url = "https://api.fxapi.com/v1/latest"
        self.logger = logging.getLogger('FXAPI')
        
    async def get_real_time_quote(self, symbol: str) -> Optional[MarketDataPoint]:
        """Get real-time quote from FX API"""
        try:
            start_time = time.time()
            
            # Parse symbol
            base_currency = symbol[:3]
            quote_currency = symbol[4:7]
            
            params = {
                'base': base_currency,
                'symbols': quote_currency
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    latency = (time.time() - start_time) * 1000
                    
                    if response.status != 200:
                        return None
                        
                    data = await response.json()
                    
                    if 'rates' not in data or quote_currency not in data['rates']:
                        return None
                    
                    rate = data['rates'][quote_currency]
                    
                    return MarketDataPoint(
                        symbol=symbol,
                        timestamp=datetime.now(TIMEZONE),
                        bid=rate * 0.9999,
                        ask=rate * 1.0001,
                        open=rate,
                        high=rate,
                        low=rate,
                        close=rate,
                        volume=0,
                        source='fxapi',
                        latency_ms=latency
                    )
                    
        except Exception as e:
            self.logger.error(f"FX API error for {symbol}: {e}")
            return None

class RedundantDataManager:
    """Manages multiple data sources with automatic failover"""
    
    def __init__(self):
        self.logger = logging.getLogger('RedundantDataManager')
        self.data_sources = []
        self.source_status = {}
        self.quality_validator = DataQualityValidator()
        self.data_cache = {}
        self.failover_threshold = 5.0  # seconds
        self.quality_threshold = 0.7
        self.active_connections = {}
        
        # Initialize database
        self._initialize_database()
        
        # Setup data providers
        self._setup_data_providers()
        
        # Start health monitoring
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_sources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def _initialize_database(self):
        """Initialize data quality tracking database"""
        try:
            conn = sqlite3.connect(DATABASE_CONFIG['signals_db'])
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS data_quality_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    source TEXT NOT NULL,
                    latency_ms REAL,
                    quality_score REAL,
                    status TEXT,
                    issues TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS data_source_health (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    source TEXT NOT NULL,
                    status TEXT NOT NULL,
                    uptime_pct REAL,
                    avg_latency_ms REAL,
                    error_rate REAL
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Database initialization error: {e}")
    
    def _setup_data_providers(self):
        """Setup all data providers"""
        try:
            # Primary: Alpha Vantage (if API key available)
            alpha_vantage_key = "demo"  # Replace with actual API key
            if alpha_vantage_key and alpha_vantage_key != "demo":
                self.data_sources.append({
                    'name': 'alphavantage',
                    'provider': AlphaVantageProvider(alpha_vantage_key),
                    'priority': 1,
                    'status': DataSourceStatus.ACTIVE
                })
            
            # Secondary: Yahoo Finance
            self.data_sources.append({
                'name': 'yahoo',
                'provider': YahooFinanceProvider(),
                'priority': 2,
                'status': DataSourceStatus.ACTIVE
            })
            
            # Tertiary: FX API
            self.data_sources.append({
                'name': 'fxapi',
                'provider': FXAPIProvider(),
                'priority': 3,
                'status': DataSourceStatus.ACTIVE
            })
            
            self.logger.info(f"Initialized {len(self.data_sources)} data sources")
            
        except Exception as e:
            self.logger.error(f"Data provider setup error: {e}")
    
    async def get_real_time_data(self, symbol: str) -> Optional[MarketDataPoint]:
        """Get real-time data with automatic failover"""
        attempts = 0
        max_attempts = len(self.data_sources)
        
        # Sort by priority and status
        available_sources = [
            source for source in self.data_sources 
            if source['status'] in [DataSourceStatus.ACTIVE, DataSourceStatus.DEGRADED]
        ]
        available_sources.sort(key=lambda x: x['priority'])
        
        for source in available_sources:
            if attempts >= max_attempts:
                break
                
            attempts += 1
            provider = source['provider']
            source_name = source['name']
            
            try:
                self.logger.debug(f"Trying {source_name} for {symbol}")
                
                # Get data from source
                data = await provider.get_real_time_quote(symbol)
                
                if data is None:
                    self.logger.warning(f"{source_name} returned no data for {symbol}")
                    continue
                
                # Validate data quality
                is_valid, quality_score, issues = self.quality_validator.validate_tick(data)
                data.quality_score = quality_score
                
                # Log quality metrics
                self._log_data_quality(data, issues)
                
                if is_valid and quality_score >= self.quality_threshold:
                    self.logger.debug(f"Got valid data from {source_name} for {symbol}")
                    self.data_cache[symbol] = data
                    return data
                else:
                    self.logger.warning(f"Poor quality data from {source_name}: {issues}")
                    
            except Exception as e:
                self.logger.error(f"Error getting data from {source_name}: {e}")
                # Mark source as degraded
                source['status'] = DataSourceStatus.DEGRADED
                continue
        
        # If all sources failed, try cached data
        if symbol in self.data_cache:
            cached_data = self.data_cache[symbol]
            age_minutes = (datetime.now(TIMEZONE) - cached_data.timestamp).total_seconds() / 60
            
            if age_minutes < 5:  # Use cached data if less than 5 minutes old
                self.logger.warning(f"Using cached data for {symbol} ({age_minutes:.1f}m old)")
                return cached_data
        
        self.logger.error(f"All data sources failed for {symbol}")
        return None
    
    def _log_data_quality(self, data: MarketDataPoint, issues: str):
        """Log data quality metrics"""
        try:
            conn = sqlite3.connect(DATABASE_CONFIG['signals_db'])
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO data_quality_metrics 
                (timestamp, symbol, source, latency_ms, quality_score, status, issues)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                data.timestamp.isoformat(),
                data.symbol,
                data.source,
                data.latency_ms,
                data.quality_score,
                'valid' if data.quality_score >= self.quality_threshold else 'invalid',
                issues
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error logging data quality: {e}")
    
    def _monitor_sources(self):
        """Monitor data source health"""
        while self.monitoring_active:
            try:
                for source in self.data_sources:
                    # Test each source with a sample symbol
                    test_symbol = "EUR/USD"
                    start_time = time.time()
                    
                    try:
                        # This would be an async call in practice
                        asyncio.run(source['provider'].get_real_time_quote(test_symbol))
                        response_time = (time.time() - start_time) * 1000
                        
                        # Update source status
                        if response_time < 1000:  # < 1 second
                            source['status'] = DataSourceStatus.ACTIVE
                        elif response_time < 5000:  # < 5 seconds
                            source['status'] = DataSourceStatus.DEGRADED
                        else:
                            source['status'] = DataSourceStatus.FAILED
                            
                    except Exception:
                        source['status'] = DataSourceStatus.FAILED
                
                # Sleep for 60 seconds before next check
                time.sleep(60)
                
            except Exception as e:
                self.logger.error(f"Monitor error: {e}")
                time.sleep(60)
    
    def get_source_status(self) -> Dict[str, Any]:
        """Get status of all data sources"""
        status = {}
        for source in self.data_sources:
            status[source['name']] = {
                'status': source['status'].value,
                'priority': source['priority']
            }
        return status
    
    def get_data_quality_report(self, hours: int = 24) -> Dict[str, Any]:
        """Get data quality report for the last N hours"""
        try:
            conn = sqlite3.connect(DATABASE_CONFIG['signals_db'])
            cursor = conn.cursor()
            
            cutoff_time = datetime.now(TIMEZONE) - timedelta(hours=hours)
            
            cursor.execute('''
                SELECT source, 
                       COUNT(*) as total_requests,
                       AVG(quality_score) as avg_quality,
                       AVG(latency_ms) as avg_latency,
                       SUM(CASE WHEN status = 'valid' THEN 1 ELSE 0 END) as valid_count
                FROM data_quality_metrics 
                WHERE timestamp > ?
                GROUP BY source
            ''', (cutoff_time.isoformat(),))
            
            results = cursor.fetchall()
            conn.close()
            
            report = {}
            for row in results:
                source, total, avg_quality, avg_latency, valid_count = row
                report[source] = {
                    'total_requests': total,
                    'success_rate': (valid_count / total) * 100 if total > 0 else 0,
                    'avg_quality_score': avg_quality,
                    'avg_latency_ms': avg_latency
                }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating quality report: {e}")
            return {}
    
    def shutdown(self):
        """Shutdown the data manager"""
        self.monitoring_active = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=5)
        self.logger.info("Redundant Data Manager shutdown complete")

# Example usage and testing
async def test_redundant_data_manager():
    """Test the redundant data manager"""
    manager = RedundantDataManager()
    
    test_symbols = ["EUR/USD", "GBP/USD", "USD/JPY"]
    
    for symbol in test_symbols:
        print(f"\nTesting {symbol}:")
        data = await manager.get_real_time_data(symbol)
        
        if data:
            print(f"  Source: {data.source}")
            print(f"  Price: {data.close:.5f}")
            print(f"  Quality: {data.quality_score:.2f}")
            print(f"  Latency: {data.latency_ms:.1f}ms")
        else:
            print(f"  Failed to get data")
    
    # Print status report
    print("\nData Source Status:")
    status = manager.get_source_status()
    for source, info in status.items():
        print(f"  {source}: {info['status']} (priority: {info['priority']})")
    
    # Print quality report
    print("\nData Quality Report (last 1 hour):")
    quality_report = manager.get_data_quality_report(1)
    for source, metrics in quality_report.items():
        print(f"  {source}:")
        print(f"    Success Rate: {metrics['success_rate']:.1f}%")
        print(f"    Avg Quality: {metrics['avg_quality_score']:.2f}")
        print(f"    Avg Latency: {metrics['avg_latency_ms']:.1f}ms")
    
    manager.shutdown()

if __name__ == "__main__":
    asyncio.run(test_redundant_data_manager())