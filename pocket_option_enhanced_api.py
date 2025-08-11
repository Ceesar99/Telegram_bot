#!/usr/bin/env python3
"""
Enhanced Pocket Option API Integration
Optimized for binary options trading with advanced data collection and automation
"""

import asyncio
import aiohttp
import websockets
import json
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import base64
import hashlib
import hmac
from dataclasses import dataclass
import threading
from queue import Queue

@dataclass
class TradingSignal:
    asset: str
    direction: str  # 'call' or 'put'
    amount: float
    expiry_time: int  # in minutes
    confidence: float
    timestamp: datetime

class EnhancedPocketOptionAPI:
    def __init__(self, demo_mode=True):
        self.logger = logging.getLogger('EnhancedPocketOptionAPI')
        self.demo_mode = demo_mode
        self.session = None
        self.websocket = None
        self.is_connected = False
        self.user_id = None
        self.balance = 0.0
        self.assets = {}
        self.active_trades = {}
        self.data_queue = Queue()
        self.callbacks = {
            'candle': [],
            'trade_result': [],
            'balance_update': []
        }
        
        # API endpoints
        self.base_url = "https://pocketoption.com"
        self.ws_url = "wss://pocketoption.com/ws"
        self.api_endpoints = {
            'login': '/api/login',
            'trade': '/api/trade',
            'assets': '/api/assets',
            'history': '/api/history',
            'balance': '/api/balance'
        }
        
    async def connect(self, email=None, password=None, ssid=None):
        """Connect to Pocket Option platform"""
        try:
            # Create session
            self.session = aiohttp.ClientSession()
            
            # Authenticate
            if ssid:
                success = await self._authenticate_with_ssid(ssid)
            elif email and password:
                success = await self._authenticate_with_credentials(email, password)
            else:
                self.logger.error("No authentication credentials provided")
                return False
            
            if not success:
                return False
            
            # Connect to websocket
            await self._connect_websocket()
            
            # Subscribe to data feeds
            await self._subscribe_to_feeds()
            
            self.is_connected = True
            self.logger.info("Connected to Pocket Option platform")
            return True
            
        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            return False
    
    async def _authenticate_with_ssid(self, ssid):
        """Authenticate using session ID"""
        try:
            # Parse SSID to extract session data
            session_data = self._parse_ssid(ssid)
            if not session_data:
                return False
            
            # Set authentication headers
            self.session.headers.update({
                'Authorization': f"Bearer {session_data.get('session_id')}",
                'User-Agent': session_data.get('user_agent', 'Mozilla/5.0'),
                'X-Requested-With': 'XMLHttpRequest'
            })
            
            # Validate session
            async with self.session.get(f"{self.base_url}/api/profile") as response:
                if response.status == 200:
                    profile_data = await response.json()
                    self.user_id = profile_data.get('user_id')
                    self.balance = profile_data.get('balance', 0.0)
                    return True
                    
            return False
            
        except Exception as e:
            self.logger.error(f"SSID authentication failed: {e}")
            return False
    
    async def _authenticate_with_credentials(self, email, password):
        """Authenticate using email and password"""
        try:
            login_data = {
                'email': email,
                'password': password,
                'demo': self.demo_mode
            }
            
            async with self.session.post(f"{self.base_url}{self.api_endpoints['login']}", 
                                        json=login_data) as response:
                if response.status == 200:
                    auth_data = await response.json()
                    self.user_id = auth_data.get('user_id')
                    self.balance = auth_data.get('balance', 0.0)
                    
                    # Set authentication token
                    token = auth_data.get('token')
                    if token:
                        self.session.headers.update({
                            'Authorization': f"Bearer {token}"
                        })
                    
                    return True
                    
            return False
            
        except Exception as e:
            self.logger.error(f"Credential authentication failed: {e}")
            return False
    
    def _parse_ssid(self, ssid):
        """Parse SSID string to extract session data"""
        try:
            # Remove the WebSocket frame prefix
            if ssid.startswith('42["auth",'):
                json_str = ssid[10:-1]  # Remove '42["auth",' and closing ']'
                session_info = json.loads(json_str)
                
                if 'session' in session_info:
                    # Parse PHP session string
                    session_str = session_info['session']
                    session_data = self._parse_php_session(session_str)
                    session_data.update({
                        'user_id': session_info.get('uid'),
                        'demo_mode': session_info.get('isDemo', 1) == 1,
                        'platform': session_info.get('platform', 2)
                    })
                    return session_data
                    
        except Exception as e:
            self.logger.error(f"SSID parsing failed: {e}")
        
        return None
    
    def _parse_php_session(self, session_str):
        """Parse PHP session string"""
        try:
            # This is a simplified parser for the PHP session format
            # In practice, you might need a more robust parser
            import re
            
            pattern = r's:\d+:"([^"]+)";s:\d+:"([^"]+)";'
            matches = re.findall(pattern, session_str)
            
            session_data = {}
            for i in range(0, len(matches), 2):
                if i + 1 < len(matches):
                    key = matches[i][0] if isinstance(matches[i], tuple) else matches[i]
                    value = matches[i + 1][0] if isinstance(matches[i + 1], tuple) else matches[i + 1]
                    session_data[key] = value
            
            return session_data
            
        except Exception as e:
            self.logger.error(f"PHP session parsing failed: {e}")
            return {}
    
    async def _connect_websocket(self):
        """Connect to WebSocket for real-time data"""
        try:
            self.websocket = await websockets.connect(self.ws_url)
            
            # Start message handler
            asyncio.create_task(self._websocket_message_handler())
            
            self.logger.info("WebSocket connected")
            return True
            
        except Exception as e:
            self.logger.error(f"WebSocket connection failed: {e}")
            return False
    
    async def _websocket_message_handler(self):
        """Handle incoming WebSocket messages"""
        try:
            async for message in self.websocket:
                await self._process_websocket_message(message)
                
        except Exception as e:
            self.logger.error(f"WebSocket message handler error: {e}")
    
    async def _process_websocket_message(self, message):
        """Process incoming WebSocket message"""
        try:
            # Parse message
            if message.startswith('42'):
                # Socket.IO message format
                json_str = message[2:]
                data = json.loads(json_str)
                
                if isinstance(data, list) and len(data) >= 2:
                    event_type = data[0]
                    event_data = data[1] if len(data) > 1 else {}
                    
                    # Handle different event types
                    if event_type == 'candle':
                        await self._handle_candle_data(event_data)
                    elif event_type == 'trade_result':
                        await self._handle_trade_result(event_data)
                    elif event_type == 'balance':
                        await self._handle_balance_update(event_data)
                    elif event_type == 'assets':
                        await self._handle_assets_update(event_data)
                        
        except Exception as e:
            self.logger.error(f"Message processing error: {e}")
    
    async def _handle_candle_data(self, data):
        """Handle incoming candle data"""
        try:
            asset = data.get('asset')
            if not asset:
                return
            
            candle = {
                'asset': asset,
                'timestamp': data.get('timestamp', time.time()),
                'open': float(data.get('open', 0)),
                'high': float(data.get('high', 0)),
                'low': float(data.get('low', 0)),
                'close': float(data.get('close', 0)),
                'volume': float(data.get('volume', 0))
            }
            
            # Add to data queue
            self.data_queue.put(('candle', candle))
            
            # Call callbacks
            for callback in self.callbacks['candle']:
                try:
                    await callback(candle)
                except Exception as e:
                    self.logger.error(f"Candle callback error: {e}")
                    
        except Exception as e:
            self.logger.error(f"Candle data handling error: {e}")
    
    async def _handle_trade_result(self, data):
        """Handle trade result"""
        try:
            trade_id = data.get('trade_id')
            result = data.get('result')  # 'win' or 'loss'
            payout = float(data.get('payout', 0))
            
            if trade_id in self.active_trades:
                trade = self.active_trades[trade_id]
                trade['result'] = result
                trade['payout'] = payout
                trade['closed_at'] = datetime.now()
                
                # Update balance
                if result == 'win':
                    self.balance += payout
                
                # Call callbacks
                for callback in self.callbacks['trade_result']:
                    try:
                        await callback(trade)
                    except Exception as e:
                        self.logger.error(f"Trade result callback error: {e}")
                
                # Remove from active trades
                del self.active_trades[trade_id]
                
        except Exception as e:
            self.logger.error(f"Trade result handling error: {e}")
    
    async def _handle_balance_update(self, data):
        """Handle balance update"""
        try:
            new_balance = float(data.get('balance', self.balance))
            if new_balance != self.balance:
                old_balance = self.balance
                self.balance = new_balance
                
                # Call callbacks
                for callback in self.callbacks['balance_update']:
                    try:
                        await callback({'old_balance': old_balance, 'new_balance': new_balance})
                    except Exception as e:
                        self.logger.error(f"Balance update callback error: {e}")
                        
        except Exception as e:
            self.logger.error(f"Balance update handling error: {e}")
    
    async def _handle_assets_update(self, data):
        """Handle assets update"""
        try:
            if isinstance(data, dict):
                self.assets.update(data)
                
        except Exception as e:
            self.logger.error(f"Assets update handling error: {e}")
    
    async def _subscribe_to_feeds(self):
        """Subscribe to data feeds"""
        try:
            # Subscribe to real-time candle data for major pairs
            major_pairs = ['EURUSD_OTC', 'GBPUSD_OTC', 'USDJPY_OTC', 'AUDUSD_OTC']
            
            for pair in major_pairs:
                subscribe_msg = json.dumps(['subscribe', {'asset': pair, 'type': 'candle'}])
                await self.websocket.send(f"42{subscribe_msg}")
            
            # Subscribe to balance updates
            balance_msg = json.dumps(['subscribe', {'type': 'balance'}])
            await self.websocket.send(f"42{balance_msg}")
            
            self.logger.info("Subscribed to data feeds")
            
        except Exception as e:
            self.logger.error(f"Subscription error: {e}")
    
    async def get_assets(self):
        """Get available trading assets"""
        try:
            if not self.session:
                return []
            
            async with self.session.get(f"{self.base_url}{self.api_endpoints['assets']}") as response:
                if response.status == 200:
                    assets_data = await response.json()
                    return assets_data.get('assets', [])
                    
            return []
            
        except Exception as e:
            self.logger.error(f"Get assets error: {e}")
            return []
    
    async def get_candle_data(self, asset, timeframe='1m', count=100):
        """Get historical candle data"""
        try:
            params = {
                'asset': asset,
                'timeframe': timeframe,
                'count': count
            }
            
            async with self.session.get(f"{self.base_url}/api/candles", params=params) as response:
                if response.status == 200:
                    candles_data = await response.json()
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(candles_data.get('candles', []))
                    if not df.empty:
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                        df = df.set_index('timestamp')
                        
                    return df
                    
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Get candle data error: {e}")
            return pd.DataFrame()
    
    async def place_trade(self, signal: TradingSignal):
        """Place a binary options trade"""
        try:
            if not self.is_connected:
                self.logger.error("Not connected to platform")
                return None
            
            trade_data = {
                'asset': signal.asset,
                'direction': signal.direction,
                'amount': signal.amount,
                'expiry': signal.expiry_time,
                'demo': self.demo_mode
            }
            
            async with self.session.post(f"{self.base_url}{self.api_endpoints['trade']}", 
                                        json=trade_data) as response:
                if response.status == 200:
                    trade_result = await response.json()
                    trade_id = trade_result.get('trade_id')
                    
                    if trade_id:
                        # Store active trade
                        trade_info = {
                            'trade_id': trade_id,
                            'asset': signal.asset,
                            'direction': signal.direction,
                            'amount': signal.amount,
                            'expiry': signal.expiry_time,
                            'confidence': signal.confidence,
                            'opened_at': signal.timestamp,
                            'status': 'active'
                        }
                        
                        self.active_trades[trade_id] = trade_info
                        
                        self.logger.info(f"Trade placed: {trade_id} - {signal.asset} {signal.direction} ${signal.amount}")
                        return trade_info
                        
            return None
            
        except Exception as e:
            self.logger.error(f"Place trade error: {e}")
            return None
    
    async def close_trade(self, trade_id):
        """Close an active trade early (if supported)"""
        try:
            close_data = {'trade_id': trade_id}
            
            async with self.session.post(f"{self.base_url}/api/close_trade", 
                                        json=close_data) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get('success', False)
                    
            return False
            
        except Exception as e:
            self.logger.error(f"Close trade error: {e}")
            return False
    
    def register_callback(self, event_type: str, callback: Callable):
        """Register a callback for events"""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
    
    def get_balance(self):
        """Get current balance"""
        return self.balance
    
    def get_active_trades(self):
        """Get active trades"""
        return list(self.active_trades.values())
    
    async def get_trade_history(self, limit=100):
        """Get trade history"""
        try:
            params = {'limit': limit}
            
            async with self.session.get(f"{self.base_url}{self.api_endpoints['history']}", 
                                       params=params) as response:
                if response.status == 200:
                    history_data = await response.json()
                    return history_data.get('trades', [])
                    
            return []
            
        except Exception as e:
            self.logger.error(f"Get trade history error: {e}")
            return []
    
    async def get_real_time_data(self, asset, duration=60):
        """Get real-time data for specified duration"""
        data_points = []
        start_time = time.time()
        
        def data_collector(candle_data):
            if candle_data['asset'] == asset:
                data_points.append(candle_data)
        
        # Register temporary callback
        self.register_callback('candle', data_collector)
        
        # Wait for specified duration
        await asyncio.sleep(duration)
        
        # Remove callback (in practice, you'd want a better callback management system)
        if data_collector in self.callbacks['candle']:
            self.callbacks['candle'].remove(data_collector)
        
        # Convert to DataFrame
        if data_points:
            df = pd.DataFrame(data_points)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df.set_index('timestamp')
            return df
        
        return pd.DataFrame()
    
    async def disconnect(self):
        """Disconnect from platform"""
        try:
            if self.websocket:
                await self.websocket.close()
            
            if self.session:
                await self.session.close()
            
            self.is_connected = False
            self.logger.info("Disconnected from Pocket Option platform")
            
        except Exception as e:
            self.logger.error(f"Disconnect error: {e}")

class PocketOptionDataCollector:
    """Enhanced data collector for Pocket Option"""
    
    def __init__(self, api: EnhancedPocketOptionAPI):
        self.api = api
        self.logger = logging.getLogger('PocketOptionDataCollector')
        self.data_storage = {}
        self.collection_active = False
    
    async def start_collection(self, assets: List[str], storage_duration_hours=24):
        """Start collecting real-time data"""
        self.collection_active = True
        
        # Register callback for candle data
        self.api.register_callback('candle', self._store_candle_data)
        
        self.logger.info(f"Started data collection for {len(assets)} assets")
        
        # Keep collection running
        while self.collection_active:
            await asyncio.sleep(1)
    
    async def _store_candle_data(self, candle_data):
        """Store incoming candle data"""
        asset = candle_data['asset']
        
        if asset not in self.data_storage:
            self.data_storage[asset] = []
        
        self.data_storage[asset].append(candle_data)
        
        # Limit storage to prevent memory issues
        if len(self.data_storage[asset]) > 1440:  # 24 hours of minute data
            self.data_storage[asset] = self.data_storage[asset][-1440:]
    
    def get_data_for_asset(self, asset: str) -> pd.DataFrame:
        """Get collected data for specific asset"""
        if asset in self.data_storage and self.data_storage[asset]:
            df = pd.DataFrame(self.data_storage[asset])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df.set_index('timestamp')
            return df
        
        return pd.DataFrame()
    
    def stop_collection(self):
        """Stop data collection"""
        self.collection_active = False

# Example usage and testing
async def test_enhanced_api():
    """Test the enhanced API"""
    # Initialize API
    api = EnhancedPocketOptionAPI(demo_mode=True)
    
    # Connect (you would use real credentials here)
    # await api.connect(ssid="your_real_ssid_here")
    
    # For testing, we'll simulate some operations
    print("Enhanced Pocket Option API initialized")
    print("Demo mode:", api.demo_mode)
    
    # Test signal creation
    signal = TradingSignal(
        asset="EURUSD_OTC",
        direction="call",
        amount=1.0,
        expiry_time=3,
        confidence=0.85,
        timestamp=datetime.now()
    )
    
    print(f"Created test signal: {signal}")

if __name__ == "__main__":
    asyncio.run(test_enhanced_api())