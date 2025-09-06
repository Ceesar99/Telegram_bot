import requests
import websocket
import json
import threading
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import logging
from typing import Dict, List, Optional, Callable
import os
import ssl
import urllib.parse
from config import (
    POCKET_OPTION_SSID, POCKET_OPTION_BASE_URL, POCKET_OPTION_WS_URL,
    CURRENCY_PAIRS, OTC_PAIRS, TIMEZONE, MARKET_TIMEZONE
)
import sqlite3

class PocketOptionAPI:
    def __init__(self):
        self.ssid = POCKET_OPTION_SSID
        self.base_url = POCKET_OPTION_BASE_URL
        self.ws_url = POCKET_OPTION_WS_URL
        self.ws = None
        self.connected = False
        self.data_callbacks = {}
        self.market_data = {}
        self.session = requests.Session()
        self.logger = self._setup_logger()
        self.is_weekend = False
        self.available_pairs = []
        self.server_time_offset = 0  # Server time offset in seconds
        self.last_time_sync = None
        self._init_orders_db()
        
        # Initialize session
        self._setup_session()
        
        # Synchronize with server time
        self._sync_server_time()

        # External library integration (optional)
        self.external = None
        self.external_connected = False
        self.demo_mode = os.getenv("POCKET_OPTION_DEMO", "true").lower() in ("1", "true", "yes")
        self.data_source = os.getenv("POCKET_OPTION_DATA_SOURCE", "http").lower()  # http|external
        if os.getenv("POCKET_OPTION_EXTERNAL", "false").lower() in ("1", "true", "yes"):
            self._init_external_client()
        
    def _setup_logger(self):
        logger = logging.getLogger('PocketOptionAPI')
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler('/workspace/logs/pocket_option_api.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def _setup_session(self):
        """Setup session with proper headers and authentication"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36 Edg/137.',
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'X-Requested-With': 'XMLHttpRequest'
        }
        self.session.headers.update(headers)
        
        # Set authentication cookie
        self.session.cookies.set('ssid', self._extract_session_id())
        
    def _extract_session_id(self):
        """Extract session ID from SSID"""
        try:
            # Parse the SSID string to extract session ID
            ssid_data = json.loads(self.ssid[2:])  # Remove '42' prefix
            if isinstance(ssid_data, list) and len(ssid_data) > 1:
                auth_data = ssid_data[1]
                if 'session' in auth_data:
                    # Extract session ID from session string
                    session_str = auth_data['session']
                    # Parse PHP session format
                    if 'session_id' in session_str:
                        import re
                        match = re.search(r's:32:"([^"]+)"', session_str)
                        if match:
                            return match.group(1)
            return "8ddc70c84462c00f33c4e55cd07348c2"  # Default session ID
        except Exception as e:
            self.logger.error(f"Failed to extract session ID: {e}")
            return "8ddc70c84462c00f33c4e55cd07348c2"
    
    def _sync_server_time(self):
        """Synchronize with Pocket Option server time"""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/time")
            if response.status_code == 200:
                server_time = response.json().get('time', time.time())
                local_time = time.time()
                self.server_time_offset = server_time - local_time
                self.last_time_sync = datetime.now(TIMEZONE)
                self.logger.info(f"Server time synchronized. Offset: {self.server_time_offset:.3f}s")
            else:
                self.logger.warning("Failed to sync with server time, using local time")
        except Exception as e:
            self.logger.error(f"Error syncing server time: {e}")
    
    def get_server_time(self):
        """Get current server time"""
        # Re-sync every hour
        if (self.last_time_sync is None or 
            (datetime.now(TIMEZONE) - self.last_time_sync).total_seconds() > 3600):
            self._sync_server_time()
        
        return datetime.fromtimestamp(time.time() + self.server_time_offset, tz=TIMEZONE)
    
    def get_entry_time(self, advance_minutes=1):
        """Get precise entry time for signals"""
        server_time = self.get_server_time()
        # Round to next minute boundary and add advance time
        next_minute = server_time.replace(second=0, microsecond=0) + timedelta(minutes=1)
        entry_time = next_minute + timedelta(minutes=advance_minutes)
        return entry_time
    
    def check_market_hours(self):
        """Check if we're in weekend/OTC hours"""
        now = datetime.now(TIMEZONE)
        weekday = now.weekday()
        
        # Weekend: Saturday (5) and Sunday (6)
        self.is_weekend = weekday in [5, 6]
        
        # Friday 17:00 EST to Sunday 17:00 EST is weekend
        if weekday == 4:  # Friday
            if now.hour >= 22:  # 17:00 EST = 22:00 UTC
                self.is_weekend = True
        elif weekday == 6:  # Sunday
            if now.hour < 22:  # Before 17:00 EST
                self.is_weekend = True
                
        return self.is_weekend
    
    def get_available_pairs(self):
        """Get list of available currency pairs based on market hours"""
        self.check_market_hours()
        
        if self.is_weekend:
            self.available_pairs = OTC_PAIRS.copy()
            self.logger.info("Using OTC pairs for weekend trading")
        else:
            self.available_pairs = CURRENCY_PAIRS.copy()
            self.logger.info("Using regular pairs for weekday trading")
        
        return self.available_pairs
    
    def get_market_data(self, symbol: str, timeframe: str = "1m", limit: int = 100):
        """Get historical market data for a symbol"""
        # Try external library if selected
        if self.data_source == "external" and self.external_connected:
            try:
                df = self._get_market_data_external(symbol, timeframe=timeframe, limit=limit)
                if df is not None:
                    return df
            except Exception as e:
                self.logger.warning(f"External data source failed, falling back to HTTP: {e}")
        try:
            # Map symbol to Pocket Option format
            po_symbol = self._map_symbol(symbol)
            
            endpoint = f"/api/v1/history"
            params = {
                'symbol': po_symbol,
                'timeframe': timeframe,
                'limit': limit,
                'timestamp': int(time.time())
            }
            
            response = self.session.get(f"{self.base_url}{endpoint}", params=params)
            
            if response.status_code == 200:
                data = response.json()
                return self._process_market_data(data, symbol)
            else:
                self.logger.error(f"Failed to get market data: {response.status_code}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting market data for {symbol}: {e}")
            return None

    # --- External client integration helpers ---
    def _init_external_client(self):
        try:
            # Prefer pocketoptionapi from binaryoptionstoolsv2
            try:
                from pocketoptionapi.stable_api import PocketOption  # type: ignore
            except Exception:
                PocketOption = None
            if PocketOption is None:
                self.logger.warning("pocketoptionapi not available; skip external integration")
                return
            self.external = PocketOption(self.ssid, self.demo_mode)
            self.external_connected = bool(self.external.connect())
            if self.external_connected:
                self.logger.info("Connected to Pocket Option via external library")
            else:
                self.logger.warning("External Pocket Option connection failed")
        except Exception as e:
            self.external = None
            self.external_connected = False
            self.logger.error(f"Failed to initialize external Pocket Option client: {e}")

    def _get_market_data_external(self, symbol: str, timeframe: str = "1m", limit: int = 100):
        try:
            if not self.external_connected or not self.external:
                return None
            # Map timeframe to external expected format if necessary
            tf_map = {"1m": 60, "2m": 120, "3m": 180, "5m": 300}
            seconds = tf_map.get(timeframe, 60)
            po_symbol = self._map_symbol(symbol)
            # Many libs expose get_candles(symbol, period, amount)
            if hasattr(self.external, "get_candles"):
                candles = self.external.get_candles(po_symbol, seconds, limit)  # type: ignore
                # Expect list of dicts with keys: time, open, close, min, max, volume
                if candles:
                    df = pd.DataFrame([
                        {
                            'timestamp': pd.to_datetime(c.get('time') or c.get('t') or c.get('timestamp'), unit='s'),
                            'open': float(c.get('open') or c.get('o')),
                            'high': float(c.get('max') or c.get('h') or c.get('high')),
                            'low': float(c.get('min') or c.get('l') or c.get('low')),
                            'close': float(c.get('close') or c.get('c')),
                            'volume': float(c.get('volume', 0))
                        }
                        for c in candles
                    ])
                    df.set_index('timestamp', inplace=True)
                    df.sort_index(inplace=True)
                    return df
            return None
        except Exception as e:
            self.logger.error(f"External market data retrieval failed: {e}")
            return None
    
    def _map_symbol(self, symbol: str):
        """Map standard symbol to Pocket Option symbol format"""
        symbol_mapping = {
            # Forex pairs
            'EUR/USD': 'EURUSD',
            'GBP/USD': 'GBPUSD',
            'USD/JPY': 'USDJPY',
            'USD/CHF': 'USDCHF',
            'AUD/USD': 'AUDUSD',
            'USD/CAD': 'USDCAD',
            'NZD/USD': 'NZDUSD',
            'EUR/GBP': 'EURGBP',
            'EUR/JPY': 'EURJPY',
            'GBP/JPY': 'GBPJPY',
            
            # OTC pairs
            'EUR/USD OTC': 'EURUSD_OTC',
            'GBP/USD OTC': 'GBPUSD_OTC',
            'USD/JPY OTC': 'USDJPY_OTC',
            'AUD/USD OTC': 'AUDUSD_OTC',
            'USD/CAD OTC': 'USDCAD_OTC',
            
            # Crypto
            'BTC/USD': 'BTCUSD',
            'ETH/USD': 'ETHUSD',
            'LTC/USD': 'LTCUSD',
            
            # Commodities
            'XAU/USD': 'XAUUSD',
            'XAG/USD': 'XAGUSD',
            'OIL/USD': 'OILUSD',
            
            # Indices
            'SPX500': 'SPX500',
            'NASDAQ': 'NAS100',
            'DAX30': 'GER30',
            'FTSE100': 'UK100',
            'NIKKEI': 'JPN225'
        }
        
        return symbol_mapping.get(symbol, symbol.replace('/', ''))
    
    def _process_market_data(self, data, symbol):
        """Process raw market data into pandas DataFrame"""
        try:
            if 'data' not in data:
                return None
            
            df_data = []
            for candle in data['data']:
                df_data.append({
                    'timestamp': pd.to_datetime(candle['time'], unit='s'),
                    'open': float(candle['open']),
                    'high': float(candle['high']),
                    'low': float(candle['low']),
                    'close': float(candle['close']),
                    'volume': float(candle.get('volume', 0))
                })
            
            df = pd.DataFrame(df_data)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error processing market data: {e}")
            return None
    
    def get_current_price(self, symbol: str):
        """Get current price for a symbol"""
        try:
            po_symbol = self._map_symbol(symbol)
            
            endpoint = f"/api/v1/price"
            params = {'symbol': po_symbol}
            
            response = self.session.get(f"{self.base_url}{endpoint}", params=params)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'symbol': symbol,
                    'price': float(data.get('price', 0)),
                    'timestamp': datetime.now(TIMEZONE),
                    'bid': float(data.get('bid', 0)),
                    'ask': float(data.get('ask', 0))
                }
            else:
                self.logger.error(f"Failed to get current price: {response.status_code}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting current price for {symbol}: {e}")
            return None
    
    def connect_websocket(self):
        """Connect to Pocket Option WebSocket for real-time data"""
        try:
            def on_message(ws, message):
                self._handle_websocket_message(message)
            
            def on_error(ws, error):
                self.logger.error(f"WebSocket error: {error}")
                self.connected = False
            
            def on_close(ws, close_status_code, close_msg):
                self.logger.info("WebSocket connection closed")
                self.connected = False
            
            def on_open(ws):
                self.logger.info("WebSocket connection opened")
                self.connected = True
                # Send authentication message
                auth_message = self.ssid
                ws.send(auth_message)
                
                # Subscribe to all available pairs
                self._subscribe_to_pairs()
            
            # Create WebSocket connection
            websocket.enableTrace(False)
            self.ws = websocket.WebSocketApp(
                self.ws_url,
                header=[f"Cookie: ssid={self._extract_session_id()}"],
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open
            )
            
            # Start WebSocket in a separate thread
            self.ws_thread = threading.Thread(target=self.ws.run_forever, kwargs={
                'sslopt': {"cert_reqs": ssl.CERT_NONE}
            })
            self.ws_thread.daemon = True
            self.ws_thread.start()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect WebSocket: {e}")
            return False
    
    def _handle_websocket_message(self, message):
        """Handle incoming WebSocket messages"""
        try:
            if message.startswith('42'):
                # Socket.IO message format
                data = json.loads(message[2:])
                
                if isinstance(data, list) and len(data) > 1:
                    event_type = data[0]
                    event_data = data[1]
                    
                    if event_type == 'price':
                        self._process_price_update(event_data)
                    elif event_type == 'candle':
                        self._process_candle_update(event_data)
                        
        except Exception as e:
            self.logger.error(f"Error handling WebSocket message: {e}")
    
    def _process_price_update(self, data):
        """Process real-time price updates"""
        try:
            symbol = data.get('symbol', '')
            price = float(data.get('price', 0))
            timestamp = datetime.now(TIMEZONE)
            
            # Store in market data
            if symbol not in self.market_data:
                self.market_data[symbol] = {}
            
            self.market_data[symbol]['current_price'] = price
            self.market_data[symbol]['last_update'] = timestamp
            
            # Call registered callbacks
            if symbol in self.data_callbacks:
                for callback in self.data_callbacks[symbol]:
                    callback(symbol, price, timestamp)
                    
        except Exception as e:
            self.logger.error(f"Error processing price update: {e}")
    
    def _process_candle_update(self, data):
        """Process real-time candle updates"""
        try:
            symbol = data.get('symbol', '')
            candle_data = {
                'open': float(data.get('open', 0)),
                'high': float(data.get('high', 0)),
                'low': float(data.get('low', 0)),
                'close': float(data.get('close', 0)),
                'volume': float(data.get('volume', 0)),
                'timestamp': pd.to_datetime(data.get('time'), unit='s')
            }
            
            # Store in market data
            if symbol not in self.market_data:
                self.market_data[symbol] = {}
            
            if 'candles' not in self.market_data[symbol]:
                self.market_data[symbol]['candles'] = []
            
            self.market_data[symbol]['candles'].append(candle_data)
            
            # Keep only last 1000 candles
            if len(self.market_data[symbol]['candles']) > 1000:
                self.market_data[symbol]['candles'] = self.market_data[symbol]['candles'][-1000:]
                
        except Exception as e:
            self.logger.error(f"Error processing candle update: {e}")
    
    def _subscribe_to_pairs(self):
        """Subscribe to real-time data for all available pairs"""
        if not self.connected or not self.ws:
            return
        
        available_pairs = self.get_available_pairs()
        
        for pair in available_pairs:
            po_symbol = self._map_symbol(pair)
            
            # Subscribe to price updates
            subscribe_message = json.dumps([
                "subscribe",
                {"symbol": po_symbol, "type": "price"}
            ])
            self.ws.send(f"42{subscribe_message}")
            
            # Subscribe to candle updates
            candle_message = json.dumps([
                "subscribe", 
                {"symbol": po_symbol, "type": "candle", "timeframe": "1m"}
            ])
            self.ws.send(f"42{candle_message}")
            
        self.logger.info(f"Subscribed to {len(available_pairs)} currency pairs")

    # --- Order persistence and execution stubs ---
    def _init_orders_db(self):
        """Initialize simple orders/fills tables in SQLite for persistence."""
        try:
            conn = sqlite3.connect("/workspace/data/signals.db")
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    client_order_id TEXT UNIQUE,
                    timestamp TEXT,
                    symbol TEXT,
                    side TEXT,
                    amount REAL,
                    duration INTEGER,
                    entry_time TEXT,
                    status TEXT,
                    error TEXT
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS fills (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    client_order_id TEXT,
                    fill_time TEXT,
                    result TEXT,
                    payout REAL,
                    pnl REAL
                )
                """
            )
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"Error initializing orders db: {e}")

    def _persist_order(self, client_order_id: str, symbol: str, side: str, amount: float, duration: int, entry_time: datetime, status: str, error: Optional[str] = None):
        try:
            conn = sqlite3.connect("/workspace/data/signals.db")
            cur = conn.cursor()
            cur.execute(
                """
                INSERT OR IGNORE INTO orders (client_order_id, timestamp, symbol, side, amount, duration, entry_time, status, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (client_order_id, datetime.now(TIMEZONE).isoformat(), symbol, side, amount, duration, entry_time.isoformat(), status, error)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"Error persisting order: {e}")

    def _update_order_status(self, client_order_id: str, status: str, error: Optional[str] = None):
        try:
            conn = sqlite3.connect("/workspace/data/signals.db")
            cur = conn.cursor()
            cur.execute(
                "UPDATE orders SET status = ?, error = ? WHERE client_order_id = ?",
                (status, error, client_order_id)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"Error updating order status: {e}")

    def _persist_fill(self, client_order_id: str, result: str, payout: float, pnl: float):
        try:
            conn = sqlite3.connect("/workspace/data/signals.db")
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO fills (client_order_id, fill_time, result, payout, pnl)
                VALUES (?, ?, ?, ?, ?)
                """,
                (client_order_id, datetime.now(TIMEZONE).isoformat(), result, payout, pnl)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"Error persisting fill: {e}")

    def execute_trade(self, symbol_or_signal, side: Optional[str] = None, amount: Optional[float] = None, duration_minutes: Optional[int] = None, client_order_id: Optional[str] = None) -> Dict:
        """
        Submit a binary option trade request (stub). Accepts either a signal dict or explicit params.
        Persists the order and simulates acceptance. Real placement should integrate broker API.
        """
        try:
            # Unpack when first argument is a signal dict
            if isinstance(symbol_or_signal, dict):
                signal = symbol_or_signal
                symbol = signal.get('pair') or signal.get('symbol') or 'EUR/USD'
                raw_dir = signal.get('direction') or signal.get('signal') or 'CALL'
                side = 'BUY' if str(raw_dir).upper() in ('CALL', 'BUY') else 'SELL'
                amount = float(signal.get('position_size') or signal.get('risk_amount') or 1.0)
                duration_minutes = int(signal.get('recommended_duration') or signal.get('duration') or 3)
            else:
                symbol = str(symbol_or_signal)
                side = side or 'BUY'
                amount = float(amount or 1.0)
                duration_minutes = int(duration_minutes or 3)

            if client_order_id is None:
                client_order_id = f"PO_{int(time.time()*1000)}"

            entry_time = self.get_entry_time(advance_minutes=1)
            self._persist_order(client_order_id, symbol, side, amount, duration_minutes, entry_time, status="accepted")
            self.logger.info(f"Order accepted {client_order_id}: {symbol} {side} {amount} {duration_minutes}m")
            return {
                "client_order_id": client_order_id,
                "status": "accepted",
                "entry_time": entry_time
            }
        except Exception as e:
            try:
                self._persist_order(client_order_id or "", symbol if 'symbol' in locals() else "", side or "", float(amount or 0), int(duration_minutes or 0), datetime.now(TIMEZONE), status="rejected", error=str(e))
            except Exception:
                pass
            self.logger.error(f"Execute trade failed: {e}")
            return {"client_order_id": client_order_id, "status": "rejected", "error": str(e)}
    
    def register_data_callback(self, symbol: str, callback: Callable):
        """Register callback for real-time data updates"""
        if symbol not in self.data_callbacks:
            self.data_callbacks[symbol] = []
        
        self.data_callbacks[symbol].append(callback)
    
    def get_market_volatility(self, symbol: str, periods: int = 20):
        """Calculate market volatility for a symbol"""
        try:
            data = self.get_market_data(symbol, limit=periods + 10)
            if data is None or len(data) < periods:
                return None
            
            # Calculate returns
            returns = data['close'].pct_change().dropna()
            
            # Calculate volatility (standard deviation of returns)
            volatility = returns.std() * np.sqrt(1440)  # Annualized for 1-minute data
            
            return {
                'symbol': symbol,
                'volatility': volatility,
                'avg_return': returns.mean(),
                'periods': len(returns)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility for {symbol}: {e}")
            return None
    
    def get_support_resistance(self, symbol: str, periods: int = 50):
        """Calculate support and resistance levels"""
        try:
            data = self.get_market_data(symbol, limit=periods + 10)
            if data is None or len(data) < periods:
                return None
            
            # Calculate rolling max and min
            resistance = data['high'].rolling(window=20).max().iloc[-1]
            support = data['low'].rolling(window=20).min().iloc[-1]
            current_price = data['close'].iloc[-1]
            
            return {
                'symbol': symbol,
                'resistance': resistance,
                'support': support,
                'current_price': current_price,
                'price_position': (current_price - support) / (resistance - support) if resistance != support else 0.5
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating support/resistance for {symbol}: {e}")
            return None
    
    def is_low_volatility_period(self, symbol: str):
        """Check if current period has low volatility suitable for trading"""
        volatility_data = self.get_market_volatility(symbol)
        
        if volatility_data is None:
            return False
        
        # Define low volatility threshold (adjust based on your strategy)
        low_volatility_threshold = 0.02  # 2% annualized volatility
        
        return volatility_data['volatility'] < low_volatility_threshold
    
    def get_expiry_time(self, duration_minutes: int):
        """Calculate expiry time for binary options"""
        now = datetime.now(TIMEZONE)
        expiry = now + timedelta(minutes=duration_minutes)
        
        # Format for display
        start_time = now.strftime("%H:%M")
        end_time = expiry.strftime("%H:%M")
        
        return {
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration_minutes,
            'expiry_timestamp': expiry
        }
    
    def validate_trading_conditions(self, symbol: str):
        """Validate if conditions are suitable for trading"""
        conditions = {
            'market_open': True,
            'low_volatility': False,
            'data_available': False,
            'spread_acceptable': True
        }
        
        try:
            # Check if market data is available
            current_price = self.get_current_price(symbol)
            if current_price:
                conditions['data_available'] = True
            
            # Check volatility
            if self.is_low_volatility_period(symbol):
                conditions['low_volatility'] = True
            
            # Check market hours
            self.check_market_hours()
            
            return conditions
            
        except Exception as e:
            self.logger.error(f"Error validating trading conditions: {e}")
            return conditions
    
    def disconnect(self):
        """Disconnect from WebSocket and cleanup"""
        try:
            self.connected = False
            if self.ws:
                self.ws.close()
            self.logger.info("Disconnected from Pocket Option API")
        except Exception as e:
            self.logger.error(f"Error disconnecting: {e}")
    
    def get_account_info(self):
        """Get account information"""
        try:
            endpoint = "/api/v1/account"
            response = self.session.get(f"{self.base_url}{endpoint}")
            
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"Failed to get account info: {response.status_code}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return None