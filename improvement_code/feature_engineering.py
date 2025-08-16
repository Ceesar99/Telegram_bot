
# ADVANCED FEATURE ENGINEERING

def add_multi_timeframe_features(self, data):
    """Add multi-timeframe indicators"""
    
    # 5-minute indicators
    data['rsi_5m'] = talib.RSI(data['close'], timeperiod=14)
    data['macd_5m'] = talib.MACD(data['close'])[0]
    
    # 15-minute indicators  
    data['rsi_15m'] = talib.RSI(data['close'].resample('15T').last(), timeperiod=14)
    data['macd_15m'] = talib.MACD(data['close'].resample('15T').last())[0]
    
    # 1-hour indicators
    data['rsi_1h'] = talib.RSI(data['close'].resample('1H').last(), timeperiod=14)
    data['macd_1h'] = talib.MACD(data['close'].resample('1H').last())[0]
    
    return data

def add_volatility_features(self, data):
    """Add volatility-based features"""
    
    # ATR-based volatility
    data['atr'] = talib.ATR(data['high'], data['low'], data['close'])
    data['atr_ratio'] = data['atr'] / data['close']
    
    # Bollinger Band width
    bb_upper, bb_middle, bb_lower = talib.BBANDS(data['close'])
    data['bb_width'] = (bb_upper - bb_lower) / bb_middle
    
    # Volatility clustering
    data['volatility_20'] = data['close'].rolling(20).std()
    data['volatility_ratio'] = data['volatility_20'] / data['close']
    
    return data

def add_momentum_features(self, data):
    """Add momentum indicators"""
    
    # Price momentum
    data['momentum_5'] = data['close'].pct_change(5)
    data['momentum_10'] = data['close'].pct_change(10)
    data['momentum_20'] = data['close'].pct_change(20)
    
    # Rate of change
    data['roc_5'] = talib.ROC(data['close'], timeperiod=5)
    data['roc_10'] = talib.ROC(data['close'], timeperiod=10)
    
    # Williams %R
    data['williams_r'] = talib.WILLR(data['high'], data['low'], data['close'])
    
    return data
