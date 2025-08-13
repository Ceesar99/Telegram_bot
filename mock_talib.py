"""
Mock TA-Lib module for basic technical analysis functions
Provides simple implementations of common technical indicators
"""

import numpy as np
import pandas as pd

def SMA(data, timeperiod=14):
    """Simple Moving Average"""
    if isinstance(data, (list, np.ndarray)):
        data = pd.Series(data)
    return data.rolling(window=timeperiod).mean().values

def EMA(data, timeperiod=14):
    """Exponential Moving Average"""
    if isinstance(data, (list, np.ndarray)):
        data = pd.Series(data)
    return data.ewm(span=timeperiod).mean().values

def RSI(data, timeperiod=14):
    """Relative Strength Index"""
    if isinstance(data, (list, np.ndarray)):
        data = pd.Series(data)
    
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=timeperiod).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=timeperiod).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def MACD(data, fastperiod=12, slowperiod=26, signalperiod=9):
    """MACD - Moving Average Convergence/Divergence"""
    if isinstance(data, (list, np.ndarray)):
        data = pd.Series(data)
    
    ema_fast = data.ewm(span=fastperiod).mean()
    ema_slow = data.ewm(span=slowperiod).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signalperiod).mean()
    histogram = macd_line - signal_line
    
    return macd_line.values, signal_line.values, histogram.values

def BBANDS(data, timeperiod=20, nbdevup=2, nbdevdn=2):
    """Bollinger Bands"""
    if isinstance(data, (list, np.ndarray)):
        data = pd.Series(data)
    
    sma = data.rolling(window=timeperiod).mean()
    std = data.rolling(window=timeperiod).std()
    
    upper_band = sma + (std * nbdevup)
    lower_band = sma - (std * nbdevdn)
    
    return upper_band.values, sma.values, lower_band.values

def STOCH(high, low, close, fastk_period=5, slowk_period=3, slowd_period=3):
    """Stochastic Oscillator"""
    if isinstance(high, (list, np.ndarray)):
        high = pd.Series(high)
        low = pd.Series(low)
        close = pd.Series(close)
    
    lowest_low = low.rolling(window=fastk_period).min()
    highest_high = high.rolling(window=fastk_period).max()
    
    k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
    slowk = k_percent.rolling(window=slowk_period).mean()
    slowd = slowk.rolling(window=slowd_period).mean()
    
    return slowk.values, slowd.values

def ATR(high, low, close, timeperiod=14):
    """Average True Range"""
    if isinstance(high, (list, np.ndarray)):
        high = pd.Series(high)
        low = pd.Series(low)
        close = pd.Series(close)
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=timeperiod).mean()
    
    return atr.values

def ADX(high, low, close, timeperiod=14):
    """Average Directional Index"""
    if isinstance(high, (list, np.ndarray)):
        high = pd.Series(high)
        low = pd.Series(low)
        close = pd.Series(close)
    
    # Simplified ADX calculation
    tr = ATR(high, low, close, 1)
    up = high.diff()
    down = -low.diff()
    
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    
    plus_dm_series = pd.Series(plus_dm).rolling(window=timeperiod).mean()
    minus_dm_series = pd.Series(minus_dm).rolling(window=timeperiod).mean()
    tr_series = pd.Series(tr).rolling(window=timeperiod).mean()
    
    plus_di = 100 * (plus_dm_series / tr_series)
    minus_di = 100 * (minus_dm_series / tr_series)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=timeperiod).mean()
    
    return adx.values

def OBV(close, volume):
    """On Balance Volume"""
    if isinstance(close, (list, np.ndarray)):
        close = pd.Series(close)
        volume = pd.Series(volume)
    
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    return obv.values

def WILLR(high, low, close, timeperiod=14):
    """Williams %R"""
    if isinstance(high, (list, np.ndarray)):
        high = pd.Series(high)
        low = pd.Series(low)
        close = pd.Series(close)
    
    highest_high = high.rolling(window=timeperiod).max()
    lowest_low = low.rolling(window=timeperiod).min()
    
    willr = -100 * (highest_high - close) / (highest_high - lowest_low)
    return willr.values

# Add more functions as needed
def CCI(high, low, close, timeperiod=14):
    """Commodity Channel Index"""
    if isinstance(high, (list, np.ndarray)):
        high = pd.Series(high)
        low = pd.Series(low)
        close = pd.Series(close)
    
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(window=timeperiod).mean()
    mad = tp.rolling(window=timeperiod).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    
    cci = (tp - sma_tp) / (0.015 * mad)
    return cci.values