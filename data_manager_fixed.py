import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

class DataManager:
    """Fixed Data Manager for training purposes"""
    
    def __init__(self):
        self.logger = logging.getLogger('DataManager')
        
    def create_sample_data(self, samples=10000):
        """Create sample market data for training"""
        dates = pd.date_range(start='2023-01-01', periods=samples, freq='1min')
        
        # Generate realistic price data
        np.random.seed(42)
        base_price = 1.1000
        returns = np.random.normal(0, 0.0001, samples)
        
        prices = [base_price]
        for i in range(1, samples):
            new_price = prices[-1] * (1 + returns[i])
            prices.append(max(0.5, min(2.0, new_price)))
        
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.00005))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.00005))) for p in prices],
            'close': prices,
            'volume': np.random.randint(100, 1000, samples)
        })
        
        # Ensure OHLC consistency
        data['high'] = data[['open', 'close', 'high']].max(axis=1)
        data['low'] = data[['open', 'close', 'low']].min(axis=1)
        
        return data
    
    def get_market_data(self, symbol, timeframe='1m', limit=1000):
        """Get market data - returns sample data for now"""
        return self.create_sample_data(limit)