#!/usr/bin/env python3
"""
Create Feature Scaler for LSTM Model

This script creates a feature_scaler.pkl file that the LSTM model can load.
The scaler needs to be fitted with sample data to have the proper scale parameters.
"""

import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler

def create_feature_scaler():
    """Create and save a feature scaler for the LSTM model"""
    
    # Create models directory if it doesn't exist
    models_dir = '/workspace/models'
    os.makedirs(models_dir, exist_ok=True)
    
    # Create a StandardScaler (same as in lstm_model.py)
    feature_scaler = StandardScaler()
    
    # Generate sample feature data to fit the scaler
    # Based on the technical indicators in lstm_model.py, we need features for:
    # - price_change, price_volatility
    # - rsi, rsi_signal
    # - macd, macd_signal, macd_histogram, macd_crossover
    # - bb_position, bb_squeeze
    # - stoch_k, stoch_d, stoch_signal
    # - williams_r, cci, adx, atr_normalized
    # - ema signals, sma signals
    # - volume_ratio, obv
    # - price_position
    
    # Generate realistic sample data
    np.random.seed(42)  # For reproducibility
    
    # Number of samples and features
    n_samples = 1000
    n_features = 50  # Approximate number of features from technical indicators
    
    # Generate sample features with realistic ranges
    sample_features = np.random.randn(n_samples, n_features)
    
    # Set realistic ranges for specific features
    # RSI: 0-100
    sample_features[:, 2] = np.random.uniform(0, 100, n_samples)
    # MACD: typically -2 to 2
    sample_features[:, 4] = np.random.uniform(-2, 2, n_samples)
    # Bollinger Bands position: 0-1
    sample_features[:, 8] = np.random.uniform(0, 1, n_samples)
    # Stochastic: 0-100
    sample_features[:, 11] = np.random.uniform(0, 100, n_samples)
    # Williams %R: -100 to 0
    sample_features[:, 13] = np.random.uniform(-100, 0, n_samples)
    # CCI: typically -300 to 300
    sample_features[:, 14] = np.random.uniform(-300, 300, n_samples)
    # ADX: 0-100
    sample_features[:, 15] = np.random.uniform(0, 100, n_samples)
    # Price position: 0-1
    sample_features[:, -1] = np.random.uniform(0, 1, n_samples)
    
    # Fit the scaler with the sample data
    feature_scaler.fit(sample_features)
    
    # Save the fitted scaler
    scaler_path = os.path.join(models_dir, 'feature_scaler.pkl')
    joblib.dump(feature_scaler, scaler_path)
    
    print(f"✅ Feature scaler created and saved to: {scaler_path}")
    print(f"   Scaler fitted with {n_samples} samples of {n_features} features")
    print(f"   Mean: {feature_scaler.mean_[:5]}... (showing first 5)")
    print(f"   Scale: {feature_scaler.scale_[:5]}... (showing first 5)")
    
    return scaler_path

if __name__ == "__main__":
    try:
        create_feature_scaler()
        print("🎉 Feature scaler creation completed successfully!")
    except Exception as e:
        print(f"❌ Error creating feature scaler: {e}")
        raise