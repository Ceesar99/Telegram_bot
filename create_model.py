#!/usr/bin/env python3
"""
Create Basic LSTM Model for Trading Bot
This script creates a simple LSTM model to avoid the missing model error
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os
import logging

def setup_logging():
    """Setup logging for the model creation process"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def create_basic_model():
    """Create a basic LSTM model for trading signals"""
    logger = setup_logging()
    
    try:
        # Create models directory if it doesn't exist
        models_dir = "/workspace/models"
        os.makedirs(models_dir, exist_ok=True)
        
        logger.info("Creating basic LSTM model...")
        
        # Create a simple LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(60, 20)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Save the model
        model_path = os.path.join(models_dir, "lstm_trading_model.h5")
        model.save(model_path)
        
        logger.info(f"Model saved successfully to {model_path}")
        
        # Create a simple scaler file as well
        import joblib
        scaler_path = os.path.join(models_dir, "price_scaler.pkl")
        
        # Create dummy scaler data
        dummy_data = np.random.randn(100, 20)
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaler.fit(dummy_data)
        
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        return False

if __name__ == "__main__":
    success = create_basic_model()
    if success:
        print("✅ Basic LSTM model created successfully!")
    else:
        print("❌ Failed to create model")