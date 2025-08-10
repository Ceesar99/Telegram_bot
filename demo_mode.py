#!/usr/bin/env python3
"""
Enhanced Demo Mode for Trading Bot
This script runs the trading bot in demo mode with completely simulated market data
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import time

# Add the workspace to Python path
sys.path.append('/workspace')

from config import TELEGRAM_USER_ID
from telegram_bot import TradingBot

def setup_logging():
    """Setup logging for demo mode"""
    # Create logs directory if it doesn't exist
    os.makedirs('/workspace/logs', exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('/workspace/logs/demo_mode.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def create_demo_data():
    """Create simulated market data for demo purposes"""
    logger = logging.getLogger('DemoMode')

    try:
        # Create data directory if it doesn't exist
        data_dir = "/workspace/data"
        os.makedirs(data_dir, exist_ok=True)

        # Generate simulated OHLCV data for EUR/USD
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='1min')
        n_periods = len(dates)

        # Simulate realistic price movements
        np.random.seed(42)  # For reproducible results

        # Start with base price
        base_price = 1.1000

        # Generate price changes with some trend and volatility
        price_changes = np.random.normal(0, 0.0001, n_periods)  # Small random changes
        trend = np.linspace(0, 0.001, n_periods)  # Slight upward trend

        # Add some volatility clusters
        volatility = np.random.exponential(0.0001, n_periods)
        price_changes += volatility * np.random.normal(0, 1, n_periods)

        # Apply trend
        price_changes += trend

        # Calculate prices
        prices = base_price + np.cumsum(price_changes)

        # Generate OHLCV data
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            # Simulate OHLC from close price
            volatility_factor = np.random.uniform(0.0001, 0.0005)

            high = price * (1 + np.random.uniform(0, volatility_factor))
            low = price * (1 - np.random.uniform(0, volatility_factor))
            open_price = price * (1 + np.random.uniform(-volatility_factor/2, volatility_factor/2))
            close_price = price
            volume = np.random.uniform(1000, 10000)

            data.append({
                'timestamp': date,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume
            })

        # Create DataFrame
        df = pd.DataFrame(data)

        # Save to CSV
        csv_path = os.path.join(data_dir, "demo_eurusd_data.csv")
        df.to_csv(csv_path, index=False)

        logger.info(f"Demo data created with {len(df)} records")
        logger.info(f"Data saved to {csv_path}")

        return df

    except Exception as e:
        logger.error(f"Error creating demo data: {e}")
        return None

def create_demo_models():
    """Create demo LSTM model and scaler if they don't exist"""
    logger = logging.getLogger('DemoMode')
    
    try:
        models_dir = "/workspace/models"
        os.makedirs(models_dir, exist_ok=True)
        
        # Check if model already exists
        model_path = os.path.join(models_dir, "lstm_trading_model.h5")
        scaler_path = os.path.join(models_dir, "price_scaler.pkl")
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            logger.info("Demo models already exist, skipping creation")
            return True
        
        logger.info("Creating demo LSTM model and scaler...")
        
        # Import required libraries
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam
        import joblib
        from sklearn.preprocessing import MinMaxScaler
        
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
        model.save(model_path)
        logger.info(f"Demo model saved to {model_path}")

        # Create a simple scaler
        dummy_data = np.random.randn(100, 20)
        scaler = MinMaxScaler()
        scaler.fit(dummy_data)

        joblib.dump(scaler, scaler_path)
        logger.info(f"Demo scaler saved to {scaler_path}")

        return True

    except Exception as e:
        logger.error(f"Error creating demo models: {e}")
        return False

async def run_demo_bot():
    """Run the trading bot in demo mode"""
    logger = setup_logging()

    try:
        logger.info("🚀 Starting Trading Bot in ENHANCED DEMO MODE")
        logger.info("📊 Using completely simulated market data")
        logger.info("🤖 Telegram bot will be available for testing")
        logger.info("🔧 Bypassing all external API dependencies")

        demo_data = create_demo_data()
        if demo_data is None:
            logger.error("Failed to create demo data")
            return

        demo_models = create_demo_models()
        if not demo_models:
            logger.error("Failed to create demo models")
            return

        bot = TradingBot()
        bot.bot_status['demo_mode'] = True
        bot.bot_status['demo_data_available'] = True

        logger.info("✅ Enhanced demo mode initialized successfully")
        logger.info("📱 Send /start to your Telegram bot to begin testing")
        logger.info("🎯 Bot will use simulated data for all operations")
        logger.info("🚫 No external market data will be fetched")

        # Check if we're already in an event loop
        try:
            loop = asyncio.get_running_loop()
            logger.info("Event loop already running, using existing loop")
            # If we're already in an event loop, just run the bot directly
            result = bot.run()
            if asyncio.iscoroutine(result):
                await result
            elif asyncio.isfuture(result):
                await result
        except RuntimeError:
            logger.info("No event loop running, creating new one")
            # If no event loop is running, we can use asyncio.run
            await bot.run()

    except KeyboardInterrupt:
        logger.info("🛑 Demo mode stopped by user")
    except Exception as e:
        logger.error(f"❌ Error in demo mode: {e}")
        raise

def main():
    """Main entry point that handles event loop properly"""
    try:
        # Check if we're already in an event loop
        try:
            loop = asyncio.get_running_loop()
            print("Event loop already running, cannot use asyncio.run()")
            print("Please run this script from a context where no event loop is active")
            return
        except RuntimeError:
            # No event loop running, safe to use asyncio.run
            asyncio.run(run_demo_bot())
    except KeyboardInterrupt:
        print("\n🛑 Demo mode stopped by user")
    except Exception as e:
        print(f"❌ Error in demo mode: {e}")
        raise

if __name__ == "__main__":
    main()