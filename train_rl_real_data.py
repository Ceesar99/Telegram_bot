#!/usr/bin/env python3
"""
🤖 REINFORCEMENT LEARNING AGENT TRAINING WITH REAL MARKET DATA
Train advanced RL agent for real-time trading using PPO and DQN algorithms
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
from datetime import datetime
import gymnasium as gym
from gymnasium import spaces
import warnings
warnings.filterwarnings('ignore')

# RL Libraries
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import torch

# Add project path
sys.path.append('/workspace')

class TradingEnvironment(gym.Env):
    """Advanced trading environment for RL agents"""
    
    def __init__(self, data, initial_balance=10000, transaction_cost=0.001, max_position=1.0):
        super(TradingEnvironment, self).__init__()
        
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        
        # Environment state
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0.0  # -1 to 1 (short to long)
        self.total_value = initial_balance
        self.trades_count = 0
        self.winning_trades = 0
        
        # Feature columns (exclude non-feature columns)
        exclude_cols = ['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume', 
                       'date', 'dividends', 'stock splits', 'future_returns', 'returns']
        self.feature_columns = [col for col in data.columns if col not in exclude_cols]
        
        # Action space: 0=hold, 1=buy, 2=sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space (normalized features + portfolio state)
        n_features = len(self.feature_columns)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(n_features + 4,), dtype=np.float32
        )
        
        # Initialize data
        self._prepare_data()
        
    def _prepare_data(self):
        """Prepare and normalize data"""
        # Fill NaN values
        self.data = self.data.ffill().fillna(0)
        
        # Normalize features
        for col in self.feature_columns:
            if col in self.data.columns:
                self.data[col] = (self.data[col] - self.data[col].mean()) / (self.data[col].std() + 1e-8)
        
        # Calculate returns for reward calculation
        self.data['price_change'] = self.data['close'].pct_change().fillna(0)
        
    def reset(self, seed=None, options=None):
        """Reset environment"""
        super().reset(seed=seed)
        self.current_step = 50  # Start after some history
        self.balance = self.initial_balance
        self.position = 0.0
        self.total_value = self.initial_balance
        self.trades_count = 0
        self.winning_trades = 0
        
        return self._get_observation(), {}
        
    def _get_observation(self):
        """Get current observation"""
        if self.current_step >= len(self.data):
            self.current_step = len(self.data) - 1
            
        # Get features
        features = []
        for col in self.feature_columns:
            if col in self.data.columns:
                features.append(self.data.iloc[self.current_step][col])
            else:
                features.append(0.0)
        
        # Add portfolio state
        portfolio_state = [
            self.position,  # Current position
            self.balance / self.initial_balance,  # Normalized balance
            self.total_value / self.initial_balance,  # Normalized total value
            self.trades_count / 100.0  # Normalized trade count
        ]
        
        obs = np.array(features + portfolio_state, dtype=np.float32)
        return obs
        
    def step(self, action):
        """Execute action and return next state, reward, done, info"""
        if self.current_step >= len(self.data) - 1:
            return self._get_observation(), 0, True, {}
            
        # Current and next prices
        current_price = self.data.iloc[self.current_step]['close']
        next_price = self.data.iloc[self.current_step + 1]['close']
        
        # Calculate price change
        price_change = (next_price - current_price) / current_price
        
        # Execute action
        reward = self._execute_action(action, price_change, current_price)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = (self.current_step >= len(self.data) - 1) or (self.total_value <= 0.1 * self.initial_balance)
        
        # Calculate total value
        self.total_value = self.balance + (self.position * self.balance * (1 + price_change))
        
        info = {
            'total_value': self.total_value,
            'balance': self.balance,
            'position': self.position,
            'trades_count': self.trades_count,
            'win_rate': self.winning_trades / max(1, self.trades_count)
        }
        
        return self._get_observation(), reward, done, info
        
    def _execute_action(self, action, price_change, current_price):
        """Execute trading action and calculate reward"""
        reward = 0
        
        # Action mapping: 0=hold, 1=buy, 2=sell
        if action == 1:  # Buy
            if self.position < self.max_position:
                # Calculate transaction cost
                transaction_cost = self.balance * self.transaction_cost
                new_position = min(self.max_position, self.position + 0.1)
                position_change = new_position - self.position
                
                # Update position and balance
                if position_change > 0 and self.balance > transaction_cost:
                    self.position = new_position
                    self.balance -= transaction_cost
                    self.trades_count += 1
                    
                    # Reward based on future price movement
                    reward = position_change * price_change * 100
                    if price_change > 0:
                        self.winning_trades += 1
                        
        elif action == 2:  # Sell
            if self.position > -self.max_position:
                # Calculate transaction cost
                transaction_cost = self.balance * self.transaction_cost
                new_position = max(-self.max_position, self.position - 0.1)
                position_change = self.position - new_position
                
                # Update position and balance
                if position_change > 0 and self.balance > transaction_cost:
                    self.position = new_position
                    self.balance -= transaction_cost
                    self.trades_count += 1
                    
                    # Reward based on future price movement (inverse for short)
                    reward = position_change * (-price_change) * 100
                    if price_change < 0:
                        self.winning_trades += 1
                        
        # Holding position reward
        if action == 0:  # Hold
            reward = abs(self.position) * price_change * 50  # Reward for holding profitable positions
            
        # Risk penalty for extreme positions
        risk_penalty = abs(self.position) * 0.1
        reward -= risk_penalty
        
        # Drawdown penalty
        if self.total_value < 0.8 * self.initial_balance:
            reward -= 5
            
        return reward

class RLTrainingCallback(BaseCallback):
    """Custom callback for RL training monitoring"""
    
    def __init__(self, verbose=0):
        super(RLTrainingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_values = []
        
    def _on_step(self) -> bool:
        # Log training progress
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]
            if 'total_value' in info:
                self.episode_values.append(info['total_value'])
                
        return True

class RealDataRLTrainer:
    """Train RL agents with real market data"""
    
    def __init__(self):
        self.setup_logging()
        self.models = {}
        self.environments = {}
        
    def setup_logging(self):
        """Setup logging"""
        log_dir = "/workspace/logs"
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{log_dir}/rl_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('RLTrainer')
        
    def load_and_prepare_data(self, data_file):
        """Load and prepare real market data for RL training"""
        self.logger.info(f"📊 Loading real market data from {data_file}")
        
        data = pd.read_csv(data_file)
        self.logger.info(f"✅ Loaded {len(data):,} records")
        
        # Convert datetime
        if 'datetime' in data.columns:
            data['datetime'] = pd.to_datetime(data['datetime'])
        else:
            data['datetime'] = pd.to_datetime(data['date'])
            
        # Sort by datetime and symbol
        data = data.sort_values(['symbol', 'datetime']).reset_index(drop=True)
        
        # Focus on one symbol for RL training (better convergence)
        symbols = data['symbol'].unique()
        main_symbol = symbols[0] if len(symbols) > 0 else 'AAPL'
        data = data[data['symbol'] == main_symbol].copy()
        
        self.logger.info(f"📈 Using {len(data):,} records from {main_symbol}")
        
        # Add technical features
        data = self._add_technical_features(data)
        
        return data
        
    def _add_technical_features(self, data):
        """Add technical analysis features"""
        df = data.copy()
        
        # Price features
        df['returns'] = df['close'].pct_change()
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'price_sma_ratio_{period}'] = df['close'] / df[f'sma_{period}']
            
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Volume features
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Volatility
        df['volatility'] = df['returns'].rolling(20).std()
        
        # Drop rows with NaN values
        df = df.dropna()
        
        self.logger.info(f"✅ Added technical features, final shape: {df.shape}")
        return df
        
    def create_training_environment(self, data):
        """Create trading environment for training"""
        # Split data for training (80%) and validation (20%)
        split_idx = int(len(data) * 0.8)
        train_data = data[:split_idx].copy()
        val_data = data[split_idx:].copy()
        
        # Create environments
        train_env = TradingEnvironment(train_data)
        val_env = TradingEnvironment(val_data)
        
        self.logger.info(f"📊 Training data: {len(train_data)} records")
        self.logger.info(f"📊 Validation data: {len(val_data)} records")
        
        return train_env, val_env
        
    def train_ppo_agent(self, train_env, val_env):
        """Train PPO agent"""
        self.logger.info("🚀 Training PPO agent...")
        
        # Wrap environments
        train_env = Monitor(train_env)
        val_env = Monitor(val_env)
        
        # Create PPO model
        model = PPO(
            'MlpPolicy',
            train_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1,
            device='cpu'  # Use CPU for stability
        )
        
        # Training callback
        callback = RLTrainingCallback()
        
        # Train the model (reduced for demo)
        total_timesteps = 20000
        self.logger.info(f"🎯 Training for {total_timesteps:,} timesteps...")
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )
        
        # Save model
        model_dir = "/workspace/models/production/rl"
        os.makedirs(model_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = f"{model_dir}/ppo_agent_{timestamp}.zip"
        model.save(model_path)
        
        self.models['ppo'] = model
        self.logger.info(f"✅ PPO agent saved to {model_path}")
        
        # Evaluate model
        self._evaluate_agent(model, val_env, "PPO")
        
        return model
        
    def train_dqn_agent(self, train_env, val_env):
        """Train DQN agent"""
        self.logger.info("🎯 Training DQN agent...")
        
        # Wrap environments
        train_env = Monitor(train_env)
        val_env = Monitor(val_env)
        
        # Create DQN model
        model = DQN(
            'MlpPolicy',
            train_env,
            learning_rate=1e-3,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=32,
            tau=1.0,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1000,
            exploration_fraction=0.1,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            verbose=1,
            device='cpu'
        )
        
        # Training callback
        callback = RLTrainingCallback()
        
        # Train the model (reduced for demo)
        total_timesteps = 15000
        self.logger.info(f"🎯 Training for {total_timesteps:,} timesteps...")
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )
        
        # Save model
        model_dir = "/workspace/models/production/rl"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = f"{model_dir}/dqn_agent_{timestamp}.zip"
        model.save(model_path)
        
        self.models['dqn'] = model
        self.logger.info(f"✅ DQN agent saved to {model_path}")
        
        # Evaluate model
        self._evaluate_agent(model, val_env, "DQN")
        
        return model
        
    def _evaluate_agent(self, model, env, agent_name):
        """Evaluate trained agent"""
        self.logger.info(f"📊 Evaluating {agent_name} agent...")
        
        obs = env.reset()
        total_reward = 0
        step_count = 0
        episode_values = []
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            if 'total_value' in info:
                episode_values.append(info['total_value'])
            
            if done:
                break
                
        final_value = episode_values[-1] if episode_values else 10000
        total_return = (final_value - 10000) / 10000 * 100
        
        self.logger.info(f"✅ {agent_name} Evaluation Results:")
        self.logger.info(f"   Total Reward: {total_reward:.2f}")
        self.logger.info(f"   Total Return: {total_return:.2f}%")
        self.logger.info(f"   Final Value: ${final_value:.2f}")
        self.logger.info(f"   Steps: {step_count}")
        
        return {
            'total_reward': total_reward,
            'total_return': total_return,
            'final_value': final_value,
            'steps': step_count
        }
        
    def train_all_agents(self, data_file):
        """Train all RL agents"""
        start_time = datetime.now()
        self.logger.info("🚀 Starting RL agents training with real data...")
        
        # Load and prepare data
        data = self.load_and_prepare_data(data_file)
        
        # Create environments
        train_env, val_env = self.create_training_environment(data)
        
        # Train agents
        ppo_model = self.train_ppo_agent(train_env, val_env)
        dqn_model = self.train_dqn_agent(train_env, val_env)
        
        training_time = datetime.now() - start_time
        self.logger.info(f"🎉 RL TRAINING COMPLETED in {training_time}")
        
        print("\n" + "="*80)
        print("🤖 REINFORCEMENT LEARNING AGENTS PERFORMANCE")
        print("="*80)
        print(f"⏱️  Total Training Time: {training_time}")
        print("💾 Models saved to: /workspace/models/production/rl/")
        print("✅ PPO and DQN agents trained successfully!")

if __name__ == "__main__":
    trainer = RealDataRLTrainer()
    
    # Use the real market data we collected
    data_files = [
        "/workspace/data/real_training_data/market_data_7day.csv",
        "/workspace/data/real_training_data/real_market_data_20250816_094716.csv"
    ]
    
    # Find existing data file
    data_file = None
    for file in data_files:
        if os.path.exists(file):
            data_file = file
            break
    
    if data_file:
        trainer.train_all_agents(data_file)
    else:
        print("❌ No real market data file found!")