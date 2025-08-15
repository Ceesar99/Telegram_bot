import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque, namedtuple
import random
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import gym
from gym import spaces
import pandas as pd
from datetime import datetime, timedelta
import pickle
import warnings
warnings.filterwarnings('ignore')

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

@dataclass
class TradingAction:
    """Trading action representation"""
    action_type: str  # 'BUY', 'SELL', 'HOLD'
    position_size: float
    confidence: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

@dataclass
class MarketState:
    """Comprehensive market state representation"""
    prices: np.ndarray
    technical_indicators: np.ndarray
    volume_data: np.ndarray
    sentiment_scores: np.ndarray
    economic_indicators: np.ndarray
    portfolio_state: Dict[str, float]
    market_regime: str
    timestamp: datetime

class TradingEnvironment(gym.Env):
    """
    Advanced trading environment for reinforcement learning
    Simulates realistic market conditions with transaction costs, slippage, and market impact
    """
    
    def __init__(self, 
                 price_data: np.ndarray,
                 feature_data: np.ndarray,
                 initial_balance: float = 10000,
                 transaction_cost: float = 0.001,
                 max_position_size: float = 0.1,
                 lookback_window: int = 100,
                 slippage_bps: float = 5.0,
                 exposure_penalty: float = 0.001):
        super().__init__()
        
        self.price_data = price_data
        self.feature_data = feature_data
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position_size = max_position_size
        self.lookback_window = lookback_window
        self.slippage_bps = slippage_bps
        self.exposure_penalty = exposure_penalty
        
        # State space: features + portfolio state
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(feature_data.shape[1] + 5,),  # +5 for portfolio state
            dtype=np.float32
        )
        
        # Action space: [action_type, position_size]
        # action_type: 0=HOLD, 1=BUY, 2=SELL
        # position_size: 0.0 to 1.0 (as fraction of max position)
        self.action_space = spaces.Box(
            low=np.array([0, 0]), 
            high=np.array([2, 1]), 
            dtype=np.float32
        )
        
        # Environment state
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0.0  # Current position (-1 to 1, negative = short)
        self.entry_price = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_balance = initial_balance
        
        # Performance tracking
        self.trade_history = []
        self.balance_history = [initial_balance]
        self.position_history = [0.0]
        
        self.logger = logging.getLogger('TradingEnvironment')
        
    def _estimate_slippage(self, notional: float) -> float:
        """Estimate slippage cost in price terms using basis points"""
        return notional * (self.slippage_bps / 10000.0)
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_balance = self.initial_balance
        
        self.trade_history = []
        self.balance_history = [self.initial_balance]
        self.position_history = [0.0]
        
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step in the environment"""
        # Parse action
        action_type = int(np.clip(action[0], 0, 2))
        position_size = float(np.clip(action[1], 0, 1))
        
        # Calculate reward
        reward = self._calculate_reward(action_type, position_size)
        
        # Execute trade
        self._execute_trade(action_type, position_size)
        
        # Update step
        self.current_step += 1
        
        # Check if episode is done
        done = (self.current_step >= len(self.price_data) - 1 or 
                self.balance <= 0.1 * self.initial_balance)
        
        # Get next observation
        next_obs = self._get_observation() if not done else np.zeros(self.observation_space.shape)
        
        # Info dictionary
        info = {
            'balance': self.balance,
            'position': self.position,
            'total_trades': self.total_trades,
            'win_rate': self.winning_trades / max(1, self.total_trades),
            'total_pnl': self.total_pnl,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self._calculate_sharpe_ratio()
        }
        
        return next_obs, reward, done, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation (state)"""
        if self.current_step >= len(self.feature_data):
            return np.zeros(self.observation_space.shape)
        
        # Market features
        market_features = self.feature_data[self.current_step]
        
        # Portfolio state
        current_price = self.price_data[self.current_step]
        unrealized_pnl = 0.0
        if self.position != 0:
            unrealized_pnl = self.position * (current_price - self.entry_price) / self.entry_price
        
        portfolio_state = np.array([
            self.balance / self.initial_balance,  # Normalized balance
            self.position,  # Current position
            unrealized_pnl,  # Unrealized P&L
            self.total_trades / 1000.0,  # Normalized trade count
            self.max_drawdown  # Max drawdown
        ])
        
        # Combine market and portfolio features
        observation = np.concatenate([market_features, portfolio_state])
        
        return observation.astype(np.float32)
    
    def _calculate_reward(self, action_type: int, position_size: float) -> float:
        """Calculate reward with broker-like costs and exposure penalties"""
        if self.current_step >= len(self.price_data) - 1:
            return 0.0
        
        current_price = self.price_data[self.current_step]
        next_price = self.price_data[self.current_step + 1]
        price_change = (next_price - current_price) / current_price
        
        # Base reward from price movement, scaled by position
        reward = 0.0
        if action_type == 1:  # BUY
            reward = price_change * position_size
        elif action_type == 2:  # SELL
            reward = -price_change * position_size
        else:  # HOLD
            reward = 0.0
        
        # Transaction costs (proportional to notional)
        if action_type != 0:
            reward -= self.transaction_cost * position_size
            # Slippage cost
            notional = position_size * self.max_position_size
            reward -= self._estimate_slippage(notional)
        
        # Penalize excessive trading and exposure
        if hasattr(self, 'prev_position'):
            position_change = abs(self.position - self.prev_position)
            reward -= 0.001 * position_change
        # Exposure penalty grows with absolute position
        reward -= self.exposure_penalty * abs(self.position)
        
        # Risk-adjusted reward (penalize volatility of returns)
        if len(self.balance_history) > 10:
            recent_returns = np.diff(self.balance_history[-10:]) / self.balance_history[-11:-1]
            volatility = np.std(recent_returns)
            reward -= 0.1 * volatility
        
        return reward * 100  # Scale reward
    
    def _execute_trade(self, action_type: int, position_size: float):
        """Execute the trading action"""
        current_price = self.price_data[self.current_step]
        self.prev_position = self.position
        
        if action_type == 1:  # BUY
            if self.position <= 0:  # Can buy if no long position or have short position
                trade_size = position_size * self.max_position_size
                self.position = min(self.max_position_size, self.position + trade_size)
                self.entry_price = current_price
                self._record_trade('BUY', trade_size, current_price)
                
        elif action_type == 2:  # SELL
            if self.position >= 0:  # Can sell if no short position or have long position
                trade_size = position_size * self.max_position_size
                self.position = max(-self.max_position_size, self.position - trade_size)
                self.entry_price = current_price
                self._record_trade('SELL', trade_size, current_price)
        
        # Update balance based on unrealized P&L
        if self.position != 0:
            unrealized_pnl = self.position * (current_price - self.entry_price) / self.entry_price
            current_balance = self.initial_balance * (1 + self.total_pnl + unrealized_pnl)
        else:
            current_balance = self.initial_balance * (1 + self.total_pnl)
        
        self.balance = current_balance
        self.balance_history.append(self.balance)
        self.position_history.append(self.position)
        
        # Update peak balance and drawdown
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        
        drawdown = (self.peak_balance - self.balance) / self.peak_balance
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
    
    def _record_trade(self, action: str, size: float, price: float):
        """Record trade for analysis"""
        self.total_trades += 1
        
        trade_record = {
            'timestamp': self.current_step,
            'action': action,
            'size': size,
            'price': price,
            'balance': self.balance
        }
        
        self.trade_history.append(trade_record)
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio of the strategy"""
        if len(self.balance_history) < 2:
            return 0.0
        
        returns = np.diff(self.balance_history) / self.balance_history[:-1]
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        return np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized

class DQNNetwork(nn.Module):
    """Deep Q-Network for trading decisions"""
    
    def __init__(self, state_size: int, action_size: int, hidden_sizes: List[int] = [256, 256, 128]):
        super().__init__()
        
        layers = []
        prev_size = state_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.BatchNorm1d(hidden_size)
            ])
            prev_size = hidden_size
        
        # Remove the last BatchNorm for the final layer
        layers = layers[:-1]
        
        # Output layers for action type and position size
        self.feature_layers = nn.Sequential(*layers)
        
        # Action type head (3 actions: HOLD, BUY, SELL)
        self.action_type_head = nn.Sequential(
            nn.Linear(prev_size, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        
        # Position size head (continuous value 0-1)
        self.position_size_head = nn.Sequential(
            nn.Linear(prev_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Value head for advantage estimation
        self.value_head = nn.Sequential(
            nn.Linear(prev_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        features = self.feature_layers(x)
        
        action_type_q = self.action_type_head(features)
        position_size = self.position_size_head(features)
        value = self.value_head(features)
        
        return {
            'action_type_q': action_type_q,
            'position_size': position_size,
            'value': value
        }

class ReplayBuffer:
    """Experience replay buffer for DQN"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience: Experience):
        """Add experience to buffer"""
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample batch of experiences"""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class PPOAgent:
    """Proximal Policy Optimization agent for trading"""
    
    def __init__(self, state_size: int, action_size: int, lr: float = 3e-4, device: str = 'cpu'):
        self.device = torch.device(device)
        self.state_size = state_size
        self.action_size = action_size
        
        # Actor-Critic network
        self.policy_net = DQNNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # PPO parameters
        self.gamma = 0.99
        self.eps_clip = 0.2
        self.k_epochs = 4
        self.c1 = 0.5  # Value loss coefficient
        self.c2 = 0.01  # Entropy coefficient
        
        # Training statistics
        self.training_stats = {
            'policy_losses': [],
            'value_losses': [],
            'total_rewards': [],
            'episode_lengths': []
        }
        
        self.logger = logging.getLogger('PPOAgent')
    
    def get_action(self, state: np.ndarray, training: bool = True) -> Tuple[np.ndarray, Dict]:
        """Get action from policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.policy_net(state_tensor)
            action_type_q = outputs['action_type_q']
            position_size = outputs['position_size']
            
            # Get action type using epsilon-greedy for exploration
            if training and random.random() < 0.1:  # 10% random actions during training
                action_type = random.randint(0, 2)
            else:
                action_type = torch.argmax(action_type_q, dim=1).item()
            
            # Get position size
            pos_size = position_size.item()
            
            # Add noise for exploration during training
            if training:
                pos_size += np.random.normal(0, 0.1)
                pos_size = np.clip(pos_size, 0, 1)
        
        action = np.array([action_type, pos_size])
        
        action_info = {
            'action_type_q_values': action_type_q.cpu().numpy(),
            'position_size_raw': position_size.item(),
            'value_estimate': outputs['value'].item()
        }
        
        return action, action_info
    
    def update_policy(self, states: List[np.ndarray], actions: List[np.ndarray], 
                     rewards: List[float], next_states: List[np.ndarray], 
                     dones: List[bool]) -> Dict[str, float]:
        """Update policy using PPO"""
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        actions_tensor = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)
        
        # Calculate returns and advantages
        returns = self._calculate_returns(rewards_tensor, dones_tensor)
        
        # Get current values
        with torch.no_grad():
            current_outputs = self.policy_net(states_tensor)
            current_values = current_outputs['value'].squeeze()
            advantages = returns - current_values
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update policy for k epochs
        total_policy_loss = 0
        total_value_loss = 0
        
        for _ in range(self.k_epochs):
            # Forward pass
            outputs = self.policy_net(states_tensor)
            action_type_q = outputs['action_type_q']
            position_size_pred = outputs['position_size'].squeeze()
            values = outputs['value'].squeeze()
            
            # Action type loss (using advantage as weight)
            action_types = actions_tensor[:, 0].long()
            action_type_loss = F.cross_entropy(action_type_q, action_types, reduction='none')
            action_type_loss = (action_type_loss * advantages.detach()).mean()
            
            # Position size loss (MSE with advantage weighting)
            position_size_target = actions_tensor[:, 1]
            position_size_loss = F.mse_loss(position_size_pred, position_size_target, reduction='none')
            position_size_loss = (position_size_loss * advantages.detach().abs()).mean()
            
            # Value loss
            value_loss = F.mse_loss(values, returns)
            
            # Entropy for exploration
            action_probs = F.softmax(action_type_q, dim=1)
            entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(1).mean()
            
            # Total loss
            total_loss = action_type_loss + position_size_loss + self.c1 * value_loss - self.c2 * entropy
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=0.5)
            self.optimizer.step()
            
            total_policy_loss += (action_type_loss + position_size_loss).item()
            total_value_loss += value_loss.item()
        
        return {
            'policy_loss': total_policy_loss / self.k_epochs,
            'value_loss': total_value_loss / self.k_epochs,
            'entropy': entropy.item()
        }
    
    def _calculate_returns(self, rewards: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """Calculate discounted returns"""
        returns = torch.zeros_like(rewards)
        running_return = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_return = 0
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return
        
        return returns
    
    def save_model(self, filepath: str):
        """Save model to file"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load model from file"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint['training_stats']

class RLTradingEngine:
    """Main reinforcement learning trading engine"""
    
    def __init__(self, 
                 price_data: np.ndarray,
                 feature_data: np.ndarray,
                 initial_balance: float = 10000,
                 paper_trading_only: bool = True):
        
        self.logger = logging.getLogger('RLTradingEngine')
        self.paper_trading_only = paper_trading_only
        
        # Create environment
        self.env = TradingEnvironment(
            price_data=price_data,
            feature_data=feature_data,
            initial_balance=initial_balance
        )
        
        # Create agent
        self.agent = PPOAgent(
            state_size=self.env.observation_space.shape[0],
            action_size=2,  # [action_type, position_size]
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Training parameters
        self.batch_size = 64
        self.update_frequency = 100
        self.training_episodes = 0
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.best_performance = {
            'reward': float('-inf'),
            'sharpe_ratio': float('-inf'),
            'win_rate': 0.0
        }
        
        # Replay/policy stats persistence path
        self.persistence_prefix = 'rl_training'
    
    def _persist_training_state(self, suffix: str = ""):
        try:
            payload = {
                'agent_stats': self.agent.training_stats,
                'episodes': self.training_episodes,
                'episode_rewards': self.episode_rewards[-100:],
                'episode_lengths': self.episode_lengths[-100:]
            }
            with open(f"{self.persistence_prefix}_stats{suffix}.pkl", 'wb') as f:
                pickle.dump(payload, f)
        except Exception as e:
            self.logger.warning(f"Failed to persist training state: {e}")

    def train(self, episodes: int = 1000, save_frequency: int = 100) -> Dict[str, List[float]]:
        """Train the RL agent"""
        self.logger.info(f"Starting training for {episodes} episodes...")
        
        # Training loop
        for episode in range(episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            # Collect experience for one episode
            states, actions, rewards, next_states, dones = [], [], [], [], []
            
            while True:
                # Get action from agent
                action, action_info = self.agent.get_action(state, training=True)
                
                # Take step in environment
                next_state, reward, done, info = self.env.step(action)
                
                # Store experience
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)
                
                episode_reward += reward
                episode_length += 1
                
                state = next_state
                
                if done:
                    break
            
            # Update policy
            if len(states) > 0:
                update_stats = self.agent.update_policy(states, actions, rewards, next_states, dones)
                self.agent.training_stats['policy_losses'].append(update_stats['policy_loss'])
                self.agent.training_stats['value_losses'].append(update_stats['value_loss'])
            
            # Record episode statistics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.agent.training_stats['total_rewards'].append(episode_reward)
            self.agent.training_stats['episode_lengths'].append(episode_length)
            
            # Check for best performance
            if info['sharpe_ratio'] > self.best_performance['sharpe_ratio']:
                self.best_performance.update({
                    'reward': episode_reward,
                    'sharpe_ratio': info['sharpe_ratio'],
                    'win_rate': info['win_rate']
                })
            
            # Logging
            if episode % 50 == 0:
                avg_reward = np.mean(self.episode_rewards[-50:])
                avg_length = np.mean(self.episode_lengths[-50:])
                
                self.logger.info(f"Episode {episode}: "
                               f"Avg Reward={avg_reward:.2f}, "
                               f"Avg Length={avg_length:.1f}, "
                               f"Sharpe={info['sharpe_ratio']:.3f}, "
                               f"Win Rate={info['win_rate']:.3f}")
                # Persist stats snapshot
                self._persist_training_state(suffix=f"_{episode}")
            
            # Save model periodically
            if episode % save_frequency == 0 and episode > 0:
                self.save_model(f'rl_model_episode_{episode}.pth')
                self._persist_training_state(suffix=f"_{episode}")
        
        self.training_episodes += episodes
        return self.agent.training_stats
    
    def evaluate(self, episodes: int = 10) -> Dict[str, float]:
        """Evaluate the trained agent"""
        self.logger.info(f"Evaluating agent for {episodes} episodes...")
        
        if not self.paper_trading_only:
            self.logger.warning("Kill switch disabled: live trading not permitted in RL engine. Forcing paper only.")
            self.paper_trading_only = True
        
        evaluation_results = {
            'total_returns': [],
            'sharpe_ratios': [],
            'win_rates': [],
            'max_drawdowns': [],
            'total_trades': []
        }
        
        for episode in range(episodes):
            state = self.env.reset()
            
            while True:
                action, _ = self.agent.get_action(state, training=False)
                next_state, reward, done, info = self.env.step(action)
                state = next_state
                
                if done:
                    # Record results
                    total_return = (self.env.balance - self.env.initial_balance) / self.env.initial_balance
                    evaluation_results['total_returns'].append(total_return)
                    evaluation_results['sharpe_ratios'].append(info['sharpe_ratio'])
                    evaluation_results['win_rates'].append(info['win_rate'])
                    evaluation_results['max_drawdowns'].append(info['max_drawdown'])
                    evaluation_results['total_trades'].append(info['total_trades'])
                    break
        
        # Calculate summary statistics
        summary = {
            'avg_return': np.mean(evaluation_results['total_returns']),
            'avg_sharpe': np.mean(evaluation_results['sharpe_ratios']),
            'avg_win_rate': np.mean(evaluation_results['win_rates']),
            'avg_max_drawdown': np.mean(evaluation_results['max_drawdowns']),
            'avg_trades': np.mean(evaluation_results['total_trades']),
            'return_std': np.std(evaluation_results['total_returns'])
        }
        
        self.logger.info(f"Evaluation Results: {summary}")
        return summary
    
    def predict(self, current_state: np.ndarray) -> TradingAction:
        """Make trading prediction using trained agent"""
        action, action_info = self.agent.get_action(current_state, training=False)
        
        action_types = ['HOLD', 'BUY', 'SELL']
        action_type = action_types[int(action[0])]
        position_size = float(action[1])
        confidence = float(np.max(action_info['action_type_q_values']))
        
        return TradingAction(
            action_type=action_type,
            position_size=position_size,
            confidence=confidence
        )
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        self.agent.save_model(filepath)
        
        # Also save environment parameters
        env_params = {
            'initial_balance': self.env.initial_balance,
            'transaction_cost': self.env.transaction_cost,
            'max_position_size': self.env.max_position_size,
            'lookback_window': self.env.lookback_window
        }
        
        with open(filepath.replace('.pth', '_env_params.pkl'), 'wb') as f:
            pickle.dump(env_params, f)
        
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        self.agent.load_model(filepath)
        
        # Load environment parameters if available
        env_params_file = filepath.replace('.pth', '_env_params.pkl')
        try:
            with open(env_params_file, 'rb') as f:
                env_params = pickle.load(f)
                # Update environment parameters
                for key, value in env_params.items():
                    if hasattr(self.env, key):
                        setattr(self.env, key, value)
        except FileNotFoundError:
            self.logger.warning("Environment parameters file not found")
        
        self.logger.info(f"Model loaded from {filepath}")
    
    def get_training_report(self) -> Dict[str, Any]:
        """Generate comprehensive training report"""
        return {
            'training_episodes': self.training_episodes,
            'best_performance': self.best_performance,
            'recent_performance': {
                'avg_reward_last_100': np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else 0,
                'avg_length_last_100': np.mean(self.episode_lengths[-100:]) if len(self.episode_lengths) >= 100 else 0
            },
            'training_stats': self.agent.training_stats
        }