#!/usr/bin/env python3
"""
🤖 TRANSFORMER TRAINING SYSTEM - ADVANCED FINANCIAL PREDICTIONS
Implements sophisticated transformer model training for financial markets:
1. Multi-head attention mechanisms
2. Positional encoding for time series
3. Advanced training with large datasets
4. Financial-specific optimizations
"""

import argparse
import sys
import os
import logging
from datetime import datetime, timedelta
import warnings
import traceback
import numpy as np
import pandas as pd
import json
import time
import math
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append('/workspace')

# Import PyTorch components
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset, TensorDataset
    import torch.nn.functional as F
    PYTORCH_AVAILABLE = True
except ImportError:
    print("⚠️ PyTorch not available, using mock implementation")
    PYTORCH_AVAILABLE = False

# Import other components
from data_manager_fixed import DataManager

def setup_transformer_logging():
    """Setup logging for transformer training"""
    log_dir = '/workspace/logs'
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/transformer_training_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger('TransformerTraining')

class FinancialTransformerOptimized(nn.Module):
    """Optimized Financial Transformer for trading predictions"""
    
    def __init__(self, input_dim=24, d_model=256, num_heads=8, num_layers=6, 
                 d_ff=1024, max_length=1000, num_classes=3, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.positional_encoding = self._create_positional_encoding(max_length, d_model)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output layers
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_ff // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff // 2, d_ff // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff // 4, num_classes)
        )
        
        # Confidence estimation head
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_ff // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff // 4, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _create_positional_encoding(self, max_length, d_model):
        """Create sinusoidal positional encoding"""
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Input projection and positional encoding
        x = self.input_projection(x) * math.sqrt(self.d_model)
        
        # Add positional encoding
        if seq_len <= self.positional_encoding.size(1):
            x = x + self.positional_encoding[:, :seq_len, :]
        
        # Pass through transformer
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        x = self.layer_norm(x)
        
        # Use last token for classification
        last_hidden = x[:, -1, :]
        
        # Get predictions
        logits = self.classifier(last_hidden)
        confidence = self.confidence_head(last_hidden)
        
        return {
            'logits': logits,
            'confidence': confidence,
            'hidden_states': x
        }

class FinancialDatasetAdvanced(Dataset):
    """Advanced dataset for financial transformer training"""
    
    def __init__(self, data, sequence_length=60, prediction_horizon=2):
        self.data = data
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        # Prepare features and labels
        self.features, self.labels = self._prepare_data()
    
    def _prepare_data(self):
        """Prepare features and labels for transformer training"""
        features = []
        labels = []
        
        # Group by currency pair
        for pair in self.data['pair'].unique():
            pair_data = self.data[self.data['pair'] == pair].copy()
            
            # Calculate technical indicators
            pair_data = self._calculate_features(pair_data)
            
            # Create sequences
            for i in range(len(pair_data) - self.sequence_length - self.prediction_horizon):
                # Extract features
                feature_cols = ['returns', 'volatility', 'rsi', 'macd', 'bb_upper', 'bb_lower',
                               'sma_5', 'sma_10', 'sma_20', 'ema_12', 'ema_26', 'volume_ratio',
                               'high_low_ratio', 'open_close_ratio', 'price_position', 'momentum_5',
                               'momentum_10', 'atr', 'williams_r', 'cci', 'stoch_k', 'stoch_d',
                               'adx', 'di_plus']
                
                # Ensure we have all features
                available_cols = [col for col in feature_cols if col in pair_data.columns]
                if len(available_cols) < 10:  # Need minimum features
                    continue
                
                # Pad missing features with zeros
                feature_data = np.zeros((self.sequence_length, 24))
                for j, col in enumerate(available_cols[:24]):
                    if j < 24:
                        feature_data[:, j] = pair_data[col].iloc[i:i+self.sequence_length].fillna(0)
                
                # Generate label (future price movement)
                current_price = pair_data['close'].iloc[i + self.sequence_length]
                future_price = pair_data['close'].iloc[i + self.sequence_length + self.prediction_horizon]
                
                price_change = (future_price - current_price) / current_price
                
                # Multi-class classification
                if price_change > 0.0002:  # Strong buy
                    label = 0
                elif price_change < -0.0002:  # Strong sell
                    label = 1
                else:  # Hold
                    label = 2
                
                features.append(feature_data)
                labels.append(label)
        
        return np.array(features), np.array(labels)
    
    def _calculate_features(self, data):
        """Calculate comprehensive technical features"""
        # Price-based features
        data['returns'] = data['close'].pct_change()
        data['volatility'] = data['returns'].rolling(20).std()
        data['high_low_ratio'] = (data['high'] - data['low']) / data['close']
        data['open_close_ratio'] = (data['close'] - data['open']) / data['open']
        data['price_position'] = (data['close'] - data['low']) / (data['high'] - data['low'])
        
        # Moving averages
        data['sma_5'] = data['close'].rolling(5).mean()
        data['sma_10'] = data['close'].rolling(10).mean()
        data['sma_20'] = data['close'].rolling(20).mean()
        data['ema_12'] = data['close'].ewm(span=12).mean()
        data['ema_26'] = data['close'].ewm(span=26).mean()
        
        # Momentum indicators
        data['momentum_5'] = data['close'] / data['close'].shift(5) - 1
        data['momentum_10'] = data['close'] / data['close'].shift(10) - 1
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        data['macd'] = data['ema_12'] - data['ema_26']
        
        # Bollinger Bands
        sma_20 = data['close'].rolling(20).mean()
        std_20 = data['close'].rolling(20).std()
        data['bb_upper'] = sma_20 + (std_20 * 2)
        data['bb_lower'] = sma_20 - (std_20 * 2)
        
        # Volume indicators
        data['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        
        # Additional indicators
        data['atr'] = self._calculate_atr(data)
        data['williams_r'] = self._calculate_williams_r(data)
        data['cci'] = self._calculate_cci(data)
        data['stoch_k'], data['stoch_d'] = self._calculate_stochastic(data)
        data['adx'], data['di_plus'] = self._calculate_adx(data)
        
        # Normalize relative to close price
        price_cols = ['sma_5', 'sma_10', 'sma_20', 'ema_12', 'ema_26', 'bb_upper', 'bb_lower']
        for col in price_cols:
            if col in data.columns:
                data[col] = data[col] / data['close'] - 1
        
        return data.fillna(0)
    
    def _calculate_atr(self, data, period=14):
        """Calculate Average True Range"""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(period).mean() / data['close']
    
    def _calculate_williams_r(self, data, period=14):
        """Calculate Williams %R"""
        highest_high = data['high'].rolling(period).max()
        lowest_low = data['low'].rolling(period).min()
        return -100 * (highest_high - data['close']) / (highest_high - lowest_low)
    
    def _calculate_cci(self, data, period=20):
        """Calculate Commodity Channel Index"""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        sma_tp = typical_price.rolling(period).mean()
        mad = typical_price.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        return (typical_price - sma_tp) / (0.015 * mad)
    
    def _calculate_stochastic(self, data, k_period=14, d_period=3):
        """Calculate Stochastic Oscillator"""
        lowest_low = data['low'].rolling(k_period).min()
        highest_high = data['high'].rolling(k_period).max()
        k_percent = 100 * (data['close'] - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(d_period).mean()
        return k_percent, d_percent
    
    def _calculate_adx(self, data, period=14):
        """Calculate Average Directional Index"""
        plus_dm = data['high'].diff()
        minus_dm = -data['low'].diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = self._calculate_atr(data, 1) * data['close']
        plus_di = 100 * (plus_dm.rolling(period).mean() / tr.rolling(period).mean())
        minus_di = 100 * (minus_dm.rolling(period).mean() / tr.rolling(period).mean())
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        return adx, plus_di
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.LongTensor([self.labels[idx]])[0]

def create_transformer_training_data(samples=50000):
    """Create comprehensive training data for transformer"""
    logger = logging.getLogger('TransformerTraining')
    logger.info(f"Creating transformer training data with {samples:,} samples...")
    
    # Generate sophisticated multi-pair data
    start_date = datetime.now() - timedelta(days=600)
    end_date = datetime.now()
    dates = pd.date_range(start=start_date, end=end_date, freq='1min')[:samples]
    n_samples = len(dates)
    
    np.random.seed(42)
    
    # Currency pairs with sophisticated patterns
    pairs_config = {
        'EUR/USD': {'base': 1.1000, 'vol': 0.00008, 'trend_strength': 0.3},
        'GBP/USD': {'base': 1.3000, 'vol': 0.00012, 'trend_strength': 0.4},
        'USD/JPY': {'base': 110.00, 'vol': 0.00010, 'trend_strength': 0.35},
        'AUD/USD': {'base': 0.7500, 'vol': 0.00015, 'trend_strength': 0.45},
        'USD/CAD': {'base': 1.2500, 'vol': 0.00009, 'trend_strength': 0.25},
        'EUR/GBP': {'base': 0.8500, 'vol': 0.00007, 'trend_strength': 0.2},
        'USD/CHF': {'base': 0.9200, 'vol': 0.00009, 'trend_strength': 0.3}
    }
    
    all_data = []
    
    for pair, config in pairs_config.items():
        logger.info(f"Generating transformer data for {pair}...")
        
        # Advanced market simulation
        returns = np.random.normal(0, config['vol'], n_samples)
        
        # Multi-timeframe patterns
        hours = dates.hour
        days = dates.dayofweek
        
        # Session effects (more sophisticated)
        london_session = ((hours >= 8) & (hours <= 16)).astype(float)
        ny_session = ((hours >= 13) & (hours <= 21)).astype(float)
        asian_session = ((hours >= 21) | (hours <= 6)).astype(float)
        
        session_multiplier = (1.0 + 0.5 * london_session + 0.4 * ny_session + 0.2 * asian_session)
        
        # Weekly patterns
        weekly_multiplier = np.where(days < 5, 1.0, 0.3)  # Weekends quieter
        
        # Market regime simulation (more complex)
        regime_length = 1440 * 5  # 5 days
        n_regimes = n_samples // regime_length + 1
        
        # 6 different market regimes
        regimes = np.random.choice([0, 1, 2, 3, 4, 5], n_regimes, 
                                 p=[0.15, 0.15, 0.25, 0.2, 0.15, 0.1])
        regime_series = np.repeat(regimes, regime_length)[:n_samples]
        
        # Regime characteristics
        regime_effects = {
            0: {'vol_mult': 0.6, 'trend': 0.00004, 'name': 'Strong Bull'},
            1: {'vol_mult': 0.7, 'trend': -0.00003, 'name': 'Strong Bear'},
            2: {'vol_mult': 0.4, 'trend': 0.0, 'name': 'Tight Range'},
            3: {'vol_mult': 1.0, 'trend': 0.00001, 'name': 'Normal Market'},
            4: {'vol_mult': 2.5, 'trend': 0.0, 'name': 'High Volatility'},
            5: {'vol_mult': 4.0, 'trend': -0.00008, 'name': 'Crisis Mode'}
        }
        
        # Generate price series with advanced patterns
        volatility = np.ones(n_samples) * config['vol']
        prices = [config['base']]
        
        # Trend persistence
        trend_momentum = 0
        
        for i in range(1, n_samples):
            regime = regime_series[i]
            regime_effect = regime_effects[regime]
            
            # GARCH-like volatility clustering
            volatility[i] = (0.85 * volatility[i-1] + 
                           0.1 * abs(returns[i-1]) + 
                           0.05 * config['vol'])
            
            # Trend momentum (trends persist)
            if np.random.random() < config['trend_strength']:
                trend_momentum = 0.7 * trend_momentum + 0.3 * regime_effect['trend']
            else:
                trend_momentum *= 0.9
            
            # Combined effects
            vol_factor = (volatility[i] * regime_effect['vol_mult'] * 
                         session_multiplier[i] * weekly_multiplier[i])
            
            total_change = (returns[i] * vol_factor + 
                          regime_effect['trend'] + 
                          trend_momentum)
            
            # News events (rare but impactful)
            if np.random.random() < 0.0005:  # 0.05% chance
                news_impact = np.random.normal(0, 0.003)
                total_change += news_impact
            
            new_price = prices[-1] * (1 + total_change)
            
            # Realistic bounds
            bounds = {
                'EUR/USD': (0.95, 1.25), 'GBP/USD': (1.15, 1.45),
                'USD/JPY': (95, 125), 'AUD/USD': (0.65, 0.85),
                'USD/CAD': (1.15, 1.35), 'EUR/GBP': (0.80, 0.90),
                'USD/CHF': (0.88, 0.96)
            }
            
            min_p, max_p = bounds.get(pair, (new_price * 0.8, new_price * 1.2))
            new_price = max(min_p, min(max_p, new_price))
            prices.append(new_price)
        
        # Create comprehensive OHLCV data
        pair_data = pd.DataFrame({
            'timestamp': dates,
            'pair': pair,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.000005))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.000005))) for p in prices],
            'close': prices,
            'volume': (np.random.randint(500, 2000, n_samples) * 
                      session_multiplier * weekly_multiplier),
            'regime': regime_series,
            'volatility_raw': volatility
        })
        
        # Ensure OHLC consistency
        pair_data['high'] = pair_data[['open', 'close', 'high']].max(axis=1)
        pair_data['low'] = pair_data[['open', 'close', 'low']].min(axis=1)
        
        all_data.append(pair_data)
    
    # Combine all pairs
    combined_data = pd.concat(all_data, ignore_index=True)
    combined_data = combined_data.sort_values('timestamp').reset_index(drop=True)
    
    logger.info(f"Transformer training data created:")
    logger.info(f"  Total samples: {len(combined_data):,}")
    logger.info(f"  Currency pairs: {len(pairs_config)}")
    logger.info(f"  Market regimes: 6 (Bull, Bear, Range, Normal, Volatile, Crisis)")
    
    return combined_data

def train_transformer_model(training_data, epochs=50, batch_size=32):
    """Train transformer model with advanced techniques"""
    logger = logging.getLogger('TransformerTraining')
    
    if not PYTORCH_AVAILABLE:
        logger.error("❌ PyTorch not available, cannot train transformer")
        return {'status': 'failed', 'error': 'PyTorch not available'}
    
    try:
        logger.info("=" * 60)
        logger.info(f"🤖 TRAINING FINANCIAL TRANSFORMER ({epochs} epochs)")
        logger.info("=" * 60)
        
        # Create dataset
        logger.info("Creating transformer dataset...")
        dataset = FinancialDatasetAdvanced(training_data, sequence_length=60)
        
        if len(dataset) < 1000:
            logger.error("❌ Insufficient data for transformer training")
            return {'status': 'failed', 'error': 'Insufficient data'}
        
        # Split data
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        logger.info(f"Dataset created: {len(train_dataset)} train, {len(val_dataset)} val")
        
        # Initialize model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = FinancialTransformerOptimized(
            input_dim=24,
            d_model=128,  # Reduced for faster training
            num_heads=8,
            num_layers=4,  # Reduced for faster training
            d_ff=512,
            dropout=0.1
        ).to(device)
        
        logger.info(f"Model initialized on device: {device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Optimizer and scheduler
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Loss function with class weights (handle imbalanced classes)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_val_loss = float('inf')
        best_val_acc = 0
        training_history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(data)
                loss = criterion(outputs['logits'], target)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Statistics
                train_loss += loss.item()
                _, predicted = torch.max(outputs['logits'].data, 1)
                train_total += target.size(0)
                train_correct += (predicted == target).sum().item()
                
                if batch_idx % 50 == 0:
                    logger.info(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, '
                              f'Loss: {loss.item():.4f}')
            
            # Validation phase
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    
                    outputs = model(data)
                    loss = criterion(outputs['logits'], target)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs['logits'].data, 1)
                    val_total += target.size(0)
                    val_correct += (predicted == target).sum().item()
            
            # Calculate metrics
            train_loss /= len(train_loader)
            train_acc = train_correct / train_total
            val_loss /= len(val_loader)
            val_acc = val_correct / val_total
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                
                # Save model
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path = f"/workspace/models/transformer_financial_{timestamp}.pt"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_acc': val_acc,
                    'val_loss': val_loss
                }, model_path)
            
            # Record history
            training_history['train_loss'].append(train_loss)
            training_history['train_acc'].append(train_acc)
            training_history['val_loss'].append(val_loss)
            training_history['val_acc'].append(val_acc)
            
            # Log progress
            logger.info(f'Epoch {epoch+1}/{epochs}: '
                       f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                       f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            
            # Early stopping
            if epoch > 10 and val_acc < best_val_acc - 0.02:
                logger.info("Early stopping triggered")
                break
        
        training_time = time.time() - start_time
        
        # Performance evaluation
        if best_val_acc >= 0.75:
            grade = "EXCELLENT"
        elif best_val_acc >= 0.65:
            grade = "GOOD"
        elif best_val_acc >= 0.55:
            grade = "ACCEPTABLE"
        else:
            grade = "NEEDS IMPROVEMENT"
        
        logger.info("✅ Transformer training completed!")
        logger.info(f"Training time: {training_time:.1f} seconds ({training_time/60:.1f} minutes)")
        logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        logger.info(f"Performance grade: {grade}")
        
        return {
            'status': 'success',
            'model_path': model_path,
            'training_time': training_time,
            'epochs_trained': epoch + 1,
            'best_val_accuracy': best_val_acc,
            'best_val_loss': best_val_loss,
            'final_train_accuracy': train_acc,
            'performance_grade': grade,
            'training_history': training_history,
            'model_parameters': sum(p.numel() for p in model.parameters()),
            'samples_trained': len(dataset)
        }
        
    except Exception as e:
        logger.error(f"❌ Transformer training failed: {e}")
        logger.error(traceback.format_exc())
        return {'status': 'failed', 'error': str(e)}

def main():
    """Main transformer training pipeline"""
    parser = argparse.ArgumentParser(description='Transformer model training system')
    parser.add_argument('--samples', type=int, default=50000, help='Training samples')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--quick', action='store_true', help='Quick training mode')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_transformer_logging()
    logger.info("🤖 TRANSFORMER TRAINING SYSTEM - ADVANCED FINANCIAL PREDICTIONS")
    logger.info(f"Configuration: {args.samples:,} samples, {args.epochs} epochs")
    
    try:
        # Adjust for quick mode
        if args.quick:
            args.samples = 20000
            args.epochs = 20
            logger.info("Quick mode enabled - reduced samples and epochs")
        
        # Step 1: Create training data
        logger.info("Step 1: Creating transformer training data...")
        training_data = create_transformer_training_data(args.samples)
        
        # Step 2: Train transformer
        logger.info("Step 2: Training transformer model...")
        results = train_transformer_model(
            training_data, 
            epochs=args.epochs, 
            batch_size=args.batch_size
        )
        
        # Step 3: Generate report
        logger.info("Step 3: Generating training report...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"/workspace/logs/transformer_training_report_{timestamp}.json"
        
        report = {
            'timestamp': timestamp,
            'configuration': {
                'samples': args.samples,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'quick_mode': args.quick
            },
            'training_results': results,
            'data_info': {
                'total_samples': len(training_data),
                'currency_pairs': len(training_data['pair'].unique()),
                'date_range': {
                    'start': str(training_data['timestamp'].min()),
                    'end': str(training_data['timestamp'].max())
                }
            }
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Final status
        if results['status'] == 'success':
            logger.info("🎉 TRANSFORMER TRAINING COMPLETED SUCCESSFULLY!")
            logger.info(f"Best validation accuracy: {results['best_val_accuracy']:.4f}")
            logger.info(f"Performance grade: {results['performance_grade']}")
            logger.info(f"Model saved: {results['model_path']}")
            logger.info(f"Report saved: {report_file}")
            
            if results['best_val_accuracy'] >= 0.7:
                logger.info("✅ Transformer ready for production deployment!")
            elif results['best_val_accuracy'] >= 0.6:
                logger.info("✅ Transformer ready for staging deployment")
            elif results['best_val_accuracy'] >= 0.55:
                logger.info("⚠️ Transformer acceptable, consider more training")
            else:
                logger.info("❌ Transformer needs significant improvement")
                
            return 0
        else:
            logger.error("❌ Transformer training failed!")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Transformer training pipeline failed: {e}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())