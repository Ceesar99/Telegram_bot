#!/usr/bin/env python3
"""
🤖 TRANSFORMER MODELS TRAINING WITH REAL MARKET DATA
Train advanced transformer models for multi-timeframe trading analysis
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Deep Learning Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Add project path
sys.path.append('/workspace')

class TradingDataset(Dataset):
    """Dataset class for transformer training"""
    
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        
        # Linear transformations
        Q = self.query(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        output = self.output(context)
        return output

class TransformerBlock(nn.Module):
    """Transformer block with attention and feed-forward"""
    
    def __init__(self, d_model, n_heads=8, d_ff=512, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Self-attention with residual connection
        attention_output = self.attention(x)
        x = self.norm1(x + attention_output)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x

class TradingTransformer(nn.Module):
    """Advanced transformer model for trading predictions"""
    
    def __init__(self, input_dim, d_model=128, n_heads=8, n_layers=6, d_ff=512, 
                 max_seq_len=60, dropout=0.1, output_dim=1):
        super(TradingTransformer, self).__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(max_seq_len, d_model)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) 
            for _ in range(n_layers)
        ])
        
        # Output layers
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, output_dim)
        )
        
    def _create_positional_encoding(self, max_len, d_model):
        """Create positional encoding"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        if seq_len <= self.pos_encoding.size(1):
            x += self.pos_encoding[:, :seq_len, :].to(x.device)
        
        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        # Global pooling and classification
        x = x.transpose(1, 2)  # (batch, d_model, seq_len)
        x = self.global_pool(x)  # (batch, d_model, 1)
        x = x.squeeze(-1)  # (batch, d_model)
        
        output = self.classifier(x)
        return output

class RealDataTransformerTrainer:
    """Train transformer models with real market data"""
    
    def __init__(self):
        self.setup_logging()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.scalers = {}
        self.performance_metrics = {}
        
    def setup_logging(self):
        """Setup logging"""
        log_dir = "/workspace/logs"
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{log_dir}/transformer_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('TransformerTrainer')
        
    def load_and_prepare_data(self, data_file):
        """Load and prepare real market data"""
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
        
        # Focus on main symbols for transformer training
        main_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
        data = data[data['symbol'].isin(main_symbols)].copy()
        
        self.logger.info(f"📈 Using {len(data):,} records from {len(data['symbol'].unique())} symbols")
        
        # Add features
        data = self._add_technical_features(data)
        
        return data
        
    def _add_technical_features(self, data):
        """Add technical analysis features for each symbol"""
        processed_data = []
        
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol].copy()
            
            if len(symbol_data) < 60:  # Skip if insufficient data
                continue
                
            # Price features
            symbol_data['returns'] = symbol_data['close'].pct_change()
            symbol_data['log_returns'] = np.log(symbol_data['close'] / symbol_data['close'].shift(1))
            symbol_data['high_low_ratio'] = symbol_data['high'] / symbol_data['low']
            symbol_data['close_open_ratio'] = symbol_data['close'] / symbol_data['open']
            
            # Moving averages and ratios
            for period in [5, 10, 20, 50]:
                symbol_data[f'sma_{period}'] = symbol_data['close'].rolling(period).mean()
                symbol_data[f'ema_{period}'] = symbol_data['close'].ewm(span=period).mean()
                symbol_data[f'price_sma_ratio_{period}'] = symbol_data['close'] / symbol_data[f'sma_{period}']
                
            # RSI
            delta = symbol_data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            symbol_data['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = symbol_data['close'].ewm(span=12).mean()
            exp2 = symbol_data['close'].ewm(span=26).mean()
            symbol_data['macd'] = exp1 - exp2
            symbol_data['macd_signal'] = symbol_data['macd'].ewm(span=9).mean()
            symbol_data['macd_histogram'] = symbol_data['macd'] - symbol_data['macd_signal']
            
            # Bollinger Bands
            symbol_data['bb_middle'] = symbol_data['close'].rolling(20).mean()
            bb_std = symbol_data['close'].rolling(20).std()
            symbol_data['bb_upper'] = symbol_data['bb_middle'] + (bb_std * 2)
            symbol_data['bb_lower'] = symbol_data['bb_middle'] - (bb_std * 2)
            symbol_data['bb_position'] = (symbol_data['close'] - symbol_data['bb_lower']) / (symbol_data['bb_upper'] - symbol_data['bb_lower'])
            
            # Volume features
            symbol_data['volume_sma'] = symbol_data['volume'].rolling(20).mean()
            symbol_data['volume_ratio'] = symbol_data['volume'] / symbol_data['volume_sma']
            symbol_data['price_volume'] = symbol_data['close'] * symbol_data['volume']
            
            # Volatility features
            symbol_data['volatility'] = symbol_data['returns'].rolling(20).std()
            symbol_data['volatility_ratio'] = symbol_data['volatility'] / symbol_data['volatility'].rolling(60).mean()
            
            # Target variable (future returns)
            symbol_data['future_returns'] = symbol_data['returns'].shift(-1)
            
            processed_data.append(symbol_data)
            
        combined_data = pd.concat(processed_data, ignore_index=True)
        combined_data = combined_data.dropna()
        
        self.logger.info(f"✅ Processed data shape: {combined_data.shape}")
        return combined_data
        
    def create_sequences(self, data, seq_length=60):
        """Create sequences for transformer training"""
        self.logger.info(f"🔧 Creating sequences of length {seq_length}...")
        
        # Feature columns (exclude metadata and target)
        exclude_cols = ['datetime', 'symbol', 'future_returns', 'date', 'dividends', 'stock splits']
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        sequences = []
        targets = []
        
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol].copy()
            
            if len(symbol_data) < seq_length + 1:
                continue
                
            # Extract features and targets
            features = symbol_data[feature_cols].values
            target_values = symbol_data['future_returns'].values
            
            # Create sequences
            for i in range(len(features) - seq_length):
                seq = features[i:i+seq_length]
                target = target_values[i+seq_length-1]
                
                if not np.isnan(target):
                    sequences.append(seq)
                    targets.append(target)
        
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        self.logger.info(f"✅ Created {len(sequences):,} sequences")
        self.logger.info(f"📊 Sequence shape: {sequences.shape}")
        
        return sequences, targets, feature_cols
        
    def train_transformer_model(self, X_train, y_train, X_val, y_val, input_dim):
        """Train transformer model"""
        self.logger.info("🚀 Training Transformer model...")
        
        # Create datasets
        train_dataset = TradingDataset(X_train, y_train)
        val_dataset = TradingDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Create model
        model = TradingTransformer(
            input_dim=input_dim,
            d_model=128,
            n_heads=8,
            n_layers=4,  # Reduced for faster training
            d_ff=256,
            max_seq_len=60,
            dropout=0.1,
            output_dim=1
        ).to(self.device)
        
        # Optimizer and loss
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        
        # Training loop
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        epochs = 50  # Reduced for demo
        self.logger.info(f"🎯 Training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_losses = []
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_x).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_losses.append(loss.item())
            
            # Validation
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_x).squeeze()
                    val_loss = criterion(outputs, batch_y)
                    val_losses.append(val_loss.item())
            
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            
            scheduler.step(avg_val_loss)
            
            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), '/tmp/best_transformer_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self.logger.info("Early stopping triggered")
                    break
        
        # Load best model
        model.load_state_dict(torch.load('/tmp/best_transformer_model.pth'))
        
        # Evaluate
        model.eval()
        train_preds = []
        val_preds = []
        train_targets = []
        val_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = model(batch_x).squeeze()
                train_preds.extend(outputs.cpu().numpy())
                train_targets.extend(batch_y.cpu().numpy())
                
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = model(batch_x).squeeze()
                val_preds.extend(outputs.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(train_targets, train_preds))
        val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
        val_r2 = r2_score(val_targets, val_preds)
        val_mae = mean_absolute_error(val_targets, val_preds)
        
        self.models['transformer'] = model
        self.performance_metrics['transformer'] = {
            'train_rmse': train_rmse,
            'val_rmse': val_rmse,
            'val_r2': val_r2,
            'val_mae': val_mae
        }
        
        self.logger.info(f"✅ Transformer - Val RMSE: {val_rmse:.6f}, R²: {val_r2:.4f}")
        return model
        
    def save_models(self):
        """Save trained models"""
        self.logger.info("💾 Saving transformer models...")
        
        model_dir = "/workspace/models/production/transformer"
        os.makedirs(model_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save transformer model
        if 'transformer' in self.models:
            model_path = f"{model_dir}/trading_transformer_{timestamp}.pth"
            torch.save(self.models['transformer'].state_dict(), model_path)
            self.logger.info(f"✅ Saved transformer model to {model_path}")
        
        # Save metrics
        metrics_path = f"{model_dir}/metrics_{timestamp}.json"
        import json
        with open(metrics_path, 'w') as f:
            json.dump(self.performance_metrics, f, indent=2)
        
        self.logger.info(f"📊 Saved metrics to {metrics_path}")
        
    def train_all_models(self, data_file):
        """Train all transformer models"""
        start_time = datetime.now()
        self.logger.info("🚀 Starting transformer models training with real data...")
        
        # Load and prepare data
        data = self.load_and_prepare_data(data_file)
        
        # Create sequences
        sequences, targets, feature_cols = self.create_sequences(data)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            sequences, targets, test_size=0.2, random_state=42
        )
        
        self.logger.info(f"📊 Train sequences: {X_train.shape[0]}, Val sequences: {X_val.shape[0]}")
        
        # Train transformer
        self.train_transformer_model(X_train, y_train, X_val, y_val, X_train.shape[2])
        
        # Save models
        self.save_models()
        
        training_time = datetime.now() - start_time
        self.logger.info(f"🎉 TRANSFORMER TRAINING COMPLETED in {training_time}")
        
        print("\n" + "="*80)
        print("🤖 TRANSFORMER MODELS PERFORMANCE SUMMARY")
        print("="*80)
        
        for model_name, metrics in self.performance_metrics.items():
            print(f"\n📊 {model_name.upper()}:")
            print(f"   Val RMSE: {metrics['val_rmse']:.6f}")
            print(f"   Val R²:   {metrics['val_r2']:.4f}")
            print(f"   Val MAE:  {metrics['val_mae']:.6f}")
            
        print(f"\n⏱️  Total Training Time: {training_time}")
        print("💾 Models saved to: /workspace/models/production/transformer/")

if __name__ == "__main__":
    trainer = RealDataTransformerTrainer()
    
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
        trainer.train_all_models(data_file)
    else:
        print("❌ No real market data file found!")