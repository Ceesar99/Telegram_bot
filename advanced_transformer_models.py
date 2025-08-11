import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import math
import warnings
warnings.filterwarnings('ignore')

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):
    """Advanced positional encoding for financial time series"""
    
    def __init__(self, d_model: int, max_length: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class MultiHeadSelfAttention(nn.Module):
    """Enhanced multi-head self-attention for financial data"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        
        # Linear transformations
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model)
        
        output = self.w_o(context)
        return output, attention_weights

class FeedForward(nn.Module):
    """Position-wise feed-forward network"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # GELU works better than ReLU for transformers
        
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class TransformerEncoderLayer(nn.Module):
    """Enhanced transformer encoder layer with residual connections and layer normalization"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output, attention_weights = self.self_attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attention_weights

class FinancialTransformer(nn.Module):
    """Advanced transformer model specifically designed for financial market prediction"""
    
    def __init__(self, 
                 input_dim: int,
                 d_model: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 d_ff: int = 1024,
                 max_length: int = 1000,
                 num_classes: int = 3,  # BUY, SELL, HOLD
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_length)
        
        # Transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output layers for classification
        self.output_norm = nn.LayerNorm(d_model)
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
            nn.Linear(d_model, d_ff // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff // 2, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Input projection and positional encoding
        x = self.input_projection(x) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Store attention weights for analysis
        attention_weights = []
        
        # Pass through transformer layers
        for layer in self.transformer_layers:
            x, attn_weights = layer(x, mask)
            attention_weights.append(attn_weights)
        
        x = self.output_norm(x)
        
        # Use the last token for classification (or you could use mean pooling)
        last_hidden = x[:, -1, :]  # Take last time step
        
        # Classification output
        logits = self.classifier(last_hidden)
        
        # Confidence estimation
        confidence = self.confidence_head(last_hidden)
        
        return {
            'logits': logits,
            'confidence': confidence,
            'attention_weights': attention_weights,
            'hidden_states': x
        }

class FinancialDataset(Dataset):
    """Dataset class for financial time series data"""
    
    def __init__(self, data: np.ndarray, targets: np.ndarray, sequence_length: int = 100):
        self.data = data
        self.targets = targets
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.data) - self.sequence_length
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.sequence_length]
        y = self.targets[idx + self.sequence_length]
        
        return torch.FloatTensor(x), torch.LongTensor([y])

class TransformerTrainer:
    """Advanced trainer for financial transformer models"""
    
    def __init__(self, model: FinancialTransformer, device: torch.device = device):
        self.model = model.to(device)
        self.device = device
        self.logger = logging.getLogger('TransformerTrainer')
        
        # Loss functions
        self.classification_loss = nn.CrossEntropyLoss()
        self.confidence_loss = nn.BCELoss()
        
        # Optimizer with learning rate scheduling
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        # Training statistics
        self.training_stats = {
            'losses': [],
            'accuracies': [],
            'confidences': []
        }
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        total_confidence = 0
        
        for batch_idx, (data, targets) in enumerate(dataloader):
            data = data.to(self.device)
            targets = targets.squeeze().to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(data)
            
            # Classification loss
            cls_loss = self.classification_loss(outputs['logits'], targets)
            
            # Confidence loss (confidence should be high for correct predictions)
            predictions = torch.argmax(outputs['logits'], dim=1)
            correct_mask = (predictions == targets).float()
            conf_loss = self.confidence_loss(outputs['confidence'].squeeze(), correct_mask)
            
            # Combined loss
            total_loss_batch = cls_loss + 0.1 * conf_loss
            
            # Backward pass
            total_loss_batch.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Statistics
            total_loss += total_loss_batch.item()
            correct_predictions += (predictions == targets).sum().item()
            total_predictions += targets.size(0)
            total_confidence += outputs['confidence'].mean().item()
            
            if batch_idx % 100 == 0:
                self.logger.info(f'Epoch {epoch}, Batch {batch_idx}, Loss: {total_loss_batch.item():.4f}')
        
        # Update learning rate
        self.scheduler.step()
        
        epoch_stats = {
            'loss': total_loss / len(dataloader),
            'accuracy': correct_predictions / total_predictions,
            'avg_confidence': total_confidence / len(dataloader),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        self.training_stats['losses'].append(epoch_stats['loss'])
        self.training_stats['accuracies'].append(epoch_stats['accuracy'])
        self.training_stats['confidences'].append(epoch_stats['avg_confidence'])
        
        return epoch_stats
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        all_confidences = []
        
        with torch.no_grad():
            for data, targets in dataloader:
                data = data.to(self.device)
                targets = targets.squeeze().to(self.device)
                
                outputs = self.model(data)
                
                # Loss calculation
                cls_loss = self.classification_loss(outputs['logits'], targets)
                
                predictions = torch.argmax(outputs['logits'], dim=1)
                correct_mask = (predictions == targets).float()
                conf_loss = self.confidence_loss(outputs['confidence'].squeeze(), correct_mask)
                
                total_loss += (cls_loss + 0.1 * conf_loss).item()
                correct_predictions += (predictions == targets).sum().item()
                total_predictions += targets.size(0)
                all_confidences.extend(outputs['confidence'].cpu().numpy())
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': correct_predictions / total_predictions,
            'avg_confidence': np.mean(all_confidences),
            'confidence_std': np.std(all_confidences)
        }
    
    def predict(self, data: np.ndarray, return_attention: bool = False) -> Dict[str, Any]:
        """Make predictions with the trained model"""
        self.model.eval()
        
        with torch.no_grad():
            # Convert to tensor and add batch dimension
            x = torch.FloatTensor(data).unsqueeze(0).to(self.device)
            
            outputs = self.model(x)
            
            # Get predictions
            logits = outputs['logits'].cpu().numpy()[0]
            probabilities = F.softmax(outputs['logits'], dim=1).cpu().numpy()[0]
            confidence = outputs['confidence'].cpu().numpy()[0, 0]
            
            prediction = np.argmax(logits)
            
            result = {
                'prediction': int(prediction),
                'probabilities': probabilities,
                'confidence': float(confidence),
                'logits': logits
            }
            
            if return_attention:
                result['attention_weights'] = [
                    attn.cpu().numpy() for attn in outputs['attention_weights']
                ]
            
            return result

class MultiTimeframeTransformer:
    """Multi-timeframe transformer for comprehensive market analysis"""
    
    def __init__(self, input_dim: int):
        self.logger = logging.getLogger('MultiTimeframeTransformer')
        
        # Different models for different timeframes
        self.models = {
            '1m': FinancialTransformer(input_dim, d_model=128, num_layers=4),
            '5m': FinancialTransformer(input_dim, d_model=256, num_layers=6),
            '15m': FinancialTransformer(input_dim, d_model=512, num_layers=8),
            '1h': FinancialTransformer(input_dim, d_model=256, num_layers=6)
        }
        
        # Meta-learner to combine predictions from different timeframes
        self.meta_learner = nn.Sequential(
            nn.Linear(len(self.models) * 3, 128),  # 3 classes per model
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3)
        ).to(device)
        
        self.trainers = {
            timeframe: TransformerTrainer(model)
            for timeframe, model in self.models.items()
        }
    
    def train_all_timeframes(self, data_dict: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                           epochs: int = 50) -> Dict[str, List[float]]:
        """Train all timeframe models"""
        results = {}
        
        for timeframe, (X, y) in data_dict.items():
            if timeframe in self.models:
                self.logger.info(f"Training {timeframe} model...")
                
                # Create dataset and dataloader
                dataset = FinancialDataset(X, y)
                dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
                
                # Train the model
                timeframe_results = []
                for epoch in range(epochs):
                    stats = self.trainers[timeframe].train_epoch(dataloader, epoch)
                    timeframe_results.append(stats)
                    
                    if epoch % 10 == 0:
                        self.logger.info(f"{timeframe} Epoch {epoch}: "
                                       f"Loss={stats['loss']:.4f}, "
                                       f"Acc={stats['accuracy']:.4f}")
                
                results[timeframe] = timeframe_results
        
        return results
    
    def predict_multi_timeframe(self, data_dict: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Make predictions using all timeframe models and combine them"""
        individual_predictions = {}
        all_probabilities = []
        
        # Get predictions from each timeframe model
        for timeframe, data in data_dict.items():
            if timeframe in self.models:
                prediction = self.trainers[timeframe].predict(data)
                individual_predictions[timeframe] = prediction
                all_probabilities.append(prediction['probabilities'])
        
        # Combine predictions using meta-learner
        if all_probabilities:
            combined_input = torch.FloatTensor(np.concatenate(all_probabilities)).unsqueeze(0).to(device)
            
            with torch.no_grad():
                meta_output = self.meta_learner(combined_input)
                meta_probabilities = F.softmax(meta_output, dim=1).cpu().numpy()[0]
                meta_prediction = np.argmax(meta_probabilities)
        else:
            meta_probabilities = np.array([0.33, 0.33, 0.34])
            meta_prediction = 2  # HOLD as default
        
        return {
            'individual_predictions': individual_predictions,
            'meta_prediction': int(meta_prediction),
            'meta_probabilities': meta_probabilities,
            'consensus_strength': self._calculate_consensus(individual_predictions)
        }
    
    def _calculate_consensus(self, predictions: Dict[str, Dict]) -> float:
        """Calculate consensus strength among different timeframe predictions"""
        if not predictions:
            return 0.0
        
        pred_values = [pred['prediction'] for pred in predictions.values()]
        
        # Calculate agreement percentage
        most_common = max(set(pred_values), key=pred_values.count)
        agreement_count = pred_values.count(most_common)
        consensus_strength = agreement_count / len(pred_values)
        
        return consensus_strength

# Enhanced feature preprocessing for transformers
class TransformerFeatureProcessor:
    """Advanced feature preprocessing specifically for transformer models"""
    
    def __init__(self):
        self.logger = logging.getLogger('TransformerFeatureProcessor')
        self.scalers = {}
        
    def prepare_financial_features(self, price_data: np.ndarray, 
                                 volume_data: np.ndarray = None,
                                 additional_features: np.ndarray = None) -> np.ndarray:
        """Prepare comprehensive features for transformer input"""
        features = []
        
        # Price-based features
        if len(price_data.shape) == 1:
            price_data = price_data.reshape(-1, 1)
        
        # Normalize prices
        price_returns = np.diff(price_data.flatten()) / price_data[:-1].flatten()
        price_returns = np.concatenate([[0], price_returns])  # Add zero for first element
        
        # Log returns
        log_returns = np.diff(np.log(price_data.flatten()))
        log_returns = np.concatenate([[0], log_returns])
        
        features.extend([
            price_data.flatten(),
            price_returns,
            log_returns
        ])
        
        # Technical indicators
        if len(price_data) >= 20:
            sma_20 = self._calculate_sma(price_data.flatten(), 20)
            ema_12 = self._calculate_ema(price_data.flatten(), 12)
            rsi_14 = self._calculate_rsi(price_data.flatten(), 14)
            
            features.extend([sma_20, ema_12, rsi_14])
        
        # Volume features (if available)
        if volume_data is not None:
            volume_sma = self._calculate_sma(volume_data, 20) if len(volume_data) >= 20 else volume_data
            features.append(volume_sma)
        
        # Additional features
        if additional_features is not None:
            features.append(additional_features)
        
        # Stack all features
        feature_matrix = np.column_stack(features)
        
        # Handle any NaN values
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        
        return feature_matrix
    
    def _calculate_sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Simple Moving Average"""
        sma = np.convolve(data, np.ones(period)/period, mode='same')
        sma[:period-1] = data[:period-1]  # Fill initial values
        return sma
    
    def _calculate_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average"""
        alpha = 2.0 / (period + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]
        
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    def _calculate_rsi(self, data: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate Relative Strength Index"""
        delta = np.diff(data)
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        
        avg_gains = np.convolve(gains, np.ones(period)/period, mode='same')
        avg_losses = np.convolve(losses, np.ones(period)/period, mode='same')
        
        rs = avg_gains / (avg_losses + 1e-10)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        
        # Add initial value
        rsi = np.concatenate([[50], rsi])  # Start with neutral RSI
        
        return rsi