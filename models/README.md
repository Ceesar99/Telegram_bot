# AI Models Directory

This directory contains the trained AI models used by the trading bot system.

## 🤖 Expected Model Files

### Core LSTM Model
- **`lstm_model.h5`** - Main LSTM neural network model for signal prediction
- **`model_metadata.json`** - Model configuration and training information
- **`scaler.pkl`** - Data preprocessing scaler for feature normalization

### Model Training Data
- **`features_config.json`** - Feature engineering configuration
- **`training_history.json`** - Training history and performance metrics

## 📊 Model Information

### Current Status: **MISSING** ⚠️
The AI models are not included in this repository because:
1. **Size Constraints** - Model files can be 100MB+ 
2. **Training Required** - Models need to be trained on your specific trading data
3. **Customization** - Models should be tailored to your trading strategy

## 🚀 How to Generate Models

### Option 1: Auto-Training (Recommended)
```bash
# The system will automatically train models on first run
python3 unified_trading_system.py --mode hybrid --train-models
```

### Option 2: Manual Training
```bash
# Run the LSTM model training directly
python3 lstm_model.py --train --data-source auto
```

### Option 3: Load Pre-trained Models
If you have existing models, place them in this directory:
```
/workspace/models/
├── lstm_model.h5           # Main LSTM model
├── scaler.pkl              # Feature scaler
├── model_metadata.json     # Model info
└── features_config.json    # Feature configuration
```

## 📈 Model Performance Targets

- **Accuracy**: 95%+ signal prediction accuracy
- **Precision**: 90%+ positive signal precision  
- **Recall**: 85%+ signal detection rate
- **F1-Score**: 87%+ balanced performance metric

## 🔄 Model Retraining

Models are automatically retrained:
- **Daily**: Performance-based retraining if accuracy drops below 90%
- **Weekly**: Scheduled retraining with new market data
- **Manual**: Use `--retrain` flag when running the system

## 🛠️ Model Architecture

### LSTM Network Structure
```
Input Layer (20 features) 
    ↓
LSTM Layer 1 (50 units) + Dropout (0.2)
    ↓  
LSTM Layer 2 (50 units) + Dropout (0.2)
    ↓
LSTM Layer 3 (50 units) + Dropout (0.2) 
    ↓
Dense Layer (25 units) + ReLU
    ↓
Output Layer (3 units) + Softmax
    ↓
[BUY, SELL, HOLD] predictions
```

### Features Used (20 indicators)
- RSI, MACD, Bollinger Bands
- Stochastic, Williams %R, CCI  
- ADX, ATR, EMA (9,21,50,200)
- SMA (10,20,50,100)
- Volume indicators
- Price action patterns

## 📝 Notes

- Models are saved in HDF5 format for TensorFlow/Keras compatibility
- Scalers use sklearn's StandardScaler for feature normalization
- All models include comprehensive metadata for version tracking
- Training logs are automatically saved to `/workspace/logs/`

## 🔍 Troubleshooting

### Model Loading Issues
```python
# Test model loading
from lstm_model import LSTMTradingModel
model = LSTMTradingModel()
model.load_model()  # Should work without errors
```

### Retraining Models
```bash
# Force model retraining
python3 lstm_model.py --retrain --force
```

### Model Validation
```bash
# Validate model performance
python3 lstm_model.py --validate --test-data recent
```

---

**Status**: Ready for model training and deployment
**Last Updated**: 2025-08-11
**Version**: 1.0.0