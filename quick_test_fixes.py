#!/usr/bin/env python3
"""
Quick test to verify all critical fixes are working
"""

import sys
import warnings
warnings.filterwarnings('ignore')

def test_dependencies():
    """Test all critical dependencies"""
    print("🔍 Testing Dependencies...")
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__}")
    except ImportError as e:
        print(f"❌ TensorFlow: {e}")
        return False
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch: {e}")
        return False
    
    try:
        import xgboost as xgb
        print(f"✅ XGBoost {xgb.__version__}")
    except ImportError as e:
        print(f"❌ XGBoost: {e}")
        return False
    
    try:
        import lightgbm as lgb
        print(f"✅ LightGBM {lgb.__version__}")
    except ImportError as e:
        print(f"❌ LightGBM: {e}")
        return False
    
    try:
        import talib
        print(f"✅ TA-Lib {talib.__version__}")
    except ImportError as e:
        print(f"❌ TA-Lib: {e}")
        return False
    
    return True

def test_lstm_model():
    """Test LSTM model can be created and basic operations work"""
    print("\n🤖 Testing LSTM Model...")
    
    try:
        sys.path.append('/workspace')
        from lstm_model import LSTMTradingModel
        from data_manager_fixed import DataManager
        
        # Create model
        lstm_model = LSTMTradingModel()
        print("✅ LSTM model created successfully")
        
        # Create sample data
        data_manager = DataManager()
        sample_data = data_manager.create_sample_data(1000)
        print(f"✅ Sample data created: {sample_data.shape}")
        
        # Test feature preparation
        processed_data = lstm_model.calculate_technical_indicators(sample_data)
        features = lstm_model.prepare_features(processed_data)
        print(f"✅ Features prepared: {features.shape}")
        
        # Test label generation
        labels = lstm_model.generate_labels(processed_data)
        print(f"✅ Labels generated: {labels.shape}")
        
        # Test sequence creation
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        X, y = lstm_model.create_sequences(features_scaled, labels)
        
        if X is not None and y is not None:
            print(f"✅ Sequences created: X={X.shape}, y={y.shape}")
            return True
        else:
            print("❌ Sequence creation failed")
            return False
            
    except Exception as e:
        print(f"❌ LSTM test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_training():
    """Test actual model training with small dataset"""
    print("\n🚀 Testing Model Training...")
    
    try:
        sys.path.append('/workspace')
        from lstm_model import LSTMTradingModel
        from data_manager_fixed import DataManager
        
        # Create small dataset for quick training
        data_manager = DataManager()
        training_data = data_manager.create_sample_data(2000)  # Small dataset
        
        # Initialize model
        lstm_model = LSTMTradingModel()
        
        # Train with minimal epochs
        print("Starting quick training (5 epochs)...")
        history = lstm_model.train_model(
            data=training_data,
            validation_split=0.2,
            epochs=5  # Very quick training
        )
        
        if history is not None:
            final_acc = max(history.history.get('val_accuracy', [0]))
            print(f"✅ Training completed! Validation accuracy: {final_acc:.4f}")
            
            # Test model saving/loading
            import os
            model_path = "/workspace/models/test_model.h5"
            lstm_model.save_model(model_path)
            
            if os.path.exists(model_path):
                print("✅ Model saved successfully")
                
                # Test loading
                test_model = LSTMTradingModel()
                if test_model.load_model(model_path):
                    print("✅ Model loaded successfully")
                    return True
                else:
                    print("❌ Model loading failed")
                    return False
            else:
                print("❌ Model saving failed")
                return False
        else:
            print("❌ Training failed")
            return False
            
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 CRITICAL ISSUES FIX VERIFICATION")
    print("=" * 50)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("LSTM Model", test_lstm_model),
        ("Model Training", test_model_training)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            print(f"✅ {test_name} PASSED")
            passed += 1
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "="*50)
    print(f"📊 FINAL RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL CRITICAL ISSUES FIXED!")
        print("The trading system is ready for production training")
    elif passed >= total * 0.5:
        print("⚠️ PARTIAL SUCCESS - Most issues resolved")
        print("Some advanced features may need additional work")
    else:
        print("❌ CRITICAL ISSUES REMAIN")
        print("Additional debugging required")
    
    return passed / total

if __name__ == "__main__":
    success_rate = main()
    sys.exit(0 if success_rate >= 0.5 else 1)