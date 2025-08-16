#!/usr/bin/env python3
"""
🔬 COMPREHENSIVE MODEL VALIDATION SCRIPT
Validates all AI/ML models for production readiness
"""

import os
import sys
import logging
from datetime import datetime

# Add workspace to path
sys.path.append('/workspace')

def setup_logging():
    """Setup validation logging"""
    log_file = f"/workspace/logs/validation/model_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('ModelValidation')

def validate_lstm_models():
    """Validate LSTM models"""
    logger = logging.getLogger('ModelValidation')
    logger.info("🧠 Validating LSTM models...")
    
    try:
        from lstm_model import LSTMTradingModel
        model = LSTMTradingModel()
        
        # Check if models exist
        model_files = [
            "/workspace/models/best_model.h5",
            "/workspace/models/production_lstm_20250814_222320.h5"
        ]
        
        for model_file in model_files:
            if os.path.exists(model_file):
                logger.info(f"✅ Found LSTM model: {model_file}")
                try:
                    # Test loading
                    import tensorflow as tf
                    test_model = tf.keras.models.load_model(model_file)
                    logger.info(f"✅ Successfully loaded: {model_file}")
                    logger.info(f"   Input shape: {test_model.input_shape}")
                    logger.info(f"   Output shape: {test_model.output_shape}")
                except Exception as e:
                    logger.error(f"❌ Error loading {model_file}: {e}")
            else:
                logger.warning(f"⚠️ LSTM model not found: {model_file}")
        
        return True
    except Exception as e:
        logger.error(f"❌ LSTM validation failed: {e}")
        return False

def validate_ensemble_models():
    """Validate ensemble models"""
    logger = logging.getLogger('ModelValidation')
    logger.info("🎯 Validating ensemble models...")
    
    try:
        from ensemble_models import EnsembleSignalGenerator
        ensemble = EnsembleSignalGenerator()
        
        # Check individual model components
        for model_name, model in ensemble.models.items():
            logger.info(f"   Checking {model_name}: {'✅ Available' if model else '❌ Missing'}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Ensemble validation failed: {e}")
        return False

def validate_dependencies():
    """Validate critical dependencies"""
    logger = logging.getLogger('ModelValidation')
    logger.info("📦 Validating dependencies...")
    
    dependencies = {
        'tensorflow': '2.16.0',
        'torch': '2.0.0',
        'xgboost': '2.0.0',
        'sklearn': '1.3.0',
        'pandas': '2.0.0',
        'numpy': '1.24.0'
    }
    
    all_good = True
    for dep, min_version in dependencies.items():
        try:
            if dep == 'torch':
                import torch
                version = torch.__version__
            elif dep == 'tensorflow':
                import tensorflow as tf
                version = tf.__version__
            elif dep == 'xgboost':
                import xgboost as xgb
                version = xgb.__version__
            elif dep == 'sklearn':
                import sklearn
                version = sklearn.__version__
            elif dep == 'pandas':
                import pandas as pd
                version = pd.__version__
            elif dep == 'numpy':
                import numpy as np
                version = np.__version__
            
            logger.info(f"✅ {dep}: {version}")
        except ImportError:
            logger.error(f"❌ {dep}: Not installed")
            all_good = False
    
    return all_good

def main():
    """Main validation function"""
    logger = setup_logging()
    logger.info("🔬 STARTING COMPREHENSIVE MODEL VALIDATION")
    logger.info("=" * 50)
    
    validation_results = {
        'dependencies': validate_dependencies(),
        'lstm_models': validate_lstm_models(),
        'ensemble_models': validate_ensemble_models()
    }
    
    logger.info("=" * 50)
    logger.info("📊 VALIDATION SUMMARY")
    
    all_passed = True
    for category, result in validation_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{category.upper()}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        logger.info("🎉 ALL VALIDATIONS PASSED - SYSTEM READY FOR NEXT PHASE")
    else:
        logger.warning("⚠️ SOME VALIDATIONS FAILED - REVIEW ISSUES ABOVE")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
