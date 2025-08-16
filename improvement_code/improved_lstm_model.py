
# IMPROVED LSTM MODEL - FIX OVERFITTING

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import TimeSeriesSplit

def build_improved_lstm_model(self, input_shape):
    """Build LSTM model with proper regularization to fix overfitting"""
    
    model = Sequential([
        # Input LSTM layer with regularization
        LSTM(64, return_sequences=True, 
             dropout=0.3, recurrent_dropout=0.3,
             kernel_regularizer=l2(0.01),
             input_shape=input_shape),
        BatchNormalization(),
        
        # Second LSTM layer
        LSTM(32, return_sequences=True,
             dropout=0.3, recurrent_dropout=0.3,
             kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        
        # Third LSTM layer
        LSTM(16, dropout=0.3, recurrent_dropout=0.3,
             kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        
        # Dense layers with regularization
        Dense(8, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.4),
        Dense(3, activation='softmax')
    ])
    
    return model

def train_with_cross_validation(self, X, y):
    """Train model with time series cross-validation"""
    
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = self.build_improved_lstm_model(X_train.shape[1:])
        model.compile(optimizer='adam', 
                     loss='categorical_crossentropy', 
                     metrics=['accuracy'])
        
        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Learning rate reduction
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        scores.append(history.history['val_accuracy'][-1])
    
    return np.mean(scores)
