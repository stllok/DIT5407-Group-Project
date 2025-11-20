"""
Advanced LSTM Model for HKO Temperature Prediction

This script implements an advanced LSTM model (1-2 stacked layers, 50-100 units, with dropout)
for predicting Hong Kong Observatory daily temperature data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import the existing HKO data reader
from hko_data_reader import read_hko_daily_csv, HKODailyRecord


def prepare_sequences(data: List[float], sequence_length: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare sequences for LSTM training.
    
    Parameters:
        data: List of temperature values
        sequence_length: Length of input sequences
        
    Returns:
        X: Input sequences (samples, sequence_length, features)
        y: Target values (samples,)
    """
    X, y = [], []
    
    for i in range(sequence_length, len(data)):
        # Use previous sequence_length values to predict the next value
        X.append(data[i-sequence_length:i])
        y.append(data[i])
    
    X = np.array(X)
    y = np.array(y)
    
    # Reshape X to have shape (samples, sequence_length, 1) for LSTM
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    return X, y


def create_lstm_model(sequence_length: int = 30, 
                     lstm_units: int = 64, 
                     num_layers: int = 2, 
                     dropout_rate: float = 0.2) -> tf.keras.Model:
    """
    Create an advanced LSTM model with stacked layers and dropout.
    
    Parameters:
        sequence_length: Length of input sequences
        lstm_units: Number of LSTM units (50-100)
        num_layers: Number of LSTM layers (1-2)
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Compiled Keras model
    """
    model = Sequential()
    
    # First LSTM layer
    if num_layers == 1:
        model.add(LSTM(lstm_units, activation='tanh', 
                      input_shape=(sequence_length, 1)))
        model.add(Dropout(dropout_rate))
    else:
        # First LSTM layer with return_sequences=True for stacking
        model.add(LSTM(lstm_units, activation='tanh', return_sequences=True,
                      input_shape=(sequence_length, 1)))
        model.add(Dropout(dropout_rate))
        
        # Second LSTM layer
        model.add(LSTM(lstm_units // 2, activation='tanh'))
        model.add(Dropout(dropout_rate))
    
    # Output layer for regression
    model.add(Dense(1))
    
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='mse', 
                  metrics=['mae'])
    
    return model


def preprocess_data(records: List[HKODailyRecord], method: str = 'interpolate') -> pd.Series:
    """
    Preprocess HKO data by handling missing values.
    
    Parameters:
        records: List of HKODailyRecord objects
        method: Method for handling missing values ('forward_fill' or 'interpolate')
        
    Returns:
        Pandas Series with processed temperature data
    """
    # Convert records to DataFrame
    df = pd.DataFrame([
        {'date': record.date, 'temperature': record.value}
        for record in records
    ])
    
    # Set date as index and sort
    df = df.set_index('date').sort_index()
    
    # Handle missing values
    if method == 'forward_fill':
        df['temperature'] = df['temperature'].fillna(method='ffill')
    elif method == 'interpolate':
        df['temperature'] = df['temperature'].interpolate(method='linear')
    else:
        raise ValueError("Method must be 'forward_fill' or 'interpolate'")
    
    # Fill any remaining NaN values with backward fill
    df['temperature'] = df['temperature'].fillna(method='bfill')
    
    return df['temperature']


def split_data(X: np.ndarray, y: np.ndarray, train_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and testing sets.
    
    Parameters:
        X: Input sequences
        y: Target values
        train_ratio: Ratio of training data
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    split_index = int(len(X) * train_ratio)
    
    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]
    
    return X_train, X_test, y_train, y_test


def plot_results(y_true: np.ndarray, y_pred: np.ndarray, title: str = "LSTM Predictions") -> None:
    """
    Plot prediction results.
    
    Parameters:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
    """
    plt.figure(figsize=(15, 5))
    
    # Plot predictions vs actual
    plt.subplot(1, 3, 1)
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Temperature (°C)')
    plt.ylabel('Predicted Temperature (°C)')
    plt.title(f'{title} - Predictions vs Actual')
    plt.grid(True, alpha=0.3)
    
    # Plot time series
    plt.subplot(1, 3, 2)
    plt.plot(y_true, label='Actual', alpha=0.8)
    plt.plot(y_pred, label='Predicted', alpha=0.8)
    plt.xlabel('Time Steps')
    plt.ylabel('Temperature (°C)')
    plt.title(f'{title} - Time Series')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot residuals
    plt.subplot(1, 3, 3)
    residuals = y_true - y_pred
    plt.scatter(range(len(residuals)), residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Time Steps')
    plt.ylabel('Residuals (°C)')
    plt.title(f'{title} - Residuals')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/advanced_lstm_pred_vs_actual.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_error_distribution(errors: np.ndarray) -> None:
    """
    Plot error distribution histogram.
    
    Parameters:
        errors: Prediction errors (y_true - y_pred)
    """
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Prediction Error (°C)')
    plt.ylabel('Frequency')
    plt.title('LSTM Prediction Error Distribution')
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    plt.axvline(mean_error, color='red', linestyle='--', 
                label=f'Mean: {mean_error:.2f}°C')
    plt.axvline(mean_error + std_error, color='orange', linestyle='--', 
                label=f'+1 STD: {mean_error + std_error:.2f}°C')
    plt.axvline(mean_error - std_error, color='orange', linestyle='--', 
                label=f'-1 STD: {mean_error - std_error:.2f}°C')
    
    plt.legend()
    plt.savefig('figures/advanced_lstm_error_hist.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_training_history(history: tf.keras.callbacks.History) -> None:
    """
    Plot training history.
    
    Parameters:
        history: Keras training history
    """
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE (°C)')
    plt.title('Training and Validation MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/advanced_lstm_training_history.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main function to run the advanced LSTM model."""
    print("=== Advanced LSTM Model for HKO Temperature Prediction ===")
    
    # Configuration
    SEQUENCE_LENGTH = 30  # Use 30 days to predict next day
    LSTM_UNITS = 64  # Number of LSTM units (within 50-100 range)
    NUM_LAYERS = 2  # Number of LSTM layers (1-2)
    DROPOUT_RATE = 0.2  # Dropout rate for regularization
    MISSING_VALUE_METHOD = 'interpolate'  # 'forward_fill' or 'interpolate'
    EPOCHS = 50
    BATCH_SIZE = 32
    
    # Load data
    print("\n1. Loading HKO data...")
    data_file = Path("daily_HKO_GMT_ALL.csv")
    
    try:
        records = read_hko_daily_csv(data_file)
        print(f"   Loaded {len(records)} records")
    except Exception as e:
        print(f"   Error loading data: {e}")
        return
    
    # Preprocess data
    print(f"\n2. Preprocessing data (missing value method: {MISSING_VALUE_METHOD})...")
    temperature_series = preprocess_data(records, method=MISSING_VALUE_METHOD)
    print(f"   Data range: {temperature_series.min():.1f}°C to {temperature_series.max():.1f}°C")
    print(f"   Data points: {len(temperature_series)}")
    
    # Prepare sequences
    print(f"\n3. Preparing sequences (sequence length: {SEQUENCE_LENGTH})...")
    X, y = prepare_sequences(temperature_series.values, sequence_length=SEQUENCE_LENGTH)
    print(f"   Generated {len(X)} sequences")
    
    # Normalize data
    print("\n4. Normalizing data...")
    scaler = MinMaxScaler()
    y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()
    
    # Split data
    print("\n5. Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y_scaled, train_ratio=0.8)
    print(f"   Training set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")
    
    # Create and train model
    print(f"\n6. Creating LSTM model ({LSTM_UNITS} units, {NUM_LAYERS} layers, {DROPOUT_RATE} dropout)...")
    model = create_lstm_model(sequence_length=SEQUENCE_LENGTH, 
                             lstm_units=LSTM_UNITS, 
                             num_layers=NUM_LAYERS, 
                             dropout_rate=DROPOUT_RATE)
    
    print("\nModel architecture:")
    model.summary()
    
    # Setup callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
    
    print(f"\n7. Training model for {EPOCHS} epochs...")
    history = model.fit(X_train, y_train, 
                       epochs=EPOCHS, 
                       batch_size=BATCH_SIZE,
                       validation_split=0.2,
                       callbacks=[early_stopping, reduce_lr],
                       verbose=1)
    
    # Make predictions
    print("\n8. Making predictions...")
    y_pred_scaled = model.predict(X_test, verbose=0)
    
    # Inverse transform predictions
    y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    mse = mean_squared_error(y_test_original, y_pred)
    mae = mean_absolute_error(y_test_original, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"\n9. Model Performance:")
    print(f"   RMSE: {rmse:.3f}°C")
    print(f"   MAE: {mae:.3f}°C")
    print(f"   MSE: {mse:.3f}°C²")
    
    # Plot results
    print("\n10. Generating plots...")
    plot_results(y_test_original, y_pred, title="Advanced LSTM")
    
    errors = y_test_original - y_pred
    plot_error_distribution(errors)
    
    plot_training_history(history)
    
    # Save model
    model_path = "figures/advanced_lstm_model.keras"
    model.save(model_path)
    print(f"\n11. Model saved to: {model_path}")
    
    print("\n=== Training Complete ===")
    print(f"Final RMSE: {rmse:.3f}°C")
    print(f"Plots saved: figures/advanced_lstm_pred_vs_actual.png, figures/advanced_lstm_error_hist.png, figures/advanced_lstm_training_history.png")


if __name__ == "__main__":
    main()