"""
Bidirectional LSTM Forecasting Script for HKO Temperature Prediction

This script implements a sequence-to-one prediction model using bidirectional LSTM
for forecasting Hong Kong Observatory daily temperature data. It includes:
- Sequence-to-one prediction (30 days → next day's temperature)
- Bidirectional LSTM architecture
- Hyperparameter tuning with Keras Tuner
- Missing value handling with forward-fill and interpolation
- 50-100 epochs with Adam optimizer and MSE loss
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional
import argparse
import json

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import keras_tuner as kt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import the existing HKO data reader
from hko_data_reader import read_hko_daily_csv, HKODailyRecord


def load_and_preprocess_data(file_path: Path, missing_value_strategy: str = "interpolate") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load HKO data and handle missing values.
    
    Parameters:
        file_path: Path to the CSV file
        missing_value_strategy: Strategy for handling missing values ("forward_fill" or "interpolate")
    
    Returns:
        dates: Array of dates
        values: Array of temperature values with missing values handled
    """
    print(f"Loading data from {file_path}...")
    
    # Read data using existing reader function
    records = read_hko_daily_csv(file_path)
    
    # Extract dates and values, filtering out None values
    valid_records = [(record.date, record.value) for record in records if record.value is not None]
    
    if not valid_records:
        raise ValueError("No valid temperature records found in the data")
    
    dates = np.array([record[0] for record in valid_records])
    values = np.array([record[1] for record in valid_records], dtype=float)
    
    # Handle missing values using pandas for better handling
    values_series = pd.Series(values)
    
    if missing_value_strategy == "forward_fill":
        values_series = values_series.fillna(method='ffill')
    elif missing_value_strategy == "interpolate":
        values_series = values_series.interpolate(method='linear')
    else:
        raise ValueError(f"Unknown missing value strategy: {missing_value_strategy}")
    
    values = values_series.values
    
    # Remove any remaining NaN values
    valid_mask = ~np.isnan(values)
    dates = dates[valid_mask]
    values = values[valid_mask]
    
    print(f"Loaded {len(values)} valid temperature records")
    print(f"Date range: {dates[0]} to {dates[-1]}")
    print(f"Temperature range: {values.min():.1f}°C to {values.max():.1f}°C")
    
    return dates, values


def create_sequences(data: np.ndarray, sequence_length: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for sequence-to-one prediction.
    
    Parameters:
        data: Time series data
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


def split_data(X: np.ndarray, y: np.ndarray, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training, validation, and test sets.
    
    Parameters:
        X: Input sequences
        y: Target values
        train_ratio: Ratio for training data
        val_ratio: Ratio for validation data
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    n_samples = len(X)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_val = X[n_train:n_train + n_val]
    y_val = y[n_train:n_train + n_val]
    X_test = X[n_train + n_val:]
    y_test = y[n_train + n_val:]
    
    print(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


class TemperatureHyperModel(kt.HyperModel):
    """
    HyperModel for temperature forecasting with bidirectional LSTM.
    """
    
    def __init__(self, input_shape: Tuple[int, int]):
        self.input_shape = input_shape
    
    def build(self, hp):
        model = Sequential()
        
        # Bidirectional LSTM layers
        lstm_units_1 = hp.Int('lstm_units_1', min_value=32, max_value=128, step=32)
        lstm_units_2 = hp.Int('lstm_units_2', min_value=16, max_value=64, step=16)
        
        # First bidirectional LSTM layer
        model.add(Bidirectional(LSTM(
            units=lstm_units_1,
            return_sequences=True,
            input_shape=self.input_shape
        )))
        
        dropout_rate_1 = hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)
        model.add(Dropout(dropout_rate_1))
        
        # Second bidirectional LSTM layer
        model.add(Bidirectional(LSTM(units=lstm_units_2)))
        
        dropout_rate_2 = hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1)
        model.add(Dropout(dropout_rate_2))
        
        # Dense output layer
        model.add(Dense(1))
        
        # Learning rate
        learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model


def build_model(input_shape: Tuple[int, int], lstm_units_1: int = 64, lstm_units_2: int = 32,
                dropout_1: float = 0.2, dropout_2: float = 0.2, learning_rate: float = 0.001) -> tf.keras.Model:
    """
    Build bidirectional LSTM model with specified hyperparameters.
    
    Parameters:
        input_shape: Shape of input sequences
        lstm_units_1: Units in first LSTM layer
        lstm_units_2: Units in second LSTM layer
        dropout_1: Dropout rate after first LSTM
        dropout_2: Dropout rate after second LSTM
        learning_rate: Learning rate for Adam optimizer
    
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        Bidirectional(LSTM(units=lstm_units_1, return_sequences=True, input_shape=input_shape)),
        Dropout(dropout_1),
        Bidirectional(LSTM(units=lstm_units_2)),
        Dropout(dropout_2),
        Dense(1)
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def train_model(model: tf.keras.Model, X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray, epochs: int = 100, batch_size: int = 32) -> tf.keras.callbacks.History:
    """
    Train the model with early stopping and learning rate reduction.
    
    Parameters:
        model: Keras model to train
        X_train: Training input sequences
        y_train: Training target values
        X_val: Validation input sequences
        y_val: Validation target values
        epochs: Maximum number of epochs
        batch_size: Batch size for training
    
    Returns:
        Training history
    """
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]
    
    print(f"Training model for up to {epochs} epochs...")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return history


def evaluate_model(model: tf.keras.Model, X_test: np.ndarray, y_test: np.ndarray, scaler: MinMaxScaler) -> dict:
    """
    Evaluate model performance on test data.
    
    Parameters:
        model: Trained Keras model
        X_test: Test input sequences
        y_test: Test target values
        scaler: Scaler used for normalization
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test, verbose=0)
    
    # Inverse transform predictions and targets
    y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    mse = mean_squared_error(y_test_original, y_pred_original)
    mae = mean_absolute_error(y_test_original, y_pred_original)
    rmse = np.sqrt(mse)
    
    metrics = {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'y_pred': y_pred_original,
        'y_test': y_test_original
    }
    
    return metrics


def plot_results(history: tf.keras.callbacks.History, metrics: dict, save_path: Optional[Path] = None):
    """
    Plot training history and prediction results.
    
    Parameters:
        history: Training history
        metrics: Evaluation metrics
        save_path: Path to save plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training history
    axes[0, 0].plot(history.history['loss'], label='Training Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MSE Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # MAE history
    axes[0, 1].plot(history.history['mae'], label='Training MAE')
    axes[0, 1].plot(history.history['val_mae'], label='Validation MAE')
    axes[0, 1].set_title('Model MAE')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Predictions vs Actual
    y_test = metrics['y_test']
    y_pred = metrics['y_pred']
    axes[1, 0].scatter(y_test, y_pred, alpha=0.6)
    axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[1, 0].set_xlabel('Actual Temperature (°C)')
    axes[1, 0].set_ylabel('Predicted Temperature (°C)')
    axes[1, 0].set_title('Predictions vs Actual')
    axes[1, 0].grid(True)
    
    # Residuals
    residuals = y_test - y_pred
    axes[1, 1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Residuals (°C)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Residual Distribution')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plots saved to {save_path}")
    else:
        plt.show()


def perform_hyperparameter_tuning(X_train: np.ndarray, y_train: np.ndarray,
                                  X_val: np.ndarray, y_val: np.ndarray,
                                  max_trials: int = 20, epochs: int = 50) -> dict:
    """
    Perform hyperparameter tuning using Keras Tuner.
    
    Parameters:
        X_train: Training input sequences
        y_train: Training target values
        X_val: Validation input sequences
        y_val: Validation target values
        max_trials: Maximum number of hyperparameter combinations to try
        epochs: Number of epochs per trial
    
    Returns:
        Best hyperparameters
    """
    print(f"Performing hyperparameter tuning with {max_trials} trials...")
    
    hypermodel = TemperatureHyperModel(input_shape=(X_train.shape[1], X_train.shape[2]))
    
    tuner = kt.RandomSearch(
        hypermodel,
        objective='val_loss',
        max_trials=max_trials,
        directory='hyperparameter_tuning',
        project_name='temperature_forecasting'
    )
    
    # Perform search
    tuner.search(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=[EarlyStopping(monitor='val_loss', patience=5)],
        verbose=0
    )
    
    # Get best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    print("Best hyperparameters found:")
    print(f"  LSTM units (layer 1): {best_hps.get('lstm_units_1')}")
    print(f"  LSTM units (layer 2): {best_hps.get('lstm_units_2')}")
    print(f"  Dropout rate 1: {best_hps.get('dropout_1'):.2f}")
    print(f"  Dropout rate 2: {best_hps.get('dropout_2'):.2f}")
    print(f"  Learning rate: {best_hps.get('learning_rate')}")
    
    return best_hps


def main():
    parser = argparse.ArgumentParser(description='HKO Temperature Forecasting with Bidirectional LSTM')
    parser.add_argument('--data-file', type=str, default='daily_HKO_GMT_ALL.csv',
                        help='Path to HKO data file')
    parser.add_argument('--sequence-length', type=int, default=30,
                        help='Length of input sequences')
    parser.add_argument('--missing-strategy', type=str, default='interpolate',
                        choices=['forward_fill', 'interpolate'],
                        help='Strategy for handling missing values')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--tune-hyperparameters', action='store_true',
                        help='Perform hyperparameter tuning')
    parser.add_argument('--max-trials', type=int, default=20,
                        help='Maximum trials for hyperparameter tuning')
    parser.add_argument('--output-dir', type=str, default='forecasting_results',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Load and preprocess data
        data_file = Path(args.data_file)
        dates, values = load_and_preprocess_data(data_file, args.missing_strategy)
        
        # Normalize data
        scaler = MinMaxScaler()
        values_scaled = scaler.fit_transform(values.reshape(-1, 1)).flatten()
        
        # Create sequences
        print(f"Creating sequences with length {args.sequence_length}...")
        X, y = create_sequences(values_scaled, args.sequence_length)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
        
        if args.tune_hyperparameters:
            # Perform hyperparameter tuning
            best_hps = perform_hyperparameter_tuning(
                X_train, y_train, X_val, y_val,
                max_trials=args.max_trials, epochs=50
            )
            
            # Build model with best hyperparameters
            model = build_model(
                input_shape=(X_train.shape[1], X_train.shape[2]),
                lstm_units_1=best_hps.get('lstm_units_1'),
                lstm_units_2=best_hps.get('lstm_units_2'),
                dropout_1=best_hps.get('dropout_1'),
                dropout_2=best_hps.get('dropout_2'),
                learning_rate=best_hps.get('learning_rate')
            )
        else:
            # Use default hyperparameters
            model = build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        # Train model
        history = train_model(
            model, X_train, y_train, X_val, y_val,
            epochs=args.epochs, batch_size=args.batch_size
        )
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test, scaler)
        
        # Print results
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        print(f"Test MSE: {metrics['mse']:.4f}")
        print(f"Test MAE: {metrics['mae']:.4f}")
        print(f"Test RMSE: {metrics['rmse']:.4f}")
        
        # Save results
        results = {
            'mse': float(metrics['mse']),
            'mae': float(metrics['mae']),
            'rmse': float(metrics['rmse']),
            'sequence_length': args.sequence_length,
            'missing_strategy': args.missing_strategy,
            'epochs': len(history.history['loss']),
            'batch_size': args.batch_size
        }
        
        if args.tune_hyperparameters:
            results.update({
                'lstm_units_1': int(best_hps.get('lstm_units_1')),
                'lstm_units_2': int(best_hps.get('lstm_units_2')),
                'dropout_1': float(best_hps.get('dropout_1')),
                'dropout_2': float(best_hps.get('dropout_2')),
                'learning_rate': float(best_hps.get('learning_rate'))
            })
        
        # Save results to JSON
        results_file = output_dir / 'results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_file}")
        
        # Save model
        model_file = output_dir / 'bidirectional_lstm_model.keras'
        model.save(model_file)
        print(f"Model saved to {model_file}")
        
        # Plot results
        plot_file = output_dir / 'forecasting_results.png'
        plot_results(history, metrics, plot_file)
        
        print(f"\nAll results saved to {output_dir}/")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())