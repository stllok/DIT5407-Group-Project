"""
Baseline RNN Model for HKO Temperature Prediction

This script implements a simple RNN model (1-2 layers, 50-100 units) using Keras/TensorFlow
for predicting Hong Kong Observatory daily temperature data.
"""

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Import the existing HKO data reader
from hko_data_reader import HKODailyRecord, read_hko_daily_csv


def prepare_sequences(
    data: List[float], sequence_length: int = 30
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare sequences for RNN training.

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
        X.append(data[i - sequence_length : i])
        y.append(data[i])

    X = np.array(X)
    y = np.array(y)

    # Reshape X to have shape (samples, sequence_length, 1) for RNN
    X = X.reshape((X.shape[0], X.shape[1], 1))

    return X, y


def create_rnn_model(sequence_length: int = 30, units: int = 64) -> tf.keras.Model:
    """
    Create a simple RNN model with 1-2 layers and 50-100 units.

    Parameters:
        sequence_length: Length of input sequences
        units: Number of RNN units (50-100)

    Returns:
        Compiled Keras model
    """
    model = Sequential(
        [
            SimpleRNN(
                units,
                activation="tanh",
                return_sequences=True,
                input_shape=(sequence_length, 1),
            ),
            SimpleRNN(units // 2, activation="tanh"),
            Dense(1),  # Output layer for regression
        ]
    )

    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])

    return model


def preprocess_data(
    records: List[HKODailyRecord], method: str = "interpolate"
) -> pd.Series:
    """
    Preprocess HKO data by handling missing values.

    Parameters:
        records: List of HKODailyRecord objects
        method: Method for handling missing values ('forward_fill' or 'interpolate')

    Returns:
        Pandas Series with processed temperature data
    """
    # Convert records to DataFrame
    df = pd.DataFrame(
        [{"date": record.date, "temperature": record.value} for record in records]
    )

    # Set date as index and sort
    df = df.set_index("date").sort_index()

    # Handle missing values
    if method == "forward_fill":
        df["temperature"] = df["temperature"].fillna(method="ffill")
    elif method == "interpolate":
        df["temperature"] = df["temperature"].interpolate(method="linear")
    else:
        raise ValueError("Method must be 'forward_fill' or 'interpolate'")

    # Fill any remaining NaN values with backward fill
    df["temperature"] = df["temperature"].fillna(method="bfill")

    return df["temperature"]


def split_data(
    X: np.ndarray, y: np.ndarray, train_ratio: float = 0.8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def plot_results(
    y_true: np.ndarray, y_pred: np.ndarray, title: str = "RNN Predictions"
) -> None:
    """
    Plot prediction results.

    Parameters:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
    """
    plt.figure(figsize=(12, 6))

    # Plot predictions vs actual
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2)
    plt.xlabel("Actual Temperature (°C)")
    plt.ylabel("Predicted Temperature (°C)")
    plt.title(f"{title} - Predictions vs Actual")
    plt.grid(True, alpha=0.3)

    # Plot time series
    plt.subplot(1, 2, 2)
    plt.plot(y_true, label="Actual", alpha=0.8)
    plt.plot(y_pred, label="Predicted", alpha=0.8)
    plt.xlabel("Time Steps")
    plt.ylabel("Temperature (°C)")
    plt.title(f"{title} - Time Series")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/baseline_rnn_pred_vs_actual.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_error_distribution(errors: np.ndarray) -> None:
    """
    Plot error distribution histogram.

    Parameters:
        errors: Prediction errors (y_true - y_pred)
    """
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=30, alpha=0.7, edgecolor="black")
    plt.xlabel("Prediction Error (°C)")
    plt.ylabel("Frequency")
    plt.title("RNN Prediction Error Distribution")
    plt.grid(True, alpha=0.3)

    # Add statistics
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    plt.axvline(
        mean_error, color="red", linestyle="--", label=f"Mean: {mean_error:.2f}°C"
    )
    plt.axvline(
        mean_error + std_error,
        color="orange",
        linestyle="--",
        label=f"+1 STD: {mean_error + std_error:.2f}°C",
    )
    plt.axvline(
        mean_error - std_error,
        color="orange",
        linestyle="--",
        label=f"-1 STD: {mean_error - std_error:.2f}°C",
    )

    plt.legend()
    plt.savefig("figures/baseline_rnn_error_hist.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    """Main function to run the baseline RNN model."""
    print("=== Baseline RNN Model for HKO Temperature Prediction ===")

    # Configuration
    SEQUENCE_LENGTH = 30  # Use 30 days to predict next day
    RNN_UNITS = 100  # Number of RNN units (within 50-100 range)
    MISSING_VALUE_METHOD = "interpolate"  # 'forward_fill' or 'interpolate'
    EPOCHS = 50
    BATCH_SIZE = 128

    # Load data
    print("\n1. Loading HKO data...")
    data_file = Path("daily_HKO_GMT_ALL.csv")
    validation_file = Path("daily_HKO_GMT_2025.csv")

    try:
        records = read_hko_daily_csv(data_file)
        validator_records = read_hko_daily_csv(validation_file)
        print(f"   Loaded {len(records)} records")
        print(f"   Loaded {len(validator_records)} validation records")
    except Exception as e:
        print(f"   Error loading data: {e}")
        return

    # Preprocess data
    print(f"\n2. Preprocessing data (missing value method: {MISSING_VALUE_METHOD})...")
    temperature_series = preprocess_data(records, method=MISSING_VALUE_METHOD)
    validate_temperature_series = preprocess_data(
        validator_records, method=MISSING_VALUE_METHOD
    )
    print(
        f"   Training data range: {temperature_series.min():.1f}°C to {temperature_series.max():.1f}°C"
    )
    print(f"   Training data points: {len(temperature_series)}")
    print(
        f"   Validation data range: {validate_temperature_series.min():.1f}°C to {validate_temperature_series.max():.1f}°C"
    )
    print(f"   Validation data points: {len(validate_temperature_series)}")

    # Prepare sequences
    print(f"\n3. Preparing sequences (sequence length: {SEQUENCE_LENGTH})...")
    X_train, y_train = prepare_sequences(
        temperature_series.values, sequence_length=SEQUENCE_LENGTH
    )
    X_val, y_val = prepare_sequences(
        validate_temperature_series.values, sequence_length=SEQUENCE_LENGTH
    )
    print(f"   Training sequences: {len(X_train)} samples")
    print(f"   Validation sequences: {len(X_val)} samples")

    # Normalize data using training data's scaler
    print("\n4. Normalizing data...")
    scaler = MinMaxScaler()
    # Fit scaler on the full training temperature series and transform the data before creating sequences
    scaled_temperature_values = scaler.fit_transform(
        temperature_series.values.reshape(-1, 1)
    ).flatten()

    # Apply the same transformation to validation data
    scaled_validate_values = scaler.transform(
        validate_temperature_series.values.reshape(-1, 1)
    ).flatten()

    # Now create sequences using the scaled values
    X_train_scaled, y_train_scaled = prepare_sequences(
        scaled_temperature_values, sequence_length=SEQUENCE_LENGTH
    )
    X_val_scaled, y_val_scaled = prepare_sequences(
        scaled_validate_values, sequence_length=SEQUENCE_LENGTH
    )

    # Reshape X to have shape (samples, sequence_length, 1) for LSTM
    X_train_scaled = X_train_scaled.reshape(
        (X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    )
    X_val_scaled = X_val_scaled.reshape(
        (X_val_scaled.shape[0], X_val_scaled.shape[1], 1)
    )

    # Create and train model
    print(f"\n5. Creating RNN model ({RNN_UNITS} units)...")
    model = create_rnn_model(sequence_length=SEQUENCE_LENGTH, units=RNN_UNITS)

    print("\nModel architecture:")
    model.summary()

    print(f"\n6. Training model for {EPOCHS} epochs...")
    reduce_lr = ReduceLROnPlateau(
        monitor="val_mae",
        patience=3,
        # 3 epochs 內acc沒下降就要調整LR
        verbose=1,
        factor=0.5,
        # LR降為0.5
        min_lr=0.000001,
        # 最小 LR 到0.000001就不再下降
        mode="min",
    )
    history = model.fit(
        X_train_scaled,
        y_train_scaled,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val_scaled, y_val_scaled),  # Using validation dataset
        callbacks=[reduce_lr],
        verbose=1,
    )

    # Make predictions on validation set
    print("\n7. Making predictions on validation data...")
    y_pred_scaled = model.predict(X_val_scaled, verbose=0)

    # Inverse transform predictions
    y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
    y_val_original = scaler.inverse_transform(y_val_scaled.reshape(-1, 1)).flatten()

    # Calculate metrics
    mse = mean_squared_error(y_val_original, y_pred)
    mae = mean_absolute_error(y_val_original, y_pred)
    rmse = np.sqrt(mse)

    print("\n8. Model Performance on Validation Data:")
    print(f"   RMSE: {rmse:.3f}°C")
    print(f"   MAE: {mae:.3f}°C")
    print(f"   MSE: {mse:.3f}°C²")

    # Plot results
    print("\n9. Generating plots...")
    plot_results(y_val_original, y_pred, title="Baseline RNN - Validation Data")

    errors = y_val_original - y_pred
    plot_error_distribution(errors)

    # Save model and scaler
    import joblib

    model_path = "figures/baseline_rnn_model.keras"
    scaler_path = "figures/baseline_rnn_scaler.pkl"
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    print(f"\n10. Model saved to: {model_path}")
    print(f"    Scaler saved to: {scaler_path}")

    print("\n=== Training Complete ===")
    print(f"Final RMSE on Validation: {rmse:.3f}°C")
    print(
        "Plots saved: figures/baseline_rnn_pred_vs_actual.png, figures/baseline_rnn_error_hist.png"
    )


if __name__ == "__main__":
    main()
