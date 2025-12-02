import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# Load the 2025 validation data
df_2025 = pd.read_csv('daily_HKO_GMT_2025.csv')

# Process the data - handle missing values marked as '***'
df_2025['數值/Value'] = df_2025['數值/Value'].replace('***', np.nan)
df_2025['數值/Value'] = pd.to_numeric(df_2025['數值/Value'], errors='coerce')

# Remove rows with missing temperature values
df_2025_clean = df_2025.dropna(subset=['數值/Value']).copy()
temperatures = df_2025_clean['數值/Value'].values

print(f"Temperature stats:")
print(f"Min: {temperatures.min()}")
print(f"Max: {temperatures.max()}")
print(f"Mean: {temperatures.mean():.4f}")
print(f"Std: {temperatures.std():.4f}")

# Check the first few values
print(f"\nFirst 10 temperature values: {temperatures[:10]}")

# Prepare sequence-to-one data (30 days input -> next day temperature output)
def create_sequences(data, seq_length=30):
    """
    Create sequences for sequence-to-one prediction
    Each input sequence has seq_length days, output is the next day's temperature
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

X, y_true = create_sequences(temperatures, seq_length=30)
print(f"\nAfter creating sequences:")
print(f"X shape: {X.shape}")
print(f"y_true shape: {y_true.shape}")
print(f"y_true stats: min={y_true.min():.4f}, max={y_true.max():.4f}, mean={y_true.mean():.4f}")

# Reshape input to match model expectations (samples, timesteps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))
print(f"X reshaped: {X.shape}")

# Load models and make predictions
advanced_lstm_model = keras.models.load_model('figures/advanced_lstm_model.keras')
baseline_rnn_model = keras.models.load_model('figures/baseline_rnn_model.keras')

y_pred_lstm = advanced_lstm_model.predict(X, verbose=0).flatten()
y_pred_rnn = baseline_rnn_model.predict(X, verbose=0).flatten()

print(f"\nLSTM predictions stats: min={y_pred_lstm.min():.4f}, max={y_pred_lstm.max():.4f}, mean={y_pred_lstm.mean():.4f}")
print(f"RNN predictions stats: min={y_pred_rnn.min():.4f}, max={y_pred_rnn.max():.4f}, mean={y_pred_rnn.mean():.4f}")

# Calculate errors manually
lstm_errors = y_true - y_pred_lstm
rnn_errors = y_true - y_pred_rnn

print(f"\nLSTM errors stats: min={lstm_errors.min():.4f}, max={lstm_errors.max():.4f}, mean={lstm_errors.mean():.4f}")
print(f"RNN errors stats: min={rnn_errors.min():.4f}, max={rnn_errors.max():.4f}, mean={rnn_errors.mean():.4f}")