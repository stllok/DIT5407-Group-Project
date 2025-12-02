import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import warnings
warnings.filterwarnings('ignore')

print("Loading models and scalers...")
# Load both models and their corresponding scalers
try:
    advanced_lstm_model = keras.models.load_model('figures/advanced_lstm_model.keras')
    advanced_lstm_scaler = joblib.load('figures/advanced_lstm_scaler.pkl')
    print("Advanced LSTM model and scaler loaded successfully!")
except:
    print("Advanced LSTM model or scaler not found, trying with original model...")
    advanced_lstm_model = keras.models.load_model('figures/advanced_lstm_model.keras')
    # We'll need to recreate the scaler
    print("Loading original training data to recreate scaler...")
    df_train = pd.read_csv('daily_HKO_GMT_ALL.csv')
    df_train['數值/Value'] = df_train['數值/Value'].replace('***', np.nan)
    df_train['數值/Value'] = pd.to_numeric(df_train['數值/Value'], errors='coerce')
    df_train_clean = df_train.dropna(subset=['數值/Value']).copy()
    train_temperatures = df_train_clean['數值/Value'].values
    
    from sklearn.preprocessing import MinMaxScaler
    advanced_lstm_scaler = MinMaxScaler()
    advanced_lstm_scaler.fit(train_temperatures.reshape(-1, 1))
    print("Recreated Advanced LSTM scaler from training data")

try:
    baseline_rnn_model = keras.models.load_model('figures/baseline_rnn_model.keras')
    baseline_rnn_scaler = joblib.load('figures/baseline_rnn_scaler.pkl')
    print("Baseline RNN model and scaler loaded successfully!")
except:
    print("Baseline RNN model or scaler not found, trying with original model...")
    baseline_rnn_model = keras.models.load_model('figures/baseline_rnn_model.keras')
    # We'll need to recreate the scaler
    print("Loading original training data to recreate scaler...")
    df_train = pd.read_csv('daily_HKO_GMT_ALL.csv')
    df_train['數值/Value'] = df_train['數值/Value'].replace('***', np.nan)
    df_train['數值/Value'] = pd.to_numeric(df_train['數值/Value'], errors='coerce')
    df_train_clean = df_train.dropna(subset=['數值/Value']).copy()
    train_temperatures = df_train_clean['數值/Value'].values
    
    from sklearn.preprocessing import MinMaxScaler
    baseline_rnn_scaler = MinMaxScaler()
    baseline_rnn_scaler.fit(train_temperatures.reshape(-1, 1))
    print("Recreated Baseline RNN scaler from training data")

print(f"Advanced LSTM model: {advanced_lstm_model.name if hasattr(advanced_lstm_model, 'name') else 'Unknown'}")
print(f"Baseline RNN model: {baseline_rnn_model.name if hasattr(baseline_rnn_model, 'name') else 'Unknown'}")

# Load the 2025 validation data
print("\nLoading 2025 validation data...")
df_2025 = pd.read_csv('daily_HKO_GMT_2025.csv')

# Process the data - handle missing values marked as '***'
df_2025['數值/Value'] = df_2025['數值/Value'].replace('***', np.nan)
df_2025['數值/Value'] = pd.to_numeric(df_2025['數值/Value'], errors='coerce')

print(f"Data shape before cleaning: {df_2025.shape}")
print(f"Missing values: {df_2025['數值/Value'].isna().sum()}")

# Remove rows with missing temperature values
df_2025_clean = df_2025.dropna(subset=['數值/Value']).copy()
print(f"Cleaned data shape: {df_2025_clean.shape}")

# Extract temperature values
temperatures = df_2025_clean['數值/Value'].values
print(f"Temperature data shape: {temperatures.shape}")
print(f"Temperature range: {temperatures.min():.2f}°C - {temperatures.max():.2f}°C")

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

print("\nCreating sequences...")
X, y_true = create_sequences(temperatures, seq_length=30)
print(f"Input sequences shape: {X.shape}")
print(f"Target values shape: {y_true.shape}")

# Scale the input sequences using the loaded scalers (both models should use the same scaler in theory)
# For consistency, I'll scale using the advanced LSTM scaler
X_scaled = np.zeros_like(X, dtype=np.float32)
for i in range(X.shape[0]):
    X_scaled[i] = advanced_lstm_scaler.transform(X[i].reshape(-1, 1)).flatten()

# y_true is the actual target values that we want to compare against
y_true_scaled = advanced_lstm_scaler.transform(y_true.reshape(-1, 1)).flatten()

# Reshape input to match model expectations (samples, timesteps, features)
X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
print(f"Reshaped input shape: {X_scaled.shape}")

# Make predictions using both models
print("\nMaking predictions with Advanced LSTM model...")
y_pred_lstm_scaled = advanced_lstm_model.predict(X_scaled, verbose=0)

print("Making predictions with Baseline RNN model...")
y_pred_rnn_scaled = baseline_rnn_model.predict(X_scaled, verbose=0)

# Inverse transform the predictions back to actual temperature values
y_pred_lstm = advanced_lstm_scaler.inverse_transform(y_pred_lstm_scaled).flatten()
y_pred_rnn = baseline_rnn_scaler.inverse_transform(y_pred_rnn_scaled).flatten()

# y_true is already in the original scale
y_true_original = advanced_lstm_scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()

print(f"LSTM predictions range: {y_pred_lstm.min():.2f}°C - {y_pred_lstm.max():.2f}°C")
print(f"RNN predictions range: {y_pred_rnn.min():.2f}°C - {y_pred_rnn.max():.2f}°C")
print(f"True values range: {y_true_original.min():.2f}°C - {y_true_original.max():.2f}°C")

# Calculate metrics
print("\nCalculating metrics...")

# MAE and RMSE for LSTM model
mae_lstm = mean_absolute_error(y_true_original, y_pred_lstm)
rmse_lstm = np.sqrt(mean_squared_error(y_true_original, y_pred_lstm))

# MAE and RMSE for RNN model
mae_rnn = mean_absolute_error(y_true_original, y_pred_rnn)
rmse_rnn = np.sqrt(mean_squared_error(y_true_original, y_pred_rnn))

print(f"\nModel Evaluation Results:")
print(f"Advanced LSTM Model:")
print(f"  MAE: {mae_lstm:.4f}°C")
print(f"  RMSE: {rmse_lstm:.4f}°C")

print(f"\nBaseline RNN Model:")
print(f"  MAE: {mae_rnn:.4f}°C")
print(f"  RMSE: {rmse_rnn:.4f}°C")

# Calculate errors for visualization
lstm_errors = y_true_original - y_pred_lstm
rnn_errors = y_true_original - y_pred_rnn

print(f"\nError statistics:")
print(f"LSTM Mean Error: {np.mean(lstm_errors):.4f}°C, Std Error: {np.std(lstm_errors):.4f}°C")
print(f"RNN Mean Error: {np.mean(rnn_errors):.4f}°C, Std Error: {np.std(rnn_errors):.4f}°C")

# Create visualizations
plt.style.use('default')
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Model Evaluation: Advanced LSTM vs Baseline RNN on 2025 Data', fontsize=16)

# 1. Actual vs Predicted temperatures plot for LSTM
axes[0, 0].scatter(y_true_original, y_pred_lstm, alpha=0.6, color='blue', label='Predictions')
axes[0, 0].plot([y_true_original.min(), y_true_original.max()], [y_true_original.min(), y_true_original.max()], 'r--', lw=2, label='Perfect Prediction')
axes[0, 0].set_xlabel('Actual Temperature (°C)')
axes[0, 0].set_ylabel('Predicted Temperature (°C)')
axes[0, 0].set_title(f'Advanced LSTM: Actual vs Predicted\nMAE: {mae_lstm:.4f}, RMSE: {rmse_lstm:.4f}')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Actual vs Predicted temperatures plot for RNN
axes[0, 1].scatter(y_true_original, y_pred_rnn, alpha=0.6, color='green', label='Predictions')
axes[0, 1].plot([y_true_original.min(), y_true_original.max()], [y_true_original.min(), y_true_original.max()], 'r--', lw=2, label='Perfect Prediction')
axes[0, 1].set_xlabel('Actual Temperature (°C)')
axes[0, 1].set_ylabel('Predicted Temperature (°C)')
axes[0, 1].set_title(f'Baseline RNN: Actual vs Predicted\nMAE: {mae_rnn:.4f}, RMSE: {rmse_rnn:.4f}')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Error distributions
axes[1, 0].hist(lstm_errors, bins=30, alpha=0.6, label='LSTM Errors', color='blue', density=True)
axes[1, 0].hist(rnn_errors, bins=30, alpha=0.6, label='RNN Errors', color='green', density=True)
axes[1, 0].axvline(x=0, color='red', linestyle='--', label='Zero Error Line')
axes[1, 0].set_xlabel('Prediction Error (°C)')
axes[1, 0].set_ylabel('Density')
axes[1, 0].set_title('Distribution of Prediction Errors')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Time series plot of predictions vs actual
axes[1, 1].plot(y_true_original, label='Actual Temperature', color='black', linewidth=2)
axes[1, 1].plot(y_pred_lstm, label=f'LSTM Predictions (MAE: {mae_lstm:.4f})', alpha=0.8)
axes[1, 1].plot(y_pred_rnn, label=f'RNN Predictions (MAE: {mae_rnn:.4f})', alpha=0.8)
axes[1, 1].set_xlabel('Time Index')
axes[1, 1].set_ylabel('Temperature (°C)')
axes[1, 1].set_title('Time Series: Actual vs Predicted Temperatures')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/model_comparison_2025_fixed.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nActual vs Predicted temperatures plot saved as 'figures/model_comparison_2025_fixed.png'")

# Create separate error distribution visualization
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(lstm_errors, bins=30, alpha=0.7, label=f'LSTM Errors\n(MAE: {mae_lstm:.4f})', color='blue', density=True)
plt.axvline(x=0, color='red', linestyle='--', label='Zero Error')
plt.xlabel('Prediction Error (°C)')
plt.ylabel('Density')
plt.title('LSTM Error Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(rnn_errors, bins=30, alpha=0.7, label=f'RNN Errors\n(MAE: {mae_rnn:.4f})', color='green', density=True)
plt.axvline(x=0, color='red', linestyle='--', label='Zero Error')
plt.xlabel('Prediction Error (°C)')
plt.ylabel('Density')
plt.title('RNN Error Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/error_distributions_fixed.png', dpi=300, bbox_inches='tight')
plt.show()

print("Error distribution plot saved as 'figures/error_distributions_fixed.png'")

# Calculate and visualize confidence intervals
def calculate_confidence_intervals(errors, confidence_level=0.95):
    """Calculate confidence intervals based on prediction errors"""
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    lower_bound = np.percentile(errors, lower_percentile)
    upper_bound = np.percentile(errors, upper_percentile)
    return lower_bound, upper_bound

lstm_conf_lower, lstm_conf_upper = calculate_confidence_intervals(lstm_errors)
rnn_conf_lower, rnn_conf_upper = calculate_confidence_intervals(rnn_errors)

# Create confidence interval visualization
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.fill_between(range(len(y_true_original)), y_true_original + lstm_conf_lower, y_true_original + lstm_conf_upper, 
                 alpha=0.2, color='blue', label='95% Confidence Interval')
plt.plot(y_true_original, label='Actual Temperature', color='black', linewidth=2)
plt.plot(y_pred_lstm, label='LSTM Predictions', color='blue', alpha=0.8)
plt.xlabel('Time Index')
plt.ylabel('Temperature (°C)')
plt.title(f'LSTM Predictions with Confidence Intervals\n95% CI: [{lstm_conf_lower:.3f}, {lstm_conf_upper:.3f}]')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.fill_between(range(len(y_true_original)), y_true_original + rnn_conf_lower, y_true_original + rnn_conf_upper, 
                 alpha=0.2, color='green', label='95% Confidence Interval')
plt.plot(y_true_original, label='Actual Temperature', color='black', linewidth=2)
plt.plot(y_pred_rnn, label='RNN Predictions', color='green', alpha=0.8)
plt.xlabel('Time Index')
plt.ylabel('Temperature (°C)')
plt.title(f'RNN Predictions with Confidence Intervals\n95% CI: [{rnn_conf_lower:.3f}, {rnn_conf_upper:.3f}]')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/confidence_intervals_fixed.png', dpi=300, bbox_inches='tight')
plt.show()

print("Confidence intervals plot saved as 'figures/confidence_intervals_fixed.png'")

# Generate final analysis report
print("\n" + "="*60)
print("FINAL ANALYSIS REPORT")
print("="*60)

print(f"\nDataset Information:")
print(f"- Validation dataset: 2025 daily temperature data")
print(f"- Total data points: {len(y_true_original)}")
print(f"- Input sequence length: 30 days")
print(f"- Output: Next day's minimum temperature")

print(f"\nModel Performance Comparison:")
print(f"- Advanced LSTM Model:")
print(f"  * MAE: {mae_lstm:.4f}°C")
print(f"  * RMSE: {rmse_lstm:.4f}°C")
print(f"  * Mean Error: {np.mean(lstm_errors):.4f}°C")
print(f"  * Std Error: {np.std(lstm_errors):.4f}°C")

print(f"\n- Baseline RNN Model:")
print(f"  * MAE: {mae_rnn:.4f}°C")
print(f"  * RMSE: {rmse_rnn:.4f}°C")
print(f"  * Mean Error: {np.mean(rnn_errors):.4f}°C")
print(f"  * Std Error: {np.std(rnn_errors):.4f}°C")

# Determine better model based on both metrics
lstm_better_mae = mae_lstm < mae_rnn
lstm_better_rmse = rmse_lstm < rmse_rnn

print(f"\nPerformance Analysis:")
if lstm_better_mae and lstm_better_rmse:
    print(f"- Advanced LSTM outperforms Baseline RNN on both metrics")
elif not lstm_better_mae and not lstm_better_rmse:
    print(f"- Baseline RNN outperforms Advanced LSTM on both metrics")
else:
    print(f"- Mixed results: LSTM performs better on some metrics, RNN on others")

print(f"\nLSTM vs RNN Improvement:")
mae_improvement = ((mae_rnn - mae_lstm) / mae_rnn) * 100
rmse_improvement = ((rmse_rnn - rmse_lstm) / rmse_rnn) * 100

print(f"- MAE Improvement: {mae_improvement:.2f}%")
print(f"- RMSE Improvement: {rmse_improvement:.2f}%")

# Confidence interval analysis
print(f"\nConfidence Interval Analysis:")
print(f"- LSTM 95% CI: [{lstm_conf_lower:.3f}, {lstm_conf_upper:.3f}]")
print(f"- RNN 95% CI: [{rnn_conf_lower:.3f}, {rnn_conf_upper:.3f}]")

print(f"\nError Distribution Analysis:")
lstm_bias = "overestimates" if np.mean(lstm_errors) > 0 else "underestimates"
rnn_bias = "overestimates" if np.mean(rnn_errors) > 0 else "underestimates"
print(f"- LSTM {lstm_bias} temperatures by {np.mean(lstm_errors):.4f}°C on average (positive = overestimate, negative = underestimate)")
print(f"- RNN {rnn_bias} temperatures by {np.mean(rnn_errors):.4f}°C on average (positive = overestimate, negative = underestimate)")

print(f"\nConclusion:")
if mae_lstm < mae_rnn:
    print(f"- The Advanced LSTM model performs better with lower MAE, indicating more accurate predictions on average.")
else:
    print(f"- The Baseline RNN model performs better with lower MAE, indicating more accurate predictions on average.")

print(f"\nAll visualizations and evaluation results have been saved to the 'figures' directory.")
print("="*60)

# Save detailed metrics to a CSV file
results_df = pd.DataFrame({
    'Model': ['Advanced LSTM', 'Baseline RNN'],
    'MAE': [mae_lstm, mae_rnn],
    'RMSE': [rmse_lstm, rmse_rnn],
    'Mean_Error': [np.mean(lstm_errors), np.mean(rnn_errors)],
    'Std_Error': [np.std(lstm_errors), np.std(rnn_errors)]
})

results_df.to_csv('forecasting_results/model_evaluation_results_fixed.csv', index=False)
print(f"\nDetailed evaluation results saved to 'forecasting_results/model_evaluation_results_fixed.csv'")

print(f"\nScript execution completed successfully!")