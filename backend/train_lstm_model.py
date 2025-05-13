import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os

# Define directories
DATA_DIR = r"C:\Users\algha\OneDrive\Documents\Market_Eye\data"
FORECASTS_DIR = r"C:\Users\algha\OneDrive\Documents\Market_Eye\forecasts"

# Create forecasts directory if it doesn't exist
if not os.path.exists(FORECASTS_DIR):
    os.makedirs(FORECASTS_DIR)

def create_sequences(data, seq_length):
    """Vectorized creation of sequences for LSTM input."""
    num_samples = len(data) - seq_length
    if num_samples <= 0:
        return np.array([]), np.array([])  # Return empty arrays if not enough data

    # Create indices for sequences
    indices = np.arange(num_samples)[:, None] + np.arange(seq_length + 1)
    sequences = data[indices]
    
    X = sequences[:, :-1]  # All but the last element in each sequence
    y = sequences[:, -1]    # Last element in each sequence (target)
    return X, y

def train_lstm_model(ticker):
    """Train an LSTM model on historical stock data for the given ticker."""
    # Load the stock data
    file_path = os.path.join(DATA_DIR, f"{ticker}_stock_data.csv")
    if not os.path.exists(file_path):
        print(f"No data file found for {ticker} at {file_path}")
        return None, None
    
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter data to the last 5 years to reduce training time
    end_date = df['date'].max()
    start_date = end_date - pd.Timedelta(days=5*365)  # Approximately 5 years
    df = df[df['date'] >= start_date].copy()

    if df.empty:
        print(f"No data available for {ticker} in the last 5 years.")
        return None, None

    # Prepare the data for LSTM
    data = df[['close']].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # Create sequences for training (reduced sequence length)
    sequence_length = 30  # Reduced from 60 to 30
    X, y = create_sequences(scaled_data, sequence_length)

    if X.size == 0 or y.size == 0:
        print(f"Not enough data to create sequences for {ticker} with sequence length {sequence_length}.")
        return None, None

    # Split into training and validation sets (80-20 split)
    train_size = int(len(X) * 0.8)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    # Build the LSTM model (simplified architecture)
    model = Sequential([
        LSTM(32, input_shape=(sequence_length, 1)),  # Reduced to 1 layer with 32 units
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    # Train the model (reduced epochs and increased batch size)
    model.fit(X_train, y_train, epochs=3, batch_size=64, validation_data=(X_val, y_val), verbose=1)

    return model, scaler

def forecast_january_2025(ticker):
    """Generate forecasts for January 2025 using the trained LSTM model."""
    # Load the trained model and scaler
    model, scaler = train_lstm_model(ticker)
    if model is None or scaler is None:
        print(f"Failed to train model for {ticker}.")
        return

    # Load the stock data again
    file_path = os.path.join(DATA_DIR, f"{ticker}_stock_data.csv")
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])

    # Prepare the last 30 days of data for forecasting (match sequence length)
    sequence_length = 30  # Match the reduced sequence length
    last_sequence = scaler.transform(df[['close']].tail(sequence_length).values)
    last_sequence = last_sequence.reshape((1, sequence_length, 1))

    # Generate forecast dates for January 2025
    forecast_dates = pd.date_range(start="2025-01-01", end="2025-01-31", freq='D')
    forecast_values = []

    current_sequence = last_sequence.copy()
    for _ in range(len(forecast_dates)):
        predicted_scaled = model.predict(current_sequence, verbose=0)
        forecast_values.append(predicted_scaled[0, 0])
        
        # Update the sequence with the predicted value
        current_sequence = np.roll(current_sequence, -1, axis=1)
        current_sequence[0, -1, 0] = predicted_scaled[0, 0]

    # Inverse transform the forecasted values
    forecast_values = np.array(forecast_values).reshape(-1, 1)
    forecast_values = scaler.inverse_transform(forecast_values).flatten()

    # Create a DataFrame for the forecast
    forecast_df = pd.DataFrame({
        'date': forecast_dates,
        'forecasted_close': forecast_values
    })

    # Save the forecast to a CSV file
    forecast_file_path = os.path.join(FORECASTS_DIR, f"{ticker}_jan2025_forecast.csv")
    forecast_df.to_csv(forecast_file_path, index=False)
    print(f"Saved January 2025 forecast for {ticker} to {forecast_file_path}")