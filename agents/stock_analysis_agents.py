import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
import os
import google.generativeai as genai
import logging
from sklearn.metrics import mean_squared_error

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyCVHSezcMFg_zGhaNRt4wnrPZETfl6yEQQ"
genai.configure(api_key=GEMINI_API_KEY)

# Define directories (consistent with app.py and train_lstm_model.py)
DATA_DIR = r"C:\Users\algha\OneDrive\Documents\Market_Eye\data"
FORECASTS_DIR = r"C:\Users\algha\OneDrive\Documents\Market_Eye\forecasts"
KAGGLE_DATA_PATH = r"C:\Users\algha\OneDrive\Documents\Market_Eye\World-Stock-Prices-Dataset.csv"

# Ensure directories exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
if not os.path.exists(FORECASTS_DIR):
    os.makedirs(FORECASTS_DIR)

# Define sector mapping for tickers
SECTOR_MAPPING = {
    "AAPL": "Tech",
    "GOOGL": "Tech",
    "MSFT": "Tech",
    "AMZN": "Tech",
    "TSLA": "Tech",
    "JPM": "Finance",
    "NKE": "Sportswear"
}

# Standalone Function: Data Collection
def collect_stock_data(tickers, full_historical=True):
    """
    Collect stock data for specified tickers from the Kaggle dataset.
    
    Parameters:
    - tickers (list): List of stock tickers to collect data for.
    - full_historical (bool): If True, load data from Jan 3, 2000, to Dec 31, 2024 for analytics.
                             If False, load Jan 2025 data for forecast evaluation.
    
    Returns:
    - pd.DataFrame: Combined DataFrame containing stock data for the specified tickers and date range.
    """
    # Define date ranges based on full_historical parameter
    cutoff_date = pd.to_datetime("2024-12-31", utc=True) if full_historical else pd.to_datetime("2025-01-31", utc=True)
    start_date = pd.to_datetime("2000-01-03", utc=True) if full_historical else pd.to_datetime("2025-01-01", utc=True)

    # Load the Kaggle dataset directly
    logging.info(f"Loading Kaggle dataset from {KAGGLE_DATA_PATH} for {'historical analytics' if full_historical else 'January 2025 evaluation'}...")
    try:
        df = pd.read_csv(KAGGLE_DATA_PATH)
    except Exception as e:
        logging.error(f"Error loading Kaggle dataset: {e}")
        return None

    # Standardize column names to lowercase
    df = df.rename(columns={
        'Date': 'date',
        'Ticker': 'ticker',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })

    # Ensure required columns are present
    required_columns = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logging.error(f"Missing required columns in dataset: {missing_columns}")
        return None

    # Convert date column to datetime with UTC timezone
    df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce')
    if df['date'].isnull().any():
        logging.error("Some dates could not be converted to datetime format.")
        return None

    # Filter for specified tickers and date range
    df = df[df['ticker'].isin(tickers)]
    df = df[(df['date'] >= start_date) & (df['date'] <= cutoff_date)]

    if df.empty:
        logging.warning(f"No data found for tickers {tickers} between {start_date} and {cutoff_date}.")
        return None

    # Sort by date
    df = df.sort_values('date')
    logging.info(f"Collected {len(df)} rows for tickers {tickers} between {start_date} and {cutoff_date}.")
    return df

# Standalone Function: Data Processing, Forecasting, and Sector Comparison
def process_data_and_forecast(combined_data):
    if combined_data is None:
        return None

    # Compute analytics (high, low, trend, 2020 growth%, annual growth)
    tickers = combined_data['ticker'].unique() if 'ticker' in combined_data else ["Unknown"]
    analytics = {}
    for ticker in tickers:
        ticker_data = combined_data[combined_data['ticker'] == ticker].copy() if 'ticker' in combined_data else combined_data.copy()
        
        # Verify that 'date' is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(ticker_data['date']):
            logging.error(f"'date' column for {ticker} is not in datetime format: {ticker_data['date'].dtype}")
            ticker_data.loc[:, 'date'] = pd.to_datetime(ticker_data['date'], errors='coerce')
            if ticker_data['date'].isnull().any():
                logging.error(f"Failed to convert some dates for {ticker}. Null dates: {ticker_data[ticker_data['date'].isnull()]['date']}")
                continue
        
        # Compute 2020 growth%
        growth_2020 = 0
        year_2020_data = ticker_data[(ticker_data['date'].dt.year == 2020)]
        if not year_2020_data.empty:
            start_price = year_2020_data.iloc[0]['close']
            end_price = year_2020_data.iloc[-1]['close']
            if start_price != 0:  # Avoid division by zero
                growth_2020 = ((end_price - start_price) / start_price) * 100

        # Compute annual growth for each year
        annual_growth = {}
        years = ticker_data['date'].dt.year.unique()
        for year in years:
            year_data = ticker_data[ticker_data['date'].dt.year == year]
            if not year_data.empty:
                start_price = year_data.iloc[0]['close']
                end_price = year_data.iloc[-1]['close']
                if start_price != 0:
                    growth = ((end_price - start_price) / start_price) * 100
                    annual_growth[str(year)] = growth
                else:
                    annual_growth[str(year)] = 0

        # Compute recent metrics (last 60 days)
        recent_data = ticker_data.tail(60)
        high = recent_data['close'].max() if not recent_data.empty else 0
        low = recent_data['close'].min() if not recent_data.empty else 0
        trend = ((recent_data['close'].iloc[-1] - recent_data['close'].iloc[-30]) / recent_data['close'].iloc[-30]) * 100 if len(recent_data) >= 30 else 0
        avg_close = recent_data['close'].mean() if not recent_data.empty else 0

        analytics[ticker] = {
            "high": high,
            "low": low,
            "trend": trend,
            "growth_2020": growth_2020,
            "avg_close": avg_close,
            "annual_growth": annual_growth  # Add annual growth for each year
        }

    # Compute sector-level analytics
    sector_analytics = {}
    for sector in set(SECTOR_MAPPING.values()):
        sector_tickers = [ticker for ticker, s in SECTOR_MAPPING.items() if s == sector and ticker in analytics]
        if not sector_tickers:
            continue
        
        sector_growth_2020 = np.mean([analytics[ticker]["growth_2020"] for ticker in sector_tickers]) if sector_tickers else 0
        sector_trend = np.mean([analytics[ticker]["trend"] for ticker in sector_tickers]) if sector_tickers else 0
        sector_avg_close = np.mean([analytics[ticker]["avg_close"] for ticker in sector_tickers]) if sector_tickers else 0
        
        sector_analytics[sector] = {
            "average_2020_growth": sector_growth_2020,
            "average_price_trend": sector_trend,
            "average_closing_price": sector_avg_close,
            "tickers": sector_tickers
        }

    # Forecast using LSTM with historical data up to Dec 31, 2024
    forecasts = {}
    for ticker in tickers:
        ticker_data = combined_data[combined_data['ticker'] == ticker].copy() if 'ticker' in combined_data else combined_data.copy()
        data = ticker_data[['close']].values
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)

        sequence_length = 30
        X, y = create_sequences(scaled_data, sequence_length)
        if X.size == 0 or y.size == 0:
            logging.warning(f"Insufficient data to create sequences for {ticker}.")
            continue

        train_size = int(len(X) * 0.8)
        X_train, X_val = X[:train_size], X[train_size:]  # First 80% for training, last 20% for validation
        y_train, y_val = y[:train_size], y[train_size:]  # Match X_train and X_val splits

        # Updated LSTM model with Input layer to suppress warning
        model = Sequential([
            Input(shape=(sequence_length, 1)),  # Explicit Input layer
            LSTM(32),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=3, batch_size=64, validation_data=(X_val, y_val), verbose=1)

        last_sequence = scaler.transform(ticker_data[['close']].tail(sequence_length).values)
        last_sequence = last_sequence.reshape((1, sequence_length, 1))
        forecast_dates = pd.date_range(start="2025-01-01", end="2025-01-31", freq='D')
        forecast_values = []
        current_sequence = last_sequence.copy()
        for _ in range(len(forecast_dates)):
            predicted_scaled = model.predict(current_sequence, verbose=0)
            forecast_values.append(predicted_scaled[0, 0])
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, 0] = predicted_scaled[0, 0]

        forecast_values = np.array(forecast_values).reshape(-1, 1)
        forecast_values = scaler.inverse_transform(forecast_values).flatten()
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'forecasted_close': forecast_values,
            'ticker': ticker
        })
        forecast_file_path = os.path.join(FORECASTS_DIR, f"{ticker}_jan2025_forecast.csv")
        forecast_df.to_csv(forecast_file_path, index=False)
        forecasts[ticker] = forecast_df

    result = {
        "analytics": analytics,
        "forecasts": forecasts,
        "sector_analytics": sector_analytics
    }
    # Save analytics, forecasts, and sector analytics to JSON
    with open(os.path.join(FORECASTS_DIR, "analytics_and_forecasts.json"), 'w') as f:
        pd.Series(result).to_json(f)
    return result

def create_sequences(data, seq_length):
    num_samples = len(data) - seq_length
    if num_samples <= 0:
        return np.array([]), np.array([])
    indices = np.arange(num_samples)[:, None] + np.arange(seq_length + 1)
    sequences = data[indices]
    X = sequences[:, :-1]
    y = sequences[:, -1]
    return X, y

# Function to Generate Recommendations
def generate_recommendations(analytics_and_forecasts):
    if analytics_and_forecasts is None:
        return None

    recommendations = {}
    model = genai.GenerativeModel('gemini-1.5-pro')

    for ticker in analytics_and_forecasts['analytics'].keys():
        analytics = analytics_and_forecasts['analytics'][ticker]
        forecast_data = analytics_and_forecasts['forecasts'].get(ticker, [])
        
        # Compute average forecasted price if forecast data exists
        avg_forecasted_price = None
        if forecast_data:
            forecast_df = pd.DataFrame(forecast_data)
            if 'forecasted_close' in forecast_df.columns:
                avg_forecasted_price = forecast_df['forecasted_close'].mean()
        
        # Format the average forecasted price
        avg_forecasted_price_str = f"{avg_forecasted_price:.2f}" if avg_forecasted_price is not None else "N/A"
        
        # Prepare prompt with analytics and forecast data
        prompt = f"""
        You are a financial analyst providing investment recommendations for the stock {ticker}. Based on the following data, generate a market-trend summary and a Buy/Hold/Sell recommendation:

        - **Recent Analytics (Last 60 Days):**
          - Average Closing Price: ${analytics['avg_close']:.2f}
          - Highest Price: ${analytics['high']:.2f}
          - Lowest Price: ${analytics['low']:.2f}
          - Price Trend (Last 30 Days): {analytics['trend']:.2f}%
          - 2020 Growth Percentage: {analytics['growth_2020']:.2f}%

        - **January 2025 Forecast:**
          - Average Forecasted Price: {avg_forecasted_price_str} (if forecast data is available)

        Provide a concise market-trend summary and a clear Buy/Hold/Sell recommendation in a conversational tone.
        """
        
        try:
            logging.info(f"Generating recommendation for {ticker} using Gemini API...")
            response = model.generate_content(prompt)
            recommendation = response.text.strip()
            logging.info(f"Successfully generated recommendation for {ticker} using Gemini API.")
        except Exception as e:
            logging.warning(f"Failed to generate recommendation for {ticker} using Gemini API: {e}. Using mocked recommendation.")
            # Enhanced mocked recommendation if API call fails
            trend_direction = "positive" if analytics['trend'] > 0 else "negative or neutral"
            forecast_outlook = "continued growth" if avg_forecasted_price and avg_forecasted_price > analytics['avg_close'] else "potential decline or stability"
            recommendation_action = "Buy" if avg_forecasted_price and avg_forecasted_price > analytics['avg_close'] else "Hold"
            
            recommendation = (
                f"**Market Trend Summary:** {ticker} has shown a {trend_direction} trend with an average closing price of ${analytics['avg_close']:.2f} over the last 60 days. "
                f"The stock reached a high of ${analytics['high']:.2f} and a low of ${analytics['low']:.2f}, with a 30-day price trend of {analytics['trend']:.2f}%. "
                f"Historically, {ticker} achieved a {analytics['growth_2020']:.2f}% growth in 2020, reflecting its past performance. "
                f"Looking ahead, the average forecasted price for January 2025 is {avg_forecasted_price_str}, suggesting {forecast_outlook}.\n\n"
                f"**Recommendation:** Given the {trend_direction} trend and the forecast indicating {forecast_outlook}, I recommend a **{recommendation_action}** for {ticker}. "
                f"This position leverages the stock's {'growth potential' if recommendation_action == 'Buy' else 'stability while monitoring for better opportunities'}."
            )
        
        recommendations[ticker] = recommendation

    # Save recommendations to JSON
    with open(os.path.join(FORECASTS_DIR, "recommendations.json"), 'w') as f:
        pd.Series(recommendations).to_json(f)
    logging.info("Recommendations saved to recommendations.json.")
    return recommendations

# Function to Calculate MSE and RMSE
def evaluate_forecasts(tickers, forecast_start_date="2025-01-05", forecast_end_date="2025-01-31"):
    """
    Compare LSTM forecasts with actual data for January 2025 and calculate MSE and RMSE.
    """
    error_metrics = {}
    for ticker in tickers:
        # Load actual data (January 2025 data)
        actual_data = collect_stock_data([ticker], full_historical=False)
        if actual_data is None:
            logging.error(f"Actual data for {ticker} not found.")
            continue
        
        actual_df = actual_data
        actual_df['date'] = pd.to_datetime(actual_df['date'], utc=True, errors='coerce')
        if actual_df['date'].isnull().any():
            logging.error(f"Failed to convert some dates in actual data for {ticker}.")
            continue

        # Normalize actual_df dates to remove time component
        actual_df['date'] = actual_df['date'].dt.normalize()

        # Filter actual data for the forecast period
        start_date = pd.to_datetime(forecast_start_date, utc=True).normalize()
        end_date = pd.to_datetime(forecast_end_date, utc=True).normalize()
        actual_df = actual_df[
            (actual_df['date'] >= start_date) &
            (actual_df['date'] <= end_date)
        ]
        if actual_df.empty:
            logging.warning(f"No actual data found for {ticker} between {forecast_start_date} and {forecast_end_date}.")
            continue

        # Load forecast data
        forecast_file = os.path.join(FORECASTS_DIR, f"{ticker}_jan2025_forecast.csv")
        if not os.path.exists(forecast_file):
            logging.error(f"Forecast file for {ticker} not found at {forecast_file}.")
            continue

        forecast_df = pd.read_csv(forecast_file)
        forecast_df['date'] = pd.to_datetime(forecast_df['date'], utc=True, errors='coerce')
        if forecast_df['date'].isnull().any():
            logging.error(f"Failed to convert some dates in forecast data for {ticker}.")
            continue

        # Normalize forecast_df dates to remove time component
        forecast_df['date'] = forecast_df['date'].dt.normalize()

        # Filter forecast data for the same period
        forecast_df = forecast_df[
            (forecast_df['date'] >= start_date) &
            (forecast_df['date'] <= end_date)
        ]
        if forecast_df.empty:
            logging.warning(f"No forecast data found for {ticker} between {forecast_start_date} and {forecast_end_date}.")
            continue

        # Debug: Log sample dates from both dataframes
        logging.info(f"Sample actual dates for {ticker}: {actual_df['date'].head().tolist()}")
        logging.info(f"Sample forecast dates for {ticker}: {forecast_df['date'].head().tolist()}")

        # Merge actual and forecast data on date
        merged_df = pd.merge(
            actual_df[['date', 'close']],
            forecast_df[['date', 'forecasted_close']],
            on='date',
            how='inner'
        )
        if merged_df.empty:
            logging.warning(f"No overlapping dates found for {ticker} between actual and forecast data after normalization.")
            continue

        # Calculate MSE and RMSE
        mse = mean_squared_error(merged_df['close'], merged_df['forecasted_close'])
        rmse = np.sqrt(mse)
        error_metrics[ticker] = {"MSE": mse, "RMSE": rmse}
        logging.info(f"Error metrics for {ticker}: MSE = {mse:.2f}, RMSE = {rmse:.2f}")

    # Save error metrics to a file
    if error_metrics:
        error_metrics_df = pd.DataFrame.from_dict(error_metrics, orient='index')
        error_metrics_file = os.path.join(FORECASTS_DIR, "error_metrics.json")
        error_metrics_df.to_json(error_metrics_file)
        logging.info(f"Error metrics saved to {error_metrics_file}")
    else:
        logging.warning("No error metrics calculated for any ticker.")
    return error_metrics

# Run the Workflow
def run_stock_analysis_crew():
    # Step 1: Collect stock data (full historical data up to Dec 31, 2024)
    tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "JPM", "NKE"]
    combined_data = collect_stock_data(tickers, full_historical=True)
    if combined_data is None:
        print("Failed to collect stock data.")
        return None

    # Step 2: Process data and forecast using historical data
    analytics_and_forecasts = process_data_and_forecast(combined_data)
    if analytics_and_forecasts is None:
        print("Failed to process data and generate forecasts.")
        return None

    # Step 3: Convert DataFrames in forecasts to dictionaries for compatibility
    forecasts_dict = {}
    for ticker, forecast_df in analytics_and_forecasts['forecasts'].items():
        forecasts_dict[ticker] = forecast_df.to_dict('records')
    
    analytics_and_forecasts['forecasts'] = forecasts_dict

    # Step 4: Generate recommendations
    try:
        recommendations = generate_recommendations(analytics_and_forecasts)
    except Exception as e:
        print(f"Failed to generate recommendations: {e}")
        recommendations = None

    # Step 5: Evaluate forecasts (calculate MSE and RMSE using Jan 2025 data)
    try:
        error_metrics = evaluate_forecasts(tickers)
    except Exception as e:
        print(f"Failed to evaluate forecasts: {e}")
        error_metrics = None

    # Combine results
    final_result = {
        "analytics_and_forecasts": analytics_and_forecasts,
        "recommendations": recommendations,
        "error_metrics": error_metrics
    }
    
    # Summarize the result instead of printing the entire dictionary
    print("CrewAI Result Summary:")
    print(f"- Analytics generated for {len(final_result['analytics_and_forecasts']['analytics'])} tickers.")
    print(f"- Forecasts generated for {len(final_result['analytics_and_forecasts']['forecasts'])} tickers.")
    print(f"- Recommendations generated for {len(final_result['recommendations']) if final_result['recommendations'] else 0} tickers.")
    print(f"- Error metrics calculated for {len(final_result['error_metrics']) if final_result['error_metrics'] else 0} tickers.")
    return final_result

if __name__ == "__main__":
    result = run_stock_analysis_crew()