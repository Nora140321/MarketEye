import pandas as pd
import os
from datetime import datetime, timedelta
import schedule
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'collect_stock_data.log')),
        logging.StreamHandler()
    ]
)

# Define directories
DATA_DIR = r"C:\Users\algha\OneDrive\Documents\Market_Eye\data"
KAGGLE_DATA_PATH = r"C:\Users\algha\OneDrive\Documents\Market_Eye\World-Stock-Prices-Dataset.csv"

# Ensure data directory exists
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# List of tickers
TICKERS = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "JPM", "NKE"]

def update_stock_data(ticker, current_date):
    """
    Update stock data for a given ticker by filtering the Kaggle dataset up to the current date.
    """
    file_path = os.path.join(DATA_DIR, f"{ticker}_stock_data.csv")
    
    # Load the Kaggle dataset
    logging.info(f"Loading Kaggle dataset for {ticker}...")
    try:
        df = pd.read_csv(KAGGLE_DATA_PATH)
    except Exception as e:
        logging.error(f"Error loading Kaggle dataset: {e}")
        return

    # Standardize column names
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
        return

    # Convert date column to datetime with UTC
    df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce')
    if df['date'].isnull().any():
        logging.error("Some dates could not be converted to datetime format.")
        return

    # Convert current_date to UTC datetime for comparison
    current_date = pd.to_datetime(current_date, utc=True)

    # Filter for the specified ticker and date range (up to current_date or December 31, 2024)
    ticker_df = df[df['ticker'] == ticker]
    cutoff_date = min(pd.to_datetime("2024-12-31", utc=True), current_date)
    ticker_df = ticker_df[ticker_df['date'] <= cutoff_date]

    if ticker_df.empty:
        logging.warning(f"No data found for {ticker} up to {cutoff_date}.")
        return

    # Sort by date
    ticker_df = ticker_df.sort_values('date')
    # Save to CSV
    ticker_df.to_csv(file_path, index=False)
    logging.info(f"Updated data for {ticker} saved to {file_path} ({len(ticker_df)} rows).")

def update_all_tickers():
    """Update data for all tickers."""
    current_date = datetime.now().date()
    if current_date > datetime.strptime("2024-12-31", "%Y-%m-%d").date():
        logging.info("Data updates stopped as of December 31, 2024.")
        return

    for ticker in TICKERS:
        update_stock_data(ticker, current_date)
        time.sleep(5)  # Small delay to avoid overloading

def main():
    """
    Main function to schedule daily updates of stock data up to December 31, 2024.
    """
    # Run immediately on startup
    update_all_tickers()

    # Schedule daily updates at 8:00 AM
    schedule.every().day.at("08:00").do(update_all_tickers)

    # Keep the script running to execute scheduled tasks
    while True:
        current_date = datetime.now().date()
        if current_date > datetime.strptime("2024-12-31", "%Y-%m-%d").date():
            logging.info("Scheduling stopped as of December 31, 2024.")
            break
        schedule.run_pending()
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    main()