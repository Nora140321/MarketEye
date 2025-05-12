import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
import time  # Add this import for delays

# Define the list of stock tickers to collect data for
STOCK_TICKERS = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

# Define the data directory
DATA_DIR = "../data/"
os.makedirs(DATA_DIR, exist_ok=True)

def collect_stock_data(ticker, start_date, end_date):
    """
    Collect historical stock data for a given ticker between start_date and end_date.
    Save the data as a CSV file in the data/ directory.
    """
    try:
        # Fetch data using yfinance
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        
        # Check if data is empty
        if data.empty:
            print(f"No data found for {ticker}")
            return
        
        # Reset index to make 'Date' a column
        data.reset_index(inplace=True)
        
        # Save to CSV
        csv_path = os.path.join(DATA_DIR, f"{ticker}_historical_data.csv")
        data.to_csv(csv_path, index=False)
        print(f"Saved data for {ticker} to {csv_path}")
    except Exception as e:
        print(f"Error collecting data for {ticker}: {e}")

def main():
    # Define the time range for data collection (from 2023-12-31 to 2024-12-31)
    end_date = "2024-12-31"
    start_date = "2023-12-31"
    
    print(f"Collecting data from {start_date} to {end_date}")
    
    # Collect data for each ticker with a delay to avoid rate limiting
    for i, ticker in enumerate(STOCK_TICKERS):
        collect_stock_data(ticker, start_date, end_date)
        if i < len(STOCK_TICKERS) - 1:  # Don't sleep after the last ticker
            print(f"Waiting 5 seconds to avoid rate limiting...")
            time.sleep(5)  # Add a 5-second delay between requests

if __name__ == "__main__":
    main()