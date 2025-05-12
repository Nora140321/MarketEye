import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
import time

STOCK_TICKERS = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
DATA_DIR = "../data/"
os.makedirs(DATA_DIR, exist_ok=True)

def collect_stock_data(ticker, start_date, end_date):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        if data.empty:
            print(f"No data found for {ticker}")
            return
        data.reset_index(inplace=True)
        csv_path = os.path.join(DATA_DIR, f"{ticker}_historical_data.csv")
        data.to_csv(csv_path, index=False)
        print(f"Saved data for {ticker} to {csv_path}")
    except Exception as e:
        print(f"Error collecting data for {ticker}: {e}")

def main():
    end_date = "2024-12-31"
    start_date = "2023-12-31"
    print(f"Collecting data from {start_date} to {end_date}")
    for i, ticker in enumerate(STOCK_TICKERS):
        collect_stock_data(ticker, start_date, end_date)
        if i < len(STOCK_TICKERS) - 1:
            print(f"Waiting 5 seconds to avoid rate limiting...")
            time.sleep(5)

if __name__ == "__main__":
    main()