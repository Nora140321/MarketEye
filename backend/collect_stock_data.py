import pandas as pd
import os

KAGGLE_DATA_DIR = r"C:\Users\algha\OneDrive\Documents\Market_Eye\kaggle_data"
DATA_DIR = r"C:\Users\algha\OneDrive\Documents\Market_Eye\data"
TICKERS = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

def process_stock_data(ticker):
    kaggle_file_path = os.path.join(KAGGLE_DATA_DIR, f"{ticker}.csv")
    output_file_path = os.path.join(DATA_DIR, f"{ticker}_stock_data.csv")
    if os.path.exists(kaggle_file_path):
        df = pd.read_csv(kaggle_file_path)
        df['date'] = pd.to_datetime(df['Date'], utc=True)  # Add utc=True to handle mixed time zones
        df = df[['date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        df.to_csv(output_file_path, index=False)
        print(f"Saved stock data for {ticker} to {output_file_path}")
    else:
        print(f"No data file found for {ticker} at {kaggle_file_path}")

def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    for ticker in TICKERS:
        print(f"Processing data for {ticker}...")
        process_stock_data(ticker)

if __name__ == "__main__":
    main()