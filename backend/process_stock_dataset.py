import pandas as pd
import os

STOCK_TICKERS = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
DATA_DIR = "../data/"
os.makedirs(DATA_DIR, exist_ok=True)
DATASET_PATH = "../World-Stock-Prices-Dataset.csv"

def process_stock_data():
    try:
        df = pd.read_csv(DATASET_PATH)
        expected_columns = ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]
        if not all(col in df.columns for col in expected_columns):
            print(f"Error: Dataset does not contain all expected columns: {expected_columns}")
            return
        for ticker in STOCK_TICKERS:
            ticker_data = df[df["Ticker"] == ticker][expected_columns]
            if ticker_data.empty:
                print(f"No data found for {ticker} in the dataset")
                continue
            csv_path = os.path.join(DATA_DIR, f"{ticker}_historical_data.csv")
            ticker_data.to_csv(csv_path, index=False)
            print(f"Saved data for {ticker} to {csv_path}")
    except Exception as e:
        print(f"Error processing dataset: {e}")

if __name__ == "__main__":
    process_stock_data()