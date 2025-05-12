import pandas as pd
import os

# Define the list of stock tickers to process
STOCK_TICKERS = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

# Define the data directory
DATA_DIR = "../data/"
os.makedirs(DATA_DIR, exist_ok=True)

# Path to the World-Stock-Prices-Dataset.csv file
DATASET_PATH = "../World-Stock-Prices-Dataset.csv"

def process_stock_data():
    """
    Process World-Stock-Prices-Dataset.csv, extract data for specified tickers,
    and save each ticker's data as a CSV file in the data/ directory.
    """
    try:
        # Read the dataset
        df = pd.read_csv(DATASET_PATH)
        
        # Ensure the dataset has the expected columns
        expected_columns = ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]
        if not all(col in df.columns for col in expected_columns):
            print(f"Error: Dataset does not contain all expected columns: {expected_columns}")
            return
        
        # Filter and save data for each ticker
        for ticker in STOCK_TICKERS:
            # Extract data for the current ticker
            ticker_data = df[df["Ticker"] == ticker][expected_columns]
            
            # Check if data exists for the ticker
            if ticker_data.empty:
                print(f"No data found for {ticker} in the dataset")
                continue
            
            # Save to CSV
            csv_path = os.path.join(DATA_DIR, f"{ticker}_historical_data.csv")
            ticker_data.to_csv(csv_path, index=False)
            print(f"Saved data for {ticker} to {csv_path}")
    except Exception as e:
        print(f"Error processing dataset: {e}")

if __name__ == "__main__":
    process_stock_data()