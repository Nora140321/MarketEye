import pandas as pd

# Path to the Kaggle dataset
KAGGLE_DATA_PATH = r"C:\Users\algha\OneDrive\Documents\Market_Eye\World-Stock-Prices-Dataset.csv"

# List of tickers
TICKERS = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "JPM", "NKE"]

def inspect_kaggle_data():
    """Inspect the Kaggle World Stock Prices dataset."""
    print(f"Loading dataset from {KAGGLE_DATA_PATH}...")
    try:
        df = pd.read_csv(KAGGLE_DATA_PATH)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Display basic information
    print("Dataset Info:")
    print(df.info())
    print("\nDataset Columns:", df.columns.tolist())
    print("\nSample Data (first 5 rows):")
    print(df.head())

    # Check date range
    df['Date'] = pd.to_datetime(df['Date'], utc=True, errors='coerce')
    if df['Date'].isnull().any():
        print("Warning: Some dates could not be converted to datetime format.")
    else:
        min_date = df['Date'].min()
        max_date = df['Date'].max()
        print(f"\nDate Range: {min_date} to {max_date}")

    # Check for specified tickers
    available_tickers = df['Ticker'].unique()
    print("\nAvailable Tickers:", available_tickers)
    missing_tickers = [ticker for ticker in TICKERS if ticker not in available_tickers]
    if missing_tickers:
        print(f"Missing Tickers: {missing_tickers}")
    else:
        print("All required tickers are present in the dataset.")

if __name__ == "__main__":
    inspect_kaggle_data()