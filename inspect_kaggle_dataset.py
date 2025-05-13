import pandas as pd
import os

# Path to the downloaded dataset
dataset_path = r"C:\Users\algha\.cache\kagglehub\datasets\nelgiriyewithana\world-stock-prices-daily-updating\versions\345\World-Stock-Prices-Dataset.csv"

# Load the dataset
df = pd.read_csv(dataset_path)

# Print the first few rows to inspect the structure
print("First few rows of the dataset:")
print(df.head())

# Print the column names
print("\nColumns in the dataset:")
print(df.columns.tolist())

# Print unique tickers to confirm our tickers are present
print("\nUnique tickers in the dataset:")
print(df['Ticker'].unique())

# Filter data for our tickers (AAPL, GOOGL, MSFT, AMZN, TSLA)
tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
filtered_df = df[df['Ticker'].isin(tickers)]

# Create the kaggle_data directory if it doesn't exist
kaggle_data_dir = r"C:\Users\algha\OneDrive\Documents\Market_Eye\kaggle_data"
if not os.path.exists(kaggle_data_dir):
    os.makedirs(kaggle_data_dir)

# Save each ticker's data to a separate CSV file in kaggle_data/
for ticker in tickers:
    ticker_df = filtered_df[filtered_df['Ticker'] == ticker]
    if not ticker_df.empty:
        ticker_path = os.path.join(kaggle_data_dir, f"{ticker}.csv")
        ticker_df.to_csv(ticker_path, index=False)
        print(f"Saved data for {ticker} to {ticker_path}")
    else:
        print(f"No data found for {ticker}")