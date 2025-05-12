import pandas as pd
import sqlite3
import os

# Define the list of stock tickers
STOCK_TICKERS = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

# Define directories
DATA_DIR = "../data/"
DB_DIR = "../database/"
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "stock_data.db")

def clean_data(df):
    """
    Clean the stock data by handling missing values and ensuring correct data types.
    """
    # Drop rows with missing values in critical columns
    df = df.dropna(subset=["Date", "Open", "High", "Low", "Close", "Volume"])
    
    # Ensure correct data types
    df["Date"] = pd.to_datetime(df["Date"], utc=True)  # Handle mixed time zones
    df["Open"] = df["Open"].astype(float)
    df["High"] = df["High"].astype(float)
    df["Low"] = df["Low"].astype(float)
    df["Close"] = df["Close"].astype(float)
    df["Volume"] = df["Volume"].astype(int)
    
    # Convert Timestamp to string for SQLite compatibility
    df["Date"] = df["Date"].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    return df

def process_and_store_data():
    """
    Read CSV files, clean the data, and store it in a SQLite database.
    """
    try:
        # Connect to SQLite database (creates the file if it doesn't exist)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Drop the table if it exists to start fresh
        cursor.execute("DROP TABLE IF EXISTS stock_data")
        
        # Create a table for stock data
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stock_data (
                ticker TEXT,
                date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                PRIMARY KEY (ticker, date)
            )
        """)
        
        # Process each ticker's CSV file
        for ticker in STOCK_TICKERS:
            csv_path = os.path.join(DATA_DIR, f"{ticker}_historical_data.csv")
            if not os.path.exists(csv_path):
                print(f"CSV file not found for {ticker}: {csv_path}")
                continue
            
            # Read the CSV file
            df = pd.read_csv(csv_path)
            
            # Clean the data
            df = clean_data(df)
            
            # Add ticker column
            df["Ticker"] = ticker
            
            # Remove duplicates based on ticker and date
            df = df.drop_duplicates(subset=["Ticker", "Date"], keep="last")
            
            # Rename columns to match the database schema
            df = df.rename(columns={
                "Date": "date",
                "Ticker": "ticker",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume"
            })
            
            # Store in the database
            df.to_sql("stock_data", conn, if_exists="append", index=False)
            print(f"Stored data for {ticker} in the database")
        
        # Commit and close the connection
        conn.commit()
        conn.close()
        print(f"Database created/updated at {DB_PATH}")
    
    except Exception as e:
        print(f"Error processing and storing data: {e}")

if __name__ == "__main__":
    process_and_store_data()