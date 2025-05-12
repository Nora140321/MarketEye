import sqlite3
import pandas as pd
import os
from google.generativeai import GenerativeModel
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Define directories
DB_PATH = "../database/stock_data.db"

def fetch_stock_data(ticker):
    """
    Fetch stock data for a given ticker from the database.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        query = "SELECT date, close FROM stock_data WHERE ticker = ? ORDER BY date"
        df = pd.read_sql_query(query, conn, params=(ticker,))
        conn.close()
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def calculate_moving_average(df, window=50):
    """
    Calculate the moving average for the closing price.
    """
    df["moving_average"] = df["close"].rolling(window=window).mean()
    return df

def generate_analysis_summary(ticker, df):
    """
    Mock function to simulate Gemini API response.
    """
    try:
        recent_data = df.tail(5)
        return f"Mock analysis for {ticker}: The stock shows a stable trend based on recent data:\n{recent_data[['date', 'close', 'moving_average']].to_string(index=False)}"
    except Exception as e:
        print(f"Error generating summary for {ticker}: {e}")
        return "Unable to generate summary."

def main():
    # List of tickers to analyze
    tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
    
    # Create a reports directory
    reports_dir = "../reports/"
    os.makedirs(reports_dir, exist_ok=True)
    
    # Analyze each ticker
    for ticker in tickers:
        print(f"Analyzing {ticker}...")
        
        # Fetch data from the database
        df = fetch_stock_data(ticker)
        if df is None or df.empty:
            print(f"No data available for {ticker}")
            continue
        
        # Calculate moving average
        df = calculate_moving_average(df)
        
        # Generate analysis summary using mock function
        summary = generate_analysis_summary(ticker, df)
        print(f"Analysis for {ticker}:\n{summary}\n")
        
        # Save the summary to a file
        report_path = os.path.join(reports_dir, f"{ticker}_analysis.txt")
        with open(report_path, "w") as f:
            f.write(summary)
        print(f"Saved analysis for {ticker} to {report_path}")

if __name__ == "__main__":
    main()