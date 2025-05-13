import yfinance as yf
import pandas as pd
import os
import time

KAGGLE_DATA_DIR = r"C:\Users\algha\OneDrive\Documents\Market_Eye\kaggle_data"
TICKERS = ["AAPL"]

for ticker in TICKERS:
    stock = yf.Ticker(ticker)
    df = stock.history(start="2000-01-01", end="2025-05-12")
    df.reset_index(inplace=True)
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S%z')
    df['Brand_Name'] = ticker.lower()
    df['Ticker'] = ticker
    df['Industry_Tag'] = 'technology'
    df['Country'] = 'usa'
    df['Dividends'] = 0.0
    df['Stock Splits'] = 0.0
    df['Capital Gains'] = ''
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Brand_Name', 'Ticker', 'Industry_Tag', 'Country', 'Dividends', 'Stock Splits', 'Capital Gains']]
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Brand_Name', 'Ticker', 'Industry_Tag', 'Country', 'Dividends', 'Stock Splits', 'Capital Gains']
    output_path = os.path.join(KAGGLE_DATA_DIR, f"{ticker}.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved data for {ticker} to {output_path}")
    time.sleep(5)  # Delay to avoid rate limiting