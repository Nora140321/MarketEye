from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import sqlite3
import pandas as pd
import os

# Define directories
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database", "stock_data.db"))
REPORTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports"))

# Initialize FastAPI app
app = FastAPI(
    title="Market Eye API",
    description="API for accessing stock data, analysis, and reports",
    version="1.0.0"
)

def fetch_stock_data(ticker, days=30):
    """
    Fetch the last N days of stock data for a given ticker from the database.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        query = """
            SELECT date, close, volume
            FROM stock_data
            WHERE ticker = ?
            ORDER BY date DESC
            LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=(ticker, days))
        conn.close()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")  # Sort ascending for response
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching data for {ticker}: {e}")

def read_analysis_summary(ticker):
    """
    Read the analysis summary from the report file.
    """
    summary_path = os.path.join(REPORTS_DIR, f"{ticker}_analysis.txt")
    try:
        with open(summary_path, "r") as f:
            return f.read()
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Analysis summary not found for {ticker}: {e}")

def get_report_path(ticker):
    """
    Get the path to the PDF report for a given ticker.
    """
    report_path = os.path.join(REPORTS_DIR, f"{ticker}_report.pdf")
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail=f"Report not found for {ticker}")
    return report_path

@app.get("/stocks/{ticker}/data")
async def get_stock_data(ticker: str):
    """
    Retrieve the last 30 days of stock data for a given ticker.
    """
    data = fetch_stock_data(ticker.upper())
    return {"ticker": ticker.upper(), "data": data}

@app.get("/stocks/{ticker}/analysis")
async def get_stock_analysis(ticker: str):
    """
    Retrieve the analysis summary for a given ticker.
    """
    summary = read_analysis_summary(ticker.upper())
    return {"ticker": ticker.upper(), "analysis": summary}

@app.get("/stocks/{ticker}/report")
async def get_stock_report(ticker: str):
    """
    Serve the PDF report for a given ticker.
    """
    report_path = get_report_path(ticker.upper())
    return FileResponse(
        report_path,
        media_type="application/pdf",
        filename=f"{ticker.upper()}_report.pdf"
    )