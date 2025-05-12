import sqlite3
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Set non-interactive backend
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import os
import time
import tempfile

# Define directories
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database", "stock_data.db"))
REPORTS_DIR = os.path.abspath(os.path.dirname(__file__))
TEMP_DIR = tempfile.gettempdir()  # Use system's temp directory to avoid OneDrive interference

def fetch_stock_data(ticker, days=30):
    """
    Fetch the last N days of stock data for a given ticker from the database.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        query = """
            SELECT date, close
            FROM stock_data
            WHERE ticker = ?
            ORDER BY date DESC
            LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=(ticker, days))
        conn.close()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")  # Sort ascending for plotting
        return df
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def calculate_moving_average(df, window=10):
    """
    Calculate the moving average for the closing price.
    """
    df["moving_average"] = df["close"].rolling(window=window).mean()
    return df

def plot_stock_data(ticker, df, output_path):
    """
    Plot closing prices and moving average, and save the plot as an image.
    """
    try:
        # Debug: Print DataFrame info
        print(f"DataFrame for {ticker}:")
        print(df.head())
        print(f"DataFrame dtypes:\n{df.dtypes}")
        
        # Ensure data is valid for plotting
        if df["close"].isnull().all() or df["date"].isnull().all():
            print(f"No valid data to plot for {ticker}")
            return False
        
        plt.figure(figsize=(8, 4))
        plt.plot(df["date"], df["close"], label="Closing Price", color="blue")
        plt.plot(df["date"], df["moving_average"], label="10-Day Moving Average", color="orange")
        plt.title(f"{ticker} Stock Price Trend")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.grid()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the plot with retry mechanism
        max_retries = 3
        for attempt in range(max_retries):
            try:
                plt.savefig(output_path)
                plt.close()
                # Wait longer to ensure the file is written
                time.sleep(1.0)
                if os.path.exists(output_path):
                    print(f"Chart saved for {ticker} at {output_path}")
                    return True
                else:
                    print(f"Chart file {output_path} not found after saving (attempt {attempt + 1}/{max_retries})")
            except Exception as e:
                print(f"Error saving chart for {ticker} (attempt {attempt + 1}/{max_retries}): {e}")
            time.sleep(0.5)  # Wait before retrying
        plt.close()
        print(f"Failed to save chart for {ticker} after {max_retries} attempts")
        return False
    except Exception as e:
        print(f"Error plotting data for {ticker}: {e}")
        plt.close()
        return False

def read_analysis_summary(ticker):
    """
    Read the analysis summary from the report file.
    """
    summary_path = os.path.join(REPORTS_DIR, f"{ticker}_analysis.txt")
    try:
        with open(summary_path, "r") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading summary for {ticker}: {e}")
        return "No analysis available."

def generate_pdf_report(ticker, df, summary, output_path):
    """
    Generate a PDF report with a table, chart (if available), and analysis summary.
    Returns the path to the temporary chart file (if created) for later cleanup.
    """
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    title = Paragraph(f"Stock Analysis Report: {ticker}", styles["Title"])
    elements.append(title)
    elements.append(Spacer(1, 12))

    # Summary
    elements.append(Paragraph("Analysis Summary", styles["Heading2"]))
    summary_style = ParagraphStyle(name="Summary", parent=styles["Normal"], fontSize=10, leading=12)
    summary_paragraph = Paragraph(summary.replace("\n", "<br/>"), summary_style)
    elements.append(summary_paragraph)
    elements.append(Spacer(1, 12))

    # Table of recent data
    elements.append(Paragraph("Recent Data (Last 30 Days)", styles["Heading2"]))
    data = [["Date", "Closing Price", "10-Day Moving Average"]]
    for _, row in df.iterrows():
        data.append([
            row["date"].strftime("%Y-%m-%d"),
            f"{row['close']:.2f}",
            f"{row['moving_average']:.2f}" if pd.notnull(row["moving_average"]) else "N/A"
        ])
    table = Table(data)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 12),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
        ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
        ("FONTSIZE", (0, 1), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 12))

    # Chart (if successfully generated)
    chart_path = os.path.join(TEMP_DIR, f"temp_{ticker}_chart.png")
    chart_generated = plot_stock_data(ticker, df, chart_path)
    if chart_generated and os.path.exists(chart_path):
        try:
            elements.append(Paragraph("Price Trend Chart", styles["Heading2"]))
            chart_image = Image(chart_path, width=400, height=200)
            elements.append(chart_image)
        except Exception as e:
            print(f"Error embedding chart for {ticker}: {e}")
            elements.append(Paragraph("Chart unavailable due to embedding error.", styles["Normal"]))
    else:
        elements.append(Paragraph("Price Trend Chart", styles["Heading2"]))
        elements.append(Paragraph("Chart unavailable due to generation error.", styles["Normal"]))

    # Build the PDF
    doc.build(elements)
    print(f"Generated PDF report for {ticker} at {output_path}")
    return chart_path if chart_generated and os.path.exists(chart_path) else None

def main():
    # List of tickers to report
    tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
    
    # Keep track of temporary chart files
    temp_files = []
    
    # Generate report for each ticker
    for ticker in tickers:
        print(f"Generating report for {ticker}...")
        
        # Fetch data from the database
        df = fetch_stock_data(ticker)
        if df is None or df.empty:
            print(f"No data available for {ticker}")
            continue
        
        # Calculate moving average
        df = calculate_moving_average(df, window=10)
        
        # Read analysis summary
        summary = read_analysis_summary(ticker)
        
        # Generate PDF report
        report_path = os.path.join(REPORTS_DIR, f"{ticker}_report.pdf")
        chart_path = generate_pdf_report(ticker, df, summary, report_path)
        if chart_path:
            temp_files.append(chart_path)

    # Clean up temporary chart files after all PDFs are generated
    for chart_path in temp_files:
        try:
            if os.path.exists(chart_path):
                os.remove(chart_path)
                print(f"Removed temporary chart file {chart_path}")
        except Exception as e:
            print(f"Error removing temporary chart file {chart_path}: {e}")

if __name__ == "__main__":
    main()