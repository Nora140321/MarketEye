import streamlit as st
import requests
import pandas as pd
from io import BytesIO

# Define API base URL
API_BASE_URL = "http://127.0.0.1:8000"

# List of tickers
TICKERS = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

# Streamlit app
st.title("Market Eye: Stock Analysis Dashboard")

# Ticker selection
ticker = st.selectbox("Select a stock ticker:", TICKERS)

# Fetch and display stock data
st.header(f"Stock Data for {ticker}")
try:
    response = requests.get(f"{API_BASE_URL}/stocks/{ticker}/data")
    response.raise_for_status()
    data = response.json()
    df = pd.DataFrame(data["data"])
    df["date"] = pd.to_datetime(df["date"])
    
    # Display the stock data table
    display_df = df[["date", "close", "volume"]].copy()
    display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
    st.dataframe(display_df, use_container_width=True)
    
    # Plot the closing price over time
    st.header(f"Closing Price Trend for {ticker}")
    chart_df = df.set_index("date")[["close"]]
    st.line_chart(chart_df)
except Exception as e:
    st.error(f"Error fetching stock data: {e}")

# Fetch and display analysis summary as a table
st.header(f"Analysis Summary for {ticker}")
try:
    response = requests.get(f"{API_BASE_URL}/stocks/{ticker}/analysis")
    response.raise_for_status()
    analysis = response.json()["analysis"]
    
    # Split the analysis text into lines
    lines = analysis.split("\n")
    
    # Extract the summary text (first line) and table data
    summary_text = lines[0]  # e.g., "Mock analysis for AAPL: The stock shows a stable trend based on recent data:"
    table_lines = [line for line in lines[1:] if line.strip()]  # Skip empty lines
    
    # Display the summary text
    st.text(summary_text)
    
    # Parse the table data
    if len(table_lines) > 1:  # Ensure there's at least a header and one data row
        # Extract headers (first non-empty line after the summary)
        headers = table_lines[0].split()
        headers = [header.strip() for header in headers]
        
        # Extract data rows
        data_rows = [line.split() for line in table_lines[1:] if line.strip()]
        # Combine date and time into a single column
        table_data = []
        for row in data_rows:
            if len(row) >= 4:  # Ensure the row has enough columns
                date_time = f"{row[0]} {row[1]}"  # Combine date and time
                close = float(row[2])
                moving_average = float(row[3])
                table_data.append([date_time, close, moving_average])
        
        # Create a DataFrame for the table
        table_df = pd.DataFrame(table_data, columns=["Date", "Close", "Moving Average"])
        
        # Display the table
        st.dataframe(table_df, use_container_width=True)
    else:
        st.text("No table data available in the analysis summary.")
except Exception as e:
    st.error(f"Error fetching analysis summary: {e}")

# Download PDF report
st.header(f"Download Report for {ticker}")
if st.button("Download PDF Report"):
    try:
        response = requests.get(f"{API_BASE_URL}/stocks/{ticker}/report")
        response.raise_for_status()
        pdf_data = BytesIO(response.content)
        st.download_button(
            label="Click to download",
            data=pdf_data,
            file_name=f"{ticker}_report.pdf",
            mime="application/pdf"
        )
    except Exception as e:
        st.error(f"Error downloading report: {e}")