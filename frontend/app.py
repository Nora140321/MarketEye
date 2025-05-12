import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import requests
import pandas as pd
from io import BytesIO
from backend.auth import register_user, login_user, log_action

# Define API base URL
API_BASE_URL = "http://127.0.0.1:8000"

# List of tickers
TICKERS = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

# Initialize session state for authentication
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "username" not in st.session_state:
    st.session_state.username = None
if "page" not in st.session_state:
    st.session_state.page = "login"

# Apply custom CSS for styling
st.markdown(
    """
    <style>
    /* Main background and text color */
    .stApp {
        background-color: #1E1E1E;
        color: #E0E0E0;
    }
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #2D2D2D;
    }
    /* Table header styling */
    .stDataFrame thead th {
        background-color: #424242;
        color: #FFFFFF;
    }
    /* Table cell styling */
    .stDataFrame tbody td {
        color: #E0E0E0;
    }
    /* Button styling */
    .stButton>button {
        background-color: #26A69A;
        color: #FFFFFF;
        border-radius: 5px;
    }
    /* Expander header styling */
    .stExpander summary {
        background-color: #2D2D2D;
        color: #E0E0E0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to display the dashboard
def show_dashboard():
    st.title("Market Eye: Stock Analysis Dashboard")
    st.write(f"Welcome, {st.session_state.username}!")

    # Sidebar for ticker selection and logout
    with st.sidebar:
        st.header("Stock Selection")
        ticker = st.selectbox("Select a stock ticker:", TICKERS)
        if st.button("Logout"):
            log_action(st.session_state.user_id, "User logged out")
            st.session_state.user_id = None
            st.session_state.username = None
            st.session_state.page = "login"
            st.rerun()  # Updated from st.experimental_rerun()

    # Fetch stock data
    try:
        response = requests.get(f"{API_BASE_URL}/stocks/{ticker}/data")
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data["data"])
        df["date"] = pd.to_datetime(df["date"])
        
        # Log app usage
        log_action(st.session_state.user_id, f"Viewed stock data for {ticker}")
        
        # Prepare data for display and chart
        display_df = df[["date", "close", "volume"]].copy()
        display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
        chart_df = df.set_index("date")[["close"]]
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        display_df = pd.DataFrame()
        chart_df = pd.DataFrame()

    # Create two columns for stock data table and chart
    col1, col2 = st.columns(2)

    # Stock data table in the first column
    with col1:
        st.header(f"Stock Data for {ticker}")
        if not display_df.empty:
            st.dataframe(display_df, use_container_width=True)
        else:
            st.write("No data available to display.")

    # Closing price trend chart in the second column
    with col2:
        st.header(f"Closing Price Trend for {ticker}")
        if not chart_df.empty:
            st.line_chart(chart_df)
        else:
            st.write("No data available to plot.")

    # Analysis summary in an expander
    with st.expander("Analysis Summary"):
        st.header(f"Analysis Summary for {ticker}")
        try:
            response = requests.get(f"{API_BASE_URL}/stocks/{ticker}/analysis")
            response.raise_for_status()
            analysis = response.json()["analysis"]
            
            # Log app usage
            log_action(st.session_state.user_id, f"Viewed analysis summary for {ticker}")
            
            # Split the analysis text into lines
            lines = analysis.split("\n")
            
            # Extract the summary text (first line) and table data
            summary_text = lines[0]
            table_lines = [line for line in lines[1:] if line.strip()]
            
            # Display the summary text
            st.text(summary_text)
            
            # Parse the table data
            if len(table_lines) > 1:
                headers = table_lines[0].split()
                headers = [header.strip() for header in headers]
                
                data_rows = [line.split() for line in table_lines[1:] if line.strip()]
                table_data = []
                for row in data_rows:
                    if len(row) >= 4:
                        date_time = f"{row[0]} {row[1]}"
                        close = float(row[2])
                        moving_average = float(row[3])
                        table_data.append([date_time, close, moving_average])
                
                table_df = pd.DataFrame(table_data, columns=["Date", "Close", "Moving Average"])
                st.dataframe(table_df, use_container_width=True)
            else:
                st.text("No table data available in the analysis summary.")
        except Exception as e:
            st.error(f"Error fetching analysis summary: {e}")

    # Download PDF report section
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
            # Log app usage
            log_action(st.session_state.user_id, f"Downloaded report for {ticker}")
        except Exception as e:
            st.error(f"Error downloading report: {e}")

# Login page
def login_page():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        success, user_id, message = login_user(username, password)
        if success:
            st.session_state.user_id = user_id
            st.session_state.username = username
            st.session_state.page = "dashboard"
            st.success(message)
            st.rerun()  # Updated from st.experimental_rerun()
        else:
            st.error(message)
    if st.button("Go to Sign Up"):
        st.session_state.page = "signup"
        st.rerun()  # Updated from st.experimental_rerun()

# Sign-up page
def signup_page():
    st.title("Sign Up")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Sign Up"):
        success, message = register_user(username, password)
        if success:
            st.success(message)
            st.session_state.page = "login"
            st.rerun()  # Updated from st.experimental_rerun()
        else:
            st.error(message)
    if st.button("Go to Login"):
        st.session_state.page = "login"
        st.rerun()  # Updated from st.experimental_rerun()

# Page routing
if st.session_state.user_id is None:
    if st.session_state.page == "login":
        login_page()
    elif st.session_state.page == "signup":
        signup_page()
else:
    show_dashboard()