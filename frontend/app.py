import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)  # Suppress deprecation warnings

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../agents')))

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from backend.auth import register_user, login_user, log_action
from agents.stock_analysis_agents import run_stock_analysis_crew  # Import the CrewAI workflow
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# Set page layout to wide for better visibility
st.set_page_config(layout="wide")

# List of tickers (expanded to include Finance and Sportswear)
TICKERS = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "JPM", "NKE"]

# Define directories
FORECASTS_DIR = r"C:\Users\algha\OneDrive\Documents\Market_Eye\forecasts"
DATA_DIR = r"C:\Users\algha\OneDrive\Documents\Market_Eye\data"
REPORTS_DIR = r"C:\Users\algha\OneDrive\Documents\Market_Eye\reports"
REPORT_TEMPLATES_DIR = r"C:\Users\algha\OneDrive\Documents\Market_Eye\report_templates"

# Create directories if they don't exist
if not os.path.exists(REPORTS_DIR):
    os.makedirs(REPORTS_DIR)
if not os.path.exists(REPORT_TEMPLATES_DIR):
    os.makedirs(REPORT_TEMPLATES_DIR)

# Initialize session state for authentication
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "username" not in st.session_state:
    st.session_state.username = None
if "page" not in st.session_state:
    st.session_state.page = "login"
if "training_done" not in st.session_state:
    st.session_state.training_done = False  # Track if training has been done

# Apply custom CSS for styling
st.markdown(
    """
    <style>
    /* Main background and text color */
    .stApp {
        background-color: #1E1E1E;
        color: #E0E0E0;
        font-family: 'Absara', sans-serif; /* Set font to Absara for all text */
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
    /* Custom styling for highlighted numbers and currency */
    .highlight-number {
        font-size: 1.2em; /* Reduced size for normal font */
        color: #FF4B4B; /* Red color for numbers */
    }
    /* Welcome message styling */
    .welcome-container {
        position: absolute;
        top: 10px;
        right: 10px;
        display: flex;
        align-items: center;
    }
    .welcome-text {
        font-size: 2em; /* Increased font size */
        color: #E0E0E0;
        margin-right: 10px;
    }
    .user-photo {
        width: 40px;
        height: 40px;
        border-radius: 50%;
    }
    /* Space between welcome message and main heading */
    .main-heading {
        margin-top: 60px; /* Add space below the welcome message */
    }
    /* Table styling for historical summary */
    .historical-summary-table td {
        padding: 5px;
        color: #E0E0E0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to calculate historical volatility
def calculate_volatility(df, window=60):
    returns = df['close'].pct_change().fillna(0)  # Fill NaN with 0 to avoid issues
    volatility = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized volatility
    volatility = volatility.fillna(0)  # Fill any remaining NaN with 0
    return volatility

# Function to calculate 14-day RSI
def calculate_rsi(df, periods=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(0)  # Fill NaN with 0 to avoid issues
    return rsi

# Function to generate a PDF report (Updated: Removed Error Metrics Section)
def generate_pdf_report(ticker, historical_df, summary_metrics, recommendation, forecast_df):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []

    # Styles for the report
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    heading_style = styles['Heading2']
    normal_style = ParagraphStyle(
        name='Normal',
        fontName='Helvetica',
        fontSize=12,
        leading=14,
        spaceAfter=12
    )

    # Title
    elements.append(Paragraph(f"Market Eye Report for {ticker}", title_style))
    elements.append(Spacer(1, 12))

    # Historical Stock Data
    elements.append(Paragraph("Historical Stock Data (Last 60 Days)", heading_style))
    required_columns = ['date', 'close', 'volume']
    if not all(col in historical_df.columns for col in required_columns):
        elements.append(Paragraph(f"Error: Historical data is missing required columns. Available columns: {historical_df.columns.tolist()}", normal_style))
    else:
        historical_data = historical_df.tail(60)[required_columns].copy()
        historical_data['date'] = pd.to_datetime(historical_data['date'])
        historical_data['date'] = historical_data['date'].apply(lambda x: x.strftime("%d-%m-%Y %I %p").replace("AM", "am").replace("PM", "pm"))
        historical_data = historical_data.round({'close': 3, 'volume': 3})
        data = [['Date', 'Close', 'Volume']] + historical_data.values.tolist()
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(table)
    elements.append(Spacer(1, 12))

    # Historical Summary
    elements.append(Paragraph("Historical Summary", heading_style))
    if not summary_metrics:
        elements.append(Paragraph("Error: Summary metrics are missing.", normal_style))
    else:
        summary_text = (
            f"- Average Closing Price (Last 60 Days): ${summary_metrics['avg_close']:.2f}\n"
            f"- Recent Volatility (Last 60 Days, Annualized): {summary_metrics['recent_volatility']:.2f}\n"
            f"- Current RSI (14-Day): {summary_metrics['current_rsi']:.2f}\n"
            f"- Price Trend (Last 30 Days): {summary_metrics['price_trend']:.2f}%\n"
            f"- 2020 Growth Percentage: {summary_metrics['growth_2020']:.2f}%\n"
            f"- Support Level (Last 60 Days): ${summary_metrics['support_level']:.2f}\n"
            f"- Resistance Level (Last 60 Days): ${summary_metrics['resistance_level']:.2f}\n"
        )
        annual_growth = summary_metrics.get('annual_growth', {})
        if annual_growth:
            summary_text += "- Annual Growth (Year-over-Year):\n"
            for year, growth in annual_growth.items():
                summary_text += f"  - {year}: {growth:.2f}%\n"
        elements.append(Paragraph(summary_text, normal_style))
    elements.append(Spacer(1, 12))

    # Recommendation
    elements.append(Paragraph("Recommendation", heading_style))
    elements.append(Paragraph(recommendation, normal_style))
    elements.append(Spacer(1, 12))

    # Forecast Data
    elements.append(Paragraph("January 2025 Forecast", heading_style))
    if forecast_df.empty:
        elements.append(Paragraph("Error: Forecast data is missing.", normal_style))
    else:
        forecast_required_columns = ['date', 'forecasted_close']
        if not all(col in forecast_df.columns for col in forecast_required_columns):
            elements.append(Paragraph("Error: Forecast data is missing required columns.", normal_style))
        else:
            forecast_data = forecast_df[forecast_required_columns].copy()
            forecast_data['date'] = pd.to_datetime(forecast_data['date'])
            forecast_data['date'] = forecast_data['date'].apply(lambda x: x.strftime("%d-%m-%Y %I %p").replace("AM", "am").replace("PM", "pm"))
            forecast_data = forecast_data.round({'forecasted_close': 3})
            data = [['Date', 'Forecasted Close']] + forecast_data.values.tolist()
            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(table)
    elements.append(Spacer(1, 12))

    # Forecast vs. Real Data Comparison
    elements.append(Paragraph("Forecast vs. Real Data Comparison (January 5-31, 2025)", heading_style))
    if not all(col in historical_df.columns for col in ['date', 'close']):
        elements.append(Paragraph(f"Error: Historical data is missing required columns for comparison. Available columns: {historical_df.columns.tolist()}", normal_style))
    elif forecast_df.empty or not all(col in forecast_df.columns for col in ['date', 'forecasted_close']):
        elements.append(Paragraph("Error: Forecast data is missing or lacks required columns.", normal_style))
    else:
        # Ensure both date columns are datetime and timezone-aware
        historical_df = historical_df.copy()
        forecast_df = forecast_df.copy()
        historical_df['date'] = pd.to_datetime(historical_df['date'], utc=True)
        forecast_df['date'] = pd.to_datetime(forecast_df['date'], utc=True)
        comparison_df = pd.merge(
            historical_df[['date', 'close']],
            forecast_df[['date', 'forecasted_close']],
            on='date',
            how='inner'
        )
        if comparison_df.empty:
            elements.append(Paragraph("Error: No overlapping dates for forecast vs. real data comparison.", normal_style))
        else:
            comparison_df['date'] = comparison_df['date'].apply(lambda x: x.strftime("%d-%m-%Y %I %p").replace("AM", "am").replace("PM", "pm"))
            comparison_df = comparison_df.round({'close': 3, 'forecasted_close': 3})
            data = [['Date', 'Actual Close', 'Forecasted Close']] + comparison_df.values.tolist()
            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(table)
    elements.append(Spacer(1, 12))

    # Build the PDF
    doc.build(elements)
    buffer.seek(0)

    # Save the PDF to the reports directory with a timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_filename = f"{ticker}_report_{timestamp}.pdf"
    report_path = os.path.join(REPORTS_DIR, report_filename)
    with open(report_path, "wb") as f:
        f.write(buffer.getvalue())

    return buffer

# Function to display the dashboard
def show_dashboard():
    # Welcome message with user photo at the top right
    st.markdown(
        f"""
        <div class="welcome-container">
            <span class="welcome-text">Welcome, {st.session_state.username}!</span>
            <img src="https://via.placeholder.com/40" class="user-photo" alt="User Photo">
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="main-heading">', unsafe_allow_html=True)
    st.title("Market Eye: Stock Analysis Dashboard")
    st.markdown('</div>', unsafe_allow_html=True)

    # Stock Selection at the top of the dashboard
    st.header("Stock Selection")
    ticker = st.selectbox("Select a stock ticker:", TICKERS)

    # Collect Stock Data
    st.header("Collect Stock Data")
    if st.button("Collect Stock Data"):
        with st.spinner("Collecting stock data using CrewAI..."):
            try:
                # Run the CrewAI workflow to collect and process data
                result = run_stock_analysis_crew()
                if result is None:
                    st.error("CrewAI failed to collect or process data.")
                    return

                log_action(st.session_state.user_id, "Collected stock data using CrewAI")
                st.success("Stock data collected successfully!")

                # Load and display the collected data for all tickers
                st.subheader("Collected Stock Data")
                combined_data = pd.DataFrame()
                for t in TICKERS:
                    file_path = os.path.join(DATA_DIR, f"{t}_stock_data.csv")
                    if os.path.exists(file_path):
                        df = pd.read_csv(file_path)
                        combined_data = pd.concat([combined_data, df], ignore_index=True)
                
                if not combined_data.empty:
                    # Format the date column to show only date and time (e.g., 17-1-2022 05 am/pm)
                    combined_data['date'] = pd.to_datetime(combined_data['date'])
                    combined_data['date'] = combined_data['date'].apply(lambda x: x.strftime("%d-%m-%Y %I %p").replace("AM", "am").replace("PM", "pm"))

                    # Format numerical columns to 3 decimal places
                    combined_data = combined_data.round({'open': 3, 'high': 3, 'low': 3, 'close': 3, 'volume': 3})

                    # Display only the required columns (excluding ticker)
                    display_cols = ['date', 'open', 'high', 'low', 'close', "volume"]
                    st.dataframe(combined_data[display_cols], use_container_width=True)
                else:
                    st.warning("No collected data available to display.")
            except Exception as e:
                st.error(f"Error collecting stock data with CrewAI: {e}")

    # Training and Forecasting (Automatic)
    st.header("Train Model and Generate Forecasts")
    with st.spinner(f"Checking forecasts for {ticker}..."):
        if not st.session_state.training_done:
            file_path = os.path.join(FORECASTS_DIR, f"{ticker}_jan2025_forecast.csv")
            if os.path.exists(file_path):
                log_action(st.session_state.user_id, f"Forecasts already generated for {ticker} by CrewAI")
                st.success(f"Forecasts already generated for {ticker}!")
                st.session_state.training_done = True
            else:
                st.warning(f"No forecast data found for {ticker}. Ensure 'Collect Stock Data' has been run using CrewAI.")
                st.session_state.training_done = False

    # Historical Stock Data and Analysis (Using Local Data)
    st.header(f"Historical Stock Data and Analysis for {ticker}")
    file_path = os.path.join(DATA_DIR, f"{ticker}_stock_data.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        print(f"Loaded data for {ticker}. Columns: {df.columns.tolist()}")
        df["date"] = pd.to_datetime(df["date"])
        
        # Log app usage
        log_action(st.session_state.user_id, f"Viewed stock data for {ticker}")
        
        # Prepare data for display and chart
        display_df = df[["date", "close", "volume"]].copy()
        display_df["date"] = display_df["date"].apply(lambda x: x.strftime("%d-%m-%Y %I %p").replace("AM", "am").replace("PM", "pm"))
        display_df = display_df.round({'close': 3, 'volume': 3})
        chart_df = df.set_index("date")[["close"]]

        # Calculate additional metrics
        df['volatility'] = calculate_volatility(df)
        df['rsi'] = calculate_rsi(df)
    else:
        st.error(f"No local stock data found for {ticker}. Please run the 'Collect Stock Data' step first.")
        display_df = pd.DataFrame()
        chart_df = pd.DataFrame()
        df = pd.DataFrame()

    # Stock Data Table
    st.subheader(f"Stock Data for {ticker}")
    if not display_df.empty:
        st.dataframe(display_df, use_container_width=True)
    else:
        st.write("No data available to display.")

    # Candlestick Chart (Moved Below Table)
    st.subheader(f"Candlestick Chart for {ticker}")
    if not df.empty and all(col in df.columns for col in ['date', 'open', 'high', 'low', 'close']):
        # Overall Trend Candlestick Chart
        st.markdown("**Overall Trend**")
        st.markdown("**Explanation:** This chart shows the stock's price movements over its entire history using candlesticks. Each candlestick represents a trading day, with the body showing the opening and closing prices, and the wicks showing the high and low prices for that day.")
        fig_overall = go.Figure(data=[go.Candlestick(
            x=df['date'],  # Use full date for x-axis
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close']
        )])
        fig_overall.update_layout(
            title=f"{ticker} Overall Candlestick Chart",
            yaxis_title="Price",
            xaxis_title="Date",
            template="plotly_dark"
        )
        fig_overall.update_xaxes(
            tickformat="%d-%m-%Y",  # Show full date
            rangeslider_visible=True
        )
        st.plotly_chart(fig_overall, use_container_width=True)

        # Yearly Trends (Combined into One Chart with Different Colors)
        st.markdown("**Yearly Trends**")
        st.markdown("**Explanation:** This chart combines candlestick data for each year into a single view, with year labels on the x-axis. It helps identify long-term trends and seasonal patterns in the stock's price movements.")
        df['year'] = df['date'].dt.year
        years = sorted(df['year'].unique())
        colors = px.colors.qualitative.Plotly  # Use Plotly's qualitative colors for differentiation
        fig_yearly = go.Figure()
        
        # Track the start index for each year to create gaps
        current_idx = 0
        x_data = []
        open_data = []
        high_data = []
        low_data = []
        close_data = []
        year_labels = []
        
        for i, year in enumerate(years):
            yearly_df = df[df['year'] == year].copy()
            if not yearly_df.empty:
                # Add a small gap between years by inserting NaN values
                if i > 0:
                    x_data.append(pd.NaT)
                    open_data.append(np.nan)
                    high_data.append(np.nan)
                    low_data.append(np.nan)
                    close_data.append(np.nan)
                    year_labels.append(None)
                    current_idx += 1

                # Add data for the current year
                num_points = len(yearly_df)
                x_data.extend(list(range(current_idx, current_idx + num_points)))
                open_data.extend(yearly_df['open'].values)
                high_data.extend(yearly_df['high'].values)
                low_data.extend(yearly_df['low'].values)
                close_data.extend(yearly_df['close'].values)
                year_labels.extend([year] * num_points)
                current_idx += num_points

        # Create the combined candlestick chart
        if x_data:
            fig_yearly.add_trace(go.Candlestick(
                x=x_data,
                open=open_data,
                high=high_data,
                low=low_data,
                close=close_data,
                name="Candlestick",
                increasing_line_color=colors[0 % len(colors)],
                decreasing_line_color=colors[1 % len(colors)]
            ))

            # Update layout to show year on x-axis
            fig_yearly.update_layout(
                title=f"{ticker} Yearly Candlestick Chart",
                yaxis_title="Price",
                xaxis_title="Year",
                template="plotly_dark",
                xaxis=dict(
                    tickvals=[x_data[year_labels.index(year)] for year in years if year in year_labels],
                    ticktext=years,
                    showticklabels=True,
                    tickmode='array'
                )
            )
            st.plotly_chart(fig_yearly, use_container_width=True)
        else:
            st.write("Insufficient data to plot yearly trends.")
    else:
        st.write("No data available to plot or missing required columns.")

    # Additional Historical Analysis
    st.subheader(f"Historical Analysis for {ticker}")

    # Volatility chart
    st.subheader("Historical Volatility")
    st.markdown("""
    **What is Volatility?** Volatility measures the degree of variation in a stock's price over time, typically calculated as the annualized standard deviation of daily returns. 
    Higher volatility indicates larger price swings, suggesting greater risk but also potential for higher returns. Lower volatility implies more stable price movements, often associated with lower risk.
    """)
    st.markdown("**Explanation:** This graph shows the annualized volatility over a 60-day rolling window. It helps identify periods of high or low price fluctuation, which can indicate market uncertainty or stability.")
    if not df.empty and 'volatility' in df.columns and len(df) >= 60:
        volatility_df = df[['date', 'volatility']].copy()
        volatility_df['volatility'] = volatility_df['volatility'].fillna(0)  # Fill NaN with 0
        if volatility_df['volatility'].sum() > 0:  # Check if there are non-zero volatility values
            fig = px.line(volatility_df, x='date', y='volatility', title="Annualized Volatility (60-day)")
            fig.update_layout(template="plotly_dark")
            fig.update_xaxes(tickformat="%d-%m-%Y")  # Show full date
            st.plotly_chart(fig)
        else:
            st.write("Volatility data is zero or insufficient to plot (requires at least 60 days).")
    else:
        st.write("No volatility data available to plot or insufficient data (requires at least 60 days).")

    # RSI chart
    st.subheader("14-Day RSI")
    st.markdown("""
    **What is RSI?** The Relative Strength Index (RSI) is a momentum indicator that measures the speed and change of price movements on a scale of 0 to 100. 
    An RSI above 70 indicates the stock may be overbought (potentially overvalued, suggesting a sell), while an RSI below 30 suggests it may be oversold (potentially undervalued, suggesting a buy). 
    Values between 30 and 70 indicate neutral momentum.
    """)
    st.markdown("**Explanation:** This graph displays the 14-day RSI, helping you identify whether the stock is overbought (above 70) or oversold (below 30), which can signal potential buying or selling opportunities.")
    if not df.empty and 'rsi' in df.columns and len(df) >= 14:
        rsi_df = df[['date', 'rsi']].dropna()
        if not rsi_df.empty:
            fig = px.line(rsi_df, x='date', y='rsi', title="14-Day RSI")
            fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
            fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
            fig.update_layout(template="plotly_dark")
            fig.update_xaxes(tickformat="%d-%m-%Y")  # Show full date
            st.plotly_chart(fig)
        else:
            st.write("No RSI data available to plot (insufficient data points).")
    else:
        st.write("No RSI data available to plot or insufficient data (requires at least 14 days).")

    # Historical Summary
    st.subheader("Historical Summary")
    if not df.empty:
        # Calculate key metrics
        recent_df = df.tail(60)  # Last 60 days
        avg_close = recent_df['close'].mean()
        recent_volatility = recent_df['volatility'].iloc[-1] if not recent_df['volatility'].empty else 0
        current_rsi = recent_df['rsi'].iloc[-1] if not recent_df['rsi'].empty else 0
        price_trend = ((recent_df['close'].iloc[-1] - recent_df['close'].iloc[-30]) / recent_df['close'].iloc[-30]) * 100 if len(recent_df) >= 30 else 0
        support_level = recent_df['close'].min()
        resistance_level = recent_df['close'].max()

        # Load 2020 growth% and annual growth from CrewAI analytics
        analytics_file = os.path.join(FORECASTS_DIR, "analytics_and_forecasts.json")
        growth_2020 = 0
        annual_growth = {}
        if os.path.exists(analytics_file):
            try:
                with open(analytics_file, 'r') as f:
                    import json
                    analytics_data = json.load(f)
                if 'analytics' in analytics_data and ticker in analytics_data['analytics']:
                    growth_2020 = analytics_data['analytics'][ticker]['growth_2020']
                    annual_growth = analytics_data['analytics'][ticker].get('annual_growth', {})
            except Exception as e:
                st.warning(f"Could not load analytics for {ticker}: {e}")

        # Store metrics for PDF report
        summary_metrics = {
            'avg_close': avg_close,
            'recent_volatility': recent_volatility,
            'current_rsi': current_rsi,
            'price_trend': price_trend,
            'support_level': support_level,
            'resistance_level': resistance_level,
            'growth_2020': growth_2020,
            'annual_growth': annual_growth
        }

        # Simple explanation
        st.markdown("This section provides a summary of key metrics for the stock over the last 60 days, including its average price, volatility, RSI, and growth trends over the years.")

        # Display summary as a table
        summary_data = [
            ["Metric", "Value"],
            ["Average Closing Price (Last 60 Days)", f"${avg_close:.2f}"],
            ["Recent Volatility (Last 60 Days, Annualized)", f"{recent_volatility:.2f}"],
            ["Current RSI (14-Day)", f"{current_rsi:.2f}"],
            ["Price Trend (Last 30 Days)", f"{price_trend:.2f}%"],
            ["2020 Growth Percentage", f"{growth_2020:.2f}%"],
            ["Support Level (Last 60 Days)", f"${support_level:.2f}"],
            ["Resistance Level (Last 60 Days)", f"${resistance_level:.2f}"]
        ]
        summary_df = pd.DataFrame(summary_data[1:], columns=summary_data[0])
        st.table(summary_df)

        # Annual Growth Over Time (Graph)
        st.subheader("Annual Growth Over Time")
        st.markdown("This graph shows the year-over-year growth percentage for each year, helping you understand the stock's historical performance trends over time.")
        if annual_growth:
            annual_growth_df = pd.DataFrame(
                list(annual_growth.items()),
                columns=['Year', 'Growth (%)']
            )
            annual_growth_df['Year'] = annual_growth_df['Year'].astype(int)
            annual_growth_df = annual_growth_df.sort_values('Year')
            
            fig = px.line(
                annual_growth_df,
                x='Year',
                y='Growth (%)',
                title=f"Annual Growth Over Time for {ticker}",
                template="plotly_dark",
                markers=True
            )
            fig.update_xaxes(tickvals=annual_growth_df['Year'])
            st.plotly_chart(fig)
        else:
            st.write("No annual growth data available to plot.")
    else:
        st.write("No historical data available for summary.")
        summary_metrics = {}

    # Recommendation
    st.header("Recommendation")
    if not df.empty:
        # Load AI-generated recommendation directly as a dictionary
        recommendation_file = os.path.join(FORECASTS_DIR, "recommendations.json")
        recommendation = "No recommendation available."
        if os.path.exists(recommendation_file):
            try:
                with open(recommendation_file, 'r') as f:
                    import json
                    recommendations = json.load(f)
                if ticker in recommendations:
                    recommendation = recommendations[ticker]
                    # Parse recommendation into bullet points
                    lines = recommendation.split('\n\n')
                    if len(lines) >= 2:
                        market_summary = lines[0].replace("**Market Trend Summary:** ", "")
                        rec_action = lines[1].replace("**Recommendation:** ", "")
                        st.markdown(f"- **Market Trend Summary:** {market_summary}")
                        st.markdown(f"- **Recommendation:** {rec_action}")
                    else:
                        st.markdown(f"- {recommendation}")
                else:
                    st.markdown(f"- No recommendation found for {ticker}.")
            except Exception as e:
                st.markdown(f"- Error loading recommendation for {ticker}: {e}")
        else:
            st.markdown("- No recommendation available.")
    else:
        st.markdown("- No historical data available for generating a recommendation.")
        recommendation = "No recommendation available due to lack of historical data."

    # Individual January 2025 Forecasts (Only for the Selected Ticker)
    st.header("Individual January 2025 Forecasts")
    st.subheader(f"Forecast for {ticker}")
    file_path = os.path.join(FORECASTS_DIR, f"{ticker}_jan2025_forecast.csv")
    if os.path.exists(file_path):
        forecast_df = pd.read_csv(file_path)
        forecast_df['date'] = pd.to_datetime(forecast_df['date'])
        
        # Create a display copy with formatted dates, but keep the original forecast_df unchanged
        forecast_display_df = forecast_df.copy()
        forecast_display_df['date'] = forecast_display_df['date'].apply(lambda x: x.strftime("%d-%m-%Y %I %p").replace("AM", "am").replace("PM", "pm"))

        # Format numerical columns to 3 decimal places
        forecast_display_df = forecast_display_df.round({'forecasted_close': 3})

        # Display the forecast table
        st.dataframe(forecast_display_df[['date', 'forecasted_close']], use_container_width=True)
        
        # Plot historical vs forecast with confidence intervals
        if not df.empty and 'close' in df.columns:
            historical_df_plot = df.tail(60)[['date', 'close']].copy()
            historical_df_plot['type'] = 'Historical'
            forecast_df['type'] = 'Forecast'
            # Combine historical and forecast data
            combined_df = pd.concat([
                historical_df_plot.rename(columns={'close': 'price'}),
                forecast_df.rename(columns={'forecasted_close': 'price'})
            ])
            # Add confidence intervals (±5%)
            forecast_df['lower_bound'] = forecast_df['forecasted_close'] * 0.95
            forecast_df['upper_bound'] = forecast_df['forecasted_close'] * 1.05
            
            # Plot with full date on x-axis
            st.markdown("**Explanation:** This graph compares the last 60 days of historical closing prices with the forecasted closing prices for January 2025. The shaded area represents a confidence interval (±5%) around the forecast, indicating potential price variation.")
            fig = go.Figure()
            # Historical data
            fig.add_trace(go.Scatter(
                x=historical_df_plot['date'],
                y=historical_df_plot['close'],
                mode='lines',
                name='Historical',
                line=dict(color='blue')
            ))
            # Forecast data
            fig.add_trace(go.Scatter(
                x=forecast_df['date'],
                y=forecast_df['forecasted_close'],
                mode='lines',
                name='Forecast',
                line=dict(color='orange')
            ))
            # Confidence intervals
            fig.add_trace(go.Scatter(
                x=forecast_df['date'],
                y=forecast_df['upper_bound'],
                mode='lines',
                name='Upper Bound (+5%)',
                line=dict(color='gray', dash='dash'),
                fill=None
            ))
            fig.add_trace(go.Scatter(
                x=forecast_df['date'],
                y=forecast_df['lower_bound'],
                mode='lines',
                name='Lower Bound (-5%)',
                line=dict(color='gray', dash='dash'),
                fill='tonexty',
                fillcolor='rgba(128, 128, 128, 0.2)'
            ))
            fig.update_layout(
                title=f"{ticker} Historical vs Forecasted Close Prices",
                xaxis_title="Date",
                yaxis_title="Price",
                template="plotly_dark"
            )
            fig.update_xaxes(tickformat="%d-%m-%Y")  # Show full date
            st.plotly_chart(fig)
        else:
            # Display only the forecast chart if historical data is unavailable
            st.markdown("**Explanation:** This graph shows the forecasted closing prices for January 2025, helping you understand the predicted price trend for the stock.")
            fig = px.line(forecast_display_df, x='date', y='forecasted_close', title=f"{ticker} Forecasted Close Prices (Jan 2025)")
            fig.update_layout(template="plotly_dark")
            fig.update_xaxes(tickformat="%d-%m-%Y")  # Show full date
            st.plotly_chart(fig)
        
        # Log app usage
        log_action(st.session_state.user_id, f"Viewed forecast for {ticker}")
    else:
        st.warning(f"No forecast data available for {ticker}. Training and forecasting have been initiated automatically.")
        forecast_df = pd.DataFrame()  # Empty DataFrame for PDF report if forecast is unavailable

    # Compare January 2025 Forecasts Across All Tickers
    st.header("Compare January 2025 Forecasts Across All Tickers")
    
    # Create a combined DataFrame for the table
    combined_forecast_df = pd.DataFrame()
    forecast_dfs = []
    colors = ['orange', 'green', 'red', 'purple', 'yellow', 'blue', 'pink']  # Colors for each ticker
    
    for t in TICKERS:
        file_path = os.path.join(FORECASTS_DIR, f"{t}_jan2025_forecast.csv")
        if os.path.exists(file_path):
            df_forecast = pd.read_csv(file_path)
            df_forecast['date'] = pd.to_datetime(df_forecast['date'])
            df_forecast['ticker'] = t
            forecast_dfs.append(df_forecast)
            # Rename the forecasted_close column to the ticker name for the combined table
            df_ticker = df_forecast[['date', 'forecasted_close']].rename(columns={'forecasted_close': t})
            if combined_forecast_df.empty:
                combined_forecast_df = df_ticker
            else:
                combined_forecast_df = combined_forecast_df.merge(df_ticker, on='date', how='outer')

    # Display the combined forecast table
    if not combined_forecast_df.empty:
        st.subheader("Combined Forecast Table")
        combined_forecast_display_df = combined_forecast_df.copy()
        combined_forecast_display_df['date'] = combined_forecast_display_df['date'].apply(lambda x: x.strftime("%d-%m-%Y %I %p").replace("AM", "am").replace("PM", "pm"))
        combined_forecast_display_df = combined_forecast_display_df.round({t: 3 for t in TICKERS})
        st.dataframe(combined_forecast_display_df, use_container_width=True)

        # Plot all forecasts in a single graph
        st.subheader("Combined Forecast Graph")
        st.markdown("**Explanation:** This graph compares the forecasted closing prices for all selected tickers in January 2025, allowing you to see how different stocks are expected to perform relative to each other.")
        fig = go.Figure()
        for i, df_forecast in enumerate(forecast_dfs):
            fig.add_trace(go.Scatter(
                x=df_forecast['date'],
                y=df_forecast['forecasted_close'],
                mode='lines',
                name=df_forecast['ticker'].iloc[0],
                line=dict(color=colors[i % len(colors)])
            ))
        fig.update_layout(
            title="January 2025 Forecasted Close Prices for All Tickers",
            xaxis_title="Date",
            yaxis_title="Forecasted Close Price",
            template="plotly_dark"
        )
        fig.update_xaxes(tickformat="%d-%m-%Y")  # Show full date
        st.plotly_chart(fig)

        # Forecast Analysis and Review (Bullet Points with Pros and Cons)
        st.subheader("Forecast Analysis and Review")
        # Calculate key metrics for review
        forecast_stats = {}
        for t in TICKERS:
            if t in combined_forecast_df.columns:
                forecast_data = combined_forecast_df[t].dropna()
                if not forecast_data.empty:
                    avg_forecast = forecast_data.mean()
                    min_forecast = forecast_data.min()
                    max_forecast = forecast_data.max()
                    price_range = max_forecast - min_forecast
                    price_change = ((max_forecast - min_forecast) / min_forecast) * 100 if min_forecast != 0 else 0
                    forecast_stats[t] = {
                        "average": avg_forecast,
                        "min": min_forecast,
                        "max": max_forecast,
                        "range": price_range,
                        "change_percent": price_change
                    }

        # Generate bullet points based on the calculations
        st.markdown("**Key Metrics:**")
        for t, stats in forecast_stats.items():
            st.markdown(f"""
            - **{t}:**
              - Average Forecast: ${stats['average']:.2f}
              - Minimum Forecast: ${stats['min']:.2f}
              - Maximum Forecast: ${stats['max']:.2f}
              - Price Range: ${stats['range']:.2f}
              - Percentage Change: {stats['change_percent']:.2f}%
            """)

        # Comparative insight
        avg_prices = {t: stats['average'] for t, stats in forecast_stats.items()}
        highest_avg_ticker = max(avg_prices, key=avg_prices.get)
        lowest_avg_ticker = min(avg_prices, key=avg_prices.get)
        st.markdown("**Comparative Insight:**")
        st.markdown(f"""
        - Highest Average Forecast: {highest_avg_ticker} at ${avg_prices[highest_avg_ticker]:.2f}—consider for potential growth.
        - Lowest Average Forecast: {lowest_avg_ticker} at ${avg_prices[lowest_avg_ticker]:.2f}—may require caution.
        """)

        # Pros and Cons Table
        st.markdown("**Pros and Cons:**")
        pros_cons_data = [['Ticker', 'Pros', 'Cons']]
        for t, stats in forecast_stats.items():
            pros = []
            cons = []
            # Pros
            if stats['change_percent'] > 5:
                pros.append("✅ Significant growth potential")
            if stats['average'] == avg_prices[highest_avg_ticker]:
                pros.append("✅ Highest average forecast")
            if stats['range'] < 5:
                pros.append("✅ Low volatility (stable forecast)")
            # Cons
            if stats['change_percent'] < -5:
                cons.append("⚠ Notable downward trend")
            if stats['average'] == avg_prices[lowest_avg_ticker]:
                cons.append("⚠ Lowest average forecast")
            if stats['range'] > 10:
                cons.append("⚠ High volatility (risky forecast)")
            # If no pros or cons, add a neutral comment
            if not pros:
                pros.append("No significant pros")
            if not cons:
                cons.append("No significant cons")
            pros_cons_data.append([t, ", ".join(pros), ", ".join(cons)])

        pros_cons_table = pd.DataFrame(pros_cons_data[1:], columns=pros_cons_data[0])
        st.table(pros_cons_table)

        # Overview
        st.markdown("**Overview:**")
        overview = "Based on the January 2025 forecasts, "
        high_growth = [t for t, stats in forecast_stats.items() if stats['change_percent'] > 5]
        high_risk = [t for t, stats in forecast_stats.items() if stats['range'] > 10]
        if high_growth:
            overview += f"{', '.join(high_growth)} show promising growth potential, making them attractive for investors seeking gains. "
        if high_risk:
            overview += f"However, {', '.join(high_risk)} exhibit high volatility, indicating potential risk—proceed with caution. "
        overview += f"Overall, {highest_avg_ticker} stands out with the highest average forecast, while {lowest_avg_ticker} may require careful monitoring due to its lower forecast."
        st.markdown(overview)

    else:
        st.warning("No forecast data available for comparison. Please run the training and forecasting step for all tickers.")

    # Sector Comparison
    st.header("Sector Comparison")
    st.markdown("**Explanation:** This section compares key metrics across different sectors (Tech, Finance, Sportswear) to help you understand sector-level performance trends.")
    analytics_file = os.path.join(FORECASTS_DIR, "analytics_and_forecasts.json")
    if os.path.exists(analytics_file):
        try:
            with open(analytics_file, 'r') as f:
                import json
                analytics_data = json.load(f)
            if 'sector_analytics' in analytics_data:
                sector_analytics = analytics_data['sector_analytics']
                st.subheader("Sector-Level Metrics")
                sector_data = []
                for sector, metrics in sector_analytics.items():
                    sector_data.append([
                        sector,
                        ", ".join(metrics['tickers']),
                        f"{metrics['average_2020_growth']:.2f}%",
                        f"{metrics['average_price_trend']:.2f}%",
                        f"${metrics['average_closing_price']:.2f}"
                    ])
                sector_df = pd.DataFrame(sector_data, columns=[
                    "Sector", "Tickers", "Average 2020 Growth", "Average Price Trend (Last 30 Days)", "Average Closing Price (Last 60 Days)"
                ])
                st.table(sector_df)

                # Plot sector comparisons
                st.subheader("Sector Comparison Graphs")
                # Bar chart for average 2020 growth
                fig_growth = px.bar(
                    sector_df,
                    x="Sector",
                    y="Average 2020 Growth",
                    title="Average 2020 Growth by Sector",
                    text="Average 2020 Growth",
                    template="plotly_dark"
                )
                fig_growth.update_traces(texttemplate='%{text}', textposition='auto')
                st.plotly_chart(fig_growth)

                # Bar chart for average price trend
                fig_trend = px.bar(
                    sector_df,
                    x="Sector",
                    y="Average Price Trend (Last 30 Days)",
                    title="Average Price Trend (Last 30 Days) by Sector",
                    text="Average Price Trend (Last 30 Days)",
                    template="plotly_dark"
                )
                fig_trend.update_traces(texttemplate='%{text}', textposition='auto')
                st.plotly_chart(fig_trend)

                # Bar chart for average closing price
                fig_price = px.bar(
                    sector_df,
                    x="Sector",
                    y="Average Closing Price (Last 60 Days)",
                    title="Average Closing Price (Last 60 Days) by Sector",
                    text="Average Closing Price (Last 60 Days)",
                    template="plotly_dark"
                )
                fig_price.update_traces(texttemplate='%{text}', textposition='auto')
                st.plotly_chart(fig_price)
            else:
                st.warning("No sector analytics data available. Please run 'Collect Stock Data' to generate analytics.")
        except Exception as e:
            st.error(f"Error loading sector analytics: {e}")
    else:
        st.warning("No sector analytics data available. Please run 'Collect Stock Data' to generate analytics.")

    # Download PDF Report (Updated Description)
    st.header("Download PDF Report")
    st.markdown("""
    **What’s Included in the Report:**
    - Historical Stock Data (last 60 days): A table of daily closing prices and trading volumes.
    - Historical Summary: Key metrics like average closing price, volatility, RSI, price trend, and support/resistance levels.
    - Recommendation: A suggested action (buy, sell, or hold) based on the historical analysis.
    - January 2025 Forecast: A table of predicted closing prices for the selected ticker.
    - Forecast vs. Real Data Comparison: A table comparing forecasted and actual closing prices.
    """)
    if st.button("Generate and Download PDF Report"):
        # Load historical data again to ensure df is correct
        historical_file_path = os.path.join(DATA_DIR, f"{ticker}_stock_data.csv")
        if os.path.exists(historical_file_path):
            historical_df = pd.read_csv(historical_file_path)
            historical_df["date"] = pd.to_datetime(historical_df["date"])
        else:
            historical_df = pd.DataFrame()
        
        if not historical_df.empty and not forecast_df.empty:
            pdf_buffer = generate_pdf_report(ticker, historical_df, summary_metrics, recommendation, forecast_df)
            st.download_button(
                label="Click to download",
                data=pdf_buffer,
                file_name=f"{ticker}_report.pdf",
                mime="application/pdf"
            )
            log_action(st.session_state.user_id, f"Downloaded report for {ticker}")
            st.success(f"Report saved to {REPORTS_DIR}")
        else:
            st.error("Cannot generate PDF report: Historical or forecast data is missing.")

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
            st.rerun()
        else:
            st.error(message)
    if st.button("Go to Sign Up"):
        st.session_state.page = "signup"
        st.rerun()

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
            st.rerun()
        else:
            st.error(message)
    if st.button("Go to Login"):
        st.session_state.page = "login"
        st.rerun()

# Page routing
if st.session_state.user_id is None:
    if st.session_state.page == "login":
        login_page()
    elif st.session_state.page == "signup":
        signup_page()
else:
    show_dashboard()