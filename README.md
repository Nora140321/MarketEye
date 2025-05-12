# Market Eye: AI-Powered Stock Analysis System

## Overview
Market Eye is an AI-powered stock analysis system that collects historical stock data, processes and stores it in a SQLite database, performs analysis using AI models, generates PDF reports with visualizations, and provides an API and frontend interface to access the results. The system supports analysis for five major stocks: AAPL, GOOGL, MSFT, AMZN, and TSLA.

## Features
- **Data Collection**: Fetches historical stock data using `yfinance`.
- **Data Processing**: Processes and stores data in a SQLite database (`stock_data.db`).
- **Analysis**: Performs AI-powered analysis (mocked due to API quota limits) and generates summaries.
- **Reporting**: Creates PDF reports with data tables and charts using `reportlab` and `matplotlib`.
- **API Service**: Exposes endpoints to access stock data, analysis summaries, and reports via `FastAPI`.
- **Frontend**: Provides a user-friendly interface to view data, charts, and download reports using `Streamlit`.

## Project Structure
- `data/`: Contains raw stock data CSV files.
- `database/`: Stores the SQLite database (`stock_data.db`).
- `backend/`: Contains scripts for data collection and processing (`process_stock_dataset.py`, `process_and_store_data.py`).
- `agents/`: Contains the analysis script (`analyze_stock_data.py`).
- `reports/`: Contains analysis summaries (`<ticker>_analysis.txt`) and PDF reports (`<ticker>_report.pdf`).
- `api/`: FastAPI application (`main.py`) for serving data and reports.
- `frontend/`: Streamlit application (`app.py`) for the web interface.
- `requirements.txt`: Lists project dependencies.

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd Market_Eye