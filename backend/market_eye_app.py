import sys
import os
import subprocess
import pandas as pd
from PySide6.QtWidgets import (QMainWindow, QApplication, QTabWidget, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget, QPushButton, QTextEdit, QProgressBar, QComboBox)
from PySide6.QtCore import QThread, Signal
import collect_stock_data  # Import the collect_stock_data script as a module
import train_lstm_model   # Import the train_lstm_model script as a module

FORECASTS_DIR = r"C:\Users\algha\OneDrive\Documents\Market_Eye\forecasts"
TICKERS = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

class WorkerThread(QThread):
    update_log = Signal(str)
    update_progress = Signal(int)
    finished = Signal()

    def __init__(self, func):
        super().__init__()
        self.func = func

    def run(self):
        try:
            # Simulate progress updates
            for i in range(1, 101):
                self.update_progress.emit(i)
                self.msleep(50)  # Simulate work being done
            # Run the actual function
            self.func()
            self.update_log.emit("Process completed successfully!")
        except Exception as e:
            self.update_log.emit(f"Error: {str(e)}")
        finally:
            self.finished.emit()

class MarketEyeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Market Eye - AI-Powered Stock Analysis")
        self.setGeometry(100, 100, 1000, 700)

        # Create a tab widget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Data Collection Tab
        self.data_tab = QWidget()
        self.data_layout = QVBoxLayout()
        self.data_collect_button = QPushButton("Collect Stock Data")
        self.data_collect_button.clicked.connect(self.run_data_collection)
        self.data_layout.addWidget(self.data_collect_button)
        self.data_progress = QProgressBar()
        self.data_progress.setValue(0)
        self.data_layout.addWidget(self.data_progress)
        self.data_log = QTextEdit()
        self.data_log.setReadOnly(True)
        self.data_layout.addWidget(self.data_log)
        self.data_tab.setLayout(self.data_layout)
        self.tabs.addTab(self.data_tab, "Data Collection")

        # Training Tab
        self.train_tab = QWidget()
        self.train_layout = QVBoxLayout()
        self.ticker_combo = QComboBox()
        self.ticker_combo.addItems(["All"] + TICKERS)
        self.train_layout.addWidget(self.ticker_combo)
        self.train_button = QPushButton("Train Model and Forecast")
        self.train_button.clicked.connect(self.run_training)
        self.train_layout.addWidget(self.train_button)
        self.train_progress = QProgressBar()
        self.train_progress.setValue(0)
        self.train_layout.addWidget(self.train_progress)
        self.train_log = QTextEdit()
        self.train_log.setReadOnly(True)
        self.train_layout.addWidget(self.train_log)
        self.train_tab.setLayout(self.train_layout)
        self.tabs.addTab(self.train_tab, "Training")

        # Forecasts Tab
        self.forecast_tabs = QTabWidget()
        for ticker in TICKERS:
            tab = QWidget()
            layout = QVBoxLayout()
            table = QTableWidget()
            self.load_forecast_data(ticker, table)
            layout.addWidget(table)
            tab.setLayout(layout)
            self.forecast_tabs.addTab(tab, ticker)
        self.tabs.addTab(self.forecast_tabs, "Forecasts")

    def run_data_collection(self):
        self.data_log.clear()
        self.data_progress.setValue(0)
        self.data_collect_button.setEnabled(False)

        def collect_data():
            collect_stock_data.main()  # Call the main function from collect_stock_data.py

        self.worker = WorkerThread(collect_data)
        self.worker.update_log.connect(self.data_log.append)
        self.worker.update_progress.connect(self.data_progress.setValue)
        self.worker.finished.connect(lambda: self.data_collect_button.setEnabled(True))
        self.worker.start()

    def run_training(self):
        self.train_log.clear()
        self.train_progress.setValue(0)
        self.train_button.setEnabled(False)

        selected_ticker = self.ticker_combo.currentText()

        def train_and_forecast():
            if selected_ticker == "All":
                for ticker in TICKERS:
                    self.train_log.append(f"Training model for {ticker}...")
                    train_lstm_model.train_lstm_model(ticker)
                    self.train_log.append(f"Forecasting for {ticker}...")
                    train_lstm_model.forecast_january_2025(ticker)
            else:
                self.train_log.append(f"Training model for {selected_ticker}...")
                train_lstm_model.train_lstm_model(selected_ticker)
                self.train_log.append(f"Forecasting for {selected_ticker}...")
                train_lstm_model.forecast_january_2025(selected_ticker)
            # Refresh the forecasts tab
            for i in range(self.forecast_tabs.count()):
                ticker = self.forecast_tabs.tabText(i)
                table = self.forecast_tabs.widget(i).layout().itemAt(0).widget()
                self.load_forecast_data(ticker, table)

        self.worker = WorkerThread(train_and_forecast)
        self.worker.update_log.connect(self.train_log.append)
        self.worker.update_progress.connect(self.train_progress.setValue)
        self.worker.finished.connect(lambda: self.train_button.setEnabled(True))
        self.worker.start()

    def load_forecast_data(self, ticker, table):
        file_path = os.path.join(FORECASTS_DIR, f"{ticker}_jan2025_forecast.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            table.setRowCount(len(df))
            table.setColumnCount(2)
            table.setHorizontalHeaderLabels(["Date", "Forecasted Close"])
            for i, row in df.iterrows():
                table.setItem(i, 0, QTableWidgetItem(str(row['date'])))
                table.setItem(i, 1, QTableWidgetItem(f"{row['forecasted_close']:.2f}"))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MarketEyeApp()
    window.show()
    sys.exit(app.exec())