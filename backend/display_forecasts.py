import sys
from PySide6.QtWidgets import QMainWindow, QApplication, QTabWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget
import pandas as pd
import os

FORECASTS_DIR = r"C:\Users\algha\OneDrive\Documents\Market_Eye\forecasts"
TICKERS = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

class ForecastApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Market Eye - Stock Forecasts")
        self.setGeometry(100, 100, 800, 600)

        # Create a tab widget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Add a tab for each ticker
        for ticker in TICKERS:
            tab = QWidget()
            layout = QVBoxLayout()
            table = QTableWidget()
            self.load_forecast_data(ticker, table)
            layout.addWidget(table)
            tab.setLayout(layout)
            self.tabs.addTab(tab, ticker)

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
    window = ForecastApp()
    window.show()
    sys.exit(app.exec())