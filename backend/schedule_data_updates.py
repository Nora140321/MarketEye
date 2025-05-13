import schedule
import time
import collect_stock_data

def job():
    print("Running daily stock data update...")
    collect_stock_data.main()
    print("Stock data update completed.")

# Schedule the job to run daily at 12:00 AM
schedule.every().day.at("00:00").do(job)

def main():
    print("Starting stock data update scheduler...")
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    main()