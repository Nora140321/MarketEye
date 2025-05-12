import sqlite3
import os

# Define database path
DB_PATH = os.path.join(os.path.dirname(__file__), "../database/users.db")

def create_connection():
    """
    Create a connection to the SQLite database.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

def create_tables(conn):
    """
    Create the users and activity_logs tables if they don't exist.
    """
    try:
        cursor = conn.cursor()
        # Create users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hashed TEXT NOT NULL,
                registration_date TEXT NOT NULL
            )
        """)
        # Create activity_logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS activity_logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                action TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        """)
        conn.commit()
        print("Tables created successfully.")
    except Exception as e:
        print(f"Error creating tables: {e}")

def setup_database():
    """
    Set up the database and tables.
    """
    conn = create_connection()
    if conn is None:
        return
    create_tables(conn)
    conn.close()

if __name__ == "__main__":
    setup_database()