import sqlite3
import bcrypt
import os
from datetime import datetime

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

def hash_password(password):
    """
    Hash a password using bcrypt.
    """
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def verify_password(password, hashed):
    """
    Verify a password against its hash.
    """
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

def register_user(username, password):
    """
    Register a new user with a hashed password.
    """
    conn = create_connection()
    if conn is None:
        return False, "Database connection failed."

    try:
        cursor = conn.cursor()
        # Check if username already exists
        cursor.execute("SELECT username FROM users WHERE username = ?", (username,))
        if cursor.fetchone():
            conn.close()
            return False, "Username already exists."

        # Hash the password
        hashed_password = hash_password(password)
        registration_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Insert the new user
        cursor.execute(
            "INSERT INTO users (username, password_hashed, registration_date) VALUES (?, ?, ?)",
            (username, hashed_password, registration_date)
        )
        conn.commit()
        
        # Get the user_id of the newly registered user
        cursor.execute("SELECT user_id FROM users WHERE username = ?", (username,))
        user_id = cursor.fetchone()[0]
        
        # Log the registration action
        log_action(user_id, "User registered")
        conn.close()
        return True, "Registration successful."
    except Exception as e:
        conn.close()
        return False, f"Registration failed: {e}"

def login_user(username, password):
    """
    Authenticate a user and log the login action.
    """
    conn = create_connection()
    if conn is None:
        return False, None, "Database connection failed."

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT user_id, password_hashed FROM users WHERE username = ?", (username,))
        result = cursor.fetchone()
        if not result:
            conn.close()
            return False, None, "Invalid username or password."

        user_id, hashed_password = result
        if verify_password(password, hashed_password):
            # Log the login action
            log_action(user_id, "User logged in")
            conn.close()
            return True, user_id, "Login successful."
        else:
            conn.close()
            return False, None, "Invalid username or password."
    except Exception as e:
        conn.close()
        return False, None, f"Login failed: {e}"

def log_action(user_id, action):
    """
    Log a user action into the activity_logs table.
    """
    conn = create_connection()
    if conn is None:
        return

    try:
        cursor = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            "INSERT INTO activity_logs (user_id, action, timestamp) VALUES (?, ?, ?)",
            (user_id, action, timestamp)
        )
        conn.commit()
    except Exception as e:
        print(f"Error logging action: {e}")
    finally:
        conn.close()