import sqlite3
import os
from pathlib import Path
from datetime import datetime

def get_db_connection(project_path: Path):
    """Establishes a connection to the project's local history.db."""
    db_path = project_path / "history.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # This lets us access columns by name
    return conn

def init_project_db(project_path: Path):
    """Creates the necessary tables if they don't exist."""
    conn = get_db_connection(project_path)
    cursor = conn.cursor()
    
    # Create Threads table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS threads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_archived BOOLEAN DEFAULT 0
        )
    ''')
    
    # Create Messages table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (thread_id) REFERENCES threads (id) ON DELETE CASCADE
        )
    ''')
    
    conn.commit()
    conn.close()
    