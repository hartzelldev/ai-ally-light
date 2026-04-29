import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Optional

class ProjectDatabase:
    def __init__(self, project_path: Path):
        self.db_path = project_path / "history.db"
        self._init_db()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initializes tables for a specific project."""
        with self._get_connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS threads (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_archived BOOLEAN DEFAULT 0
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (thread_id) REFERENCES threads (id) ON DELETE CASCADE
                )
            ''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_id)')

    def get_threads(self, limit: Optional[int] = None, offset: int = 0):
        """Fetches active threads, newest first."""
        # Use last_updated to match your schema's column name
        query = "SELECT * FROM threads WHERE is_archived = 0 ORDER BY last_updated DESC"
        if limit:
            query += f" LIMIT {limit} OFFSET {offset}"
        
        with self._get_connection() as conn:
            return [dict(row) for row in conn.execute(query).fetchall()]

    def get_thread_by_id(self, thread_id: int):
        """Fetches a single thread by its ID. (Now correctly inside the class)"""
        query = "SELECT * FROM threads WHERE id = ?"
        with self._get_connection() as conn:
            row = conn.execute(query, (thread_id,)).fetchone()
            return dict(row) if row else None

    def get_messages(self, thread_id: int):
        """Fetches all messages for a specific thread to display in the chat."""
        query = "SELECT role, content, timestamp FROM messages WHERE thread_id = ? ORDER BY timestamp ASC"
        with self._get_connection() as conn:
            return [dict(row) for row in conn.execute(query, (thread_id,)).fetchall()]

    def create_thread(self, name: str) -> int:
        with self._get_connection() as conn:
            cursor = conn.execute("INSERT INTO threads (name) VALUES (?)", (name,))
            return cursor.lastrowid

    def rename_thread(self, thread_id: int, new_name: str):
        with self._get_connection() as conn:
            conn.execute("UPDATE threads SET name = ? WHERE id = ?", (new_name, thread_id))

    def delete_thread(self, thread_id: int):
        with self._get_connection() as conn:
            conn.execute("DELETE FROM threads WHERE id = ?", (thread_id,))

    def save_message(self, thread_id: int, role: str, content: str):
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO messages (thread_id, role, content) VALUES (?, ?, ?)",
                (thread_id, role, content)
            )
            conn.execute(
                "UPDATE threads SET last_updated = CURRENT_TIMESTAMP WHERE id = ?",
                (thread_id,)
            )
            