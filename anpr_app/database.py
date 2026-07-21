import datetime
import hashlib
import os
import sqlite3

from .config import DB, UPLOAD_DIR, VIDEO_DIR


def get_db():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(VIDEO_DIR, exist_ok=True)

    with get_db() as db:
        db.execute(
            """CREATE TABLE IF NOT EXISTS watchlist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate TEXT UNIQUE NOT NULL,
                owner TEXT,
                reason TEXT,
                added TEXT
            )"""
        )
        db.execute(
            """CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate TEXT,
                matched INTEGER,
                confidence REAL,
                timestamp TEXT,
                image_path TEXT,
                detected_by TEXT
            )"""
        )
        db.execute(
            """CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                role TEXT DEFAULT 'user',
                created TEXT
            )"""
        )
        db.execute(
            """CREATE TABLE IF NOT EXISTS authorities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                role TEXT,
                phone TEXT,
                added TEXT
            )"""
        )

        admin_pass = hashlib.sha256("admin123".encode()).hexdigest()
        db.execute(
            """INSERT OR IGNORE INTO users (username, password, role, created)
            VALUES (?, ?, ?, ?)""",
            ("admin", admin_pass, "admin", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )

    print("Database ready")


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()
