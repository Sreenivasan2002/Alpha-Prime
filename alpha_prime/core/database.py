"""
Database Layer for Alpha-Prime Trading System
SQLite-based persistence for trades, signals, portfolio, and agent logs.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
from loguru import logger

from alpha_prime.core.config import settings

DB_PATH = str(settings.db_path)


def get_connection():
    """Get a database connection with WAL mode for concurrency"""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def db_session():
    """Context manager for database sessions"""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        conn.close()


def init_database():
    """Initialize all database tables"""
    with db_session() as conn:
        cursor = conn.cursor()

        # Portfolio state
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS portfolio (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                capital REAL,
                invested REAL,
                available REAL,
                total_value REAL,
                daily_pnl REAL,
                total_pnl REAL,
                positions_json TEXT
            )
        """)

        # Trade history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                price REAL NOT NULL,
                order_type TEXT DEFAULT 'MARKET',
                stop_loss REAL,
                target REAL,
                status TEXT DEFAULT 'PENDING',
                order_id TEXT,
                rationale TEXT,
                agent_name TEXT,
                pnl REAL DEFAULT 0.0
            )
        """)

        # Agent signals and decisions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                symbol TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                strength REAL,
                agent_name TEXT,
                reasoning TEXT,
                indicators_json TEXT,
                action_taken TEXT
            )
        """)

        # Agent activity logs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                agent_name TEXT NOT NULL,
                log_type TEXT NOT NULL,
                message TEXT,
                details_json TEXT
            )
        """)

        # Watchlist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS watchlist (
                symbol TEXT PRIMARY KEY,
                added_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                sector TEXT,
                notes TEXT
            )
        """)

        # Market data cache
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_cache (
                cache_key TEXT PRIMARY KEY,
                data_json TEXT,
                cached_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # System state
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_state (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        logger.info("Database initialized successfully")
        _migrate_trades_trading_mode(conn)


def _migrate_trades_trading_mode(conn):
    """Add trading_mode column to trades if missing (paper vs live separation)."""
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(trades)")
    columns = [row[1] for row in cursor.fetchall()]
    if "trading_mode" not in columns:
        cursor.execute("ALTER TABLE trades ADD COLUMN trading_mode TEXT DEFAULT 'paper'")
        cursor.execute("UPDATE trades SET trading_mode = 'paper' WHERE trading_mode IS NULL")
        conn.commit()
        logger.info("Migrated trades table: added trading_mode column")


# ---- Trade Operations ----

def record_trade(symbol: str, action: str, quantity: int, price: float,
                 order_type: str = "MARKET", stop_loss: float = None,
                 target: float = None, rationale: str = "", agent_name: str = "",
                 order_id: str = "", status: str = "EXECUTED",
                 trading_mode: str = None) -> int:
    """Record a trade in the database. trading_mode should be 'paper' or 'live'."""
    if trading_mode is None:
        trading_mode = getattr(settings.trading, "mode", "paper")
    with db_session() as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(trades)")
        columns = [row[1] for row in cursor.fetchall()]
        if "trading_mode" in columns:
            cursor.execute("""
                INSERT INTO trades (symbol, action, quantity, price, order_type,
                                  stop_loss, target, rationale, agent_name, order_id, status, trading_mode)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (symbol, action, quantity, price, order_type, stop_loss, target,
                  rationale, agent_name, order_id, status, trading_mode))
        else:
            cursor.execute("""
                INSERT INTO trades (symbol, action, quantity, price, order_type,
                                  stop_loss, target, rationale, agent_name, order_id, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (symbol, action, quantity, price, order_type, stop_loss, target,
                  rationale, agent_name, order_id, status))
        return cursor.lastrowid


def get_trades(limit: int = 50, symbol: str = None, trading_mode: str = None) -> list:
    """Get recent trades. If trading_mode is 'paper' or 'live', only that mode is returned."""
    with db_session() as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(trades)")
        columns = [row[1] for row in cursor.fetchall()]
        has_mode = "trading_mode" in columns
        if trading_mode and has_mode:
            if symbol:
                cursor.execute(
                    "SELECT * FROM trades WHERE symbol = ? AND trading_mode = ? ORDER BY timestamp DESC LIMIT ?",
                    (symbol, trading_mode, limit))
            else:
                cursor.execute(
                    "SELECT * FROM trades WHERE trading_mode = ? ORDER BY timestamp DESC LIMIT ?",
                    (trading_mode, limit))
        else:
            if symbol:
                cursor.execute(
                    "SELECT * FROM trades WHERE symbol = ? ORDER BY timestamp DESC LIMIT ?",
                    (symbol, limit))
            else:
                cursor.execute(
                    "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?", (limit,))
        return [dict(row) for row in cursor.fetchall()]


def get_open_positions(trading_mode: str = None) -> list:
    """Get currently open positions from trade history. If trading_mode is 'paper' or 'live', only that mode."""
    with db_session() as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(trades)")
        columns = [row[1] for row in cursor.fetchall()]
        has_mode = "trading_mode" in columns
        if trading_mode and has_mode:
            cursor.execute("""
                SELECT symbol,
                       SUM(CASE WHEN action = 'BUY' THEN quantity ELSE -quantity END) as net_qty,
                       AVG(CASE WHEN action = 'BUY' THEN price END) as avg_buy_price
                FROM trades
                WHERE status = 'EXECUTED' AND trading_mode = ?
                GROUP BY symbol
                HAVING net_qty > 0
            """, (trading_mode,))
        else:
            cursor.execute("""
                SELECT symbol,
                       SUM(CASE WHEN action = 'BUY' THEN quantity ELSE -quantity END) as net_qty,
                       AVG(CASE WHEN action = 'BUY' THEN price END) as avg_buy_price
                FROM trades
                WHERE status = 'EXECUTED'
                GROUP BY symbol
                HAVING net_qty > 0
            """)
        return [dict(row) for row in cursor.fetchall()]


# ---- Signal Operations ----

def record_signal(symbol: str, signal_type: str, strength: float,
                  agent_name: str, reasoning: str, indicators: dict = None,
                  action_taken: str = None):
    """Record an agent signal"""
    with db_session() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO signals (symbol, signal_type, strength, agent_name,
                               reasoning, indicators_json, action_taken)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (symbol, signal_type, strength, agent_name, reasoning,
              json.dumps(indicators) if indicators else None, action_taken))


def get_signals(limit: int = 50, symbol: str = None) -> list:
    """Get recent signals"""
    with db_session() as conn:
        cursor = conn.cursor()
        if symbol:
            cursor.execute(
                "SELECT * FROM signals WHERE symbol = ? ORDER BY timestamp DESC LIMIT ?",
                (symbol, limit))
        else:
            cursor.execute(
                "SELECT * FROM signals ORDER BY timestamp DESC LIMIT ?", (limit,))
        return [dict(row) for row in cursor.fetchall()]


# ---- Agent Log Operations ----

def log_agent_activity(agent_name: str, log_type: str, message: str,
                       details: dict = None):
    """Log agent activity"""
    with db_session() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO agent_logs (agent_name, log_type, message, details_json)
            VALUES (?, ?, ?, ?)
        """, (agent_name, log_type, message,
              json.dumps(details) if details else None))


def get_agent_logs(agent_name: str = None, limit: int = 100) -> list:
    """Get agent logs"""
    with db_session() as conn:
        cursor = conn.cursor()
        if agent_name:
            cursor.execute(
                "SELECT * FROM agent_logs WHERE agent_name = ? ORDER BY timestamp DESC LIMIT ?",
                (agent_name, limit))
        else:
            cursor.execute(
                "SELECT * FROM agent_logs ORDER BY timestamp DESC LIMIT ?", (limit,))
        return [dict(row) for row in cursor.fetchall()]


# ---- Portfolio Operations ----

def save_portfolio_snapshot(capital: float, invested: float, available: float,
                           total_value: float, daily_pnl: float, total_pnl: float,
                           positions: dict):
    """Save a portfolio snapshot"""
    with db_session() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO portfolio (capital, invested, available, total_value,
                                  daily_pnl, total_pnl, positions_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (capital, invested, available, total_value, daily_pnl, total_pnl,
              json.dumps(positions)))


def get_portfolio_history(limit: int = 100) -> list:
    """Get portfolio value history"""
    with db_session() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM portfolio ORDER BY timestamp DESC LIMIT ?", (limit,))
        return [dict(row) for row in cursor.fetchall()]


# ---- Watchlist Operations ----

def add_to_watchlist(symbol: str, sector: str = "", notes: str = ""):
    """Add a symbol to watchlist"""
    with db_session() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO watchlist (symbol, sector, notes)
            VALUES (?, ?, ?)
        """, (symbol, sector, notes))


def remove_from_watchlist(symbol: str):
    """Remove a symbol from watchlist"""
    with db_session() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM watchlist WHERE symbol = ?", (symbol,))


def get_watchlist() -> list:
    """Get all watchlist symbols"""
    with db_session() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM watchlist ORDER BY added_at DESC")
        return [dict(row) for row in cursor.fetchall()]


# ---- System State Operations ----

def set_system_state(key: str, value: str):
    """Set a system state value"""
    with db_session() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO system_state (key, value, updated_at)
            VALUES (?, ?, datetime('now'))
        """, (key, value))


def get_system_state(key: str, default: str = None) -> str:
    """Get a system state value"""
    with db_session() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM system_state WHERE key = ?", (key,))
        row = cursor.fetchone()
        return row["value"] if row else default


# ---- Cache Operations ----

def cache_set(key: str, data: dict):
    """Set cached data"""
    with db_session() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO market_cache (cache_key, data_json, cached_at)
            VALUES (?, ?, datetime('now'))
        """, (key, json.dumps(data)))


def cache_get(key: str, max_age_minutes: int = 5) -> dict:
    """Get cached data if not expired"""
    with db_session() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT data_json FROM market_cache
            WHERE cache_key = ?
            AND cached_at > datetime('now', ?)
        """, (key, f"-{max_age_minutes} minutes"))
        row = cursor.fetchone()
        return json.loads(row["data_json"]) if row else None


# Initialize on import
init_database()
