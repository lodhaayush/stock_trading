"""Database helpers for the stock_trading pipeline."""

import sqlite3
from datetime import datetime, timezone

from stock_trading.config import DB_PATH


def get_connection(db_path=None):
    """Return a configured SQLite connection."""
    if db_path is None:
        db_path = DB_PATH
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA cache_size = -65536")
    conn.execute("PRAGMA mmap_size = 268435456")
    return conn


def init_db(conn):
    """Create the schema tables if they do not exist."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS tickers (
            ticker       TEXT PRIMARY KEY,
            name         TEXT,
            exchange     TEXT,
            sector       TEXT,
            industry     TEXT,
            market_cap   REAL,
            trailing_pe  REAL,
            forward_pe   REAL,
            dividend_yield REAL,
            beta         REAL,
            last_updated TEXT
        );

        CREATE TABLE IF NOT EXISTS daily_prices (
            ticker    TEXT,
            date      TEXT,
            open      REAL,
            high      REAL,
            low       REAL,
            close     REAL,
            volume    INTEGER,
            adj_close REAL,
            PRIMARY KEY (ticker, date)
        ) WITHOUT ROWID;

        CREATE INDEX IF NOT EXISTS idx_daily_prices_date
            ON daily_prices (date);

        CREATE TABLE IF NOT EXISTS download_log (
            ticker          TEXT PRIMARY KEY,
            last_download   TEXT,
            last_price_date TEXT,
            status          TEXT DEFAULT 'pending',
            error_message   TEXT,
            retry_count     INTEGER DEFAULT 0
        );
        """
    )


def upsert_prices(conn, rows):
    """Bulk insert or replace daily price rows.

    Parameters
    ----------
    conn : sqlite3.Connection
    rows : list[tuple]
        Each tuple: (ticker, date, open, high, low, close, volume, adj_close)
    """
    with conn:
        conn.executemany(
            "INSERT OR REPLACE INTO daily_prices "
            "(ticker, date, open, high, low, close, volume, adj_close) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )


def get_last_price_date(conn, ticker):
    """Return the most recent date string for *ticker*, or None."""
    row = conn.execute(
        "SELECT MAX(date) AS max_date FROM daily_prices WHERE ticker = ?",
        (ticker,),
    ).fetchone()
    if row is None:
        return None
    return row["max_date"]


def upsert_ticker(conn, ticker, name=None, exchange=None):
    """Insert or replace a row in the tickers table."""
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO tickers (ticker, name, exchange) "
            "VALUES (?, ?, ?)",
            (ticker, name, exchange),
        )


def update_download_log(conn, ticker, status, last_price_date=None, error_message=None):
    """Insert or replace a download_log entry with the current timestamp."""
    now = datetime.now(timezone.utc).isoformat()
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO download_log "
            "(ticker, last_download, last_price_date, status, error_message) "
            "VALUES (?, ?, ?, ?, ?)",
            (ticker, now, last_price_date, status, error_message),
        )


def query_prices(conn, ticker, start_date=None, end_date=None):
    """Return daily_prices rows for *ticker*, optionally filtered by date range."""
    sql = "SELECT * FROM daily_prices WHERE ticker = ?"
    params = [ticker]
    if start_date is not None:
        sql += " AND date >= ?"
        params.append(start_date)
    if end_date is not None:
        sql += " AND date <= ?"
        params.append(end_date)
    sql += " ORDER BY date"
    return conn.execute(sql, params).fetchall()


def query_all_fundamentals(conn):
    """Return all tickers with their fundamental data."""
    return conn.execute(
        "SELECT ticker, name, sector, industry, market_cap, "
        "trailing_pe, forward_pe, dividend_yield, beta "
        "FROM tickers"
    ).fetchall()


def query_recent_prices(conn, lookback_days=250):
    """Return recent price data for all tickers in one query.

    Uses the idx_daily_prices_date index for efficient date filtering.
    Results are ordered by (ticker, date) for groupby partitioning.
    """
    from datetime import timedelta

    cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    return conn.execute(
        "SELECT ticker, date, open, high, low, close, volume, adj_close "
        "FROM daily_prices WHERE date >= ? ORDER BY ticker, date",
        (cutoff,),
    ).fetchall()
