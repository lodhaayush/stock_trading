"""Fetch fundamental data for tickers via yfinance."""

import logging
import time
from datetime import datetime, timezone

import yfinance as yf

from stock_trading import config

logger = logging.getLogger(__name__)

# Map yfinance info keys to DB column names
_FIELD_MAP = {
    "sector": "sector",
    "industry": "industry",
    "marketCap": "market_cap",
    "trailingPE": "trailing_pe",
    "forwardPE": "forward_pe",
    "dividendYield": "dividend_yield",
    "beta": "beta",
}


def fetch_ticker_fundamentals(ticker):
    """Fetch fundamental data for a single ticker.

    Returns a dict with DB column keys, or None on failure.
    """
    try:
        info = yf.Ticker(ticker).info
        return {db_col: info.get(yf_key) for yf_key, db_col in _FIELD_MAP.items()}
    except Exception:
        logger.warning("Failed to fetch fundamentals for %s", ticker)
        return None


def fetch_all_fundamentals(conn, limit=None):
    """Fetch fundamentals for all tickers in the database.

    Returns a summary dict with processed, updated, and failed counts.
    """
    rows = conn.execute("SELECT ticker FROM tickers").fetchall()
    tickers = [r["ticker"] for r in rows]
    if limit is not None:
        tickers = tickers[:limit]

    processed = 0
    updated = 0
    failed = 0

    for i, ticker in enumerate(tickers):
        data = fetch_ticker_fundamentals(ticker)
        processed += 1

        if data is not None:
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                "UPDATE tickers SET sector=?, industry=?, market_cap=?, "
                "trailing_pe=?, forward_pe=?, dividend_yield=?, beta=?, "
                "last_updated=? WHERE ticker=?",
                (
                    data["sector"],
                    data["industry"],
                    data["market_cap"],
                    data["trailing_pe"],
                    data["forward_pe"],
                    data["dividend_yield"],
                    data["beta"],
                    now,
                    ticker,
                ),
            )
            conn.commit()
            updated += 1
        else:
            failed += 1

        if (i + 1) % 100 == 0:
            logger.info("Progress: %d / %d tickers processed", i + 1, len(tickers))

        time.sleep(config.FUNDAMENTALS_DELAY_SECONDS)

    return {"processed": processed, "updated": updated, "failed": failed}
