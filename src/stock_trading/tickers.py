"""Ticker fetching and syncing for the stock_trading pipeline."""

import logging
import re
from io import StringIO

import requests

from stock_trading import db
from stock_trading.config import (
    NASDAQ_LISTED_URL,
    OTHER_LISTED_URL,
    SEC_EDGAR_HEADERS,
    SEC_EDGAR_URL,
)

logger = logging.getLogger(__name__)


def fetch_nasdaq_tickers():
    """Fetch tickers from NASDAQ FTP (nasdaqlisted.txt and otherlisted.txt).

    Returns a list of dicts with keys: ticker, name, exchange.
    """
    tickers = []

    # nasdaqlisted.txt
    resp = requests.get(NASDAQ_LISTED_URL, timeout=30)
    resp.raise_for_status()
    lines = resp.text.strip().splitlines()
    header = lines[0].split("|")
    sym_idx = header.index("Symbol")
    name_idx = header.index("Security Name")
    for line in lines[1:]:
        if line.startswith("File Creation Time"):
            continue
        cols = line.split("|")
        if len(cols) <= max(sym_idx, name_idx):
            continue
        tickers.append({
            "ticker": cols[sym_idx].strip(),
            "name": cols[name_idx].strip(),
            "exchange": "NASDAQ",
        })

    # otherlisted.txt
    resp = requests.get(OTHER_LISTED_URL, timeout=30)
    resp.raise_for_status()
    lines = resp.text.strip().splitlines()
    header = lines[0].split("|")
    sym_idx = header.index("ACT Symbol")
    name_idx = header.index("Security Name")
    exch_idx = header.index("Exchange")
    for line in lines[1:]:
        if line.startswith("File Creation Time"):
            continue
        cols = line.split("|")
        if len(cols) <= max(sym_idx, name_idx, exch_idx):
            continue
        tickers.append({
            "ticker": cols[sym_idx].strip(),
            "name": cols[name_idx].strip(),
            "exchange": cols[exch_idx].strip(),
        })

    return tickers


def fetch_sec_edgar_tickers():
    """Fetch tickers from SEC EDGAR as a fallback.

    Returns a list of dicts with keys: ticker, name, exchange.
    """
    resp = requests.get(SEC_EDGAR_URL, headers=SEC_EDGAR_HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    tickers = []
    for entry in data.values():
        tickers.append({
            "ticker": entry["ticker"],
            "name": entry["title"],
            "exchange": "UNKNOWN",
        })
    return tickers


_SPECIAL_CHARS = re.compile(r"[.\$\-]")
_TEST_WORD = re.compile(r"\btest\b", re.IGNORECASE)
_EXCLUDE_KEYWORDS = {"warrant", "unit", "right", "preferred"}


def filter_tickers(tickers):
    """Filter out non-common-stock tickers.

    Removes symbols containing '.', '$', or '-', symbols longer than 5 chars,
    and names containing warrant/unit/right/preferred (case-insensitive)
    or 'test' as a standalone word.
    """
    result = []
    for t in tickers:
        symbol = t["ticker"]
        name = t.get("name", "")

        if _SPECIAL_CHARS.search(symbol):
            continue
        if len(symbol) > 5:
            continue

        name_lower = name.lower()
        if any(kw in name_lower for kw in _EXCLUDE_KEYWORDS):
            continue
        if _TEST_WORD.search(name):
            continue

        result.append(t)
    return result


def fetch_all_tickers():
    """Fetch tickers from NASDAQ, falling back to SEC EDGAR on failure.

    Applies filter_tickers to the result.
    """
    try:
        tickers = fetch_nasdaq_tickers()
        logger.info("Fetched %d tickers from NASDAQ", len(tickers))
    except Exception:
        logger.warning("NASDAQ fetch failed, falling back to SEC EDGAR")
        tickers = fetch_sec_edgar_tickers()
        logger.info("Fetched %d tickers from SEC EDGAR", len(tickers))
    return filter_tickers(tickers)


def sync_tickers(conn):
    """Sync fetched tickers into the database.

    Returns a dict with keys: total, new, updated.
    """
    tickers = fetch_all_tickers()

    # Get existing tickers from DB
    existing = {
        row["ticker"]
        for row in conn.execute("SELECT ticker FROM tickers").fetchall()
    }

    new_count = 0
    updated_count = 0
    for t in tickers:
        if t["ticker"] in existing:
            updated_count += 1
        else:
            new_count += 1
        db.upsert_ticker(conn, t["ticker"], name=t["name"], exchange=t["exchange"])

    logger.info(
        "Synced %d tickers: %d new, %d updated",
        len(tickers),
        new_count,
        updated_count,
    )
    return {"total": len(tickers), "new": new_count, "updated": updated_count}
