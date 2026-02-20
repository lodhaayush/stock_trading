"""Incremental daily update orchestrator for the stock_trading pipeline."""

import logging
from datetime import datetime, timedelta

import yfinance as yf

from stock_trading import db, fundamentals, tickers as tickers_mod
from stock_trading.config import BATCH_SIZE
from stock_trading.downloader import download_batch, _yf_symbol

logger = logging.getLogger(__name__)


def _has_recent_split(ticker, since_date):
    """Check if a ticker has had a stock split since *since_date*.

    Uses yfinance's splits data rather than heuristic price detection.
    Returns True if a split occurred on or after since_date.
    """
    try:
        yf_sym = _yf_symbol(ticker)
        splits = yf.Ticker(yf_sym).splits
        if splits.empty:
            return False
        # Normalize index to tz-naive for comparison
        split_dates = splits.index.tz_localize(None) if splits.index.tz else splits.index
        return any(d.strftime("%Y-%m-%d") >= since_date for d in split_dates)
    except Exception as exc:
        logger.debug("Could not check splits for %s: %s", ticker, exc)
        return False


def run_daily_update(conn, include_fundamentals=False):
    """Orchestrate a daily incremental price update.

    Steps:
        1. Sync the ticker universe via ``tickers.sync_tickers``.
        2. Query all tickers from the DB and group by last price date.
        3. For tickers with no price data use ``period="max"``; for others
           determine start date from the last price date.
        4. Download in batches via ``download_batch``.
        5. Optionally fetch fundamentals.

    Returns
    -------
    dict
        ``{"tickers_synced": N, "prices_updated": N, "new_rows": N}``
    """
    # Step 1 -- sync ticker universe
    sync_result = tickers_mod.sync_tickers(conn)
    tickers_synced = sync_result["total"]
    logger.info("Ticker sync complete: %d tickers", tickers_synced)

    # Step 2 -- get all tickers and group by last price date
    rows = conn.execute("SELECT ticker FROM tickers ORDER BY ticker").fetchall()
    all_tickers = [r["ticker"] for r in rows]

    fresh_tickers = []  # no price data -> period="max"
    incremental_groups = {}  # start_date -> list of tickers

    for ticker in all_tickers:
        last_date = db.get_last_price_date(conn, ticker)
        if last_date is None:
            fresh_tickers.append(ticker)
        else:
            # Start from the day after last_price_date
            start = (datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            incremental_groups.setdefault(start, []).append(ticker)

    total_updated = 0
    total_rows = 0

    # Step 3 & 4 -- download fresh tickers in batches with period="max"
    if fresh_tickers:
        logger.info("Downloading %d tickers with no prior data (period=max)", len(fresh_tickers))
        for i in range(0, len(fresh_tickers), BATCH_SIZE):
            batch = fresh_tickers[i : i + BATCH_SIZE]
            stats = download_batch(conn, batch, period="max")
            total_updated += stats["downloaded"]
            total_rows += stats["rows"]

    # Step 4 -- download incremental updates grouped by start date
    split_tickers = []

    for start_date, ticker_list in incremental_groups.items():
        # Check for splits before downloading incremental data
        for ticker in ticker_list:
            if _has_recent_split(ticker, start_date):
                logger.info(
                    "Split detected for %s since %s, scheduling full re-download",
                    ticker, start_date,
                )
                split_tickers.append(ticker)

        # Remove split tickers from incremental list
        batch_tickers = [t for t in ticker_list if t not in split_tickers]
        if not batch_tickers:
            continue

        logger.info("Incremental update for %d tickers from %s", len(batch_tickers), start_date)
        for i in range(0, len(batch_tickers), BATCH_SIZE):
            batch = batch_tickers[i : i + BATCH_SIZE]
            # Use yf.download with start= for incremental updates
            try:
                if len(batch) == 1:
                    data = yf.download(batch[0], start=start_date, threads=True)
                else:
                    data = yf.download(batch, start=start_date, group_by="ticker", threads=True)

                if data is not None and not data.empty:
                    # Process through download_batch-like logic by storing directly
                    # We reuse the batch processing from download_batch by
                    # calling it; the data has already been fetched above so we
                    # patch through.  Instead, process each ticker manually.
                    for ticker in batch:
                        try:
                            if len(batch) == 1:
                                ticker_df = data.copy()
                            else:
                                if ticker not in data.columns.get_level_values(0):
                                    continue
                                ticker_df = data[ticker].copy()

                            ticker_df.columns = [c.lower() for c in ticker_df.columns]
                            if "close" in ticker_df.columns:
                                ticker_df = ticker_df.dropna(subset=["close"])
                            if ticker_df.empty:
                                continue

                            dates = ticker_df.index
                            if hasattr(dates, "tz") and dates.tz is not None:
                                dates = dates.tz_localize(None)

                            price_rows = []
                            for idx, dt in enumerate(dates):
                                date_str = dt.strftime("%Y-%m-%d")
                                row = ticker_df.iloc[idx]
                                price_rows.append((
                                    ticker,
                                    date_str,
                                    float(row.get("open", 0) or 0),
                                    float(row.get("high", 0) or 0),
                                    float(row.get("low", 0) or 0),
                                    float(row.get("close", 0) or 0),
                                    int(row.get("volume", 0) or 0),
                                    float(row.get("adj close", row.get("close", 0)) or 0),
                                ))

                            db.upsert_prices(conn, price_rows)
                            last_d = price_rows[-1][1] if price_rows else None
                            db.update_download_log(conn, ticker, "complete", last_price_date=last_d)
                            total_updated += 1
                            total_rows += len(price_rows)
                        except Exception as exc:
                            logger.error("Error processing ticker %s: %s", ticker, exc)
                else:
                    logger.info("No new data for batch starting %s", start_date)
            except Exception as exc:
                logger.error("yf.download failed for incremental batch: %s", exc)

    # Re-download full history for tickers with detected splits
    if split_tickers:
        logger.info(
            "Re-downloading %d tickers with detected splits: %s",
            len(split_tickers), split_tickers[:10],
        )
        for i in range(0, len(split_tickers), BATCH_SIZE):
            batch = split_tickers[i : i + BATCH_SIZE]
            stats = download_batch(conn, batch, period="max")
            total_updated += stats["downloaded"]
            total_rows += stats["rows"]

    # Step 5 -- optionally fetch fundamentals
    if include_fundamentals:
        fundamentals.fetch_all_fundamentals(conn)

    summary = {
        "tickers_synced": tickers_synced,
        "prices_updated": total_updated,
        "new_rows": total_rows,
    }
    logger.info("Daily update complete: %s", summary)
    return summary


def detect_delistings(conn):
    """Compare current exchange listings with tickers stored in the DB.

    Returns
    -------
    list[str]
        Tickers present in the DB but absent from current exchange listings.
    """
    current = tickers_mod.fetch_all_tickers()
    current_symbols = {t["ticker"] for t in current}

    db_rows = conn.execute("SELECT ticker FROM tickers").fetchall()
    db_symbols = {r["ticker"] for r in db_rows}

    delisted = sorted(db_symbols - current_symbols)
    if delisted:
        logger.info("Detected %d potentially delisted tickers: %s", len(delisted), delisted)
    return delisted
