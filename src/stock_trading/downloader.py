"""Download historical price data via yfinance."""

import logging
import time

import yfinance as yf

from stock_trading import db
from stock_trading.config import (
    BATCH_DELAY_SECONDS,
    BATCH_SIZE,
    MAX_RETRIES,
    RETRY_BASE_DELAY_SECONDS,
)

logger = logging.getLogger(__name__)


def download_batch(conn, tickers, period="max"):
    """Download price data for a batch of tickers and store in the database.

    Parameters
    ----------
    conn : sqlite3.Connection
    tickers : list[str]
    period : str
        yfinance period string (default ``"max"``).

    Returns
    -------
    dict
        ``{"downloaded": N, "failed": N, "rows": N}``
    """
    stats = {"downloaded": 0, "failed": 0, "rows": 0}

    if not tickers:
        return stats

    try:
        if len(tickers) == 1:
            data = yf.download(tickers[0], period=period, threads=True)
        else:
            data = yf.download(tickers, period=period, group_by="ticker", threads=True)
    except Exception as exc:
        logger.error("yf.download failed for batch: %s", exc)
        for t in tickers:
            db.update_download_log(conn, t, "failed", error_message=str(exc))
            stats["failed"] += 1
        return stats

    if data is None or data.empty:
        for t in tickers:
            db.update_download_log(conn, t, "no_data")
        stats["failed"] += len(tickers)
        return stats

    for ticker in tickers:
        try:
            if len(tickers) == 1:
                ticker_df = data.copy()
            else:
                if ticker not in data.columns.get_level_values(0):
                    db.update_download_log(conn, ticker, "no_data")
                    stats["failed"] += 1
                    continue
                ticker_df = data[ticker].copy()

            # Normalize column names to lowercase
            ticker_df.columns = [c.lower() for c in ticker_df.columns]

            # Drop rows where close is NaN (no data)
            if "close" in ticker_df.columns:
                ticker_df = ticker_df.dropna(subset=["close"])

            if ticker_df.empty:
                db.update_download_log(conn, ticker, "no_data")
                stats["failed"] += 1
                continue

            # Strip timezone from index and convert to YYYY-MM-DD strings
            dates = ticker_df.index
            if hasattr(dates, "tz") and dates.tz is not None:
                dates = dates.tz_localize(None)

            rows = []
            for i, dt in enumerate(dates):
                date_str = dt.strftime("%Y-%m-%d")
                row = ticker_df.iloc[i]
                rows.append((
                    ticker,
                    date_str,
                    float(row.get("open", 0) or 0),
                    float(row.get("high", 0) or 0),
                    float(row.get("low", 0) or 0),
                    float(row.get("close", 0) or 0),
                    int(row.get("volume", 0) or 0),
                    float(row.get("adj close", 0) or 0),
                ))

            db.upsert_prices(conn, rows)
            last_date = rows[-1][1] if rows else None
            db.update_download_log(conn, ticker, "complete", last_price_date=last_date)
            stats["downloaded"] += 1
            stats["rows"] += len(rows)
        except Exception as exc:
            logger.error("Error processing ticker %s: %s", ticker, exc)
            db.update_download_log(conn, ticker, "failed", error_message=str(exc))
            stats["failed"] += 1

    return stats


def download_all(conn, limit=None, tickers_list=None, resume=True):
    """Orchestrate downloading price data for all tickers.

    Parameters
    ----------
    conn : sqlite3.Connection
    limit : int | None
        Maximum number of tickers to process.
    tickers_list : list[str] | None
        Explicit list of tickers. If ``None``, all tickers from DB are used.
    resume : bool
        Skip tickers already marked ``complete`` in download_log.

    Returns
    -------
    dict
        Overall stats: ``{"downloaded": N, "failed": N, "rows": N}``
    """
    if tickers_list is not None:
        all_tickers = list(tickers_list)
    else:
        rows = conn.execute("SELECT ticker FROM tickers ORDER BY ticker").fetchall()
        all_tickers = [r["ticker"] for r in rows]

    if resume:
        completed = conn.execute(
            "SELECT ticker FROM download_log WHERE status = 'complete'"
        ).fetchall()
        completed_set = {r["ticker"] for r in completed}
        all_tickers = [t for t in all_tickers if t not in completed_set]

    if limit is not None:
        all_tickers = all_tickers[:limit]

    total_stats = {"downloaded": 0, "failed": 0, "rows": 0}

    # Split into batches
    batches = [
        all_tickers[i : i + BATCH_SIZE]
        for i in range(0, len(all_tickers), BATCH_SIZE)
    ]
    total_batches = len(batches)

    for batch_num, batch in enumerate(batches, start=1):
        logger.info(
            "Batch %d/%d — %d tickers (cumulative: %d downloaded, %d rows)",
            batch_num,
            total_batches,
            len(batch),
            total_stats["downloaded"],
            total_stats["rows"],
        )

        for attempt in range(MAX_RETRIES + 1):
            try:
                batch_stats = download_batch(conn, batch)
                total_stats["downloaded"] += batch_stats["downloaded"]
                total_stats["failed"] += batch_stats["failed"]
                total_stats["rows"] += batch_stats["rows"]
                break
            except Exception as exc:
                if attempt < MAX_RETRIES:
                    delay = RETRY_BASE_DELAY_SECONDS * (2 ** attempt)
                    logger.warning(
                        "Batch %d failed (attempt %d/%d), retrying in %.1fs: %s",
                        batch_num,
                        attempt + 1,
                        MAX_RETRIES,
                        delay,
                        exc,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "Batch %d failed after %d retries: %s",
                        batch_num,
                        MAX_RETRIES,
                        exc,
                    )
                    for t in batch:
                        db.update_download_log(conn, t, "failed", error_message=str(exc))
                        total_stats["failed"] += 1

        if batch_num < total_batches:
            time.sleep(BATCH_DELAY_SECONDS)

    logger.info(
        "Download complete — %d downloaded, %d failed, %d rows",
        total_stats["downloaded"],
        total_stats["failed"],
        total_stats["rows"],
    )
    return total_stats


def retry_failed(conn):
    """Re-download tickers whose status is ``failed`` in download_log.

    Resets their retry_count then delegates to :func:`download_all`.
    """
    failed = conn.execute(
        "SELECT ticker FROM download_log WHERE status = 'failed'"
    ).fetchall()
    failed_tickers = [r["ticker"] for r in failed]

    if not failed_tickers:
        logger.info("No failed tickers to retry.")
        return {"downloaded": 0, "failed": 0, "rows": 0}

    # Reset retry_count for failed tickers
    with conn:
        conn.executemany(
            "UPDATE download_log SET retry_count = 0 WHERE ticker = ?",
            [(t,) for t in failed_tickers],
        )

    return download_all(conn, tickers_list=failed_tickers, resume=False)


def get_download_status(conn):
    """Return a summary of download progress.

    Returns
    -------
    dict
        ``{"pending": N, "complete": N, "failed": N, "no_data": N,
           "total_tickers": N, "total_rows": N}``
    """
    total_tickers = conn.execute("SELECT COUNT(*) AS cnt FROM tickers").fetchone()["cnt"]
    total_rows = conn.execute("SELECT COUNT(*) AS cnt FROM daily_prices").fetchone()["cnt"]

    status_rows = conn.execute(
        "SELECT status, COUNT(*) AS cnt FROM download_log GROUP BY status"
    ).fetchall()
    status_counts = {r["status"]: r["cnt"] for r in status_rows}

    return {
        "pending": status_counts.get("pending", 0),
        "complete": status_counts.get("complete", 0),
        "failed": status_counts.get("failed", 0),
        "no_data": status_counts.get("no_data", 0),
        "total_tickers": total_tickers,
        "total_rows": total_rows,
    }
