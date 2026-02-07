"""Tests for stock_trading.downloader module."""

from unittest.mock import patch

import pandas as pd
import pytest

from stock_trading.db import init_db, update_download_log, upsert_prices, upsert_ticker
from stock_trading.downloader import (
    download_all,
    download_batch,
    get_download_status,
    retry_failed,
)


def _make_multi_ticker_df(tickers, dates):
    """Build a MultiIndex DataFrame mimicking yf.download with group_by='ticker'.

    yfinance with group_by='ticker' produces columns with level 0 = ticker symbol
    and level 1 = price field.
    """
    fields = ["Open", "High", "Low", "Close", "Volume", "Adj Close"]
    index = pd.DatetimeIndex(dates)
    arrays = []
    for ticker in tickers:
        for field in fields:
            arrays.append((ticker, field))
    columns = pd.MultiIndex.from_tuples(arrays, names=["Ticker", "Price"])
    data = {}
    for ticker in tickers:
        for field in fields:
            if field == "Volume":
                data[(ticker, field)] = [1000000] * len(dates)
            else:
                data[(ticker, field)] = [100.0 + i for i in range(len(dates))]
    df = pd.DataFrame(data, index=index, columns=columns)
    return df


def _make_single_ticker_df(dates):
    """Build a simple DataFrame mimicking yf.download for a single ticker."""
    index = pd.DatetimeIndex(dates)
    data = {
        "Open": [100.0 + i for i in range(len(dates))],
        "High": [101.0 + i for i in range(len(dates))],
        "Low": [99.0 + i for i in range(len(dates))],
        "Close": [100.5 + i for i in range(len(dates))],
        "Volume": [1000000] * len(dates),
        "Adj Close": [100.5 + i for i in range(len(dates))],
    }
    return pd.DataFrame(data, index=index)


class TestDownloadBatchMultiTicker:
    @patch("stock_trading.downloader.yf.download")
    def test_multi_ticker_download(self, mock_download, in_memory_db):
        init_db(in_memory_db)
        tickers = ["AAPL", "GOOG"]
        dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
        mock_download.return_value = _make_multi_ticker_df(tickers, dates)

        stats = download_batch(in_memory_db, tickers)

        assert stats["downloaded"] == 2
        assert stats["failed"] == 0
        assert stats["rows"] == 6  # 3 rows * 2 tickers

        # Verify data in database
        count = in_memory_db.execute("SELECT COUNT(*) AS cnt FROM daily_prices").fetchone()
        assert count["cnt"] == 6

        # Verify download_log
        for t in tickers:
            row = in_memory_db.execute(
                "SELECT * FROM download_log WHERE ticker = ?", (t,)
            ).fetchone()
            assert row["status"] == "complete"

    @patch("stock_trading.downloader.yf.download")
    def test_multi_ticker_missing_one(self, mock_download, in_memory_db):
        """When a ticker is requested but not in the download result."""
        init_db(in_memory_db)
        dates = ["2024-01-01"]
        # Only AAPL is returned, GOOG is missing from result
        mock_download.return_value = _make_multi_ticker_df(["AAPL"], dates)

        stats = download_batch(in_memory_db, ["AAPL", "GOOG"])

        assert stats["downloaded"] == 1
        assert stats["failed"] == 1


class TestDownloadBatchSingleTicker:
    @patch("stock_trading.downloader.yf.download")
    def test_single_ticker_download(self, mock_download, in_memory_db):
        init_db(in_memory_db)
        dates = ["2024-01-01", "2024-01-02"]
        mock_download.return_value = _make_single_ticker_df(dates)

        stats = download_batch(in_memory_db, ["AAPL"])

        assert stats["downloaded"] == 1
        assert stats["failed"] == 0
        assert stats["rows"] == 2

        count = in_memory_db.execute("SELECT COUNT(*) AS cnt FROM daily_prices").fetchone()
        assert count["cnt"] == 2

        row = in_memory_db.execute(
            "SELECT * FROM download_log WHERE ticker = 'AAPL'"
        ).fetchone()
        assert row["status"] == "complete"


class TestDownloadBatchErrors:
    @patch("stock_trading.downloader.yf.download")
    def test_empty_dataframe(self, mock_download, in_memory_db):
        init_db(in_memory_db)
        mock_download.return_value = pd.DataFrame()

        stats = download_batch(in_memory_db, ["AAPL"])

        assert stats["downloaded"] == 0
        assert stats["failed"] == 1

    @patch("stock_trading.downloader.yf.download")
    def test_exception_from_yf(self, mock_download, in_memory_db):
        init_db(in_memory_db)
        mock_download.side_effect = Exception("Network error")

        stats = download_batch(in_memory_db, ["AAPL", "GOOG"])

        assert stats["downloaded"] == 0
        assert stats["failed"] == 2

        for t in ["AAPL", "GOOG"]:
            row = in_memory_db.execute(
                "SELECT * FROM download_log WHERE ticker = ?", (t,)
            ).fetchone()
            assert row["status"] == "failed"

    def test_empty_ticker_list(self, in_memory_db):
        init_db(in_memory_db)
        stats = download_batch(in_memory_db, [])
        assert stats == {"downloaded": 0, "failed": 0, "rows": 0}


class TestDownloadAll:
    @patch("stock_trading.downloader.time.sleep")
    @patch("stock_trading.downloader.download_batch")
    def test_resume_skips_completed(self, mock_batch, mock_sleep, in_memory_db):
        init_db(in_memory_db)
        # Insert tickers
        for t in ["AAPL", "GOOG", "MSFT"]:
            upsert_ticker(in_memory_db, t)
        # Mark AAPL as complete
        update_download_log(in_memory_db, "AAPL", "complete")

        mock_batch.return_value = {"downloaded": 2, "failed": 0, "rows": 100}

        stats = download_all(in_memory_db, resume=True)

        # Should only have called with GOOG and MSFT (AAPL skipped)
        assert mock_batch.call_count == 1
        called_tickers = mock_batch.call_args[0][1]
        assert "AAPL" not in called_tickers
        assert set(called_tickers) == {"GOOG", "MSFT"}

    @patch("stock_trading.downloader.time.sleep")
    @patch("stock_trading.downloader.download_batch")
    def test_no_resume_includes_all(self, mock_batch, mock_sleep, in_memory_db):
        init_db(in_memory_db)
        for t in ["AAPL", "GOOG"]:
            upsert_ticker(in_memory_db, t)
        update_download_log(in_memory_db, "AAPL", "complete")

        mock_batch.return_value = {"downloaded": 2, "failed": 0, "rows": 100}

        stats = download_all(in_memory_db, resume=False)

        called_tickers = mock_batch.call_args[0][1]
        assert "AAPL" in called_tickers

    @patch("stock_trading.downloader.time.sleep")
    @patch("stock_trading.downloader.download_batch")
    def test_limit_restricts_tickers(self, mock_batch, mock_sleep, in_memory_db):
        init_db(in_memory_db)
        for t in ["AAPL", "GOOG", "MSFT", "TSLA"]:
            upsert_ticker(in_memory_db, t)

        mock_batch.return_value = {"downloaded": 2, "failed": 0, "rows": 50}

        stats = download_all(in_memory_db, limit=2)

        called_tickers = mock_batch.call_args[0][1]
        assert len(called_tickers) == 2

    @patch("stock_trading.downloader.time.sleep")
    @patch("stock_trading.downloader.download_batch")
    def test_explicit_tickers_list(self, mock_batch, mock_sleep, in_memory_db):
        init_db(in_memory_db)

        mock_batch.return_value = {"downloaded": 1, "failed": 0, "rows": 10}

        stats = download_all(in_memory_db, tickers_list=["AAPL"])

        called_tickers = mock_batch.call_args[0][1]
        assert called_tickers == ["AAPL"]


class TestGetDownloadStatus:
    def test_returns_correct_counts(self, in_memory_db):
        init_db(in_memory_db)

        # Add tickers
        for t in ["AAPL", "GOOG", "MSFT", "TSLA"]:
            upsert_ticker(in_memory_db, t)

        # Add some prices
        upsert_prices(in_memory_db, [
            ("AAPL", "2024-01-01", 100.0, 105.0, 99.0, 104.0, 1000000, 104.0),
            ("AAPL", "2024-01-02", 104.0, 106.0, 103.0, 105.0, 1100000, 105.0),
            ("GOOG", "2024-01-01", 140.0, 142.0, 139.0, 141.0, 500000, 141.0),
        ])

        # Set download statuses
        update_download_log(in_memory_db, "AAPL", "complete")
        update_download_log(in_memory_db, "GOOG", "complete")
        update_download_log(in_memory_db, "MSFT", "failed")
        update_download_log(in_memory_db, "TSLA", "no_data")

        status = get_download_status(in_memory_db)

        assert status["total_tickers"] == 4
        assert status["total_rows"] == 3
        assert status["complete"] == 2
        assert status["failed"] == 1
        assert status["no_data"] == 1
        assert status["pending"] == 0

    def test_empty_db(self, in_memory_db):
        init_db(in_memory_db)
        status = get_download_status(in_memory_db)
        assert status["total_tickers"] == 0
        assert status["total_rows"] == 0
        assert status["complete"] == 0
        assert status["failed"] == 0
        assert status["pending"] == 0


class TestRetryFailed:
    @patch("stock_trading.downloader.time.sleep")
    @patch("stock_trading.downloader.download_batch")
    def test_retries_failed_tickers(self, mock_batch, mock_sleep, in_memory_db):
        init_db(in_memory_db)
        update_download_log(in_memory_db, "AAPL", "complete")
        update_download_log(in_memory_db, "GOOG", "failed")
        update_download_log(in_memory_db, "MSFT", "failed")

        mock_batch.return_value = {"downloaded": 2, "failed": 0, "rows": 100}

        stats = retry_failed(in_memory_db)

        called_tickers = mock_batch.call_args[0][1]
        assert set(called_tickers) == {"GOOG", "MSFT"}
        assert stats["downloaded"] == 2

    @patch("stock_trading.downloader.time.sleep")
    def test_no_failed_tickers(self, mock_sleep, in_memory_db):
        init_db(in_memory_db)
        update_download_log(in_memory_db, "AAPL", "complete")

        stats = retry_failed(in_memory_db)
        assert stats == {"downloaded": 0, "failed": 0, "rows": 0}
