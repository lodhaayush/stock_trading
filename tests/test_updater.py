"""Tests for stock_trading.updater module."""

from unittest.mock import MagicMock, patch, call

import pandas as pd
import pytest

from stock_trading.db import init_db, upsert_ticker, upsert_prices
from stock_trading.updater import run_daily_update, detect_delistings


class TestRunDailyUpdate:
    @patch("stock_trading.updater.download_batch")
    @patch("stock_trading.updater.tickers_mod.sync_tickers")
    def test_calls_sync_and_processes_fresh_tickers(
        self, mock_sync, mock_download_batch, in_memory_db
    ):
        """run_daily_update syncs tickers then downloads fresh ones with period='max'."""
        init_db(in_memory_db)
        # Insert tickers that sync_tickers would populate
        upsert_ticker(in_memory_db, "AAPL", name="Apple", exchange="NASDAQ")
        upsert_ticker(in_memory_db, "GOOG", name="Alphabet", exchange="NASDAQ")

        mock_sync.return_value = {"total": 2, "new": 2, "updated": 0}
        mock_download_batch.return_value = {"downloaded": 2, "failed": 0, "rows": 100}

        result = run_daily_update(in_memory_db)

        mock_sync.assert_called_once_with(in_memory_db)
        assert mock_download_batch.call_count >= 1
        # Should have used period="max" since no price data exists
        _, kwargs = mock_download_batch.call_args
        assert kwargs.get("period") == "max" or mock_download_batch.call_args[0][2] == "max"
        assert result["tickers_synced"] == 2
        assert result["prices_updated"] == 2
        assert result["new_rows"] == 100

    @patch("stock_trading.updater.yf.download")
    @patch("stock_trading.updater.download_batch")
    @patch("stock_trading.updater.tickers_mod.sync_tickers")
    def test_incremental_update_uses_start_date(
        self, mock_sync, mock_download_batch, mock_yf_download, in_memory_db
    ):
        """Tickers with existing price data get incremental updates."""
        init_db(in_memory_db)
        upsert_ticker(in_memory_db, "AAPL", name="Apple", exchange="NASDAQ")
        # Insert existing price data so AAPL has a last_price_date
        upsert_prices(in_memory_db, [
            ("AAPL", "2024-01-10", 100.0, 105.0, 99.0, 104.0, 1000000, 104.0),
        ])

        mock_sync.return_value = {"total": 1, "new": 0, "updated": 1}
        # No fresh tickers, so download_batch should not be called
        mock_yf_download.return_value = pd.DataFrame()  # empty - no new data

        result = run_daily_update(in_memory_db)

        mock_sync.assert_called_once()
        # download_batch should NOT be called (no fresh tickers)
        mock_download_batch.assert_not_called()
        # yf.download should be called with start date = day after last price date
        mock_yf_download.assert_called_once()
        call_kwargs = mock_yf_download.call_args
        assert call_kwargs[1]["start"] == "2024-01-11"

    @patch("stock_trading.updater.yf.download")
    @patch("stock_trading.updater.download_batch")
    @patch("stock_trading.updater.tickers_mod.sync_tickers")
    def test_mixed_fresh_and_incremental(
        self, mock_sync, mock_download_batch, mock_yf_download, in_memory_db
    ):
        """Both fresh and incremental tickers are processed."""
        init_db(in_memory_db)
        upsert_ticker(in_memory_db, "AAPL", name="Apple", exchange="NASDAQ")
        upsert_ticker(in_memory_db, "GOOG", name="Alphabet", exchange="NASDAQ")
        # AAPL has data, GOOG doesn't
        upsert_prices(in_memory_db, [
            ("AAPL", "2024-01-10", 100.0, 105.0, 99.0, 104.0, 1000000, 104.0),
        ])

        mock_sync.return_value = {"total": 2, "new": 0, "updated": 2}
        mock_download_batch.return_value = {"downloaded": 1, "failed": 0, "rows": 50}
        mock_yf_download.return_value = pd.DataFrame()  # no new incremental data

        result = run_daily_update(in_memory_db)

        # download_batch called for fresh ticker (GOOG)
        mock_download_batch.assert_called_once()
        batch_tickers = mock_download_batch.call_args[0][1]
        assert "GOOG" in batch_tickers
        assert "AAPL" not in batch_tickers

        # yf.download called for incremental ticker (AAPL)
        mock_yf_download.assert_called_once()

        assert result["tickers_synced"] == 2
        assert result["prices_updated"] == 1  # only GOOG from download_batch
        assert result["new_rows"] == 50

    @patch("stock_trading.updater.download_batch")
    @patch("stock_trading.updater.tickers_mod.sync_tickers")
    def test_empty_db_no_tickers(
        self, mock_sync, mock_download_batch, in_memory_db
    ):
        """With no tickers in DB, summary shows zero updates."""
        init_db(in_memory_db)
        mock_sync.return_value = {"total": 0, "new": 0, "updated": 0}

        result = run_daily_update(in_memory_db)

        assert result["tickers_synced"] == 0
        assert result["prices_updated"] == 0
        assert result["new_rows"] == 0
        mock_download_batch.assert_not_called()

    @patch("stock_trading.updater.fundamentals")
    @patch("stock_trading.updater.download_batch")
    @patch("stock_trading.updater.tickers_mod.sync_tickers")
    def test_include_fundamentals_calls_fetch(
        self, mock_sync, mock_download_batch, mock_fundamentals, in_memory_db
    ):
        """When include_fundamentals=True, fundamentals.fetch_all_fundamentals is called."""
        init_db(in_memory_db)
        mock_sync.return_value = {"total": 0, "new": 0, "updated": 0}

        result = run_daily_update(in_memory_db, include_fundamentals=True)

        mock_fundamentals.fetch_all_fundamentals.assert_called_once_with(in_memory_db)


class TestDetectDelistings:
    @patch("stock_trading.updater.tickers_mod.fetch_all_tickers")
    def test_detects_missing_tickers(self, mock_fetch, in_memory_db):
        """Tickers in DB but not in current listings are detected."""
        init_db(in_memory_db)
        upsert_ticker(in_memory_db, "AAPL", name="Apple", exchange="NASDAQ")
        upsert_ticker(in_memory_db, "GOOG", name="Alphabet", exchange="NASDAQ")
        upsert_ticker(in_memory_db, "DELIST", name="Delisted Co", exchange="NASDAQ")

        # Current listings only have AAPL and GOOG
        mock_fetch.return_value = [
            {"ticker": "AAPL", "name": "Apple", "exchange": "NASDAQ"},
            {"ticker": "GOOG", "name": "Alphabet", "exchange": "NASDAQ"},
        ]

        delisted = detect_delistings(in_memory_db)

        assert delisted == ["DELIST"]

    @patch("stock_trading.updater.tickers_mod.fetch_all_tickers")
    def test_no_delistings(self, mock_fetch, in_memory_db):
        """When all DB tickers are in current listings, returns empty list."""
        init_db(in_memory_db)
        upsert_ticker(in_memory_db, "AAPL", name="Apple", exchange="NASDAQ")

        mock_fetch.return_value = [
            {"ticker": "AAPL", "name": "Apple", "exchange": "NASDAQ"},
            {"ticker": "GOOG", "name": "Alphabet", "exchange": "NASDAQ"},
        ]

        delisted = detect_delistings(in_memory_db)

        assert delisted == []

    @patch("stock_trading.updater.tickers_mod.fetch_all_tickers")
    def test_empty_db(self, mock_fetch, in_memory_db):
        """With no tickers in DB, no delistings are reported."""
        init_db(in_memory_db)

        mock_fetch.return_value = [
            {"ticker": "AAPL", "name": "Apple", "exchange": "NASDAQ"},
        ]

        delisted = detect_delistings(in_memory_db)

        assert delisted == []

    @patch("stock_trading.updater.tickers_mod.fetch_all_tickers")
    def test_multiple_delistings_sorted(self, mock_fetch, in_memory_db):
        """Multiple delisted tickers are returned in sorted order."""
        init_db(in_memory_db)
        for t in ["ZZZ", "AAA", "MMM", "AAPL"]:
            upsert_ticker(in_memory_db, t)

        mock_fetch.return_value = [
            {"ticker": "AAPL", "name": "Apple", "exchange": "NASDAQ"},
        ]

        delisted = detect_delistings(in_memory_db)

        assert delisted == ["AAA", "MMM", "ZZZ"]
