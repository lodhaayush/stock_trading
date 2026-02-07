"""Tests for stock_trading.fundamentals module."""

from unittest.mock import MagicMock, patch

from stock_trading.db import init_db, upsert_ticker
from stock_trading.fundamentals import fetch_all_fundamentals, fetch_ticker_fundamentals


class TestFetchTickerFundamentals:
    @patch("stock_trading.fundamentals.yf")
    def test_returns_mapped_fields(self, mock_yf):
        mock_yf.Ticker.return_value.info = {
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "marketCap": 3000000000000,
            "trailingPE": 28.5,
            "forwardPE": 25.0,
            "dividendYield": 0.005,
            "beta": 1.2,
        }
        result = fetch_ticker_fundamentals("AAPL")
        assert result == {
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "market_cap": 3000000000000,
            "trailing_pe": 28.5,
            "forward_pe": 25.0,
            "dividend_yield": 0.005,
            "beta": 1.2,
        }
        mock_yf.Ticker.assert_called_once_with("AAPL")

    @patch("stock_trading.fundamentals.yf")
    def test_missing_fields_return_none(self, mock_yf):
        mock_yf.Ticker.return_value.info = {
            "sector": "Technology",
        }
        result = fetch_ticker_fundamentals("AAPL")
        assert result["sector"] == "Technology"
        assert result["industry"] is None
        assert result["market_cap"] is None
        assert result["trailing_pe"] is None
        assert result["forward_pe"] is None
        assert result["dividend_yield"] is None
        assert result["beta"] is None

    @patch("stock_trading.fundamentals.yf")
    def test_exception_returns_none(self, mock_yf):
        mock_yf.Ticker.return_value.info.__getitem__ = MagicMock(
            side_effect=RuntimeError("API error")
        )
        mock_yf.Ticker.side_effect = RuntimeError("API error")
        result = fetch_ticker_fundamentals("AAPL")
        assert result is None


class TestFetchAllFundamentals:
    @patch("stock_trading.fundamentals.time")
    @patch("stock_trading.fundamentals.fetch_ticker_fundamentals")
    def test_processes_tickers_and_updates_db(self, mock_fetch, mock_time, in_memory_db):
        init_db(in_memory_db)
        upsert_ticker(in_memory_db, "AAPL", name="Apple Inc.", exchange="NASDAQ")
        upsert_ticker(in_memory_db, "GOOG", name="Alphabet Inc.", exchange="NASDAQ")

        mock_fetch.side_effect = [
            {
                "sector": "Technology",
                "industry": "Consumer Electronics",
                "market_cap": 3e12,
                "trailing_pe": 28.5,
                "forward_pe": 25.0,
                "dividend_yield": 0.005,
                "beta": 1.2,
            },
            {
                "sector": "Communication Services",
                "industry": "Internet Content",
                "market_cap": 2e12,
                "trailing_pe": 22.0,
                "forward_pe": 20.0,
                "dividend_yield": None,
                "beta": 1.1,
            },
        ]

        result = fetch_all_fundamentals(in_memory_db)

        assert result == {"processed": 2, "updated": 2, "failed": 0}

        row = in_memory_db.execute(
            "SELECT * FROM tickers WHERE ticker='AAPL'"
        ).fetchone()
        assert row["sector"] == "Technology"
        assert row["market_cap"] == 3e12
        assert row["last_updated"] is not None

        row = in_memory_db.execute(
            "SELECT * FROM tickers WHERE ticker='GOOG'"
        ).fetchone()
        assert row["sector"] == "Communication Services"

    @patch("stock_trading.fundamentals.time")
    @patch("stock_trading.fundamentals.fetch_ticker_fundamentals")
    def test_respects_limit(self, mock_fetch, mock_time, in_memory_db):
        init_db(in_memory_db)
        upsert_ticker(in_memory_db, "AAPL", name="Apple Inc.", exchange="NASDAQ")
        upsert_ticker(in_memory_db, "GOOG", name="Alphabet Inc.", exchange="NASDAQ")
        upsert_ticker(in_memory_db, "MSFT", name="Microsoft Corp.", exchange="NASDAQ")

        mock_fetch.return_value = {
            "sector": "Technology",
            "industry": "Software",
            "market_cap": 1e12,
            "trailing_pe": 30.0,
            "forward_pe": 28.0,
            "dividend_yield": 0.008,
            "beta": 0.9,
        }

        result = fetch_all_fundamentals(in_memory_db, limit=2)

        assert result["processed"] == 2
        assert mock_fetch.call_count == 2

    @patch("stock_trading.fundamentals.time")
    @patch("stock_trading.fundamentals.fetch_ticker_fundamentals")
    def test_handles_failed_ticker(self, mock_fetch, mock_time, in_memory_db):
        init_db(in_memory_db)
        upsert_ticker(in_memory_db, "AAPL", name="Apple Inc.", exchange="NASDAQ")
        upsert_ticker(in_memory_db, "BAD", name="Bad Corp.", exchange="NASDAQ")

        mock_fetch.side_effect = [
            {
                "sector": "Technology",
                "industry": "Consumer Electronics",
                "market_cap": 3e12,
                "trailing_pe": 28.5,
                "forward_pe": 25.0,
                "dividend_yield": 0.005,
                "beta": 1.2,
            },
            None,
        ]

        result = fetch_all_fundamentals(in_memory_db)

        assert result == {"processed": 2, "updated": 1, "failed": 1}

        row = in_memory_db.execute(
            "SELECT sector FROM tickers WHERE ticker='BAD'"
        ).fetchone()
        assert row["sector"] is None
