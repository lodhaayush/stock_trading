"""Tests for stock_trading.tickers module."""

from unittest.mock import MagicMock, patch

import pytest

from stock_trading import db
from stock_trading.tickers import (
    fetch_all_tickers,
    fetch_nasdaq_tickers,
    fetch_sec_edgar_tickers,
    filter_tickers,
    sync_tickers,
)


# ---------------------------------------------------------------------------
# filter_tickers tests
# ---------------------------------------------------------------------------

class TestFilterTickers:
    def test_keeps_valid_common_stock(self):
        tickers = [
            {"ticker": "AAPL", "name": "Apple Inc. - Common Stock", "exchange": "NASDAQ"},
            {"ticker": "MSFT", "name": "Microsoft Corporation", "exchange": "NASDAQ"},
            {"ticker": "GOOGL", "name": "Alphabet Inc.", "exchange": "NASDAQ"},
        ]
        result = filter_tickers(tickers)
        assert len(result) == 3
        assert {t["ticker"] for t in result} == {"AAPL", "MSFT", "GOOGL"}

    def test_removes_symbols_with_dot(self):
        tickers = [
            {"ticker": "BRK.B", "name": "Berkshire Hathaway", "exchange": "NYSE"},
        ]
        assert filter_tickers(tickers) == []

    def test_removes_symbols_with_dollar(self):
        tickers = [
            {"ticker": "FOO$", "name": "Some Ticker", "exchange": "NYSE"},
        ]
        assert filter_tickers(tickers) == []

    def test_removes_symbols_with_dash(self):
        tickers = [
            {"ticker": "ABC-D", "name": "Some Ticker", "exchange": "NYSE"},
        ]
        assert filter_tickers(tickers) == []

    def test_removes_long_symbols(self):
        tickers = [
            {"ticker": "ABCDEF", "name": "Long Symbol Inc.", "exchange": "NYSE"},
        ]
        assert filter_tickers(tickers) == []

    def test_keeps_five_char_symbol(self):
        tickers = [
            {"ticker": "ABCDE", "name": "Five Char Inc.", "exchange": "NYSE"},
        ]
        result = filter_tickers(tickers)
        assert len(result) == 1

    def test_removes_warrants(self):
        tickers = [
            {"ticker": "ABCW", "name": "ABC Warrant", "exchange": "NYSE"},
            {"ticker": "XYZ", "name": "XYZ Warrants", "exchange": "NYSE"},
        ]
        assert filter_tickers(tickers) == []

    def test_removes_units(self):
        tickers = [
            {"ticker": "ABCU", "name": "ABC Unit", "exchange": "NYSE"},
        ]
        assert filter_tickers(tickers) == []

    def test_removes_rights(self):
        tickers = [
            {"ticker": "ABCR", "name": "ABC Right", "exchange": "NYSE"},
        ]
        assert filter_tickers(tickers) == []

    def test_removes_preferred(self):
        tickers = [
            {"ticker": "ABCP", "name": "ABC Preferred Stock", "exchange": "NYSE"},
        ]
        assert filter_tickers(tickers) == []

    def test_removes_test_standalone_word(self):
        tickers = [
            {"ticker": "ZTEST", "name": "NASDAQ test Stock", "exchange": "NASDAQ"},
        ]
        assert filter_tickers(tickers) == []

    def test_keeps_contest_in_name(self):
        """'test' inside another word should NOT be filtered."""
        tickers = [
            {"ticker": "CTST", "name": "Contest Corp", "exchange": "NYSE"},
        ]
        result = filter_tickers(tickers)
        assert len(result) == 1

    def test_case_insensitive_keyword_filter(self):
        tickers = [
            {"ticker": "WAR", "name": "Some WARRANT Issue", "exchange": "NYSE"},
            {"ticker": "UNT", "name": "Some UNIT Issue", "exchange": "NYSE"},
            {"ticker": "PFD", "name": "Some PREFERRED Issue", "exchange": "NYSE"},
        ]
        assert filter_tickers(tickers) == []


# ---------------------------------------------------------------------------
# fetch_nasdaq_tickers tests
# ---------------------------------------------------------------------------

MOCK_NASDAQ_LISTED = (
    "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares\r\n"
    "AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N\r\n"
    "MSFT|Microsoft Corporation - Common Stock|Q|N|N|100|N|N\r\n"
    "File Creation Time: 0101202500:00|||||||\r\n"
)

MOCK_OTHER_LISTED = (
    "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol\r\n"
    "IBM|International Business Machines Corporation|A|IBM|N|100|N|IBM\r\n"
    "GE|GE Aerospace|N|GE|N|100|N|GE\r\n"
    "File Creation Time: 0101202500:00|||||||\r\n"
)


class TestFetchNasdaqTickers:
    @patch("stock_trading.tickers.requests.get")
    def test_parses_both_files(self, mock_get):
        resp_nasdaq = MagicMock()
        resp_nasdaq.text = MOCK_NASDAQ_LISTED
        resp_nasdaq.raise_for_status = MagicMock()

        resp_other = MagicMock()
        resp_other.text = MOCK_OTHER_LISTED
        resp_other.raise_for_status = MagicMock()

        mock_get.side_effect = [resp_nasdaq, resp_other]

        tickers = fetch_nasdaq_tickers()

        assert len(tickers) == 4
        symbols = {t["ticker"] for t in tickers}
        assert symbols == {"AAPL", "MSFT", "IBM", "GE"}

        # Check exchange assignments
        nasdaq_tickers = [t for t in tickers if t["exchange"] == "NASDAQ"]
        assert len(nasdaq_tickers) == 2

        ibm = next(t for t in tickers if t["ticker"] == "IBM")
        assert ibm["exchange"] == "A"

    @patch("stock_trading.tickers.requests.get")
    def test_skips_file_creation_line(self, mock_get):
        resp_nasdaq = MagicMock()
        resp_nasdaq.text = MOCK_NASDAQ_LISTED
        resp_nasdaq.raise_for_status = MagicMock()

        resp_other = MagicMock()
        resp_other.text = MOCK_OTHER_LISTED
        resp_other.raise_for_status = MagicMock()

        mock_get.side_effect = [resp_nasdaq, resp_other]

        tickers = fetch_nasdaq_tickers()
        # Should not include any entry from "File Creation Time" lines
        for t in tickers:
            assert "File Creation" not in t["ticker"]
            assert "File Creation" not in t["name"]


# ---------------------------------------------------------------------------
# fetch_sec_edgar_tickers tests
# ---------------------------------------------------------------------------

class TestFetchSecEdgarTickers:
    @patch("stock_trading.tickers.requests.get")
    def test_parses_json_response(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
            "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp"},
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        tickers = fetch_sec_edgar_tickers()

        assert len(tickers) == 2
        assert tickers[0]["ticker"] == "AAPL"
        assert tickers[0]["name"] == "Apple Inc."
        assert tickers[0]["exchange"] == "UNKNOWN"
        assert tickers[1]["ticker"] == "MSFT"


# ---------------------------------------------------------------------------
# fetch_all_tickers tests
# ---------------------------------------------------------------------------

class TestFetchAllTickers:
    @patch("stock_trading.tickers.fetch_nasdaq_tickers")
    def test_uses_nasdaq_first(self, mock_nasdaq):
        mock_nasdaq.return_value = [
            {"ticker": "AAPL", "name": "Apple Inc.", "exchange": "NASDAQ"},
        ]
        result = fetch_all_tickers()
        assert len(result) == 1
        assert result[0]["ticker"] == "AAPL"
        mock_nasdaq.assert_called_once()

    @patch("stock_trading.tickers.fetch_sec_edgar_tickers")
    @patch("stock_trading.tickers.fetch_nasdaq_tickers")
    def test_falls_back_to_sec_edgar(self, mock_nasdaq, mock_sec):
        mock_nasdaq.side_effect = Exception("Connection error")
        mock_sec.return_value = [
            {"ticker": "MSFT", "name": "Microsoft Corp", "exchange": "UNKNOWN"},
        ]
        result = fetch_all_tickers()
        assert len(result) == 1
        assert result[0]["ticker"] == "MSFT"

    @patch("stock_trading.tickers.fetch_nasdaq_tickers")
    def test_applies_filter(self, mock_nasdaq):
        mock_nasdaq.return_value = [
            {"ticker": "AAPL", "name": "Apple Inc.", "exchange": "NASDAQ"},
            {"ticker": "BRK.B", "name": "Berkshire Hathaway", "exchange": "NYSE"},
            {"ticker": "WARW", "name": "Some Warrant", "exchange": "NYSE"},
        ]
        result = fetch_all_tickers()
        assert len(result) == 1
        assert result[0]["ticker"] == "AAPL"


# ---------------------------------------------------------------------------
# sync_tickers tests
# ---------------------------------------------------------------------------

class TestSyncTickers:
    @patch("stock_trading.tickers.fetch_all_tickers")
    def test_sync_new_tickers(self, mock_fetch, in_memory_db):
        db.init_db(in_memory_db)
        mock_fetch.return_value = [
            {"ticker": "AAPL", "name": "Apple Inc.", "exchange": "NASDAQ"},
            {"ticker": "MSFT", "name": "Microsoft Corp", "exchange": "NASDAQ"},
        ]

        result = sync_tickers(in_memory_db)

        assert result["total"] == 2
        assert result["new"] == 2
        assert result["updated"] == 0

        rows = in_memory_db.execute("SELECT * FROM tickers").fetchall()
        assert len(rows) == 2

    @patch("stock_trading.tickers.fetch_all_tickers")
    def test_sync_with_existing_tickers(self, mock_fetch, in_memory_db):
        db.init_db(in_memory_db)
        # Pre-insert one ticker
        db.upsert_ticker(in_memory_db, "AAPL", name="Apple", exchange="NASDAQ")

        mock_fetch.return_value = [
            {"ticker": "AAPL", "name": "Apple Inc.", "exchange": "NASDAQ"},
            {"ticker": "MSFT", "name": "Microsoft Corp", "exchange": "NASDAQ"},
        ]

        result = sync_tickers(in_memory_db)

        assert result["total"] == 2
        assert result["new"] == 1
        assert result["updated"] == 1

    @patch("stock_trading.tickers.fetch_all_tickers")
    def test_sync_returns_correct_counts(self, mock_fetch, in_memory_db):
        db.init_db(in_memory_db)

        mock_fetch.return_value = [
            {"ticker": "A", "name": "Agilent", "exchange": "NYSE"},
            {"ticker": "AA", "name": "Alcoa", "exchange": "NYSE"},
            {"ticker": "AAL", "name": "American Airlines", "exchange": "NASDAQ"},
        ]

        result = sync_tickers(in_memory_db)
        assert result == {"total": 3, "new": 3, "updated": 0}

        # Run sync again — all should be updated, none new
        result2 = sync_tickers(in_memory_db)
        assert result2 == {"total": 3, "new": 0, "updated": 3}
