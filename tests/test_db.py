"""Tests for stock_trading.db module."""

from stock_trading.db import (
    get_last_price_date,
    init_db,
    query_prices,
    update_download_log,
    upsert_prices,
    upsert_ticker,
)


class TestInitDb:
    def test_creates_all_tables(self, in_memory_db):
        init_db(in_memory_db)
        rows = in_memory_db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = [r["name"] for r in rows]
        assert "tickers" in table_names
        assert "daily_prices" in table_names
        assert "download_log" in table_names


class TestUpsertPrices:
    def test_insert_rows(self, in_memory_db):
        init_db(in_memory_db)
        rows = [
            ("AAPL", "2024-01-01", 100.0, 105.0, 99.0, 104.0, 1000000, 104.0),
            ("AAPL", "2024-01-02", 104.0, 106.0, 103.0, 105.0, 1100000, 105.0),
        ]
        upsert_prices(in_memory_db, rows)
        result = in_memory_db.execute("SELECT COUNT(*) AS cnt FROM daily_prices").fetchone()
        assert result["cnt"] == 2

    def test_update_existing_row(self, in_memory_db):
        init_db(in_memory_db)
        rows = [("AAPL", "2024-01-01", 100.0, 105.0, 99.0, 104.0, 1000000, 104.0)]
        upsert_prices(in_memory_db, rows)

        updated = [("AAPL", "2024-01-01", 101.0, 106.0, 100.0, 105.0, 1200000, 105.0)]
        upsert_prices(in_memory_db, updated)

        result = in_memory_db.execute(
            "SELECT close FROM daily_prices WHERE ticker='AAPL' AND date='2024-01-01'"
        ).fetchone()
        assert result["close"] == 105.0
        count = in_memory_db.execute("SELECT COUNT(*) AS cnt FROM daily_prices").fetchone()
        assert count["cnt"] == 1


class TestGetLastPriceDate:
    def test_returns_max_date(self, in_memory_db):
        init_db(in_memory_db)
        rows = [
            ("AAPL", "2024-01-01", 100.0, 105.0, 99.0, 104.0, 1000000, 104.0),
            ("AAPL", "2024-01-03", 104.0, 106.0, 103.0, 105.0, 1100000, 105.0),
            ("AAPL", "2024-01-02", 102.0, 104.0, 101.0, 103.0, 900000, 103.0),
        ]
        upsert_prices(in_memory_db, rows)
        assert get_last_price_date(in_memory_db, "AAPL") == "2024-01-03"

    def test_returns_none_for_missing_ticker(self, in_memory_db):
        init_db(in_memory_db)
        assert get_last_price_date(in_memory_db, "ZZZZ") is None


class TestUpsertTicker:
    def test_insert_ticker(self, in_memory_db):
        init_db(in_memory_db)
        upsert_ticker(in_memory_db, "AAPL", name="Apple Inc.", exchange="NASDAQ")
        row = in_memory_db.execute(
            "SELECT * FROM tickers WHERE ticker='AAPL'"
        ).fetchone()
        assert row["name"] == "Apple Inc."
        assert row["exchange"] == "NASDAQ"

    def test_update_ticker(self, in_memory_db):
        init_db(in_memory_db)
        upsert_ticker(in_memory_db, "AAPL", name="Apple Inc.", exchange="NASDAQ")
        upsert_ticker(in_memory_db, "AAPL", name="Apple Inc.", exchange="NYSE")
        row = in_memory_db.execute(
            "SELECT * FROM tickers WHERE ticker='AAPL'"
        ).fetchone()
        assert row["exchange"] == "NYSE"
        count = in_memory_db.execute("SELECT COUNT(*) AS cnt FROM tickers").fetchone()
        assert count["cnt"] == 1


class TestUpdateDownloadLog:
    def test_tracks_status(self, in_memory_db):
        init_db(in_memory_db)
        update_download_log(in_memory_db, "AAPL", "success", last_price_date="2024-01-03")
        row = in_memory_db.execute(
            "SELECT * FROM download_log WHERE ticker='AAPL'"
        ).fetchone()
        assert row["status"] == "success"
        assert row["last_price_date"] == "2024-01-03"
        assert row["last_download"] is not None

    def test_tracks_error(self, in_memory_db):
        init_db(in_memory_db)
        update_download_log(in_memory_db, "AAPL", "error", error_message="timeout")
        row = in_memory_db.execute(
            "SELECT * FROM download_log WHERE ticker='AAPL'"
        ).fetchone()
        assert row["status"] == "error"
        assert row["error_message"] == "timeout"


class TestQueryPrices:
    def _seed(self, conn):
        init_db(conn)
        rows = [
            ("AAPL", "2024-01-01", 100.0, 105.0, 99.0, 104.0, 1000000, 104.0),
            ("AAPL", "2024-01-02", 102.0, 104.0, 101.0, 103.0, 900000, 103.0),
            ("AAPL", "2024-01-03", 104.0, 106.0, 103.0, 105.0, 1100000, 105.0),
            ("GOOG", "2024-01-01", 140.0, 142.0, 139.0, 141.0, 500000, 141.0),
        ]
        upsert_prices(conn, rows)

    def test_all_rows_for_ticker(self, in_memory_db):
        self._seed(in_memory_db)
        result = query_prices(in_memory_db, "AAPL")
        assert len(result) == 3

    def test_with_start_date(self, in_memory_db):
        self._seed(in_memory_db)
        result = query_prices(in_memory_db, "AAPL", start_date="2024-01-02")
        assert len(result) == 2
        assert result[0]["date"] == "2024-01-02"

    def test_with_end_date(self, in_memory_db):
        self._seed(in_memory_db)
        result = query_prices(in_memory_db, "AAPL", end_date="2024-01-02")
        assert len(result) == 2
        assert result[-1]["date"] == "2024-01-02"

    def test_with_date_range(self, in_memory_db):
        self._seed(in_memory_db)
        result = query_prices(
            in_memory_db, "AAPL", start_date="2024-01-02", end_date="2024-01-02"
        )
        assert len(result) == 1
        assert result[0]["date"] == "2024-01-02"

    def test_different_ticker(self, in_memory_db):
        self._seed(in_memory_db)
        result = query_prices(in_memory_db, "GOOG")
        assert len(result) == 1
