"""Tests for stock_trading.cli module."""

from unittest.mock import patch

from click.testing import CliRunner

from stock_trading.cli import cli
from stock_trading.db import init_db, upsert_prices


class TestQueryCmd:
    SAMPLE_ROWS = [
        ("AAPL", "2024-01-01", 100.0, 105.0, 99.0, 104.0, 1000000, 104.0),
        ("AAPL", "2024-01-02", 102.0, 104.0, 101.0, 103.0, 900000, 103.0),
        ("AAPL", "2024-01-03", 104.0, 106.0, 103.0, 105.0, 1100000, 105.0),
    ]

    def _seed(self, conn):
        init_db(conn)
        upsert_prices(conn, self.SAMPLE_ROWS)

    def test_query_returns_data(self, in_memory_db):
        self._seed(in_memory_db)
        runner = CliRunner()
        with patch("stock_trading.cli.db.get_connection", return_value=in_memory_db):
            result = runner.invoke(cli, ["query", "--ticker", "AAPL"])

        assert result.exit_code == 0
        assert "Ticker: AAPL" in result.output
        assert "3 row(s)" in result.output
        assert "2024-01-01" in result.output
        assert "2024-01-02" in result.output
        assert "2024-01-03" in result.output
        # Check header columns are present
        assert "Date" in result.output
        assert "Open" in result.output
        assert "High" in result.output
        assert "Low" in result.output
        assert "Close" in result.output
        assert "Volume" in result.output
        assert "Adj Close" in result.output

    def test_query_with_start_and_end(self, in_memory_db):
        self._seed(in_memory_db)
        runner = CliRunner()
        with patch("stock_trading.cli.db.get_connection", return_value=in_memory_db):
            result = runner.invoke(
                cli,
                ["query", "--ticker", "AAPL", "--start", "2024-01-02", "--end", "2024-01-02"],
            )

        assert result.exit_code == 0
        assert "1 row(s)" in result.output
        assert "2024-01-02" in result.output
        assert "2024-01-01" not in result.output
        assert "2024-01-03" not in result.output

    def test_query_with_start_only(self, in_memory_db):
        self._seed(in_memory_db)
        runner = CliRunner()
        with patch("stock_trading.cli.db.get_connection", return_value=in_memory_db):
            result = runner.invoke(
                cli, ["query", "--ticker", "AAPL", "--start", "2024-01-02"]
            )

        assert result.exit_code == 0
        assert "2 row(s)" in result.output
        assert "2024-01-01" not in result.output

    def test_query_with_end_only(self, in_memory_db):
        self._seed(in_memory_db)
        runner = CliRunner()
        with patch("stock_trading.cli.db.get_connection", return_value=in_memory_db):
            result = runner.invoke(
                cli, ["query", "--ticker", "AAPL", "--end", "2024-01-01"]
            )

        assert result.exit_code == 0
        assert "1 row(s)" in result.output
        assert "2024-01-01" in result.output

    def test_query_no_matching_data(self, in_memory_db):
        init_db(in_memory_db)
        runner = CliRunner()
        with patch("stock_trading.cli.db.get_connection", return_value=in_memory_db):
            result = runner.invoke(cli, ["query", "--ticker", "ZZZZ"])

        assert result.exit_code == 0
        assert "No price data found for ZZZZ." in result.output

    def test_query_formats_values(self, in_memory_db):
        self._seed(in_memory_db)
        runner = CliRunner()
        with patch("stock_trading.cli.db.get_connection", return_value=in_memory_db):
            result = runner.invoke(cli, ["query", "--ticker", "AAPL"])

        assert result.exit_code == 0
        # Check that numeric values appear formatted with two decimals
        assert "100.00" in result.output
        assert "105.00" in result.output
        assert "1000000" in result.output

    def test_query_requires_ticker(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["query"])
        assert result.exit_code != 0
        assert "Missing option '--ticker'" in result.output or "required" in result.output.lower()
