"""Tests for stock_trading.charting module."""

from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from stock_trading.charting import (
    rows_to_dataframe,
    compute_indicators,
    render_chart,
    chart_ticker,
)
from stock_trading.cli import cli
from stock_trading.db import init_db, upsert_prices


def _make_sample_rows(ticker="AAPL", n=50):
    """Generate n sample price row tuples for seeding the DB."""
    rows = []
    base = 100.0
    for i in range(n):
        day = f"2024-01-{(i % 28) + 1:02d}" if i < 28 else f"2024-02-{(i - 28) + 1:02d}"
        o = base + i * 0.5
        h = o + 2.0
        lo = o - 1.0
        c = o + 1.0
        v = 1_000_000 + i * 10_000
        adj = c
        rows.append((ticker, day, o, h, lo, c, v, adj))
    return rows


def _make_df(n=50):
    """Create a synthetic OHLCV DataFrame for testing indicators."""
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(np.random.default_rng(42).standard_normal(n) * 0.5)
    return pd.DataFrame({
        "Open": close - 0.5,
        "High": close + 1.0,
        "Low": close - 1.0,
        "Close": close,
        "Volume": np.random.default_rng(42).integers(500_000, 2_000_000, n),
    }, index=dates)


class TestRowsToDataframe:
    def test_returns_none_for_empty_rows(self):
        assert rows_to_dataframe([]) is None

    def test_correct_columns(self, in_memory_db):
        init_db(in_memory_db)
        upsert_prices(in_memory_db, _make_sample_rows(n=5))
        from stock_trading.db import query_prices
        rows = query_prices(in_memory_db, "AAPL")
        df = rows_to_dataframe(rows)
        assert isinstance(df.index, pd.DatetimeIndex)
        for col in ("Open", "High", "Low", "Close", "Volume", "Adj Close"):
            assert col in df.columns
        assert "ticker" not in df.columns

    def test_index_name_is_date(self, in_memory_db):
        init_db(in_memory_db)
        upsert_prices(in_memory_db, _make_sample_rows(n=3))
        from stock_trading.db import query_prices
        rows = query_prices(in_memory_db, "AAPL")
        df = rows_to_dataframe(rows)
        assert df.index.name == "Date"

    def test_row_count_matches(self, in_memory_db):
        init_db(in_memory_db)
        sample = _make_sample_rows(n=10)
        upsert_prices(in_memory_db, sample)
        from stock_trading.db import query_prices
        rows = query_prices(in_memory_db, "AAPL")
        df = rows_to_dataframe(rows)
        assert len(df) == len(sample)


class TestComputeIndicators:
    @patch("stock_trading.charting.mpf.make_addplot")
    def test_sma_adds_column_and_addplot(self, mock_addplot):
        df = _make_df(50)
        mock_addplot.return_value = MagicMock()
        indicators = {"sma": [20], "volume": False}
        addplots, kwargs = compute_indicators(df, indicators)
        assert "SMA_20" in df.columns
        assert len(addplots) >= 1

    @patch("stock_trading.charting.mpf.make_addplot")
    def test_multiple_sma_periods(self, mock_addplot):
        df = _make_df(50)
        mock_addplot.return_value = MagicMock()
        indicators = {"sma": [10, 20], "volume": False}
        addplots, _ = compute_indicators(df, indicators)
        assert "SMA_10" in df.columns
        assert "SMA_20" in df.columns
        assert len(addplots) == 2

    @patch("stock_trading.charting.mpf.make_addplot")
    def test_ema_adds_column(self, mock_addplot):
        df = _make_df(50)
        mock_addplot.return_value = MagicMock()
        indicators = {"ema": [12], "volume": False}
        addplots, _ = compute_indicators(df, indicators)
        assert "EMA_12" in df.columns
        assert len(addplots) == 1

    @patch("stock_trading.charting.mpf.make_addplot")
    def test_rsi_creates_three_addplots(self, mock_addplot):
        df = _make_df(50)
        mock_addplot.return_value = MagicMock()
        indicators = {"rsi": 14, "volume": False}
        addplots, _ = compute_indicators(df, indicators)
        # RSI line + 70 line + 30 line
        assert len(addplots) == 3

    @patch("stock_trading.charting.mpf.make_addplot")
    def test_macd_creates_three_addplots(self, mock_addplot):
        df = _make_df(50)
        mock_addplot.return_value = MagicMock()
        indicators = {"macd": (12, 26, 9), "volume": False}
        addplots, _ = compute_indicators(df, indicators)
        # MACD line + signal + histogram
        assert len(addplots) == 3
        assert "MACD" in df.columns
        assert "MACD_Signal" in df.columns
        assert "MACD_Hist" in df.columns

    @patch("stock_trading.charting.mpf.make_addplot")
    def test_bbands_creates_three_addplots(self, mock_addplot):
        df = _make_df(50)
        mock_addplot.return_value = MagicMock()
        indicators = {"bbands": 20, "volume": False}
        addplots, _ = compute_indicators(df, indicators)
        assert len(addplots) == 3
        assert "BB_Upper" in df.columns
        assert "BB_Mid" in df.columns
        assert "BB_Lower" in df.columns

    @patch("stock_trading.charting.mpf.make_addplot")
    def test_volume_kwarg(self, mock_addplot):
        df = _make_df(50)
        mock_addplot.return_value = MagicMock()
        _, kwargs = compute_indicators(df, {"volume": True})
        assert kwargs["volume"] is True
        _, kwargs = compute_indicators(df, {"volume": False})
        assert kwargs["volume"] is False


class TestRenderChart:
    @patch("stock_trading.charting.mpf.plot")
    def test_display_mode(self, mock_plot):
        df = _make_df(5)
        render_chart(df, "TEST")
        mock_plot.assert_called_once()
        call_kwargs = mock_plot.call_args[1]
        assert "savefig" not in call_kwargs

    @patch("stock_trading.charting.mpf.plot")
    def test_save_mode(self, mock_plot):
        df = _make_df(5)
        render_chart(df, "TEST", output="/tmp/test.png")
        mock_plot.assert_called_once()
        call_kwargs = mock_plot.call_args[1]
        assert call_kwargs["savefig"] == "/tmp/test.png"

    @patch("stock_trading.charting.mpf.plot")
    def test_addplots_passed(self, mock_plot):
        df = _make_df(5)
        fake_addplot = [MagicMock()]
        render_chart(df, "TEST", addplots=fake_addplot)
        call_kwargs = mock_plot.call_args[1]
        assert call_kwargs["addplot"] == fake_addplot

    @patch("stock_trading.charting.mpf.plot")
    def test_extra_kwargs_forwarded(self, mock_plot):
        df = _make_df(5)
        render_chart(df, "TEST", volume=True)
        call_kwargs = mock_plot.call_args[1]
        assert call_kwargs["volume"] is True


class TestChartTicker:
    @patch("stock_trading.charting.mpf.plot")
    def test_renders_from_db(self, mock_plot, in_memory_db):
        init_db(in_memory_db)
        upsert_prices(in_memory_db, _make_sample_rows(n=10))
        result = chart_ticker(in_memory_db, "AAPL")
        assert result is True
        mock_plot.assert_called_once()

    @patch("stock_trading.charting.mpf.plot")
    def test_returns_false_for_missing_ticker(self, mock_plot, in_memory_db):
        init_db(in_memory_db)
        result = chart_ticker(in_memory_db, "ZZZZ")
        assert result is False
        mock_plot.assert_not_called()

    @patch("stock_trading.charting.mpf.plot")
    def test_date_range_filter(self, mock_plot, in_memory_db):
        init_db(in_memory_db)
        upsert_prices(in_memory_db, _make_sample_rows(n=28))
        result = chart_ticker(in_memory_db, "AAPL", start="2024-01-10", end="2024-01-20")
        assert result is True
        mock_plot.assert_called_once()


class TestChartCmd:
    SAMPLE_ROWS = _make_sample_rows(n=50)

    def _seed(self, conn):
        init_db(conn)
        upsert_prices(conn, self.SAMPLE_ROWS)

    @patch("stock_trading.charting.mpf.plot")
    def test_chart_basic(self, mock_plot, in_memory_db):
        self._seed(in_memory_db)
        runner = CliRunner()
        with patch("stock_trading.cli.db.get_connection", return_value=in_memory_db):
            result = runner.invoke(cli, ["chart", "--ticker", "AAPL"])
        assert result.exit_code == 0
        mock_plot.assert_called_once()

    @patch("stock_trading.charting.mpf.plot")
    def test_chart_no_data(self, mock_plot, in_memory_db):
        init_db(in_memory_db)
        runner = CliRunner()
        with patch("stock_trading.cli.db.get_connection", return_value=in_memory_db):
            result = runner.invoke(cli, ["chart", "--ticker", "ZZZZ"])
        assert result.exit_code == 0
        assert "No price data found" in result.output
        mock_plot.assert_not_called()

    @patch("stock_trading.charting.mpf.plot")
    def test_chart_with_output(self, mock_plot, in_memory_db):
        self._seed(in_memory_db)
        runner = CliRunner()
        with patch("stock_trading.cli.db.get_connection", return_value=in_memory_db):
            result = runner.invoke(cli, ["chart", "--ticker", "AAPL", "-o", "/tmp/chart.png"])
        assert result.exit_code == 0
        assert "Chart saved to /tmp/chart.png" in result.output

    @patch("stock_trading.charting.mpf.plot")
    def test_chart_with_sma(self, mock_plot, in_memory_db):
        self._seed(in_memory_db)
        runner = CliRunner()
        with patch("stock_trading.cli.db.get_connection", return_value=in_memory_db):
            result = runner.invoke(cli, ["chart", "--ticker", "AAPL", "--sma", "10,20"])
        assert result.exit_code == 0
        mock_plot.assert_called_once()

    @patch("stock_trading.charting.mpf.plot")
    def test_chart_with_indicators(self, mock_plot, in_memory_db):
        self._seed(in_memory_db)
        runner = CliRunner()
        with patch("stock_trading.cli.db.get_connection", return_value=in_memory_db):
            result = runner.invoke(cli, [
                "chart", "--ticker", "AAPL",
                "--sma", "10,20",
                "--rsi", "14",
                "--macd", "12,26,9",
            ])
        assert result.exit_code == 0
        mock_plot.assert_called_once()

    @patch("stock_trading.charting.mpf.plot")
    def test_chart_invalid_macd(self, mock_plot, in_memory_db):
        self._seed(in_memory_db)
        runner = CliRunner()
        with patch("stock_trading.cli.db.get_connection", return_value=in_memory_db):
            result = runner.invoke(cli, ["chart", "--ticker", "AAPL", "--macd", "12,26"])
        assert "Error: --macd requires 3 comma-separated values" in result.output
        mock_plot.assert_not_called()
