"""Tests for the movers feature (rank_movers + CLI command)."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from stock_trading.cli import cli
from stock_trading.db import init_db, upsert_prices
from stock_trading.screener import rank_movers


def _seed_movers_db(conn, tickers_config):
    """Seed DB with fully controlled price/volume data for movers tests.

    Each entry in tickers_config:
        {
            "ticker": "AAPL",
            "name": "Apple",
            "sector": "Technology",
            "market_cap": 3e12,
            "prices": [
                # (date_str, open, high, low, close, volume)
                ("2026-02-02", 100, 101, 99, 100, 1_000_000),
                ...
            ]
        }
    """
    init_db(conn)
    for t in tickers_config:
        conn.execute(
            "INSERT OR REPLACE INTO tickers "
            "(ticker, name, exchange, sector, industry, market_cap) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                t["ticker"], t.get("name", t["ticker"]),
                t.get("exchange", "NASDAQ"), t.get("sector", "Technology"),
                t.get("industry", "Software"), t.get("market_cap", 1e9),
            ),
        )
        rows = [
            (t["ticker"], p[0], p[1], p[2], p[3], p[4], p[5], p[4])
            for p in t["prices"]
        ]
        upsert_prices(conn, rows)
    conn.commit()


def _make_prices(base_price, daily_returns, base_volume, volume_multipliers,
                 start_date="2026-01-20"):
    """Generate price tuples from daily returns and volume multipliers."""
    dates = pd.bdate_range(start_date, periods=len(daily_returns))
    prices = []
    price = base_price
    for i, (ret, vol_mult) in enumerate(zip(daily_returns, volume_multipliers)):
        open_ = price
        price = price * (1 + ret)
        close = price
        high = max(open_, close) * 1.005
        low = min(open_, close) * 0.995
        volume = int(base_volume * vol_mult)
        prices.append((dates[i].strftime("%Y-%m-%d"), open_, high, low, close, volume))
    return prices


# ---------------------------------------------------------------------------
# rank_movers unit tests
# ---------------------------------------------------------------------------

class TestRankMovers:
    def test_returns_expected_columns(self, in_memory_db):
        prices = _make_prices(100, [0.01] * 20, 1_000_000, [1.0] * 20)
        _seed_movers_db(in_memory_db, [
            {"ticker": "AAPL", "name": "Apple", "prices": prices},
        ])
        result = rank_movers(in_memory_db, days=5)
        expected_cols = {
            "ticker", "first_date", "last_date", "first_close", "last_close",
            "return_pct", "avg_volume", "baseline_avg_volume", "volume_surge",
            "name", "sector", "market_cap",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_return_calculation(self, in_memory_db):
        # 15 baseline days at 0% return, then 5 days at +2% each
        returns = [0.0] * 15 + [0.02] * 5
        prices = _make_prices(100, returns, 1_000_000, [1.0] * 20)
        _seed_movers_db(in_memory_db, [
            {"ticker": "TEST", "prices": prices},
        ])
        result = rank_movers(in_memory_db, days=5)
        assert len(result) == 1
        # 5 close prices → 4 close-to-close intervals: (1.02^4 - 1) ≈ 0.08243
        assert result.iloc[0]["return_pct"] == pytest.approx(1.02**4 - 1, rel=1e-4)

    def test_volume_surge_calculation(self, in_memory_db):
        # 15 baseline days at 1x volume, then 5 days at 2x volume
        vol_mults = [1.0] * 15 + [2.0] * 5
        prices = _make_prices(100, [0.01] * 20, 1_000_000, vol_mults)
        _seed_movers_db(in_memory_db, [
            {"ticker": "TEST", "prices": prices},
        ])
        result = rank_movers(in_memory_db, days=5)
        # Avg period volume = 2M, baseline avg = (15*1M + 5*2M)/20 = 1.25M
        # surge = 2M / 1.25M = 1.6
        assert result.iloc[0]["volume_surge"] == pytest.approx(1.6, rel=1e-2)

    def test_sorted_descending_by_return(self, in_memory_db):
        configs = []
        for ticker, ret in [("UP", 0.05), ("FLAT", 0.0), ("DOWN", -0.03)]:
            returns = [0.0] * 15 + [ret] * 5
            prices = _make_prices(100, returns, 1_000_000, [1.0] * 20)
            configs.append({"ticker": ticker, "prices": prices})
        _seed_movers_db(in_memory_db, configs)
        result = rank_movers(in_memory_db, days=5)
        tickers = result["ticker"].tolist()
        assert tickers == ["UP", "FLAT", "DOWN"]

    def test_insufficient_data_excluded(self, in_memory_db):
        # Only 3 days of data, but days=5
        prices = _make_prices(100, [0.01] * 3, 1_000_000, [1.0] * 3)
        _seed_movers_db(in_memory_db, [
            {"ticker": "SHORT", "prices": prices},
        ])
        result = rank_movers(in_memory_db, days=5)
        assert result.empty

    def test_zero_volume_baseline(self, in_memory_db):
        prices = _make_prices(100, [0.01] * 20, 0, [0.0] * 20)
        _seed_movers_db(in_memory_db, [
            {"ticker": "ZERO", "prices": prices},
        ])
        result = rank_movers(in_memory_db, days=5)
        assert len(result) == 1
        assert pd.isna(result.iloc[0]["volume_surge"])

    def test_empty_db_returns_empty(self, in_memory_db):
        init_db(in_memory_db)
        result = rank_movers(in_memory_db, days=5)
        assert result.empty

    def test_joins_fundamentals(self, in_memory_db):
        prices = _make_prices(100, [0.01] * 20, 1_000_000, [1.0] * 20)
        _seed_movers_db(in_memory_db, [
            {"ticker": "AAPL", "name": "Apple Inc", "sector": "Technology",
             "market_cap": 3e12, "prices": prices},
        ])
        result = rank_movers(in_memory_db, days=5)
        row = result.iloc[0]
        assert row["name"] == "Apple Inc"
        assert row["sector"] == "Technology"
        assert row["market_cap"] == 3e12

    def test_custom_days_parameter(self, in_memory_db):
        # 20 days at 0%, then 10 days at +1%
        returns = [0.0] * 20 + [0.01] * 10
        prices = _make_prices(100, returns, 1_000_000, [1.0] * 30)
        _seed_movers_db(in_memory_db, [
            {"ticker": "TEST", "prices": prices},
        ])
        result = rank_movers(in_memory_db, days=10)
        assert len(result) == 1
        assert result.iloc[0]["return_pct"] == pytest.approx(1.01**9 - 1, rel=1e-4)

    def test_days_less_than_one_raises(self, in_memory_db):
        with pytest.raises(ValueError, match="days must be >= 1"):
            rank_movers(in_memory_db, days=0)


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------

class TestMoversCmd:
    TICKERS = [
        {
            "ticker": "GAIN", "name": "Gainer Corp", "sector": "Technology",
            "market_cap": 5e9,
            "prices": _make_prices(100, [0.0] * 15 + [0.03] * 5, 1_000_000,
                                   [1.0] * 15 + [2.0] * 5),
        },
        {
            "ticker": "LOSS", "name": "Loser Inc", "sector": "Healthcare",
            "market_cap": 2e9,
            "prices": _make_prices(100, [0.0] * 15 + [-0.02] * 5, 1_000_000,
                                   [1.0] * 20),
        },
    ]

    def _seed(self, conn):
        _seed_movers_db(conn, self.TICKERS)

    def test_basic_output(self, in_memory_db):
        self._seed(in_memory_db)
        runner = CliRunner()
        with patch("stock_trading.cli.db.get_connection", return_value=in_memory_db):
            result = runner.invoke(cli, ["movers", "--top", "5"])
        assert result.exit_code == 0
        assert "GAIN" in result.output
        assert "Return" in result.output

    def test_gainers_sort(self, in_memory_db):
        self._seed(in_memory_db)
        runner = CliRunner()
        with patch("stock_trading.cli.db.get_connection", return_value=in_memory_db):
            result = runner.invoke(cli, ["movers", "--sort", "gainers"])
        lines = [l for l in result.output.split("\n") if "GAIN" in l or "LOSS" in l]
        assert lines[0].index("GAIN") < len(lines[0])
        # GAIN should appear before LOSS (higher return)
        gain_rank = next(i for i, l in enumerate(lines) if "GAIN" in l)
        loss_rank = next(i for i, l in enumerate(lines) if "LOSS" in l)
        assert gain_rank < loss_rank

    def test_losers_sort(self, in_memory_db):
        self._seed(in_memory_db)
        runner = CliRunner()
        with patch("stock_trading.cli.db.get_connection", return_value=in_memory_db):
            result = runner.invoke(cli, ["movers", "--sort", "losers"])
        lines = [l for l in result.output.split("\n") if "GAIN" in l or "LOSS" in l]
        # LOSS should appear first (most negative return)
        loss_rank = next(i for i, l in enumerate(lines) if "LOSS" in l)
        gain_rank = next(i for i, l in enumerate(lines) if "GAIN" in l)
        assert loss_rank < gain_rank

    def test_sector_filter(self, in_memory_db):
        self._seed(in_memory_db)
        runner = CliRunner()
        with patch("stock_trading.cli.db.get_connection", return_value=in_memory_db):
            result = runner.invoke(cli, ["movers", "--sector", "Technology"])
        assert result.exit_code == 0
        assert "GAIN" in result.output
        assert "LOSS" not in result.output

    def test_min_volume_surge_filter(self, in_memory_db):
        self._seed(in_memory_db)
        runner = CliRunner()
        with patch("stock_trading.cli.db.get_connection", return_value=in_memory_db):
            result = runner.invoke(cli, ["movers", "--min-volume-surge", "1.3"])
        assert result.exit_code == 0
        # GAIN has 2x volume in period vs 1.25x baseline → surge ~1.6
        assert "GAIN" in result.output
        # LOSS has uniform volume → surge ~1.0
        assert "LOSS" not in result.output

    def test_empty_db(self, in_memory_db):
        init_db(in_memory_db)
        runner = CliRunner()
        with patch("stock_trading.cli.db.get_connection", return_value=in_memory_db):
            result = runner.invoke(cli, ["movers"])
        assert result.exit_code == 0
        assert "No results" in result.output
