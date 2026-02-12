"""Tests for stock_trading.screener module."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from stock_trading.cli import cli
from stock_trading.db import init_db, upsert_prices
from stock_trading.screener import (
    _detect_swings,
    _score_bbands,
    _score_ma_crossover,
    _score_macd,
    _score_momentum,
    _score_rsi,
    compute_fundamental_scores,
    compute_technical_score,
    score_universe,
)


def _make_ohlcv_df(n=250, base_price=100.0, seed=42):
    """Generate a synthetic OHLCV DataFrame for testing."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = base_price + np.cumsum(rng.standard_normal(n) * 0.5)
    return pd.DataFrame({
        "Open": close - 0.5,
        "High": close + 1.0,
        "Low": close - 1.0,
        "Close": close,
        "Volume": rng.integers(500_000, 2_000_000, n),
    }, index=dates)


def _seed_db(conn, tickers_data, n_prices=250):
    """Seed the in-memory DB with tickers and price data."""
    init_db(conn)
    rng = np.random.default_rng(42)
    for t in tickers_data:
        conn.execute(
            "INSERT OR REPLACE INTO tickers "
            "(ticker, name, exchange, sector, industry, market_cap, "
            "trailing_pe, forward_pe, dividend_yield, beta, "
            "target_median, target_high, target_low, num_analysts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                t["ticker"], t.get("name", t["ticker"]),
                t.get("exchange", "NASDAQ"), t.get("sector", "Technology"),
                t.get("industry", "Software"), t.get("market_cap", 1e9),
                t.get("trailing_pe", 20.0), t.get("forward_pe", 18.0),
                t.get("dividend_yield", 0.01), t.get("beta", 1.0),
                t.get("target_median"), t.get("target_high"),
                t.get("target_low"), t.get("num_analysts"),
            ),
        )
        base = 100.0 + rng.random() * 50
        rows = []
        dates = pd.date_range(end=pd.Timestamp.today(), periods=n_prices, freq="B")
        close_prices = base + np.cumsum(rng.standard_normal(n_prices) * 0.5)
        for i, dt in enumerate(dates):
            c = close_prices[i]
            rows.append((
                t["ticker"], dt.strftime("%Y-%m-%d"),
                c - 0.5, c + 1.0, c - 1.0, c,
                int(rng.integers(500_000, 2_000_000)), c,
            ))
        upsert_prices(conn, rows)
    conn.commit()


class TestScoreRsi:
    def test_oversold(self):
        assert _score_rsi(25.0) == 1.0

    def test_slightly_oversold(self):
        assert _score_rsi(40.0) == 0.5

    def test_neutral(self):
        assert _score_rsi(50.0) == 0.0

    def test_slightly_overbought(self):
        assert _score_rsi(65.0) == -0.5

    def test_overbought(self):
        assert _score_rsi(75.0) == -1.0

    def test_nan_returns_zero(self):
        assert _score_rsi(float("nan")) == 0.0

    def test_boundary_30(self):
        assert _score_rsi(30.0) == 0.5

    def test_boundary_55(self):
        assert _score_rsi(55.0) == 0.0

    def test_boundary_70(self):
        assert _score_rsi(70.0) == -0.5


class TestScoreMacd:
    def test_bullish_crossover(self):
        assert _score_macd(1.0, 0.5, 0.5, -0.1) == 1.0

    def test_bearish_crossover(self):
        assert _score_macd(-0.5, 0.1, -0.6, 0.1) == -1.0

    def test_growing_histogram(self):
        assert _score_macd(1.0, 0.5, 0.5, 0.3) == 0.5

    def test_shrinking_histogram(self):
        assert _score_macd(-1.0, -0.5, -0.5, -0.3) == -0.5

    def test_nan_returns_zero(self):
        assert _score_macd(float("nan"), 0.5, 0.5, 0.3) == 0.0

    def test_neutral_histogram(self):
        assert _score_macd(1.0, 0.5, 0.5, 0.5) == 0.0


class TestScoreMaCrossover:
    def test_golden_cross_price_above(self):
        assert _score_ma_crossover(110, 105, 100) == 1.0

    def test_golden_cross_price_below(self):
        assert _score_ma_crossover(100, 105, 100) == -0.25

    def test_death_cross_price_above_sma50(self):
        assert _score_ma_crossover(100, 95, 100) == 0.5

    def test_death_cross_full(self):
        assert _score_ma_crossover(90, 95, 100) == -1.0

    def test_nan_returns_zero(self):
        assert _score_ma_crossover(float("nan"), 105, 100) == 0.0


class TestScoreBbands:
    def test_below_lower_band(self):
        assert _score_bbands(95, 100, 110, 120) == 1.0

    def test_near_lower_band(self):
        assert _score_bbands(102, 100, 110, 120) == 0.5

    def test_middle(self):
        assert _score_bbands(110, 100, 110, 120) == 0.0

    def test_near_upper(self):
        assert _score_bbands(118, 100, 110, 120) == -0.5

    def test_above_upper(self):
        assert _score_bbands(125, 100, 110, 120) == -1.0

    def test_nan_returns_zero(self):
        assert _score_bbands(float("nan"), 100, 110, 120) == 0.0

    def test_equal_bands_returns_zero(self):
        assert _score_bbands(100, 100, 100, 100) == 0.0


class TestDetectSwings:
    def test_finds_swing_highs_and_lows(self):
        # Create data with clear peaks and valleys
        n = 50
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        # Sine wave creates predictable peaks/valleys
        t = np.linspace(0, 4 * np.pi, n)
        prices = 100 + 10 * np.sin(t)
        highs = pd.Series(prices + 1, index=dates)
        lows = pd.Series(prices - 1, index=dates)

        swing_highs, swing_lows = _detect_swings(highs, lows, window=3)
        assert len(swing_highs) > 0
        assert len(swing_lows) > 0

    def test_returns_empty_for_short_data(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        highs = pd.Series([101, 102, 103, 102, 101], index=dates)
        lows = pd.Series([99, 100, 101, 100, 99], index=dates)
        swing_highs, swing_lows = _detect_swings(highs, lows, window=5)
        assert swing_highs == []
        assert swing_lows == []

    def test_flat_data(self):
        n = 30
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        highs = pd.Series([100.0] * n, index=dates)
        lows = pd.Series([99.0] * n, index=dates)
        swing_highs, swing_lows = _detect_swings(highs, lows, window=5)
        # All bars tie for max/min so all qualify as swings in flat data
        assert isinstance(swing_highs, list)
        assert isinstance(swing_lows, list)


class TestScoreMomentum:
    def _make_trending_up(self, n=200):
        """Create data with clear higher highs and higher lows."""
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        base = np.linspace(100, 150, n)  # steadily rising
        noise = np.sin(np.linspace(0, 16 * np.pi, n)) * 5
        prices = base + noise
        return pd.Series(prices + 1, index=dates), pd.Series(prices - 1, index=dates)

    def _make_trending_down(self, n=200):
        """Create data with clear lower highs and lower lows."""
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        base = np.linspace(150, 100, n)  # steadily falling
        noise = np.sin(np.linspace(0, 16 * np.pi, n)) * 5
        prices = base + noise
        return pd.Series(prices + 1, index=dates), pd.Series(prices - 1, index=dates)

    def test_uptrend_positive_score(self):
        highs, lows = self._make_trending_up()
        score = _score_momentum(highs, lows)
        assert score > 0.0

    def test_downtrend_negative_score(self):
        highs, lows = self._make_trending_down()
        score = _score_momentum(highs, lows)
        assert score < 0.0

    def test_score_in_range(self):
        highs, lows = self._make_trending_up()
        score = _score_momentum(highs, lows)
        assert -1.0 <= score <= 1.0

    def test_insufficient_data_returns_zero(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        highs = pd.Series([101, 102, 103, 102, 101], index=dates)
        lows = pd.Series([99, 100, 101, 100, 99], index=dates)
        score = _score_momentum(highs, lows, window=5)
        assert score == 0.0


class TestComputeTechnicalScore:
    def test_returns_dict_with_expected_keys(self):
        df = _make_ohlcv_df(n=250)
        result = compute_technical_score(df)
        assert result is not None
        for key in ("price", "rsi_value", "rsi_score", "macd_score",
                     "ma_crossover_score", "bbands_score", "momentum_score",
                     "technical_score"):
            assert key in result

    def test_technical_score_in_range(self):
        df = _make_ohlcv_df(n=250)
        result = compute_technical_score(df)
        assert 0.0 <= result["technical_score"] <= 1.0

    def test_returns_none_for_short_df(self):
        df = _make_ohlcv_df(n=10)
        assert compute_technical_score(df) is None

    def test_returns_none_for_none_input(self):
        assert compute_technical_score(None) is None

    def test_works_with_minimum_rows(self):
        df = _make_ohlcv_df(n=26)
        result = compute_technical_score(df)
        assert result is not None
        assert 0.0 <= result["technical_score"] <= 1.0


class TestComputeFundamentalScores:
    def test_scores_in_range(self):
        df = pd.DataFrame({
            "ticker": ["A", "B", "C"],
            "trailing_pe": [15.0, 25.0, 35.0],
            "forward_pe": [14.0, 22.0, np.nan],
            "dividend_yield": [0.02, 0.01, 0.03],
            "beta": [1.0, 1.5, 0.8],
            "market_cap": [1e10, 5e10, 1e11],
        })
        result = compute_fundamental_scores(df)
        assert all(0.0 <= s <= 1.0 for s in result["fundamental_score"])

    def test_negative_pe_gets_zero_score(self):
        df = pd.DataFrame({
            "ticker": ["A", "B"],
            "trailing_pe": [-5.0, 20.0],
            "forward_pe": [np.nan, 18.0],
            "dividend_yield": [0.01, 0.02],
            "beta": [1.0, 1.0],
            "market_cap": [1e9, 1e10],
        })
        result = compute_fundamental_scores(df)
        assert result.loc[result["ticker"] == "A", "pe_score"].iloc[0] == 0.0

    def test_forward_pe_preferred_over_trailing(self):
        df = pd.DataFrame({
            "ticker": ["A"],
            "trailing_pe": [30.0],
            "forward_pe": [15.0],
            "dividend_yield": [0.01],
            "beta": [1.0],
            "market_cap": [1e9],
        })
        result = compute_fundamental_scores(df)
        assert result["pe"].iloc[0] == 15.0

    def test_missing_fundamentals_get_neutral_scores(self):
        df = pd.DataFrame({
            "ticker": ["A"],
            "trailing_pe": [np.nan],
            "forward_pe": [np.nan],
            "dividend_yield": [np.nan],
            "beta": [np.nan],
            "market_cap": [np.nan],
        })
        result = compute_fundamental_scores(df)
        assert 0.0 <= result["fundamental_score"].iloc[0] <= 1.0


class TestScoreUniverse:
    def test_returns_sorted_dataframe(self, in_memory_db):
        tickers_data = [
            {"ticker": "AAPL", "name": "Apple", "sector": "Technology",
             "market_cap": 3e12, "trailing_pe": 28.0, "forward_pe": 25.0,
             "dividend_yield": 0.005, "beta": 1.2,
             "target_median": 250.0, "target_high": 280.0, "target_low": 200.0,
             "num_analysts": 30},
            {"ticker": "GOOG", "name": "Alphabet", "sector": "Communication",
             "market_cap": 2e12, "trailing_pe": 22.0, "forward_pe": 20.0,
             "dividend_yield": 0.0, "beta": 1.1,
             "target_median": 200.0, "target_high": 230.0, "target_low": 170.0,
             "num_analysts": 25},
            {"ticker": "JNJ", "name": "J&J", "sector": "Healthcare",
             "market_cap": 4e11, "trailing_pe": 15.0, "forward_pe": 14.0,
             "dividend_yield": 0.025, "beta": 0.7,
             "target_median": 180.0, "target_high": 200.0, "target_low": 150.0,
             "num_analysts": 20},
        ]
        _seed_db(in_memory_db, tickers_data, n_prices=250)
        result = score_universe(in_memory_db, lookback_days=400)
        assert not result.empty
        assert len(result) == 3
        scores = result["composite_score"].tolist()
        assert scores == sorted(scores, reverse=True)
        for col in ("ticker", "composite_score", "technical_score", "fundamental_score"):
            assert col in result.columns

    def test_empty_db_returns_empty(self, in_memory_db):
        init_db(in_memory_db)
        result = score_universe(in_memory_db)
        assert result.empty

    def test_custom_weights(self, in_memory_db):
        tickers_data = [
            {"ticker": "AAPL", "name": "Apple", "market_cap": 3e12,
             "trailing_pe": 28.0, "forward_pe": 25.0,
             "dividend_yield": 0.005, "beta": 1.2},
        ]
        _seed_db(in_memory_db, tickers_data, n_prices=250)
        result = score_universe(
            in_memory_db, lookback_days=400,
            weights={"technical": 1.0, "fundamental": 0.0},
        )
        assert not result.empty
        row = result.iloc[0]
        assert abs(row["composite_score"] - row["technical_score"]) < 1e-6


class TestRecommendCmd:
    TICKERS_DATA = [
        {"ticker": "AAPL", "name": "Apple", "sector": "Technology",
         "market_cap": 3e12, "trailing_pe": 28.0, "forward_pe": 25.0,
         "dividend_yield": 0.005, "beta": 1.2,
         "target_median": 250.0, "target_high": 280.0, "target_low": 200.0,
         "num_analysts": 30},
        {"ticker": "GOOG", "name": "Alphabet", "sector": "Communication",
         "market_cap": 2e12, "trailing_pe": 22.0, "forward_pe": 20.0,
         "dividend_yield": 0.0, "beta": 1.1,
         "target_median": 200.0, "target_high": 230.0, "target_low": 170.0,
         "num_analysts": 25},
    ]

    def _seed(self, conn):
        _seed_db(conn, self.TICKERS_DATA, n_prices=250)

    def test_basic_output(self, in_memory_db):
        self._seed(in_memory_db)
        runner = CliRunner()
        with patch("stock_trading.cli.db.get_connection", return_value=in_memory_db):
            result = runner.invoke(cli, ["recommend", "--top", "5"])
        assert result.exit_code == 0
        assert "AAPL" in result.output or "GOOG" in result.output
        assert "Composite" in result.output

    def test_sector_filter(self, in_memory_db):
        self._seed(in_memory_db)
        runner = CliRunner()
        with patch("stock_trading.cli.db.get_connection", return_value=in_memory_db):
            result = runner.invoke(cli, ["recommend", "--sector", "Technology"])
        assert result.exit_code == 0
        assert "AAPL" in result.output
        # GOOG is Communication, should be filtered out
        assert "GOOG" not in result.output

    def test_empty_db(self, in_memory_db):
        init_db(in_memory_db)
        runner = CliRunner()
        with patch("stock_trading.cli.db.get_connection", return_value=in_memory_db):
            result = runner.invoke(cli, ["recommend"])
        assert result.exit_code == 0
        assert "No results" in result.output

    def test_custom_weights(self, in_memory_db):
        self._seed(in_memory_db)
        runner = CliRunner()
        with patch("stock_trading.cli.db.get_connection", return_value=in_memory_db):
            result = runner.invoke(cli, [
                "recommend",
                "--technical-weight", "0.8",
                "--fundamental-weight", "0.2",
            ])
        assert result.exit_code == 0
        assert "80%" in result.output

    def test_min_market_cap_filter(self, in_memory_db):
        tickers = [
            {"ticker": "BIG", "name": "Big Corp", "sector": "Tech",
             "market_cap": 5e12, "trailing_pe": 20.0, "forward_pe": 18.0,
             "dividend_yield": 0.01, "beta": 1.0},
            {"ticker": "TINY", "name": "Tiny Inc", "sector": "Tech",
             "market_cap": 1e6, "trailing_pe": 10.0, "forward_pe": 9.0,
             "dividend_yield": 0.02, "beta": 0.8},
        ]
        _seed_db(in_memory_db, tickers, n_prices=250)
        runner = CliRunner()
        with patch("stock_trading.cli.db.get_connection", return_value=in_memory_db):
            result = runner.invoke(cli, ["recommend", "--min-market-cap", "1e9"])
        assert result.exit_code == 0
        assert "BIG" in result.output
        assert "TINY" not in result.output


class TestTargetUpsideScore:
    def test_higher_upside_gets_higher_score(self):
        """Ticker with 50% upside should score higher than 10% upside."""
        df = pd.DataFrame({
            "ticker": ["A", "B", "C"],
            "trailing_pe": [20.0, 20.0, 20.0],
            "forward_pe": [18.0, 18.0, 18.0],
            "dividend_yield": [0.01, 0.01, 0.01],
            "beta": [1.0, 1.0, 1.0],
            "market_cap": [1e10, 1e10, 1e10],
            "price": [100.0, 100.0, 100.0],
            "target_median": [150.0, 110.0, 90.0],
        })
        result = compute_fundamental_scores(df)
        # A has 50% upside, B has 10%, C has -10%
        assert result.loc[0, "target_upside_score"] > result.loc[1, "target_upside_score"]
        assert result.loc[1, "target_upside_score"] > result.loc[2, "target_upside_score"]

    def test_missing_target_gets_zero_score(self):
        """Tickers without target_median should get 0.0 upside score."""
        df = pd.DataFrame({
            "ticker": ["A", "B"],
            "trailing_pe": [20.0, 20.0],
            "forward_pe": [18.0, 18.0],
            "dividend_yield": [0.01, 0.01],
            "beta": [1.0, 1.0],
            "market_cap": [1e10, 1e10],
            "price": [100.0, 100.0],
            "target_median": [150.0, np.nan],
        })
        result = compute_fundamental_scores(df)
        assert result.loc[1, "target_upside_score"] == 0.0

    def test_missing_price_gets_zero_score(self):
        """Tickers without price should get 0.0 upside score."""
        df = pd.DataFrame({
            "ticker": ["A"],
            "trailing_pe": [20.0],
            "forward_pe": [18.0],
            "dividend_yield": [0.01],
            "beta": [1.0],
            "market_cap": [1e10],
            "price": [np.nan],
            "target_median": [150.0],
        })
        result = compute_fundamental_scores(df)
        assert result.loc[0, "target_upside_score"] == 0.0

    def test_no_price_column_backward_compat(self):
        """Function should work without price column."""
        df = pd.DataFrame({
            "ticker": ["A"],
            "trailing_pe": [20.0],
            "forward_pe": [18.0],
            "dividend_yield": [0.01],
            "beta": [1.0],
            "market_cap": [1e10],
        })
        result = compute_fundamental_scores(df)
        assert result.loc[0, "target_upside_score"] == 0.0
        assert 0.0 <= result.loc[0, "fundamental_score"] <= 1.0

    def test_zero_price_treated_as_nan(self):
        """Price of 0 should not cause division error."""
        df = pd.DataFrame({
            "ticker": ["A"],
            "trailing_pe": [20.0],
            "forward_pe": [18.0],
            "dividend_yield": [0.01],
            "beta": [1.0],
            "market_cap": [1e10],
            "price": [0.0],
            "target_median": [150.0],
        })
        result = compute_fundamental_scores(df)
        assert result.loc[0, "target_upside_score"] == 0.0
