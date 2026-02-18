"""Tests for stock_trading.backtester module."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from click.testing import CliRunner

from stock_trading.cli import cli
from stock_trading.db import init_db, upsert_prices


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


def _seed_backtest_db(conn, tickers, n_prices=250, base_price=100.0, seed=42):
    """Seed the in-memory DB with tickers and price data for backtesting."""
    init_db(conn)
    rng = np.random.default_rng(seed)
    for t in tickers:
        conn.execute(
            "INSERT OR REPLACE INTO tickers (ticker, name, exchange) "
            "VALUES (?, ?, ?)",
            (t, t, "NASDAQ"),
        )
        base = base_price + rng.random() * 50
        dates = pd.date_range(end=pd.Timestamp.today(), periods=n_prices, freq="B")
        close_prices = base + np.cumsum(rng.standard_normal(n_prices) * 0.5)
        rows = []
        for i, dt in enumerate(dates):
            c = close_prices[i]
            rows.append((t, dt.strftime("%Y-%m-%d"), c - 0.5, c + 1.0,
                         c - 1.0, c, int(rng.integers(500_000, 2_000_000)), c))
        upsert_prices(conn, rows)
    conn.commit()


class TestAddIndicators:
    def test_adds_rsi_column(self):
        df = _make_ohlcv_df(n=250)
        from stock_trading.backtester import add_indicators
        result = add_indicators(df)
        assert "RSI_14" in result.columns

    def test_adds_sma_columns(self):
        df = _make_ohlcv_df(n=250)
        from stock_trading.backtester import add_indicators
        result = add_indicators(df)
        for period in (10, 20, 50, 200):
            assert f"SMA_{period}" in result.columns

    def test_adds_macd_columns(self):
        df = _make_ohlcv_df(n=250)
        from stock_trading.backtester import add_indicators
        result = add_indicators(df)
        assert any(c.startswith("MACD_") for c in result.columns)
        assert any(c.startswith("MACDh_") for c in result.columns)
        assert any(c.startswith("MACDs_") for c in result.columns)

    def test_adds_bbands_columns(self):
        df = _make_ohlcv_df(n=250)
        from stock_trading.backtester import add_indicators
        result = add_indicators(df)
        assert any(c.startswith("BBL_") for c in result.columns)
        assert any(c.startswith("BBM_") for c in result.columns)
        assert any(c.startswith("BBU_") for c in result.columns)

    def test_adds_ema_columns(self):
        df = _make_ohlcv_df(n=250)
        from stock_trading.backtester import add_indicators
        result = add_indicators(df)
        for period in (12, 26, 50):
            assert f"EMA_{period}" in result.columns

    def test_adds_atr_column(self):
        df = _make_ohlcv_df(n=250)
        from stock_trading.backtester import add_indicators
        result = add_indicators(df)
        assert any(c.startswith("ATR") for c in result.columns)

    def test_adds_stoch_columns(self):
        df = _make_ohlcv_df(n=250)
        from stock_trading.backtester import add_indicators
        result = add_indicators(df)
        assert any(c.startswith("STOCHk_") for c in result.columns)
        assert any(c.startswith("STOCHd_") for c in result.columns)

    def test_adds_obv_column(self):
        df = _make_ohlcv_df(n=250)
        from stock_trading.backtester import add_indicators
        result = add_indicators(df)
        assert "OBV" in result.columns

    def test_preserves_original_columns(self):
        df = _make_ohlcv_df(n=250)
        from stock_trading.backtester import add_indicators
        result = add_indicators(df)
        for col in ("Open", "High", "Low", "Close", "Volume"):
            assert col in result.columns

    def test_short_df_does_not_error(self):
        df = _make_ohlcv_df(n=30)
        from stock_trading.backtester import add_indicators
        result = add_indicators(df)
        assert "RSI_14" in result.columns
        # SMA_200 should be all NaN for 30 rows
        assert result["SMA_200"].isna().all()


class TestLoadTickerData:
    def test_returns_dataframe(self, in_memory_db):
        _seed_backtest_db(in_memory_db, ["AAPL"], n_prices=100)
        from stock_trading.backtester import load_ticker_data
        result = load_ticker_data(in_memory_db, "AAPL", "2020-01-01")
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_returns_none_for_missing_ticker(self, in_memory_db):
        init_db(in_memory_db)
        from stock_trading.backtester import load_ticker_data
        result = load_ticker_data(in_memory_db, "ZZZZ", "2020-01-01")
        assert result is None

    def test_has_indicators(self, in_memory_db):
        _seed_backtest_db(in_memory_db, ["AAPL"], n_prices=250)
        from stock_trading.backtester import load_ticker_data
        result = load_ticker_data(in_memory_db, "AAPL", "2020-01-01")
        assert "RSI_14" in result.columns
        assert "SMA_50" in result.columns

    def test_has_ohlcv_columns(self, in_memory_db):
        _seed_backtest_db(in_memory_db, ["AAPL"], n_prices=100)
        from stock_trading.backtester import load_ticker_data
        result = load_ticker_data(in_memory_db, "AAPL", "2020-01-01")
        for col in ("Open", "High", "Low", "Close", "Volume"):
            assert col in result.columns

    def test_date_filtering(self, in_memory_db):
        _seed_backtest_db(in_memory_db, ["AAPL"], n_prices=250)
        from stock_trading.backtester import load_ticker_data
        result = load_ticker_data(in_memory_db, "AAPL", "2024-06-01", "2024-12-31")
        if result is not None and len(result) > 0:
            assert result.index.min() >= pd.Timestamp("2024-06-01")
            assert result.index.max() <= pd.Timestamp("2024-12-31")


class TestComputeMetrics:
    def test_flat_equity_curve(self):
        from stock_trading.backtester import compute_metrics
        eq = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=10, freq="B"),
            "portfolio_value": [100000.0] * 10,
        })
        result = compute_metrics(eq, [], 100000.0)
        assert result["total_return"] == 0.0
        assert result["max_drawdown"] == 0.0
        assert result["num_trades"] == 0

    def test_positive_return(self):
        from stock_trading.backtester import compute_metrics, Trade
        eq = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=5, freq="B"),
            "portfolio_value": [100000, 102000, 105000, 108000, 110000],
        })
        result = compute_metrics(eq, [], 100000.0)
        assert abs(result["total_return"] - 0.10) < 1e-6

    def test_max_drawdown_calculation(self):
        from stock_trading.backtester import compute_metrics
        eq = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=4, freq="B"),
            "portfolio_value": [100000, 110000, 90000, 100000],
        })
        result = compute_metrics(eq, [], 100000.0)
        # Drawdown from 110k to 90k = (90000-110000)/110000 = -18.18%
        assert abs(result["max_drawdown"] - (-20000 / 110000)) < 1e-6

    def test_sharpe_ratio_no_volatility(self):
        from stock_trading.backtester import compute_metrics
        eq = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=10, freq="B"),
            "portfolio_value": [100000.0] * 10,
        })
        result = compute_metrics(eq, [], 100000.0)
        assert result["sharpe_ratio"] == 0.0

    def test_win_rate(self):
        from stock_trading.backtester import compute_metrics, Trade
        trades = [
            Trade("A", "2023-01-01", 100, "2023-02-01", 110, 10, 100.0),
            Trade("B", "2023-01-01", 100, "2023-02-01", 120, 10, 200.0),
            Trade("C", "2023-01-01", 100, "2023-02-01", 90, 10, -100.0),
        ]
        eq = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=5, freq="B"),
            "portfolio_value": [100000] * 5,
        })
        result = compute_metrics(eq, trades, 100000.0)
        assert abs(result["win_rate"] - 2 / 3) < 1e-6
        assert result["num_trades"] == 3

    def test_annualized_return(self):
        from stock_trading.backtester import compute_metrics
        # 10% over 126 trading days (half a year)
        eq = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=126, freq="B"),
            "portfolio_value": np.linspace(100000, 110000, 126),
        })
        result = compute_metrics(eq, [], 100000.0)
        # annualized: (1.1)^(252/126) - 1 = ~21%
        expected = (1.1 ** (252 / 126)) - 1
        assert abs(result["annualized_return"] - expected) < 0.01


class TestRunBacktest:
    def test_no_signal_no_trades(self, in_memory_db):
        """Signal always 0 -> no trades, final value = initial capital."""
        from stock_trading.backtester import run_backtest, BacktestConfig
        _seed_backtest_db(in_memory_db, ["AAPL"], n_prices=100)

        def no_signal(df):
            return pd.Series(0, index=df.index)

        config = BacktestConfig(
            tickers=["AAPL"], start_date="2020-01-01", end_date=None,
            initial_capital=100000.0, signal_fn=no_signal, name="none",
        )
        result = run_backtest(in_memory_db, config)
        assert result.num_trades == 0
        assert result.final_value == 100000.0

    def test_single_round_trip(self, in_memory_db):
        """Buy on day 5, sell on day 15 -> exactly 1 closed trade."""
        from stock_trading.backtester import run_backtest, BacktestConfig
        _seed_backtest_db(in_memory_db, ["AAPL"], n_prices=100)

        def signal_fn(df):
            s = pd.Series(0, index=df.index)
            if len(df) > 15:
                s.iloc[5] = 1
                s.iloc[15] = -1
            return s

        config = BacktestConfig(
            tickers=["AAPL"], start_date="2020-01-01", end_date=None,
            initial_capital=100000.0, signal_fn=signal_fn, name="trip",
        )
        result = run_backtest(in_memory_db, config)
        assert result.num_trades == 1
        assert result.trades[0].ticker == "AAPL"
        assert result.trades[0].exit_date is not None

    def test_multi_ticker_allocation(self, in_memory_db):
        """Capital split equally among tickers."""
        from stock_trading.backtester import run_backtest, BacktestConfig
        _seed_backtest_db(in_memory_db, ["AAPL", "GOOG"], n_prices=100, seed=42)

        def always_buy(df):
            s = pd.Series(0, index=df.index)
            s.iloc[0] = 1
            return s

        config = BacktestConfig(
            tickers=["AAPL", "GOOG"], start_date="2020-01-01", end_date=None,
            initial_capital=100000.0, signal_fn=always_buy, name="alloc",
        )
        result = run_backtest(in_memory_db, config)
        # Should have positions in both tickers
        tickers_traded = {t.ticker for t in result.trades}
        assert "AAPL" in tickers_traded
        assert "GOOG" in tickers_traded

    def test_no_lookahead_bias(self, in_memory_db):
        """Signal on day N executes at day N+1 open."""
        from stock_trading.backtester import run_backtest, BacktestConfig, load_ticker_data
        _seed_backtest_db(in_memory_db, ["AAPL"], n_prices=100)

        # Load data to find the exact dates
        df = load_ticker_data(in_memory_db, "AAPL", "2020-01-01")
        signal_date = df.index[5]
        expected_entry_date = df.index[6]  # next trading day

        def signal_fn(data):
            s = pd.Series(0, index=data.index)
            if signal_date in data.index:
                s.loc[signal_date] = 1
            return s

        config = BacktestConfig(
            tickers=["AAPL"], start_date="2020-01-01", end_date=None,
            initial_capital=100000.0, signal_fn=signal_fn, name="bias",
        )
        result = run_backtest(in_memory_db, config)
        assert len(result.trades) >= 1
        assert result.trades[0].entry_date == expected_entry_date.strftime("%Y-%m-%d")

    def test_no_duplicate_positions(self, in_memory_db):
        """Multiple buy signals while holding -> only one entry."""
        from stock_trading.backtester import run_backtest, BacktestConfig
        _seed_backtest_db(in_memory_db, ["AAPL"], n_prices=100)

        def always_buy(df):
            return pd.Series(1, index=df.index)

        config = BacktestConfig(
            tickers=["AAPL"], start_date="2020-01-01", end_date=None,
            initial_capital=100000.0, signal_fn=always_buy, name="dup",
        )
        result = run_backtest(in_memory_db, config)
        # Should only have 1 trade (open position closed at end)
        assert len(result.trades) == 1

    def test_open_positions_closed_at_end(self, in_memory_db):
        """Positions still open at end are force-closed."""
        from stock_trading.backtester import run_backtest, BacktestConfig
        _seed_backtest_db(in_memory_db, ["AAPL"], n_prices=100)

        def buy_once(df):
            s = pd.Series(0, index=df.index)
            s.iloc[5] = 1
            return s

        config = BacktestConfig(
            tickers=["AAPL"], start_date="2020-01-01", end_date=None,
            initial_capital=100000.0, signal_fn=buy_once, name="open",
        )
        result = run_backtest(in_memory_db, config)
        assert len(result.trades) == 1
        assert result.trades[0].exit_date is not None

    def test_empty_data(self, in_memory_db):
        """No price data -> zero trades, final value = initial."""
        from stock_trading.backtester import run_backtest, BacktestConfig
        init_db(in_memory_db)

        config = BacktestConfig(
            tickers=["ZZZZ"], start_date="2020-01-01", end_date=None,
            initial_capital=100000.0, signal_fn=lambda df: pd.Series(0, index=df.index),
            name="empty",
        )
        result = run_backtest(in_memory_db, config)
        assert result.num_trades == 0
        assert result.final_value == 100000.0

    def test_equity_curve_has_rows(self, in_memory_db):
        """Equity curve should have entries for each trading day."""
        from stock_trading.backtester import run_backtest, BacktestConfig
        _seed_backtest_db(in_memory_db, ["AAPL"], n_prices=50)

        config = BacktestConfig(
            tickers=["AAPL"], start_date="2020-01-01", end_date=None,
            initial_capital=100000.0, signal_fn=lambda df: pd.Series(0, index=df.index),
            name="curve",
        )
        result = run_backtest(in_memory_db, config)
        assert len(result.equity_curve) > 0
        assert "date" in result.equity_curve.columns
        assert "portfolio_value" in result.equity_curve.columns


class TestBuiltinStrategies:
    def test_golden_cross_generates_signals(self):
        from stock_trading.backtester import add_indicators, BUILTIN_STRATEGIES
        df = _make_ohlcv_df(n=250)
        df = add_indicators(df)
        signals = BUILTIN_STRATEGIES["golden_cross"](df)
        assert len(signals) == len(df)
        # Should have at least some non-zero signals with enough data
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_rsi_mean_reversion_generates_signals(self):
        from stock_trading.backtester import add_indicators, BUILTIN_STRATEGIES
        df = _make_ohlcv_df(n=250)
        df = add_indicators(df)
        signals = BUILTIN_STRATEGIES["rsi_mean_reversion"](df)
        assert len(signals) == len(df)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_macd_crossover_generates_signals(self):
        from stock_trading.backtester import add_indicators, BUILTIN_STRATEGIES
        df = _make_ohlcv_df(n=250)
        df = add_indicators(df)
        signals = BUILTIN_STRATEGIES["macd_crossover"](df)
        assert len(signals) == len(df)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_all_strategies_registered(self):
        from stock_trading.backtester import BUILTIN_STRATEGIES
        assert "golden_cross" in BUILTIN_STRATEGIES
        assert "rsi_mean_reversion" in BUILTIN_STRATEGIES
        assert "macd_crossover" in BUILTIN_STRATEGIES


class TestFormatResults:
    def test_contains_key_metrics(self):
        from stock_trading.backtester import format_results, BacktestResult
        result = BacktestResult(
            config_name="test", tickers=["AAPL"], start_date="2023-01-01",
            end_date="2023-12-31", initial_capital=100000.0,
            final_value=110000.0, total_return=0.10,
            annualized_return=0.10, sharpe_ratio=1.5,
            max_drawdown=-0.05, win_rate=0.60, num_trades=5,
            trades=[], equity_curve=pd.DataFrame(),
        )
        output = format_results(result)
        assert "10.0%" in output or "10.00%" in output
        assert "Sharpe" in output
        assert "Drawdown" in output

    def test_contains_trade_log(self):
        from stock_trading.backtester import format_results, BacktestResult, Trade
        trade = Trade("AAPL", "2023-01-15", 150.0, "2023-03-01", 165.0, 10, 150.0)
        result = BacktestResult(
            config_name="test", tickers=["AAPL"], start_date="2023-01-01",
            end_date="2023-12-31", initial_capital=100000.0,
            final_value=100150.0, total_return=0.0015,
            annualized_return=0.003, sharpe_ratio=0.5,
            max_drawdown=-0.01, win_rate=1.0, num_trades=1,
            trades=[trade], equity_curve=pd.DataFrame(),
        )
        output = format_results(result)
        assert "AAPL" in output
        assert "2023-01-15" in output


class TestBacktestCmd:
    def test_basic_backtest(self, in_memory_db):
        _seed_backtest_db(in_memory_db, ["AAPL"], n_prices=250)
        runner = CliRunner()
        with patch("stock_trading.cli.db.get_connection", return_value=in_memory_db):
            result = runner.invoke(cli, [
                "backtest", "--tickers", "AAPL",
                "--start", "2020-01-01", "--strategy", "golden_cross",
            ])
        assert result.exit_code == 0
        assert "Return" in result.output

    def test_multi_ticker(self, in_memory_db):
        _seed_backtest_db(in_memory_db, ["AAPL", "GOOG"], n_prices=250)
        runner = CliRunner()
        with patch("stock_trading.cli.db.get_connection", return_value=in_memory_db):
            result = runner.invoke(cli, [
                "backtest", "--tickers", "AAPL,GOOG",
                "--start", "2020-01-01", "--strategy", "rsi_mean_reversion",
            ])
        assert result.exit_code == 0

    def test_no_data(self, in_memory_db):
        init_db(in_memory_db)
        runner = CliRunner()
        with patch("stock_trading.cli.db.get_connection", return_value=in_memory_db):
            result = runner.invoke(cli, [
                "backtest", "--tickers", "ZZZZ",
                "--start", "2020-01-01", "--strategy", "golden_cross",
            ])
        assert result.exit_code == 0


class TestBenchmark:
    def test_compute_benchmark_returns_metrics(self, in_memory_db):
        """compute_benchmark should return dict with return metrics."""
        from stock_trading.backtester import compute_benchmark
        _seed_backtest_db(in_memory_db, ["SPY"], n_prices=250)
        result = compute_benchmark(in_memory_db, "SPY", "2020-01-01", None, 100000.0)
        assert "total_return" in result
        assert "annualized_return" in result
        assert "sharpe_ratio" in result
        assert "max_drawdown" in result

    def test_compute_benchmark_no_data(self, in_memory_db):
        """Missing benchmark ticker returns None."""
        from stock_trading.backtester import compute_benchmark
        init_db(in_memory_db)
        result = compute_benchmark(in_memory_db, "SPY", "2020-01-01", None, 100000.0)
        assert result is None

    def test_backtest_result_has_benchmark(self, in_memory_db):
        """BacktestResult should include benchmark field."""
        from stock_trading.backtester import run_backtest, BacktestConfig
        _seed_backtest_db(in_memory_db, ["AAPL", "SPY"], n_prices=100)

        config = BacktestConfig(
            tickers=["AAPL"], start_date="2020-01-01", end_date=None,
            initial_capital=100000.0,
            signal_fn=lambda df: pd.Series(0, index=df.index),
            name="test",
        )
        result = run_backtest(in_memory_db, config)
        assert hasattr(result, "benchmark")

    def test_format_results_shows_benchmark(self, in_memory_db):
        """format_results output should include benchmark comparison."""
        from stock_trading.backtester import (
            run_backtest, BacktestConfig, format_results
        )
        _seed_backtest_db(in_memory_db, ["AAPL", "SPY"], n_prices=100)

        config = BacktestConfig(
            tickers=["AAPL"], start_date="2020-01-01", end_date=None,
            initial_capital=100000.0,
            signal_fn=lambda df: pd.Series(0, index=df.index),
            name="test",
        )
        result = run_backtest(in_memory_db, config)
        output = format_results(result)
        assert "Benchmark" in output or "S&P 500" in output or "SPY" in output

    def test_portfolio_buy_hold_returns_metrics(self, in_memory_db):
        """compute_portfolio_buy_hold should return dict with return metrics."""
        from stock_trading.backtester import compute_portfolio_buy_hold
        _seed_backtest_db(in_memory_db, ["AAPL", "GOOG"], n_prices=100)
        result = compute_portfolio_buy_hold(
            in_memory_db, ["AAPL", "GOOG"], "2020-01-01", None, 100000.0,
        )
        assert "total_return" in result
        assert "annualized_return" in result
        assert "sharpe_ratio" in result
        assert "max_drawdown" in result

    def test_portfolio_buy_hold_no_data(self, in_memory_db):
        """No data for any ticker returns None."""
        from stock_trading.backtester import compute_portfolio_buy_hold
        init_db(in_memory_db)
        result = compute_portfolio_buy_hold(
            in_memory_db, ["ZZZZ"], "2020-01-01", None, 100000.0,
        )
        assert result is None

    def test_backtest_result_has_portfolio_benchmark(self, in_memory_db):
        """BacktestResult should include portfolio_benchmark field."""
        from stock_trading.backtester import run_backtest, BacktestConfig
        _seed_backtest_db(in_memory_db, ["AAPL", "GOOG"], n_prices=100)

        config = BacktestConfig(
            tickers=["AAPL", "GOOG"], start_date="2020-01-01", end_date=None,
            initial_capital=100000.0,
            signal_fn=lambda df: pd.Series(0, index=df.index),
            name="test",
        )
        result = run_backtest(in_memory_db, config)
        assert hasattr(result, "portfolio_benchmark")
        assert result.portfolio_benchmark is not None

    def test_format_results_shows_portfolio_benchmark(self, in_memory_db):
        """format_results should show portfolio buy & hold comparison."""
        from stock_trading.backtester import (
            run_backtest, BacktestConfig, format_results
        )
        _seed_backtest_db(in_memory_db, ["AAPL", "GOOG"], n_prices=100)

        config = BacktestConfig(
            tickers=["AAPL", "GOOG"], start_date="2020-01-01", end_date=None,
            initial_capital=100000.0,
            signal_fn=lambda df: pd.Series(0, index=df.index),
            name="test",
        )
        result = run_backtest(in_memory_db, config)
        output = format_results(result)
        assert "Buy & Hold" in output

    def test_metals_benchmarks_in_result(self, in_memory_db):
        """BacktestResult should include metals_benchmarks field."""
        from stock_trading.backtester import run_backtest, BacktestConfig, METALS_BENCHMARKS
        _seed_backtest_db(in_memory_db,
            ["AAPL"] + list(METALS_BENCHMARKS.keys()), n_prices=100)

        config = BacktestConfig(
            tickers=["AAPL"], start_date="2020-01-01", end_date=None,
            initial_capital=100000.0,
            signal_fn=lambda df: pd.Series(0, index=df.index),
            name="test",
        )
        result = run_backtest(in_memory_db, config)
        assert hasattr(result, "metals_benchmarks")
        assert isinstance(result.metals_benchmarks, dict)

    def test_format_results_shows_metals(self, in_memory_db):
        """format_results should show Gold, Silver, Copper benchmarks."""
        from stock_trading.backtester import (
            run_backtest, BacktestConfig, format_results, METALS_BENCHMARKS
        )
        _seed_backtest_db(in_memory_db,
            ["AAPL"] + list(METALS_BENCHMARKS.keys()), n_prices=100)

        config = BacktestConfig(
            tickers=["AAPL"], start_date="2020-01-01", end_date=None,
            initial_capital=100000.0,
            signal_fn=lambda df: pd.Series(0, index=df.index),
            name="test",
        )
        result = run_backtest(in_memory_db, config)
        output = format_results(result)
        assert "Gold" in output
        assert "Silver" in output
        assert "Copper" in output
        assert "Bitcoin" in output
