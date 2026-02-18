"""Backtesting framework for signal-based trading strategies."""

import dataclasses
import logging
from math import sqrt
from typing import Callable

import numpy as np
import pandas as pd
import pandas_ta as ta

from stock_trading import db
from stock_trading.charting import rows_to_dataframe

logger = logging.getLogger(__name__)

METALS_BENCHMARKS = {
    "GLD": "Gold",
    "SLV": "Silver",
    "CPER": "Copper",
}


@dataclasses.dataclass
class BacktestConfig:
    """Configuration for a backtest run."""

    tickers: list
    start_date: str
    end_date: str | None
    initial_capital: float
    signal_fn: Callable[[pd.DataFrame], pd.Series]
    name: str = "custom"


@dataclasses.dataclass
class Trade:
    """A single trade (entry and optional exit)."""

    ticker: str
    entry_date: str
    entry_price: float
    exit_date: str | None
    exit_price: float | None
    shares: int
    pnl: float | None

    @property
    def return_pct(self):
        if self.exit_price is None or self.entry_price == 0:
            return None
        return (self.exit_price - self.entry_price) / self.entry_price


@dataclasses.dataclass
class BacktestResult:
    """Results from a completed backtest."""

    config_name: str
    tickers: list
    start_date: str
    end_date: str
    initial_capital: float
    final_value: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    num_trades: int
    trades: list
    equity_curve: pd.DataFrame
    benchmark: dict | None = None
    portfolio_benchmark: dict | None = None
    metals_benchmarks: dict = dataclasses.field(default_factory=dict)


def add_indicators(df):
    """Add technical indicators to an OHLCV DataFrame.

    Adds RSI, MACD, SMA, EMA, BBands, ATR, ADX, Stochastic, and OBV.
    """
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # RSI
    rsi = ta.rsi(close, length=14)
    if rsi is not None:
        df["RSI_14"] = rsi

    # MACD
    macd = ta.macd(close, fast=12, slow=26, signal=9)
    if macd is not None:
        df = pd.concat([df, macd], axis=1)

    # SMA
    for period in (10, 20, 50, 200):
        sma = ta.sma(close, length=period)
        df[f"SMA_{period}"] = sma if sma is not None else np.nan

    # EMA
    for period in (12, 26, 50):
        ema = ta.ema(close, length=period)
        df[f"EMA_{period}"] = ema if ema is not None else np.nan

    # Bollinger Bands
    bbands = ta.bbands(close, length=20)
    if bbands is not None:
        df = pd.concat([df, bbands], axis=1)

    # ATR
    atr = ta.atr(high, low, close, length=14)
    if atr is not None:
        df[atr.name] = atr

    # ADX
    adx = ta.adx(high, low, close, length=14)
    if adx is not None:
        df = pd.concat([df, adx], axis=1)

    # Stochastic
    stoch = ta.stoch(high, low, close, k=14, d=3, smooth_k=3)
    if stoch is not None:
        df = pd.concat([df, stoch], axis=1)

    # OBV
    obv = ta.obv(close, volume)
    if obv is not None:
        df["OBV"] = obv

    return df


def load_ticker_data(conn, ticker, start_date, end_date=None):
    """Load price data and add technical indicators.

    Returns None if no data found.
    """
    rows = db.query_prices(conn, ticker, start_date, end_date)
    df = rows_to_dataframe(rows)
    if df is None:
        return None
    return add_indicators(df)


def compute_benchmark(conn, ticker, start_date, end_date, initial_capital):
    """Compute buy-and-hold benchmark returns for a ticker (e.g. SPY).

    Returns a dict with total_return, annualized_return, sharpe_ratio,
    max_drawdown, or None if no data found.
    """
    rows = db.query_prices(conn, ticker, start_date, end_date)
    df = rows_to_dataframe(rows)
    if df is None or len(df) < 2:
        return None

    close = df["Close"]
    shares = int(initial_capital // close.iloc[0])
    if shares == 0:
        return None

    # Build equity curve for buy-and-hold
    equity = shares * close
    cash_remainder = initial_capital - (shares * close.iloc[0])
    equity = equity + cash_remainder

    eq_df = pd.DataFrame({
        "date": df.index,
        "portfolio_value": equity.values,
    })

    final_value = equity.iloc[-1]
    total_return = (final_value - initial_capital) / initial_capital

    num_days = len(close)
    if total_return > -1 and num_days > 1:
        annualized_return = (1 + total_return) ** (252 / num_days) - 1
    else:
        annualized_return = 0.0

    daily_returns = close.pct_change().dropna()
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * sqrt(252)
    else:
        sharpe_ratio = 0.0

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_drawdown = drawdown.min()

    return {
        "ticker": ticker,
        "total_return": total_return,
        "annualized_return": annualized_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "final_value": final_value,
    }


def compute_portfolio_buy_hold(conn, tickers, start_date, end_date, initial_capital):
    """Compute equal-weight buy-and-hold returns across a list of tickers.

    Returns a dict with metrics, or None if no data found.
    """
    capital_per_ticker = initial_capital / len(tickers)
    ticker_dfs = {}
    for ticker in tickers:
        rows = db.query_prices(conn, ticker, start_date, end_date)
        df = rows_to_dataframe(rows)
        if df is not None and len(df) >= 2:
            ticker_dfs[ticker] = df

    if not ticker_dfs:
        return None

    # Build unified date index
    all_dates = sorted(set().union(*(df.index for df in ticker_dfs.values())))

    # For each ticker, buy as many whole shares as possible on first available day
    holdings = {}  # ticker -> (shares, cost_basis, last_close)
    cash = initial_capital
    equity_rows = []

    for date in all_dates:
        # Buy on first day each ticker appears
        for ticker, df in ticker_dfs.items():
            if ticker not in holdings and date in df.index:
                price = df.loc[date, "Open"]
                shares = int(capital_per_ticker // price)
                if shares > 0:
                    cash -= shares * price
                    holdings[ticker] = shares

        # Mark to market
        portfolio_value = cash
        for ticker, shares in holdings.items():
            df = ticker_dfs[ticker]
            if date in df.index:
                portfolio_value += shares * df.loc[date, "Close"]
            else:
                # Use last known close from the DataFrame up to this date
                before = df.index[df.index <= date]
                if len(before) > 0:
                    portfolio_value += shares * df.loc[before[-1], "Close"]

        equity_rows.append({"date": date, "portfolio_value": portfolio_value})

    if not equity_rows:
        return None

    eq = pd.DataFrame(equity_rows)
    values = eq["portfolio_value"]
    final_value = values.iloc[-1]
    total_return = (final_value - initial_capital) / initial_capital

    num_days = len(values)
    if total_return > -1 and num_days > 1:
        annualized_return = (1 + total_return) ** (252 / num_days) - 1
    else:
        annualized_return = 0.0

    daily_returns = values.pct_change().dropna()
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * sqrt(252)
    else:
        sharpe_ratio = 0.0

    running_max = values.cummax()
    drawdown = (values - running_max) / running_max
    max_drawdown = drawdown.min()

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "final_value": final_value,
    }


def compute_metrics(equity_curve, trades, initial_capital):
    """Compute performance metrics from equity curve and trade list."""
    values = equity_curve["portfolio_value"]
    final_value = values.iloc[-1]
    total_return = (final_value - initial_capital) / initial_capital

    num_days = len(values)
    if total_return > -1 and num_days > 1:
        annualized_return = (1 + total_return) ** (252 / num_days) - 1
    else:
        annualized_return = 0.0

    daily_returns = values.pct_change().dropna()
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * sqrt(252)
    else:
        sharpe_ratio = 0.0

    running_max = values.cummax()
    drawdown = (values - running_max) / running_max
    max_drawdown = drawdown.min()

    winning = sum(1 for t in trades if t.pnl is not None and t.pnl > 0)
    win_rate = winning / len(trades) if trades else 0.0

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "num_trades": len(trades),
    }


def run_backtest(conn, config):
    """Execute a backtest simulation across multiple tickers."""
    # Load data and compute signals for each ticker
    ticker_data = {}
    ticker_signals = {}
    for ticker in config.tickers:
        df = load_ticker_data(conn, ticker, config.start_date, config.end_date)
        if df is None or len(df) < 2:
            logger.warning("Skipping %s: insufficient data", ticker)
            continue
        try:
            signals = config.signal_fn(df)
        except Exception:
            logger.warning("Signal function failed for %s", ticker, exc_info=True)
            continue
        ticker_data[ticker] = df
        ticker_signals[ticker] = signals

    # Handle no data case
    if not ticker_data:
        return BacktestResult(
            config_name=config.name, tickers=config.tickers,
            start_date=config.start_date, end_date=config.end_date or "",
            initial_capital=config.initial_capital,
            final_value=config.initial_capital,
            total_return=0.0, annualized_return=0.0, sharpe_ratio=0.0,
            max_drawdown=0.0, win_rate=0.0, num_trades=0,
            trades=[],
            equity_curve=pd.DataFrame({"date": [], "portfolio_value": []}),
        )

    # Build unified date index
    all_dates = sorted(set().union(*(df.index for df in ticker_data.values())))

    # Simulation state
    cash = config.initial_capital
    capital_per_ticker = config.initial_capital / len(ticker_data)
    positions = {}  # ticker -> Trade
    closed_trades = []
    equity_rows = []
    last_close = {}  # ticker -> last known close price

    for i, date in enumerate(all_dates):
        date_str = date.strftime("%Y-%m-%d")

        for ticker, df in ticker_data.items():
            signals = ticker_signals[ticker]

            # Update last known close
            if date in df.index:
                last_close[ticker] = df.loc[date, "Close"]

            # Find previous date in this ticker's data for signal
            ticker_dates = df.index
            date_pos = ticker_dates.get_loc(date) if date in ticker_dates else None
            if date_pos is None or date_pos == 0:
                continue

            prev_date = ticker_dates[date_pos - 1]
            signal = signals.loc[prev_date] if prev_date in signals.index else 0

            if date not in df.index:
                continue
            open_price = df.loc[date, "Open"]

            if signal == 1 and ticker not in positions:
                shares = int(capital_per_ticker // open_price)
                cost = shares * open_price
                if shares > 0 and cash >= cost:
                    cash -= cost
                    positions[ticker] = Trade(
                        ticker=ticker, entry_date=date_str,
                        entry_price=open_price, exit_date=None,
                        exit_price=None, shares=shares, pnl=None,
                    )

            elif signal == -1 and ticker in positions:
                trade = positions.pop(ticker)
                proceeds = trade.shares * open_price
                cash += proceeds
                trade.exit_date = date_str
                trade.exit_price = open_price
                trade.pnl = proceeds - (trade.shares * trade.entry_price)
                closed_trades.append(trade)

        # Mark to market
        portfolio_value = cash
        for ticker, trade in positions.items():
            portfolio_value += trade.shares * last_close.get(ticker, trade.entry_price)
        equity_rows.append({"date": date_str, "portfolio_value": portfolio_value})

    # Close open positions at last known price
    for ticker in list(positions.keys()):
        trade = positions.pop(ticker)
        close_price = last_close.get(ticker, trade.entry_price)
        trade.exit_date = equity_rows[-1]["date"] if equity_rows else trade.entry_date
        trade.exit_price = close_price
        trade.pnl = (trade.shares * close_price) - (trade.shares * trade.entry_price)
        closed_trades.append(trade)

    equity_curve = pd.DataFrame(equity_rows)
    if equity_curve.empty:
        final_value = config.initial_capital
        end_date = config.end_date or config.start_date
    else:
        final_value = equity_curve["portfolio_value"].iloc[-1]
        end_date = equity_curve["date"].iloc[-1]

    metrics = compute_metrics(equity_curve, closed_trades, config.initial_capital) \
        if not equity_curve.empty else {
            "total_return": 0.0, "annualized_return": 0.0, "sharpe_ratio": 0.0,
            "max_drawdown": 0.0, "win_rate": 0.0, "num_trades": 0,
        }

    # Compute S&P 500 benchmark
    benchmark = compute_benchmark(
        conn, "SPY", config.start_date, config.end_date, config.initial_capital,
    )

    # Compute portfolio buy-and-hold benchmark (same tickers)
    portfolio_benchmark = compute_portfolio_buy_hold(
        conn, config.tickers, config.start_date, config.end_date,
        config.initial_capital,
    )

    # Compute metals benchmarks
    metals = {}
    for ticker, label in METALS_BENCHMARKS.items():
        bm = compute_benchmark(
            conn, ticker, config.start_date, config.end_date,
            config.initial_capital,
        )
        if bm is not None:
            metals[label] = bm

    return BacktestResult(
        config_name=config.name, tickers=config.tickers,
        start_date=config.start_date, end_date=end_date,
        initial_capital=config.initial_capital, final_value=final_value,
        total_return=metrics["total_return"],
        annualized_return=metrics["annualized_return"],
        sharpe_ratio=metrics["sharpe_ratio"],
        max_drawdown=metrics["max_drawdown"],
        win_rate=metrics["win_rate"],
        num_trades=metrics["num_trades"],
        trades=closed_trades, equity_curve=equity_curve,
        benchmark=benchmark,
        portfolio_benchmark=portfolio_benchmark,
        metals_benchmarks=metals,
    )


# --- Built-in strategies ---------------------------------------------------

def golden_cross_signal(df):
    """Buy when SMA_50 crosses above SMA_200, sell when it crosses below."""
    signal = pd.Series(0, index=df.index)
    if "SMA_50" not in df.columns or "SMA_200" not in df.columns:
        return signal
    above = df["SMA_50"] > df["SMA_200"]
    signal[above & ~above.shift(1).fillna(False)] = 1
    signal[~above & above.shift(1).fillna(False)] = -1
    return signal


def rsi_mean_reversion_signal(df):
    """Buy when RSI drops below 30, sell when RSI rises above 70."""
    signal = pd.Series(0, index=df.index)
    if "RSI_14" not in df.columns:
        return signal
    signal[df["RSI_14"] < 30] = 1
    signal[df["RSI_14"] > 70] = -1
    return signal


def macd_crossover_signal(df):
    """Buy on MACD bullish crossover, sell on bearish crossover."""
    signal = pd.Series(0, index=df.index)
    hist_col = [c for c in df.columns if c.startswith("MACDh_")]
    if not hist_col:
        return signal
    hist = df[hist_col[0]]
    signal[(hist > 0) & (hist.shift(1) <= 0)] = 1
    signal[(hist < 0) & (hist.shift(1) >= 0)] = -1
    return signal


BUILTIN_STRATEGIES = {
    "golden_cross": golden_cross_signal,
    "rsi_mean_reversion": rsi_mean_reversion_signal,
    "macd_crossover": macd_crossover_signal,
}


# --- Output formatting ------------------------------------------------------

def format_results(result):
    """Format backtest results as a human-readable string."""
    lines = [
        f"Backtest Results: {result.config_name}",
        f"Tickers: {', '.join(result.tickers)}",
        f"Period: {result.start_date} to {result.end_date}",
        "",
        f"  Initial Capital:    ${result.initial_capital:,.2f}",
        f"  Final Value:        ${result.final_value:,.2f}",
        f"  Total Return:       {result.total_return:+.2%}",
        f"  Annualized Return:  {result.annualized_return:+.2%}",
        f"  Sharpe Ratio:       {result.sharpe_ratio:.2f}",
        f"  Max Drawdown:       {result.max_drawdown:.2%}",
        f"  Win Rate:           {result.win_rate:.1%}"
        f" ({sum(1 for t in result.trades if t.pnl and t.pnl > 0)}"
        f"/{result.num_trades})",
        f"  Total Trades:       {result.num_trades}",
    ]

    # Benchmark comparison
    bm = result.benchmark if hasattr(result, "benchmark") else None
    if bm:
        lines.append("")
        lines.append(f"  Benchmark (S&P 500 Buy & Hold):")
        lines.append(f"    Total Return:       {bm['total_return']:+.2%}")
        lines.append(f"    Annualized Return:  {bm['annualized_return']:+.2%}")
        lines.append(f"    Sharpe Ratio:       {bm['sharpe_ratio']:.2f}")
        lines.append(f"    Max Drawdown:       {bm['max_drawdown']:.2%}")
        alpha = result.annualized_return - bm["annualized_return"]
        lines.append(f"    Strategy Alpha:     {alpha:+.2%}")
    else:
        lines.append("")
        lines.append("  Benchmark (S&P 500): SPY data not available")

    # Portfolio buy & hold benchmark
    pbm = result.portfolio_benchmark if hasattr(result, "portfolio_benchmark") else None
    if pbm:
        lines.append("")
        lines.append(f"  Portfolio Buy & Hold (same tickers):")
        lines.append(f"    Total Return:       {pbm['total_return']:+.2%}")
        lines.append(f"    Annualized Return:  {pbm['annualized_return']:+.2%}")
        lines.append(f"    Sharpe Ratio:       {pbm['sharpe_ratio']:.2f}")
        lines.append(f"    Max Drawdown:       {pbm['max_drawdown']:.2%}")
        alpha = result.annualized_return - pbm["annualized_return"]
        lines.append(f"    Strategy Alpha:     {alpha:+.2%}")

    # Metals benchmarks
    metals = result.metals_benchmarks if hasattr(result, "metals_benchmarks") else {}
    if metals:
        lines.append("")
        lines.append(f"  Metals Buy & Hold:")
        for label, mbm in metals.items():
            lines.append(
                f"    {label + ':':<9s} Return {mbm['total_return']:+.2%}  "
                f"Sharpe {mbm['sharpe_ratio']:.2f}  "
                f"Drawdown {mbm['max_drawdown']:.2%}"
            )

    if result.trades:
        lines.append("")
        lines.append("  # Ticker  Entry       Exit        Shares  Entry$     Exit$      P&L")
        for i, t in enumerate(result.trades, 1):
            pnl_str = f"${t.pnl:+,.2f}" if t.pnl is not None else "N/A"
            exit_date = t.exit_date or "open"
            exit_price = f"${t.exit_price:,.2f}" if t.exit_price else "N/A"
            lines.append(
                f"  {i:>2} {t.ticker:<7s} {t.entry_date}  {exit_date}  "
                f"{t.shares:>6}  ${t.entry_price:<9,.2f} {exit_price:<10s} {pnl_str}"
            )

    return "\n".join(lines)


def plot_equity_curve(result, output=None):
    """Plot the equity curve from a BacktestResult."""
    import matplotlib
    if output:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eq = result.equity_curve
    if eq.empty:
        logger.warning("No equity curve data to plot.")
        return

    dates = pd.to_datetime(eq["date"])
    values = eq["portfolio_value"]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, values, linewidth=1.5)
    ax.axhline(result.initial_capital, color="gray", linestyle="--", alpha=0.5)
    ax.set_title(f"Equity Curve: {result.config_name}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150)
        plt.close(fig)
    else:
        plt.show()
