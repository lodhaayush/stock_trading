"""Charting and technical analysis visualization."""

import logging

import pandas as pd
import pandas_ta as ta
import mplfinance as mpf

logger = logging.getLogger(__name__)

_INDICATOR_COLORS = {
    "sma": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
    "ema": ["#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8"],
    "bbands_upper": "#aaaaaa",
    "bbands_mid": "#888888",
    "bbands_lower": "#aaaaaa",
    "rsi": "#7f00ff",
    "macd_line": "#2962FF",
    "macd_signal": "#FF6D00",
    "macd_hist_pos": "#26A69A",
    "macd_hist_neg": "#EF5350",
    "swing_high": "#26A69A",
    "swing_low": "#EF5350",
}


def rows_to_dataframe(rows):
    """Convert sqlite3.Row objects to an mplfinance-compatible DataFrame.

    Returns None if rows is empty.
    """
    if not rows:
        return None

    data = [dict(row) for row in rows]
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df = df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "adj_close": "Adj Close",
    })
    df = df.drop(columns=["ticker"], errors="ignore")
    df.index.name = "Date"
    return df


def compute_indicators(df, indicators):
    """Compute technical indicators and return addplot list.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with DatetimeIndex.
    indicators : dict
        Indicator config, e.g. {"sma": [20, 50], "rsi": 14, "macd": (12,26,9), "volume": True}.

    Returns
    -------
    tuple[list, dict]
        (addplots, extra_kwargs) for mpf.plot.
    """
    addplots = []
    extra_kwargs = {}
    next_panel = 2 if indicators.get("volume", True) else 1

    # SMA
    for i, period in enumerate(indicators.get("sma", [])):
        col = f"SMA_{period}"
        df[col] = ta.sma(df["Close"], length=period)
        color = _INDICATOR_COLORS["sma"][i % len(_INDICATOR_COLORS["sma"])]
        addplots.append(mpf.make_addplot(df[col], panel=0, color=color, width=1.0))

    # EMA
    for i, period in enumerate(indicators.get("ema", [])):
        col = f"EMA_{period}"
        df[col] = ta.ema(df["Close"], length=period)
        color = _INDICATOR_COLORS["ema"][i % len(_INDICATOR_COLORS["ema"])]
        addplots.append(mpf.make_addplot(df[col], panel=0, color=color, width=1.0,
                                          linestyle="dashed"))

    # Bollinger Bands
    bbands_period = indicators.get("bbands")
    if bbands_period:
        bbands = ta.bbands(df["Close"], length=bbands_period)
        bbl_col = [c for c in bbands.columns if c.startswith("BBL_")][0]
        bbm_col = [c for c in bbands.columns if c.startswith("BBM_")][0]
        bbu_col = [c for c in bbands.columns if c.startswith("BBU_")][0]
        df["BB_Lower"] = bbands[bbl_col]
        df["BB_Mid"] = bbands[bbm_col]
        df["BB_Upper"] = bbands[bbu_col]
        addplots.append(mpf.make_addplot(df["BB_Upper"], panel=0,
                                          color=_INDICATOR_COLORS["bbands_upper"], width=0.7))
        addplots.append(mpf.make_addplot(df["BB_Mid"], panel=0,
                                          color=_INDICATOR_COLORS["bbands_mid"],
                                          width=0.7, linestyle="dotted"))
        addplots.append(mpf.make_addplot(df["BB_Lower"], panel=0,
                                          color=_INDICATOR_COLORS["bbands_lower"], width=0.7))

    # RSI
    rsi_period = indicators.get("rsi")
    if rsi_period:
        rsi_col = f"RSI_{rsi_period}"
        df[rsi_col] = ta.rsi(df["Close"], length=rsi_period)
        addplots.append(mpf.make_addplot(df[rsi_col], panel=next_panel,
                                          color=_INDICATOR_COLORS["rsi"],
                                          ylabel=f"RSI({rsi_period})"))
        df["RSI_70"] = 70
        df["RSI_30"] = 30
        addplots.append(mpf.make_addplot(df["RSI_70"], panel=next_panel,
                                          color="gray", width=0.5, linestyle="dashed"))
        addplots.append(mpf.make_addplot(df["RSI_30"], panel=next_panel,
                                          color="gray", width=0.5, linestyle="dashed"))
        next_panel += 1

    # MACD
    macd_params = indicators.get("macd")
    if macd_params:
        fast, slow, signal = macd_params
        macd_df = ta.macd(df["Close"], fast=fast, slow=slow, signal=signal)
        macd_col = [c for c in macd_df.columns if c.startswith("MACD_")][0]
        hist_col = [c for c in macd_df.columns if c.startswith("MACDh_")][0]
        signal_col = [c for c in macd_df.columns if c.startswith("MACDs_")][0]
        df["MACD"] = macd_df[macd_col]
        df["MACD_Signal"] = macd_df[signal_col]
        df["MACD_Hist"] = macd_df[hist_col]

        hist_colors = [_INDICATOR_COLORS["macd_hist_pos"]
                       if v >= 0 else _INDICATOR_COLORS["macd_hist_neg"]
                       for v in df["MACD_Hist"].fillna(0)]

        addplots.append(mpf.make_addplot(df["MACD"], panel=next_panel,
                                          color=_INDICATOR_COLORS["macd_line"], ylabel="MACD"))
        addplots.append(mpf.make_addplot(df["MACD_Signal"], panel=next_panel,
                                          color=_INDICATOR_COLORS["macd_signal"]))
        addplots.append(mpf.make_addplot(df["MACD_Hist"], panel=next_panel,
                                          type="bar", color=hist_colors))
        next_panel += 1

    # Momentum (swing high/low markers)
    momentum_window = indicators.get("momentum")
    if momentum_window:
        from stock_trading.screener import _detect_swings

        swing_highs, swing_lows = _detect_swings(df["High"], df["Low"],
                                                   window=momentum_window)

        # Create scatter series for swing highs (markers above highs)
        sh_series = pd.Series(float("nan"), index=df.index)
        for pos, val in swing_highs:
            sh_series.iloc[pos] = df["High"].iloc[pos]
        addplots.append(mpf.make_addplot(sh_series, panel=0, type="scatter",
                                          markersize=50, marker="v",
                                          color=_INDICATOR_COLORS["swing_high"]))

        # Create scatter series for swing lows (markers below lows)
        sl_series = pd.Series(float("nan"), index=df.index)
        for pos, val in swing_lows:
            sl_series.iloc[pos] = df["Low"].iloc[pos]
        addplots.append(mpf.make_addplot(sl_series, panel=0, type="scatter",
                                          markersize=50, marker="^",
                                          color=_INDICATOR_COLORS["swing_low"]))

    extra_kwargs["volume"] = indicators.get("volume", True)
    return addplots, extra_kwargs


def render_chart(df, ticker, addplots=None, output=None, **kwargs):
    """Render a candlestick chart.

    If output is given, saves to that file path. Otherwise displays in a window.
    """
    style = mpf.make_mpf_style(base_mpf_style="charles", gridstyle="-", gridcolor="#e0e0e0")

    plot_kwargs = {
        "type": "candle",
        "style": style,
        "title": f"\n{ticker}",
        "datetime_format": "%Y-%m-%d",
        "figscale": 1.3,
        "figratio": (16, 9),
        "tight_layout": True,
    }
    plot_kwargs.update(kwargs)

    if addplots:
        plot_kwargs["addplot"] = addplots

    if output:
        plot_kwargs["savefig"] = output
        logger.info("Saving chart to %s", output)

    mpf.plot(df, **plot_kwargs)

    if output:
        logger.info("Chart saved to %s", output)


def chart_ticker(conn, ticker, start=None, end=None, indicators=None, output=None):
    """Generate a chart for a ticker from database data.

    Returns True if chart was rendered, False if no data found.
    """
    from stock_trading import db as db_mod

    rows = db_mod.query_prices(conn, ticker, start_date=start, end_date=end)
    df = rows_to_dataframe(rows)
    if df is None:
        logger.warning("No price data found for %s", ticker)
        return False

    if indicators is None:
        indicators = {"volume": True}

    addplots, extra_kwargs = compute_indicators(df, indicators)
    render_chart(df, ticker, addplots=addplots, output=output, **extra_kwargs)
    return True
