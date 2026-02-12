"""Stock screener: score and rank tickers by technical + fundamental signals."""

import logging
from itertools import groupby

import numpy as np
import pandas as pd
import pandas_ta as ta

from stock_trading import db

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {
    "technical": 0.6,
    "fundamental": 0.4,
}

TECHNICAL_WEIGHTS = {
    "rsi": 0.20,
    "macd": 0.0,
    "ma_crossover": 0.30,
    "bbands": 0.0,
    "momentum": 0.50,
}


def _score_rsi(rsi_value):
    """Score RSI on a [-1, +1] scale. NaN -> 0.0."""
    if pd.isna(rsi_value):
        return 0.0
    if rsi_value < 30:
        return 1.0
    elif rsi_value < 45:
        return 0.5
    elif rsi_value <= 55:
        return 0.0
    elif rsi_value <= 70:
        return -0.5
    else:
        return -1.0


def _score_macd(macd_val, signal_val, hist_val, prev_hist_val):
    """Score MACD on a [-1, +1] scale."""
    if any(pd.isna(v) for v in (macd_val, signal_val, hist_val)):
        return 0.0
    if pd.isna(prev_hist_val):
        prev_hist_val = 0.0

    if prev_hist_val <= 0 < hist_val:
        return 1.0
    if prev_hist_val >= 0 > hist_val:
        return -1.0
    if hist_val > 0 and hist_val > prev_hist_val:
        return 0.5
    if hist_val < 0 and hist_val < prev_hist_val:
        return -0.5
    return 0.0


def _score_ma_crossover(price, sma50, sma200):
    """Score SMA 50/200 crossover on a [-1, +1] scale."""
    if any(pd.isna(v) for v in (price, sma50, sma200)):
        return 0.0
    if sma50 > sma200:
        if price > sma50:
            return 1.0
        else:
            return -0.25
    else:
        if price > sma50:
            return 0.5
        else:
            return -1.0


def _score_bbands(price, bb_lower, bb_mid, bb_upper):
    """Score Bollinger Bands position on a [-1, +1] scale."""
    if any(pd.isna(v) for v in (price, bb_lower, bb_mid, bb_upper)):
        return 0.0
    if bb_upper == bb_lower:
        return 0.0

    position = (price - bb_lower) / (bb_upper - bb_lower)

    if position <= 0.0:
        return 1.0
    elif position <= 0.2:
        return 0.5
    elif position <= 0.8:
        return 0.0
    elif position < 1.0:
        return -0.5
    else:
        return -1.0


def _detect_swings(highs, lows, window=5):
    """Detect swing highs and swing lows in price data.

    A swing high is a bar whose High is the maximum in a window of
    [i-window, i+window]. Swing lows are analogous with minimums.

    Returns (swing_highs, swing_lows) where each is a list of
    (position, value) tuples.
    """
    swing_highs = []
    swing_lows = []
    n = len(highs)

    for i in range(window, n - window):
        high_window = highs.iloc[i - window:i + window + 1]
        if highs.iloc[i] == high_window.max():
            swing_highs.append((i, highs.iloc[i]))

        low_window = lows.iloc[i - window:i + window + 1]
        if lows.iloc[i] == low_window.min():
            swing_lows.append((i, lows.iloc[i]))

    return swing_highs, swing_lows


def _score_momentum(highs, lows, window=5, n_swings=4):
    """Score momentum based on higher highs / lower lows pattern.

    Returns a score on [-1, +1]:
      +1 = all higher highs + higher lows (strong bullish)
      -1 = all lower highs + lower lows (strong bearish)
       0 = mixed or insufficient data
    """
    swing_highs, swing_lows = _detect_swings(highs, lows, window=window)

    # Need at least n_swings+1 points to count n_swings transitions
    min_points = n_swings + 1
    if len(swing_highs) < min_points and len(swing_lows) < min_points:
        return 0.0

    hh_count = 0  # higher highs
    lh_count = 0  # lower highs
    hl_count = 0  # higher lows
    ll_count = 0  # lower lows

    # Count higher highs vs lower highs from recent swing highs
    if len(swing_highs) >= min_points:
        recent_highs = swing_highs[-min_points:]
        for i in range(1, len(recent_highs)):
            if recent_highs[i][1] > recent_highs[i - 1][1]:
                hh_count += 1
            elif recent_highs[i][1] < recent_highs[i - 1][1]:
                lh_count += 1

    # Count higher lows vs lower lows from recent swing lows
    if len(swing_lows) >= min_points:
        recent_lows = swing_lows[-min_points:]
        for i in range(1, len(recent_lows)):
            if recent_lows[i][1] > recent_lows[i - 1][1]:
                hl_count += 1
            elif recent_lows[i][1] < recent_lows[i - 1][1]:
                ll_count += 1

    # Net: positive = bullish, negative = bearish
    # Max possible net = 2 * n_swings (all HH + all HL)
    net = (hh_count + hl_count) - (ll_count + lh_count)
    max_net = 2 * n_swings
    return max(-1.0, min(1.0, net / max_net)) if max_net > 0 else 0.0


def compute_technical_score(df):
    """Compute technical indicator scores for a single ticker's OHLCV DataFrame.

    Returns a dict with individual scores and combined technical_score [0, 1],
    or None if the DataFrame has fewer than 26 rows.
    """
    if df is None or len(df) < 26:
        return None

    close = df["Close"]
    latest = len(df) - 1

    # Compute indicators
    rsi_series = ta.rsi(close, length=14)
    macd_df = ta.macd(close, fast=12, slow=26, signal=9)
    sma50 = ta.sma(close, length=50)
    sma200 = ta.sma(close, length=200)
    bbands = ta.bbands(close, length=20)

    # Extract latest values
    rsi_val = rsi_series.iloc[latest] if rsi_series is not None else np.nan
    price = close.iloc[latest]

    # MACD
    if macd_df is not None and not macd_df.empty:
        macd_col = [c for c in macd_df.columns if c.startswith("MACD_")][0]
        signal_col = [c for c in macd_df.columns if c.startswith("MACDs_")][0]
        hist_col = [c for c in macd_df.columns if c.startswith("MACDh_")][0]
        macd_val = macd_df[macd_col].iloc[latest]
        signal_val = macd_df[signal_col].iloc[latest]
        hist_val = macd_df[hist_col].iloc[latest]
        prev_hist_val = macd_df[hist_col].iloc[latest - 1] if latest > 0 else np.nan
    else:
        macd_val = signal_val = hist_val = prev_hist_val = np.nan

    # SMA
    sma50_val = sma50.iloc[latest] if sma50 is not None else np.nan
    sma200_val = sma200.iloc[latest] if sma200 is not None else np.nan

    # Bollinger Bands
    if bbands is not None and not bbands.empty:
        bbl = [c for c in bbands.columns if c.startswith("BBL_")][0]
        bbm = [c for c in bbands.columns if c.startswith("BBM_")][0]
        bbu = [c for c in bbands.columns if c.startswith("BBU_")][0]
        bb_lower = bbands[bbl].iloc[latest]
        bb_mid = bbands[bbm].iloc[latest]
        bb_upper = bbands[bbu].iloc[latest]
    else:
        bb_lower = bb_mid = bb_upper = np.nan

    # Score each signal
    rsi_score = _score_rsi(rsi_val)
    macd_score = _score_macd(macd_val, signal_val, hist_val, prev_hist_val)
    ma_score = _score_ma_crossover(price, sma50_val, sma200_val)
    bb_score = _score_bbands(price, bb_lower, bb_mid, bb_upper)
    mom_score = _score_momentum(df["High"], df["Low"])

    raw = (
        TECHNICAL_WEIGHTS["rsi"] * rsi_score
        + TECHNICAL_WEIGHTS["macd"] * macd_score
        + TECHNICAL_WEIGHTS["ma_crossover"] * ma_score
        + TECHNICAL_WEIGHTS["bbands"] * bb_score
        + TECHNICAL_WEIGHTS["momentum"] * mom_score
    )
    technical_score = (raw + 1.0) / 2.0

    return {
        "price": price,
        "rsi_value": rsi_val,
        "rsi_score": rsi_score,
        "macd_score": macd_score,
        "ma_crossover_score": ma_score,
        "bbands_score": bb_score,
        "momentum_score": mom_score,
        "technical_score": technical_score,
    }


def compute_fundamental_scores(fundamentals_df):
    """Compute fundamental scores for the entire universe using percentile ranking.

    Returns the DataFrame with added score columns and a combined fundamental_score [0, 1].
    """
    df = fundamentals_df.copy()

    # Ensure numeric types for all scoring columns
    for col in ("trailing_pe", "forward_pe", "dividend_yield", "beta", "market_cap",
                "target_median", "price"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # P/E: forward preferred, fallback trailing. Lower is better.
    df["pe"] = df["forward_pe"].fillna(df["trailing_pe"])
    df.loc[df["pe"] <= 0, "pe"] = np.nan
    df["pe_score"] = 1.0 - df["pe"].rank(pct=True, na_option="keep")
    df["pe_score"] = df["pe_score"].fillna(0.0)

    # Dividend yield: higher is better, capped at 10%.
    div_capped = df["dividend_yield"].clip(upper=0.10)
    df["dividend_score"] = div_capped.rank(pct=True, na_option="keep").fillna(0.0)

    # Beta: closer to 1.0 is better.
    beta_distance = (df["beta"] - 1.0).abs()
    df["beta_score"] = 1.0 - beta_distance.rank(pct=True, na_option="keep")
    df["beta_score"] = df["beta_score"].fillna(0.5)

    # Market cap: larger is more stable.
    df["market_cap_score"] = df["market_cap"].rank(pct=True, na_option="keep").fillna(0.0)

    # Target upside: (target_median - price) / price. Higher upside = better.
    if "price" in df.columns and "target_median" in df.columns:
        df.loc[df["price"] <= 0, "price"] = np.nan
        df["target_upside"] = (df["target_median"] - df["price"]) / df["price"]
        df["target_upside_score"] = df["target_upside"].rank(
            pct=True, na_option="keep"
        ).fillna(0.0)
    else:
        df["target_upside"] = np.nan
        df["target_upside_score"] = 0.0

    # Combined
    df["fundamental_score"] = (
        0.25 * df["pe_score"]
        + 0.20 * df["dividend_score"]
        + 0.15 * df["beta_score"]
        + 0.25 * df["market_cap_score"]
        + 0.15 * df["target_upside_score"]
    )

    return df


def score_universe(conn, lookback_days=250, weights=None):
    """Score and rank all tickers in the database.

    Returns a DataFrame sorted by composite_score descending.
    """
    from tqdm import tqdm

    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()

    total_w = weights["technical"] + weights["fundamental"]
    w_tech = weights["technical"] / total_w
    w_fund = weights["fundamental"] / total_w

    # Load all data
    logger.info("Loading fundamental data...")
    fund_rows = db.query_all_fundamentals(conn)
    fund_df = pd.DataFrame([dict(r) for r in fund_rows])

    if fund_df.empty:
        logger.warning("No tickers found in database.")
        return pd.DataFrame()

    logger.info("Loading recent prices (last %d days)...", lookback_days)
    price_rows = db.query_recent_prices(conn, lookback_days=lookback_days)

    if not price_rows:
        logger.warning("No price data found.")
        return pd.DataFrame()

    # Partition price rows by ticker
    logger.info("Computing technical scores...")
    tech_results = []

    grouped = groupby(price_rows, key=lambda r: r["ticker"])
    ticker_groups = {ticker: list(rows) for ticker, rows in grouped}

    for ticker in tqdm(ticker_groups, desc="Scoring", unit="ticker"):
        rows = ticker_groups[ticker]
        data = [dict(r) for r in rows]
        tdf = pd.DataFrame(data)
        tdf["date"] = pd.to_datetime(tdf["date"])
        tdf = tdf.set_index("date").sort_index()
        tdf = tdf.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume", "adj_close": "Adj Close",
        })

        result = compute_technical_score(tdf)
        if result is not None:
            result["ticker"] = ticker
            tech_results.append(result)

    if not tech_results:
        logger.warning("No tickers had enough price data for scoring.")
        return pd.DataFrame()

    tech_df = pd.DataFrame(tech_results)

    # Enrich fundamentals with current price for target upside scoring
    fund_with_price = pd.merge(
        fund_df, tech_df[["ticker", "price"]], on="ticker", how="left",
    )

    # Compute fundamental scores
    logger.info("Computing fundamental scores...")
    fund_scored = compute_fundamental_scores(fund_with_price)

    # Merge — drop price from fund_scored to avoid duplicate with tech_df
    fund_scored = fund_scored.drop(columns=["price"], errors="ignore")
    merged = pd.merge(tech_df, fund_scored, on="ticker", how="inner")

    # Composite score
    merged["composite_score"] = (
        w_tech * merged["technical_score"]
        + w_fund * merged["fundamental_score"]
    )

    merged = merged.sort_values("composite_score", ascending=False)

    output_cols = [
        "ticker", "name", "sector", "composite_score",
        "technical_score", "fundamental_score",
        "price", "rsi_value", "pe", "market_cap",
        "target_mean", "target_high", "target_low", "num_analysts",
        "target_upside",
    ]
    available = [c for c in output_cols if c in merged.columns]
    return merged[available].reset_index(drop=True)
