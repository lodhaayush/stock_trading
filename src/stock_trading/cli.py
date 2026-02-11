import click

from stock_trading import db
from stock_trading.config import DATA_DIR, DB_PATH
from stock_trading import downloader


@click.group()
def cli():
    """Stock trading data pipeline CLI."""
    pass


@cli.command("init-db")
def init_db_cmd():
    """Create the database schema."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = db.get_connection()
    db.init_db(conn)
    conn.close()
    click.echo(f"Database initialized at {DB_PATH}")


@cli.command("sync-tickers")
def sync_tickers_cmd():
    """Fetch and sync the ticker universe."""
    from stock_trading import tickers

    conn = db.get_connection()
    db.init_db(conn)
    result = tickers.sync_tickers(conn)
    conn.close()
    click.echo(
        f"Synced {result['total']} tickers: "
        f"{result['new']} new, {result['updated']} updated"
    )


@cli.command("download")
@click.option("--resume/--no-resume", default=True, help="Skip already-completed tickers.")
@click.option("--limit", type=int, default=None, help="Max tickers to download.")
@click.option("--tickers", type=str, default=None, help="Comma-separated ticker list.")
def download_cmd(resume, limit, tickers):
    """Download historical price data."""
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    conn = db.get_connection()
    db.init_db(conn)
    tickers_list = [t.strip() for t in tickers.split(",")] if tickers else None
    result = downloader.download_all(conn, limit=limit, tickers_list=tickers_list, resume=resume)
    conn.close()
    click.echo(
        f"Download complete: {result['downloaded']} downloaded, "
        f"{result['failed']} failed, {result['rows']} rows"
    )


@cli.command("status")
def status_cmd():
    """Show download status summary."""
    conn = db.get_connection()
    db.init_db(conn)
    status = downloader.get_download_status(conn)
    conn.close()
    click.echo(f"Total tickers:  {status['total_tickers']}")
    click.echo(f"Total rows:     {status['total_rows']}")
    click.echo(f"Complete:       {status['complete']}")
    click.echo(f"Pending:        {status['pending']}")
    click.echo(f"Failed:         {status['failed']}")
    click.echo(f"No data:        {status['no_data']}")


@cli.command("retry-failed")
def retry_failed_cmd():
    """Retry downloading failed tickers."""
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    conn = db.get_connection()
    db.init_db(conn)
    result = downloader.retry_failed(conn)
    conn.close()
    click.echo(
        f"Retry complete: {result['downloaded']} downloaded, "
        f"{result['failed']} failed, {result['rows']} rows"
    )


@cli.command("update")
@click.option("--fundamentals/--no-fundamentals", default=True, help="Fetch fundamentals data (default: on).")
def update_cmd(fundamentals):
    """Run a daily incremental update."""
    import logging

    from stock_trading import updater

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    conn = db.get_connection()
    db.init_db(conn)
    result = updater.run_daily_update(conn, include_fundamentals=fundamentals)
    conn.close()
    click.echo(
        f"Update complete: {result['tickers_synced']} tickers synced, "
        f"{result['prices_updated']} prices updated, {result['new_rows']} new rows"
    )


@cli.command("fetch-fundamentals")
@click.option("--limit", type=int, default=None, help="Max tickers to fetch fundamentals for.")
def fetch_fundamentals_cmd(limit):
    """Fetch fundamental data for all tickers."""
    import logging

    from stock_trading import fundamentals

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    conn = db.get_connection()
    db.init_db(conn)
    result = fundamentals.fetch_all_fundamentals(conn, limit=limit)
    conn.close()
    click.echo(
        f"Fundamentals complete: {result['processed']} processed, "
        f"{result['updated']} updated, {result['failed']} failed"
    )


@cli.command("query")
@click.option("--ticker", required=True, help="Ticker symbol to query.")
@click.option("--start", default=None, help="Start date (YYYY-MM-DD).")
@click.option("--end", default=None, help="End date (YYYY-MM-DD).")
def query_cmd(ticker, start, end):
    """Query stored price data for a ticker."""
    conn = db.get_connection()
    rows = db.query_prices(conn, ticker, start_date=start, end_date=end)
    conn.close()

    if not rows:
        click.echo(f"No price data found for {ticker}.")
        return

    header = f"{'Date':<12} {'Open':>10} {'High':>10} {'Low':>10} {'Close':>10} {'Volume':>12} {'Adj Close':>10}"
    separator = "-" * len(header)

    click.echo(f"Ticker: {ticker}")
    click.echo(separator)
    click.echo(header)
    click.echo(separator)
    for row in rows:
        click.echo(
            f"{row['date']:<12} "
            f"{row['open']:>10.2f} "
            f"{row['high']:>10.2f} "
            f"{row['low']:>10.2f} "
            f"{row['close']:>10.2f} "
            f"{row['volume']:>12} "
            f"{row['adj_close']:>10.2f}"
        )
    click.echo(separator)
    click.echo(f"{len(rows)} row(s)")


@cli.command("chart")
@click.option("--ticker", required=True, help="Ticker symbol to chart.")
@click.option("--start", default=None, help="Start date (YYYY-MM-DD).")
@click.option("--end", default=None, help="End date (YYYY-MM-DD).")
@click.option("--sma", default=None, help="SMA periods, comma-separated (e.g. 20,50,200).")
@click.option("--ema", default=None, help="EMA periods, comma-separated (e.g. 12,26).")
@click.option("--rsi", default=None, type=int, help="RSI period (e.g. 14).")
@click.option("--macd", default=None,
              help="MACD fast,slow,signal (e.g. 12,26,9).")
@click.option("--bbands", default=None, type=int, help="Bollinger Bands period (e.g. 20).")
@click.option("--momentum", default=None, type=int, help="Momentum swing window (e.g. 5).")
@click.option("--no-volume", is_flag=True, default=False, help="Hide volume subplot.")
@click.option("--output", "-o", default=None, help="Save chart to file instead of displaying.")
def chart_cmd(ticker, start, end, sma, ema, rsi, macd, bbands, momentum, no_volume, output):
    """Render a candlestick chart with optional technical indicators."""
    from stock_trading import charting

    conn = db.get_connection()
    db.init_db(conn)

    indicators = {}
    if sma:
        indicators["sma"] = [int(p.strip()) for p in sma.split(",")]
    if ema:
        indicators["ema"] = [int(p.strip()) for p in ema.split(",")]
    if rsi is not None:
        indicators["rsi"] = rsi
    if macd is not None:
        parts = [int(p.strip()) for p in macd.split(",")]
        if len(parts) != 3:
            click.echo("Error: --macd requires 3 comma-separated values (fast,slow,signal).")
            conn.close()
            return
        indicators["macd"] = tuple(parts)
    if bbands is not None:
        indicators["bbands"] = bbands
    if momentum is not None:
        indicators["momentum"] = momentum
    indicators["volume"] = not no_volume

    success = charting.chart_ticker(conn, ticker, start=start, end=end,
                                     indicators=indicators, output=output)
    conn.close()

    if not success:
        click.echo(f"No price data found for {ticker}.")
    elif output:
        click.echo(f"Chart saved to {output}")


@cli.command("recommend")
@click.option("--top", default=20, type=int, show_default=True,
              help="Number of top recommendations to display.")
@click.option("--sector", default=None, help="Filter results by sector.")
@click.option("--min-market-cap", default=None, type=float,
              help="Minimum market cap filter (e.g. 1e9 for $1B).")
@click.option("--technical-weight", default=0.6, type=float, show_default=True,
              help="Weight for technical score.")
@click.option("--fundamental-weight", default=0.4, type=float, show_default=True,
              help="Weight for fundamental score.")
@click.option("--lookback", default=250, type=int, show_default=True,
              help="Calendar days of price history to use.")
def recommend_cmd(top, sector, min_market_cap, technical_weight, fundamental_weight, lookback):
    """Score and rank stocks, showing top recommendations."""
    import logging

    import pandas as pd

    from stock_trading import screener

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    conn = db.get_connection()
    db.init_db(conn)

    weights = {
        "technical": technical_weight,
        "fundamental": fundamental_weight,
    }

    results = screener.score_universe(conn, lookback_days=lookback, weights=weights)
    conn.close()

    if results.empty:
        click.echo("No results. Ensure you have price and fundamental data.")
        return

    # Apply filters
    if sector:
        results = results[results["sector"].str.lower() == sector.lower()]
    if min_market_cap is not None:
        results = results[results["market_cap"] >= min_market_cap]

    if results.empty:
        click.echo("No results match the specified filters.")
        return

    display = results.head(top)

    click.echo(f"\nTop {len(display)} Stock Recommendations")
    click.echo(f"Weights: Technical={technical_weight:.0%}, Fundamental={fundamental_weight:.0%}")
    click.echo("")

    header = (
        f"{'Rank':>4}  {'Ticker':<6}  {'Name':<20}  {'Sector':<18}  "
        f"{'Composite':>9}  {'Technical':>9}  {'Fundmntl':>9}  "
        f"{'Price':>8}  {'RSI':>5}  {'P/E':>6}  {'Target':>12}  {'MktCap':>10}"
    )
    separator = "-" * len(header)

    click.echo(header)
    click.echo(separator)

    for rank, (_, row) in enumerate(display.iterrows(), start=1):
        raw_name = row.get("name")
        name = (str(raw_name) if pd.notna(raw_name) else "")[:20]
        raw_sector = row.get("sector")
        sector_val = (str(raw_sector) if pd.notna(raw_sector) else "")[:18]
        price = row.get("price", 0)
        rsi = row.get("rsi_value", 0)
        pe = row.get("pe", 0)
        mcap = row.get("market_cap", 0)

        if pd.notna(mcap) and mcap > 0:
            if mcap >= 1e12:
                mcap_str = f"{mcap / 1e12:.1f}T"
            elif mcap >= 1e9:
                mcap_str = f"{mcap / 1e9:.1f}B"
            elif mcap >= 1e6:
                mcap_str = f"{mcap / 1e6:.1f}M"
            else:
                mcap_str = f"{mcap:.0f}"
        else:
            mcap_str = "N/A"

        pe_str = f"{pe:.1f}" if pd.notna(pe) and pe > 0 else "N/A"
        rsi_str = f"{rsi:.1f}" if pd.notna(rsi) else "N/A"

        target_mean = row.get("target_mean")
        if pd.notna(target_mean) and target_mean > 0 and pd.notna(price) and price > 0:
            upside = (target_mean - price) / price * 100
            sign = "+" if upside >= 0 else ""
            target_str = f"{target_mean:.0f}({sign}{upside:.0f}%)"
        else:
            target_str = "N/A"

        click.echo(
            f"{rank:>4}  {row['ticker']:<6}  {name:<20}  {sector_val:<18}  "
            f"{row['composite_score']:>9.4f}  {row['technical_score']:>9.4f}  "
            f"{row['fundamental_score']:>9.4f}  "
            f"{price:>8.2f}  {rsi_str:>5}  {pe_str:>6}  {target_str:>12}  {mcap_str:>10}"
        )
