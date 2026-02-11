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
@click.option("--fundamentals", is_flag=True, default=False, help="Also fetch fundamentals data.")
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
@click.option("--no-volume", is_flag=True, default=False, help="Hide volume subplot.")
@click.option("--output", "-o", default=None, help="Save chart to file instead of displaying.")
def chart_cmd(ticker, start, end, sma, ema, rsi, macd, bbands, no_volume, output):
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
    indicators["volume"] = not no_volume

    success = charting.chart_ticker(conn, ticker, start=start, end=end,
                                     indicators=indicators, output=output)
    conn.close()

    if not success:
        click.echo(f"No price data found for {ticker}.")
    elif output:
        click.echo(f"Chart saved to {output}")
