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
