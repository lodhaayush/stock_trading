import click

from stock_trading import db
from stock_trading.config import DATA_DIR, DB_PATH


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
