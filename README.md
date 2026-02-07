# stock_trading

Personal stock trading data pipeline -- fetches, stores, and updates US equity price and fundamental data for screening and strategy development.

Data is stored locally in SQLite at `data/stock_trading.db`.

## Requirements

- Python >= 3.11

## Installation

```bash
git clone <repo-url> && cd stock_trading
pip install -e ".[dev]"
```

## Dependencies

yfinance, pandas, pandas-ta, click, tqdm, requests

## Quick start

```bash
stock-trading init-db
stock-trading sync-tickers
stock-trading download --limit 50
stock-trading status
```

## CLI commands

### `stock-trading init-db`

Create the database schema.

```bash
stock-trading init-db
```

### `stock-trading sync-tickers`

Fetch and sync the ticker universe from available sources.

```bash
stock-trading sync-tickers
```

### `stock-trading download`

Download historical price data for all tickers (or a subset). Resumes by default.

```bash
stock-trading download
stock-trading download --limit 100
stock-trading download --tickers AAPL,MSFT,GOOG
stock-trading download --no-resume
```

### `stock-trading status`

Show download progress stats (complete, pending, failed counts).

```bash
stock-trading status
```

### `stock-trading retry-failed`

Retry downloading any tickers that previously failed.

```bash
stock-trading retry-failed
```

### `stock-trading update`

Run a daily incremental update. Optionally include fundamentals.

```bash
stock-trading update
stock-trading update --fundamentals
```

### `stock-trading fetch-fundamentals`

Fetch company fundamental data for all tickers.

```bash
stock-trading fetch-fundamentals
stock-trading fetch-fundamentals --limit 200
```

### `stock-trading query`

Query stored price data for a specific ticker.

```bash
stock-trading query --ticker AAPL
stock-trading query --ticker AAPL --start 2024-01-01 --end 2024-06-30
```
