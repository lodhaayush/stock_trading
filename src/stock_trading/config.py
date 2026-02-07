from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "stock_trading.db"

# Download settings
BATCH_SIZE = 50
BATCH_DELAY_SECONDS = 2.0
MAX_RETRIES = 3
RETRY_BASE_DELAY_SECONDS = 5.0

# Fundamentals throttle
FUNDAMENTALS_DELAY_SECONDS = 0.5

# NASDAQ FTP URLs
NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

# SEC EDGAR fallback
SEC_EDGAR_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_EDGAR_HEADERS = {"User-Agent": "stock-trading-pipeline admin@example.com"}

# Fundamental fields to extract from yfinance Ticker.info
FUNDAMENTAL_FIELDS = [
    "sector",
    "industry",
    "marketCap",
    "trailingPE",
    "forwardPE",
    "dividendYield",
    "beta",
]
