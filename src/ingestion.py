import logging
import yaml
import yfinance as yf
import pandas as pd
from pathlib import Path

"""
Data Ingestion Module
Downloads raw OHLCV Forex data from Yahoo Finance and saves to Parquet.
"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def download_ohlcv(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str = "1d"
) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance."""
    logger.info(f"Downloading {symbol} from {start_date} to {end_date}")

    df = yf.download(
        tickers=symbol,
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        raise ValueError(f"No data returned for {symbol}. Check ticker or date range.")

    # Flatten multi-level columns if present (yfinance quirk)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    logger.info(f"Downloaded {len(df)} rows | Columns: {list(df.columns)}")
    return df


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean raw OHLCV data."""
    required_columns = ["Open", "High", "Low", "Close", "Volume"]

    # Check required columns exist
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    initial_len = len(df)

    # Drop rows where Close is null
    df = df.dropna(subset=["Close"])

    # Drop duplicate index entries
    df = df[~df.index.duplicated(keep="first")]

    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Sort chronologically
    df = df.sort_index()

    dropped = initial_len - len(df)
    if dropped > 0:
        logger.warning(f"Dropped {dropped} invalid rows during validation")

    logger.info(f"Validation passed | {len(df)} clean rows remain")
    logger.info(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
    return df


def save_to_parquet(df: pd.DataFrame, path: str) -> None:
    """Save DataFrame to Parquet file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=True, engine="pyarrow")
    logger.info(f"Saved to {output_path} | Size: {output_path.stat().st_size / 1024:.1f} KB")


def load_from_parquet(path: str) -> pd.DataFrame:
    """Load DataFrame from Parquet file."""
    df = pd.read_parquet(path, engine="pyarrow")
    logger.info(f"Loaded {len(df)} rows from {path}")
    return df


def run_ingestion(config_path: str = "config.yaml") -> pd.DataFrame:
    """Full ingestion pipeline: download → validate → save."""
    config = load_config(config_path)

    symbol     = config["data"]["symbol"]
    start_date = config["data"]["start_date"]
    end_date   = config["data"]["end_date"]
    interval   = config["data"]["interval"]
    raw_path   = config["data"]["raw_path"]

    output_file = f"{raw_path}/{symbol.replace('=', '_')}_ohlcv.parquet"

    # Download
    df = download_ohlcv(symbol, start_date, end_date, interval)

    # Validate
    df = validate_ohlcv(df)

    # Save
    save_to_parquet(df, output_file)

    return df


if __name__ == "__main__":
    df = run_ingestion()
    print("\n--- Data Sample ---")
    print(df.tail(5))
    print(f"\nShape: {df.shape}")
    print(f"Dtypes:\n{df.dtypes}")
