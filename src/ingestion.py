"""
Data Ingestion Module
Downloads raw OHLCV data for primary symbol + macro indicators.
"""

import logging
import yaml
import yfinance as yf
import pandas as pd
from pathlib import Path

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
        raise ValueError(f"No data returned for {symbol}.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    logger.info(f"Downloaded {len(df)} rows | Columns: {list(df.columns)}")
    return df


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    initial_len = len(df)
    df = df.dropna(subset=["Close"])
    df = df[~df.index.duplicated(keep="first")]

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df = df.sort_index()
    dropped = initial_len - len(df)
    if dropped > 0:
        logger.warning(f"Dropped {dropped} invalid rows during validation")

    logger.info(f"Validation passed | {len(df)} clean rows remain")
    logger.info(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
    return df


def save_to_parquet(df: pd.DataFrame, path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=True, engine="pyarrow")
    logger.info(f"Saved to {output_path} | Size: {output_path.stat().st_size / 1024:.1f} KB")


def load_from_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path, engine="pyarrow")
    logger.info(f"Loaded {len(df)} rows from {path}")
    return df


def run_ingestion(config_path: str = "config.yaml") -> pd.DataFrame:
    config     = load_config(config_path)
    symbol     = config["data"]["symbol"]
    start_date = config["data"]["start_date"]
    from datetime import date as _date
    end_date   = str(_date.today())
    interval   = config["data"]["interval"]
    raw_path   = config["data"]["raw_path"]

    # Download primary symbol (EURUSD)
    df = download_ohlcv(symbol, start_date, end_date, interval)
    df = validate_ohlcv(df)
    output_file = f"{raw_path}/{symbol.replace('=', '_')}_ohlcv.parquet"
    save_to_parquet(df, output_file)

    # Download macro symbols
    macro_symbols = config["data"].get("macro_symbols", [])
    for macro_symbol in macro_symbols:
        try:
            macro_df = download_ohlcv(macro_symbol, start_date, end_date, interval)
            macro_df = validate_ohlcv(macro_df)
            macro_file = f"{raw_path}/{macro_symbol.replace('=', '_').replace('^', '')}_ohlcv.parquet"
            save_to_parquet(macro_df, macro_file)
        except Exception as e:
            logger.warning(f"Could not download {macro_symbol}: {e}")

    return df


if __name__ == "__main__":
    df = run_ingestion()
    print("\n--- Primary Data Sample ---")
    print(df.tail(3))
    print(f"\nShape: {df.shape}")
