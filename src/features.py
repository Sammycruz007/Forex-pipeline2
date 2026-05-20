
import logging
import yaml
import pandas as pd
import numpy as np
import ta
from pathlib import Path

"""
Feature Store Module
Transforms raw OHLCV data into ML-ready feature matrix.
"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Moving averages and trend indicators."""
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]

    df["sma_10"]  = ta.trend.sma_indicator(close, window=10)
    df["sma_20"]  = ta.trend.sma_indicator(close, window=20)
    df["sma_50"]  = ta.trend.sma_indicator(close, window=50)
    df["ema_10"]  = ta.trend.ema_indicator(close, window=10)
    df["ema_20"]  = ta.trend.ema_indicator(close, window=20)

    df["price_vs_sma20"] = (close - df["sma_20"]) / df["sma_20"]
    df["price_vs_sma50"] = (close - df["sma_50"]) / df["sma_50"]

    macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["macd"]           = macd.macd()
    df["macd_signal"]    = macd.macd_signal()
    df["macd_histogram"] = macd.macd_diff()

    adx = ta.trend.ADXIndicator(high, low, close, window=14)
    df["adx"] = adx.adx()

    logger.info("Trend features added")
    return df


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Momentum and oscillator indicators."""
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]

    df["rsi_14"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    df["rsi_7"]  = ta.momentum.RSIIndicator(close, window=7).rsi()

    stoch = ta.momentum.StochasticOscillator(high, low, close, window=14)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    df["roc_5"]  = ta.momentum.ROCIndicator(close, window=5).roc()
    df["roc_10"] = ta.momentum.ROCIndicator(close, window=10).roc()

    logger.info("Momentum features added")
    return df


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Volatility indicators."""
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]

    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_upper"]    = bb.bollinger_hband()
    df["bb_lower"]    = bb.bollinger_lband()
    df["bb_middle"]   = bb.bollinger_mavg()
    df["bb_width"]    = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
    df["bb_position"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    df["atr_14"] = ta.volatility.AverageTrueRange(
        high, low, close, window=14
    ).average_true_range()

    logger.info("Volatility features added")
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lag features and price-based proxies (replaces volume for Forex)."""
    close = df["Close"]

    # Lagged returns
    for lag in [1, 2, 3, 5, 10]:
        df[f"return_lag_{lag}"] = close.pct_change(periods=lag)

    # Daily range as % of close
    df["hl_range"] = (df["High"] - df["Low"]) / df["Close"]

    # Overnight gap
    df["gap"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)

    # Rolling volatility — std of returns over 10 days
    # This replaces volume as a "market activity" proxy for Forex
    df["volatility_10"] = close.pct_change().rolling(window=10).std()

    # Rolling volatility over 20 days
    df["volatility_20"] = close.pct_change().rolling(window=20).std()

    logger.info("Lag + volatility proxy features added")
    return df


def add_target(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    Binary target: will price be higher in N days?
    1 = Up, 0 = Down
    Last N rows are dropped — no future data available for them.
    """
    future_close        = df["Close"].shift(-horizon)
    df["target"]        = (future_close > df["Close"]).astype(int)
    df["future_return"] = (future_close - df["Close"]) / df["Close"]

    df = df.iloc[:-horizon]

    up_pct = df["target"].mean() * 100
    logger.info(
        f"Target created | Horizon: {horizon} days | "
        f"Up: {up_pct:.1f}% | Down: {100 - up_pct:.1f}%"
    )
    return df


def build_feature_store(config_path: str = "config.yaml") -> pd.DataFrame:
    """Full feature engineering pipeline."""
    config        = load_config(config_path)
    symbol        = config["data"]["symbol"]
    horizon       = config["features"]["target_horizon"]
    raw_path      = config["data"]["raw_path"]
    features_path = config["data"]["features_path"]

    # Load raw data
    raw_file = f"{raw_path}/{symbol.replace('=', '_')}_ohlcv.parquet"
    logger.info(f"Loading raw data from {raw_file}")
    df = pd.read_parquet(raw_file, engine="pyarrow")
    logger.info(f"Raw data shape: {df.shape}")

    # Build features — order matters
    df = add_trend_features(df)
    df = add_momentum_features(df)
    df = add_volatility_features(df)
    df = add_lag_features(df)
    df = add_target(df, horizon=horizon)

    # Drop NaN rows from indicator warmup periods
    before  = len(df)
    df      = df.dropna()
    dropped = before - len(df)
    logger.info(f"Dropped {dropped} NaN rows from indicator warmup")
    logger.info(f"Final feature matrix shape: {df.shape}")

    # Sanity check
    if len(df) == 0:
        # Debug: show which columns still have NaNs before dropna
        nan_counts = df.isnull().sum()
        logger.error("DataFrame is empty after dropna. NaN counts per column:")
        logger.error(nan_counts[nan_counts > 0].to_string())
        raise ValueError("Feature store is empty. Check NaN debug output above.")

    # Save
    Path(features_path).mkdir(parents=True, exist_ok=True)
    output_file = f"{features_path}/{symbol.replace('=', '_')}_features.parquet"
    df.to_parquet(output_file, index=True, engine="pyarrow")
    logger.info(f"Feature store saved to {output_file}")

    return df


if __name__ == "__main__":
    df = build_feature_store()

    print("\n--- Feature Matrix Sample (last 3 rows) ---")
    print(df.tail(3).T.to_string())

    print(f"\n--- Summary ---")
    print(f"Shape         : {df.shape}")
    print(f"Features      : {df.shape[1] - 2} (excl. target + future_return)")
    print(f"Date range    : {df.index[0].date()} → {df.index[-1].date()}")
    print(f"Target balance: {df['target'].value_counts().to_dict()}")
    print(f"Any NaNs      : {df.isnull().sum().sum()}")
