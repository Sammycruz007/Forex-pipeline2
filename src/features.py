"""
Feature Store Module
Transforms raw OHLCV data into ML-ready feature matrix.
"""

import logging
import yaml
import pandas as pd
import numpy as np
import ta
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
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
    close = df["Close"]

    for lag in [1, 2, 3, 5, 10]:
        df[f"return_lag_{lag}"] = close.pct_change(periods=lag)

    df["hl_range"]      = (df["High"] - df["Low"]) / df["Close"]
    df["gap"]           = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)
    df["volatility_10"] = close.pct_change().rolling(window=10).std()
    df["volatility_20"] = close.pct_change().rolling(window=20).std()

    logger.info("Lag + volatility proxy features added")
    return df


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Market regime features.
    Helps model distinguish trending vs ranging,
    and bull vs bear conditions.
    """
    close = df["Close"]

    # Price position relative to MAs
    df["above_sma20"] = (close > df["sma_20"]).astype(int)
    df["above_sma50"] = (close > df["sma_50"]).astype(int)

    # Short MA above long MA = bullish regime
    df["sma_cross"] = (df["sma_10"] > df["sma_20"]).astype(int)

    # Consecutive up/down day streaks
    daily_return = close.pct_change()

    df["up_streak"] = (
        daily_return.gt(0)
        .groupby((daily_return.gt(0) != daily_return.gt(0).shift()).cumsum())
        .cumcount() + 1
    ) * daily_return.gt(0)

    df["down_streak"] = (
        daily_return.lt(0)
        .groupby((daily_return.lt(0) != daily_return.lt(0).shift()).cumsum())
        .cumcount() + 1
    ) * daily_return.lt(0)

    # RSI regime flags
    df["rsi_oversold"]   = (df["rsi_14"] < 30).astype(int)
    df["rsi_overbought"] = (df["rsi_14"] > 70).astype(int)

    # Volatility regime — is short-term vol above long-term vol?
    df["vol_expanding"] = (df["volatility_10"] > df["volatility_20"]).astype(int)

    logger.info("Regime features added")
    return df


def add_target(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
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
    config        = load_config(config_path)
    symbol        = config["data"]["symbol"]
    horizon       = config["features"]["target_horizon"]
    raw_path      = config["data"]["raw_path"]
    features_path = config["data"]["features_path"]

    raw_file = f"{raw_path}/{symbol.replace('=', '_')}_ohlcv.parquet"
    logger.info(f"Loading raw data from {raw_file}")
    df = pd.read_parquet(raw_file, engine="pyarrow")
    logger.info(f"Raw data shape: {df.shape}")

    # Pipeline — order matters
    df = add_trend_features(df)
    df = add_momentum_features(df)
    df = add_volatility_features(df)
    df = add_lag_features(df)
    df = add_regime_features(df)      # ← new
    df = add_target(df, horizon=horizon)

    before  = len(df)
    df      = df.dropna()
    dropped = before - len(df)
    logger.info(f"Dropped {dropped} NaN rows from indicator warmup")
    logger.info(f"Final feature matrix shape: {df.shape}")

    if len(df) == 0:
        nan_counts = df.isnull().sum()
        logger.error(f"Empty DataFrame. NaN counts:\n{nan_counts[nan_counts > 0]}")
        raise ValueError("Feature store is empty after dropna.")

    Path(features_path).mkdir(parents=True, exist_ok=True)
    output_file = f"{features_path}/{symbol.replace('=', '_')}_features.parquet"
    df.to_parquet(output_file, index=True, engine="pyarrow")
    logger.info(f"Feature store saved to {output_file}")

    return df


if __name__ == "__main__":
    df = build_feature_store()
    print(f"\nShape         : {df.shape}")
    print(f"Features      : {df.shape[1] - 2} (excl. target + future_return)")
    print(f"Date range    : {df.index[0].date()} → {df.index[-1].date()}")
    print(f"Target balance: {df['target'].value_counts().to_dict()}")
    print(f"Any NaNs      : {df.isnull().sum().sum()}")
