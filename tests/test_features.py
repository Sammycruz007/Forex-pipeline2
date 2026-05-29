"""
Unit Tests - Feature Engineering
Tests each feature function in isolation with synthetic data.
"""

import pytest
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, ".")

from src.features import (
    add_trend_features,
    add_momentum_features,
    add_volatility_features,
    add_lag_features,
    add_regime_features,
    add_target,
)


@pytest.fixture
def sample_ohlcv():
    """200 rows of synthetic OHLCV — enough for all indicator warmup periods."""
    np.random.seed(42)
    n      = 200
    dates  = pd.date_range("2022-01-01", periods=n, freq="B")
    close  = 1.10 + np.cumsum(np.random.normal(0, 0.002, n))
    high   = close + np.abs(np.random.normal(0, 0.001, n))
    low    = close - np.abs(np.random.normal(0, 0.001, n))
    open_  = close + np.random.normal(0, 0.001, n)
    volume = np.zeros(n)

    df = pd.DataFrame({
        "Open": open_, "High": high, "Low": low,
        "Close": close, "Volume": volume
    }, index=dates)
    return df


@pytest.fixture
def sample_with_trend(sample_ohlcv):
    """OHLCV + trend features."""
    return add_trend_features(sample_ohlcv.copy())


@pytest.fixture
def sample_with_momentum(sample_with_trend):
    """OHLCV + trend + momentum features."""
    return add_momentum_features(sample_with_trend.copy())


@pytest.fixture
def sample_full(sample_with_momentum):
    """
    OHLCV + all features needed for regime tests.
    Regime features depend on: sma_10, sma_20, sma_50, rsi_14, volatility_10/20.
    """
    df = add_volatility_features(sample_with_momentum.copy())
    df = add_lag_features(df)
    return df


class TestTrendFeatures:

    def test_sma_columns_created(self, sample_with_trend):
        for col in ["sma_10", "sma_20", "sma_50"]:
            assert col in sample_with_trend.columns, f"Missing: {col}"

    def test_ema_columns_created(self, sample_with_trend):
        for col in ["ema_10"]:
            assert col in sample_with_trend.columns, f"Missing: {col}"

    def test_macd_columns_created(self, sample_with_trend):
        for col in ["macd_signal", "macd_histogram"]:
            assert col in sample_with_trend.columns, f"Missing: {col}"

    def test_sma_is_smooth(self, sample_with_trend):
        price_std = sample_with_trend["Close"].std()
        sma_std   = sample_with_trend["sma_20"].dropna().std()
        assert sma_std < price_std, "SMA should be smoother than raw price"

    def test_price_vs_sma_is_ratio(self, sample_with_trend):
        vals = sample_with_trend["price_vs_sma20"].dropna()
        assert vals.abs().max() < 0.5, "price_vs_sma20 should be a small ratio"


class TestMomentumFeatures:

    def test_rsi_columns_created(self, sample_with_momentum):
        for col in ["rsi_14"]:
            assert col in sample_with_momentum.columns

    def test_rsi_within_range(self, sample_with_momentum):
        rsi = sample_with_momentum["rsi_14"].dropna()
        assert rsi.between(0, 100).all(), \
            f"RSI out of range: min={rsi.min():.2f}, max={rsi.max():.2f}"

    def test_stoch_columns_created(self, sample_with_momentum):
        for col in ["stoch_k", "stoch_d"]:
            assert col in sample_with_momentum.columns

    def test_roc_columns_created(self, sample_with_momentum):
        for col in ["roc_5", "roc_10"]:
            assert col in sample_with_momentum.columns


class TestTarget:

    def test_target_is_binary(self, sample_with_trend):
        df     = add_target(sample_with_trend.copy(), horizon=5)
        df     = df.dropna()
        unique = set(df["target"].unique())
        assert unique <= {0, 1}, f"Target must be 0 or 1, got {unique}"

    def test_target_removes_last_n_rows(self, sample_with_trend):
        horizon = 5
        before  = len(sample_with_trend)
        df      = add_target(sample_with_trend.copy(), horizon=horizon)
        assert len(df) == before - horizon, \
            f"Expected {before - horizon} rows, got {len(df)}"

    def test_target_correct_direction(self):
        """Manually verify target logic with known prices."""
        dates = pd.date_range("2022-01-01", periods=10, freq="B")
        close = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
        df    = pd.DataFrame({
            "Open": close, "High": close, "Low": close,
            "Close": close, "Volume": [0]*10
        }, index=dates)
        df = add_target(df, horizon=5)
        # Row 0: close=1.0, close[5]=1.5 → UP → target=1
        assert df["target"].iloc[0] == 1, "Upward move should give target=1"

    def test_future_return_is_numeric(self, sample_with_trend):
        df = add_target(sample_with_trend.copy(), horizon=5)
        assert pd.api.types.is_numeric_dtype(df["future_return"]), \
            "future_return must be numeric"


class TestRegimeFeatures:

    def test_binary_regime_flags(self, sample_full):
        """Regime features need trend + momentum + lag features first."""
        df = add_regime_features(sample_full.copy())
        for col in ["above_sma20", "above_sma50", "sma_cross"]:
            unique = set(df[col].dropna().unique())
            assert unique <= {0, 1}, \
                f"{col} should be binary, got {unique}"

    def test_streaks_are_non_negative(self, sample_full):
        df = add_regime_features(sample_full.copy())
        assert (df["up_streak"]   >= 0).all(), "up_streak must be non-negative"
        assert (df["down_streak"] >= 0).all(), "down_streak must be non-negative"

    def test_rsi_flags_are_binary(self, sample_full):
        df = add_regime_features(sample_full.copy())
        for col in ["rsi_oversold", "rsi_overbought"]:
            unique = set(df[col].dropna().unique())
            assert unique <= {0, 1}, f"{col} must be binary"

    def test_vol_expanding_is_binary(self, sample_full):
        df = add_regime_features(sample_full.copy())
        unique = set(df["vol_expanding"].dropna().unique())
        assert unique <= {0, 1}, "vol_expanding must be binary"
