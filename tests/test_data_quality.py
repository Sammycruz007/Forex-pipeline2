"""
Data Quality Tests
Validates the feature store and raw data integrity.
These run fast — no model loading needed.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture(scope="module")
def config():
    import yaml
    with open("config.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def feature_store(config):
    symbol = config["data"]["symbol"]
    path   = f"{config['data']['features_path']}/{symbol.replace('=','_')}_features.parquet"
    return pd.read_parquet(path, engine="pyarrow")


@pytest.fixture(scope="module")
def raw_data(config):
    symbol = config["data"]["symbol"]
    path   = f"{config['data']['raw_path']}/{symbol.replace('=','_')}_ohlcv.parquet"
    return pd.read_parquet(path, engine="pyarrow")


# ─────────────────────────────────────────────
# Raw data tests
# ─────────────────────────────────────────────

class TestRawData:

    def test_raw_file_exists(self, config):
        symbol = config["data"]["symbol"]
        path   = Path(f"{config['data']['raw_path']}/{symbol.replace('=','_')}_ohlcv.parquet")
        assert path.exists(), f"Raw data file not found: {path}"

    def test_raw_has_required_columns(self, raw_data):
        required = ["Open", "High", "Low", "Close", "Volume"]
        for col in required:
            assert col in raw_data.columns, f"Missing column: {col}"

    def test_raw_index_is_datetime(self, raw_data):
        assert isinstance(raw_data.index, pd.DatetimeIndex), \
            "Raw data index must be DatetimeIndex"

    def test_raw_index_is_sorted(self, raw_data):
        assert raw_data.index.is_monotonic_increasing, \
            "Raw data index must be chronologically sorted"

    def test_raw_no_duplicate_dates(self, raw_data):
        duplicates = raw_data.index.duplicated().sum()
        assert duplicates == 0, f"Found {duplicates} duplicate dates"

    def test_raw_close_is_positive(self, raw_data):
        assert (raw_data["Close"] > 0).all(), \
            "All Close prices must be positive"

    def test_raw_high_gte_low(self, raw_data):
        assert (raw_data["High"] >= raw_data["Low"]).all(), \
            "High must always be >= Low"

    def test_raw_sufficient_rows(self, raw_data):
        assert len(raw_data) >= 500, \
            f"Expected at least 500 rows, got {len(raw_data)}"


# ─────────────────────────────────────────────
# Feature store tests
# ─────────────────────────────────────────────

class TestFeatureStore:

    def test_feature_store_exists(self, config):
        symbol = config["data"]["symbol"]
        path   = Path(f"{config['data']['features_path']}/{symbol.replace('=','_')}_features.parquet")
        assert path.exists(), f"Feature store not found: {path}"

    def test_no_nans_in_features(self, feature_store):
        nan_counts = feature_store.isnull().sum()
        cols_with_nan = nan_counts[nan_counts > 0]
        assert len(cols_with_nan) == 0, \
            f"NaN values found in columns:\n{cols_with_nan}"

    def test_target_is_binary(self, feature_store):
        unique_vals = set(feature_store["target"].unique())
        assert unique_vals <= {0, 1}, \
            f"Target must be binary (0/1), got: {unique_vals}"

    def test_target_has_both_classes(self, feature_store):
        counts = feature_store["target"].value_counts()
        assert 0 in counts.index, "Target missing class 0 (Down)"
        assert 1 in counts.index, "Target missing class 1 (Up)"

    def test_target_not_too_imbalanced(self, feature_store):
        ratio = feature_store["target"].mean()
        assert 0.35 <= ratio <= 0.65, \
            f"Target severely imbalanced: {ratio:.2f} — expected between 0.35 and 0.65"

    def test_rsi_within_range(self, feature_store):
        if "rsi_14" in feature_store.columns:
            assert feature_store["rsi_14"].between(0, 100).all(), \
                "RSI values must be between 0 and 100"

    def test_index_is_chronological(self, feature_store):
        assert feature_store.index.is_monotonic_increasing, \
            "Feature store index must be chronologically sorted"

    def test_sufficient_training_samples(self, feature_store):
        assert len(feature_store) >= 500, \
            f"Need at least 500 samples for training, got {len(feature_store)}"

    def test_feature_count(self, feature_store):
        drop_cols    = ["target", "future_return", "Volume",
                        "Open", "High", "Low", "Close"]
        feature_cols = [c for c in feature_store.columns if c not in drop_cols]
        assert len(feature_cols) >= 30, \
            f"Expected at least 30 features, got {len(feature_cols)}"

    def test_no_infinite_values(self, feature_store):
        numeric = feature_store.select_dtypes(include=[np.number])
        inf_counts = np.isinf(numeric).sum().sum()
        assert inf_counts == 0, \
            f"Found {inf_counts} infinite values in feature store"

    def test_close_price_reasonable(self, feature_store):
        """EURUSD should be between 0.5 and 2.0"""
        if "Close" in feature_store.columns:
            assert feature_store["Close"].between(0.5, 2.0).all(), \
                "EURUSD Close prices outside reasonable range (0.5–2.0)"
