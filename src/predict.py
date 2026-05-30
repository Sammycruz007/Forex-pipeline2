"""
Prediction Module
Loads trained model and generates trading signals.
Uses raw OHLCV for inference to get the freshest possible signal.
"""

import logging
import joblib
import yaml
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict

sys.path.insert(0, ".")

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model(config: dict):
    """
    Load production model — whichever won the last training run.
    Reads production_model_name.txt to know which model won.
    Falls back to lgbm_model.joblib for backward compatibility.
    """
    model_path    = Path(config["model"]["model_path"])
    cols_path     = model_path / "feature_cols.joblib"
    name_path     = model_path / "production_model_name.txt"
    prod_path     = model_path / "production_model.joblib"

    # Read winner name
    if name_path.exists():
        with open(name_path) as f:
            model_name = f.read().strip()
    else:
        model_name = "LightGBM"

    # Load production model
    if prod_path.exists():
        model = joblib.load(prod_path)
    elif (model_path / "lgbm_model.joblib").exists():
        model      = joblib.load(model_path / "lgbm_model.joblib")
        model_name = "LightGBM"
    else:
        raise FileNotFoundError("No trained model found. Run src/train.py first.")

    feature_cols = joblib.load(cols_path)
    logger.info(f"Loaded production model: {model_name}")
    return model, feature_cols, model_name


def load_inference_features(config: dict) -> pd.DataFrame:
    """
    Build features from raw OHLCV WITHOUT dropping the last 5 rows.
    Training drops the last 5 rows because they have no future target.
    For inference we don't need a target — we need the freshest features.
    This gives us signals based on the most recent available close.
    """
    from src.features import (
        add_trend_features,
        add_momentum_features,
        add_volatility_features,
        add_lag_features,
        add_regime_features,
        add_macro_features,
    )

    symbol   = config["data"]["symbol"]
    raw_path = config["data"]["raw_path"]

    # Load raw OHLCV
    raw_file = f"{raw_path}/{symbol.replace('=', '_')}_ohlcv.parquet"
    df       = pd.read_parquet(raw_file, engine="pyarrow")

    # Build all features — no target, no row dropping
    df = add_trend_features(df)
    df = add_momentum_features(df)
    df = add_volatility_features(df)
    df = add_lag_features(df)
    df = add_regime_features(df)
    df = add_macro_features(df, config)

    # Drop only NaN warmup rows
    df = df.dropna()

    logger.info(f"Inference features built | Latest date: {df.index[-1].date()}")
    logger.info(f"Shape: {df.shape}")
    return df


def load_latest_features(config: dict) -> pd.DataFrame:
    """Load the training feature store (fallback)."""
    symbol        = config["data"]["symbol"]
    features_path = config["data"]["features_path"]
    path = f"{features_path}/{symbol.replace('=', '_')}_features.parquet"
    df   = pd.read_parquet(path, engine="pyarrow")
    return df


def generate_signal(config_path: str = "config.yaml") -> Dict:
    """
    Generate trading signal for the next N days.
    Uses freshest available features for inference.
    """
    config                          = load_config(config_path)
    model, feature_cols, model_name = load_model(config)

    # Use inference features — freshest available data
    try:
        df = load_inference_features(config)
    except Exception as e:
        logger.warning(f"Inference features failed ({e}), falling back to feature store")
        df = load_latest_features(config)

    # Align to model's expected feature columns
    valid_cols    = [c for c in feature_cols if c in df.columns]
    latest        = df[valid_cols].iloc[-1:]
    latest_date   = df.index[-1]
    current_price = float(df["Close"].iloc[-1])

    # Predict
    direction_pred  = int(model.predict(latest)[0])
    proba           = model.predict_proba(latest)[0]
    up_probability  = float(round(proba[1], 4))
    dn_probability  = float(round(proba[0], 4))
    direction_label = "UP" if direction_pred == 1 else "DOWN"

    # Confidence
    confidence = max(up_probability, dn_probability)
    if confidence >= 0.65:
        confidence_label = "HIGH"
    elif confidence >= 0.55:
        confidence_label = "MEDIUM"
    else:
        confidence_label = "LOW"

    # 5-day forecast
    horizon          = config["features"]["target_horizon"]
    feature_window   = df[valid_cols].copy()
    daily_forecasts  = []

    for day in range(1, horizon + 1):
        row       = feature_window.iloc[-1:]
        day_pred  = int(model.predict(row)[0])
        day_proba = model.predict_proba(row)[0]
        daily_forecasts.append({
            "day":       day,
            "direction": "UP" if day_pred == 1 else "DOWN",
            "up_prob":   float(round(day_proba[1], 4)),
            "dn_prob":   float(round(day_proba[0], 4)),
        })

    return {
        "symbol":          config["data"]["symbol"],
        "as_of_date":      str(latest_date.date()),
        "current_price":   current_price,
        "signal":          direction_label,
        "up_probability":  up_probability,
        "dn_probability":  dn_probability,
        "confidence":      confidence_label,
        "horizon_days":    horizon,
        "daily_forecasts": daily_forecasts,
        "model_used":      model_name,
    }


if __name__ == "__main__":
    import json
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    signal = generate_signal()
    print(json.dumps(signal, indent=2))
