"""
Prediction Module
Loads trained model and generates trading signals.
"""

import logging
import joblib
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        import yaml
        return yaml.safe_load(f)


def load_model(config: dict):
    """Load the best model — LightGBM if available, else baseline."""
    model_path = Path(config["model"]["model_path"])

    lgbm_path     = model_path / "lgbm_model.joblib"
    baseline_path = model_path / "baseline_model.joblib"
    cols_path     = model_path / "feature_cols.joblib"

    if lgbm_path.exists():
        model = joblib.load(lgbm_path)
        model_name = "LightGBM"
    elif baseline_path.exists():
        model = joblib.load(baseline_path)
        model_name = "LogisticRegression"
    else:
        raise FileNotFoundError("No trained model found. Run src/train.py first.")

    feature_cols = joblib.load(cols_path)
    logger.info(f"Loaded model: {model_name}")
    return model, feature_cols, model_name


def load_latest_features(config: dict) -> pd.DataFrame:
    """Load the feature store."""
    symbol        = config["data"]["symbol"]
    features_path = config["data"]["features_path"]
    path = f"{features_path}/{symbol.replace('=', '_')}_features.parquet"
    df   = pd.read_parquet(path, engine="pyarrow")
    return df


def generate_signal(config_path: str = "config.yaml") -> Dict:
    """
    Generate trading signal for the next N days.
    Returns direction, probability, and per-day forecasts.
    """
    config                       = load_config(config_path)
    model, feature_cols, model_name = load_model(config)
    df                           = load_latest_features(config)

    # Use the most recent row as the current state
    latest         = df[feature_cols].iloc[-1:]
    latest_date    = df.index[-1]
    current_price  = float(df["Close"].iloc[-1])

    # Predict direction and probability
    direction_pred = int(model.predict(latest)[0])
    proba          = model.predict_proba(latest)[0]
    up_probability = float(round(proba[1], 4))
    dn_probability = float(round(proba[0], 4))

    direction_label = "UP" if direction_pred == 1 else "DOWN"

    # Confidence level
    confidence = max(up_probability, dn_probability)
    if confidence >= 0.65:
        confidence_label = "HIGH"
    elif confidence >= 0.55:
        confidence_label = "MEDIUM"
    else:
        confidence_label = "LOW"

    # Generate 5-day forecast using rolling predictions
    horizon       = config["features"]["target_horizon"]
    daily_forecasts = []
    feature_window  = df[feature_cols].copy()

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
    signal = generate_signal()
    print(json.dumps(signal, indent=2))
