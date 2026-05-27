"""
FastAPI Trading Signal Server
Serves live directional signals for Forex pairs.
"""

import logging
import os
import os
from datetime import datetime
from typing import List, Dict

import yaml
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sys
sys.path.append(".")
from src.predict import generate_signal, load_model, load_config, load_latest_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────

app = FastAPI(
    title="Forex Trading Signal API",
    description="ML-powered directional signal for Forex pairs",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Response schemas
# ─────────────────────────────────────────────

class DailyForecast(BaseModel):
    day:       int
    direction: str
    up_prob:   float
    dn_prob:   float


class SignalResponse(BaseModel):
    symbol:          str
    as_of_date:      str
    current_price:   float
    signal:          str
    up_probability:  float
    dn_probability:  float
    confidence:      str
    horizon_days:    int
    daily_forecasts: List[DailyForecast]
    model_used:      str
    generated_at:    str


class HealthResponse(BaseModel):
    status:      str
    model_loaded: bool
    timestamp:   str


class HistoryResponse(BaseModel):
    symbol:  str
    records: int
    data:    List[Dict]


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/", tags=["Root"])
def root():
    return {
        "message": "Forex Trading Signal API",
        "docs":    "/docs",
        "signal":  "/signal",
        "health":  "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Check if the API and model are healthy."""
    try:
        config = load_config()
        load_model(config)
        model_loaded = True
    except Exception:
        model_loaded = False

    return HealthResponse(
        status      = "healthy" if model_loaded else "degraded",
        model_loaded = model_loaded,
        timestamp   = datetime.utcnow().isoformat()
    )


@app.get("/signal", response_model=SignalResponse, tags=["Signal"])
def get_signal():
    """
    Get the current trading signal for the configured Forex pair.
    Returns direction (UP/DOWN), probability, and 5-day forecast.
    """
    try:
        signal = generate_signal()
        signal["generated_at"] = datetime.utcnow().isoformat()
        return SignalResponse(**signal)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Signal generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history", response_model=HistoryResponse, tags=["Data"])
def get_history(days: int = 30):
    """
    Return recent OHLCV + feature data.
    Used by the Streamlit dashboard to draw the chart.
    """
    try:
        config = load_config()
        df     = load_latest_features(config)

        # Return last N days of price data
        recent = df[["Close", "High", "Low", "Open"]].tail(days).copy()
        recent.index = recent.index.astype(str)

        records = [
            {
                "date":  date,
                "open":  round(float(row["Open"]),  5),
                "high":  round(float(row["High"]),  5),
                "low":   round(float(row["Low"]),   5),
                "close": round(float(row["Close"]), 5),
            }
            for date, row in recent.iterrows()
        ]

        return HistoryResponse(
            symbol  = config["data"]["symbol"],
            records = len(records),
            data    = records
        )
    except Exception as e:
        logger.error(f"History fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/features/importance", tags=["Model"])
def get_feature_importance():
    """Return top 15 feature importances from the LightGBM model."""
    try:
        import joblib
        from pathlib import Path
        config     = load_config()
        model_path = Path(config["model"]["model_path"])
        model      = joblib.load(model_path / "lgbm_model.joblib")
        cols       = joblib.load(model_path / "feature_cols.joblib")

        importance = sorted(
            zip(cols, model.feature_importances_),
            key=lambda x: x[1],
            reverse=True
        )[:15]

        return {
            "features": [
                {"feature": f, "importance": int(i)}
                for f, i in importance
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
