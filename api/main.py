"""
FastAPI Trading Signal Server
Serves live directional signals for Forex pairs.
"""

import logging
import os
from datetime import datetime
from typing import List, Dict, Optional

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

PROBA_THRESHOLD = 0.65


def load_backtest_metrics(config: dict) -> dict:
    """
    Read model_metrics.json written by evaluate.py's run_evaluation().
    Returns {} if the file doesn't exist yet (e.g. before the first
    evaluate.py run) so callers can degrade gracefully.
    """
    import json
    from pathlib import Path

    model_path   = Path(config["model"]["model_path"])
    metrics_path = model_path / "model_metrics.json"

    if not metrics_path.exists():
        logger.warning(f"model_metrics.json not found at {metrics_path}")
        return {}

    with open(metrics_path) as f:
        return json.load(f)

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

class Forecast(BaseModel):
    horizon_days: int
    direction:    str
    up_prob:      float
    dn_prob:      float


class SignalResponse(BaseModel):
    symbol:          str
    as_of_date:      str
    current_price:   float
    signal:          str
    up_probability:  float
    dn_probability:  float
    confidence:      str
    horizon_days:    int
    forecast:        Forecast
    model_used:      str
    generated_at:    str
    precision_at_threshold: Optional[float] = None
    proba_threshold:        Optional[float] = None
    accuracy:               Optional[float] = None


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
    Get the weekly trading signal — locked on Sunday, stable all week.
    Falls back to live calculation if no locked signal exists.
    """
    try:
        import sys
        sys.path.insert(0, ".")
        from src.signal_store import load_weekly_signal, is_signal_current

        # Try to serve locked weekly signal first
        if is_signal_current():
            signal = load_weekly_signal()
            signal["generated_at"] = datetime.utcnow().isoformat()
            logger.info("Serving locked weekly signal")
        else:
            # Fallback to live calculation
            signal = generate_signal()
            signal["generated_at"] = datetime.utcnow().isoformat()
            logger.info("Serving live signal (no locked signal found)")

        # Attach backtest-level precision (measured at proba >= threshold)
        # from evaluate.py's persisted metrics — this is NOT computed from
        # this single live prediction, it reflects the last backtest run.
        config           = load_config()
        backtest_metrics = load_backtest_metrics(config)
        signal["precision_at_threshold"] = backtest_metrics.get("precision_at_threshold")
        signal["proba_threshold"]        = PROBA_THRESHOLD
        signal["accuracy"]               = backtest_metrics.get("accuracy")

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
    """
    Return top 15 feature importances from the production XGBoost model.

    The production model is a CalibratedClassifierCV wrapping several
    fold-fitted XGBClassifier instances (one per inner_cv fold) rather
    than a single raw model — there's no top-level .feature_importances_.
    We average importances across each fold's base estimator for a
    stable ranking instead of arbitrarily picking one fold.
    """
    try:
        import joblib
        import numpy as np
        from pathlib import Path

        config     = load_config()
        model_path = Path(config["model"]["model_path"])
        prod_path  = model_path / "production_model.joblib"
        xgb_path   = model_path / "xgb_model.joblib"

        if prod_path.exists():
            model = joblib.load(prod_path)
        elif xgb_path.exists():
            model = joblib.load(xgb_path)
        else:
            raise FileNotFoundError("No trained model found. Run src/train.py first.")

        cols = joblib.load(model_path / "feature_cols.joblib")

        if hasattr(model, "calibrated_classifiers_"):
            # CalibratedClassifierCV — average importances across folds
            fold_importances = [
                cc.estimator.feature_importances_
                for cc in model.calibrated_classifiers_
            ]
            importances = np.mean(fold_importances, axis=0)
        elif hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        else:
            raise HTTPException(
                status_code=500,
                detail="Loaded model exposes no feature importances"
            )

        importance = sorted(
            zip(cols, importances),
            key=lambda x: x[1],
            reverse=True
        )[:15]

        return {
            "features": [
                {"feature": f, "importance": float(round(i, 6))}
                for f, i in importance
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
