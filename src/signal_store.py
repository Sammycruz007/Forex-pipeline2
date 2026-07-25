"""
Signal Store — locks weekly signal on Sunday
API reads from here instead of recalculating live.
"""

import json
import logging
from pathlib import Path
from datetime import datetime, date
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

SIGNAL_FILE = "models/weekly_signal.json"
DEFAULT_HORIZON_DAYS = 2  # fallback only — config is always preferred


def _load_target_horizon(config_path: str = "config.yaml") -> int:
    """
    Read features.target_horizon from config so the signal's validity
    window always matches what the model actually predicted, instead
    of a hardcoded business-day count that can silently drift out of
    sync with the model (previously hardcoded to 5, from the old
    5-day horizon).
    """
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return config["features"]["target_horizon"]
    except Exception as e:
        logger.warning(
            f"Could not read target_horizon from config ({e}); "
            f"falling back to {DEFAULT_HORIZON_DAYS} days"
        )
        return DEFAULT_HORIZON_DAYS


def save_weekly_signal(signal: dict, config_path: str = "config.yaml") -> None:
    """Save signal with timestamp — called by pipeline on Sunday."""
    horizon_days = _load_target_horizon(config_path)

    signal["locked_at"]    = datetime.utcnow().isoformat()
    signal["valid_from"]   = str(date.today())
    signal["valid_until"]  = str(
        pd.bdate_range(date.today(), periods=horizon_days)[-1].date()
    )

    with open(SIGNAL_FILE, "w") as f:
        json.dump(signal, f, indent=2)

    logger.info(
        f"Weekly signal locked until {signal['valid_until']} "
        f"(horizon: {horizon_days} business days)"
    )


def load_weekly_signal() -> dict:
    """Load the locked weekly signal."""
    path = Path(SIGNAL_FILE)
    if not path.exists():
        return None

    with open(path) as f:
        signal = json.load(f)

    logger.info(
        f"Loaded locked signal: {signal['signal']} "
        f"(valid until {signal.get('valid_until', 'unknown')})"
    )
    return signal


def is_signal_current() -> bool:
    """Check if locked signal is from this week."""
    path = Path(SIGNAL_FILE)
    if not path.exists():
        return False

    with open(path) as f:
        signal = json.load(f)

    valid_until = pd.to_datetime(signal.get("valid_until")).date()
    return date.today() <= valid_until
