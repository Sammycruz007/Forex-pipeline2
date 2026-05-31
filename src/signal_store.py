"""
Signal Store — locks weekly signal on Sunday
API reads from here instead of recalculating live.
"""

import json
import logging
from pathlib import Path
from datetime import datetime, date
import pandas as pd

logger = logging.getLogger(__name__)

SIGNAL_FILE = "models/weekly_signal.json"


def save_weekly_signal(signal: dict) -> None:
    """Save signal with timestamp — called by pipeline on Sunday."""
    signal["locked_at"]    = datetime.utcnow().isoformat()
    signal["valid_from"]   = str(date.today())
    signal["valid_until"]  = str(
        pd.bdate_range(date.today(), periods=5)[-1].date()
    )

    with open(SIGNAL_FILE, "w") as f:
        json.dump(signal, f, indent=2)

    logger.info(f"Weekly signal locked until {signal['valid_until']}")


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
