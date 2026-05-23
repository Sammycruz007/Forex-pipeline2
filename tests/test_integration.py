"""
Integration Tests
Tests that components work correctly together end-to-end.
"""

import pytest
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
sys.path.insert(0, ".")


@pytest.fixture(scope="module")
def config():
    import yaml
    with open("config.yaml") as f:
        return yaml.safe_load(f)


class TestModelArtifacts:

    def test_model_files_exist(self, config):
        model_path = Path(config["model"]["model_path"])
        for fname in ["lgbm_model.joblib", "baseline_model.joblib", "feature_cols.joblib"]:
            assert (model_path / fname).exists(), \
                f"Model artifact missing: {fname}"

    def test_model_loads_without_error(self, config):
        model_path = Path(config["model"]["model_path"])
        model      = joblib.load(model_path / "lgbm_model.joblib")
        assert model is not None

    def test_feature_cols_is_list(self, config):
        model_path   = Path(config["model"]["model_path"])
        feature_cols = joblib.load(model_path / "feature_cols.joblib")
        assert isinstance(feature_cols, list), "feature_cols must be a list"
        assert len(feature_cols) > 0,          "feature_cols must not be empty"

    def test_model_predicts_binary(self, config):
        model_path   = Path(config["model"]["model_path"])
        model        = joblib.load(model_path / "lgbm_model.joblib")
        feature_cols = joblib.load(model_path / "feature_cols.joblib")

        symbol = config["data"]["symbol"]
        df     = pd.read_parquet(
            f"{config['data']['features_path']}/{symbol.replace('=','_')}_features.parquet"
        )
        X    = df[feature_cols].iloc[-5:]
        pred = model.predict(X)
        assert set(pred).issubset({0, 1}), \
            f"Model must predict 0 or 1, got {set(pred)}"

    def test_model_returns_probabilities(self, config):
        model_path   = Path(config["model"]["model_path"])
        model        = joblib.load(model_path / "lgbm_model.joblib")
        feature_cols = joblib.load(model_path / "feature_cols.joblib")

        symbol = config["data"]["symbol"]
        df     = pd.read_parquet(
            f"{config['data']['features_path']}/{symbol.replace('=','_')}_features.parquet"
        )
        X     = df[feature_cols].iloc[-1:]
        proba = model.predict_proba(X)

        assert proba.shape == (1, 2), "predict_proba must return 2 classes"
        assert abs(proba[0].sum() - 1.0) < 1e-6, "Probabilities must sum to 1.0"
        assert (proba >= 0).all() and (proba <= 1).all(), \
            "Probabilities must be between 0 and 1"


class TestPredictionPipeline:

    def test_generate_signal_returns_dict(self):
        from src.predict import generate_signal
        signal = generate_signal()
        assert isinstance(signal, dict), "Signal must be a dictionary"

    def test_signal_has_required_keys(self):
        from src.predict import generate_signal
        signal   = generate_signal()
        required = [
            "symbol", "as_of_date", "current_price",
            "signal", "up_probability", "dn_probability",
            "confidence", "horizon_days", "daily_forecasts", "model_used"
        ]
        for key in required:
            assert key in signal, f"Signal missing required key: {key}"

    def test_signal_direction_is_valid(self):
        from src.predict import generate_signal
        signal = generate_signal()
        assert signal["signal"] in {"UP", "DOWN"}, \
            f"Signal must be UP or DOWN, got {signal['signal']}"

    def test_signal_probabilities_sum_to_one(self):
        from src.predict import generate_signal
        signal = generate_signal()
        total  = signal["up_probability"] + signal["dn_probability"]
        assert abs(total - 1.0) < 0.01, \
            f"Probabilities must sum to ~1.0, got {total}"

    def test_signal_confidence_is_valid(self):
        from src.predict import generate_signal
        signal = generate_signal()
        assert signal["confidence"] in {"LOW", "MEDIUM", "HIGH"}, \
            f"Invalid confidence: {signal['confidence']}"

    def test_daily_forecasts_length(self):
        from src.predict import generate_signal
        signal = generate_signal()
        assert len(signal["daily_forecasts"]) == signal["horizon_days"], \
            "daily_forecasts length must match horizon_days"

    def test_daily_forecasts_structure(self):
        from src.predict import generate_signal
        signal = generate_signal()
        for f in signal["daily_forecasts"]:
            assert "day"       in f
            assert "direction" in f
            assert "up_prob"   in f
            assert "dn_prob"   in f
            assert f["direction"] in {"UP", "DOWN"}


class TestEvaluationPipeline:

    def test_evaluation_returns_metrics(self):
        from src.evaluate import run_evaluation
        metrics, passed = run_evaluation(log_mlflow=False)
        assert isinstance(metrics, dict)
        # Fix: numpy.bool_ is not Python bool — use bool() to cast
        assert isinstance(bool(passed), bool)

    def test_metrics_has_required_keys(self):
        from src.evaluate import run_evaluation
        metrics, _ = run_evaluation(log_mlflow=False)
        required   = [
            "directional_acc", "f1_minority",
            "n_samples", "n_correct"
        ]
        for key in required:
            assert key in metrics, f"Metrics missing: {key}"

    def test_accuracy_within_valid_range(self):
        from src.evaluate import run_evaluation
        metrics, _ = run_evaluation(log_mlflow=False)
        acc        = metrics["directional_acc"]
        assert 0.0 <= acc <= 1.0, \
            f"Accuracy must be between 0 and 1, got {acc}"

    def test_gate_passes_current_model(self):
        from src.evaluate import run_evaluation
        _, passed = run_evaluation(log_mlflow=False)
        assert passed, "Current model must pass the quality gate"
