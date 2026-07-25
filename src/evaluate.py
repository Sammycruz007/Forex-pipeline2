import os
"""
Evaluation Module
Standalone backtest used by both Airflow and GitHub Actions CI/CD.
Returns exit code 0 (pass) or 1 (fail) — critical for CI/CD gate.
"""

import sys
import logging
import yaml
import joblib
import mlflow
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

PROBA_THRESHOLD = 0.65


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_features(config: dict) -> pd.DataFrame:
    symbol        = config["data"]["symbol"]
    features_path = config["data"]["features_path"]
    path = f"{features_path}/{symbol.replace('=', '_')}_features.parquet"

    if not Path(path).exists():
        raise FileNotFoundError(f"Feature store not found at {path}. Run src/features.py first.")

    df = pd.read_parquet(path, engine="pyarrow")
    logger.info(f"Loaded feature store: {df.shape}")
    return df


def load_model(config: dict):
    model_path   = Path(config["model"]["model_path"])
    prod_path    = model_path / "production_model.joblib"
    xgb_path     = model_path / "xgb_model.joblib"
    name_path    = model_path / "production_model_name.txt"
    cols_path    = model_path / "feature_cols.joblib"

    if not cols_path.exists():
        raise FileNotFoundError("feature_cols.joblib not found. Run src/train.py first.")

    feature_cols = joblib.load(cols_path)

    if prod_path.exists():
        model = joblib.load(prod_path)
    elif xgb_path.exists():
        model = joblib.load(xgb_path)
    else:
        raise FileNotFoundError("No trained model found. Run src/train.py first.")

    if name_path.exists():
        model_name = name_path.read_text().strip()
    else:
        model_name = "XGBoost"

    logger.info(f"Loaded model: {model_name}")
    return model, feature_cols, model_name


def run_backtest(
    model,
    feature_cols: list,
    df: pd.DataFrame,
    test_size: float = 0.2
) -> Tuple[Dict, pd.DataFrame]:
    """
    Walk-forward backtest on the holdout test set.
    Always uses the LAST test_size% of data chronologically.
    Never shuffles — time order is sacred.

    Metric suite matches src/train.py's evaluate_model():
      - AUC-ROC, PR-AUC (threshold-independent, raw probabilities)
      - Precision measured ONLY on rows where proba >= PROBA_THRESHOLD
      - Recall, F1 (standard 0.5-threshold predictions)
      - Max / mean probability among predictions >= PROBA_THRESHOLD
    """
    drop_cols    = ["target", "future_return", "Volume",
                    "Open", "High", "Low", "Close"]
    valid_cols   = [c for c in feature_cols if c in df.columns]
    X            = df[valid_cols]
    y            = df["target"]

    split_idx    = int(len(X) * (1 - test_size))
    X_test       = X.iloc[split_idx:]
    y_test       = y.iloc[split_idx:]
    dates_test   = df.index[split_idx:]

    logger.info(f"Backtest period: {dates_test[0].date()} → {dates_test[-1].date()}")
    logger.info(f"Backtest samples: {len(X_test)}")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    high_conf_mask = y_prob >= PROBA_THRESHOLD
    n_high_conf    = int(high_conf_mask.sum())

    if n_high_conf > 0:
        precision_at_threshold = round(
            precision_score(y_test[high_conf_mask], y_pred[high_conf_mask]), 4
        )
        max_proba  = round(float(np.max(y_prob[high_conf_mask])), 4)
        mean_proba = round(float(np.mean(y_prob[high_conf_mask])), 4)
    else:
        precision_at_threshold = float("nan")
        max_proba  = float("nan")
        mean_proba = float("nan")

    metrics = {
        "auc_roc":                round(roc_auc_score(y_test, y_prob), 4),
        "pr_auc":                 round(average_precision_score(y_test, y_prob), 4),
        "precision_at_threshold": precision_at_threshold,
        "recall":                 round(recall_score(y_test, y_pred), 4),
        "f1_minority":            round(f1_score(y_test, y_pred, average="binary", pos_label=1), 4),
        "accuracy":               round(accuracy_score(y_test, y_pred), 4),
        "n_high_conf":            n_high_conf,
        "max_proba_high_conf":    max_proba,
        "mean_proba_high_conf":   mean_proba,
        "n_samples":              len(y_test),
        "n_correct":              int((y_pred == y_test).sum()),
        "test_start":             str(dates_test[0].date()),
        "test_end":               str(dates_test[-1].date()),
    }

    # Per-class breakdown
    cm = confusion_matrix(y_test, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics["down_recall"] = round(tn / (tn + fp) if (tn + fp) > 0 else 0, 4)
        metrics["up_recall"]   = round(tp / (tp + fn) if (tp + fn) > 0 else 0, 4)

    # Build results dataframe for detailed analysis
    results_df = pd.DataFrame({
        "date":      dates_test,
        "actual":    y_test.values,
        "predicted": y_pred,
        "up_prob":   y_prob,
        "correct":   (y_pred == y_test.values).astype(int)
    }).set_index("date")

    # Rolling accuracy — shows if model degrades over time
    results_df["rolling_acc_20"] = (
        results_df["correct"].rolling(20).mean()
    )

    logger.info(f"  AUC-ROC                    : {metrics['auc_roc']:.4f}")
    logger.info(f"  PR-AUC                     : {metrics['pr_auc']:.4f}")
    logger.info(f"  Precision (proba >= {PROBA_THRESHOLD}) : {metrics['precision_at_threshold']} "
                f"(n={n_high_conf})")
    logger.info(f"  Recall                     : {metrics['recall']:.4f}")
    logger.info(f"  Max proba  (>= {PROBA_THRESHOLD})       : {metrics['max_proba_high_conf']}")
    logger.info(f"  Mean proba (>= {PROBA_THRESHOLD})       : {metrics['mean_proba_high_conf']}")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=['Down', 'Up'])}")
    return metrics, results_df


def check_gate(metrics: Dict, config: dict) -> bool:
    """
    Quality gate — used by both Airflow and GitHub Actions.
    Returns True if model passes, False if it fails.
    Exit code matters for CI/CD: 0 = pass, 1 = fail.
    """
    min_acc       = config["thresholds"]["min_directional_accuracy"]
    min_f1        = config["thresholds"]["min_f1_score"]
    min_precision = config["thresholds"].get("min_precision_at_threshold", 0.0)

    acc_pass       = metrics["accuracy"]    >= min_acc
    f1_pass        = metrics["f1_minority"] >= min_f1
    precision_val  = metrics["precision_at_threshold"]
    precision_pass = (not np.isnan(precision_val)) and (precision_val >= min_precision)

    logger.info(f"\n{'='*55}")
    logger.info(f"  QUALITY GATE RESULTS")
    logger.info(f"{'='*55}")
    logger.info(f"  Directional Accuracy       : {metrics['accuracy']:.4f} "
                f"(min: {min_acc}) → {'PASS ✓' if acc_pass else 'FAIL ✗'}")
    logger.info(f"  F1 Minority Class          : {metrics['f1_minority']:.4f} "
                f"(min: {min_f1}) → {'PASS ✓' if f1_pass else 'FAIL ✗'}")
    logger.info(f"  Precision (proba >= {PROBA_THRESHOLD}) : {precision_val} "
                f"(min: {min_precision}) → {'PASS ✓' if precision_pass else 'FAIL ✗'}")
    logger.info(f"  AUC-ROC                    : {metrics['auc_roc']:.4f}")
    logger.info(f"  PR-AUC                     : {metrics['pr_auc']:.4f}")
    logger.info(f"  Up Recall                  : {metrics.get('up_recall', 0):.4f}")
    logger.info(f"  Down Recall                : {metrics.get('down_recall', 0):.4f}")
    logger.info(f"  Test Samples               : {metrics['n_samples']}")
    logger.info(f"  Correct Predictions        : {metrics['n_correct']}")
    logger.info(f"  Test Period                : {metrics['test_start']} → {metrics['test_end']}")
    logger.info(f"{'='*55}")

    passed = acc_pass and f1_pass and precision_pass
    logger.info(f"  OVERALL GATE: {'✅ PASSED' if passed else '❌ FAILED'}")
    logger.info(f"{'='*55}")
    return passed


def log_to_mlflow(
    metrics: Dict,
    model_name: str,
    gate_passed: bool,
    config: dict
) -> None:
    """Log backtest results to MLflow for audit trail."""
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "./mlruns"))
    mlflow.set_experiment(config["model"]["experiment_name"])

    with mlflow.start_run(run_name=f"backtest_{model_name}"):
        numeric_metrics = {
            k: v for k, v in metrics.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        mlflow.log_metrics(numeric_metrics)
        mlflow.log_param("model_name",   model_name)
        mlflow.log_param("gate_passed",  str(gate_passed))
        mlflow.log_param("test_start", str(metrics["test_start"]))
        mlflow.log_param("test_end", str(metrics["test_end"]))
    logger.info("Backtest results logged to MLflow")


def run_evaluation(
    config_path: str = "config.yaml",
    log_mlflow:  bool = True
) -> Tuple[Dict, bool]:
    """
    Full evaluation pipeline.
    Called by Airflow, GitHub Actions, and directly.
    """
    config   = load_config(config_path)
    df       = load_features(config)
    model, feature_cols, model_name = load_model(config)

    metrics, results_df = run_backtest(
        model, feature_cols, df,
        test_size=config["model"]["test_size"]
    )

    gate_passed = check_gate(metrics, config)

    # Persist metrics so the API (main.py) can serve them without
    # recomputing a full backtest on every /signal request.
    try:
        import json
        model_path = Path(config["model"]["model_path"])
        model_path.mkdir(parents=True, exist_ok=True)
        metrics_out = {
            k: (None if isinstance(v, float) and np.isnan(v) else v)
            for k, v in metrics.items()
        }
        with open(model_path / "model_metrics.json", "w") as f:
            json.dump(metrics_out, f, indent=2)
        logger.info(f"Metrics written to {model_path / 'model_metrics.json'}")
    except Exception as e:
        logger.warning(f"Could not write model_metrics.json (non-critical): {e}")

    if log_mlflow:
        try:
            log_to_mlflow(metrics, model_name, gate_passed, config)
        except Exception as e:
            logger.warning(f"MLflow logging failed (non-critical): {e}")

    return metrics, gate_passed


if __name__ == "__main__":
    metrics, passed = run_evaluation()

    print("\n--- Backtest Summary ---")
    for k, v in metrics.items():
        print(f"  {k:<25}: {v}")

    # Exit code for CI/CD
    # 0 = success (gate passed) → GitHub Actions marks step as green
    # 1 = failure (gate failed) → GitHub Actions marks step as red, blocks merge
    sys.exit(0 if passed else 1)
