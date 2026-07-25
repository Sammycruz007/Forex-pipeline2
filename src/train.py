"""
Production Training Pipeline
Trains XGBoost only, with fixed deterministic hyperparameters.
Calibrates probabilities with isotonic regression via out-of-fold
TimeSeriesSplit (CalibratedClassifierCV handles the base-model-fit +
calibration internally — no manual prefit split), matching the pattern
used in ml/signal_ranker.py.
Best (only) model promoted to production.
"""

import os
import glob
import logging
import warnings
import yaml
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

import xgboost as xgb
import mlflow

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

PROBA_THRESHOLD = 0.70

# Deterministic best params (from boosting_experiment.py)
XGB_BEST_PARAMS = {
    "objective":        "binary:logistic",
    "eval_metric":      "logloss",
    "verbosity":        0,
    "random_state":     42,
    "n_estimators":     400,
    "learning_rate":    0.02885863560550515,
    "max_depth":        4,
    "min_child_weight": 7,
    "subsample":        0.8914001595404614,
    "colsample_bytree": 0.7454571588517109,
    "reg_alpha":        0.029597654785768302,
    "reg_lambda":       0.28862852174095415,
    "scale_pos_weight": 1.2256482393129562,
}


# ─────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_features(config: dict) -> pd.DataFrame:
    symbol        = config["data"]["symbol"]
    features_path = config["data"]["features_path"]
    path = f"{features_path}/{symbol.replace('=','_')}_features.parquet"
    df   = pd.read_parquet(path, engine="pyarrow")
    logger.info(f"Loaded feature store: {df.shape}")
    return df


def prepare_data(df: pd.DataFrame, config: dict) -> Tuple:
    drop_cols    = ["target", "future_return", "Volume",
                    "Open", "High", "Low", "Close"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols]
    y = df["target"]
    logger.info(f"Features: {len(feature_cols)} | Samples: {len(X)}")
    return X, y


def time_series_split(X, y, test_size=0.2):
    split_idx        = int(len(X) * (1 - test_size))
    X_train, X_test  = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test  = y.iloc[:split_idx], y.iloc[split_idx:]
    logger.info(f"Train: {len(X_train)} | {X_train.index[0].date()} → {X_train.index[-1].date()}")
    logger.info(f"Test : {len(X_test)}  | {X_test.index[0].date()}  → {X_test.index[-1].date()}")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def evaluate_model(y_test, y_pred, y_pred_proba, model_name: str) -> Dict:
    """
    Full metric suite:
      - AUC-ROC, PR-AUC (computed on raw probabilities, threshold-independent)
      - Precision measured ONLY on rows where proba >= PROBA_THRESHOLD
      - Recall, F1 (on standard 0.5-threshold predictions, for reference)
      - Max / mean probability among predictions >= PROBA_THRESHOLD
    """
    high_conf_mask = y_pred_proba >= PROBA_THRESHOLD
    n_high_conf    = int(high_conf_mask.sum())

    if n_high_conf > 0:
        precision_at_threshold = round(
            precision_score(y_test[high_conf_mask], y_pred[high_conf_mask]), 4
        )
        max_proba  = round(float(np.max(y_pred_proba[high_conf_mask])), 4)
        mean_proba = round(float(np.mean(y_pred_proba[high_conf_mask])), 4)
    else:
        precision_at_threshold = float("nan")
        max_proba  = float("nan")
        mean_proba = float("nan")

    metrics = {
        "auc_roc":                round(roc_auc_score(y_test, y_pred_proba), 4),
        "pr_auc":                 round(average_precision_score(y_test, y_pred_proba), 4),
        "precision_at_threshold": precision_at_threshold,
        "recall":                 round(recall_score(y_test, y_pred), 4),
        "f1_minority":             round(f1_score(y_test, y_pred, average="binary", pos_label=1), 4),
        "accuracy":                round(accuracy_score(y_test, y_pred), 4),
        "n_high_conf":             n_high_conf,
        "max_proba_high_conf":     max_proba,
        "mean_proba_high_conf":    mean_proba,
    }

    logger.info(f"\n{'='*55}")
    logger.info(f"  {model_name}")
    logger.info(f"{'='*55}")
    logger.info(f"  AUC-ROC                    : {metrics['auc_roc']:.4f}")
    logger.info(f"  PR-AUC                     : {metrics['pr_auc']:.4f}")
    logger.info(f"  Precision (proba >= {PROBA_THRESHOLD}) : {metrics['precision_at_threshold']} "
                f"(n={n_high_conf})")
    logger.info(f"  Recall                     : {metrics['recall']:.4f}")
    logger.info(f"  F1 Score (Up class)        : {metrics['f1_minority']:.4f}")
    logger.info(f"  Max proba  (>= {PROBA_THRESHOLD})       : {metrics['max_proba_high_conf']}")
    logger.info(f"  Mean proba (>= {PROBA_THRESHOLD})       : {metrics['mean_proba_high_conf']}")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=['Down','Up'])}")
    return metrics


# ─────────────────────────────────────────────
# Baseline — Logistic Regression
# ─────────────────────────────────────────────

def train_baseline(X_train, y_train, X_test, y_test, config):
    tscv        = TimeSeriesSplit(n_splits=5, gap=10)
    best_weight = 1.0
    best_f1     = 0.0

    for up_weight in [1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0]:
        fold_f1s = []
        for train_idx, val_idx in tscv.split(X_train):
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=1000, random_state=42,
                    class_weight={0: 1.0, 1: up_weight}, C=0.1
                ))
            ])
            model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            preds = model.predict(X_train.iloc[val_idx])
            fold_f1s.append(f1_score(
                y_train.iloc[val_idx], preds,
                average="binary", pos_label=1
            ))
        mean_f1 = np.mean(fold_f1s)
        logger.info(f"  up_weight={up_weight:.1f} → CV F1: {mean_f1:.4f}")
        if mean_f1 > best_f1:
            best_f1     = mean_f1
            best_weight = up_weight

    logger.info(f"Best up_weight: {best_weight}")
    base = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000, random_state=42,
            class_weight={0: 1.0, 1: best_weight}, C=0.1
        ))
    ])

    calib_cv = TimeSeriesSplit(n_splits=5, gap=10)
    final = CalibratedClassifierCV(base, method="isotonic", cv=calib_cv)
    final.fit(X_train, y_train)

    y_proba = final.predict_proba(X_test)[:, 1]
    metrics = evaluate_model(y_test, final.predict(X_test), y_proba, "Logistic Regression (Baseline)")
    return final, metrics

# ─────────────────────────────────────────────
# XGBoost — deterministic best params + isotonic calibration
# ─────────────────────────────────────────────

def train_xgboost(X_train, y_train, X_test, y_test):
    """
    Trains XGBoost with fixed deterministic params, wrapped in isotonic
    calibration. Follows the same pattern as ml/signal_ranker.py:
    CalibratedClassifierCV is handed a TimeSeriesSplit directly and
    performs the base-model-fit + out-of-fold calibration internally
    in a single .fit() call — there's no separate manual prefit step,
    so the isotonic regressor is always calibrated on folds the base
    model didn't train on within that fold, without us having to carve
    out and permanently sacrifice a chunk of the training set up front.
    """
    logger.info("Training XGBoost (fixed deterministic params)...")

    gap = 10
    inner_cv = TimeSeriesSplit(n_splits=5, gap=gap)

    base_model = xgb.XGBClassifier(**XGB_BEST_PARAMS)

    calibrated_model = CalibratedClassifierCV(
        estimator=base_model, method="isotonic", cv=inner_cv
    )

    # Diagnostic: out-of-fold CV scores within the train window only.
    # These never touch X_test/y_test — purely a sanity check on
    # calibration-fold consistency before the final OOS evaluation below.
    cv_auc_roc = cross_val_score(
        calibrated_model, X_train, y_train, cv=inner_cv,
        scoring="roc_auc", n_jobs=-1
    )
    cv_pr_auc = cross_val_score(
        calibrated_model, X_train, y_train, cv=inner_cv,
        scoring="average_precision", n_jobs=-1
    )
    logger.info(
        f"  CV (train window only) | "
        f"AUC-ROC: {cv_auc_roc.mean():.4f} ± {cv_auc_roc.std():.4f} | "
        f"PR-AUC: {cv_pr_auc.mean():.4f} ± {cv_pr_auc.std():.4f} | gap={gap}"
    )

    # Final fit on the full training window — CalibratedClassifierCV
    # internally refits base models + isotonic regressors out-of-fold
    # across inner_cv, then this becomes the model used for OOS scoring.
    calibrated_model.fit(X_train, y_train)

    y_pred       = calibrated_model.predict(X_test)
    y_pred_proba = calibrated_model.predict_proba(X_test)[:, 1]
    metrics      = evaluate_model(y_test, y_pred, y_pred_proba, "XGBoost (Calibrated)")

    return calibrated_model, metrics


# ─────────────────────────────────────────────
# Model Selection & Promotion
# ─────────────────────────────────────────────

def cleanup_old_models(model_path: Path):
    """
    Delete previously saved model artifacts before writing new ones.
    Prevents stale files / git merge conflicts when CI commits model
    files back to the repo.
    """
    patterns = ["*.joblib", "*.cbm", "*.txt"]
    removed  = []
    for pattern in patterns:
        for f in glob.glob(str(model_path / pattern)):
            os.remove(f)
            removed.append(f)
    if removed:
        logger.info(f"Removed {len(removed)} old model artifact(s): {removed}")
    else:
        logger.info("No old model artifacts found to remove.")


def promote_winner(
    models: dict,
    metrics: dict,
    feature_cols: list,
    config: dict
) -> Tuple[str, float]:
    """
    XGBoost is the only trained model — promote it directly to production.
    """
    model_path = Path(config["model"]["model_path"])
    model_path.mkdir(parents=True, exist_ok=True)

    cleanup_old_models(model_path)

    winner_name = "XGBoost"
    winner_acc  = metrics[winner_name]["accuracy"]

    logger.info(f"\n{'='*55}")
    logger.info(f"  MODEL SELECTION RESULTS")
    logger.info(f"{'='*55}")
    logger.info(f"  {'Model':<28} {'Accuracy':>10} {'F1 Up':>10}")
    logger.info(f"  {'-'*50}")
    for name, m in metrics.items():
        marker = " ← PROMOTED" if name == winner_name else ""
        logger.info(f"  {name:<28} {m['accuracy']*100:>9.2f}% {m['f1_minority']:>10.4f}{marker}")
    logger.info(f"{'='*55}")

    # Save winner as production model (and backward-compatible aliases)
    joblib.dump(models[winner_name], model_path / "production_model.joblib")
    joblib.dump(models[winner_name], model_path / "xgb_model.joblib")

    with open(model_path / "production_model_name.txt", "w") as f:
        f.write(winner_name)

    joblib.dump(feature_cols, model_path / "feature_cols.joblib")

    logger.info(f"Winner '{winner_name}' promoted to production")
    logger.info(f"All models saved to {model_path}/")

    return winner_name, winner_acc


# ─────────────────────────────────────────────
# Quality Gate
# ─────────────────────────────────────────────

def check_gate(metrics: Dict, config: dict, model_name: str) -> bool:
    min_acc       = config["thresholds"]["min_directional_accuracy"]
    min_f1        = config["thresholds"]["min_f1_score"]
    min_precision = config["thresholds"].get("min_precision_at_threshold", 0.0)

    acc_pass       = metrics["accuracy"]                >= min_acc
    f1_pass        = metrics["f1_minority"]             >= min_f1
    precision_val  = metrics["precision_at_threshold"]
    precision_pass = (not np.isnan(precision_val)) and (precision_val >= min_precision)
    passed         = acc_pass and f1_pass and precision_pass

    logger.info(f"\n{'='*55}")
    logger.info(f"  Quality Gate — {model_name} (Production)")
    logger.info(f"{'='*55}")
    logger.info(f"  Directional Accuracy       : {metrics['accuracy']:.4f} "
                f"(min: {min_acc}) → {'PASS ✓' if acc_pass else 'FAIL ✗'}")
    logger.info(f"  F1 Minority Class          : {metrics['f1_minority']:.4f} "
                f"(min: {min_f1}) → {'PASS ✓' if f1_pass else 'FAIL ✗'}")
    logger.info(f"  Precision (proba >= {PROBA_THRESHOLD}) : {precision_val} "
                f"(min: {min_precision}) → {'PASS ✓' if precision_pass else 'FAIL ✗'}")
    logger.info(f"  Overall Gate               : {'PASSED ✓' if passed else 'FAILED ✗'}")
    return passed


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def run_training(config_path: str = "config.yaml") -> Dict:
    config = load_config(config_path)
    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", "./mlruns")
    )

    df                               = load_features(config)
    X, y                             = prepare_data(df, config)
    X_train, X_test, y_train, y_test = time_series_split(
        X, y, test_size=config["model"]["test_size"]
    )

    all_models  = {}
    all_metrics = {}

    logger.info("\n" + "="*55)
    logger.info("  TRAINING MODELS")
    logger.info("="*55)

    baseline, baseline_metrics = train_baseline(X_train, y_train, X_test, y_test, config)
    all_models["Baseline"]     = baseline
    all_metrics["Baseline"]    = baseline_metrics

    xgb_model, xgb_metrics = train_xgboost(X_train, y_train, X_test, y_test)
    all_models["XGBoost"]  = xgb_model
    all_metrics["XGBoost"] = xgb_metrics

    # Promote winner (XGBoost only)
    winner_name, winner_acc = promote_winner(
        all_models, all_metrics,
        list(X_train.columns), config
    )

    # Gate on winner only
    gate_passed = check_gate(all_metrics[winner_name], config, winner_name)

    # Generate and lock weekly signal after successful training
    try:
        import sys
        sys.path.insert(0, ".")
        from src.predict import generate_signal
        from src.signal_store import save_weekly_signal
        signal = generate_signal()
        save_weekly_signal(signal)
        logger.info(f"Weekly signal locked: {signal['signal']} "
                    f"{signal['confidence']} ({signal['up_probability']*100:.1f}%)")
    except Exception as e:
        logger.warning(f"Signal locking failed (non-critical): {e}")

    print(f"\nBaseline Accuracy               : {baseline_metrics['accuracy']*100:.2f}%")
    print(f"XGBoost  Accuracy                : {xgb_metrics['accuracy']*100:.2f}%")
    print(f"XGBoost  AUC-ROC                 : {xgb_metrics['auc_roc']:.4f}")
    print(f"XGBoost  PR-AUC                  : {xgb_metrics['pr_auc']:.4f}")
    print(f"XGBoost  Precision (>= {PROBA_THRESHOLD})     : {xgb_metrics['precision_at_threshold']}")
    print(f"XGBoost  Recall                  : {xgb_metrics['recall']:.4f}")
    print(f"XGBoost  F1 (Up)                 : {xgb_metrics['f1_minority']:.4f}")
    print(f"XGBoost  Max proba (>= {PROBA_THRESHOLD})     : {xgb_metrics['max_proba_high_conf']}")
    print(f"XGBoost  Mean proba (>= {PROBA_THRESHOLD})    : {xgb_metrics['mean_proba_high_conf']}")
    print(f"Winner                           : {winner_name} ({winner_acc*100:.2f}%)")
    print(f"Gate Passed                      : {gate_passed}")

    return {
        "all_metrics":  all_metrics,
        "winner":       winner_name,
        "winner_acc":   winner_acc,
        "gate_passed":  gate_passed,
        "feature_cols": list(X_train.columns)
    }


if __name__ == "__main__":
    run_training()
