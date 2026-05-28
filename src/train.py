import os
"""
Model Training Module
Primary: Logistic Regression ensemble with tuned class weights
Secondary: LightGBM with Optuna
All experiments logged to MLflow.
"""

import logging
import warnings
import yaml
import joblib
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report
)
from sklearn.model_selection import TimeSeriesSplit

import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_features(config: dict) -> pd.DataFrame:
    symbol        = config["data"]["symbol"]
    features_path = config["data"]["features_path"]
    path = f"{features_path}/{symbol.replace('=', '_')}_features.parquet"
    df   = pd.read_parquet(path, engine="pyarrow")
    logger.info(f"Loaded feature store: {df.shape}")
    return df


def prepare_data(df: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, pd.Series]:
    # Explicitly define kept features after V2 pruning experiment
    # Removed: zero-importance and redundant features
    # rsi_7, ema_20, macd, gap, return_lag_1
    # above_sma20, above_sma50, sma_cross (redundant with price_vs_sma)
    # rsi_oversold, rsi_overbought (redundant with rsi_14)
    # vol_expanding (redundant with volatility_10/20)
    # up_streak, down_streak (noisy)
    # gold_above_sma20, dxy_above_sma20, tnx_above_sma20 (redundant with rsi)
    # tnx_return_1d (borderline, Tier 2)

    drop_cols = [
        "target", "future_return", "Volume",
        "Open", "High", "Low", "Close"
    ]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols]
    y = df["target"]
    logger.info(f"Features: {len(feature_cols)} | Samples: {len(X)}")
    return X, y


def time_series_train_test_split(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2
) -> Tuple:
    split_idx        = int(len(X) * (1 - test_size))
    X_train, X_test  = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test  = y.iloc[:split_idx], y.iloc[split_idx:]
    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)}")
    logger.info(f"Train: {X_train.index[0].date()} → {X_train.index[-1].date()}")
    logger.info(f"Test : {X_test.index[0].date()}  → {X_test.index[-1].date()}")
    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test, model_name) -> Dict[str, float]:
    y_pred  = model.predict(X_test)
    metrics = {
        "accuracy":        round(accuracy_score(y_test, y_pred), 4),
        "f1_score":        round(f1_score(y_test, y_pred, average="weighted"), 4),
        "f1_minority":     round(f1_score(y_test, y_pred, average="binary", pos_label=1), 4),
        "directional_acc": round(accuracy_score(y_test, y_pred), 4),
    }
    logger.info(f"\n{'='*50}")
    logger.info(f"  {model_name} Results")
    logger.info(f"{'='*50}")
    logger.info(f"  Directional Accuracy : {metrics['accuracy']*100:.2f}%")
    logger.info(f"  F1 Score (weighted)  : {metrics['f1_score']:.4f}")
    logger.info(f"  F1 Score (Up class)  : {metrics['f1_minority']:.4f}")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=['Down','Up'])}")
    return metrics


# ─────────────────────────────────────────────
# Baseline — tuned class weights via CV
# ─────────────────────────────────────────────

def find_best_class_weight(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict
) -> float:
    """
    Search for the class weight on the Up class that
    maximises F1 minority on walk-forward CV folds.
    """
    tscv         = TimeSeriesSplit(n_splits=5)
    best_weight  = 1.0
    best_f1      = 0.0

    # Try a range of weights for the Up (minority) class
    for up_weight in [1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0]:
        fold_f1s = []
        for train_idx, val_idx in tscv.split(X_train):
            X_f = X_train.iloc[train_idx]
            y_f = y_train.iloc[train_idx]
            X_v = X_train.iloc[val_idx]
            y_v = y_train.iloc[val_idx]

            model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=1000,
                    random_state=config["model"]["random_state"],
                    class_weight={0: 1.0, 1: up_weight},
                    C=0.1
                ))
            ])
            model.fit(X_f, y_f)
            preds = model.predict(X_v)
            fold_f1s.append(
                f1_score(y_v, preds, average="binary", pos_label=1)
            )

        mean_f1 = np.mean(fold_f1s)
        logger.info(f"  up_weight={up_weight:.1f} → CV F1 minority: {mean_f1:.4f}")

        if mean_f1 > best_f1:
            best_f1     = mean_f1
            best_weight = up_weight

    logger.info(f"Best up_weight: {best_weight} (CV F1: {best_f1:.4f})")
    return best_weight


def train_baseline(
    X_train, y_train, X_test, y_test, config
) -> Tuple[Pipeline, Dict]:
    logger.info("Finding best class weight for Logistic Regression...")
    mlflow.set_experiment(config["model"]["experiment_name"])

    best_weight = find_best_class_weight(X_train, y_train, config)

    with mlflow.start_run(run_name="baseline_logistic_regression"):
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                random_state=config["model"]["random_state"],
                class_weight={0: 1.0, 1: best_weight},
                C=0.1
            ))
        ])
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test, "Logistic Regression")
        mlflow.log_params({
            "model_type":   "logistic_regression",
            "C":            0.1,
            "up_weight":    best_weight
        })
        mlflow.log_metrics(metrics)
        # Model saved via joblib — skip mlflow artifact logging in CI

    return model, metrics


# ─────────────────────────────────────────────
# LightGBM + Optuna
# ─────────────────────────────────────────────

def optuna_objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict
) -> float:
    params = {
        "objective":         "binary",
        "metric":            "binary_logloss",
        "verbosity":         -1,
        "boosting_type":     "gbdt",
        "random_state":      config["model"]["random_state"],
        "n_estimators":      trial.suggest_int("n_estimators", 50, 300),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves":        trial.suggest_int("num_leaves", 15, 50),
        "max_depth":         trial.suggest_int("max_depth", 3, 6),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
        "subsample":         trial.suggest_float("subsample", 0.6, 0.9),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 0.9),
        "reg_alpha":         trial.suggest_float("reg_alpha", 0.1, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        # Also tune class weight for Up class
        "scale_pos_weight":  trial.suggest_float("scale_pos_weight", 0.8, 2.0),
    }

    tscv   = TimeSeriesSplit(n_splits=5)
    scores = []

    for train_idx, val_idx in tscv.split(X_train):
        X_f_train = X_train.iloc[train_idx]
        y_f_train = y_train.iloc[train_idx]
        X_f_val   = X_train.iloc[val_idx]
        y_f_val   = y_train.iloc[val_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_f_train, y_f_train,
            eval_set=[(X_f_val, y_f_val)],
            callbacks=[lgb.early_stopping(30, verbose=False),
                       lgb.log_evaluation(-1)]
        )
        # Optimise for balanced F1 — not just accuracy
        preds  = model.predict(X_f_val)
        f1     = f1_score(y_f_val, preds, average="macro")
        scores.append(f1)

    return np.mean(scores)


def train_lightgbm(
    X_train, y_train, X_test, y_test, config
) -> Tuple[lgb.LGBMClassifier, Dict, pd.DataFrame]:
    logger.info("Starting Optuna search for LightGBM...")
    mlflow.set_experiment(config["model"]["experiment_name"])

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42)
    )
    study.optimize(
        lambda trial: optuna_objective(trial, X_train, y_train, config),
        n_trials=config["model"]["lightgbm"]["n_trials"],
        timeout=config["model"]["lightgbm"]["timeout"],
        show_progress_bar=True
    )

    best_params = study.best_params
    logger.info(f"Best CV Macro F1 : {study.best_value:.4f}")
    logger.info(f"Best params      : {best_params}")

    with mlflow.start_run(run_name="lightgbm_optuna"):
        final_params = {
            "objective":    "binary",
            "verbosity":    -1,
            "random_state": config["model"]["random_state"],
            **best_params
        }
        final_model = lgb.LGBMClassifier(**final_params)
        final_model.fit(X_train, y_train)

        metrics = evaluate_model(final_model, X_test, y_test, "LightGBM + Optuna")
        metrics["cv_macro_f1"] = round(study.best_value, 4)

        mlflow.log_params(final_params)
        mlflow.log_metrics(metrics)
        # Model saved via joblib — skip mlflow artifact logging in CI

        importance_df = pd.DataFrame({
            "feature":    X_train.columns,
            "importance": final_model.feature_importances_
        }).sort_values("importance", ascending=False)

        logger.info(f"\nTop 10 Features:\n{importance_df.head(10).to_string(index=False)}")

    return final_model, metrics, importance_df


# ─────────────────────────────────────────────
# Save & Gate
# ─────────────────────────────────────────────

def save_models(baseline, lgbm_model, feature_cols, config) -> None:
    model_path = Path(config["model"]["model_path"])
    model_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(baseline,     model_path / "baseline_model.joblib")
    joblib.dump(lgbm_model,   model_path / "lgbm_model.joblib")
    joblib.dump(feature_cols, model_path / "feature_cols.joblib")
    logger.info(f"Models saved to {model_path}/")


def check_quality_gate(metrics: Dict, config: dict, model_name: str) -> bool:
    min_acc = config["thresholds"]["min_directional_accuracy"]
    min_f1  = config["thresholds"]["min_f1_score"]
    acc_pass = metrics["directional_acc"] >= min_acc
    f1_pass  = metrics["f1_minority"]     >= min_f1

    logger.info(f"\n{'='*50}")
    logger.info(f"  Quality Gate — {model_name}")
    logger.info(f"{'='*50}")
    logger.info(f"  Directional Accuracy : {metrics['directional_acc']:.4f} "
                f"(min: {min_acc}) → {'PASS ✓' if acc_pass else 'FAIL ✗'}")
    logger.info(f"  F1 Minority Class    : {metrics['f1_minority']:.4f} "
                f"(min: {min_f1}) → {'PASS ✓' if f1_pass else 'FAIL ✗'}")
    passed = acc_pass and f1_pass
    logger.info(f"  Overall Gate         : {'PASSED ✓' if passed else 'FAILED ✗'}")
    return passed


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def run_training(config_path: str = "config.yaml") -> Dict:
    config = load_config(config_path)
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "./mlruns"))

    df                               = load_features(config)
    X, y                             = prepare_data(df, config)
    X_train, X_test, y_train, y_test = time_series_train_test_split(
        X, y, test_size=config["model"]["test_size"]
    )

    baseline_model, baseline_metrics = train_baseline(
        X_train, y_train, X_test, y_test, config
    )
    lgbm_model, lgbm_metrics, importance_df = train_lightgbm(
        X_train, y_train, X_test, y_test, config
    )

    save_models(baseline_model, lgbm_model, list(X_train.columns), config)

    # Gate both models — report which passes
    baseline_gate = check_quality_gate(baseline_metrics, config, "Logistic Regression")
    lgbm_gate     = check_quality_gate(lgbm_metrics,     config, "LightGBM")
    gate_passed   = baseline_gate or lgbm_gate

    return {
        "baseline_metrics": baseline_metrics,
        "lgbm_metrics":     lgbm_metrics,
        "gate_passed":      gate_passed,
        "feature_cols":     list(X_train.columns),
        "importance":       importance_df
    }


if __name__ == "__main__":
    results = run_training()
    print(f"\nBaseline Accuracy : {results['baseline_metrics']['accuracy']*100:.2f}%")
    print(f"LightGBM Accuracy : {results['lgbm_metrics']['accuracy']*100:.2f}%")
    print(f"Gate Passed       : {results['gate_passed']}")
