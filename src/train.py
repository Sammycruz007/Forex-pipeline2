"""
Production Training Pipeline
Trains LightGBM, XGBoost, and CatBoost.
Best model automatically promoted to production.
"""

import os
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
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import TimeSeriesSplit

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import optuna
from optuna.samplers import TPESampler
import mlflow

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


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


def evaluate_model(y_test, y_pred, model_name: str) -> Dict:
    metrics = {
        "accuracy":        round(accuracy_score(y_test, y_pred), 4),
        "f1_weighted":     round(f1_score(y_test, y_pred, average="weighted"), 4),
        "f1_minority":     round(f1_score(y_test, y_pred, average="binary", pos_label=1), 4),
        "directional_acc": round(accuracy_score(y_test, y_pred), 4),
    }
    logger.info(f"\n{'='*55}")
    logger.info(f"  {model_name}")
    logger.info(f"{'='*55}")
    logger.info(f"  Directional Accuracy : {metrics['accuracy']*100:.2f}%")
    logger.info(f"  F1 Score (Up class)  : {metrics['f1_minority']:.4f}")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=['Down','Up'])}")
    return metrics


# ─────────────────────────────────────────────
# Baseline — Logistic Regression
# ─────────────────────────────────────────────

def train_baseline(X_train, y_train, X_test, y_test, config):
    tscv        = TimeSeriesSplit(n_splits=5)
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
    final = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000, random_state=42,
            class_weight={0: 1.0, 1: best_weight}, C=0.1
        ))
    ])
    final.fit(X_train, y_train)
    metrics = evaluate_model(y_test, final.predict(X_test), "Logistic Regression")
    return final, metrics


# ─────────────────────────────────────────────
# LightGBM — deterministic best params
# ─────────────────────────────────────────────

def train_lightgbm(X_train, y_train, X_test, y_test):
    logger.info("Training LightGBM...")
    params = {
        "objective":         "binary",
        "verbosity":         -1,
        "random_state":      42,
        "n_estimators":      235,
        "learning_rate":     0.013805483481639545,
        "num_leaves":        39,
        "max_depth":         4,
        "min_child_samples": 98,
        "subsample":         0.8808881942639996,
        "colsample_bytree":  0.7966972765292112,
        "reg_alpha":         0.2833237907250693,
        "reg_lambda":        6.950384442633107,
        "scale_pos_weight":  1.2256482393129562,
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    metrics = evaluate_model(y_test, model.predict(X_test), "LightGBM")
    return model, metrics


# ─────────────────────────────────────────────
# XGBoost + Optuna
# ─────────────────────────────────────────────

def xgb_objective(trial, X_train, y_train):
    params = {
        "objective":        "binary:logistic",
        "eval_metric":      "logloss",
        "verbosity":        0,
        "random_state":     42,
        "n_estimators":     trial.suggest_int("n_estimators", 50, 400),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth":        trial.suggest_int("max_depth", 3, 8),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.8, 2.0),
    }
    tscv   = TimeSeriesSplit(n_splits=5)
    scores = []
    for train_idx, val_idx in tscv.split(X_train):
        m = xgb.XGBClassifier(**params)
        m.fit(
            X_train.iloc[train_idx], y_train.iloc[train_idx],
            eval_set=[(X_train.iloc[val_idx], y_train.iloc[val_idx])],
            verbose=False
        )
        preds = m.predict(X_train.iloc[val_idx])
        scores.append(f1_score(y_train.iloc[val_idx], preds, average="macro"))
    return np.mean(scores)


def train_xgboost(X_train, y_train, X_test, y_test):
    logger.info("Training XGBoost with Optuna (100 trials)...")
    study = optuna.create_study(direction="maximize", sampler=TPESampler())
    study.optimize(
        lambda trial: xgb_objective(trial, X_train, y_train),
        n_trials=100,
        show_progress_bar=True
    )
    logger.info(f"Best CV F1: {study.best_value:.4f}")
    best = {
        "objective": "binary:logistic", "verbosity": 0,
        "random_state": 42, **study.best_params
    }
    model = xgb.XGBClassifier(**best)
    model.fit(X_train, y_train)
    metrics = evaluate_model(y_test, model.predict(X_test), "XGBoost")
    return model, metrics, best


# ─────────────────────────────────────────────
# CatBoost + Optuna
# ─────────────────────────────────────────────

def cat_objective(trial, X_train, y_train):
    params = {
        "loss_function":       "Logloss",
        "eval_metric":         "Accuracy",
        "verbose":             False,
        "random_seed":         42,
        "iterations":          trial.suggest_int("iterations", 100, 600),
        "learning_rate":       trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "depth":               trial.suggest_int("depth", 3, 8),
        "l2_leaf_reg":         trial.suggest_float("l2_leaf_reg", 0.1, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "border_count":        trial.suggest_int("border_count", 32, 255),
        "scale_pos_weight":    trial.suggest_float("scale_pos_weight", 0.8, 2.0),
    }
    tscv   = TimeSeriesSplit(n_splits=5)
    scores = []
    for train_idx, val_idx in tscv.split(X_train):
        m = CatBoostClassifier(**params)
        m.fit(
            X_train.iloc[train_idx], y_train.iloc[train_idx],
            eval_set=(X_train.iloc[val_idx], y_train.iloc[val_idx]),
            early_stopping_rounds=30,
            verbose=False
        )
        preds = m.predict(X_train.iloc[val_idx])
        scores.append(f1_score(y_train.iloc[val_idx], preds, average="macro"))
    return np.mean(scores)


def train_catboost(X_train, y_train, X_test, y_test):
    logger.info("Training CatBoost with Optuna (100 trials)...")
    study = optuna.create_study(direction="maximize", sampler=TPESampler())
    study.optimize(
        lambda trial: cat_objective(trial, X_train, y_train),
        n_trials=100,
        show_progress_bar=True
    )
    logger.info(f"Best CV F1: {study.best_value:.4f}")
    best = {
        "loss_function": "Logloss",
        "verbose": False,
        "random_seed": 42,
        **study.best_params
    }
    model = CatBoostClassifier(**best)
    model.fit(X_train, y_train, verbose=False)
    metrics = evaluate_model(y_test, model.predict(X_test), "CatBoost")
    return model, metrics, best


# ─────────────────────────────────────────────
# Model Selection & Promotion
# ─────────────────────────────────────────────

def promote_winner(
    models: dict,
    metrics: dict,
    feature_cols: list,
    config: dict
) -> Tuple[str, float]:
    """
    Compare all trained models.
    Promote the winner to production.
    Save all models for audit trail.
    """
    model_path = Path(config["model"]["model_path"])
    model_path.mkdir(parents=True, exist_ok=True)

    # Find winner
    winner_name = max(metrics, key=lambda k: metrics[k]["accuracy"])
    winner_acc  = metrics[winner_name]["accuracy"]

    logger.info(f"\n{'='*55}")
    logger.info(f"  MODEL SELECTION RESULTS")
    logger.info(f"{'='*55}")
    logger.info(f"  {'Model':<20} {'Accuracy':>10} {'F1 Up':>10}")
    logger.info(f"  {'-'*42}")
    for name, m in sorted(metrics.items(), key=lambda x: x[1]["accuracy"], reverse=True):
        marker = " ← PROMOTED" if name == winner_name else ""
        logger.info(f"  {name:<20} {m['accuracy']*100:>9.2f}% {m['f1_minority']:>10.4f}{marker}")
    logger.info(f"{'='*55}")

    # Save all models for audit trail
    joblib.dump(models["LightGBM"], model_path / "lgbm_model.joblib")
    joblib.dump(models["XGBoost"],  model_path / "xgb_model.joblib")
    models["CatBoost"].save_model(str(model_path / "catboost_model.cbm"))
    joblib.dump(models["Baseline"], model_path / "baseline_model.joblib")

    # Promote winner as production model
    if winner_name == "CatBoost":
        models[winner_name].save_model(str(model_path / "production_model.cbm"))
        # Also save as joblib wrapper for predict.py compatibility
        joblib.dump(models[winner_name], model_path / "production_model.joblib")
    else:
        joblib.dump(models[winner_name], model_path / "production_model.joblib")

    # Save winner name and feature cols
    with open(model_path / "production_model_name.txt", "w") as f:
        f.write(winner_name)

    joblib.dump(feature_cols, model_path / "feature_cols.joblib")

    # Also save as lgbm_model.joblib for backward compat with predict.py
    if winner_name != "LightGBM":
        joblib.dump(models[winner_name], model_path / "lgbm_model.joblib")

    logger.info(f"Winner '{winner_name}' promoted to production")
    logger.info(f"All models saved to {model_path}/")

    return winner_name, winner_acc


# ─────────────────────────────────────────────
# Quality Gate
# ─────────────────────────────────────────────

def check_gate(metrics: Dict, config: dict, model_name: str) -> bool:
    min_acc = config["thresholds"]["min_directional_accuracy"]
    min_f1  = config["thresholds"]["min_f1_score"]
    acc_pass = metrics["directional_acc"] >= min_acc
    f1_pass  = metrics["f1_minority"]     >= min_f1
    passed   = acc_pass and f1_pass

    logger.info(f"\n{'='*55}")
    logger.info(f"  Quality Gate — {model_name} (Production)")
    logger.info(f"{'='*55}")
    logger.info(f"  Directional Accuracy : {metrics['directional_acc']:.4f} "
                f"(min: {min_acc}) → {'PASS ✓' if acc_pass else 'FAIL ✗'}")
    logger.info(f"  F1 Minority Class    : {metrics['f1_minority']:.4f} "
                f"(min: {min_f1}) → {'PASS ✓' if f1_pass else 'FAIL ✗'}")
    logger.info(f"  Overall Gate         : {'PASSED ✓' if passed else 'FAILED ✗'}")
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

    # Train all models
    logger.info("\n" + "="*55)
    logger.info("  TRAINING ALL MODELS")
    logger.info("="*55)

    baseline, baseline_metrics       = train_baseline(X_train, y_train, X_test, y_test, config)
    all_models["Baseline"]           = baseline
    all_metrics["Baseline"]          = baseline_metrics

    lgbm_model, lgbm_metrics         = train_lightgbm(X_train, y_train, X_test, y_test)
    all_models["LightGBM"]           = lgbm_model
    all_metrics["LightGBM"]          = lgbm_metrics

    xgb_model, xgb_metrics, _        = train_xgboost(X_train, y_train, X_test, y_test)
    all_models["XGBoost"]            = xgb_model
    all_metrics["XGBoost"]           = xgb_metrics

    cat_model, cat_metrics, _        = train_catboost(X_train, y_train, X_test, y_test)
    all_models["CatBoost"]           = cat_model
    all_metrics["CatBoost"]          = cat_metrics

    # Promote winner
    winner_name, winner_acc = promote_winner(
        all_models, all_metrics,
        list(X_train.columns), config
    )

    # Gate on winner only
    gate_passed = check_gate(all_metrics[winner_name], config, winner_name)

    print(f"\nBaseline Accuracy : {baseline_metrics['accuracy']*100:.2f}%")
    print(f"LightGBM Accuracy : {lgbm_metrics['accuracy']*100:.2f}%")
    print(f"XGBoost  Accuracy : {xgb_metrics['accuracy']*100:.2f}%")
    print(f"CatBoost Accuracy : {cat_metrics['accuracy']*100:.2f}%")
    print(f"Winner            : {winner_name} ({winner_acc*100:.2f}%)")
    print(f"Gate Passed       : {gate_passed}")

    return {
        "all_metrics":  all_metrics,
        "winner":       winner_name,
        "winner_acc":   winner_acc,
        "gate_passed":  gate_passed,
        "feature_cols": list(X_train.columns)
    }


if __name__ == "__main__":
    run_training()
