"""
Boosting Models Experiment — CatBoost vs XGBoost vs LightGBM
Fair comparison on identical data and train/test split.
"""

import logging
import warnings
import yaml
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def load_data(config):
    symbol        = config["data"]["symbol"]
    features_path = config["data"]["features_path"]
    path          = f"{features_path}/{symbol.replace('=','_')}_features.parquet"
    df            = pd.read_parquet(path, engine="pyarrow")

    drop_cols    = ["target", "future_return", "Volume",
                    "Open", "High", "Low", "Close"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols]
    y = df["target"]

    split_idx        = int(len(X) * 0.8)
    X_train, X_test  = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test  = y.iloc[:split_idx], y.iloc[split_idx:]

    logger.info(f"Features : {len(feature_cols)}")
    logger.info(f"Train    : {len(X_train)} | {X_train.index[0].date()} → {X_train.index[-1].date()}")
    logger.info(f"Test     : {len(X_test)}  | {X_test.index[0].date()}  → {X_test.index[-1].date()}")

    return X_train, X_test, y_train, y_test


def evaluate(y_test, y_pred, model_name):
    metrics = {
        "accuracy":    round(accuracy_score(y_test, y_pred), 4),
        "f1_weighted": round(f1_score(y_test, y_pred, average="weighted"), 4),
        "f1_minority": round(f1_score(y_test, y_pred, average="binary", pos_label=1), 4),
    }
    logger.info(f"\n{'='*55}")
    logger.info(f"  {model_name}")
    logger.info(f"{'='*55}")
    logger.info(f"  Accuracy : {metrics['accuracy']*100:.2f}%")
    logger.info(f"  F1 Up    : {metrics['f1_minority']:.4f}")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=['Down','Up'])}")
    return metrics


# ─────────────────────────────────────────────
# LightGBM — our known best (baseline)
# ─────────────────────────────────────────────

def run_lightgbm(X_train, y_train, X_test, y_test):
    logger.info("\nTraining LightGBM (known best params)...")
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
    return evaluate(y_test, model.predict(X_test), "LightGBM (best params)")


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
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train.iloc[train_idx], y_train.iloc[train_idx],
            eval_set=[(X_train.iloc[val_idx], y_train.iloc[val_idx])],
            verbose=False
        )
        preds = model.predict(X_train.iloc[val_idx])
        scores.append(f1_score(y_train.iloc[val_idx], preds, average="macro"))
    return np.mean(scores)


def run_xgboost(X_train, y_train, X_test, y_test):
    logger.info("\nTraining XGBoost with Optuna (100 trials)...")

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
    return evaluate(y_test, model.predict(X_test), "XGBoost + Optuna"), model, best


# ─────────────────────────────────────────────
# CatBoost + Optuna
# ─────────────────────────────────────────────

def cat_objective(trial, X_train, y_train):
    params = {
        "loss_function":     "Logloss",
        "eval_metric":       "Accuracy",
        "verbose":           False,
        "random_seed":       42,
        "iterations":        trial.suggest_int("iterations", 100, 600),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "depth":             trial.suggest_int("depth", 3, 8),
        "l2_leaf_reg":       trial.suggest_float("l2_leaf_reg", 0.1, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "border_count":      trial.suggest_int("border_count", 32, 255),
        "scale_pos_weight":  trial.suggest_float("scale_pos_weight", 0.8, 2.0),
    }

    tscv   = TimeSeriesSplit(n_splits=5)
    scores = []
    for train_idx, val_idx in tscv.split(X_train):
        model = CatBoostClassifier(**params)
        model.fit(
            X_train.iloc[train_idx], y_train.iloc[train_idx],
            eval_set=(X_train.iloc[val_idx], y_train.iloc[val_idx]),
            early_stopping_rounds=30,
            verbose=False
        )
        preds = model.predict(X_train.iloc[val_idx])
        scores.append(f1_score(y_train.iloc[val_idx], preds, average="macro"))
    return np.mean(scores)


def run_catboost(X_train, y_train, X_test, y_test):
    logger.info("\nTraining CatBoost with Optuna (100 trials)...")

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
    return evaluate(y_test, model.predict(X_test), "CatBoost + Optuna"), model, best


# ─────────────────────────────────────────────
# Final Comparison
# ─────────────────────────────────────────────

def print_comparison(results: dict):
    logger.info(f"\n{'='*60}")
    logger.info(f"  FINAL COMPARISON")
    logger.info(f"{'='*60}")
    logger.info(f"  {'Model':<25} {'Accuracy':>10} {'F1 Up':>10}")
    logger.info(f"  {'-'*47}")

    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1]["accuracy"],
        reverse=True
    )

    for name, metrics in sorted_results:
        marker = " ← WINNER" if name == sorted_results[0][0] else ""
        logger.info(
            f"  {name:<25} {metrics['accuracy']*100:>9.2f}% "
            f"{metrics['f1_minority']:>10.4f}{marker}"
        )

    logger.info(f"{'='*60}")

    winner_name, winner_metrics = sorted_results[0]
    lgbm_acc = results["LightGBM"]["accuracy"]

    if winner_name != "LightGBM":
        diff = winner_metrics["accuracy"] - lgbm_acc
        logger.info(
            f"\n  {winner_name} BEATS LightGBM by +{diff*100:.2f}%"
            f" → Integrate into production"
        )
    else:
        logger.info(f"\n  LightGBM remains the best model")
        logger.info(f"  Keep current production setup")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    config = load_config()
    X_train, X_test, y_train, y_test = load_data(config)

    results = {}

    # LightGBM baseline
    lgbm_metrics = run_lightgbm(X_train, y_train, X_test, y_test)
    results["LightGBM"] = lgbm_metrics

    # XGBoost
    xgb_metrics, xgb_model, xgb_params = run_xgboost(
        X_train, y_train, X_test, y_test
    )
    results["XGBoost"] = xgb_metrics

    # CatBoost
    cat_metrics, cat_model, cat_params = run_catboost(
        X_train, y_train, X_test, y_test
    )
    results["CatBoost"] = cat_metrics

    # Final comparison
    print_comparison(results)

    # Save winner if it beats LightGBM
    winner = max(results.items(), key=lambda x: x[1]["accuracy"])
    if winner[0] == "XGBoost" and winner[1]["accuracy"] > lgbm_metrics["accuracy"]:
        joblib.dump(xgb_model,  "models/xgb_model.joblib")
        joblib.dump(xgb_params, "models/xgb_params.joblib")
        logger.info("XGBoost saved — beats LightGBM")
    elif winner[0] == "CatBoost" and winner[1]["accuracy"] > lgbm_metrics["accuracy"]:
        cat_model.save_model("models/catboost_model.cbm")
        joblib.dump(cat_params, "models/catboost_params.joblib")
        logger.info("CatBoost saved — beats LightGBM")
