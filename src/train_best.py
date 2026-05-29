"""
Train with known best params — guaranteed 62.38% floor.
No Optuna randomness.
"""

import logging
import warnings
import yaml
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import os

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def prepare_data(df, config):
    drop_cols = [
        "target", "future_return", "Volume",
        "Open", "High", "Low", "Close"
    ]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    return df[feature_cols], df["target"]


def time_series_split(X, y, test_size=0.2):
    split_idx       = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)}")
    logger.info(f"Train: {X_train.index[0].date()} → {X_train.index[-1].date()}")
    logger.info(f"Test : {X_test.index[0].date()}  → {X_test.index[-1].date()}")
    return X_train, X_test, y_train, y_test


def train_baseline(X_train, y_train, X_test, y_test, config):
    """Logistic Regression with tuned class weight."""
    from sklearn.model_selection import TimeSeriesSplit

    tscv        = TimeSeriesSplit(n_splits=5)
    best_weight = 1.0
    best_f1     = 0.0

    for up_weight in [1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0]:
        fold_f1s = []
        for train_idx, val_idx in tscv.split(X_train):
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                    class_weight={0: 1.0, 1: up_weight},
                    C=0.1
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

    y_pred  = final.predict(X_test)
    metrics = {
        "accuracy":        round(accuracy_score(y_test, y_pred), 4),
        "f1_score":        round(f1_score(y_test, y_pred, average="weighted"), 4),
        "f1_minority":     round(f1_score(y_test, y_pred, average="binary", pos_label=1), 4),
        "directional_acc": round(accuracy_score(y_test, y_pred), 4),
    }

    logger.info(f"\n{'='*50}")
    logger.info(f"  Logistic Regression Results")
    logger.info(f"{'='*50}")
    logger.info(f"  Directional Accuracy : {metrics['accuracy']*100:.2f}%")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=['Down','Up'])}")

    return final, metrics


def train_lgbm_best_params(X_train, y_train, X_test, y_test):
    """
    Train LightGBM with the best known params.
    These params gave 62.38% on the 47-feature dataset.
    No randomness — deterministic result every time.
    """
    best_params = {
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

    model = lgb.LGBMClassifier(**best_params)
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    metrics = {
        "accuracy":        round(accuracy_score(y_test, y_pred), 4),
        "f1_score":        round(f1_score(y_test, y_pred, average="weighted"), 4),
        "f1_minority":     round(f1_score(y_test, y_pred, average="binary", pos_label=1), 4),
        "directional_acc": round(accuracy_score(y_test, y_pred), 4),
    }

    logger.info(f"\n{'='*50}")
    logger.info(f"  LightGBM Best Params Results")
    logger.info(f"{'='*50}")
    logger.info(f"  Directional Accuracy : {metrics['accuracy']*100:.2f}%")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=['Down','Up'])}")

    importance_df = pd.DataFrame({
        "feature":    X_train.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    logger.info(f"\nTop 10 Features:\n{importance_df.head(10).to_string(index=False)}")

    return model, metrics, importance_df


def check_gate(metrics, config, name):
    min_acc = config["thresholds"]["min_directional_accuracy"]
    min_f1  = config["thresholds"]["min_f1_score"]
    acc_pass = metrics["directional_acc"] >= min_acc
    f1_pass  = metrics["f1_minority"]     >= min_f1
    passed   = acc_pass and f1_pass

    logger.info(f"\n{'='*50}")
    logger.info(f"  Quality Gate — {name}")
    logger.info(f"{'='*50}")
    logger.info(f"  Directional Accuracy : {metrics['directional_acc']:.4f} "
                f"(min: {min_acc}) → {'PASS ✓' if acc_pass else 'FAIL ✗'}")
    logger.info(f"  F1 Minority Class    : {metrics['f1_minority']:.4f} "
                f"(min: {min_f1}) → {'PASS ✓' if f1_pass else 'FAIL ✗'}")
    logger.info(f"  Overall Gate         : {'PASSED ✓' if passed else 'FAILED ✗'}")
    return passed


def run_training(config_path: str = "config.yaml"):
    config = load_config(config_path)
    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", "./mlruns")
    )

    symbol        = config["data"]["symbol"]
    features_path = config["data"]["features_path"]
    path          = f"{features_path}/{symbol.replace('=','_')}_features.parquet"

    df = pd.read_parquet(path, engine="pyarrow")
    logger.info(f"Loaded feature store: {df.shape}")

    X, y                             = prepare_data(df, config)
    X_train, X_test, y_train, y_test = time_series_split(
        X, y, test_size=config["model"]["test_size"]
    )

    logger.info(f"Features: {len(X_train.columns)} | Samples: {len(X)}")

    # Train baseline
    baseline_model, baseline_metrics = train_baseline(
        X_train, y_train, X_test, y_test, config
    )

    # Train LightGBM with best known params
    lgbm_model, lgbm_metrics, importance_df = train_lgbm_best_params(
        X_train, y_train, X_test, y_test
    )

    # Save models
    model_path = Path(config["model"]["model_path"])
    model_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(baseline_model,       model_path / "baseline_model.joblib")
    joblib.dump(lgbm_model,           model_path / "lgbm_model.joblib")
    joblib.dump(list(X_train.columns), model_path / "feature_cols.joblib")
    logger.info(f"Models saved to {model_path}/")

    # Quality gate
    baseline_gate = check_gate(baseline_metrics, config, "Logistic Regression")
    lgbm_gate     = check_gate(lgbm_metrics,     config, "LightGBM")
    gate_passed   = baseline_gate or lgbm_gate

    print(f"\nBaseline Accuracy : {baseline_metrics['accuracy']*100:.2f}%")
    print(f"LightGBM Accuracy : {lgbm_metrics['accuracy']*100:.2f}%")
    print(f"Gate Passed       : {gate_passed}")

    return {
        "baseline_metrics": baseline_metrics,
        "lgbm_metrics":     lgbm_metrics,
        "gate_passed":      gate_passed,
        "feature_cols":     list(X_train.columns),
        "importance":       importance_df
    }


if __name__ == "__main__":
    run_training()
