"""
LSTM Experiment — Standalone
Tests whether LSTM beats LightGBM 61.65% on same data/split.
Uses same 47 features, same train/test split.
No integration with main pipeline until results are confirmed.
"""

import logging
import warnings
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from pathlib import Path

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)


# ─────────────────────────────────────────────
# Config & Data
# ─────────────────────────────────────────────

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_and_prepare(config: dict):
    """
    Load feature store and prepare same split as LightGBM.
    Identical setup ensures fair comparison.
    """
    symbol        = config["data"]["symbol"]
    features_path = config["data"]["features_path"]
    path          = f"{features_path}/{symbol.replace('=','_')}_features.parquet"

    df = pd.read_parquet(path, engine="pyarrow")

    # Same drop cols as train.py
    drop_cols    = ["target", "future_return", "Volume",
                    "Open", "High", "Low", "Close"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].values.astype(np.float32)
    y = df["target"].values.astype(np.int64)

    # Same 80/20 chronological split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    logger.info(f"Features : {len(feature_cols)}")
    logger.info(f"Train    : {len(X_train)} rows")
    logger.info(f"Test     : {len(X_test)} rows")
    logger.info(f"Dates    : {df.index[0].date()} → {df.index[-1].date()}")

    return X_train, X_test, y_train, y_test, feature_cols


# ─────────────────────────────────────────────
# Scaling
# ─────────────────────────────────────────────

def scale_features(X_train, X_test):
    """
    StandardScaler fit on training set only.
    Prevents leakage of test set statistics into training.
    """
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    return X_train, X_test, scaler


# ─────────────────────────────────────────────
# Sequence Dataset
# ─────────────────────────────────────────────

class ForexSequenceDataset(Dataset):
    """
    Converts flat feature rows into sequences for LSTM.

    sequence_length = 20 means:
      For each prediction, LSTM sees the last 20 days of features.
      This gives it temporal memory that LightGBM lacks.

    Example:
      Row 100 target → LSTM sees features from rows 80-100
      Row 101 target → LSTM sees features from rows 81-101
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sequence_length: int = 20
    ):
        self.X              = torch.FloatTensor(X)
        self.y              = torch.LongTensor(y)
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.X) - self.sequence_length

    def __getitem__(self, idx):
        # Sequence of last N days
        x_seq = self.X[idx: idx + self.sequence_length]
        # Target is the label for the LAST day in the sequence
        y_val = self.y[idx + self.sequence_length - 1]
        return x_seq, y_val


# ─────────────────────────────────────────────
# LSTM Model
# ─────────────────────────────────────────────

class ForexLSTM(nn.Module):
    """
    Two-layer bidirectional LSTM for direction prediction.

    Bidirectional: reads sequence forward AND backward.
    Dropout: prevents overfitting on small dataset.
    Final layer: maps LSTM output to binary prediction.
    """

    def __init__(
        self,
        input_size:   int,
        hidden_size:  int = 128,
        num_layers:   int = 2,
        dropout:      float = 0.3,
        bidirectional: bool = True
    ):
        super(ForexLSTM, self).__init__()

        self.hidden_size   = hidden_size
        self.num_layers    = num_layers
        self.bidirectional = bidirectional
        self.directions    = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size   = input_size,
            hidden_size  = hidden_size,
            num_layers   = num_layers,
            dropout      = dropout if num_layers > 1 else 0,
            bidirectional = bidirectional,
            batch_first  = True
        )

        self.dropout = nn.Dropout(dropout)

        # Batch normalisation — stabilises training
        self.batch_norm = nn.BatchNorm1d(hidden_size * self.directions)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * self.directions, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)   # 2 classes: Down, Up
        )

    def forward(self, x):
        # x shape: (batch, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)

        # Take only the last timestep output
        last_output = lstm_out[:, -1, :]

        # Normalise and classify
        normed = self.batch_norm(last_output)
        output = self.classifier(self.dropout(normed))
        return output


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    sequence_length: int = 20,
    hidden_size:     int = 128,
    num_layers:      int = 2,
    epochs:          int = 50,
    batch_size:      int = 64,
    learning_rate:   float = 0.001,
    patience:        int = 10
) -> ForexLSTM:
    """
    Train LSTM with early stopping.
    patience=10 means stop if val loss doesn't improve for 10 epochs.
    """
    input_size = X_train.shape[1]

    # Datasets
    train_dataset = ForexSequenceDataset(X_train, y_train, sequence_length)
    test_dataset  = ForexSequenceDataset(X_test,  y_test,  sequence_length)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=False,   # Never shuffle time series
        drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, drop_last=False
    )

    # Model
    model = ForexLSTM(
        input_size   = input_size,
        hidden_size  = hidden_size,
        num_layers   = num_layers,
        dropout      = 0.3,
        bidirectional = True
    )

    logger.info(f"\nLSTM Architecture:")
    logger.info(f"  Input size    : {input_size}")
    logger.info(f"  Hidden size   : {hidden_size}")
    logger.info(f"  Layers        : {num_layers} (bidirectional)")
    logger.info(f"  Parameters    : {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"  Sequence len  : {sequence_length} days")
    logger.info(f"  Epochs        : {epochs} (patience={patience})")

    # Loss and optimizer
    # Weight the Up class slightly more — same logic as LightGBM scale_pos_weight
    class_weights = torch.FloatTensor([1.0, 1.226])
    criterion     = nn.CrossEntropyLoss(weight=class_weights)
    optimizer     = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler     = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, verbose=False
    )

    # Training loop with early stopping
    best_val_loss  = float("inf")
    best_model_state = None
    patience_count = 0

    logger.info("\nTraining:")
    for epoch in range(epochs):
        # ── Train ──────────────────────────────
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss   = criterion(output, y_batch)
            loss.backward()
            # Gradient clipping — prevents exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ── Validate ───────────────────────────
        model.eval()
        val_loss  = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                output    = model(X_batch)
                loss      = criterion(output, y_batch)
                val_loss += loss.item()
                preds     = torch.argmax(output, dim=1)
                all_preds.extend(preds.numpy())
                all_labels.extend(y_batch.numpy())

        val_loss /= len(test_loader)
        val_acc   = accuracy_score(all_labels, all_preds)

        scheduler.step(val_loss)

        # Log every 10 epochs
        if (epoch + 1) % 10 == 0:
            logger.info(
                f"  Epoch {epoch+1:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc*100:.2f}%"
            )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_model_state = model.state_dict().copy()
            patience_count   = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break

    # Restore best model
    model.load_state_dict(best_model_state)
    return model, test_loader, all_labels


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def evaluate_lstm(
    model: ForexLSTM,
    test_loader: DataLoader,
) -> dict:
    """Full evaluation on test set."""
    model.eval()
    all_preds  = []
    all_labels = []
    all_probs  = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            output = model(X_batch)
            probs  = torch.softmax(output, dim=1)
            preds  = torch.argmax(output, dim=1)
            all_preds.extend(preds.numpy())
            all_labels.extend(y_batch.numpy())
            all_probs.extend(probs[:, 1].numpy())

    metrics = {
        "accuracy":        round(accuracy_score(all_labels, all_preds), 4),
        "f1_weighted":     round(f1_score(all_labels, all_preds, average="weighted"), 4),
        "f1_minority":     round(f1_score(all_labels, all_preds, average="binary", pos_label=1), 4),
        "directional_acc": round(accuracy_score(all_labels, all_preds), 4),
    }

    logger.info(f"\n{'='*55}")
    logger.info(f"  LSTM Results")
    logger.info(f"{'='*55}")
    logger.info(f"  Directional Accuracy : {metrics['accuracy']*100:.2f}%")
    logger.info(f"  F1 Score (weighted)  : {metrics['f1_weighted']:.4f}")
    logger.info(f"  F1 Score (Up class)  : {metrics['f1_minority']:.4f}")
    logger.info(f"\n{classification_report(all_labels, all_preds, target_names=['Down','Up'])}")

    return metrics


# ─────────────────────────────────────────────
# Comparison
# ─────────────────────────────────────────────

def compare_with_lightgbm(lstm_metrics: dict) -> None:
    """Direct comparison against our known LightGBM baseline."""
    lgbm_acc = 0.6165
    lgbm_f1  = 0.6247
    lstm_acc = lstm_metrics["accuracy"]
    lstm_f1  = lstm_metrics["f1_minority"]

    logger.info(f"\n{'='*55}")
    logger.info(f"  HEAD-TO-HEAD COMPARISON")
    logger.info(f"{'='*55}")
    logger.info(f"  {'Model':<20} {'Accuracy':>10} {'F1 Up':>10}")
    logger.info(f"  {'-'*42}")
    logger.info(f"  {'LightGBM (V2)':<20} {lgbm_acc*100:>9.2f}% {lgbm_f1:>10.4f}")
    logger.info(f"  {'LSTM':<20} {lstm_acc*100:>9.2f}% {lstm_f1:>10.4f}")
    logger.info(f"  {'-'*42}")

    diff = lstm_acc - lgbm_acc
    if diff > 0.01:
        logger.info(f"  VERDICT: LSTM WINS by +{diff*100:.2f}% → Integrate into ensemble")
    elif diff > -0.01:
        logger.info(f"  VERDICT: ROUGHLY EQUAL ({diff*100:+.2f}%) → Not worth added complexity")
    else:
        logger.info(f"  VERDICT: LIGHTGBM WINS by {abs(diff)*100:.2f}% → Keep LightGBM only")
    logger.info(f"{'='*55}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    config = load_config()

    # Load data
    X_train, X_test, y_train, y_test, feature_cols = load_and_prepare(config)

    # Scale
    X_train_s, X_test_s, scaler = scale_features(X_train, X_test)

    # Train LSTM
    logger.info("\nStarting LSTM training...")
    model, test_loader, _ = train_lstm(
        X_train_s, y_train,
        X_test_s,  y_test,
        sequence_length = 20,
        hidden_size     = 128,
        num_layers      = 2,
        epochs          = 100,
        batch_size      = 64,
        learning_rate   = 0.001,
        patience        = 15
    )

    # Evaluate
    metrics = evaluate_lstm(model, test_loader)

    # Compare
    compare_with_lightgbm(metrics)

    # Save model if it beats LightGBM
    if metrics["accuracy"] > 0.6165:
        torch.save(model.state_dict(), "models/lstm_model.pt")
        import joblib
        joblib.dump(scaler, "models/lstm_scaler.joblib")
        logger.info("LSTM model saved — beats LightGBM")
    else:
        logger.info("LSTM not saved — does not beat LightGBM")
