"""
train.py

Fine-tunes xlm-roberta-base for 3-class hate speech classification,
per Section 2.1/2.2 of the Milestone 2 report, for ONE condition
(baseline = no emoji, or proposed = with emoji) across all 5 folds
defined in folds.json (Section 4.3).

Implements:
  - AdamW optimiser, lr=2e-5, dropout=0.1, max_len=128           (Table 2.2)
  - class-weighted CrossEntropyLoss for the imbalance strategy    (Section 1.2)
  - early stopping on validation loss divergence                 (Section 4.2)
  - per-fold checkpoint + metrics saved to outputs/<condition>/fold_k/

Run once per condition:
    python train.py --condition with_emoji  --data ../data/data_with_emoji.csv \
        --folds ../data/folds.json --outdir ../outputs --epochs 6

    python train.py --condition no_emoji    --data ../data/data_no_emoji.csv \
        --folds ../data/folds.json --outdir ../outputs --epochs 6

Requires a GPU for realistic runtime (Colab T4/A100, or a local CUDA
GPU). CPU-only will run but very slowly for 5 folds x 6 epochs.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                              roc_auc_score, confusion_matrix)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                           get_linear_schedule_with_warmup)

from dataset import HateSpeechDataset, LABEL2ID, ID2LABEL

MODEL_NAME = "xlm-roberta-base"


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(y_true, y_pred, y_proba):
    """Macro-F1 (primary, Section 4.1) + secondary metrics (Section 4.2)."""
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")  # can happen if a class is absent from a tiny fold
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "macro_auc_roc": auc,
        "per_class": {
            ID2LABEL[i]: {"precision": precision[i], "recall": recall[i], "f1": f1[i]}
            for i in range(3)
        },
        "confusion_matrix": cm,  # rows=true, cols=pred, order = neutral, offensive, hate
        "confusion_matrix_labels": ["neutral", "offensive", "hate"],
    }


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_proba, total_loss = [], [], [], 0.0
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"].to(device)
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            outputs = model(**inputs)
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * labels.size(0)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=-1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_proba.extend(probs.tolist())
    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(all_labels, all_preds, np.array(all_proba))
    metrics["loss"] = avg_loss
    return metrics


def train_one_fold(fold_num, train_df, test_df, tokenizer, device, args):
    set_seed(args.seed)

    # class-weighted loss (Section 1.2 imbalance strategy, item 3)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1, 2]),
        y=[LABEL2ID[l] for l in train_df["label"]],
    )
    class_weights_t = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"  Class weights (neutral, offensive, hate): {class_weights}")

    train_ds = HateSpeechDataset(train_df["clean_text"], train_df["label"], tokenizer, args.max_len)
    test_ds = HateSpeechDataset(test_df["clean_text"], test_df["label"], tokenizer, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3, hidden_dropout_prob=0.1
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )
    loss_fn = nn.CrossEntropyLoss(weight=class_weights_t)

    best_val_loss = float("inf")
    best_state = None
    patience, patience_counter = args.patience, 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            labels = batch["labels"].to(device)
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item() * labels.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_metrics = evaluate(model, test_loader, device)
        val_loss = val_metrics["loss"]
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                         "val_macro_f1": val_metrics["macro_f1"]})
        print(f"  Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_macro_f1={val_metrics['macro_f1']:.4f}")

        # Early stopping: revert to best checkpoint if val loss diverges (Section 4.2)
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

    # revert to best checkpoint before final evaluation
    if best_state is not None:
        model.load_state_dict(best_state)
    final_metrics = evaluate(model, test_loader, device)
    final_metrics["history"] = history
    return model, final_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=["with_emoji", "no_emoji"])
    ap.add_argument("--data", required=True, help="Processed CSV matching the condition "
                                                    "(data_with_emoji.csv or data_no_emoji.csv)")
    ap.add_argument("--folds", required=True, help="folds.json from make_folds.py")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_model", action="store_true",
                     help="Save the fine-tuned weights per fold (large; off by default).")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cpu":
        print("WARNING: no GPU detected. 5 folds x XLM-R fine-tuning on CPU will be very slow. "
              "Use Colab/Kaggle with a GPU runtime for the real run.")

    df = pd.read_csv(args.data, encoding="utf-8-sig").reset_index(drop=True)
    with open(args.folds, "r", encoding="utf-8") as f:
        fold_spec = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    outdir = Path(args.outdir) / args.condition
    outdir.mkdir(parents=True, exist_ok=True)

    all_fold_metrics = []
    for fold in fold_spec["folds"]:
        fold_num = fold["fold"]
        print(f"\n=== Condition: {args.condition} | Fold {fold_num}/{len(fold_spec['folds'])} ===")
        n_idx = len(fold["train_rows"]) + len(fold["test_rows"])
        assert n_idx == len(df) and max(fold["train_rows"] + fold["test_rows"]) < len(df), (
            f"folds.json covers {n_idx} rows but --data has {len(df)} rows. "
            "Both conditions must use CSVs with identical rows (re-run preprocessing.py + make_folds.py).")
        train_df = df.iloc[fold["train_rows"]].reset_index(drop=True)
        test_df = df.iloc[fold["test_rows"]].reset_index(drop=True)

        t0 = time.time()
        model, metrics = train_one_fold(fold_num, train_df, test_df, tokenizer, device, args)
        elapsed = time.time() - t0
        metrics["fold"] = fold_num
        metrics["train_seconds"] = round(elapsed, 1)
        all_fold_metrics.append(metrics)

        fold_dir = outdir / f"fold_{fold_num}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        with open(fold_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        if args.save_model:
            model.save_pretrained(fold_dir / "model")
            tokenizer.save_pretrained(fold_dir / "model")

        print(f"  Fold {fold_num} done in {elapsed/60:.1f} min. "
              f"Test macro-F1={metrics['macro_f1']:.4f}")

    summary_path = outdir / "all_folds_metrics.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_fold_metrics, f, indent=2)
    print(f"\nSaved all-fold metrics to {summary_path}")
    print("Next: run evaluate.py to aggregate mean +/- std across folds, "
          "then stats_test.py to compare against the other condition.")


if __name__ == "__main__":
    main()
