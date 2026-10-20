"""Fine-tune a light BERT-family regressor on cleaned OCR text (HuggingFace Transformers).

Model: AutoModelForSequenceClassification(num_labels=1, problem_type="regression"), i.e. a
linear head on the [CLS] representation trained with MSE. Default backbone
distilbert-base-uncased (--model microsoft/deberta-v3-small also supported).

Design decisions
  * Long CVs: cleaned CVs are <= 163 word pieces (mean 97) - far below 512 - so nothing is
    truncated or chunked here. For robustness (Streamlit uploads) inputs longer than
    --max-length keep the first 75 % (work history: titles + date ranges) and the last
    25 % (education: graduation years) of the tokens instead of cutting the tail blindly.
  * Target scaling: y is standardised with the TRAIN mean/std, (y - mu) / sigma, so the
    regression head starts in a well-conditioned range; predictions are mapped back to
    years and clipped to [0, cap] before any metric.
  * Early stopping: evaluation every epoch, best checkpoint = lowest validation MAE in
    years (metric_for_best_model="mae"), patience --patience epochs.
  * Reproducibility: fixed seed for weights init, data order and dropout.

Outputs
  models/<name>/best/                       model + tokenizer + target_scaler.json
  output/predictions/<name>_{val,test}.csv
  output/metrics/<name>_val.json            val metrics + hyper-parameters + timing + GPU
  output/tables/<name>_history.csv          per-epoch train loss / val metrics

Sensitivity flags (README Q2/Q3): --loss huber, --truncation head, --text-source ref, --seed N.

Usage:  python -m src.train_bert [--model distilbert-base-uncased] [--lr 5e-5] [--epochs 30]
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, EarlyStoppingCallback,
    Trainer, TrainingArguments,
)

from src.evaluation import regression_metrics, report_validation, save_predictions
from src.split import load_split
from src.utils import MODELS_DIR, PROCESSED_DIR, SEED, TABLES_DIR, clip_target, ensure_dirs, get_logger, hardware_info, load_json, save_json, set_seed

log = get_logger("train_bert")
HEAD_SHARE = 0.75


# ---------------------------------------------------------------------------
# Tokenisation with head+tail truncation
# ---------------------------------------------------------------------------
def encode(tokenizer, texts: list[str], max_length: int, head_share: float = HEAD_SHARE) -> tuple[dict[str, list], int]:
    """Tokenise; inputs over max_length keep the first `head_share` and the last (1 - head_share)
    of the body tokens (head_share=1.0 is plain head-only truncation)."""
    body = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    head = int(body * head_share)
    input_ids, n_trunc = [], 0
    for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]:
        if len(ids) > body:
            ids = ids[:head] + (ids[-(body - head):] if body > head else [])
            n_trunc += 1
        input_ids.append(tokenizer.build_inputs_with_special_tokens(ids))
    return {"input_ids": input_ids, "attention_mask": [[1] * len(x) for x in input_ids]}, n_trunc


class TargetScaler:
    def __init__(self, mean: float, std: float):
        self.mean, self.std = float(mean), float(std)

    @classmethod
    def fit(cls, y) -> "TargetScaler":
        return cls(np.mean(y), np.std(y))

    def transform(self, y):
        return (np.asarray(y, float) - self.mean) / self.std

    def inverse(self, z):
        return np.asarray(z, float) * self.std + self.mean

    def to_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}


def make_dataset(tokenizer, df: pd.DataFrame, scaler: TargetScaler, max_length: int,
                 head_share: float = HEAD_SHARE) -> tuple[Dataset, int]:
    enc, n_trunc = encode(tokenizer, df["text"].tolist(), max_length, head_share)
    enc["labels"] = scaler.transform(df["exp_year"]).astype(np.float32).tolist()
    return Dataset.from_dict(enc), n_trunc


class HuberTrainer(Trainer):
    """Trainer with a Huber loss on the standardised target (robust to outlier labels)."""

    huber_delta = 1.0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = torch.nn.functional.huber_loss(outputs.logits.squeeze(-1).float(), labels.float(), delta=self.huber_delta)
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Inference helpers (also used by the Streamlit app)
# ---------------------------------------------------------------------------
def load_regressor(path: Path, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()
    meta = load_json(Path(path) / "target_scaler.json")
    cfg = {"max_length": meta.get("max_length", 512), "head_share": meta.get("head_share", HEAD_SHARE)}
    return model, tok, TargetScaler(meta["mean"], meta["std"]), cfg


@torch.no_grad()
def predict_texts(model, tokenizer, scaler: TargetScaler, texts: list[str], max_length: int = 512,
                  head_share: float = HEAD_SHARE, batch_size: int = 64) -> np.ndarray:
    enc, _ = encode(tokenizer, texts, max_length, head_share)
    collator = DataCollatorWithPadding(tokenizer)
    out = []
    for i in range(0, len(texts), batch_size):
        batch = collator([{k: enc[k][j] for k in enc} for j in range(i, min(i + batch_size, len(texts)))])
        logits = model(**{k: v.to(model.device) for k, v in batch.items()}).logits.squeeze(-1)
        out.append(logits.float().cpu().numpy())
    return clip_target(scaler.inverse(np.concatenate(out)))


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--name", default=None, help="output name (default bert_<backbone>)")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--loss", choices=("mse", "huber"), default="mse")
    parser.add_argument("--truncation", choices=("head_tail", "head"), default="head_tail",
                        help="strategy for inputs longer than --max-length")
    parser.add_argument("--text-source", choices=("ocr", "ref"), default="ocr",
                        help="ref = reference text, ANALYSIS ONLY (README Q2), never the reported model")
    parser.add_argument("--no-test", action="store_true", help="skip test predictions (sensitivity runs)")
    args = parser.parse_args()

    name = args.name or "bert_" + args.model.split("/")[-1].replace("-", "_").replace(".", "_")
    out_dir = MODELS_DIR / name
    ensure_dirs()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        log.warning("CUDA not available: training on CPU will be slow")

    head_share = HEAD_SHARE if args.truncation == "head_tail" else 1.0
    text_path = PROCESSED_DIR / ("cvs.parquet" if args.text_source == "ocr" else "cvs_ref.parquet")
    train, val, test = (load_split(s, text_path=text_path) for s in ("train", "val", "test"))
    scaler = TargetScaler.fit(train["exp_year"])
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ds_train, tr_trunc = make_dataset(tokenizer, train, scaler, args.max_length, head_share)
    ds_val, va_trunc = make_dataset(tokenizer, val, scaler, args.max_length, head_share)
    lengths = [len(x) for x in ds_train["input_ids"]]
    log.info("Tokens per CV (train): mean %.0f, max %d | truncated train %d, val %d | target scaler mu=%.3f sigma=%.3f",
             np.mean(lengths), max(lengths), tr_trunc, va_trunc, scaler.mean, scaler.std)

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1, problem_type="regression")

    def compute_metrics(eval_pred):
        preds = scaler.inverse(np.asarray(eval_pred.predictions).reshape(-1))
        years = scaler.inverse(np.asarray(eval_pred.label_ids).reshape(-1))
        m = regression_metrics(years, preds)
        return {"mae": m["mae"], "rmse": m["rmse"], "acc_pm1": m["acc_pm1"]}

    fp16 = device == "cuda" and not args.no_fp16 and "deberta" not in args.model.lower()  # DeBERTa-v3 is unstable in fp16
    targs = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        eval_strategy="epoch", save_strategy="epoch", logging_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="mae", greater_is_better=False, save_total_limit=2,
        learning_rate=args.lr, num_train_epochs=args.epochs, warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay, lr_scheduler_type="linear",
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=64,
        fp16=fp16, bf16=device == "cuda" and "deberta" in args.model.lower() and not args.no_fp16,
        seed=args.seed, data_seed=args.seed, dataloader_num_workers=0, report_to="none", disable_tqdm=True,
    )
    trainer = (HuberTrainer if args.loss == "huber" else Trainer)(model=model, args=targs, train_dataset=ds_train, eval_dataset=ds_val,
                      data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=compute_metrics,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)])

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    trainer.train()
    train_s = time.perf_counter() - t0

    history = pd.DataFrame(trainer.state.log_history)
    hist = (history.groupby("epoch").agg({c: "last" for c in ("loss", "eval_loss", "eval_mae", "eval_rmse", "eval_acc_pm1") if c in history})
            .reset_index())
    hist.to_csv(TABLES_DIR / f"{name}_history.csv", index=False)
    best_epoch = float(hist.loc[hist["eval_mae"].idxmin(), "epoch"])
    log.info("Per-epoch history:\n%s", hist.round(4).to_string(index=False))

    best_dir = out_dir / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    save_json({**scaler.to_dict(), "max_length": args.max_length, "head_share": head_share, "backbone": args.model,
               "text_source": args.text_source}, best_dir / "target_scaler.json")
    shutil.rmtree(out_dir / "checkpoints", ignore_errors=True)

    model = trainer.model.eval()
    t1 = time.perf_counter()
    va_pred = predict_texts(model, tokenizer, scaler, val["text"].tolist(), args.max_length, head_share)
    infer_ms = 1000 * (time.perf_counter() - t1) / len(val)

    hw = hardware_info()
    info = {
        "params": {"backbone": args.model, "lr": args.lr, "max_epochs": args.epochs, "batch_size": args.batch_size,
                   "max_length": args.max_length, "warmup_ratio": args.warmup_ratio, "weight_decay": args.weight_decay,
                   "patience": args.patience, "seed": args.seed, "fp16": fp16, "bf16": targs.bf16, "loss": args.loss,
                   "target_scaling": "z-score (train)", "truncation": args.truncation, "head_share": head_share,
                   "n_truncated_train": tr_trunc, "n_truncated_val": va_trunc, "text_source": args.text_source,
                   "n_parameters": int(sum(p.numel() for p in model.parameters()))},
        "best_epoch": best_epoch, "epochs_run": float(hist["epoch"].max()),
        "train_time_s": round(train_s, 1), "train_time_per_epoch_s": round(train_s / max(hist["epoch"].max(), 1), 1),
        "inference_ms_per_cv": round(infer_ms, 2),
        "peak_gpu_mem_mib": round(torch.cuda.max_memory_allocated() / 2**20) if device == "cuda" else None,
        "hardware": {"gpu": hw.get("gpu"), "cuda": hw.get("cuda"), "torch": hw.get("torch")},
    }
    log.info("Trained %s in %.0fs (%.1fs/epoch), best epoch %.0f, peak GPU mem %s MiB, %.2f ms/CV inference",
             name, train_s, info["train_time_per_epoch_s"], best_epoch, info["peak_gpu_mem_mib"], infer_ms)
    report_validation(name, val["id"], val["exp_year"], va_pred, info)
    if not args.no_test:
        te_pred = predict_texts(model, tokenizer, scaler, test["text"].tolist(), args.max_length, head_share)
        save_predictions(name, "test", test["id"], test["exp_year"], te_pred)


if __name__ == "__main__":
    main()
