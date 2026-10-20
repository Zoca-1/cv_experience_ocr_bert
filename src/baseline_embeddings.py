"""Baseline 2: frozen sentence embeddings + Ridge regression.

Encoder: sentence-transformers/all-MiniLM-L6-v2 by default (--encoder BAAI/bge-small-en-v1.5
is the alternative). Ridge alpha is tuned on validation; the encoder is never fine-tuned,
which is exactly what separates this baseline from the BERT regressor.

Sequence length: cleaned CVs are <= 163 word pieces (mean 97) against the encoder's 256
limit, so no CV is truncated. For longer inputs (e.g. Streamlit uploads) `encode_long`
splits the text on line boundaries into chunks that fit, embeds each chunk and averages
them weighted by chunk length - instead of silently dropping the tail (education years).
Embeddings are cached in data/processed/ keyed by encoder and text hash.

Usage:  python -m src.baseline_embeddings [--encoder NAME]
"""
from __future__ import annotations

import argparse
import hashlib
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

from src.evaluation import report_validation, save_predictions
from src.split import load_split
from src.utils import (
    MODELS_DIR, PROCESSED_DIR, SEED, TABLES_DIR, clip_target, ensure_dirs, get_device, get_logger, set_seed,
)

log = get_logger("baseline_embeddings")
MODEL_NAME = "baseline_embeddings"
DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
ALPHAS = np.logspace(-4, 3, 15)


def chunk_by_lines(text: str, tokenizer, max_tokens: int) -> list[str]:
    """Greedy line packing so each chunk fits the encoder window."""
    chunks, cur, cur_len = [], [], 0
    for line in text.splitlines():
        n = len(tokenizer.tokenize(line)) + 1
        if cur and cur_len + n > max_tokens:
            chunks.append("\n".join(cur))
            cur, cur_len = [], 0
        cur.append(line)
        cur_len += n
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def encode_long(model, texts: list[str], batch_size: int = 64) -> tuple[np.ndarray, int]:
    """Embed texts; any text longer than the window is chunked and length-weighted averaged."""
    tok, limit = model.tokenizer, model.get_max_seq_length() - 2
    pieces, owner, weights, n_chunked = [], [], [], 0
    for i, t in enumerate(texts):
        chunks = [t] if len(tok.tokenize(t)) <= limit else chunk_by_lines(t, tok, limit)
        n_chunked += len(chunks) > 1
        for c in chunks:
            pieces.append(c)
            owner.append(i)
            weights.append(max(len(c), 1))
    emb = model.encode(pieces, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True,
                       show_progress_bar=False)
    out = np.zeros((len(texts), emb.shape[1]), dtype=np.float32)
    wsum = np.zeros(len(texts), dtype=np.float32)
    np.add.at(out, owner, emb * np.asarray(weights, np.float32)[:, None])
    np.add.at(wsum, owner, weights)
    out /= wsum[:, None]
    return out / np.linalg.norm(out, axis=1, keepdims=True), n_chunked


def cached_embeddings(model, encoder: str, ids: pd.Series, texts: pd.Series) -> np.ndarray:
    key = hashlib.sha256((encoder + "\x00" + "\x00".join(ids) + "\x00".join(texts)).encode()).hexdigest()[:16]
    path = PROCESSED_DIR / f"emb_{encoder.replace('/', '__')}_{key}.npy"
    if path.exists():
        return np.load(path)
    emb, n_chunked = encode_long(model, texts.tolist())
    if n_chunked:
        log.info("%d texts exceeded the %d-token window and were chunked", n_chunked, model.get_max_seq_length())
    np.save(path, emb)
    return emb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--encoder", default=DEFAULT_ENCODER)
    args = parser.parse_args()
    ensure_dirs()
    set_seed()
    from sentence_transformers import SentenceTransformer

    train, val, test = (load_split(s) for s in ("train", "val", "test"))
    device = get_device()
    t0 = time.perf_counter()
    model = SentenceTransformer(args.encoder, device=device)
    x = {name: cached_embeddings(model, args.encoder, d["id"], d["text"])
         for name, d in (("train", train), ("val", val), ("test", test))}
    enc_s = time.perf_counter() - t0
    log.info("Encoded %d CVs with %s on %s in %.1fs (dim %d, window %d tokens)",
             sum(map(len, x.values())), args.encoder, device, enc_s, x["train"].shape[1], model.get_max_seq_length())

    grid = pd.DataFrame([
        {"alpha": float(a), "val_mae": mean_absolute_error(
            val["exp_year"], clip_target(Ridge(alpha=a, random_state=SEED).fit(x["train"], train["exp_year"]).predict(x["val"])))}
        for a in ALPHAS]).sort_values("val_mae").reset_index(drop=True)
    grid.to_csv(TABLES_DIR / f"{MODEL_NAME}_grid.csv", index=False)
    alpha = float(grid.loc[0, "alpha"])
    if alpha in (ALPHAS.min(), ALPHAS.max()):
        log.warning("Best alpha %.3g is at the edge of the grid", alpha)
    log.info("Best alpha %.3g -> val MAE %.3f", alpha, grid.loc[0, "val_mae"])

    ridge = Ridge(alpha=alpha, random_state=SEED).fit(x["train"], train["exp_year"])
    joblib.dump({"encoder": args.encoder, "ridge": ridge}, MODELS_DIR / f"{MODEL_NAME}.joblib")
    report_validation(MODEL_NAME, val["id"], val["exp_year"], ridge.predict(x["val"]),
                      {"params": {"encoder": args.encoder, "alpha": alpha, "dim": int(x["train"].shape[1]),
                                  "max_seq_length": model.get_max_seq_length(), "normalize_embeddings": True},
                       "encode_time_s": round(enc_s, 1), "device": device})
    save_predictions(MODEL_NAME, "test", test["id"], test["exp_year"], ridge.predict(x["test"]))


if __name__ == "__main__":
    main()
