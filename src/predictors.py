"""Uniform `texts -> predicted years` interface over every trained model.

Used by the probe / noise / sensitivity experiments (src/experiments.py) and by the
Streamlit estimator (app.py), so a CV uploaded in the app goes through exactly the same
cleaning + model code as the evaluation.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from src.utils import MODELS_DIR, clip_target, get_device, get_logger, load_json

log = get_logger("predictors")

Predictor = Callable[[list[str]], np.ndarray]

LABELS = {
    "baseline_median": "Median",
    "baseline_tfidf": "TF-IDF + Ridge",
    "baseline_embeddings": "MiniLM embeddings + Ridge",
    "baseline_rules": "Rule-based date parser",
    "bert_distilbert_base_uncased": "DistilBERT regressor",
    "bert_deberta_v3_small": "DeBERTa-v3-small regressor",
}


def _median() -> Predictor:
    m = load_json(MODELS_DIR / "baseline_median.json")["median"]
    return lambda texts: np.full(len(texts), m, dtype=float)


def _tfidf(name: str = "baseline_tfidf") -> Predictor:
    import joblib

    pipe = joblib.load(MODELS_DIR / f"{name}.joblib")
    return lambda texts: clip_target(pipe.predict(list(texts)))


def _embeddings() -> Predictor:
    import joblib
    from sentence_transformers import SentenceTransformer

    from src.baseline_embeddings import encode_long

    bundle = joblib.load(MODELS_DIR / "baseline_embeddings.joblib")
    enc = SentenceTransformer(bundle["encoder"], device=get_device())
    return lambda texts: clip_target(bundle["ridge"].predict(encode_long(enc, list(texts))[0]))


def _rules() -> Predictor:
    from src.baseline_rules import load_model, predict

    imp = load_model()
    return lambda texts: clip_target(predict(pd.Series(list(texts)), imp)[0])


def _bert(name: str) -> Predictor:
    from src.train_bert import load_regressor, predict_texts

    model, tok, scaler, cfg = load_regressor(MODELS_DIR / name / "best")
    return lambda texts: predict_texts(model, tok, scaler, list(texts), cfg["max_length"], cfg["head_share"])


def load_predictor(name: str) -> Predictor:
    if name == "baseline_median":
        return _median()
    if name.startswith("baseline_tfidf"):
        return _tfidf(name)
    if name == "baseline_embeddings":
        return _embeddings()
    if name == "baseline_rules":
        return _rules()
    if name.startswith("bert_"):
        return _bert(name)
    raise ValueError(f"unknown model {name!r}")


def load_all(names=None) -> dict[str, Predictor]:
    """Load every model whose artefacts exist; missing ones are skipped with a warning."""
    out = {}
    for name in names or LABELS:
        try:
            out[name] = load_predictor(name)
        except (FileNotFoundError, OSError) as e:
            log.warning("Model %s unavailable (%s)", name, e)
    return out
