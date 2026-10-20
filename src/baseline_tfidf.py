"""Baseline 1: TF-IDF + Ridge regression, n-grams and regularisation tuned on validation.

Grid (selected by validation MAE, fitted on train only so the comparison with BERT,
which uses val for early stopping, is like for like):
  * word n-grams (1,1) / (1,2) / (1,3)   and   char_wb n-grams (2,5)
  * min_df 1 / 2, sublinear tf
  * alpha in logspace(-4, 2)
The default token pattern splits "2018.09-2023.01" into 2018 / 09 / 2023 / 01: a bag of
n-grams sees WHICH years occur, but not the difference between start and end that the
target is built from - a structural limit discussed in README Q1.

Usage:  python -m src.baseline_tfidf
"""
from __future__ import annotations

import itertools
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

from src.evaluation import report_validation, save_predictions
from src.split import load_split
from src.utils import MODELS_DIR, SEED, TABLES_DIR, clip_target, ensure_dirs, get_logger, set_seed

log = get_logger("baseline_tfidf")
MODEL_NAME = "baseline_tfidf"

VECTORIZERS = {
    "word_1-1": dict(analyzer="word", ngram_range=(1, 1)),
    "word_1-2": dict(analyzer="word", ngram_range=(1, 2)),
    "word_1-3": dict(analyzer="word", ngram_range=(1, 3)),
    "char_wb_2-5": dict(analyzer="char_wb", ngram_range=(2, 5)),
}
MIN_DF = (1, 2)
ALPHAS = np.logspace(-4, 2, 13)


def build(vec: str, min_df: int, alpha: float) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, sublinear_tf=True, min_df=min_df, **VECTORIZERS[vec])),
        ("ridge", Ridge(alpha=alpha, random_state=SEED)),
    ])


def main() -> None:
    ensure_dirs()
    set_seed()
    train, val, test = (load_split(s) for s in ("train", "val", "test"))

    t0 = time.perf_counter()
    rows = []
    for vec, min_df in itertools.product(VECTORIZERS, MIN_DF):
        tfidf = TfidfVectorizer(lowercase=True, sublinear_tf=True, min_df=min_df, **VECTORIZERS[vec])
        x_tr, x_va = tfidf.fit_transform(train["text"]), None
        x_va = tfidf.transform(val["text"])
        for alpha in ALPHAS:
            pred = Ridge(alpha=alpha, random_state=SEED).fit(x_tr, train["exp_year"]).predict(x_va)
            rows.append({"vectorizer": vec, "min_df": min_df, "alpha": float(alpha), "n_features": x_tr.shape[1],
                         "val_mae": mean_absolute_error(val["exp_year"], clip_target(pred))})
    grid = pd.DataFrame(rows).sort_values("val_mae").reset_index(drop=True)
    grid.to_csv(TABLES_DIR / f"{MODEL_NAME}_grid.csv", index=False)
    best = grid.iloc[0]
    log.info("Grid: %d configs in %.1fs. Best: %s min_df=%d alpha=%.3g (%d features) -> val MAE %.3f",
             len(grid), time.perf_counter() - t0, best.vectorizer, best.min_df, best.alpha, best.n_features, best.val_mae)
    log.info("Best per vectorizer:\n%s", grid.groupby("vectorizer").first()[["min_df", "alpha", "val_mae"]].round(4).to_string())

    t1 = time.perf_counter()
    model = build(best.vectorizer, int(best.min_df), float(best.alpha)).fit(train["text"], train["exp_year"])
    fit_s = time.perf_counter() - t1
    joblib.dump(model, MODELS_DIR / f"{MODEL_NAME}.joblib")

    params = {"vectorizer": best.vectorizer, **{k: v for k, v in VECTORIZERS[best.vectorizer].items()},
              "min_df": int(best.min_df), "alpha": float(best.alpha), "n_features": int(best.n_features)}
    report_validation(MODEL_NAME, val["id"], val["exp_year"], model.predict(val["text"]),
                      {"params": params, "fit_time_s": round(fit_s, 2), "grid_size": len(grid)})
    save_predictions(MODEL_NAME, "test", test["id"], test["exp_year"], model.predict(test["text"]))


if __name__ == "__main__":
    main()
