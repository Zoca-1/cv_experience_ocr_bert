"""Analysis experiments behind README questions 1-3 (run after the final evaluation).

Q1  What does BERT learn that the baselines cannot?  -> counterfactual date probes
    P1 shift_all_+5    every year in the CV +5 (ranges and graduation): spans unchanged,
                       the ideal prediction change is 0.
    P2 oldest_start_-5 the oldest job starts 5 years earlier: the true sum of spans grows
                       by 5 (clipped at the cap) - the ideal change is +5.
    P1b shift_all_-3   every year -3: same as P1 but staying inside the train year range
                       (1999-2023), so a change cannot be blamed on unseen year tokens.
    P3 dates_removed   every date range deleted: how much does each model rely on dates?
    Probes are also broken down by number of jobs in the CV (q1_probes_by_njobs.csv).
Q2  How much does OCR noise cost?
    (a) the OCR-trained models applied to the REFERENCE text of the same test CVs,
    (b) models trained and tested on reference text (TF-IDF refit here, DistilBERT via
        `train_bert --text-source ref`), and the render-truncated vs clean subsets,
    (c) dose-response: synthetic OCR noise injected into the test text (digit
        substitutions, and generic character substitutions) at increasing rates.
Q3  Methodological sensitivity: TF-IDF number handling and cleaning; BERT loss (MSE vs
    Huber), long-text strategy (head vs head+tail at a forced 64-token window) and seed.
    BERT variants are trained by scripts/run_sensitivity.sh; this module collects them.

All experiments are reported on the test set purely descriptively - no model or
hyper-parameter is selected from them (selection happened on validation).

Outputs: output/tables/q1_probes.csv, q2_ocr_vs_ref.csv, q2_noise_curve.csv,
         q3_sensitivity.csv and output/metrics/experiments.json

Usage:  python -m src.experiments [--skip-noise]
"""
from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd

from src.evaluation import load_predictions, regression_metrics
from src.predictors import LABELS, load_all, load_predictor
from src.preprocessing import DATE_RANGE_RE, PROCESSED_REF_PATH
from src.split import load_split
from src.utils import (
    METRICS_DIR, MODELS_DIR, PREDICTIONS_DIR, SEED, TABLES_DIR, clip_target, ensure_dirs, get_logger,
    get_target_cap, load_json, save_json,
)

log = get_logger("experiments")
YEAR_TOKEN = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
NOISE_RATES = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20)


# ---------------------------------------------------------------------------
# Q1 probes
# ---------------------------------------------------------------------------
def shift_all_years(text: str, k: int = 5) -> str:
    return YEAR_TOKEN.sub(lambda m: str(int(m.group(1)) + k), text)


def oldest_start_earlier(text: str, k: int = 5) -> str:
    matches = list(DATE_RANGE_RE.finditer(text))
    if not matches:
        return text
    m = min(matches, key=lambda m: (int(m.group(1)), int(m.group(2))))
    return text[: m.start(1)] + str(int(m.group(1)) - k) + text[m.end(1):]


def remove_dates(text: str) -> str:
    return DATE_RANGE_RE.sub("", text)


def run_probes(test: pd.DataFrame, predictors: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    cap = get_target_cap()
    texts = test["text"].tolist()
    expected_p2 = np.clip(test["exp_year"] + 5, 0, cap) - test["exp_year"]
    probes = {"shift_all_+5": [shift_all_years(t) for t in texts],
              "shift_all_-3": [shift_all_years(t, -3) for t in texts],
              "oldest_start_-5": [oldest_start_earlier(t) for t in texts],
              "dates_removed": [remove_dates(t) for t in texts]}
    from src.preprocessing import date_ranges

    n_jobs = np.array([len(date_ranges(t)) for t in texts])
    rows, by_jobs = [], []
    for name, predict in predictors.items():
        base = predict(texts)
        for probe, mod in probes.items():
            delta = predict(mod) - base
            for k in np.unique(n_jobs):
                sel = n_jobs == k
                by_jobs.append({"model": name, "probe": probe, "n_jobs": int(k), "n": int(sel.sum()),
                                "mean_delta": float(delta[sel].mean()),
                                "ideal": float(expected_p2[sel].mean()) if probe == "oldest_start_-5" else 0.0})
            expected = expected_p2.values if probe == "oldest_start_-5" else np.zeros(len(base))
            rows.append({"model": name, "label": LABELS.get(name, name), "probe": probe,
                         "mean_delta": float(delta.mean()), "mean_abs_delta": float(np.abs(delta).mean()),
                         "expected_mean_delta": float(expected.mean()) if probe != "dates_removed" else np.nan,
                         "mae_vs_expected": float(np.abs(delta - expected).mean()) if probe != "dates_removed" else np.nan,
                         "response_ratio": float(delta.mean() / expected.mean()) if probe == "oldest_start_-5" else np.nan})
        r = {x["probe"]: x for x in rows[-len(probes):]}
        log.info("[probe] %-30s P1 d=%+.3f  P1b d=%+.3f  P2 d=%+.3f (ideal %.3f)  P3 |d|=%.3f", name,
                 r["shift_all_+5"]["mean_delta"], r["shift_all_-3"]["mean_delta"], r["oldest_start_-5"]["mean_delta"],
                 r["oldest_start_-5"]["expected_mean_delta"], r["dates_removed"]["mean_abs_delta"])
    return pd.DataFrame(rows), pd.DataFrame(by_jobs)


# ---------------------------------------------------------------------------
# Q2 OCR noise
# ---------------------------------------------------------------------------
def inject_noise(text: str, rate: float, kind: str, rng: np.random.Generator) -> str:
    """Substitute characters at `rate`: kind='digits' only touches digits (OCR digit
    confusions such as 0<->8, 1<->7), kind='chars' touches any alphanumeric character."""
    if rate <= 0:
        return text
    chars = list(text)
    for i, c in enumerate(chars):
        if (c.isdigit() if kind == "digits" else c.isalnum()) and rng.random() < rate:
            pool = "0123456789" if c.isdigit() else "abcdefghijklmnopqrstuvwxyz"
            chars[i] = rng.choice([p for p in pool if p != c.lower()])
    return "".join(chars)


def run_ocr_vs_ref(test: pd.DataFrame, predictors: dict) -> pd.DataFrame:
    ref = load_split("test", text_path=PROCESSED_REF_PATH).set_index("id").loc[test["id"], "text"].tolist()
    rows = []
    for name, predict in predictors.items():
        for source, texts in (("ocr", test["text"].tolist()), ("reference", ref)):
            m = regression_metrics(test["exp_year"], predict(texts))
            rows.append({"model": name, "label": LABELS.get(name, name), "trained_on": "ocr", "tested_on": source, **m})
        for flag, g in test.groupby("truncated"):
            m = regression_metrics(g["exp_year"], predict(g["text"].tolist()))
            rows.append({"model": name, "label": LABELS.get(name, name), "trained_on": "ocr",
                         "tested_on": f"ocr_{'truncated' if flag else 'clean'}_subset", **m})

    # trained AND tested on reference text
    import joblib
    from src.baseline_tfidf import build

    p = load_json(METRICS_DIR / "baseline_tfidf_val.json")["params"]
    tr_ref = load_split("train", text_path=PROCESSED_REF_PATH)
    tfidf_ref = build(p["vectorizer"], p["min_df"], p["alpha"]).fit(tr_ref["text"], tr_ref["exp_year"])
    rows.append({"model": "baseline_tfidf", "label": LABELS["baseline_tfidf"], "trained_on": "reference",
                 "tested_on": "reference", **regression_metrics(test["exp_year"], tfidf_ref.predict(ref))})
    ref_bert = PREDICTIONS_DIR / "bert_distilbert_ref_test.csv"
    if ref_bert.exists():
        pr = pd.read_csv(ref_bert, dtype={"id": str}).set_index("id").loc[test["id"]]
        rows.append({"model": "bert_distilbert_base_uncased", "label": LABELS["bert_distilbert_base_uncased"],
                     "trained_on": "reference", "tested_on": "reference", **regression_metrics(pr["y_true"], pr["y_pred_raw"])})
    else:
        log.warning("bert_distilbert_ref not trained: run scripts/run_sensitivity.sh for the Q2 ref-trained BERT")
    return pd.DataFrame(rows)


def run_noise_curve(test: pd.DataFrame, predictors: dict) -> pd.DataFrame:
    rows = []
    for kind in ("digits", "chars"):
        for rate in NOISE_RATES:
            rng = np.random.default_rng([SEED, int(rate * 1000), kind == "digits"])
            noisy = [inject_noise(t, rate, kind, rng) for t in test["text"]]
            for name, predict in predictors.items():
                m = regression_metrics(test["exp_year"], predict(noisy))
                rows.append({"noise": kind, "rate": rate, "model": name, "label": LABELS.get(name, name),
                             "mae": m["mae"], "acc_pm1": m["acc_pm1"], "acc_pm2": m["acc_pm2"]})
            log.info("[noise] %-6s rate %.2f: %s", kind, rate,
                     {n: r["mae"] for n, r in zip(predictors, rows[-len(predictors):])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Q3 sensitivity
# ---------------------------------------------------------------------------
BERT_VARIANTS = {
    "bert_distilbert_base_uncased": ("BERT loss / truncation / seed", "reference run: MSE, head+tail, 512, seed 42"),
    "bert_distilbert_huber": ("BERT loss", "Huber loss (delta=1 on z-scored target)"),
    "bert_distilbert_len64_headtail": ("BERT long-text strategy", "64-token window, head 75% + tail 25%"),
    "bert_distilbert_len64_head": ("BERT long-text strategy", "64-token window, head only"),
    "bert_distilbert_seed43": ("BERT seed", "seed 43"),
    "bert_distilbert_seed44": ("BERT seed", "seed 44"),
}


def tfidf_variants(train: pd.DataFrame, test: pd.DataFrame, val: pd.DataFrame) -> list[dict]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline

    from src.utils import OCR_DIR, load_json as lj

    p = load_json(METRICS_DIR / "baseline_tfidf_val.json")["params"]
    base = dict(lowercase=True, sublinear_tf=True, min_df=p["min_df"], analyzer=p["analyzer"],
                ngram_range=tuple(p["ngram_range"]))
    raw = {s: d["id"].map(lambda i: lj(OCR_DIR / f"{i}.json")["text"]).tolist() for s, d in
           (("train", train), ("val", val), ("test", test))}

    def digits_masked(texts):
        return [re.sub(r"\d", "0", t) for t in texts]

    variants = [
        ("TF-IDF numbers", "best config (char_wb 2-5, cleaned OCR text)", base, lambda s: s),
        ("TF-IDF numbers", "every digit replaced by 0 (numbers carry no value)", base, digits_masked),
        ("TF-IDF numbers", "word 1-3 n-grams, dates kept as single tokens '2018.09-2023.01'",
         dict(base, analyzer="word", ngram_range=(1, 3), token_pattern=r"(?u)\b[\w.\-]+\b"), lambda s: s),
        ("TF-IDF cleaning", "raw OCR text (no canonicalization, PII kept)", base, None),
    ]
    rows = []
    for group, desc, kw, transform in variants:
        pipe = Pipeline([("tfidf", TfidfVectorizer(**kw)), ("ridge", Ridge(alpha=p["alpha"], random_state=SEED))])
        if transform is None:
            pipe.fit(raw["train"], train["exp_year"])
            pv, pt = pipe.predict(raw["val"]), pipe.predict(raw["test"])
        else:
            pipe.fit(transform(train["text"].tolist()), train["exp_year"])
            pv, pt = pipe.predict(transform(val["text"].tolist())), pipe.predict(transform(test["text"].tolist()))
        rows.append({"group": group, "variant": desc, "val_mae": regression_metrics(val["exp_year"], pv)["mae"],
                     "test_mae": regression_metrics(test["exp_year"], pt)["mae"]})
    return rows


def run_sensitivity(train, val, test) -> pd.DataFrame:
    rows = tfidf_variants(train, test, val)
    for name, (group, desc) in BERT_VARIANTS.items():
        vpath = METRICS_DIR / f"{name}_val.json"
        tpath = PREDICTIONS_DIR / f"{name}_test.csv"
        if not (vpath.exists() and tpath.exists()):
            log.warning("Sensitivity run %s missing (scripts/run_sensitivity.sh)", name)
            continue
        info = load_json(vpath)
        t = load_predictions(name, "test")
        rows.append({"group": group, "variant": desc, "val_mae": info["metrics"]["mae"],
                     "test_mae": regression_metrics(t["y_true"], t["y_pred_raw"])["mae"],
                     "best_epoch": info.get("best_epoch"), "train_time_s": info.get("train_time_s"),
                     "n_truncated_val": info["params"].get("n_truncated_val")})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-noise", action="store_true")
    args = parser.parse_args()
    ensure_dirs()
    train, val, test = (load_split(s) for s in ("train", "val", "test"))
    predictors = load_all()
    log.info("Loaded predictors: %s", list(predictors))

    probes, probes_jobs = run_probes(test, predictors)
    probes.to_csv(TABLES_DIR / "q1_probes.csv", index=False)
    probes_jobs.to_csv(TABLES_DIR / "q1_probes_by_njobs.csv", index=False)
    q2 = run_ocr_vs_ref(test, predictors)
    q2.to_csv(TABLES_DIR / "q2_ocr_vs_ref.csv", index=False)
    noise = None if args.skip_noise else run_noise_curve(test, predictors)
    if noise is not None:
        noise.to_csv(TABLES_DIR / "q2_noise_curve.csv", index=False)
    sens = run_sensitivity(train, val, test)
    sens.to_csv(TABLES_DIR / "q3_sensitivity.csv", index=False)
    log.info("Q2 OCR vs reference:\n%s", q2[["label", "trained_on", "tested_on", "n", "mae", "acc_pm1"]].round(3).to_string(index=False))
    log.info("Q3 sensitivity:\n%s", sens.round(3).to_string(index=False))
    save_json({"q1_probes": probes.to_dict("records"), "q2_ocr_vs_ref": q2.to_dict("records"),
               "q2_noise_curve": None if noise is None else noise.to_dict("records"),
               "q3_sensitivity": sens.to_dict("records")}, METRICS_DIR / "experiments.json")


if __name__ == "__main__":
    main()
