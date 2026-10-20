"""Evaluation metrics and prediction I/O shared by every model.

Discipline: model scripts call `report_validation` (metrics on VAL only, used for model
selection) and `save_predictions` for TEST without computing test metrics. Test metrics
are produced once, for all models together, by the final evaluation (Step 6).

Metrics: MAE, RMSE, R^2, Acc +/-k (|error| <= k years, k = 1, 2, 3), band accuracy and
band macro-F1 (bands from utils.EXPERIENCE_BANDS). Predictions are clipped to
[0, cap] before scoring, the same range the target lives in.

`python -m src.evaluation` runs the final TEST evaluation of every trained model:
  output/tables/model_comparison.csv     all metrics + train time + inference ms/CV
  output/tables/bootstrap_mae_gains.csv  primary BERT's MAE gain vs each model, paired
                                         bootstrap (1,000 resamples of the test CVs), 95 % CI
  output/tables/mae_by_{occupation,band,truncated}.csv
  output/tables/fairness_mae.csv, fairness_gaps.csv   MAE by sex / race (evaluation only)
  output/tables/bert_top10_errors.csv    largest BERT errors with diagnosed causes
  output/metrics/final_evaluation.json
The primary BERT is the BERT run with the lowest VALIDATION MAE (never chosen on test).
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, r2_score

from src.utils import (
    BAND_LABELS, METRICS_DIR, MODELS_DIR, PREDICTIONS_DIR, RAW_METADATA_PATH, SEED, TABLES_DIR, clip_target,
    ensure_dirs, get_device, get_logger, get_target_cap, load_json, save_json, to_bands,
)

log = get_logger("evaluation")

ACC_KS = (1, 2, 3)


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    p = clip_target(y_pred)
    err = np.abs(p - y)
    bt, bp = to_bands(y), to_bands(p)
    out = {
        "n": int(len(y)),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "r2": float(r2_score(y, p)),
        **{f"acc_pm{k}": float(np.mean(err <= k)) for k in ACC_KS},
        "band_accuracy": float(np.mean(bt == bp)),
        "band_macro_f1": float(f1_score(bt, bp, labels=list(BAND_LABELS), average="macro", zero_division=0)),
    }
    return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in out.items()}


def save_predictions(model: str, split: str, ids, y_true, y_pred, **extra) -> pd.DataFrame:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"id": list(ids), "y_true": np.asarray(y_true, float),
                       "y_pred": clip_target(y_pred), "y_pred_raw": np.asarray(y_pred, float), **extra})
    df.to_csv(PREDICTIONS_DIR / f"{model}_{split}.csv", index=False)
    return df


def report_validation(model: str, ids, y_true, y_pred, info: dict[str, Any] | None = None) -> dict:
    """Save val predictions + val metrics (with training info) and log them."""
    save_predictions(model, "val", ids, y_true, y_pred)
    metrics = regression_metrics(y_true, y_pred)
    save_json({"model": model, "split": "val", "metrics": metrics, **(info or {})}, METRICS_DIR / f"{model}_val.json")
    log.info("[%s] VAL  MAE %.3f | RMSE %.3f | R2 %.3f | Acc+-1 %.3f | Acc+-2 %.3f | band acc %.3f | band F1 %.3f",
             model, metrics["mae"], metrics["rmse"], metrics["r2"], metrics["acc_pm1"], metrics["acc_pm2"],
             metrics["band_accuracy"], metrics["band_macro_f1"])
    return metrics


# ---------------------------------------------------------------------------
# Final evaluation on the held-out TEST set (Step 6)
# ---------------------------------------------------------------------------
# (name, label, family). The primary BERT is picked by VALIDATION MAE among the BERT runs.
MODELS: tuple[tuple[str, str, str], ...] = (
    ("baseline_median", "Median", "baseline"),
    ("baseline_tfidf", "TF-IDF + Ridge", "baseline"),
    ("baseline_embeddings", "MiniLM embeddings + Ridge", "baseline"),
    ("baseline_rules", "Rule-based date parser", "reference"),
    ("bert_distilbert_base_uncased", "DistilBERT regressor", "bert"),
    ("bert_deberta_v3_small", "DeBERTa-v3-small regressor", "bert"),
)
N_BOOT = 1000
CI = 0.95


def load_predictions(model: str, split: str) -> pd.DataFrame:
    path = PREDICTIONS_DIR / f"{model}_{split}.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path.name}: train the model first (see README).")
    return pd.read_csv(path, dtype={"id": str})


def available_models() -> list[tuple[str, str, str]]:
    have = [m for m in MODELS if (PREDICTIONS_DIR / f"{m[0]}_test.csv").exists()]
    missing = [m[0] for m in MODELS if m not in have]
    if missing:
        log.warning("Skipping models without test predictions: %s", missing)
    return have


def primary_bert(models) -> str:
    berts = [m for m, _, fam in models if fam == "bert"]
    if not berts:
        raise SystemExit("No BERT predictions found")
    return min(berts, key=lambda m: load_json(METRICS_DIR / f"{m}_val.json")["metrics"]["mae"])


def paired_bootstrap(y, pred_a, pred_b, n_boot: int = N_BOOT, seed: int = SEED, ci: float = CI) -> dict:
    """MAE(a) - MAE(b) with a paired percentile bootstrap over CVs (same resample for both)."""
    y = np.asarray(y, dtype=float)
    ea, eb = np.abs(clip_target(pred_a) - y), np.abs(clip_target(pred_b) - y)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    deltas = ea[idx].mean(1) - eb[idx].mean(1)
    lo, hi = np.quantile(deltas, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    gain = ea.mean() - eb.mean()
    return {"mae_a": float(ea.mean()), "mae_b": float(eb.mean()), "gain": float(gain),
            "gain_rel": float(gain / ea.mean()) if ea.mean() else np.nan,
            "ci_low": float(lo), "ci_high": float(hi), "p_boot_le_0": float(np.mean(deltas <= 0)), "n_boot": n_boot}


def inference_ms_per_cv(model: str, texts: list[str]) -> float | None:
    """Text-to-prediction latency per CV (OCR excluded, reported separately)."""
    import joblib

    t0 = time.perf_counter()
    if model == "baseline_median":
        _ = np.full(len(texts), load_json(MODELS_DIR / "baseline_median.json")["median"])
    elif model == "baseline_tfidf":
        joblib.load(MODELS_DIR / "baseline_tfidf.joblib").predict(texts)
        t0 = time.perf_counter()
        joblib.load(MODELS_DIR / "baseline_tfidf.joblib").predict(texts)
    elif model == "baseline_rules":
        from src.baseline_rules import load_model, predict
        predict(pd.Series(texts), load_model())
    elif model == "baseline_embeddings":
        from sentence_transformers import SentenceTransformer
        from src.baseline_embeddings import encode_long
        bundle = joblib.load(MODELS_DIR / "baseline_embeddings.joblib")
        enc = SentenceTransformer(bundle["encoder"], device=get_device())
        encode_long(enc, texts[:32])                                 # warm-up
        t0 = time.perf_counter()
        bundle["ridge"].predict(encode_long(enc, texts)[0])
    elif model.startswith("bert_"):
        return load_json(METRICS_DIR / f"{model}_val.json").get("inference_ms_per_cv")
    else:
        return None
    return round(1000 * (time.perf_counter() - t0) / len(texts), 3)


def train_time_s(model: str) -> float | None:
    info = load_json(METRICS_DIR / f"{model}_val.json")
    for k in ("train_time_s", "fit_time_s", "encode_time_s"):
        if k in info:
            return info[k]
    return 0.0


def breakdown(test: pd.DataFrame, preds: dict[str, pd.Series], by: str) -> pd.DataFrame:
    rows = []
    for key, g in test.groupby(by):
        row = {by: key, "n": len(g)}
        for m, p in preds.items():
            row[m] = float(np.mean(np.abs(clip_target(p.loc[g.index]) - g["exp_year"])))
        rows.append(row)
    return pd.DataFrame(rows)


# --- error analysis --------------------------------------------------------
def job_structure(ref_text: str) -> dict:
    from src.preprocessing import canonicalize, date_ranges
    r = sorted(date_ranges(canonicalize(ref_text)), key=lambda t: (t[0], t[1]))
    months = [(y1 * 12 + m1, y2 * 12 + m2) for y1, m1, y2, m2 in r]
    gaps = [max(0, s2 - e1) for (_, e1), (s2, _) in zip(months, months[1:])]
    overlaps = [max(0, e1 - s2) for (_, e1), (s2, _) in zip(months, months[1:])]
    return {"n_jobs": len(r), "gap_years": sum(gaps) / 12, "max_gap_years": max(gaps, default=0) / 12,
            "overlap_years": sum(overlaps) / 12,
            "first_to_last_years": (months[-1][1] - months[0][0]) / 12 if months else 0.0}


def error_causes(row: pd.Series, cap: int) -> list[str]:
    """Diagnosed causes, most explanatory first (data issues before model limitations)."""
    causes = []
    if row["truncated"]:
        causes.append("render truncation (end month missing in image)")
    if row["exp_year_raw"] > cap:
        causes.append(f"target clipped at {cap}")
    if row["ocr_span_err"] > 0.05:
        causes.append("OCR date error")
    if row["exp_year"] >= 15:
        causes.append("sparse high-experience region (>=15 y, ~5% of train)")
    if row["y_pred"] >= cap - 1e-6:
        causes.append(f"over-prediction clipped at the {cap}-year cap")
    if row["gap_years"] >= 1:
        causes.append(f"employment gap ({row['gap_years']:.1f} y)")
    if row["overlap_years"] > 0:
        causes.append("overlapping jobs")
    if row["n_jobs"] >= 4:
        causes.append("4 jobs to add up")
    if row["n_jobs"] == 1 and row["exp_year"] >= 8:
        causes.append("single long tenure")
    return causes or ["no data issue: arithmetic error of the model"]


def top_errors(test: pd.DataFrame, model: str, k: int = 10) -> pd.DataFrame:
    cap = get_target_cap()
    d = test.copy()
    d["y_pred"] = clip_target(d[f"pred_{model}"])
    d["error"] = d["y_pred"] - d["exp_year"]
    d["abs_error"] = d["error"].abs()
    d = d.sort_values("abs_error", ascending=False).head(k).copy()
    struct = pd.DataFrame([job_structure(t) for t in d["text_ref"]], index=d.index)
    d = d.join(struct)
    d["ocr_span_err"] = d["span_abs_err"]
    d["causes"] = d.apply(lambda r: "; ".join(error_causes(r, cap)), axis=1)
    d["primary_cause"] = d["causes"].str.split("; ").str[0]
    keep = ["id", "occupation", "exp_year", "y_pred", "error", "abs_error", "pred_baseline_rules", "n_jobs",
            "gap_years", "overlap_years", "first_to_last_years", "truncated", "primary_cause", "causes", "text"]
    return d[keep].rename(columns={"pred_baseline_rules": "rules_pred"})


def run_final_evaluation() -> None:
    from src.split import load_split

    models = available_models()
    bert = primary_bert(models)
    test = load_split("test")
    meta = pd.read_parquet(RAW_METADATA_PATH, columns=["id", "sex", "race", "exp_year_raw", "text_ref"])
    q = pd.read_csv(TABLES_DIR / "ocr_quality_per_cv.csv", dtype={"id": str}, usecols=["id", "span_abs_err"])
    test = test.merge(meta, on="id", validate="one_to_one").merge(q, on="id", validate="one_to_one").set_index("id", drop=False)

    preds = {}
    for m, _, _ in models:
        p = load_predictions(m, "test").set_index("id")
        if set(p.index) != set(test.index):
            raise SystemExit(f"{m}: test predictions do not cover the frozen test ids")
        preds[m] = p.loc[test.index, "y_pred_raw"]
        test[f"pred_{m}"] = preds[m]

    # 1) comparison table
    rows = []
    texts = test["text"].tolist()
    for m, label, fam in models:
        met = regression_metrics(test["exp_year"], preds[m])
        val = load_json(METRICS_DIR / f"{m}_val.json")["metrics"]
        rows.append({"model": m, "label": label, "family": fam, **met, "val_mae": val["mae"],
                     "share_err_gt2": float(np.mean(np.abs(clip_target(preds[m]) - test["exp_year"]) > 2)),
                     "share_err_gt5": float(np.mean(np.abs(clip_target(preds[m]) - test["exp_year"]) > 5)),
                     "train_time_s": train_time_s(m), "inference_ms_per_cv": inference_ms_per_cv(m, texts),
                     "primary_bert": m == bert})
    comp = pd.DataFrame(rows)
    ocr_ms = load_json(METRICS_DIR / "ocr_runtime.json")["runtime_per_cv_s"]["mean"] * 1000
    comp["end_to_end_ms_per_cv"] = comp["inference_ms_per_cv"] + ocr_ms
    comp.to_csv(TABLES_DIR / "model_comparison.csv", index=False)
    log.info("TEST comparison (primary BERT by val MAE: %s):\n%s", bert,
             comp[["label", "mae", "rmse", "r2", "acc_pm1", "acc_pm2", "acc_pm3", "band_accuracy", "band_macro_f1",
                   "train_time_s", "inference_ms_per_cv"]].round(3).to_string(index=False))

    # 2) paired bootstrap of BERT's MAE gain vs every other model
    boots = []
    for m, label, _ in models:
        if m == bert:
            continue
        b = paired_bootstrap(test["exp_year"], preds[m], preds[bert])
        boots.append({"bert": bert, "vs_model": m, "vs_label": label, **b})
    boot = pd.DataFrame(boots)
    boot.to_csv(TABLES_DIR / "bootstrap_mae_gains.csv", index=False)
    log.info("BERT MAE gain vs others (paired bootstrap, %d resamples, %.0f%% CI):\n%s", N_BOOT, 100 * CI,
             boot[["vs_label", "mae_a", "mae_b", "gain", "ci_low", "ci_high"]].round(4).to_string(index=False))

    # 3) breakdowns
    test["band"] = to_bands(test["exp_year"])
    for by in ("occupation", "band", "truncated"):
        breakdown(test, preds, by).to_csv(TABLES_DIR / f"mae_by_{by}.csv", index=False)

    # 4) fairness: sex / race are used HERE ONLY, never as features
    fair_rows = []
    for attr in ("sex", "race"):
        for grp, g in test.groupby(attr):
            for m, label, _ in models:
                e = np.abs(clip_target(preds[m].loc[g.index]) - g["exp_year"])
                fair_rows.append({"attribute": attr, "group": grp, "n": len(g), "model": m, "label": label,
                                  "mae": float(e.mean()), "mean_signed_error": float((clip_target(preds[m].loc[g.index]) - g["exp_year"]).mean()),
                                  "share_err_gt2": float((e > 2).mean())})
    fair = pd.DataFrame(fair_rows)
    # Gap = max - min group MAE. Its bootstrap CI can never contain 0 (the statistic is >= 0),
    # so significance comes from a permutation test: shuffle group labels, recompute the gap.
    gaps = []
    rng = np.random.default_rng(SEED)
    for attr in ("sex", "race"):
        labels = test[attr].to_numpy()
        groups = sorted(set(labels))
        perms = [rng.permutation(labels) for _ in range(N_BOOT)]
        for m, label, _ in models:
            e = np.abs(clip_target(preds[m]) - test["exp_year"]).to_numpy()
            obs = {g: float(e[labels == g].mean()) for g in groups}
            gap = max(obs.values()) - min(obs.values())
            null = np.array([max(e[pl == g].mean() for g in groups) - min(e[pl == g].mean() for g in groups) for pl in perms])
            gaps.append({"attribute": attr, "model": m, "label": label, "max_group_mae_gap": gap,
                         "perm_p_value": float((np.sum(null >= gap) + 1) / (len(null) + 1)),
                         "null_gap_mean": float(null.mean()), "worst_group": max(obs, key=obs.get),
                         "best_group": min(obs, key=obs.get), **{f"mae_{g}": v for g, v in obs.items()}})
    fair.to_csv(TABLES_DIR / "fairness_mae.csv", index=False)
    pd.DataFrame(gaps).to_csv(TABLES_DIR / "fairness_gaps.csv", index=False)

    # 5) top-10 BERT errors with causes
    top = top_errors(test, bert)
    top.to_csv(TABLES_DIR / "bert_top10_errors.csv", index=False)
    log.info("Top-10 %s errors:\n%s", bert, top[["id", "occupation", "exp_year", "y_pred", "error", "rules_pred",
                                                 "n_jobs", "gap_years", "primary_cause"]].round(2).to_string(index=False))

    # predictions joined with meta for the app / figures (sex/race excluded)
    test.drop(columns=["sex", "race", "text_ref"]).to_csv(PREDICTIONS_DIR / "test_all_models.csv", index=False)
    save_json({"primary_bert": bert, "n_test": len(test), "n_boot": N_BOOT, "ci": CI,
               "models": [m for m, _, _ in models], "ocr_ms_per_cv": ocr_ms,
               "comparison": comp.drop(columns=["family"]).to_dict("records"),
               "bootstrap": boot.to_dict("records"), "fairness_gaps": gaps}, METRICS_DIR / "final_evaluation.json")


def main() -> None:
    import argparse
    argparse.ArgumentParser(description="Final TEST evaluation of all models (Step 6)").parse_args()
    ensure_dirs()
    run_final_evaluation()


if __name__ == "__main__":
    main()
