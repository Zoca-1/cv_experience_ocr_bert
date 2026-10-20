"""Bonus baseline: rule-based date parser - sum of employment spans read from OCR text.

PopResume builds exp_year from the job history: on the reference text the sum of
(end - start) over all "YYYY.MM-YYYY.MM" ranges reproduces the target with MAE 0.02 y.
This model applies that rule to the cleaned OCR text, so it is a transparent reference
for "how much of the task is just reading and adding dates" (README Q1/Q2) - not a
learned model and not one of the required baselines.

Truncated ranges ("2020.01-2021." / "2023.0", the 38 CVs cut in the rendered image)
cannot be read exactly. The missing end month is imputed with the median end month
observed in COMPLETE train ranges with the same end year and the same visible month
digit (falling back to the same end year, then the global median), i.e. the imputation
is learned on train only.

Usage:  python -m src.baseline_rules
"""
from __future__ import annotations

import re
import time

import numpy as np
import pandas as pd

from src.evaluation import report_validation, save_predictions
from src.preprocessing import DATE_RANGE_RE, span_years
from src.split import load_split
from src.utils import MODELS_DIR, ensure_dirs, get_logger, load_json, save_json

log = get_logger("baseline_rules")
MODEL_NAME = "baseline_rules"
PARTIAL_RANGE_RE = re.compile(r"\b((?:19|20)\d{2})\.(\d{2})\s*-\s*((?:19|20)\d{2})(?:\.(\d?))?\s*$", re.M)


def fit_month_imputer(texts: pd.Series) -> dict:
    """Median end month by (end_year, first month digit) and by end_year, from complete ranges."""
    ends = pd.DataFrame([(y2, m2) for t in texts for _, _, y2, m2 in map(lambda r: tuple(map(int, r)), DATE_RANGE_RE.findall(t))],
                        columns=["year", "month"])
    ends["digit"] = ends["month"] // 10
    return {
        "by_year_digit": {f"{y}|{d}": int(g.median()) for (y, d), g in ends.groupby(["year", "digit"])["month"]},
        "by_year": {str(y): int(g.median()) for y, g in ends.groupby("year")["month"]},
        "global": int(ends["month"].median()),
    }


def impute_month(imp: dict, year: int, digit: str | None) -> int:
    if digit:
        m = imp["by_year_digit"].get(f"{year}|{digit}")
        if m is not None:
            return m
    return imp["by_year"].get(str(year), imp["global"])


def parse_ranges(text: str, imp: dict) -> tuple[list[tuple[int, int, int, int]], int]:
    """Complete ranges + truncated ones with an imputed end month; returns (ranges, n_imputed)."""
    ranges = [tuple(map(int, r)) for r in DATE_RANGE_RE.findall(text)]
    n_imp = 0
    for m in PARTIAL_RANGE_RE.finditer(text):
        if DATE_RANGE_RE.search(m.group(0)):
            continue                                    # complete range, already counted
        y1, m1, y2 = int(m.group(1)), int(m.group(2)), int(m.group(3))
        ranges.append((y1, m1, y2, impute_month(imp, y2, m.group(4))))
        n_imp += 1
    return ranges, n_imp


def predict(texts: pd.Series, imp: dict) -> tuple[np.ndarray, np.ndarray]:
    out = [parse_ranges(t, imp) for t in texts]
    return np.array([span_years(r) for r, _ in out]), np.array([n for _, n in out])


def load_model() -> dict:
    return load_json(MODELS_DIR / f"{MODEL_NAME}.json")


def main() -> None:
    ensure_dirs()
    train, val, test = (load_split(s) for s in ("train", "val", "test"))
    t0 = time.perf_counter()
    imp = fit_month_imputer(train["text"])
    save_json(imp, MODELS_DIR / f"{MODEL_NAME}.json")

    tr_pred, _ = predict(train["text"], imp)
    log.info("Train MAE of the rule (sanity, nothing is fitted but the month imputer): %.4f",
             np.abs(np.clip(tr_pred, 0, None) - train["exp_year"]).mean())
    va_pred, va_imp = predict(val["text"], imp)
    te_pred, te_imp = predict(test["text"], imp)
    log.info("Ranges with imputed end month: val %d CVs, test %d CVs", (va_imp > 0).sum(), (te_imp > 0).sum())
    report_validation(MODEL_NAME, val["id"], val["exp_year"], va_pred,
                      {"params": {"imputer": "median end month by (end year, visible digit) from train"},
                       "n_val_imputed_cvs": int((va_imp > 0).sum()), "fit_time_s": round(time.perf_counter() - t0, 2)})
    save_predictions(MODEL_NAME, "test", test["id"], test["exp_year"], te_pred, n_imputed=te_imp)


if __name__ == "__main__":
    main()
