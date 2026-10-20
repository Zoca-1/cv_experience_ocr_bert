"""Baseline 0: predict the training-set median for every CV.

The floor every learned model must beat; the median (not the mean) minimises MAE,
the headline metric.

Usage:  python -m src.baseline_median
"""
from __future__ import annotations

import numpy as np

from src.evaluation import report_validation, save_predictions
from src.split import load_split
from src.utils import MODELS_DIR, ensure_dirs, get_logger, save_json

log = get_logger("baseline_median")
MODEL_NAME = "baseline_median"


def main() -> None:
    ensure_dirs()
    train, val, test = (load_split(s, with_text=False) for s in ("train", "val", "test"))
    median = float(np.median(train["exp_year"]))
    save_json({"median": median, "n_train": len(train)}, MODELS_DIR / f"{MODEL_NAME}.json")
    log.info("Train median exp_year = %.3f", median)

    report_validation(MODEL_NAME, val["id"], val["exp_year"], np.full(len(val), median), {"params": {"median": median}})
    save_predictions(MODEL_NAME, "test", test["id"], test["exp_year"], np.full(len(test), median))


if __name__ == "__main__":
    main()
