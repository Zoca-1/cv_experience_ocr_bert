"""Evaluation metrics and prediction I/O shared by every model.

Discipline: model scripts call `report_validation` (metrics on VAL only, used for model
selection) and `save_predictions` for TEST without computing test metrics. Test metrics
are produced once, for all models together, by the final evaluation (Step 6).

Metrics: MAE, RMSE, R^2, Acc +/-k (|error| <= k years, k = 1, 2, 3), band accuracy and
band macro-F1 (bands from utils.EXPERIENCE_BANDS). Predictions are clipped to
[0, cap] before scoring, the same range the target lives in.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, r2_score

from src.utils import BAND_LABELS, METRICS_DIR, PREDICTIONS_DIR, clip_target, get_logger, save_json, to_bands

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
