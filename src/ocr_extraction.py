"""PaddleOCR extraction with reading-order reconstruction and a per-CV JSON cache.

For every image in data/raw/images/ this module runs PaddleOCR (lang="en"), keeps
`rec_texts`, `rec_scores` and `rec_boxes`, rebuilds the natural reading order and writes
data/ocr/<id>.json atomically. Already-cached CVs are skipped, so the run is resumable.

Reading order - recursive XY-cut with a CV-specific guard
  * A region is split VERTICALLY (columns, left -> right) when a whitespace gutter crosses
    it top to bottom, and HORIZONTALLY (top -> bottom) at its widest whitespace band
    otherwise. Vertical cuts are tried first so a sidebar + main column is never
    interleaved line by line.
  * Guard: CVs very often right-align dates / locations on the same row as the job
    title ("Staff Nurse, Mercy Hospital ........ 2015 - Present"). That also produces a
    vertical gutter, and cutting there would emit all titles and then all dates. A
    vertical cut is therefore REJECTED when the smaller side
      - is mostly date/location-like and row-aligned with the other side, or
      - is sparse (<= half the items) and almost entirely row-aligned with it.
    The region then falls back to horizontal cuts, which keep title and date on one line.
  * Leaves are grouped into lines (vertical overlap >= 50 % of the smaller box height)
    and each line is read left -> right.

Usage:  python -m src.ocr_extraction [--limit N] [--device gpu:0] [--force]
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils import (
    METRICS_DIR, OCR_DIR, RAW_IMAGES_DIR, ROOT_DIR, TABLES_DIR, ensure_dirs, get_logger,
    hardware_info, load_json, save_json, set_seed,
)

log = get_logger("ocr_extraction")

OCR_CACHE_VERSION = 1
LOW_CONFIDENCE = 0.80

# ---------------------------------------------------------------------------
# Reading order
# ---------------------------------------------------------------------------
_MONTHS = r"jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec"
_DATE_RE = re.compile(rf"\b(19|20)\d{{2}}\b|\b(present|current|now)\b|\b({_MONTHS})[a-z]*\.?\b", re.I)
_LOCATION_RE = re.compile(r"^[A-Z][A-Za-z .'-]+,\s*[A-Z]{2}$|^remote$", re.I)


@dataclass
class Box:
    idx: int            # position in PaddleOCR's raw output
    text: str
    score: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def h(self) -> float:
        return max(self.y2 - self.y1, 1.0)

    @property
    def char_w(self) -> float:
        return (self.x2 - self.x1) / max(len(self.text), 1)


def _annotation_like(text: str) -> bool:
    t = text.strip()
    return bool(_DATE_RE.search(t)) or bool(_LOCATION_RE.match(t))


def _v_overlap(a: Box, b: Box) -> float:
    return max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1)) / min(a.h, b.h)


def _gaps(intervals: list[tuple[float, float]], min_gap: float) -> list[tuple[float, float]]:
    """Whitespace gaps (start, end) >= min_gap in the union of 1-D intervals."""
    intervals = sorted(intervals)
    gaps, cur_end = [], intervals[0][1]
    for s, e in intervals[1:]:
        if s - cur_end >= min_gap:
            gaps.append((cur_end, s))
        cur_end = max(cur_end, e)
    return gaps


class ReadingOrder:
    """Recursive XY-cut tuned for resumes (see module docstring)."""

    def __init__(self, page_width: float, min_col_gap_frac: float = 0.02,
                 annot_share: float = 0.5, annot_align: float = 0.5,
                 sparse_ratio: float = 0.5, sparse_align: float = 0.8):
        self.page_width = page_width
        self.min_col_gap_frac = min_col_gap_frac
        self.annot_share = annot_share
        self.annot_align = annot_align
        self.sparse_ratio = sparse_ratio
        self.sparse_align = sparse_align

    # -- public --------------------------------------------------------------
    def order(self, boxes: list[Box]) -> list[list[list[Box]]]:
        """Return blocks -> lines -> boxes in reading order."""
        if not boxes:
            return []
        self._med_h = float(np.median([b.h for b in boxes]))
        self._min_vgap = max(self.min_col_gap_frac * self.page_width,
                             2.0 * float(np.median([b.char_w for b in boxes])))
        self._min_hgap = 0.3 * self._med_h
        blocks: list[list[Box]] = []
        self._cut(boxes, blocks)
        return [self._lines(b) for b in blocks]

    # -- recursion -------------------------------------------------------------
    def _cut(self, boxes: list[Box], out: list[list[Box]]) -> None:
        if len(boxes) <= 1:
            out.append(boxes)
            return
        parts = self._vertical_cut(boxes) or self._horizontal_cut(boxes)
        if parts is None:
            out.append(boxes)
            return
        for p in parts:
            self._cut(p, out)

    def _vertical_cut(self, boxes: list[Box]) -> list[list[Box]] | None:
        gaps = _gaps([(b.x1, b.x2) for b in boxes], self._min_vgap)
        for g0, g1 in sorted(gaps, key=lambda g: g[1] - g[0], reverse=True):
            x = (g0 + g1) / 2
            left = [b for b in boxes if b.x2 <= x]
            right = [b for b in boxes if b.x1 >= x]
            if left and right and self._is_real_column_split(left, right):
                return [left, right]
        return None

    def _horizontal_cut(self, boxes: list[Box]) -> list[list[Box]] | None:
        gaps = _gaps([(b.y1, b.y2) for b in boxes], self._min_hgap)
        if not gaps:
            return None
        g0, g1 = max(gaps, key=lambda g: g[1] - g[0])
        y = (g0 + g1) / 2
        return [[b for b in boxes if b.y2 <= y], [b for b in boxes if b.y1 >= y]]

    def _is_real_column_split(self, left: list[Box], right: list[Box]) -> bool:
        minor, major = (left, right) if len(left) < len(right) else (right, left)
        aligned = np.mean([any(_v_overlap(m, o) >= 0.5 for o in major) for m in minor])
        annot = np.mean([_annotation_like(m.text) for m in minor])
        if annot >= self.annot_share and aligned >= self.annot_align:
            return False        # title ... date / location rows
        if len(minor) / len(major) <= self.sparse_ratio and aligned >= self.sparse_align:
            return False        # sparse right-aligned annotations
        return True

    # -- leaves ------------------------------------------------------------------
    @staticmethod
    def _lines(boxes: list[Box]) -> list[list[Box]]:
        lines: list[list[Box]] = []
        for b in sorted(boxes, key=lambda b: (b.y1 + b.y2) / 2):
            if lines and any(_v_overlap(b, o) >= 0.5 for o in lines[-1]):
                lines[-1].append(b)
            else:
                lines.append([b])
        return [sorted(line, key=lambda b: b.x1) for line in lines]


def reconstruct(texts: Sequence[str], scores: Sequence[float], boxes: Any,
                page_width: float) -> dict[str, Any]:
    """Apply ReadingOrder to raw PaddleOCR output; returns ordered items, lines and text."""
    items = [Box(i, str(t), float(s), *map(float, bx))
             for i, (t, s, bx) in enumerate(zip(texts, scores, np.asarray(boxes).reshape(-1, 4)))
             if str(t).strip()]
    blocks = ReadingOrder(page_width).order(items)
    ordered, lines = [], []
    for bi, block in enumerate(blocks):
        for line in block:
            li = len(lines)
            lines.append(" ".join(b.text.strip() for b in line))
            for b in line:
                ordered.append({"text": b.text, "score": round(b.score, 4),
                                "box": [int(b.x1), int(b.y1), int(b.x2), int(b.y2)],
                                "order": len(ordered), "raw_index": b.idx, "block": bi, "line": li})
    return {"items": ordered, "lines": lines, "text": "\n".join(lines)}


# ---------------------------------------------------------------------------
# PaddleOCR
# ---------------------------------------------------------------------------
def build_ocr(device: str = "gpu:0"):
    """Create the PaddleOCR pipeline (document unwarping/orientation off: CVs are clean renders)."""
    try:
        import torch  # noqa: F401  (Windows: torch must load its CUDA DLLs before paddle)
    except ImportError:
        pass
    from paddleocr import PaddleOCR

    return PaddleOCR(lang="en", device=device,
                     use_doc_orientation_classify=False,
                     use_doc_unwarping=False,
                     use_textline_orientation=False)


def ocr_image(ocr, image: str | Path | np.ndarray) -> dict[str, Any]:
    """Run PaddleOCR on a path or an RGB array; returns raw output + reading-order text."""
    if isinstance(image, np.ndarray):
        src = image[:, :, ::-1].copy() if image.ndim == 3 else image   # PaddleOCR expects BGR
        height, width = image.shape[:2]
    else:
        src = str(image)
        from PIL import Image

        with Image.open(src) as im:
            width, height = im.size
    t0 = time.perf_counter()
    res = ocr.predict(src)[0]
    runtime = time.perf_counter() - t0

    texts = list(res["rec_texts"])
    scores = [float(s) for s in res["rec_scores"]]
    boxes = np.asarray(res["rec_boxes"]).reshape(-1, 4).astype(int).tolist()
    out = {
        "width": int(width), "height": int(height),
        "runtime_s": round(runtime, 4),
        "n_boxes": len(texts),
        "mean_confidence": round(float(np.mean(scores)), 4) if scores else None,
        "low_conf_share": round(float(np.mean(np.array(scores) < LOW_CONFIDENCE)), 4) if scores else None,
        "raw": {"rec_texts": texts, "rec_scores": scores, "rec_boxes": boxes},
    }
    out.update(reconstruct(texts, scores, boxes, page_width=width))
    return out


# ---------------------------------------------------------------------------
# GPU monitor
# ---------------------------------------------------------------------------
class GpuMonitor:
    """Samples nvidia-smi in a background thread (utilisation %, memory MiB)."""

    QUERY = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits", "-i", "0"]

    def __init__(self, interval_s: float = 1.0):
        self.interval_s = interval_s
        self.samples: list[tuple[float, float, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                line = subprocess.run(self.QUERY, capture_output=True, text=True, timeout=5).stdout.strip()
                self.samples.append(tuple(float(v) for v in line.split(",")))
            except (OSError, ValueError, subprocess.SubprocessError):
                return          # no NVIDIA driver / nvidia-smi: monitoring silently disabled
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "GpuMonitor":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=10)

    def summary(self) -> dict[str, Any]:
        if not self.samples:
            return {"available": False}
        a = np.array(self.samples)
        return {"available": True, "n_samples": len(a),
                "util_mean_pct": round(float(a[:, 0].mean()), 1), "util_max_pct": float(a[:, 0].max()),
                "mem_used_mean_mib": round(float(a[:, 1].mean()), 0), "mem_used_max_mib": float(a[:, 1].max()),
                "mem_total_mib": float(a[0, 2])}


# ---------------------------------------------------------------------------
# Batch run
# ---------------------------------------------------------------------------
def cache_path(cv_id: str) -> Path:
    return OCR_DIR / f"{cv_id}.json"


def is_cached(cv_id: str) -> bool:
    p = cache_path(cv_id)
    if not p.exists():
        return False
    try:
        return load_json(p).get("cache_version") == OCR_CACHE_VERSION
    except (ValueError, OSError):
        return False            # corrupt / truncated file -> recompute


def run(images: list[Path], device: str, force: bool) -> None:
    import paddleocr

    todo = [p for p in images if force or not is_cached(p.stem)]
    log.info("%d images, %d cached, %d to process on %s", len(images), len(images) - len(todo), len(todo), device)

    per_cv: list[dict] = []
    failures: list[tuple[str, str]] = []
    if todo:
        ocr = build_ocr(device)
        t_warm = time.perf_counter()
        ocr.predict(np.full((256, 256, 3), 255, dtype=np.uint8))       # exclude model warm-up from timings
        warmup_s = time.perf_counter() - t_warm

        with GpuMonitor() as gpu:
            t_all = time.perf_counter()
            for path in tqdm(todo, desc="PaddleOCR", unit="cv"):
                try:
                    result = ocr_image(ocr, path)
                except Exception as e:      # one bad image must not kill a 4,000-CV run
                    failures.append((path.stem, f"{type(e).__name__}: {e}"))
                    log.error("OCR failed for %s: %s", path.name, e)
                    continue
                record = {"id": path.stem, "cache_version": OCR_CACHE_VERSION,
                          "image_path": path.relative_to(ROOT_DIR).as_posix() if path.is_relative_to(ROOT_DIR) else str(path),
                          "engine": {"name": "PaddleOCR", "version": paddleocr.__version__, "lang": "en", "device": device},
                          **result}
                save_json(record, cache_path(path.stem))
                per_cv.append({"id": path.stem, "runtime_s": result["runtime_s"], "n_boxes": result["n_boxes"],
                               "mean_confidence": result["mean_confidence"]})
            wall_s = time.perf_counter() - t_all
        run_info = {"n_processed": len(per_cv), "n_failed": len(failures), "failures": failures[:50],
                    "wall_time_s": round(wall_s, 1), "warmup_s": round(warmup_s, 2),
                    "throughput_cv_per_min": round(60 * len(per_cv) / wall_s, 1) if wall_s else None,
                    "gpu": gpu.summary()}
    else:
        run_info = {"n_processed": 0, "n_failed": 0}

    write_report(images, run_info, device)
    if failures:
        raise SystemExit(f"{len(failures)} images failed; re-run to retry them (successful ones are cached).")


def write_report(images: list[Path], run_info: dict, device: str) -> None:
    """Aggregate over ALL cached CVs (not only this run) so the report is always complete."""
    rows = []
    for p in images:
        if is_cached(p.stem):
            r = load_json(cache_path(p.stem))
            rows.append({"id": r["id"], "runtime_s": r["runtime_s"], "n_boxes": r["n_boxes"],
                         "mean_confidence": r["mean_confidence"], "low_conf_share": r["low_conf_share"],
                         "n_lines": len(r["lines"]), "n_chars": len(r["text"]),
                         "device": r["engine"]["device"]})
    if not rows:
        log.warning("No cached OCR results to report.")
        return
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "ocr_per_cv.csv", index=False)
    t = df["runtime_s"]
    report = {
        "n_cvs_cached": len(df),
        "runtime_per_cv_s": {"mean": round(t.mean(), 4), "median": round(t.median(), 4),
                             "p95": round(t.quantile(0.95), 4), "max": round(t.max(), 4)},
        "mean_confidence": round(float(df["mean_confidence"].mean()), 4),
        "low_conf_box_share": round(float(df["low_conf_share"].mean()), 4),
        "boxes_per_cv_mean": round(float(df["n_boxes"].mean()), 1),
        "chars_per_cv_mean": round(float(df["n_chars"].mean()), 1),
        "devices": df["device"].value_counts().to_dict(),
        "last_run": run_info,
        "hardware": hardware_info(),
    }
    save_json(report, METRICS_DIR / "ocr_runtime.json")
    log.info("OCR: %d CVs | %.3f s/CV (median %.3f) | mean conf %.3f | %.1f%% boxes < %.2f",
             len(df), t.mean(), t.median(), report["mean_confidence"],
             100 * report["low_conf_box_share"], LOW_CONFIDENCE)
    if run_info.get("gpu", {}).get("available"):
        g = run_info["gpu"]
        log.info("GPU: util mean %.0f%% (max %.0f%%) | mem max %.0f / %.0f MiB",
                 g["util_mean_pct"], g["util_max_pct"], g["mem_used_max_mib"], g["mem_total_mib"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images-dir", type=Path, default=RAW_IMAGES_DIR)
    parser.add_argument("--device", default="gpu:0", help="gpu:0 or cpu")
    parser.add_argument("--limit", type=int, default=None, help="process only the first N images (smoke test)")
    parser.add_argument("--force", action="store_true", help="ignore the cache and recompute")
    args = parser.parse_args()

    ensure_dirs()
    set_seed()
    images = sorted(p for p in args.images_dir.glob("*.png") if not p.name.endswith(".tmp.png"))
    if not images:
        raise SystemExit(f"No images in {args.images_dir}. Run `python -m src.download_data` first.")
    run(images[: args.limit] if args.limit else images, args.device, args.force)


if __name__ == "__main__":
    main()
