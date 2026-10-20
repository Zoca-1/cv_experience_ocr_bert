"""OCR quality vs the PopResume reference text: CER / WER, date and year accuracy.

Both sides go through the same `canonicalize` used for model input (one dash glyph, one
bullet glyph): the renderer draws "•" and "–" where the reference text has "-", which is
a rendering convention, not an OCR error. Unnormalized CER is reported too, so the
effect of normalization is visible.

Metrics
  * CER / WER per CV and corpus-level (micro), whitespace collapsed (layout-insensitive).
  * Date accuracy
      - employment ranges ("YYYY.MM-YYYY.MM"): range-level recall, CVs with all ranges exact,
        digit accuracy (1 - CER over the concatenated date digits),
      - years: recall of reference years in the matching OCR section, split into
        employment years (WORK HISTORY) and graduation years (EDUCATION),
      - target impact: |sum of job spans from OCR dates - from reference dates| in years.
  * Error attribution for every reference range the OCR does not reproduce:
      - image_truncation: the OCR line ends in the range prefix "YYYY.MM-YYYY." AND the
        image strip to the right of that PaddleOCR box is blank (no ink). The month is
        absent from the rendered image, so no OCR engine could read it.
      - ocr_error: anything else (misread digit, dropped range, ...).

Outputs
  output/metrics/ocr_quality.json
  output/tables/ocr_quality_per_cv.csv
  output/tables/ocr_quality_by_occupation.csv
  output/tables/ocr_quality_by_category.csv
  output/tables/ocr_cer_distribution.csv
  output/tables/ocr_date_errors.csv

Usage:  python -m src.ocr_quality
"""
from __future__ import annotations

import argparse
import re
from collections import Counter

import jiwer
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.preprocessing import (
    DATE_RANGE_RE, TRUNCATED_RANGE_RE, YEAR_RE, canonicalize, date_ranges, span_years, split_sections,
)
from src.utils import METRICS_DIR, OCR_DIR, RAW_METADATA_PATH, ROOT_DIR, TABLES_DIR, ensure_dirs, get_logger, load_json, save_json

log = get_logger("ocr_quality")

BLANK_STRIP_PX = 60          # width of the strip checked for ink right of a truncated box
INK_THRESHOLD = 160          # grayscale value below which a pixel counts as ink


def flat(text: str) -> str:
    """Canonical characters, all whitespace collapsed to single spaces."""
    return re.sub(r"\s+", " ", canonicalize(text)).strip()


def year_recall(ref_lines: list[str], ocr_lines: list[str]) -> tuple[int, int]:
    """(matched, total) reference years found in the OCR lines of the same section (multiset)."""
    ref = Counter(YEAR_RE.findall("\n".join(ref_lines)))
    hyp = Counter(YEAR_RE.findall("\n".join(ocr_lines)))
    return sum((ref & hyp).values()), sum(ref.values())


def date_digits(ranges: list[tuple[int, int, int, int]]) -> str:
    return " ".join(f"{a:04d}{b:02d}{c:04d}{d:02d}" for a, b, c, d in ranges)


def strip_is_blank(image: Image.Image, box: list[int]) -> bool:
    """True when the area right of `box` (same text line) contains no ink."""
    x1, y1, x2, y2 = box
    w, _ = image.size
    if x2 + 3 >= w:
        return False                       # box touches the page edge: cannot prove absence
    strip = image.crop((x2 + 3, y1, min(w, x2 + 3 + BLANK_STRIP_PX), y2)).convert("L")
    return int(np.asarray(strip).min()) >= INK_THRESHOLD


def attribute_missing_ranges(cv_id: str, ref_ranges, ocr: dict, image_path) -> list[dict]:
    """Classify each reference range absent from the OCR text."""
    hyp = Counter(date_ranges(canonicalize(ocr["text"])))
    missing = list((Counter(ref_ranges) - hyp).elements())
    if not missing:
        return []
    items_by_line: dict[int, list[dict]] = {}
    for it in ocr["items"]:
        items_by_line.setdefault(it["line"], []).append(it)
    truncated: dict[tuple[int, int, int], list[int]] = {}
    for li, line in enumerate(ocr["lines"]):
        for m in TRUNCATED_RANGE_RE.finditer(canonicalize(line)):
            last_box = max(items_by_line.get(li, []), key=lambda it: it["box"][2])["box"]
            truncated.setdefault(tuple(map(int, m.groups())), []).append(last_box)

    image = None
    out = []
    for y1, m1, y2, m2 in missing:
        boxes = truncated.get((y1, m1, y2), [])
        category, blank = "ocr_error", None
        if boxes:
            image = image or Image.open(image_path)
            blank = strip_is_blank(image, boxes.pop(0))
            category = "image_truncation" if blank else "ocr_error"
        out.append({"id": cv_id, "ref_range": f"{y1}.{m1:02d}-{y2}.{m2:02d}", "category": category,
                    "blank_strip_verified": blank,
                    "ocr_line": next((ln for ln in ocr["lines"] if f"{y1}.{m1:02d}" in canonicalize(ln)), None)})
    if image is not None:
        image.close()
    return out


def evaluate(meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, date_errors = [], []
    for cv_id, rec in tqdm(meta.iterrows(), total=len(meta), desc="OCR quality", unit="cv"):
        ocr = load_json(OCR_DIR / f"{cv_id}.json")
        ref_raw, hyp_raw = rec["text_ref"], ocr["text"]
        ref_c, hyp_c = canonicalize(ref_raw), canonicalize(hyp_raw)
        ref_f, hyp_f = flat(ref_raw), flat(hyp_raw)

        ref_ranges, hyp_ranges = date_ranges(ref_c), date_ranges(hyp_c)
        ref_sec, hyp_sec = split_sections(ref_c.splitlines()), split_sections(hyp_c.splitlines())
        emp_hit, emp_tot = year_recall(ref_sec.get("work", []), hyp_sec.get("work", []))
        edu_hit, edu_tot = year_recall(ref_sec.get("education", []), hyp_sec.get("education", []))
        errs = attribute_missing_ranges(cv_id, ref_ranges, ocr, ROOT_DIR / rec["image_path"])
        date_errors += errs
        cats = {e["category"] for e in errs}

        rows.append({
            "id": cv_id, "occupation": rec["occupation"],
            "cer": jiwer.cer(ref_f, hyp_f), "wer": jiwer.wer(ref_f, hyp_f),
            "cer_unnormalized": jiwer.cer(re.sub(r"\s+", " ", ref_raw).strip(), re.sub(r"\s+", " ", hyp_raw).strip()),
            "n_ref_chars": len(ref_f), "n_ref_words": len(ref_f.split()),
            "n_ref_ranges": len(ref_ranges), "n_ocr_ranges": len(hyp_ranges),
            "ranges_matched": sum((Counter(ref_ranges) & Counter(hyp_ranges)).values()),
            "ranges_exact": Counter(ref_ranges) == Counter(hyp_ranges),
            "date_digit_cer": jiwer.cer(date_digits(ref_ranges), date_digits(hyp_ranges)) if ref_ranges else 0.0,
            "emp_years_hit": emp_hit, "emp_years_total": emp_tot,
            "edu_years_hit": edu_hit, "edu_years_total": edu_tot,
            "span_years_ref": span_years(ref_ranges), "span_years_ocr": span_years(hyp_ranges),
            "span_abs_err": abs(span_years(hyp_ranges) - span_years(ref_ranges)),
            "mean_confidence": ocr["mean_confidence"],
            "category": "ocr_error" if "ocr_error" in cats else "image_truncation" if cats else "exact",
            "ref_flat": ref_f, "hyp_flat": hyp_f,
        })
    return pd.DataFrame(rows), pd.DataFrame(date_errors, columns=["id", "ref_range", "category", "blank_strip_verified", "ocr_line"])


def aggregate(df: pd.DataFrame) -> dict:
    """Corpus-level metrics for any subset of rows."""
    if df.empty:
        return {"n_cvs": 0}
    return {
        "n_cvs": int(len(df)),
        "cer_micro": round(jiwer.cer(df["ref_flat"].tolist(), df["hyp_flat"].tolist()), 6),
        "wer_micro": round(jiwer.wer(df["ref_flat"].tolist(), df["hyp_flat"].tolist()), 6),
        "cer_mean": round(float(df["cer"].mean()), 6),
        "wer_mean": round(float(df["wer"].mean()), 6),
        "cer_unnormalized_mean": round(float(df["cer_unnormalized"].mean()), 6),
        "cvs_with_cer_0": round(float((df["cer"] == 0).mean()), 4),
        "range_recall": round(float(df["ranges_matched"].sum() / max(df["n_ref_ranges"].sum(), 1)), 6),
        "cvs_all_ranges_exact": round(float(df["ranges_exact"].mean()), 6),
        "date_digit_accuracy": round(float(1 - df["date_digit_cer"].mean()), 6),
        "employment_year_recall": round(float(df["emp_years_hit"].sum() / max(df["emp_years_total"].sum(), 1)), 6),
        "graduation_year_recall": round(float(df["edu_years_hit"].sum() / max(df["edu_years_total"].sum(), 1)), 6),
        "span_abs_err_mean_years": round(float(df["span_abs_err"].mean()), 4),
        "span_abs_err_max_years": round(float(df["span_abs_err"].max()), 4),
    }


def distribution(df: pd.DataFrame) -> pd.DataFrame:
    qs = [0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
    rows = [{"metric": m, **{f"p{int(q * 100)}": float(df[m].quantile(q)) for q in qs}, "mean": float(df[m].mean())}
            for m in ("cer", "wer", "cer_unnormalized", "span_abs_err")]
    edges = [0, 1e-9, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 1.0]
    hist = pd.cut(df["cer"], bins=edges, include_lowest=True, right=True).value_counts().sort_index()
    rows += [{"metric": f"cer_bin {iv}", "count": int(n), "share": round(n / len(df), 4)} for iv, n in hist.items()]
    return pd.DataFrame(rows)


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    ensure_dirs()
    meta = pd.read_parquet(RAW_METADATA_PATH).set_index("id")
    df, errs = evaluate(meta)

    by_cat = {c: aggregate(df[df["category"] == c]) for c in ("exact", "image_truncation", "ocr_error")}
    summary = {
        "normalization": "NFKC, dash variants -> '-', bullet glyphs -> '-', whitespace collapsed",
        "all": aggregate(df),
        "excluding_image_truncation": aggregate(df[df["category"] != "image_truncation"]),
        "by_category": by_cat,
        "date_errors": {
            "missing_ranges": int(len(errs)),
            "by_category": errs["category"].value_counts().to_dict(),
            "cvs_image_truncation": int((df["category"] == "image_truncation").sum()),
            "cvs_ocr_error": int((df["category"] == "ocr_error").sum()),
            "truncation_by_occupation": df[df["category"] == "image_truncation"]["occupation"].value_counts().to_dict(),
        },
        "confidence_vs_cer_spearman": round(float(df["mean_confidence"].corr(df["cer"], method="spearman")), 4),
    }
    save_json(summary, METRICS_DIR / "ocr_quality.json")

    per_cv = df.drop(columns=["ref_flat", "hyp_flat"])
    per_cv.to_csv(TABLES_DIR / "ocr_quality_per_cv.csv", index=False)
    pd.DataFrame([{"occupation": occ, **aggregate(g)} for occ, g in df.groupby("occupation")]
                 + [{"occupation": "ALL", **summary["all"]}]).to_csv(TABLES_DIR / "ocr_quality_by_occupation.csv", index=False)
    pd.DataFrame([{"category": c, **v} for c, v in by_cat.items()]).to_csv(TABLES_DIR / "ocr_quality_by_category.csv", index=False)
    distribution(df).to_csv(TABLES_DIR / "ocr_cer_distribution.csv", index=False)
    errs.to_csv(TABLES_DIR / "ocr_date_errors.csv", index=False)

    a, d = summary["all"], summary["date_errors"]
    log.info("CER %.4f%% (micro) / %.4f%% (mean) | WER %.4f%% | unnormalized CER %.3f%%",
             100 * a["cer_micro"], 100 * a["cer_mean"], 100 * a["wer_micro"], 100 * a["cer_unnormalized_mean"])
    log.info("Dates: range recall %.3f%% | CVs all ranges exact %.2f%% | digit acc %.4f%% | "
             "employment-year recall %.3f%% | graduation-year recall %.3f%%",
             100 * a["range_recall"], 100 * a["cvs_all_ranges_exact"], 100 * a["date_digit_accuracy"],
             100 * a["employment_year_recall"], 100 * a["graduation_year_recall"])
    log.info("Missing ranges: %d -> %s | CVs: %d image truncation, %d OCR error | truncation by occupation %s",
             d["missing_ranges"], d["by_category"], d["cvs_image_truncation"], d["cvs_ocr_error"], d["truncation_by_occupation"])
    log.info("Excluding image-truncated CVs: CER %.4f%%, CVs all ranges exact %.2f%%",
             100 * summary["excluding_image_truncation"]["cer_micro"], 100 * summary["excluding_image_truncation"]["cvs_all_ranges_exact"])
    log.info("Target impact (|span sum OCR - ref|): mean %.4f y, max %.2f y", a["span_abs_err_mean_years"], a["span_abs_err_max_years"])


if __name__ == "__main__":
    main()
