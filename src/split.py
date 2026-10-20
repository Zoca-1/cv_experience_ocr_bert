"""Stratified 70/15/15 train/val/test split (seed 42), frozen with a hash manifest.

Strata = occupation x experience band x image-truncation flag (the 38 CVs whose last
end month is cut in the rendered image, see ocr_quality.py), so every split gets its
proportional share of each occupation, each band and the irreducible-error CVs.

Small strata: a fixed 70/15/15 split needs enough members per stratum for both
stages (70/30, then 30 -> 15/15). Each CV gets the FINEST key whose population is
>= MIN_STRATUM, falling back through
    occupation|band|truncated  ->  occupation|truncated  ->  occupation
so tiny cells (e.g. truncated teachers in the 16+ band) are merged instead of making
the split fail or silently dropping the truncation flag.

The test set is frozen: data/splits/manifest.json stores a SHA-256 of each split's
sorted ids, `load_split` verifies it on every read, and re-running refuses to overwrite
existing splits unless --force is given.

Outputs
  data/splits/{train,val,test}.csv   id, occupation, exp_year, band, truncated, stratum
  data/splits/manifest.json          seed, ratios, sizes, sha256 per split
  output/tables/split_summary.csv    distribution check per split
  output/metrics/split.json

Usage:  python -m src.split [--force]
"""
from __future__ import annotations

import argparse
import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import (
    BAND_LABELS, METRICS_DIR, PROCESSED_DIR, SEED, SPLIT_RATIOS, SPLITS_DIR, TABLES_DIR,
    ensure_dirs, get_logger, load_json, save_json, to_bands,
)

log = get_logger("split")

SPLITS = ("train", "val", "test")
MIN_STRATUM = 20
STRATA_LEVELS = (("occupation", "band", "truncated"), ("occupation", "truncated"), ("occupation",))
MANIFEST_PATH = SPLITS_DIR / "manifest.json"
OCR_QUALITY_PER_CV = TABLES_DIR / "ocr_quality_per_cv.csv"


def ids_hash(ids) -> str:
    return hashlib.sha256("\n".join(sorted(map(str, ids))).encode()).hexdigest()


def assign_strata(df: pd.DataFrame, min_size: int = MIN_STRATUM) -> pd.Series:
    """Finest key per row with >= min_size members among rows assigned to it."""
    keys = {lvl: df[list(lvl)].astype(str).agg("|".join, axis=1) for lvl in STRATA_LEVELS}
    level = pd.Series(0, index=df.index)
    while True:
        stratum = pd.Series([keys[STRATA_LEVELS[l]][i] for i, l in level.items()], index=df.index)
        sizes = stratum.map(stratum.value_counts())
        small = (sizes < min_size) & (level < len(STRATA_LEVELS) - 1)
        if not small.any():
            return stratum
        level[small] += 1


def make_splits(df: pd.DataFrame, seed: int = SEED) -> dict[str, pd.DataFrame]:
    r = SPLIT_RATIOS
    train, rest = train_test_split(df, train_size=r["train"], stratify=df["stratum"], random_state=seed)
    val, test = train_test_split(rest, test_size=r["test"] / (r["val"] + r["test"]),
                                 stratify=rest["stratum"], random_state=seed)
    return {"train": train, "val": val, "test": test}


def summarize(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, d in splits.items():
        row = {"split": name, "n": len(d), "share": len(d) / sum(map(len, splits.values())),
               "exp_mean": d["exp_year"].mean(), "exp_median": d["exp_year"].median(), "exp_std": d["exp_year"].std(),
               "n_truncated": int(d["truncated"].sum())}
        row.update({f"occ_{o}": (d["occupation"] == o).mean() for o in sorted(d["occupation"].unique())})
        row.update({f"band_{b}": (d["band"] == b).mean() for b in BAND_LABELS})
        rows.append(row)
    return pd.DataFrame(rows)


def load_split(name: str, with_text: bool = True, text_path=None) -> pd.DataFrame:
    """Read a frozen split, verify its hash, and (optionally) join the cleaned OCR text."""
    if name not in SPLITS:
        raise ValueError(f"unknown split {name!r}")
    if not MANIFEST_PATH.exists():
        raise SystemExit("No splits found. Run `python -m src.split` first.")
    d = pd.read_csv(SPLITS_DIR / f"{name}.csv", dtype={"id": str})
    expected = load_json(MANIFEST_PATH)["splits"][name]["sha256"]
    if ids_hash(d["id"]) != expected:
        raise SystemExit(f"Split '{name}' does not match its manifest hash: the frozen split was modified.")
    if with_text:
        texts = pd.read_parquet(text_path or PROCESSED_DIR / "cvs.parquet", columns=["id", "text"])
        d = d.merge(texts, on="id", how="left", validate="one_to_one")
        if d["text"].isna().any():
            raise SystemExit(f"{d['text'].isna().sum()} ids of split '{name}' have no processed text")
    return d


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--force", action="store_true", help="overwrite the frozen splits")
    args = parser.parse_args()
    ensure_dirs()

    if MANIFEST_PATH.exists() and not args.force:
        log.info("Splits already frozen (%s). Verifying only; use --force to regenerate.", MANIFEST_PATH)
        for s in SPLITS:
            log.info("  %-5s %4d ids, hash OK", s, len(load_split(s, with_text=False)))
        return

    df = pd.read_parquet(PROCESSED_DIR / "cvs.parquet", columns=["id", "occupation", "exp_year"])
    if not OCR_QUALITY_PER_CV.exists():
        raise SystemExit("Missing ocr_quality_per_cv.csv. Run `python -m src.ocr_quality` first.")
    q = pd.read_csv(OCR_QUALITY_PER_CV, dtype={"id": str}, usecols=["id", "category"])
    df = df.merge(q, on="id", how="left", validate="one_to_one")
    if df["category"].isna().any():
        raise SystemExit("Some CVs have no OCR quality record")
    df["truncated"] = (df["category"] == "image_truncation").astype(int)
    df["band"] = to_bands(df["exp_year"])
    df = df.drop(columns="category").sort_values("id").reset_index(drop=True)   # input order-independent
    df["stratum"] = assign_strata(df)

    splits = make_splits(df, args.seed)
    all_ids = [i for d in splits.values() for i in d["id"]]
    assert len(all_ids) == len(set(all_ids)) == len(df), "splits overlap or lose rows"

    cols = ["id", "occupation", "exp_year", "band", "truncated", "stratum"]
    manifest = {"seed": args.seed, "ratios": SPLIT_RATIOS, "min_stratum": MIN_STRATUM,
                "strata_levels": ["|".join(l) for l in STRATA_LEVELS], "n_strata": int(df["stratum"].nunique()),
                "splits": {}}
    for name, d in splits.items():
        d = d.sort_values("id")[cols]
        d.to_csv(SPLITS_DIR / f"{name}.csv", index=False)
        manifest["splits"][name] = {"n": len(d), "sha256": ids_hash(d["id"])}
    save_json(manifest, MANIFEST_PATH)

    summary = summarize(splits)
    summary.to_csv(TABLES_DIR / "split_summary.csv", index=False)
    level_of = df["stratum"].str.count(r"\|").map({2: "occupation|band|truncated", 1: "occupation|truncated", 0: "occupation"})
    report = {**manifest, "rows_per_strata_level": level_of.value_counts().to_dict(),
              "max_abs_share_diff_vs_train": {
                  c: round(float(np.abs(summary[c] - summary.loc[0, c]).max()), 4)
                  for c in summary.columns if c.startswith(("occ_", "band_"))}}
    save_json(report, METRICS_DIR / "split.json")

    log.info("Strata: %d (%s)", manifest["n_strata"], report["rows_per_strata_level"])
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        log.info("Split summary:\n%s", summary.round(3).T.to_string())


if __name__ == "__main__":
    main()
