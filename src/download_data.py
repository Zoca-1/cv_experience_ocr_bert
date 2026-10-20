"""Stratified streaming sample of PopResume CV images (800 per occupation, no photo).

Strategy (avoids downloading the ~20 GB `images` config):
  1. Index   - for every image shard, stream only the tiny `id` and `has_photo` column
               chunks over HTTP range requests (hf:// filesystem), shards in parallel.
  2. Sample  - keep rows with has_photo == False, sort ids, and draw 800 without
               replacement with a seeded RNG per occupation (np.random.default_rng([SEED, k])).
  3. Fetch   - stream (in parallel) only the row groups that contain sampled ids. No-photo
               rows are stored contiguously, so photo row groups are never downloaded. Save each
               `image` to data/raw/images/<id>.png and collect the non-PII attributes.
  4. Target  - report how many sampled CVs sit at the 20-year cap; if the share exceeds
               15 %, pick the smallest larger cap that brings it under the threshold and
               persist the decision to data/raw/target_cap.json (read by utils.get_target_cap).

Outputs
  data/raw/images/<id>.png      CV images (model input goes through OCR only)
  data/raw/metadata.parquet     id, occupation, exp_year_raw, exp_year, text_ref, sex, race, edu_*
                                (text_ref comes from the `text` config: image shards have no text)
  data/raw/pii.parquet          names / state / region / age -> PII scrubbing only, never features
  data/raw/shard_index.parquet  (id, has_photo) -> shard, row group  (cache for re-runs)
  data/raw/target_cap.json      global cap decision
  output/tables/target_cap_analysis.csv

The dataset is gated: accept its terms on huggingface.co and provide a token via
`huggingface-cli login` or HF_TOKEN in the environment / .env file.

Usage:  python -m src.download_data [--n-per-occupation 800] [--cap 20] [--workers 8] [--force]
"""
from __future__ import annotations

import argparse
import io
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, TypeVar

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

from src.utils import (
    CAP_CANDIDATES, CAP_SHARE_THRESHOLD, HF_DATASET, OCCUPATIONS, RAW_IMAGES_DIR,
    RAW_INDEX_PATH, RAW_METADATA_PATH, RAW_PII_PATH, ROOT_DIR, SAMPLES_PER_OCCUPATION, SEED,
    TABLES_DIR, TARGET_CAP_PATH, TARGET_MAX, TARGET_MIN, ensure_dirs, get_logger,
    load_dotenv_if_present, save_json,
)

log = get_logger("download_data")

SHARD_GLOB = f"datasets/{HF_DATASET}/data/images/{{occupation}}-*.parquet"

TEXT_PATH = f"datasets/{HF_DATASET}/data/text/{{occupation}}.parquet"

# Columns kept as analysis metadata. sex/race are stored ONLY for the fairness check (Q4).
# The image shards carry no `text` column: the reference text is joined from the `text` config.
META_COLS = ("id", "job", "exp_year", "sex", "race", "edu_level", "edu_group")
# Columns used ONLY to scrub/mask PII in OCR text; never joined into a feature matrix.
PII_COLS = ("id", "first_name", "last_name", "state", "region", "age")

T = TypeVar("T")

READ_BLOCK_SIZE = 1 << 16       # 64 KiB
DEFAULT_WORKERS = 8


# ---------------------------------------------------------------------------
# Hugging Face access
# ---------------------------------------------------------------------------
def get_filesystem():
    from huggingface_hub import HfFileSystem

    load_dotenv_if_present()
    return HfFileSystem(token=os.environ.get("HF_TOKEN"))


def with_retries(fn: Callable[[], T], what: str, attempts: int = 5, base_delay: float = 2.0) -> T:
    """Retry transient network failures with exponential backoff; auth errors fail fast."""
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    for i in range(1, attempts + 1):
        try:
            return fn()
        except (GatedRepoError, RepositoryNotFoundError) as e:
            raise SystemExit(
                f"Cannot access {HF_DATASET}: {e.__class__.__name__}.\n"
                "The dataset is gated. Accept its terms at "
                f"https://huggingface.co/datasets/{HF_DATASET} and then either run "
                "`huggingface-cli login` or put HF_TOKEN=<token> in the .env file."
            ) from e
        except (OSError, ConnectionError, TimeoutError) as e:
            if i == attempts:
                raise
            delay = base_delay * 2 ** (i - 1)
            log.warning("%s failed (%s: %s). Retry %d/%d in %.0fs", what, type(e).__name__, e, i, attempts - 1, delay)
            time.sleep(delay)
    raise AssertionError("unreachable")


def open_parquet(fs, path: str) -> pq.ParquetFile:
    """Open a remote parquet file with small read blocks.

    Reads are latency-bound (one HTTP range request per column chunk); fsspec's default
    multi-MB read-ahead would download far more than the few KB an index read needs.
    """
    return with_retries(
        lambda: pq.ParquetFile(fs.open(path, "rb", block_size=READ_BLOCK_SIZE, cache_type="bytes")),
        f"open {path}",
    )


def list_shards(fs, occupation: str) -> list[str]:
    shards = sorted(with_retries(lambda: fs.glob(SHARD_GLOB.format(occupation=occupation)), f"list {occupation}"))
    if not shards:
        raise SystemExit(f"No image shards found for occupation '{occupation}'")
    return shards


def to_bool(series: pd.Series) -> pd.Series:
    """Robust bool cast: a plain astype(bool) would turn the string 'False' into True."""
    if series.dtype == bool:
        return series
    mapping = {"true": True, "1": True, "false": False, "0": False}
    out = series.astype(str).str.strip().str.lower().map(mapping)
    if out.isna().any():
        raise ValueError(f"Unexpected has_photo values: {series[out.isna()].unique()[:5]}")
    return out.astype(bool)


def run_parallel(fn: Callable[..., T], tasks: list[tuple], workers: int, desc: str, unit: str) -> list[T]:
    """Run fn(*task) on a thread pool (I/O-bound HTTP reads) with a progress bar."""
    results: list[T] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fn, *t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=desc, unit=unit):
            results.append(fut.result())
    return results


# ---------------------------------------------------------------------------
# 1. Index
# ---------------------------------------------------------------------------
def index_shard(fs, occupation: str, shard: str) -> pd.DataFrame:
    """(id, has_photo, shard, row_group) for every image row of one shard."""
    pf = open_parquet(fs, shard)
    tbl = with_retries(lambda: pf.read(columns=["id", "has_photo"]), f"index {shard}").to_pandas()
    rg_sizes = [pf.metadata.row_group(i).num_rows for i in range(pf.metadata.num_row_groups)]
    tbl["row_group"] = np.repeat(np.arange(len(rg_sizes)), rg_sizes)
    tbl["shard"] = shard
    tbl["occupation"] = occupation
    return tbl


def load_or_build_index(fs, force: bool, workers: int) -> pd.DataFrame:
    if RAW_INDEX_PATH.exists() and not force:
        idx = pd.read_parquet(RAW_INDEX_PATH)
        if set(OCCUPATIONS) <= set(idx["occupation"].unique()):
            log.info("Using cached shard index (%d rows) at %s", len(idx), RAW_INDEX_PATH)
            return idx
    tasks = [(fs, occ, shard) for occ in OCCUPATIONS for shard in list_shards(fs, occ)]
    idx = pd.concat(run_parallel(index_shard, tasks, workers, "index shards", "shard"), ignore_index=True)
    idx["id"] = idx["id"].astype(str)
    idx["has_photo"] = to_bool(idx["has_photo"])
    idx = idx.sort_values(["occupation", "shard", "row_group", "id"]).reset_index(drop=True)
    idx.to_parquet(RAW_INDEX_PATH, index=False)
    log.info("Indexed %d image rows in %d shards", len(idx), len(tasks))
    return idx


# ---------------------------------------------------------------------------
# 2. Sample
# ---------------------------------------------------------------------------
def sample_ids(index: pd.DataFrame, n_per_occupation: int, seed: int = SEED) -> pd.DataFrame:
    """Seeded sample of `n_per_occupation` no-photo rows per occupation (sampling without replacement)."""
    picked = []
    for k, occ in enumerate(OCCUPATIONS):
        pool = (
            index[(index["occupation"] == occ) & (~index["has_photo"])]
            .drop_duplicates("id")
            .sort_values("id", kind="stable")       # order-independent of shard listing
            .reset_index(drop=True)
        )
        if len(pool) < n_per_occupation:
            raise SystemExit(f"{occ}: only {len(pool)} no-photo CVs, need {n_per_occupation}")
        rng = np.random.default_rng([seed, k])
        rows = rng.choice(len(pool), size=n_per_occupation, replace=False)
        picked.append(pool.iloc[np.sort(rows)])
        log.info("%-36s pool=%6d (has_photo==False)  sampled=%d", occ, len(pool), n_per_occupation)
    return pd.concat(picked, ignore_index=True)


# ---------------------------------------------------------------------------
# 3. Fetch
# ---------------------------------------------------------------------------
def save_image(image_cell: Any, path) -> tuple[int, int]:
    """Write an HF `Image` struct ({'bytes', 'path'}) to PNG; returns (width, height)."""
    data = image_cell.get("bytes") if isinstance(image_cell, dict) else image_cell
    if not data:
        raise ValueError("empty image bytes")
    img = Image.open(io.BytesIO(data))
    size = img.size
    tmp = path.with_suffix(".tmp.png")
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        tmp.write_bytes(data)               # already PNG: keep the original bytes losslessly
    else:
        img.convert("RGB").save(tmp, format="PNG")
    tmp.replace(path)
    return size


def fetch_row_group(fs, shard: str, rg: int, ids: set[str], occupation_of: dict[str, str],
                    force: bool) -> tuple[list[dict], list[dict], list[tuple[str, str]]]:
    """Read one parquet row group, save its sampled no-photo images; returns (meta, pii, failures)."""
    pf = open_parquet(fs, shard)
    available = set(pf.schema_arrow.names)
    missing = set(META_COLS) - available
    if missing:
        raise SystemExit(f"{shard} is missing expected columns: {sorted(missing)}")
    attr_cols = [c for c in dict.fromkeys(META_COLS + PII_COLS) if c in available]
    need_images = force or any(not (RAW_IMAGES_DIR / f"{i}.png").exists() for i in ids)
    cols = attr_cols + ["has_photo"] + (["image"] if need_images else [])

    df = with_retries(lambda: pf.read_row_group(rg, columns=cols), f"{shard} rg={rg}").to_pandas()
    df["id"] = df["id"].astype(str)
    df = df[df["id"].isin(ids) & ~to_bool(df["has_photo"])]

    meta_rows, pii_rows, failures = [], [], []
    for rec in df.to_dict("records"):
        cv_id = rec["id"]
        img_path = RAW_IMAGES_DIR / f"{cv_id}.png"
        try:
            if need_images and (force or not img_path.exists()):
                w, h = save_image(rec["image"], img_path)
            else:
                with Image.open(img_path) as im:
                    w, h = im.size
        except Exception as e:  # corrupt bytes, disk errors: report and continue
            failures.append((cv_id, f"{type(e).__name__}: {e}"))
            continue
        meta_rows.append({
            "id": cv_id,
            "occupation": occupation_of[cv_id],   # canonical key, not the display `job`
            "exp_year_raw": float(rec["exp_year"]),
            "sex": rec["sex"],
            "race": rec["race"],
            "edu_level": rec["edu_level"],
            "edu_group": rec["edu_group"],
            "job": rec["job"],
            "image_path": img_path.relative_to(ROOT_DIR).as_posix(),
            "img_width": w,
            "img_height": h,
            "shard": shard.rsplit("/", 1)[-1],
        })
        pii_rows.append({c: rec.get(c) for c in PII_COLS})
    return meta_rows, pii_rows, failures


def fetch_sample(fs, sample: pd.DataFrame, force: bool, workers: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stream (in parallel) only the row groups holding sampled ids; return (metadata, pii)."""
    occupation_of = dict(zip(sample["id"], sample["occupation"]))
    groups = sample.groupby(["shard", "row_group"])["id"].apply(set)
    tasks = [(fs, shard, int(rg), ids, occupation_of, force) for (shard, rg), ids in groups.items()]
    log.info("Fetching %d CVs from %d row groups with %d workers", len(sample), len(tasks), workers)

    meta_rows, pii_rows, failures = [], [], []
    for m, p, f in run_parallel(fetch_row_group, tasks, workers, "fetch row groups", "rg"):
        meta_rows += m
        pii_rows += p
        failures += f

    for cv_id, err in failures:
        log.error("Failed to save %s: %s", cv_id, err)
    wanted = set(sample["id"]) - {r["id"] for r in meta_rows}
    if wanted:
        log.error("%d sampled ids were not saved: %s", len(wanted), sorted(wanted)[:10])
    if failures or wanted:
        raise SystemExit("Download incomplete; re-run to retry (finished images are cached).")
    return pd.DataFrame(meta_rows), pd.DataFrame(pii_rows)


def attach_reference_text(fs, meta: pd.DataFrame) -> pd.DataFrame:
    """Join `text_ref` (ground truth for CER/WER only) from the small `text` config by id."""
    frames = []
    for occ in OCCUPATIONS:
        def _read(occ=occ):
            with fs.open(TEXT_PATH.format(occupation=occ), "rb") as f:
                return pd.read_parquet(f, columns=["id", "exp_year", "text"])
        frames.append(with_retries(_read, f"text config {occ}"))
    ref = pd.concat(frames, ignore_index=True)
    ref["id"] = ref["id"].astype(str)
    if ref["id"].duplicated().any():
        raise SystemExit("Reference text ids are not unique across occupations")

    out = meta.merge(ref.rename(columns={"text": "text_ref", "exp_year": "_exp_text"}), on="id", how="left")
    missing = out["text_ref"].isna()
    if missing.any():
        raise SystemExit(f"{missing.sum()} sampled ids have no reference text, e.g. {out.loc[missing, 'id'].head().tolist()}")
    mismatch = ~np.isclose(out["exp_year_raw"], out["_exp_text"])
    if mismatch.any():
        raise SystemExit(f"exp_year differs between image and text configs for {mismatch.sum()} ids")
    log.info("Reference text joined for %d CVs (exp_year consistent across configs)", len(out))
    return out.drop(columns="_exp_text")


# ---------------------------------------------------------------------------
# 4. Target cap analysis
# ---------------------------------------------------------------------------
def analyse_cap(meta: pd.DataFrame, cap_override: int | None) -> dict:
    y = meta["exp_year_raw"]
    rows = []
    for cap in CAP_CANDIDATES:
        at_cap = y >= cap
        row = {"cap": cap, "n_at_cap": int(at_cap.sum()), "share_at_cap": float(at_cap.mean())}
        row.update({f"share_{occ}": float(at_cap[meta["occupation"] == occ].mean()) for occ in OCCUPATIONS})
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(TABLES_DIR / "target_cap_analysis.csv", index=False)

    base = table.set_index("cap").loc[TARGET_MAX]
    share20 = float(base["share_at_cap"])
    log.info("CVs with exp_year >= %d: %d / %d (%.1f%%), threshold %.0f%%",
             TARGET_MAX, base["n_at_cap"], len(meta), 100 * share20, 100 * CAP_SHARE_THRESHOLD)
    for occ in OCCUPATIONS:
        log.info("   %-36s %.1f%% at >= %d", occ, 100 * base[f"share_{occ}"], TARGET_MAX)

    if cap_override is not None:
        cap = cap_override
        why = f"Cap fixed manually to {cap} via --cap."
    elif share20 <= CAP_SHARE_THRESHOLD:
        cap = TARGET_MAX
        why = (f"{100 * share20:.1f}% of the sample is at >= {TARGET_MAX} years (<= {100 * CAP_SHARE_THRESHOLD:.0f}%): "
               f"clipping to [0, {TARGET_MAX}] loses little information, so the default cap is kept.")
    else:
        ok = table[(table["cap"] > TARGET_MAX) & (table["share_at_cap"] <= CAP_SHARE_THRESHOLD)]
        cap = int(ok["cap"].min()) if len(ok) else int(table["cap"].max())
        new_share = float(table.set_index("cap").loc[cap, "share_at_cap"])
        why = (f"{100 * share20:.1f}% of the sample is at >= {TARGET_MAX} years (> {100 * CAP_SHARE_THRESHOLD:.0f}%). "
               f"A pile-up that large at the cap makes the top band a catch-all and biases a regressor towards it, "
               f"so the cap is raised to {cap} ({100 * new_share:.1f}% at cap), the smallest candidate under the threshold. "
               f"It is applied globally (targets, metrics, bands) via utils.get_target_cap().")
    decision = {
        "cap": int(cap),
        "default_cap": TARGET_MAX,
        "share_at_default_cap": share20,
        "threshold": CAP_SHARE_THRESHOLD,
        "adjusted": int(cap) != TARGET_MAX,
        "justification": why,
        "n_cvs": int(len(meta)),
    }
    save_json(decision, TARGET_CAP_PATH)
    log.info("Target cap decision: %d. %s", cap, why)
    return decision


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-per-occupation", type=int, default=SAMPLES_PER_OCCUPATION)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--cap", type=int, default=None, help="override the automatic cap decision")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="parallel HTTP readers")
    parser.add_argument("--force", action="store_true", help="rebuild the index and re-download images")
    args = parser.parse_args()

    ensure_dirs()
    fs = get_filesystem()
    t0 = time.perf_counter()

    index = load_or_build_index(fs, args.force, args.workers)
    sample = sample_ids(index, args.n_per_occupation, args.seed)
    meta, pii = fetch_sample(fs, sample, args.force, args.workers)
    meta = attach_reference_text(fs, meta)

    decision = analyse_cap(meta, args.cap)
    meta["exp_year"] = np.clip(meta["exp_year_raw"], TARGET_MIN, decision["cap"])
    meta = meta.sort_values(["occupation", "id"]).reset_index(drop=True)
    meta.to_parquet(RAW_METADATA_PATH, index=False)
    pii.sort_values("id").to_parquet(RAW_PII_PATH, index=False)

    log.info("Saved %d CVs -> %s  (%.1f min)", len(meta), RAW_METADATA_PATH, (time.perf_counter() - t0) / 60)
    log.info("Per occupation:\n%s", meta.groupby("occupation")["exp_year"].describe().round(2).to_string())


if __name__ == "__main__":
    main()
