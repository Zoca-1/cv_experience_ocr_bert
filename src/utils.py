"""Shared utilities: paths, global constants, seeding, logging, I/O and hardware info.

Every other module imports its configuration from here so that paths, the
target cap and the experience bands are defined in exactly one place.

Windows note: in any process that uses both frameworks, import ``torch`` BEFORE
``paddle``. Paddle loads its own CUDA/cuDNN DLLs and a later ``import torch``
fails with ``WinError 127`` (shm.dll).
"""
from __future__ import annotations

import json
import logging
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_IMAGES_DIR = RAW_DIR / "images"
RAW_METADATA_PATH = RAW_DIR / "metadata.parquet"   # ids, targets, reference text
RAW_PII_PATH = RAW_DIR / "pii.parquet"              # names/state/age: PII scrubbing only, never features
RAW_INDEX_PATH = RAW_DIR / "shard_index.parquet"    # (id, has_photo) -> shard / row group
TARGET_CAP_PATH = RAW_DIR / "target_cap.json"       # global cap decision (see download_data.py)
OCR_DIR = DATA_DIR / "ocr"                          # one JSON per CV (cache)
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

MODELS_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
METRICS_DIR = OUTPUT_DIR / "metrics"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"

_ALL_DIRS = (
    RAW_IMAGES_DIR, OCR_DIR, PROCESSED_DIR, SPLITS_DIR, MODELS_DIR,
    FIGURES_DIR, TABLES_DIR, METRICS_DIR, PREDICTIONS_DIR,
)

# ---------------------------------------------------------------------------
# Project constants
# ---------------------------------------------------------------------------
SEED = 42

HF_DATASET = "sumin-yu/PopResume"
OCCUPATIONS: tuple[str, ...] = (
    "registered_nurses",
    "elementary_middle_school_teachers",
    "software_developers",
    "accountants_auditors",
    "construction_laborers",
)
SAMPLES_PER_OCCUPATION = 800

TARGET_COL = "exp_year"
TARGET_MIN = 0
TARGET_MAX = 20            # default cap; download_data.py may revise it once (see get_target_cap)
CAP_SHARE_THRESHOLD = 0.15
CAP_CANDIDATES = (20, 25, 30, 35, 40)

# Features that must never reach a model (sex/race only for the fairness check, Q4)
FORBIDDEN_FEATURES = ("age", "sex", "race", "name", "state", "region")
FAIRNESS_ONLY_COLS = ("sex", "race")

# Experience bands: (label, lower inclusive, upper inclusive). The last band is open-ended
# so the bands stay valid if the global cap is raised above 20.
EXPERIENCE_BANDS: tuple[tuple[str, int, int], ...] = (
    ("0-2", 0, 2),
    ("3-5", 3, 5),
    ("6-10", 6, 10),
    ("11-15", 11, 15),
    ("16+", 16, 10**6),
)
BAND_LABELS = tuple(b[0] for b in EXPERIENCE_BANDS)

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ensure_dirs(dirs: Iterable[Path] = _ALL_DIRS) -> None:
    """Create every project directory that does not exist yet."""
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = SEED) -> None:
    """Seed python, numpy, torch (if installed) and paddle (only if already imported:
    importing it costs ~10 s and is not needed outside the OCR step)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    if "paddle" in sys.modules:
        sys.modules["paddle"].seed(seed)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Console logger with a uniform format; idempotent across repeated calls."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def get_target_cap() -> int:
    """Global target cap: the decision persisted by download_data.py, else TARGET_MAX.

    Every module must use this (never a literal 20) so the cap is identical everywhere.
    """
    if TARGET_CAP_PATH.exists():
        return int(load_json(TARGET_CAP_PATH)["cap"])
    return TARGET_MAX


def clip_target(y: Any, cap: int | None = None) -> np.ndarray:
    """Clip experience years to [TARGET_MIN, cap] (cap defaults to the global decision)."""
    return np.clip(np.asarray(y, dtype=float), TARGET_MIN, get_target_cap() if cap is None else cap)


def to_band(years: float) -> str:
    """Map a (possibly fractional) number of years to its experience band label."""
    y = int(round(max(float(years), TARGET_MIN)))
    for label, lo, hi in EXPERIENCE_BANDS:
        if lo <= y <= hi:
            return label
    raise ValueError(f"Years value {years!r} does not fall in any band")


def to_bands(years: Iterable[float]) -> np.ndarray:
    return np.array([to_band(y) for y in years])


def save_json(obj: Any, path: Path) -> None:
    """Write JSON atomically (temp file + rename) so a crash never leaves a half-written cache."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)
    tmp.replace(path)


def load_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def hardware_info() -> dict[str, Any]:
    """Describe the hardware/software used, for the OCR and training timing reports."""
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "gpu": None,
    }
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
    except ImportError:
        pass
    try:
        import paddle

        info["paddle"] = paddle.__version__
        info["paddle_cuda"] = paddle.device.is_compiled_with_cuda()
    except ImportError:
        pass
    return info


def load_dotenv_if_present() -> None:
    """Load ROOT/.env (e.g. HF_TOKEN) into the environment without overriding existing vars."""
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT_DIR / ".env", override=False)
    except ImportError:
        pass


def get_device() -> str:
    """'cuda' if a GPU is visible to torch, else 'cpu'."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
