#!/usr/bin/env bash
# End-to-end reproduction of every result in the README (~1.5 h on an RTX 3060).
# Every step is cached / resumable; re-running skips finished work.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/Scripts/python}"
[ -x "$PY" ] || PY=".venv/bin/python"
export PYTHONPATH=. PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1
mkdir -p logs

step() { echo ">>> $*"; "$PY" -m "$@"; }

step src.download_data --workers 12        # ~15 min, needs HF_TOKEN (gated dataset)
step src.ocr_extraction                    # ~25 min GPU
step src.ocr_quality
step src.preprocessing
step src.split                             # frozen: verifies hashes on re-runs
step src.baseline_median
step src.baseline_tfidf
step src.baseline_embeddings
step src.baseline_rules
step src.train_bert                                        # DistilBERT, ~4 min
step src.train_bert --model microsoft/deberta-v3-small     # ~7 min
step src.evaluation
bash scripts/run_sensitivity.sh                            # ~25 min (Q2/Q3 BERT variants)
step src.experiments
step src.visualization
mkdir -p docs/figures && cp output/figures/*.png docs/figures/
echo "Done. Launch the dashboard with: streamlit run app.py"
