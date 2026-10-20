#!/usr/bin/env bash
# BERT runs behind README Q2 (reference-text training) and Q3 (loss, long-text strategy, seed).
# Each run writes output/metrics/<name>_val.json and output/predictions/<name>_{val,test}.csv;
# src/experiments.py collects them. ~20 min on an RTX 3060.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/Scripts/python}"
[ -x "$PY" ] || PY=".venv/bin/python"
export PYTHONPATH=. PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1
mkdir -p logs

run() { name=$1; shift; echo ">>> $name"; "$PY" -m src.train_bert --name "$name" "$@" > "logs/train_$name.log" 2>&1; grep -E "VAL" "logs/train_$name.log" | tail -1; }

run bert_distilbert_ref            --text-source ref
run bert_distilbert_huber          --loss huber
run bert_distilbert_len64_headtail --max-length 64 --truncation head_tail
run bert_distilbert_len64_head     --max-length 64 --truncation head
run bert_distilbert_seed43         --seed 43
run bert_distilbert_seed44         --seed 44
