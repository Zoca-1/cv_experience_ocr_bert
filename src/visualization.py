"""Static figures for the report (output/figures/*.png), built from the evaluation tables.

Design rules (shared with the Streamlit app): one recessive light surface; single-hue
marks wherever the chart is not comparing series (small multiples instead of a rainbow
of six models); a sequential blue ramp for magnitudes (heatmaps); categorical hues in a
fixed order (never cycled) only for the noise curve, where every line is also labelled
directly so identity never relies on colour alone; one y-axis per chart.

Usage:  python -m src.visualization
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from src.utils import (
    BAND_LABELS, FIGURES_DIR, METRICS_DIR, PREDICTIONS_DIR, RAW_METADATA_PATH, TABLES_DIR, ensure_dirs,
    get_logger, get_target_cap, load_json, to_bands,
)

log = get_logger("visualization")

# --- palette (reference instance of the dataviz method) ----------------------
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e6e5e1"
REF_LINE = "#8a8984"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
BLUE = SERIES[0]
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#f0efec", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])

SHORT = {
    "baseline_median": "Median",
    "baseline_tfidf": "TF-IDF + Ridge",
    "baseline_embeddings": "Embeddings + Ridge",
    "baseline_rules": "Rule parser",
    "bert_distilbert_base_uncased": "DistilBERT",
    "bert_deberta_v3_small": "DeBERTa-v3-small",
}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT, "axes.titlesize": 11,
    "axes.titleweight": "semibold", "axes.titlelocation": "left", "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "font.size": 10, "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


def _save(fig, name: str) -> None:
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("saved %s", path.relative_to(FIGURES_DIR.parents[1]))


def _models(df: pd.DataFrame) -> list[str]:
    return [m for m in SHORT if f"pred_{m}" in df.columns]


# ---------------------------------------------------------------------------
def pred_vs_true(test: pd.DataFrame, comp: pd.DataFrame) -> None:
    models, cap = _models(test), get_target_cap()
    ncol = 3
    nrow = int(np.ceil(len(models) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 4 * nrow), sharex=True, sharey=True)
    for ax, m in zip(axes.flat, models):
        pred = np.clip(test[f"pred_{m}"], 0, cap)
        ax.plot([0, cap], [0, cap], color=REF_LINE, lw=1.2, ls="--", zorder=1, label="perfect (45°)")
        ax.scatter(test["exp_year"], pred, s=9, color=BLUE, alpha=0.45, linewidths=0, zorder=2)
        mae = comp.set_index("model").loc[m, "mae"]
        ax.set_title(f"{SHORT[m]}   MAE {mae:.2f} y")
        ax.set_xlim(0, cap)
        ax.set_ylim(0, cap)
        ax.set_aspect("equal")
    for ax in axes.flat[len(models):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("True experience (years)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Predicted (years)")
    fig.suptitle("Predicted vs true experience on the test set (n = %d); dashed line = perfect prediction" % len(test),
                 x=0.01, ha="left", color=TEXT, fontsize=12)
    _save(fig, "pred_vs_true.png")


def band_confusion(test: pd.DataFrame) -> None:
    models = _models(test)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    yt = to_bands(test["exp_year"])
    for ax, m in zip(axes.flat, models):
        yp = to_bands(np.clip(test[f"pred_{m}"], 0, get_target_cap()))
        cm = pd.crosstab(pd.Categorical(yt, BAND_LABELS), pd.Categorical(yp, BAND_LABELS), dropna=False).to_numpy()
        share = cm / cm.sum(1, keepdims=True).clip(min=1)
        ax.imshow(share, cmap=BLUES, vmin=0, vmax=1)
        for i in range(len(BAND_LABELS)):
            for j in range(len(BAND_LABELS)):
                if cm[i, j]:
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9,
                            color="#ffffff" if share[i, j] > 0.55 else TEXT)
        ax.set_xticks(range(len(BAND_LABELS)), BAND_LABELS)
        ax.set_yticks(range(len(BAND_LABELS)), BAND_LABELS)
        ax.grid(False)
        ax.set_title(f"{SHORT[m]}  (band acc {np.trace(cm) / cm.sum():.2f})")
        ax.set_xlabel("Predicted band")
        ax.set_ylabel("True band")
    for ax in axes.flat[len(models):]:
        ax.set_visible(False)
    fig.suptitle("Experience-band confusion matrices (counts; shade = share of the true band)",
                 x=0.01, ha="left", color=TEXT, fontsize=12)
    fig.tight_layout()
    _save(fig, "band_confusion.png")


def mae_heatmap(table: str, by: str, title: str, fname: str, order=None) -> None:
    df = pd.read_csv(TABLES_DIR / table)
    models = [m for m in SHORT if m in df.columns]
    if order is not None:
        df = df.set_index(by).loc[[o for o in order if o in set(df[by])]].reset_index()
    vals = df[models].to_numpy().T
    fig, ax = plt.subplots(figsize=(1.6 * len(df) + 3, 0.55 * len(models) + 1.6))
    ax.imshow(vals, cmap=BLUES, norm=LogNorm(vmin=max(vals.min(), 0.01), vmax=vals.max()), aspect="auto")
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            ax.text(j, i, f"{vals[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="#ffffff" if vals[i, j] > np.sqrt(vals.max() * max(vals.min(), 0.01)) * 1.5 else TEXT)
    ax.set_xticks(range(len(df)), [f"{str(v).replace('_', ' ')}\n(n={n})" for v, n in zip(df[by], df["n"])], fontsize=8)
    ax.set_yticks(range(len(models)), [SHORT[m] for m in models])
    ax.grid(False)
    ax.set_title(title + "  (MAE in years, log-scaled shade)")
    _save(fig, fname)


def bootstrap_forest() -> None:
    b = pd.read_csv(TABLES_DIR / "bootstrap_mae_gains.csv")
    b = b.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(b) + 1.5))
    ax.axvline(0, color=REF_LINE, lw=1.2, ls="--")
    ax.hlines(range(len(b)), b["ci_low"], b["ci_high"], color=BLUE, lw=2)
    ax.scatter(b["gain"], range(len(b)), s=48, color=BLUE, zorder=3, edgecolors=SURFACE, linewidths=2)
    span = b["ci_high"].max() - min(b["ci_low"].min(), 0)
    for i, r in b.iterrows():
        ax.text(r["ci_high"] + 0.02 * span, i, f"{r['gain']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]",
                fontsize=8.5, color=TEXT_2, va="center")
    ax.set_xlim(min(b["ci_low"].min(), 0) - 0.05 * span, b["ci_high"].max() + 0.45 * span)
    ax.set_ylim(-0.6, len(b) - 0.4)
    ax.set_yticks(range(len(b)), [SHORT.get(m, m) for m in b["vs_model"]])
    bert = SHORT.get(b["bert"].iloc[0], b["bert"].iloc[0])
    ax.set_xlabel(f"MAE(model) - MAE({bert}) in years  (> 0: {bert} is better)")
    ax.set_title(f"{bert} MAE gain vs each model - paired bootstrap, {int(b['n_boot'].iloc[0])} resamples, 95% CI")
    _save(fig, "bootstrap_mae_gains.png")


def residuals(test: pd.DataFrame, bert: str) -> None:
    err = np.clip(test[f"pred_{bert}"], 0, get_target_cap()) - test["exp_year"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    axes[0].hist(err, bins=50, color=BLUE, edgecolor=SURFACE, linewidth=0.5)
    axes[0].axvline(0, color=REF_LINE, ls="--", lw=1.2)
    axes[0].set_xlabel("Signed error (pred - true, years)")
    axes[0].set_ylabel("CVs")
    axes[0].set_title(f"{SHORT[bert]}: error distribution (mean {err.mean():+.3f}, sd {err.std():.3f})")
    axes[1].scatter(test["exp_year"], err, s=9, color=BLUE, alpha=0.45, linewidths=0)
    axes[1].axhline(0, color=REF_LINE, ls="--", lw=1.2)
    axes[1].set_xlabel("True experience (years)")
    axes[1].set_ylabel("Signed error (years)")
    axes[1].set_title("Error vs true experience")
    _save(fig, "bert_residuals.png")


def eda_figures() -> None:
    meta = pd.read_parquet(RAW_METADATA_PATH, columns=["occupation", "exp_year"])
    occs = sorted(meta["occupation"].unique())
    fig, axes = plt.subplots(1, len(occs), figsize=(15, 3), sharey=True)
    for ax, occ in zip(axes, occs):
        y = meta.loc[meta["occupation"] == occ, "exp_year"]
        ax.hist(y, bins=np.arange(0, get_target_cap() + 1), color=BLUE, edgecolor=SURFACE, linewidth=0.6)
        ax.axvline(y.median(), color=TEXT_2, lw=1.2, ls="--")
        ax.set_title(f"{occ.replace('_', ' ')[:24]}\nmedian {y.median():.1f} y", fontsize=9.5)
        ax.set_xlabel("exp_year")
    axes[0].set_ylabel("CVs")
    fig.suptitle("Target distribution by occupation (n = 800 each); dashed line = median", x=0.01, ha="left", color=TEXT)
    _save(fig, "exp_distribution.png")

    q = pd.read_csv(TABLES_DIR / "ocr_quality_per_cv.csv")
    fig, ax = plt.subplots(figsize=(8, 3.2))
    edges = [0, 1e-9, 0.001, 0.0025, 0.005, 0.01]
    counts = pd.cut(q["cer"], bins=edges, include_lowest=True).value_counts().sort_index()
    labels = ["0", "(0, 0.1%]", "(0.1, 0.25%]", "(0.25, 0.5%]", "(0.5, 1%]"]
    ax.bar(labels, counts.values, color=BLUE, width=0.6)
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v}", ha="center", va="bottom", fontsize=9, color=TEXT_2)
    ax.set_ylabel("CVs")
    ax.set_xlabel("Per-CV character error rate vs reference text")
    ax.set_title(f"OCR CER distribution (mean {q['cer'].mean() * 100:.4f}%, {int((q['cer'] == 0).sum())} of {len(q)} CVs exact)")
    _save(fig, "ocr_cer_distribution.png")


def probe_figure() -> None:
    path = TABLES_DIR / "q1_probes.csv"
    if not path.exists():
        return
    p = pd.read_csv(path)
    models = [m for m in SHORT if m in set(p["model"])]
    titles = {"shift_all_+5": "P1  all years +5 (ideal 0)",
              "shift_all_-3": "P1b  all years -3 (ideal 0)",
              "oldest_start_-5": "P2  oldest start -5 y (ideal ~+5)",
              "dates_removed": "P3  dates removed (|change|)"}
    titles = {k: v for k, v in titles.items() if k in set(p["probe"])}
    fig, axes = plt.subplots(1, len(titles), figsize=(5 * len(titles), 0.5 * len(models) + 2), sharey=True)
    for ax, (probe, title) in zip(axes, titles.items()):
        d = p[p["probe"] == probe].set_index("model").loc[models]
        val = d["mean_abs_delta"] if probe == "dates_removed" else d["mean_delta"]
        ideal = d["expected_mean_delta"].iloc[0] if probe == "oldest_start_-5" else None if probe == "dates_removed" else 0
        if ideal is not None:
            ax.axvline(ideal, color=REF_LINE, ls="--", lw=1.2)
        ax.hlines(range(len(models)), 0, val, color=GRID, lw=2)
        ax.scatter(val, range(len(models)), s=48, color=BLUE, zorder=3, edgecolors=SURFACE, linewidths=2)
        for i, v in enumerate(val):
            ax.text(v, i + 0.2, f" {v:.2f}" if probe == "dates_removed" else f" {v:+.2f}", fontsize=8.5, color=TEXT_2)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("mean |change| (years)" if probe == "dates_removed" else "mean change in prediction (years)")
        ax.margins(x=0.2)
    axes[0].set_yticks(range(len(models)), [SHORT[m] for m in models])
    fig.suptitle("Q1 counterfactual date probes on the test set", x=0.01, ha="left", color=TEXT, fontsize=12)
    _save(fig, "q1_probes.png")


def noise_figure() -> None:
    path = TABLES_DIR / "q2_noise_curve.csv"
    if not path.exists():
        return
    n = pd.read_csv(path)
    keep = [m for m in ("bert_distilbert_base_uncased", "baseline_tfidf", "baseline_rules", "baseline_embeddings") if m in set(n["model"])]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, kind in zip(axes, ("digits", "chars")):
        d = n[n["noise"] == kind]
        ends = []
        for slot, m in enumerate(keep):
            s = d[d["model"] == m].sort_values("rate")
            ax.plot(100 * s["rate"], s["mae"], color=SERIES[slot], lw=2, marker="o", ms=5, markeredgecolor=SURFACE)
            ends.append([s["mae"].iloc[-1], SHORT[m], 100 * s["rate"].iloc[-1]])
        # direct labels with a minimum vertical separation so they never collide
        ends.sort()
        gap = 0.06 * (d["mae"].max() - d["mae"].min() + 1e-9)
        for i in range(1, len(ends)):
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
        for y, label, x in ends:
            ax.text(x, y, f"  {label}", color=TEXT, fontsize=8.5, va="center")
        ax.set_xlabel("substitution rate (% of " + ("digits" if kind == "digits" else "alphanumeric chars") + ")")
        ax.set_title("Synthetic " + ("digit" if kind == "digits" else "character") + " noise")
        ax.set_xlim(0, 100 * d["rate"].max() * 1.35)
    axes[0].set_ylabel("Test MAE (years)")
    axes[0].legend([plt.Line2D([], [], color=SERIES[i], lw=2) for i in range(len(keep))], [SHORT[m] for m in keep],
                   loc="upper left", fontsize=8.5)
    fig.suptitle("Q2 dose-response: test MAE as OCR noise is injected", x=0.01, ha="left", color=TEXT, fontsize=12)
    _save(fig, "q2_noise_curve.png")


def sensitivity_figure() -> None:
    path = TABLES_DIR / "q3_sensitivity.csv"
    if not path.exists():
        return
    s = pd.read_csv(path).iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(s) + 1.5))
    ax.hlines(range(len(s)), 0, s["test_mae"], color=GRID, lw=2)
    ax.scatter(s["test_mae"], range(len(s)), s=48, color=BLUE, zorder=3, edgecolors=SURFACE, linewidths=2, label="test MAE")
    for i, r in s.iterrows():
        ax.text(r["test_mae"], i + 0.15, f" {r['test_mae']:.3f}", fontsize=8.5, color=TEXT_2)
    ax.set_yticks(range(len(s)), [f"[{g}] {v}" for g, v in zip(s["group"], s["variant"])], fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Test MAE (years, log scale)")
    ax.set_title("Q3 methodological sensitivity (descriptive; nothing selected on test)")
    _save(fig, "q3_sensitivity.png")


def main() -> None:
    ensure_dirs()
    test = pd.read_csv(PREDICTIONS_DIR / "test_all_models.csv", dtype={"id": str})
    comp = pd.read_csv(TABLES_DIR / "model_comparison.csv")
    bert = load_json(METRICS_DIR / "final_evaluation.json")["primary_bert"]
    pred_vs_true(test, comp)
    band_confusion(test)
    mae_heatmap("mae_by_occupation.csv", "occupation", "Test MAE by occupation", "mae_by_occupation.png")
    mae_heatmap("mae_by_band.csv", "band", "Test MAE by experience band", "mae_by_band.png", order=BAND_LABELS)
    bootstrap_forest()
    residuals(test, bert)
    eda_figures()
    probe_figure()
    noise_figure()
    sensitivity_figure()


if __name__ == "__main__":
    main()
