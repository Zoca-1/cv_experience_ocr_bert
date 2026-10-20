"""Streamlit dashboard: CV experience estimation from PaddleOCR text (exactly 4 tabs).

    streamlit run app.py

Reads the artefacts produced by the pipeline (data/, models/, output/). Tab 4 runs
PaddleOCR + the same cleaning and model code as the evaluation on an uploaded CV.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageDraw

from src.utils import (
    BAND_LABELS, FIGURES_DIR, METRICS_DIR, OCR_DIR, PREDICTIONS_DIR, RAW_METADATA_PATH, ROOT_DIR, TABLES_DIR,
    get_target_cap, to_bands,
)

st.set_page_config(page_title="CV Experience OCR + BERT", page_icon="📄", layout="wide")

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRAY = "#8a8984"
SHORT = {
    "baseline_median": "Median", "baseline_tfidf": "TF-IDF + Ridge", "baseline_embeddings": "Embeddings + Ridge",
    "baseline_rules": "Rule parser", "bert_distilbert_base_uncased": "DistilBERT", "bert_deberta_v3_small": "DeBERTa-v3-small",
}
SEQ = ["#f0efec", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


# ---------------------------------------------------------------------------
# Data loaders (cached)
# ---------------------------------------------------------------------------
def _json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@st.cache_data
def load_table(name: str) -> pd.DataFrame | None:
    p = TABLES_DIR / name
    return pd.read_csv(p, dtype={"id": str}) if p.exists() else None


@st.cache_data
def load_metrics(name: str) -> dict | None:
    return _json(METRICS_DIR / name)


@st.cache_data
def load_meta() -> pd.DataFrame:
    m = pd.read_parquet(RAW_METADATA_PATH, columns=["id", "occupation", "exp_year", "image_path", "text_ref"])
    return m.set_index("id", drop=False)


@st.cache_data
def load_test_predictions() -> pd.DataFrame | None:
    p = PREDICTIONS_DIR / "test_all_models.csv"
    return pd.read_csv(p, dtype={"id": str}) if p.exists() else None


@st.cache_data
def load_ocr(cv_id: str) -> dict | None:
    return _json(OCR_DIR / f"{cv_id}.json")


@st.cache_resource(show_spinner="Loading models ...")
def get_predictors():
    from src.predictors import load_all
    return load_all()


@st.cache_resource(show_spinner="Loading PaddleOCR (first call downloads/initialises the models) ...")
def get_ocr():
    get_predictors()                     # Windows: torch must load its CUDA DLLs before paddle
    from src.ocr_extraction import build_ocr
    try:
        return build_ocr("gpu:0")
    except Exception:                    # no CUDA build / no GPU
        return build_ocr("cpu")


def draw_boxes(image: Image.Image, items: list[dict], max_labels: int = 400) -> Image.Image:
    img = image.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    for it in items[:max_labels]:
        x1, y1, x2, y2 = it["box"]
        d.rectangle([x1, y1, x2, y2], outline=BLUE, width=2)
        d.text((max(0, x1 - 22), y1), str(it["order"]), fill=ORANGE)
    return img


def models_present(df: pd.DataFrame) -> list[str]:
    return [m for m in SHORT if f"pred_{m}" in df.columns]


def style_fig(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=50, b=10),
                      font=dict(size=13), title_x=0.0)
    return fig


def missing(msg: str) -> None:
    st.info(f"{msg} Run the pipeline first (see README).")


# ---------------------------------------------------------------------------
st.title("📄 CV experience estimation: PaddleOCR + BERT")
st.caption("Predicting years of work experience (0-20) from CV images of the PopResume dataset.")
tab1, tab2, tab3, tab4 = st.tabs(["1 · Problem & Methodology", "2 · OCR & Exploratory Analysis",
                                  "3 · Model Results", "4 · Interactive Estimator"])

# ===========================================================================
# TAB 1 - Problem & methodology
# ===========================================================================
with tab1:
    ocr_rt, ocr_q, prep = (load_metrics(n) for n in ("ocr_runtime.json", "ocr_quality.json", "preprocessing.json"))
    cap_dec = _json(ROOT_DIR / "data" / "raw" / "target_cap.json")
    comp = load_table("model_comparison.csv")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CVs (5 occupations x 800)", "4,000")
    if ocr_q:
        c2.metric("OCR character error rate", f"{100 * ocr_q['all']['cer_micro']:.4f} %")
    if comp is not None:
        best = comp[comp["primary_bert"]].iloc[0]
        c3.metric(f"{SHORT.get(best['model'], best['model'])} test MAE", f"{best['mae']:.3f} y")
        tf = comp.set_index("model").loc["baseline_tfidf"]
        c4.metric("Best classical baseline (TF-IDF) MAE", f"{tf['mae']:.3f} y")

    st.subheader("Objective")
    st.markdown(
        "Estimate a candidate's **total years of work experience** (`exp_year`, clipped to "
        f"[0, {get_target_cap()}]) **from the CV image alone**: PaddleOCR reads the image, the text is cleaned and "
        "a fine-tuned BERT regressor predicts the years. Classical baselines quantify what fine-tuning adds.")
    st.subheader("Data")
    st.markdown(
        "- **PopResume** (`sumin-yu/PopResume`, synthetic, population-representative): streamed occupation by "
        "occupation, `has_photo == False`, **800 CVs per occupation** sampled with seed 42 → 4,000 CV images "
        "(1241 x 1754 px, single-column template).\n"
        "- The dataset's `text` column is used **only as OCR ground truth** (CER/WER); model inputs come exclusively "
        "from PaddleOCR.\n"
        "- Never used as features: age, sex, race, names, state, region (sex/race only for the fairness check).")
    if cap_dec:
        st.markdown(f"- **Target cap:** {100 * cap_dec['share_at_default_cap']:.2f}% of CVs exceed 20 years "
                    f"(threshold 15%) → cap kept at **{cap_dec['cap']}**.")
    st.subheader("Pipeline")
    st.markdown(
        "1. **OCR** - PaddleOCR 3.7 (`lang=en`, PP-OCRv6 det/rec), `rec_texts` + `rec_scores` + `rec_boxes`; "
        "reading order rebuilt with a recursive XY-cut that keeps a job title on the same line as its right-aligned "
        "dates; one JSON cache per CV." + (f" {ocr_rt['runtime_per_cv_s']['mean']:.3f} s/CV on "
                                           f"{ocr_rt['hardware'].get('gpu')}." if ocr_rt else "") + "\n"
        "2. **Cleaning** - canonical dashes/bullets, line fixes, PII removal (name, e-mail, phone, location), "
        "regex masking of direct experience/age/birth statements" +
        (f" ({prep['ocr']['masking']['cvs_affected']} CVs affected)." if prep else ".") + "\n"
        "3. **Split** - 70/15/15 stratified by occupation x experience band x render-truncation flag, frozen with a hash.\n"
        "4. **Models** - median · TF-IDF + Ridge · MiniLM embeddings + Ridge · rule-based date parser (reference) · "
        "DistilBERT and DeBERTa-v3-small regressors (MSE on z-scored target, early stopping on validation MAE).\n"
        "5. **Evaluation** - MAE, RMSE, Acc ±1/2/3, band accuracy / macro-F1, R², paired bootstrap 95% CIs.")
    if comp is not None:
        st.subheader("Model summary (test set)")
        st.dataframe(comp[["label", "mae", "rmse", "acc_pm1", "acc_pm2", "band_macro_f1", "r2", "train_time_s",
                           "inference_ms_per_cv"]].round(3), hide_index=True, use_container_width=True)
    st.subheader("Key findings & limitations")
    st.markdown(
        "- `exp_year` equals the **sum of the job date spans** printed on the CV (MAE 0.02 y on the reference text), "
        "so the task is *reading dates and adding them up*. A transparent rule parser is therefore a near-perfect "
        "reference; BERT has to learn this arithmetic from tokens.\n"
        "- OCR is nearly perfect on these clean renders; the only date losses (38 CVs, all teachers) are **cut in the "
        "rendered image itself**, not OCR errors.\n"
        "- **Limitations:** a single synthetic template (no multi-column, scanned or photographed CVs), English only, "
        "US labour-market statistics; experience is defined by the generator, not by HR judgement. Results do not "
        "transfer to real CVs without re-validation.")

# ===========================================================================
# TAB 2 - OCR & EDA
# ===========================================================================
with tab2:
    meta = load_meta()
    st.subheader("Target distribution")
    occs = sorted(meta["occupation"].unique())
    sel = st.multiselect("Occupations", occs, default=occs, key="eda_occ")
    d = meta[meta["occupation"].isin(sel)]
    c1, c2 = st.columns([2, 1])
    fig = px.histogram(d, x="exp_year", nbins=40, color_discrete_sequence=[BLUE],
                       title=f"exp_year distribution (n = {len(d)})", labels={"exp_year": "years of experience"})
    c1.plotly_chart(style_fig(fig, 360), use_container_width=True)
    stats = d.groupby("occupation")["exp_year"].describe()[["count", "mean", "50%", "std", "max"]].round(2)
    c2.dataframe(stats.rename(columns={"50%": "median"}), use_container_width=True)
    if (FIGURES_DIR / "exp_distribution.png").exists():
        with st.expander("Static figure: distribution by occupation"):
            st.image(str(FIGURES_DIR / "exp_distribution.png"))

    st.subheader("OCR quality (PaddleOCR vs reference text)")
    q_occ, q = load_table("ocr_quality_by_occupation.csv"), load_metrics("ocr_quality.json")
    if q is not None and q_occ is not None:
        a = q["all"]
        c = st.columns(5)
        c[0].metric("CER (micro)", f"{100 * a['cer_micro']:.4f} %")
        c[1].metric("WER (micro)", f"{100 * a['wer_micro']:.4f} %")
        c[2].metric("CVs with all date ranges exact", f"{100 * a['cvs_all_ranges_exact']:.2f} %")
        c[3].metric("Graduation-year recall", f"{100 * a['graduation_year_recall']:.1f} %")
        c[4].metric("Render-truncated CVs", q["date_errors"]["cvs_image_truncation"])
        show = q_occ[["occupation", "n_cvs", "cer_micro", "wer_micro", "cvs_all_ranges_exact", "date_digit_accuracy",
                      "employment_year_recall", "graduation_year_recall", "span_abs_err_mean_years"]].copy()
        for col in ("cer_micro", "wer_micro"):
            show[col] = (100 * show[col]).round(4).astype(str) + " %"
        st.dataframe(show.round(4), hide_index=True, use_container_width=True)
        st.caption("Unnormalized CER is %.3f %%: the renderer draws '•' and '–' where the reference text has '-'. "
                   "All 38 missing date ranges are verified render truncations (blank pixels right of the last box)."
                   % (100 * a["cer_unnormalized_mean"]))
        per_cv = load_table("ocr_quality_per_cv.csv")
        if per_cv is not None:
            fig = px.histogram(per_cv, x="cer", nbins=60, log_y=True, color_discrete_sequence=[BLUE],
                               title="Per-CV CER (log count)")
            st.plotly_chart(style_fig(fig, 300), use_container_width=True)
    else:
        missing("No OCR quality report yet.")

    st.subheader("Side-by-side viewer: image with PaddleOCR boxes vs extracted text")
    c1, c2, c3 = st.columns(3)
    occ_v = c1.selectbox("Occupation", occs, key="viewer_occ")
    per_cv = load_table("ocr_quality_per_cv.csv")
    only_trunc = c2.checkbox("Only render-truncated CVs", value=False)
    ids = meta.index[meta["occupation"] == occ_v]
    if only_trunc and per_cv is not None:
        ids = [i for i in ids if i in set(per_cv.loc[per_cv["category"] == "image_truncation", "id"])] or list(ids)
    cv_id = c3.selectbox("CV id", list(ids), key="viewer_id")
    ocr = load_ocr(cv_id)
    if ocr:
        from src.preprocessing import clean_cv
        left, right = st.columns([1.1, 1])
        img = Image.open(ROOT_DIR / meta.loc[cv_id, "image_path"])
        left.image(draw_boxes(img, ocr["items"]), caption=f"{cv_id}: {ocr['n_boxes']} boxes, mean confidence "
                   f"{ocr['mean_confidence']:.3f}, {ocr['runtime_s']:.2f} s; numbers = reading order", use_container_width=True)
        view = right.radio("Text", ["PaddleOCR (reading order)", "Cleaned model input", "Reference (ground truth)"], horizontal=True)
        txt = {"PaddleOCR (reading order)": ocr["text"], "Cleaned model input": clean_cv(ocr["text"]).text,
               "Reference (ground truth)": meta.loc[cv_id, "text_ref"]}[view]
        right.code(txt, language=None)
        right.markdown(f"**True exp_year:** {meta.loc[cv_id, 'exp_year']:.2f} years")
        if per_cv is not None and cv_id in set(per_cv["id"]):
            r = per_cv.set_index("id").loc[cv_id]
            right.caption(f"CER {100 * r['cer']:.3f}% · WER {100 * r['wer']:.3f}% · date ranges exact: {bool(r['ranges_exact'])} "
                          f"· category: {r['category']}")
    else:
        missing("No OCR cache for this CV.")

# ===========================================================================
# TAB 3 - Model results
# ===========================================================================
with tab3:
    comp, test = load_table("model_comparison.csv"), load_test_predictions()
    if comp is None or test is None:
        missing("No final evaluation yet.")
    else:
        st.subheader("Model comparison (held-out test set, n = %d)" % len(test))
        cols = ["label", "mae", "rmse", "r2", "acc_pm1", "acc_pm2", "acc_pm3", "band_accuracy", "band_macro_f1",
                "share_err_gt2", "share_err_gt5", "val_mae", "train_time_s", "inference_ms_per_cv", "end_to_end_ms_per_cv"]
        st.dataframe(comp[cols].round(3), hide_index=True, use_container_width=True)
        st.caption("inference_ms_per_cv = text → prediction; end_to_end adds PaddleOCR (~%.0f ms/CV). The primary BERT "
                   "was selected by validation MAE." % (comp["end_to_end_ms_per_cv"] - comp["inference_ms_per_cv"]).mean())

        models = models_present(test)
        c1, c2 = st.columns(2)
        m = c1.selectbox("Model", models, index=models.index(comp.loc[comp["primary_bert"], "model"].iloc[0]),
                         format_func=lambda x: SHORT[x])
        cap = get_target_cap()
        t = test.assign(pred=np.clip(test[f"pred_{m}"], 0, cap))
        t["abs_error"] = (t["pred"] - t["exp_year"]).abs()
        fig = px.scatter(t, x="exp_year", y="pred", color_discrete_sequence=[BLUE], opacity=0.55,
                         hover_data={"id": True, "occupation": True, "abs_error": ":.2f"},
                         labels={"exp_year": "true years", "pred": "predicted years"},
                         title=f"{SHORT[m]}: predicted vs true (MAE {t['abs_error'].mean():.3f})")
        fig.add_shape(type="line", x0=0, y0=0, x1=cap, y1=cap, line=dict(color=GRAY, dash="dash"))
        fig.update_xaxes(range=[0, cap])
        fig.update_yaxes(range=[0, cap], scaleanchor="x")
        c1.plotly_chart(style_fig(fig, 480), use_container_width=True)

        yt, yp = to_bands(t["exp_year"]), to_bands(t["pred"])
        cm = pd.crosstab(pd.Categorical(yt, BAND_LABELS), pd.Categorical(yp, BAND_LABELS), dropna=False)
        fig = px.imshow(cm.values, x=list(BAND_LABELS), y=list(BAND_LABELS), text_auto=True, color_continuous_scale=SEQ,
                        labels=dict(x="predicted band", y="true band", color="CVs"),
                        title=f"{SHORT[m]}: band confusion (accuracy {np.trace(cm.values) / cm.values.sum():.3f})")
        c2.plotly_chart(style_fig(fig, 480), use_container_width=True)

        st.subheader("Bootstrap confidence intervals for BERT's MAE gain")
        boot = load_table("bootstrap_mae_gains.csv")
        if boot is not None:
            b = boot.iloc[::-1]
            fig = go.Figure(go.Scatter(x=b["gain"], y=[SHORT.get(v, v) for v in b["vs_model"]], mode="markers",
                                       marker=dict(color=BLUE, size=11),
                                       error_x=dict(type="data", symmetric=False, array=b["ci_high"] - b["gain"],
                                                    arrayminus=b["gain"] - b["ci_low"], color=BLUE, thickness=2),
                                       hovertemplate="gain %{x:.3f} y<extra></extra>"))
            fig.add_vline(x=0, line_dash="dash", line_color=GRAY)
            fig.update_layout(title=f"MAE(model) − MAE({SHORT.get(boot['bert'].iloc[0])}), paired bootstrap "
                                    f"{int(boot['n_boot'].iloc[0])} resamples, 95% CI (> 0: BERT better)",
                              xaxis_title="years")
            st.plotly_chart(style_fig(fig, 340), use_container_width=True)
            st.dataframe(boot[["vs_label", "mae_a", "mae_b", "gain", "gain_rel", "ci_low", "ci_high"]].round(4),
                         hide_index=True, use_container_width=True)

        st.subheader("Error breakdown")
        by = st.radio("Breakdown", ["occupation", "band", "truncated"], horizontal=True)
        bd = load_table(f"mae_by_{by}.csv")
        if bd is not None:
            ms = [x for x in SHORT if x in bd.columns]
            if by == "band":
                bd = bd.set_index("band").loc[[b for b in BAND_LABELS if b in set(bd["band"])]].reset_index()
            fig = px.imshow(bd[ms].T.values, x=[f"{v} (n={n})" for v, n in zip(bd[by], bd["n"])], y=[SHORT[x] for x in ms],
                            text_auto=".2f", color_continuous_scale=SEQ, aspect="auto",
                            labels=dict(color="MAE"), title=f"Test MAE by {by}")
            st.plotly_chart(style_fig(fig, 380), use_container_width=True)

        st.subheader("Largest BERT errors")
        top = load_table("bert_top10_errors.csv")
        if top is not None:
            st.dataframe(top.drop(columns=["text"]).round(2), hide_index=True, use_container_width=True)

        st.subheader("Sensitivity and fairness experiments")
        exp = st.radio("Experiment", ["Q1 date probes", "Q2 OCR noise", "Q3 sensitivity", "Q4 fairness"],
                       horizontal=True, key="experiment")   # a radio, not st.tabs: the app has exactly 4 tabs
        if exp == "Q1 date probes":
            pr = load_table("q1_probes.csv")
            if pr is None:
                missing("No probe results.")
            else:
                st.markdown("**P1** every year +5 → ideal Δ = 0 · **P2** oldest job starts 5 y earlier → ideal Δ ≈ +5 · "
                            "**P3** all date ranges removed → how much the model relies on dates.")
                piv = pr.assign(value=np.where(pr["probe"] == "oldest_start_-5", pr["mean_delta"], pr["mean_abs_delta"]))
                st.dataframe(piv.pivot(index="label", columns="probe", values="value").round(3), use_container_width=True)
        elif exp == "Q2 OCR noise":
            q2, nc = load_table("q2_ocr_vs_ref.csv"), load_table("q2_noise_curve.csv")
            if q2 is not None:
                st.dataframe(q2[["label", "trained_on", "tested_on", "n", "mae", "acc_pm1", "acc_pm2"]].round(3),
                             hide_index=True, use_container_width=True)
            if nc is not None:
                kind = st.radio("Noise", ["digits", "chars"], horizontal=True)
                keep = [x for x in ("bert_distilbert_base_uncased", "baseline_tfidf", "baseline_rules", "baseline_embeddings")
                        if x in set(nc["model"])]
                d = nc[(nc["noise"] == kind) & nc["model"].isin(keep)].assign(model_label=lambda x: x["model"].map(SHORT))
                fig = px.line(d, x="rate", y="mae", color="model_label", markers=True,
                              category_orders={"model_label": [SHORT[k] for k in keep]},
                              color_discrete_sequence=[BLUE, ORANGE, AQUA, YELLOW],
                              labels={"rate": "substitution rate", "mae": "test MAE (years)", "model_label": "model"},
                              title=f"MAE vs injected {kind} noise")
                st.plotly_chart(style_fig(fig, 380), use_container_width=True)
        elif exp == "Q3 sensitivity":
            s = load_table("q3_sensitivity.csv")
            st.dataframe(s.round(4), hide_index=True, use_container_width=True) if s is not None else missing("No sensitivity table.")
        else:
            fg, fm = load_table("fairness_gaps.csv"), load_table("fairness_mae.csv")
            if fg is not None:
                st.markdown("Sex and race are used **only here**, never as model inputs. Gap = max − min group MAE; "
                            "p-value from 1,000 label permutations.")
                st.dataframe(fg.drop(columns=["model"]).round(4), hide_index=True, use_container_width=True)
                with st.expander("MAE by group"):
                    st.dataframe(fm.round(4), hide_index=True, use_container_width=True)

# ===========================================================================
# TAB 4 - Interactive estimator
# ===========================================================================
with tab4:
    st.subheader("Upload a CV (image or PDF)")
    st.caption("PaddleOCR runs live; the text then goes through the same cleaning and models as the evaluation. "
               "Models were trained on one synthetic template: treat predictions on real CVs as illustrative.")
    up = st.file_uploader("CV file", type=["png", "jpg", "jpeg", "pdf"])
    if up is not None:
        try:
            if up.name.lower().endswith(".pdf"):
                import fitz  # PyMuPDF

                doc = fitz.open(stream=up.read(), filetype="pdf")
                page_no = st.number_input("Page", 1, doc.page_count, 1) - 1 if doc.page_count > 1 else 0
                pix = doc[page_no].get_pixmap(dpi=150)
                image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            else:
                image = Image.open(up).convert("RGB")
        except Exception as e:  # unreadable upload
            st.error(f"Could not read the file: {e}")
            image = None
        if image is not None:
            from src.ocr_extraction import ocr_image
            from src.preprocessing import clean_cv

            predictors = get_predictors()
            with st.spinner("Running PaddleOCR ..."):
                res = ocr_image(get_ocr(), np.asarray(image))
            cleaned = clean_cv(res["text"])
            c1, c2 = st.columns([1.1, 1])
            c1.image(draw_boxes(image, res["items"]), caption=f"{res['n_boxes']} boxes · mean confidence "
                     f"{res['mean_confidence'] or 0:.3f} · OCR {res['runtime_s']:.2f} s", use_container_width=True)
            c2.markdown("**Cleaned model input** (PII removed, leakage masked)")
            c2.code(cleaned.text, language=None)
            if cleaned.masks:
                c2.warning(f"{len(cleaned.masks)} direct experience/age statement(s) masked: "
                           + "; ".join(m for _, m, _ in cleaned.masks))
            preds = {SHORT.get(n, n): float(p([cleaned.text])[0]) for n, p in predictors.items()}
            pdf = pd.DataFrame({"model": list(preds), "predicted_years": list(preds.values())})
            fig = px.bar(pdf, x="predicted_years", y="model", orientation="h", text="predicted_years",
                         color_discrete_sequence=[BLUE], title="Predictions of every model (years)")
            fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig.update_xaxes(range=[0, get_target_cap() + 1])
            st.plotly_chart(style_fig(fig, 360), use_container_width=True)
            from src.baseline_rules import load_model, parse_ranges
            ranges, n_imp = parse_ranges(cleaned.text, load_model())
            if ranges:
                st.markdown("**Date ranges read by the rule parser** (sum of spans = rule prediction):")
                st.dataframe(pd.DataFrame([{"start": f"{a}.{b:02d}", "end": f"{c}.{d:02d}", "years": ((c * 12 + d) - (a * 12 + b)) / 12}
                                           for a, b, c, d in ranges]).round(2), hide_index=True)
                if n_imp:
                    st.caption(f"{n_imp} range(s) with a truncated end month were imputed.")
            else:
                st.caption("No YYYY.MM-YYYY.MM date ranges found: the rule parser predicts 0 and BERT is extrapolating "
                           "outside its training template.")

    st.divider()
    st.subheader("Explore test-set CVs")
    test = load_test_predictions()
    if test is None:
        missing("No test predictions.")
    else:
        models = models_present(test)
        c1, c2, c3, c4 = st.columns(4)
        m = c1.selectbox("Model for errors", models, format_func=lambda x: SHORT[x], key="explore_model",
                         index=models.index("bert_distilbert_base_uncased") if "bert_distilbert_base_uncased" in models else 0)
        occ = c2.multiselect("Occupation", sorted(test["occupation"].unique()), key="explore_occ")
        yr = c3.slider("True years", 0.0, float(get_target_cap()), (0.0, float(get_target_cap())), key="explore_years")
        err = c4.slider("|error| ≥", 0.0, 5.0, 0.0, 0.05, key="explore_err")
        t = test.copy()
        t["pred"] = np.clip(t[f"pred_{m}"], 0, get_target_cap())
        t["abs_error"] = (t["pred"] - t["exp_year"]).abs()
        f = t[t["exp_year"].between(*yr) & (t["abs_error"] >= err)]
        if occ:
            f = f[f["occupation"].isin(occ)]
        st.caption(f"{len(f)} CVs match · MAE in selection {f['abs_error'].mean() if len(f) else float('nan'):.3f}")
        mode = st.radio("Show", ["Worst predictions", "Best predictions", "All (sorted by id)"], horizontal=True)
        f = f.sort_values("abs_error", ascending=mode == "Best predictions") if mode != "All (sorted by id)" else f.sort_values("id")
        show_cols = ["id", "occupation", "exp_year", "truncated"] + [f"pred_{x}" for x in models] + ["abs_error"]
        st.dataframe(f[show_cols].rename(columns={f"pred_{x}": SHORT[x] for x in models}).round(2).head(200),
                     hide_index=True, use_container_width=True)
        if len(f):
            pick = st.selectbox("Inspect CV", f["id"].head(200).tolist(), key="explore_pick")
            row = t.set_index("id").loc[pick]
            c1, c2 = st.columns([1.1, 1])
            meta = load_meta()
            ocr = load_ocr(pick)
            if ocr:
                c1.image(draw_boxes(Image.open(ROOT_DIR / meta.loc[pick, "image_path"]), ocr["items"]), use_container_width=True)
            c2.markdown(f"**True:** {row['exp_year']:.2f} y · **{SHORT[m]}:** {row['pred']:.2f} y "
                        f"(error {row['pred'] - row['exp_year']:+.2f})")
            c2.dataframe(pd.DataFrame({"model": [SHORT[x] for x in models],
                                       "prediction": [float(np.clip(row[f'pred_{x}'], 0, get_target_cap())) for x in models]}).round(2),
                         hide_index=True)
            c2.code(row["text"], language=None)
