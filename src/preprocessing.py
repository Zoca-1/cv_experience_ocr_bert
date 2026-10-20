"""Text cleaning, PII removal and leakage masking for OCR'd CVs.

Pipeline per CV (input = PaddleOCR reading-order lines from data/ocr/<id>.json):
  1. canonicalize  - NFKC, every dash variant -> "-", bullet glyphs -> "-", whitespace.
  2. fix lines     - drop empty lines, undo end-of-line hyphenation, merge wrapped
                     continuation lines (next line starts lowercase and is not a bullet).
  3. remove PII    - e-mail, phone, URL, street address, ZIP and location ("AL, USA",
                     "Austin, TX") plus the candidate's name. Names come from
                     data/raw/pii.parquet (used ONLY for scrubbing) with a heuristic
                     fallback for unseen CVs (Streamlit uploads). Lines left with nothing
                     but placeholders are dropped: they carry no signal.
  4. mask leakage  - direct statements of total experience ("12 years of experience"),
                     age ("Age: 34", "34 years old") and birth date ("DOB 1985") become
                     [EXP] / [AGE] / [DOB]. Every rule is anchored on a keyword
                     (experience / age / old / born / birth), so employment ranges
                     ("2018.09-2023.01") and graduation years ("University, 2011") are never
                     matched; a per-CV guard re-extracts them and fails loudly otherwise.

Outputs
  data/processed/cvs.parquet          id, occupation, exp_year, text  (MODEL INPUT, from OCR)
  data/processed/cvs_ref.parquet      same pipeline on the reference text - analysis only
                                      (Q2: cost of OCR noise), never a training input
  output/metrics/preprocessing.json   PII / masking counts and CVs affected
  output/tables/masking_examples.csv  every masked span with context

Usage:  python -m src.preprocessing [--selftest]
"""
from __future__ import annotations

import argparse
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd

from src.utils import (
    METRICS_DIR, OCR_DIR, PROCESSED_DIR, RAW_METADATA_PATH, RAW_PII_PATH, TABLES_DIR,
    ensure_dirs, get_logger, load_json, save_json,
)

log = get_logger("preprocessing")

PROCESSED_PATH = PROCESSED_DIR / "cvs.parquet"
PROCESSED_REF_PATH = PROCESSED_DIR / "cvs_ref.parquet"

# ---------------------------------------------------------------------------
# 1. Canonical characters (shared with ocr_quality.py)
# ---------------------------------------------------------------------------
_DASHES = re.compile(r"[‐-―−﹘﹣－]")
_BULLETS = re.compile(r"[•‣⁃∙▪▫●◦·・]")


def canonicalize(text: str) -> str:
    """NFKC + one dash and one bullet glyph; per-line whitespace collapsed, empty lines dropped."""
    text = unicodedata.normalize("NFKC", text)
    text = _DASHES.sub("-", text)
    text = _BULLETS.sub("-", text)
    lines = (re.sub(r"[ \t ]+", " ", ln).strip() for ln in text.splitlines())
    return "\n".join(ln for ln in lines if ln)


# ---------------------------------------------------------------------------
# Dates (shared with ocr_quality.py)
# ---------------------------------------------------------------------------
# Employment ranges as rendered by PopResume: "2018.09-2023.01" (after canonicalize).
DATE_RANGE_RE = re.compile(r"\b((?:19|20)\d{2})\.(\d{2})\s*-\s*((?:19|20)\d{2})\.(\d{2})\b")
# Same range with the end month fully or partly cut at end of line ("2023." / "2023.0"):
# the renderer clips long lines, see ocr_quality.
TRUNCATED_RANGE_RE = re.compile(r"\b((?:19|20)\d{2})\.(\d{2})\s*-\s*((?:19|20)\d{2})(?:\.\d?)?\s*$", re.M)
YEAR_RE = re.compile(r"(?<![\d.])(?:19|20)\d{2}(?![\d])")

SECTION_HEADINGS = {
    "WORK HISTORY": "work", "WORK EXPERIENCE": "work", "EXPERIENCE": "work",
    "PROFESSIONAL EXPERIENCE": "work", "EMPLOYMENT": "work", "EMPLOYMENT HISTORY": "work",
    "EDUCATION": "education", "SKILLS": "other", "SUMMARY": "other", "PROFILE": "other",
    "CERTIFICATIONS": "other", "LICENSES": "other", "OBJECTIVE": "other",
}


def split_sections(lines: Iterable[str]) -> dict[str, list[str]]:
    """Group lines under canonical sections; lines before the first heading are 'header'."""
    out: dict[str, list[str]] = {"header": []}
    cur = "header"
    for ln in lines:
        key = SECTION_HEADINGS.get(ln.strip().upper().rstrip(":"))
        if key:
            cur = key
            out.setdefault(cur, [])
        else:
            out.setdefault(cur, []).append(ln)
    return out


def date_ranges(text: str) -> list[tuple[int, int, int, int]]:
    return [tuple(map(int, m)) for m in DATE_RANGE_RE.findall(text)]


def span_years(ranges: Iterable[tuple[int, int, int, int]]) -> float:
    """Sum of job durations in years (the quantity exp_year is built from)."""
    return sum((y2 * 12 + m2) - (y1 * 12 + m1) for y1, m1, y2, m2 in ranges) / 12


# ---------------------------------------------------------------------------
# 2. Line fixes
# ---------------------------------------------------------------------------
def fix_lines(lines: list[str]) -> list[str]:
    out: list[str] = []
    for ln in lines:
        if out:
            prev = out[-1]
            is_bullet = ln.startswith("- ")
            is_heading = ln.strip().upper() in SECTION_HEADINGS
            prev_heading = prev.strip().upper() in SECTION_HEADINGS
            if not is_bullet and not is_heading and not prev_heading:
                if re.search(r"[A-Za-z]-$", prev) and ln[:1].islower():
                    out[-1] = prev[:-1] + ln              # "manage-" + "ment"
                    continue
                if ln[:1].islower() and not re.search(r"[.:;]$", prev):
                    out[-1] = f"{prev} {ln}"              # wrapped continuation
                    continue
        out.append(ln)
    return out


# ---------------------------------------------------------------------------
# 3. PII
# ---------------------------------------------------------------------------
US_STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire",
    "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York", "NC": "North Carolina",
    "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
    "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee",
    "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia",
}
_STATE_ALT = "|".join(sorted(list(US_STATES) + list(US_STATES.values()), key=len, reverse=True))

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
URL_RE = re.compile(r"\b(?:https?://|www\.)\S+|\b(?:linkedin|github)\.com/\S+", re.I)
PHONE_RE = re.compile(r"(?<![\w.])(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?![\w.])")
STREET_RE = re.compile(
    r"\b\d{1,5}\s+(?:[A-Z][a-z]+\.?\s+){1,3}(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Ln|Lane|"
    r"Dr|Drive|Ct|Court|Way|Pl|Place|Ter|Terrace|Pkwy|Parkway)\b\.?(?:,?\s*(?:Apt|Suite|Unit|#)\s*\w+)?")
# Location patterns are applied to the header block only: bare state codes such as IN/OR/ME
# are ordinary words, and company or school lines in the body are not addresses.
LOCATION_RES = (
    re.compile(rf"\b(?:[A-Z][a-zA-Z.'-]+\s){{0,3}}[A-Z][a-zA-Z.'-]+,\s*(?:{_STATE_ALT})\b(?:,?\s*(?:USA|US|United States))?"),
    re.compile(rf"\b(?:{_STATE_ALT}),\s*(?:USA|US|United States)\b"),
    re.compile(r"\b(?:USA|United States)\b"),
)
ZIP_RE = re.compile(r"\b\d{5}(?:-\d{4})?\b")
# A line holding only placeholders, separators and stray initials ("[NAME] J. [NAME]").
PLACEHOLDER_ONLY = re.compile(r"^(?:\[(?:NAME|EMAIL|PHONE|URL|ADDRESS|LOCATION)\]|\b[A-Z]\.?(?=\s|$)|[\s|,;/-])*$")


@dataclass
class CleanResult:
    text: str
    pii: Counter = field(default_factory=Counter)
    masks: list[tuple[str, str, str]] = field(default_factory=list)   # (rule, match, context)
    guard_ok: bool = True
    guard_msg: str = ""


def _sub(pattern: re.Pattern, repl: str, text: str, counter: Counter, key: str) -> str:
    new, n = pattern.subn(repl, text)
    counter[key] += n
    return new


def _looks_like_name(line: str) -> bool:
    words = line.split()
    return (2 <= len(words) <= 4 and not re.search(r"[\d@|,:/]", line)
            and all(re.fullmatch(r"[A-Z][A-Za-z'.-]*", w) for w in words)
            and line.strip().upper() not in SECTION_HEADINGS)


def remove_pii(lines: list[str], first_name: str | None = None, last_name: str | None = None) -> tuple[list[str], Counter]:
    counts: Counter = Counter()
    sections = split_sections(lines)
    n_header = len(sections["header"])
    full_name = re.compile(rf"\b{re.escape(first_name)}\s+{re.escape(last_name)}\b", re.I) if first_name and last_name else None
    header_tokens = [t for t in (first_name, last_name) if t]
    header_name = re.compile(r"\b(?:" + "|".join(map(re.escape, header_tokens)) + r")\b", re.I) if header_tokens else None

    out, name_found = [], False
    for i, ln in enumerate(lines):
        ln = _sub(EMAIL_RE, "[EMAIL]", ln, counts, "email")
        ln = _sub(URL_RE, "[URL]", ln, counts, "url")
        ln = _sub(PHONE_RE, "[PHONE]", ln, counts, "phone")
        ln = _sub(STREET_RE, "[ADDRESS]", ln, counts, "address")
        if full_name:
            ln = _sub(full_name, "[NAME]", ln, counts, "name")
        if i < n_header:                                   # contact block only
            if header_name:
                ln = _sub(header_name, "[NAME]", ln, counts, "name")
            elif not name_found and _looks_like_name(ln):  # unseen CV: heuristic
                ln, name_found = "[NAME]", True
                counts["name"] += 1
            for rx in LOCATION_RES:
                ln = _sub(rx, "[LOCATION]", ln, counts, "location")
            ln = _sub(ZIP_RE, "[ADDRESS]", ln, counts, "address")
        if PLACEHOLDER_ONLY.match(ln):
            counts["lines_dropped"] += 1
            continue
        out.append(ln)
    return out, counts


# ---------------------------------------------------------------------------
# 4. Leakage masking
# ---------------------------------------------------------------------------
_NUM_WORDS = ("one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
              "fifteen|sixteen|seventeen|eighteen|nineteen|twenty|twenty-five|thirty|a dozen|several|many")
NUM = rf"(?:\d{{1,2}}(?:\.\d)?\s*\+?|{_NUM_WORDS})"
YRS = r"(?:years?|yrs?\.?)"
QUAL = r"(?:over|more than|nearly|almost|about|approximately|approx\.?|around|at least|close to|roughly)"
FIELD = (r"(?:professional|combined|total|relevant|progressive|hands-on|industry|work|working|clinical|"
         r"teaching|classroom|nursing|accounting|auditing|software|development|engineering|construction|"
         r"field|practical|proven|extensive|diverse)")

MASK_RULES: tuple[tuple[str, str, re.Pattern], ...] = (
    ("exp_n_years_of_experience", "[EXP]", re.compile(
        rf"\b(?:{QUAL}\s+)?{NUM}\s*-?\s*{YRS}['’]?(?:\s+of)?(?:\s+{FIELD})*\s+(?:experience|exp\.?)\b", re.I)),
    ("exp_experience_of_n_years", "[EXP]", re.compile(
        rf"\b(?:experience|exp\.?)\s*(?:of|:|-)?\s*(?:{QUAL}\s+)?{NUM}\s*{YRS}", re.I)),
    ("exp_total_experience", "[EXP]", re.compile(
        rf"\b(?:total|overall)\s+(?:work\s+)?experience\s*[:\-]?\s*{NUM}(?:\s*{YRS})?", re.I)),
    ("exp_decade", "[EXP]", re.compile(
        rf"\b(?:{QUAL}\s+)?(?:a|one|two|three)\s+decades?\s+of(?:\s+{FIELD})*\s+experience\b", re.I)),
    ("exp_n_years_in_field", "[EXP]", re.compile(
        rf"\b(?:{QUAL}\s+)?{NUM}\s*{YRS}\s+(?:in|within)\s+(?:the\s+)?(?:\w+\s+){{0,2}}"
        r"(?:field|industry|profession|sector|workforce)\b", re.I)),
    ("age_years_old", "[AGE]", re.compile(rf"\b\d{{1,2}}\s*-?\s*{YRS}\s*-?\s*old\b", re.I)),
    ("age_label", "[AGE]", re.compile(r"\bage[d]?\s*[:\-]?\s*\d{1,2}\b", re.I)),
    ("dob", "[DOB]", re.compile(
        r"\b(?:date\s+of\s+birth|d\.?o\.?b\.?|birth\s*date|birthday|born(?:\s+(?:on|in))?)\s*[:\-]?\s*"
        r"(?:[A-Za-z]{3,9}\.?\s+)?(?:\d{1,2}[\s/.,-]+)?(?:[A-Za-z]{3,9}\.?,?\s+)?(?:\d{1,2}[\s/.,-]+)?(?:19|20)\d{2}\b", re.I)),
)


def mask_leakage(text: str) -> tuple[str, list[tuple[str, str, str]]]:
    """Replace direct experience/age/DOB statements; returns (text, [(rule, match, context)])."""
    hits: list[tuple[str, str, str]] = []
    for rule, token, rx in MASK_RULES:
        src = text

        def _repl(m: re.Match, rule=rule, token=token, src=src) -> str:
            s, e = m.span()
            hits.append((rule, m.group(0), src[max(0, s - 40): e + 40].replace("\n", " / ")))
            return token

        text = rx.sub(_repl, src)
    return text, hits


def _date_guard(before: str, after: str, masks: list[tuple[str, str, str]]) -> tuple[bool, str]:
    """Employment ranges must be untouched; years may only disappear inside [AGE]/[DOB] spans."""
    if date_ranges(before) != date_ranges(after):
        return False, "employment date ranges changed"
    removed = Counter(YEAR_RE.findall(before)) - Counter(YEAR_RE.findall(after))
    allowed = Counter(y for rule, match, _ in masks if not rule.startswith("exp") for y in YEAR_RE.findall(match))
    if removed - allowed:
        return False, f"years removed outside age/DOB masks: {dict(removed - allowed)}"
    return True, ""


# ---------------------------------------------------------------------------
def clean_cv(raw_text: str, first_name: str | None = None, last_name: str | None = None) -> CleanResult:
    """Full pipeline for one CV. Used by the batch job and by the Streamlit app."""
    text = canonicalize(raw_text)
    lines = fix_lines(text.splitlines())
    lines, pii = remove_pii(lines, first_name, last_name)
    before = "\n".join(lines)
    masked, masks = mask_leakage(before)
    ok, msg = _date_guard(before, masked, masks)
    return CleanResult(text=masked, pii=pii, masks=masks, guard_ok=ok, guard_msg=msg)


def process_corpus(texts: pd.Series, names: pd.DataFrame, source: str) -> tuple[pd.DataFrame, dict, list[dict]]:
    rows, examples = [], []
    pii_tot: Counter = Counter()
    pii_cvs: Counter = Counter()
    rule_tot: Counter = Counter()
    rule_cvs: Counter = Counter()
    guard_fail = []
    for cv_id, raw in texts.items():
        fn, ln = (names.loc[cv_id, "first_name"], names.loc[cv_id, "last_name"]) if cv_id in names.index else (None, None)
        res = clean_cv(raw, fn, ln)
        pii_tot.update(res.pii)
        pii_cvs.update({k for k, v in res.pii.items() if v})
        for rule, match, ctx in res.masks:
            rule_tot[rule] += 1
            examples.append({"source": source, "id": cv_id, "rule": rule, "match": match, "context": ctx})
        rule_cvs.update({r for r, _, _ in res.masks})
        if not res.guard_ok:
            guard_fail.append((cv_id, res.guard_msg))
        rows.append({"id": cv_id, "text": res.text, "n_chars_raw": len(raw), "n_chars": len(res.text),
                     "n_lines": res.text.count("\n") + 1, "n_masks": len(res.masks), "guard_ok": res.guard_ok,
                     **{f"pii_{k}": res.pii.get(k, 0) for k in ("name", "email", "phone", "location", "address", "url")}})
    df = pd.DataFrame(rows)
    report = {
        "source": source,
        "n_cvs": len(df),
        "pii_replacements": dict(pii_tot),
        "pii_cvs_affected": dict(pii_cvs),
        "masking": {
            "cvs_affected": int((df["n_masks"] > 0).sum()),
            "cvs_affected_share": round(float((df["n_masks"] > 0).mean()), 4),
            "total_spans": int(df["n_masks"].sum()),
            "spans_by_rule": {r: int(rule_tot.get(r, 0)) for r, _, _ in MASK_RULES},
            "cvs_by_rule": {r: int(rule_cvs.get(r, 0)) for r, _, _ in MASK_RULES},
        },
        "date_guard_failures": guard_fail[:50],
        "n_date_guard_failures": len(guard_fail),
        "chars_mean_raw": round(float(df["n_chars_raw"].mean()), 1),
        "chars_mean_clean": round(float(df["n_chars"].mean()), 1),
    }
    return df, report, examples


def load_ocr_texts(ids: Iterable[str]) -> pd.Series:
    missing = [i for i in ids if not (OCR_DIR / f"{i}.json").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} CVs have no OCR cache (e.g. {missing[:3]}). Run `python -m src.ocr_extraction`.")
    return pd.Series({i: load_json(OCR_DIR / f"{i}.json")["text"] for i in ids})


# ---------------------------------------------------------------------------
def selftest() -> None:
    """Regex regression cases: leakage is masked, dates and graduation years survive."""
    must_mask = [
        ("Registered nurse with 12 years of experience in ICU", "[EXP]"),
        ("Over 10+ years of professional nursing experience", "[EXP]"),
        ("5 yrs experience in payroll", "[EXP]"),
        ("Experience: 7 years", "[EXP]"),
        ("Total experience - 15 years", "[EXP]"),
        ("Nearly a decade of teaching experience", "[EXP]"),
        ("twelve years in the construction industry", "[EXP]"),
        ("Age: 34", "[AGE]"),
        ("34 years old, bilingual", "[AGE]"),
        ("Date of Birth: 12/03/1985", "[DOB]"),
        ("Born in 1979", "[DOB]"),
        ("DOB March 3, 1985", "[DOB]"),
    ]
    must_keep = [
        "Accountants Auditors, FairBooks Accounting, 2018.09-2023.01",
        "Master's, Accounting, Silver Ridge University, 2011",
        "- Led internal audit functions",
        "Software Developers, Agile Systems, 2004.06-2016.09",
        "- Managed 3 years of budget forecasts",            # no "experience" keyword: not a total
        "Golden Field Academy, 2017.07-2020.04",
    ]
    for text, token in must_mask:
        out, _ = mask_leakage(canonicalize(text))
        assert token in out, f"not masked: {text!r} -> {out!r}"
    for text in must_keep:
        out, hits = mask_leakage(canonicalize(text))
        assert out == canonicalize(text) and not hits, f"altered: {text!r} -> {out!r}"

    cv = ("ERIKA WALSH\nAL, USA | +1-852-711-8377 | erika.walsh@example.com\nWORK HISTORY\n"
          "Accountants Auditors, Walsh Field Partners, 2018.09–2023.01\n• Led audits\nEDUCATION\n"
          "Bachelor's, Public Accounting, Forest Hill University, 2009")
    res = clean_cv(cv, "Erika", "Walsh")
    assert "ERIKA" not in res.text and "852" not in res.text and "@" not in res.text and "AL," not in res.text, res.text
    assert "Walsh Field Partners" in res.text, "surname must not be scrubbed from company names in the body"
    assert "2018.09-2023.01" in res.text and "2009" in res.text and res.guard_ok, res.text
    assert res.text.splitlines()[0] == "WORK HISTORY", res.text
    res = clean_cv("JOHN DOE\nAustin, TX 78701\nWORK HISTORY\nNurse, Mercy, 2015.01-2020.01")
    assert "JOHN" not in res.text and "Austin" not in res.text and "78701" not in res.text, res.text
    assert fix_lines(["Designed budgeting", "methodologies for clients", "- Led audits"]) == \
        ["Designed budgeting methodologies for clients", "- Led audits"]
    log.info("selftest passed (%d mask, %d keep cases)", len(must_mask), len(must_keep))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selftest", action="store_true", help="run regex regression cases only")
    args = parser.parse_args()
    selftest()
    if args.selftest:
        return

    ensure_dirs()
    meta = pd.read_parquet(RAW_METADATA_PATH).set_index("id")
    names = pd.read_parquet(RAW_PII_PATH, columns=["id", "first_name", "last_name"]).set_index("id")
    keep = ["occupation", "exp_year", "exp_year_raw"]       # never sex/race/age/state here

    reports, examples = {}, []
    for source, texts, path in (("ocr", load_ocr_texts(meta.index), PROCESSED_PATH),
                                ("reference", meta["text_ref"], PROCESSED_REF_PATH)):
        df, rep, ex = process_corpus(texts, names, source)
        if rep["n_date_guard_failures"]:
            raise SystemExit(f"[{source}] date guard failed for {rep['n_date_guard_failures']} CVs: {rep['date_guard_failures'][:3]}")
        df = meta[keep].join(df.set_index("id")).reset_index()
        df.to_parquet(path, index=False)
        reports[source], examples = rep, examples + ex
        m = rep["masking"]
        log.info("[%s] %d CVs -> %s | PII replaced %s | masking: %d CVs affected (%.2f%%), spans %s",
                 source, len(df), path.name, rep["pii_replacements"], m["cvs_affected"],
                 100 * m["cvs_affected_share"], {k: v for k, v in m["spans_by_rule"].items() if v} or "none")

    save_json({"model_input": "ocr", **reports}, METRICS_DIR / "preprocessing.json")
    cols = ["source", "id", "rule", "match", "context"]
    pd.DataFrame(examples, columns=cols).to_csv(TABLES_DIR / "masking_examples.csv", index=False)


if __name__ == "__main__":
    main()
