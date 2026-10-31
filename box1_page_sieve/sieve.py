import re
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import fitz  # PyMuPDF

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import SOA_KEYWORDS, TEMP_CACHE_DIR, PROTOCOLS_DIR

EXTENDED_SOA_HEADERS = SOA_KEYWORDS + [
    "study calendar",
    "study flow chart",
    "study flowchart",
    "table of assessments",
    "assessment schedule",
    "schedule of study procedures",
    "schedule of evaluations",
    "study timeline",
    "time and events",
    "study visits",
    "data collection schedule",
]

TIME_HEADER_PATTERN = re.compile(
    r"\b(week|wk|day|cycle|visit|month|screening|baseline|follow[\s\-]?up|eos|eot|"
    r"d\s*[\-\+]?\d+|c\d+d\d+|v\d+|w\d+|t\d+|pre[\-\s]?treatment|post[\-\s]?treatment|"
    r"εβδομάδα|εβδ|ημέρα|κύκλος|επίσκεψη|έλεγχος|διαλογή|"
    r"woche|tag|zyklus|visite|semaine|jour|semana|día)\b",
    re.IGNORECASE,
)

CHECKMARK_PATTERN = re.compile(
    r"(?:^|\s|\|)([Xx✓✔☑✗✘•●┼√■♦\uf0fc\uf076\uf0a8\uf078])(?:\s|[a-d1-9\*†‡]|\||$)"
)


def score_pdf_page(page: fitz.Page, page_idx: int) -> Dict[str, Any]:
    text = page.get_text("text")
    text_lower = text.lower()
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if ("table of contents" in text_lower or "list of tables" in text_lower) and (
        text_lower.count("...") > 2 or text_lower.count(". .") > 2
    ):
        return {
            "page_number_1based": page_idx + 1,
            "page_index_0based": page_idx,
            "total_score": 0.0,
            "matched_keywords": [],
            "time_token_count": 0,
            "checkmark_count": len(CHECKMARK_PATTERN.findall(text)),
            "footnote_line_count": 0,
            "has_ruled_soa_table": False,
        }

    matched_keywords = [kw for kw in EXTENDED_SOA_HEADERS if kw in text_lower]
    time_matches = TIME_HEADER_PATTERN.findall(text_lower)
    check_matches = CHECKMARK_PATTERN.findall(text)
    drawings_count = len(page.get_drawings())

    footnote_lines = sum(
        1 for ln in lines if re.match(r"^([a-z]|\d+|[\*†‡])[\.\)]\s+\w+", ln)
    )

    has_ruled_soa_table = False
    try:
        tabs = page.find_tables()
        for t in tabs.tables:
            rows = t.extract()
            if len(rows) >= 3 and len(rows[0]) >= 3:
                top_text = " ".join(
                    str(c) for r in rows[:4] for c in r if c is not None
                ).lower()
                if len(TIME_HEADER_PATTERN.findall(top_text)) >= 3:
                    has_ruled_soa_table = True
                    break
    except Exception:
        pass

    if not has_ruled_soa_table and any(
        p in text_lower
        for p in ["estimand e1:", "intercurrent events:", "inclusion criteria", "exclusion criteria"]
    ):
        return {
            "page_number_1based": page_idx + 1,
            "page_index_0based": page_idx,
            "total_score": 5.0,
            "matched_keywords": matched_keywords,
            "time_token_count": len(time_matches),
            "checkmark_count": len(check_matches),
            "footnote_line_count": footnote_lines,
            "has_ruled_soa_table": False,
        }

    table_bonus = 65.0 if has_ruled_soa_table else 0.0
    keyword_score = 35.0 if matched_keywords else 0.0
    time_score = min(len(time_matches) * 2.2, 30.0)
    check_score = min(len(check_matches) * 2.0, 50.0)
    grid_score = min(drawings_count * 0.2, 15.0)

    total_score = table_bonus + keyword_score + time_score + check_score + grid_score

    return {
        "page_number_1based": page_idx + 1,
        "page_index_0based": page_idx,
        "total_score": round(total_score, 2),
        "matched_keywords": matched_keywords,
        "time_token_count": len(time_matches),
        "checkmark_count": len(check_matches),
        "footnote_line_count": footnote_lines,
        "has_ruled_soa_table": has_ruled_soa_table,
    }


def extract_protocol_synopsis_and_title_text(pdf_path: Path, max_chars: int = 3000) -> str:
    """
    Extracts the protocol title, indication/condition, and synopsis from the first 3 pages
    of any uploaded PDF so `classify_clinical_domain_from_condition` works on raw user PDF
    uploads even when no ClinicalTrials.gov API metadata is present.
    """
    try:
        doc = fitz.open(pdf_path)
        front_pages = [doc[i].get_text("text") for i in range(min(3, len(doc)))]
        doc.close()
        return "\n".join(front_pages)[:max_chars]
    except Exception:
        return ""


def extract_scheduling_prose_from_pdf(pdf_path: Path, max_chars: int = 5500) -> str:
    doc = fitz.open(pdf_path)
    scored_pages = []
    for idx, page in enumerate(doc):
        txt = page.get_text("text")
        txt_l = txt.lower()
        if "table of contents" in txt_l and txt_l.count("...") > 2:
            continue
        t_hits = len(TIME_HEADER_PATTERN.findall(txt_l))
        proc_hits = sum(
            1 for w in [
                "blood", "vital", "physical", "ecg", "mri", "biopsy", "consent",
                "questionnaire", "survey", "assessment", "baseline", "follow-up",
                "visit", "week", "day", "month", "laboratory", "sample", "dose"
            ]
            if w in txt_l
        )
        scored_pages.append((t_hits + proc_hits * 2, idx, txt))

    doc.close()
    scored_pages.sort(key=lambda x: x[0], reverse=True)
    combined = "\n\n--- PROTOCOL SECTION ---\n\n".join(p[2] for p in scored_pages[:5])
    return combined[:max_chars]


def run_page_sieve(
    pdf_path: Path,
    score_threshold: float = 45.0,
    max_soa_pages: int = 6,
    dpi: int = 150,
    excluded_pages_0based: Optional[List[int]] = None,
) -> Dict[str, Any]:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"Protocol PDF not found: {pdf_path}")

    excluded_set = set(excluded_pages_0based or [])
    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    if not excluded_set:
        print(f"[Box 1: Page Sieve] Scanning '{pdf_path.name}' ({total_pages} pages)...")
    else:
        print(f"[Box 1: Backtracking Sieve] Re-scanning '{pdf_path.name}' (excluding pages {[i+1 for i in excluded_set]})...")

    front_matter_text = "\n".join(doc[i].get_text("text") for i in range(min(3, total_pages)))[:3000]
    page_scores = [score_pdf_page(doc[i], i) for i in range(total_pages)]
    available_scores = [p for p in page_scores if p["page_index_0based"] not in excluded_set]
    if not available_scores:
        available_scores = page_scores

    ruled_indices = [p["page_index_0based"] for p in available_scores if p["has_ruled_soa_table"]]

    if ruled_indices:
        clusters: List[List[int]] = []
        for idx in ruled_indices:
            if not clusters or idx - clusters[-1][-1] > 2:
                clusters.append([idx])
            else:
                clusters[-1].append(idx)
        best_cluster = max(
            clusters,
            key=lambda cl: sum(page_scores[i]["total_score"] + page_scores[i]["checkmark_count"] * 2 for i in cl),
        )
        selected_indices = list(range(best_cluster[0], min(best_cluster[-1] + 1, best_cluster[0] + max_soa_pages)))
    else:
        sorted_avail = sorted(
            available_scores,
            key=lambda p: (len(p["matched_keywords"]) > 0, p["total_score"], p["time_token_count"]),
            reverse=True,
        )
        anchor = sorted_avail[0]["page_index_0based"]
        selected_indices = sorted(
            {
                i for i in range(max(0, anchor - 1), min(total_pages, anchor + 3))
                if i not in excluded_set and page_scores[i]["total_score"] >= 15.0
            }
        )
        if not selected_indices:
            selected_indices = [p["page_index_0based"] for p in sorted_avail[: min(3, len(sorted_avail))]]

    selected_indices = selected_indices[:max_soa_pages]

    protocol_stem = pdf_path.stem.replace(" ", "_")
    out_dir = TEMP_CACHE_DIR / protocol_stem
    img_dir = out_dir / "soa_page_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    for old_png in img_dir.glob("*.png"):
        old_png.unlink(missing_ok=True)

    mini_pdf_path = out_dir / "mini_soa_pages.pdf"
    mini_doc = fitz.open()
    rendered_images: List[str] = []

    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    for local_idx, orig_idx in enumerate(selected_indices):
        mini_doc.insert_pdf(doc, from_page=orig_idx, to_page=orig_idx)
        pix = doc[orig_idx].get_pixmap(matrix=mat, alpha=False)
        img_path = img_dir / f"soa_page_{local_idx + 1:02d}_orig_{orig_idx + 1:03d}.png"
        pix.save(str(img_path))
        rendered_images.append(str(img_path))

    mini_doc.save(str(mini_pdf_path))
    mini_doc.close()
    doc.close()

    manifest = {
        "source_pdf": str(pdf_path),
        "total_source_pages": total_pages,
        "selected_original_pages_1based": [i + 1 for i in selected_indices],
        "selected_original_pages_0based": selected_indices,
        "mini_soa_pdf_path": str(mini_pdf_path),
        "rendered_page_images": rendered_images,
        "front_matter_synopsis_text": front_matter_text,
        "page_diagnostics": [page_scores[i] for i in selected_indices],
    }

    manifest_path = out_dir / "box1_sieve_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"[Box 1: Page Sieve] Isolated {len(selected_indices)} page(s) "
        f"{manifest['selected_original_pages_1based']} -> {mini_pdf_path}"
    )
    return manifest


def create_sample_protocol_pdf(output_pdf_path: Path) -> Path:
    output_pdf_path = Path(output_pdf_path)
    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()

    for p_num, title in [(1, "1. PROTOCOL SYNOPSIS & BACKGROUND"), (2, "2. STUDY OBJECTIVES & ELIGIBILITY")]:
        page = doc.new_page(width=612, height=792)
        page.insert_text((50, 60), "PROTOCOL NCT-SIM-2026 | PHASE IIb MASH & METABOLIC TRIAL", fontsize=11)
        page.insert_text((50, 95), title, fontsize=14)
        page.insert_text((50, 130), "Standard background pharmacology and inclusion criteria.", fontsize=10)

    p3 = doc.new_page(width=612, height=792)
    p3.insert_text((40, 45), "3. SCHEDULE OF ASSESSMENTS (SoA)", fontsize=12)
    p3.insert_text((40, 65), "Cycle Length: 28 Days (4 Weeks) | Total Duration: 52 Weeks | Arms: Arm A, Arm B", fontsize=9)

    headers = ["Procedure / Assessment", "Screen\n(Wk -2)", "Baseline\n(Wk 1)", "Wk 4", "Wk 8", "Wk 12", "Wk 16-24\n(q4w)", "Wk 52\n(EOS)"]
    rows = [
        ["Informed Consent", "X", "", "", "", "", "", ""],
        ["Physical Exam & Vitals", "X", "X", "X", "X", "X", "X", "X"],
        ["Routine Safety Blood Draw (CBC/CMP)", "X", "X", "X", "X", "X", "X", "X"],
        ["Fasting Lipid & Metabolic Panel", "X a", "X a", "", "X a", "X a", "X a", "X a"],
        ["12-Lead ECG", "X", "X", "", "", "X", "", "X"],
        ["Serial 4-Hour PK Blood Sampling", "", "X b", "X b", "", "X b", "", ""],
        ["Contrast MRI-PDFF Liver Imaging", "X", "", "", "", "X", "X", "X"],
        ["Ultrasound-Guided Liver Biopsy", "X c", "", "", "", "X c", "", "X c"],
        ["Lumbar Puncture (Exploratory CSF)", "", "X d", "", "", "X d", "", ""],
        ["PRO Quality of Life Questionnaire", "", "X", "X", "X", "X", "X", "X"],
        ["Extended Neurocognitive Battery (ADAS)", "X", "X", "", "", "X", "", "X"],
    ]

    x_cols = [40, 215, 270, 325, 365, 405, 450, 520, 575]
    y_top = 85
    row_h = 26

    for c_idx, h_text in enumerate(headers):
        rect = fitz.Rect(x_cols[c_idx], y_top, x_cols[c_idx + 1], y_top + 32)
        p3.draw_rect(rect)
        p3.insert_textbox(rect, h_text, fontsize=7.5)

    y_curr = y_top + 32
    for r in rows:
        for c_idx, cell_val in enumerate(r):
            rect = fitz.Rect(x_cols[c_idx], y_curr, x_cols[c_idx + 1], y_curr + row_h)
            p3.draw_rect(rect)
            p3.insert_textbox(rect, cell_val, fontsize=7.5)
        y_curr += row_h

    footnotes_text = (
        "Schedule of Assessments Footnotes:\n"
        "a. Requires 12-hour overnight fasting prior to morning venipuncture.\n"
        "b. Serial PK sampling collected pre-dose, 1h, 2h, and 4h post-dose (4.5 hours clinic chair time).\n"
        "c. Invasive liver biopsy requires 6-hour post-procedure observation and a caregiver driver.\n"
        "d. Exploratory CSF lumbar puncture is optional for Arm B but scheduled for Arm A."
    )
    p3.insert_textbox(fitz.Rect(40, y_curr + 15, 570, y_curr + 120), footnotes_text, fontsize=8)

    p4 = doc.new_page(width=612, height=792)
    p4.insert_text((50, 60), "4. STATISTICAL ANALYSIS PLAN", fontsize=13)
    p4.insert_text((50, 100), "Standard statistical power requires >= 80% cohort retention.", fontsize=10)

    doc.save(str(output_pdf_path))
    doc.close()
    return output_pdf_path


if __name__ == "__main__":
    sample_pdf = PROTOCOLS_DIR / "sample_mash_protocol_2026.pdf"
    run_page_sieve(sample_pdf)