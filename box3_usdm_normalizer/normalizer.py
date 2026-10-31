import json
import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    OLLAMA_BASE_URL,
    MODELS,
    TEMP_CACHE_DIR,
    BURDEN_DICT_PATH,
    unload_ollama_model,
)

TELEPHONE_CALL_HEADER_REGEX = re.compile(
    r"\b(?:telephone|phone\s*call|telehealth|remote\s*call|phone\s*contact|telephone\s*call)\b",
    re.IGNORECASE,
)

LEARNED_SYNONYMS_CACHE_PATH = Path(TEMP_CACHE_DIR) / "learned_concept_synonyms_cache.json"


def _load_learned_synonyms_cache() -> Dict[str, str]:
    if LEARNED_SYNONYMS_CACHE_PATH.exists():
        try:
            data = json.loads(LEARNED_SYNONYMS_CACHE_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {str(k).casefold(): str(v) for k, v in data.items()}
        except Exception:
            return {}
    return {}


def _save_learned_synonyms_cache(cache_dict: Dict[str, str]) -> None:
    try:
        LEARNED_SYNONYMS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        LEARNED_SYNONYMS_CACHE_PATH.write_text(
            json.dumps(cache_dict, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass


def _clean_header_watermark_and_footnotes(header_str: str) -> str:
    value = re.sub(r"\s+", " ", str(header_str)).strip()
    value = re.sub(r"(?:^|\s)(?:[a-z]\s+){2,}[a-z](?=\s|$)", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"^[a-z]\s+(?=(?:week|wk|day|visit)\b)", "", value, flags=re.IGNORECASE)
    value = re.sub(r"(?<=\d)\s+[a-z](?=\s*\()", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+[a-z]\s*$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\b((?:Week|Wk|Day|Visit|V|W|D)\s*[+-]?\d{1,3})[a-z]\b", r"\1", value, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", value).strip()


def parse_and_expand_visit_header(header: str, cycle_days: int = 28, fallback_idx: int = 1) -> List[Dict[str, Any]]:
    h_clean = _clean_header_watermark_and_footnotes(
        header.replace("\u00ad", "-").replace("–", "-").replace("—", "-").replace("\n", " ").strip()
    )
    h_no_win = re.sub(r"\(\s*(?:[±\uf0b1]|\+/\-)[^)]*\)", " ", h_clean)
    h_no_win = re.sub(r"(?:[±\uf0b1]|\+/\-)\s*\d+", " ", h_no_win)
    h_lower = re.sub(r"\s+", " ", h_no_win).strip().lower()
    is_phone_visit = bool(TELEPHONE_CALL_HEADER_REGEX.search(h_lower))

    # 1. Slash-separated week/day list, e.g., "Wk 18/20/24/28/32/36"
    if "/" in h_lower and re.search(r"\d+/\d+", h_lower):
        nums = [int(n) for n in re.findall(r"\d+", h_lower) if int(n) <= 260]
        if nums:
            return [
                {
                    "visit_id": f"VISIT_WK_{wk:02d}",
                    "visit_name": f"Week {wk} (Extension from {h_clean})",
                    "source_column": h_clean,
                    "week": wk,
                    "study_day": (wk - 1) * 7 + 1,
                    "usdm_epoch": "EXTENSION_TREATMENT_EPOCH",
                    "is_expanded_cycle": True,
                    "is_telephone_call": is_phone_visit,
                }
                for wk in nums
            ]

    # 2. Screening visit OR Negative Week Range OR First-Column "Within 30 days" pre-baseline window
    neg_range_match = re.search(r"(?<!\d)(\-\d{1,2})\s*(?:\-|to)\s*(\-\d{1,2})", h_lower)
    is_before_cycle_screen = bool(
        re.search(r"\b(?:before\s+cycle|prior\s+to\s+(?:v2|day\s*1|cycle|study|surgery)|pre[\-\s]*screening|initial\s+screen)\b", h_lower)
    ) or (fallback_idx == 1 and bool(re.match(r"^within\s+\d+\s*(?:days?|weeks?)\b", h_lower)))

    if neg_range_match or is_before_cycle_screen or any(
        tok in h_lower for tok in ["screen", "διαλογή", "sélection", "selección", "selezione", "triagem", "selectie"]
    ):
        neg_nums = [int(n) for n in re.findall(r"(?<!\d)\-\d{1,2}", h_lower) if -30 <= int(n) <= -1]
        if neg_nums:
            raw_neg = min(neg_nums)
            wk = max(-8, round(raw_neg / 7)) if ("day" in h_lower or raw_neg <= -14) else max(-4, raw_neg)
            if wk == -1 and len(neg_nums) > 1:
                wk = -2
        else:
            wk = -3 if ("pre-" in h_lower or "initial" in h_lower or is_before_cycle_screen) else -2
        return [
            {
                "visit_id": f"VISIT_WK_{wk:+03d}".replace("+", ""),
                "visit_name": h_clean,
                "source_column": h_clean,
                "week": wk,
                "study_day": wk * 7,
                "usdm_epoch": "SCREENING_EPOCH",
                "is_expanded_cycle": False,
                "is_telephone_call": is_phone_visit,
            }
        ]

    # 3. Baseline visit
    if any(tok in h_lower for tok in ["baseline", "έναρξη", "basal", "basale", "aanvang", "day of enrollment"]):
        wk_m = re.search(r"(?:wk|week|εβδ|sem|sett)\s*([01])", h_lower)
        wk = int(wk_m.group(1)) if wk_m else 0
        return [
            {
                "visit_id": f"VISIT_WK_{wk:02d}",
                "visit_name": h_clean,
                "source_column": h_clean,
                "week": wk,
                "study_day": 1,
                "usdm_epoch": "BASELINE_EPOCH",
                "is_expanded_cycle": False,
                "is_telephone_call": is_phone_visit,
            }
        ]

    # 4A. Positive Week Range with frequency, e.g., "Wk 16-24 (q4w)"
    range_match = re.search(r"(?:wk|week|εβδ|woche|sem|sett|w)s?\s*(\d{1,3})\s*(?:\-|to)\s*(\d{1,3})", h_lower)
    if range_match:
        start_wk = int(range_match.group(1))
        end_wk = int(range_match.group(2))
        if 0 <= start_wk < end_wk <= 260:
            step_match = re.search(r"q(\d+)w", h_lower)
            step_wk = int(step_match.group(1)) if step_match else max(1, cycle_days // 7)
            return [
                {
                    "visit_id": f"VISIT_WK_{wk:02d}",
                    "visit_name": f"Week {wk} (Expanded from {h_clean})",
                    "source_column": h_clean,
                    "week": wk,
                    "study_day": (wk - 1) * 7 + 1 if wk > 0 else wk * 7,
                    "usdm_epoch": "TREATMENT_EPOCH",
                    "is_expanded_cycle": True,
                    "is_telephone_call": is_phone_visit,
                }
                for wk in range(start_wk, end_wk + 1, step_wk)
            ]

    # 4B. Visit Number Range, e.g., "Visits 2-9"
    vis_range_match = re.search(r"\b(?:visit|visits|visite|visita|επίσκεψη|v)\s*(\d{1,2})\s*(?:\-|to)\s*(\d{1,2})\b", h_lower)
    if vis_range_match:
        v_start = int(vis_range_match.group(1))
        v_end = int(vis_range_match.group(2))
        if 1 <= v_start < v_end <= 25:
            return [
                {
                    "visit_id": f"VISIT_WK_{(v - 1) * 2:02d}",
                    "visit_name": f"Visit {v} (Expanded from {h_clean})",
                    "source_column": h_clean,
                    "week": (v - 1) * 2,
                    "study_day": (v - 1) * 14 + 1,
                    "usdm_epoch": "TREATMENT_EPOCH",
                    "is_expanded_cycle": True,
                    "is_telephone_call": is_phone_visit,
                }
                for v in range(v_start, v_end + 1)
            ]

    # 4C. Cycle Range, e.g., "Induction (4 to 8 28-day cycles)" or "Cycles 1-6"
    cyc_range_match = (
        re.search(r"\b(?:cycle|cycles|κύκλοι|c)\s*(\d{1,2})\s*(?:\-|to)\s*(\d{1,2})\b", h_lower)
        or re.search(r"\b(\d{1,2})\s*(?:\-|to)\s*(\d{1,2})\s*(?:\d+[\-\s]*day\s+)?cycles?\b", h_lower)
    )
    if cyc_range_match:
        c_start = int(cyc_range_match.group(1))
        c_end = int(cyc_range_match.group(2))
        if 1 <= c_start < c_end <= 24:
            n_cycles = (c_start + c_end) // 2 if c_start > 1 else c_end
            step_wk = max(1, cycle_days // 7)
            return [
                {
                    "visit_id": f"VISIT_WK_{c * step_wk:02d}",
                    "visit_name": f"Cycle {c} (Expanded from {h_clean})",
                    "source_column": h_clean,
                    "week": c * step_wk,
                    "study_day": (c - 1) * cycle_days + 1,
                    "usdm_epoch": "TREATMENT_EPOCH",
                    "is_expanded_cycle": True,
                    "is_telephone_call": is_phone_visit,
                }
                for c in range(1, n_cycles + 1)
            ]

    # 5. Conditional "20 or 44" follow-up week
    if " or " in h_lower and ("wk" in h_lower or "week" in h_lower or "follow" in h_lower):
        nums = [int(n) for n in re.findall(r"\b\d{1,3}\b", h_lower) if int(n) <= 260]
        if nums:
            wk = max(nums)
            return [
                {
                    "visit_id": f"VISIT_WK_{wk:02d}",
                    "visit_name": h_clean,
                    "source_column": h_clean,
                    "week": wk,
                    "study_day": (wk - 1) * 7 + 1,
                    "usdm_epoch": "FOLLOW_UP_EPOCH",
                    "is_expanded_cycle": False,
                    "is_telephone_call": is_phone_visit,
                }
            ]

    # 6. Single Week number (both prefix "Wk 4" and suffix "1w" / "12 wks")
    wk_match = re.search(
        r"\b(?:wk|week|εβδ|εβδομάδα|woche|semaine|semana|settimana|w)s?\s*([\-+]?\d{1,3})\b",
        h_lower,
    ) or re.search(
        r"(?:^|\s)([\-+]?\d{1,3})\s*(?:w|wk|wks|week|weeks)\b",
        h_lower,
    )
    if wk_match:
        wk = int(wk_match.group(1))
        if -16 <= wk <= 260:
            return [
                {
                    "visit_id": f"VISIT_WK_{wk:+03d}".replace("+", ""),
                    "visit_name": h_clean,
                    "source_column": h_clean,
                    "week": wk,
                    "study_day": (wk - 1) * 7 + 1 if wk > 0 else (1 if wk == 0 else wk * 7),
                    "usdm_epoch": "TREATMENT_EPOCH" if wk > 0 else "SCREENING_EPOCH",
                    "is_expanded_cycle": False,
                    "is_telephone_call": is_phone_visit,
                }
            ]

    # 7. Study Day headers (Checks prefix "Day -14" FIRST so "Visit 1 Day -14" never matches "1 Day"!)
    day_match = re.search(
        r"\b(?:day|d|ημέρα|tag|jour|día|giorno|dia|dag)s?\s*([\-\+]?\d{1,3})\b",
        h_lower,
    ) or re.search(
        r"(?:^|\s)([\-\+]?\d{1,3})\s*(?:d|days?)\b",
        h_lower,
    )
    if day_match:
        day_val = int(day_match.group(1))
        if day_val > 365 and str(day_val)[:2].isdigit():
            day_val = int(str(day_val)[:2])
        wk = 0 if day_val == 0 else (max(1, round(day_val / 7)) if day_val > 0 else min(-1, round(day_val / 7)))
        return [
            {
                "visit_id": f"VISIT_DAY_{day_val:+04d}".replace("+", ""),
                "visit_name": h_clean,
                "source_column": h_clean,
                "week": wk,
                "study_day": day_val,
                "usdm_epoch": "TREATMENT_EPOCH" if day_val >= 0 else "PRE_TREATMENT_EPOCH",
                "is_expanded_cycle": False,
                "is_telephone_call": is_phone_visit,
            }
        ]

    # 8. Month headers
    mo_match = re.search(r"\b(?:month|months|mo|m|μήνας|monat|mois|mes|mese|mês|maand)\s*(\d{1,2})\b|\b(\d{1,2})[\-\s]*months?\b", h_lower)
    if mo_match:
        mo_val = int(mo_match.group(1) or mo_match.group(2))
        if 1 <= mo_val <= 60:
            wk = round(mo_val * 4.33)
            return [
                {
                    "visit_id": f"VISIT_WK_{wk:02d}",
                    "visit_name": h_clean,
                    "source_column": h_clean,
                    "week": wk,
                    "study_day": wk * 7,
                    "usdm_epoch": "TREATMENT_EPOCH",
                    "is_expanded_cycle": False,
                    "is_telephone_call": is_phone_visit,
                }
            ]

    # 9. Cycle headers
    cyc_match = re.search(r"\b(?:cycle|κύκλος|zyklus|ciclo|cyclus|c)\s*(\d{1,2})\b", h_lower)
    if cyc_match:
        c_num = int(cyc_match.group(1))
        if 1 <= c_num <= 36:
            wk = (c_num - 1) * max(1, cycle_days // 7) + 1
            return [
                {
                    "visit_id": f"VISIT_WK_{wk:02d}",
                    "visit_name": h_clean,
                    "source_column": h_clean,
                    "week": wk,
                    "study_day": (wk - 1) * 7 + 1,
                    "usdm_epoch": "TREATMENT_EPOCH",
                    "is_expanded_cycle": False,
                    "is_telephone_call": is_phone_visit,
                }
            ]

    # 10. Visit Number headers or EOS / Follow-up
    vis_match = re.search(r"\b(?:visit|visite|visita|επίσκεψη|bezoek|v)\s*([\-+]?\d{1,2})\b", h_lower)
    if vis_match:
        v_num = int(vis_match.group(1))
        wk = v_num if v_num <= 0 else (v_num - 1) * 2
        return [
            {
                "visit_id": f"VISIT_WK_{wk:+03d}".replace("+", ""),
                "visit_name": h_clean,
                "source_column": h_clean,
                "week": wk,
                "study_day": wk * 7 + 1 if wk >= 0 else wk * 7,
                "usdm_epoch": "TREATMENT_EPOCH" if wk >= 0 else "SCREENING_EPOCH",
                "is_expanded_cycle": False,
                "is_telephone_call": is_phone_visit,
            }
        ]

    if any(tok in h_lower for tok in ["follow", "eos", "eot", "end of", "completion"]):
        wk = fallback_idx * 4
        return [
            {
                "visit_id": f"VISIT_WK_{wk:02d}",
                "visit_name": h_clean,
                "source_column": h_clean,
                "week": wk,
                "study_day": wk * 7,
                "usdm_epoch": "FOLLOW_UP_EPOCH",
                "is_expanded_cycle": False,
                "is_telephone_call": is_phone_visit,
            }
        ]

    # 11. VALIDATION GATE 2
    num_any = re.findall(r"(?<!\w)([\-+]?\d{1,2})(?!\w)", h_lower)
    if num_any:
        wk = max(-16, min(104, int(num_any[0])))
    else:
        wk = (fallback_idx - 1) * 2
    return [
        {
            "visit_id": f"VISIT_WK_{wk:+03d}".replace("+", ""),
            "visit_name": h_clean,
            "source_column": h_clean,
            "week": wk,
            "study_day": wk * 7 + 1 if wk >= 0 else wk * 7,
            "usdm_epoch": "TREATMENT_EPOCH" if wk >= 0 else "SCREENING_EPOCH",
            "is_expanded_cycle": False,
            "is_telephone_call": is_phone_visit,
        }
    ]


def _repair_ocr_procedure_text(text: str) -> str:
    t = text.replace("\u00ad", "-").replace("–", "-").replace("—", "-").lower()
    t = re.sub(r"\b(ecg|fsh|hcg|hiv|cbc|cmp|mri|ct|pk)[a-z0-9]\b", r"\1", t)
    t = re.sub(r"(\w)\d+\b", r"\1", t)
    t = re.sub(r"\bhbat\b|\bhba1\b", "hba1c", t)
    t = re.sub(r"ultraso\w*d", "ultrasound", t)
    t = re.sub(r"urin[a-z\)\']{2,8}s", "urinalysis", t)
    t = re.sub(r"wei[a-z\)\']{2,6}t", "weight", t)
    t = re.sub(r"(?:2|7|9)[\-\s]*poin[tl]\s*p[a-z\,]{3,8}", "fasting glucose profile", t)
    t = re.sub(
        r"hypogly[a-z\<\:]{2,8}c|pre[\-\s]*ecla[a-z\,\)\d]{2,8}|teclmical\s*complaint|conception|pregnancy\s*outcome|delivery|iwrs|di[\;\-]?pensing|pensing\s*visit|ouutabilily|endof[lt]rial|case[\-\s]*book|bo[\-\s]*meter|bg[\-\s]*meter|handling",
        "adverse event concomitant compliance",
        t,
    )
    return t


def _classify_unmapped_procedure_with_llm(
    clean_name: str,
    burden_dict: Dict[str, Any],
    learned_cache: Dict[str, str],
) -> Optional[str]:
    """
    Innovation #2 (CENT-Local Tier-2 Constrained Zero-Shot LLM Concept Normalizer):
    Triggered ONLY when an unseen procedure on a new PDF fails Tier-1 deterministic lexical matching.
    Maps the procedure to one of the 15 canonical CDISC/LOINC/SNOMED Biomedical Concepts and
    caches the result permanently in Google Drive (`learned_concept_synonyms_cache.json`).
    """
    cache_key = clean_name.casefold()
    if cache_key in learned_cache and learned_cache[cache_key] in burden_dict:
        return learned_cache[cache_key]

    concept_menu = "\n".join(
        f"- {bc_id}: {info.get('canonical_name', '')}"
        for bc_id, info in burden_dict.items()
    )
    prompt = (
        "You are a CDISC USDM Biomedical Concept Normalizer.\n"
        "Map the following clinical trial procedure to the single closest Biomedical Concept ID from the list below:\n\n"
        f"{concept_menu}\n\n"
        f"Procedure Name: \"{clean_name}\"\n\n"
        "Return ONLY JSON in this exact format: {\"biomedical_concept_id\": \"BC_...\"}"
    )

    model = MODELS["json_normalizer"]
    try:
        resp = requests.post(
            f"{str(OLLAMA_BASE_URL).rstrip('/')}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "think": False,
                "format": "json",
                "options": {"temperature": 0.0, "seed": 42, "num_ctx": 2048, "num_predict": 64},
            },
            timeout=(10, 45),
        )
        resp.raise_for_status()
        raw_resp = resp.json().get("response", "{}")
        parsed = json.loads(raw_resp)
        chosen_id = str(parsed.get("biomedical_concept_id", "")).strip()
        if chosen_id in burden_dict:
            learned_cache[cache_key] = chosen_id
            _save_learned_synonyms_cache(learned_cache)
            return chosen_id
    except Exception:
        pass

    return None


def map_procedure_to_cdisc_concept(
    raw_name: str,
    burden_dict: Dict[str, Any],
    learned_cache: Optional[Dict[str, str]] = None,
    enable_llm_fallback: bool = True,
) -> Dict[str, Any]:
    clean_name = re.sub(r"\(\?+\)", "", raw_name).strip()
    lower_name = _repair_ocr_procedure_text(clean_name)

    best_key = None
    best_score = 0

    for bc_key, bc_data in burden_dict.items():
        score = 0
        for syn in bc_data.get("synonyms", []):
            syn_l = syn.lower()
            if len(syn_l) <= 3:
                if re.search(rf"(?<![a-zα-ω0-9]){re.escape(syn_l)}(?![a-zα-ω0-9])", lower_name):
                    score += len(syn_l) * 4
            else:
                if syn_l in lower_name:
                    score += len(syn_l) * 3
        for token in re.findall(r"[a-zA-Zα-ωΑ-Ω0-9]+", bc_data["canonical_name"].lower()):
            if bc_key == "BC_EXTENDED_COGNITIVE_BATTERY" and token in ("assessment", "battery", "extended"):
                continue
            if bc_key == "BC_TISSUE_BIOPSY" and token in ("tumor", "tissue"):
                continue
            if bc_key == "BC_SERIAL_PK_DRAW" and token in ("sampling", "blood"):
                continue
            if len(token) >= 4 and token in lower_name:
                score += len(token)
        if score > best_score:
            best_score = score
            best_key = bc_key

    if best_key and best_score >= 4:
        matched = dict(burden_dict[best_key])
        matched["biomedical_concept_id"] = best_key
        matched["mapping_method"] = "deterministic_lexical"
        return matched

    # Tier 2: Check Persistent Learned Cache or Run Zero-Shot Constrained LLM Normalizer (CENT-Local)
    if learned_cache is not None:
        cache_key = clean_name.casefold()
        if cache_key in learned_cache and learned_cache[cache_key] in burden_dict:
            c_key = learned_cache[cache_key]
            matched = dict(burden_dict[c_key])
            matched["biomedical_concept_id"] = c_key
            matched["mapping_method"] = "learned_llm_cache"
            return matched

        if enable_llm_fallback and len(clean_name) >= 4:
            llm_key = _classify_unmapped_procedure_with_llm(clean_name, burden_dict, learned_cache)
            if llm_key and llm_key in burden_dict:
                matched = dict(burden_dict[llm_key])
                matched["biomedical_concept_id"] = llm_key
                matched["mapping_method"] = "zero_shot_llm_cent"
                return matched

    fallback_key = "BC_PHYSICAL_EXAM_VITALS"
    matched = dict(burden_dict[fallback_key])
    matched["biomedical_concept_id"] = fallback_key
    matched["mapping_method"] = "default_routine_clinical"
    return matched


def apply_footnote_modifiers(
    base_concept: Dict[str, Any],
    footnote_refs: List[str],
    footnotes_dict: Dict[str, str],
) -> Dict[str, Any]:
    modified = dict(base_concept)
    applied_notes = []

    for ref in footnote_refs:
        fn_text = footnotes_dict.get(ref, "") or footnotes_dict.get(ref.lower(), "")
        if not fn_text:
            continue
        applied_notes.append(f"[{ref}] {fn_text}")
        fn_lower = fn_text.lower()

        if "fasting" in fn_lower or "overnight" in fn_lower or "νηστεία" in fn_lower:
            modified["fasting_required"] = True

        if "caregiver" in fn_lower or "driver" in fn_lower or "συνοδός" in fn_lower:
            modified["caregiver_required"] = True

        hr_match = re.search(r"(\d+(?:\.\d+)?)\s*[\-\s]?hour(?!\s+overnight|\s+fast|\s+prior)", fn_lower)
        if hr_match and ("chair" in fn_lower or "observation" in fn_lower or "post-dose" in fn_lower):
            fn_hours = float(hr_match.group(1))
            modified["chair_hours"] = max(modified["chair_hours"], fn_hours)

        if "optional" in fn_lower or "προαιρετικ" in fn_lower:
            modified["is_optional_in_some_arms"] = True
        else:
            modified.setdefault("is_optional_in_some_arms", False)

    modified["bound_footnotes"] = applied_notes
    return modified


def run_usdm_normalizer(raw_soa_json_path: Path) -> Dict[str, Any]:
    raw_soa_json_path = Path(raw_soa_json_path)
    raw_data = json.loads(raw_soa_json_path.read_text(encoding="utf-8"))
    burden_dict = json.loads(BURDEN_DICT_PATH.read_text(encoding="utf-8"))
    learned_cache = _load_learned_synonyms_cache()

    cycle_days = raw_data.get("protocol_metadata", {}).get("cycle_length_days") or 28

    column_to_visits_map: Dict[str, List[Dict[str, Any]]] = {}
    unique_visits_by_week: Dict[int, Dict[str, Any]] = {}

    for idx, col_header in enumerate(raw_data.get("visit_columns", [])):
        expanded_visits = parse_and_expand_visit_header(col_header, cycle_days=cycle_days, fallback_idx=idx + 1)
        adjusted_visits = []
        for v in expanded_visits:
            wk = v["week"]
            while wk in unique_visits_by_week and unique_visits_by_week[wk]["source_column"].casefold() != v["source_column"].casefold():
                wk += 1
            v_copy = dict(v)
            v_copy["week"] = wk
            v_copy["visit_id"] = f"VISIT_WK_{wk:+03d}".replace("+", "")
            if wk not in unique_visits_by_week:
                unique_visits_by_week[wk] = v_copy
            adjusted_visits.append(unique_visits_by_week[wk])
        column_to_visits_map[col_header] = adjusted_visits

    all_timeline_visits = [unique_visits_by_week[k] for k in sorted(unique_visits_by_week.keys())]

    raw_footnotes = raw_data.get("footnotes_dictionary", {})
    clean_footnotes = {
        k.lower(): v for k, v in raw_footnotes.items() if re.match(r"^[a-zA-Z\*†‡]$", str(k))
    }

    used_llm_fallback = False
    usdm_procedures: List[Dict[str, Any]] = []
    for proc in raw_data.get("procedures", []):
        raw_name = proc["raw_procedure_name"]
        base_concept = map_procedure_to_cdisc_concept(
            raw_name,
            burden_dict,
            learned_cache=learned_cache,
            enable_llm_fallback=True,
        )
        if base_concept.get("mapping_method") == "zero_shot_llm_cent":
            used_llm_fallback = True

        expanded_encounters_by_wk: Dict[int, Dict[str, Any]] = {}
        for sv in proc.get("scheduled_visits", []):
            col_hdr = sv["visit_header"]
            fn_refs = sv.get("footnote_refs", [])
            mod_concept = apply_footnote_modifiers(base_concept, fn_refs, clean_footnotes)

            for target_visit in column_to_visits_map.get(col_hdr, []):
                wk = target_visit["week"]
                is_phone = bool(target_visit.get("is_telephone_call", False))
                expanded_encounters_by_wk[wk] = {
                    "visit_id": target_visit["visit_id"],
                    "week": wk,
                    "study_day": target_visit["study_day"],
                    "usdm_epoch": target_visit.get("usdm_epoch", "TREATMENT_EPOCH"),
                    "footnote_refs": fn_refs,
                    "effective_chair_hours": 0.15 if is_phone else mod_concept["chair_hours"],
                    "planned_duration_iso8601": f"PT{int((0.15 if is_phone else mod_concept['chair_hours']) * 60)}M",
                    "effective_invasiveness": 0.0 if is_phone else mod_concept["invasiveness_0_10"],
                    "fasting_required": False if is_phone else mod_concept["fasting_required"],
                    "pro_cognitive_mins": mod_concept["pro_cognitive_mins"],
                    "caregiver_required": False if is_phone else mod_concept["caregiver_required"],
                    "home_health_eligible": True if is_phone else mod_concept["home_health_eligible"],
                    "converted_to_home_health": is_phone,
                    "site_cost_usd_est": 25.0 if is_phone else mod_concept.get("site_cost_usd_est", 100.0),
                    "is_optional_in_some_arms": mod_concept.get("is_optional_in_some_arms", False),
                }

        if expanded_encounters_by_wk:
            usdm_procedures.append(
                {
                    "raw_procedure_name": re.sub(r"\(\?+\)", "", raw_name).strip(),
                    "biomedical_concept_id": base_concept["biomedical_concept_id"],
                    "cdisc_code": base_concept["cdisc_code"],
                    "loinc_code": base_concept.get("loinc_code", "N/A"),
                    "snomed_code": base_concept.get("snomed_code", "N/A"),
                    "ich_m11_category": base_concept.get("ich_m11_category", "CLINICAL_ASSESSMENT"),
                    "canonical_name": base_concept["canonical_name"],
                    "endpoint_tier": base_concept["endpoint_tier"],
                    "mapping_method": base_concept["mapping_method"],
                    "scheduled_encounters": [
                        expanded_encounters_by_wk[k] for k in sorted(expanded_encounters_by_wk.keys())
                    ],
                }
            )

    if used_llm_fallback:
        unload_ollama_model(MODELS["json_normalizer"])

    usdm_schedule = {
        "usdm_version": "CDISC-USDM-v4.1-ICH-M11-CeSHarP-Compatible",
        "regulatory_standards": ["US_FDA_CDISC_USDM_v4.1", "EU_EMA_ICH_M11_2026", "EU_EMA_ICH_E6_R3", "LOINC_v2.78", "SNOMED_CT"],
        "source_pdf": raw_data.get("source_pdf"),
        "protocol_metadata": raw_data.get("protocol_metadata"),
        "clean_footnotes": clean_footnotes,
        "expanded_timeline_visits": all_timeline_visits,
        "normalized_procedures": usdm_procedures,
    }

    out_path = raw_soa_json_path.parent / "clean_usdm_schedule.json"
    out_path.write_text(json.dumps(usdm_schedule, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"[Box 3: USDM Normalizer] Expanded {len(raw_data.get('visit_columns', []))} table columns "
        f"into {len(all_timeline_visits)} discrete calendar visits "
        f"(Weeks {[v['week'] for v in all_timeline_visits]}) -> {out_path}"
    )
    return usdm_schedule


if __name__ == "__main__":
    sample_raw = TEMP_CACHE_DIR / "sample_mash_protocol_2026" / "raw_soa_grid.json"
    run_usdm_normalizer(sample_raw)