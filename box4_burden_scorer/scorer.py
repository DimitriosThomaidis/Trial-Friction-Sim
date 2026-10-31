import json
import sys
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import TEMP_CACHE_DIR
from box3_usdm_normalizer.normalizer import run_usdm_normalizer


def compute_weekly_burden_scores(usdm_schedule: Dict[str, Any], out_dir: Path) -> pd.DataFrame:
    """
    Box 4 Entrypoint:
    Aggregates all procedures scheduled at each calendar visit week and calculates:
      1. 5-Dimensional Tufts CSDD / TransCelerate Patient Burden Vector
      2. Log-Subadditive Intra-Concept Assessment Stacking Load (Innovation #3: L_stack = sum(1 + 0.35*ln(m_c)))
      3. Composite Visit Friction Index (0–100) & Estimated Site Cost ($ USD)
    """
    visits = usdm_schedule.get("expanded_timeline_visits", [])
    procedures = usdm_schedule.get("normalized_procedures", [])

    visit_rows: List[Dict[str, Any]] = []

    for v in visits:
        wk = v["week"]
        vid = v["visit_id"]

        active_procs = []
        for p in procedures:
            for enc in p.get("scheduled_encounters", []):
                if enc["week"] == wk:
                    active_procs.append(
                        {
                            "canonical_name": p["canonical_name"],
                            "biomedical_concept_id": p["biomedical_concept_id"],
                            "cdisc_code": p.get("cdisc_code", ""),
                            "loinc_code": p.get("loinc_code", ""),
                            "snomed_code": p.get("snomed_code", ""),
                            "endpoint_tier": p["endpoint_tier"],
                            **enc,
                        }
                    )

        if not active_procs:
            continue

        dedup_by_concept: Dict[str, Dict[str, Any]] = {}
        concept_counts: Dict[str, int] = {}
        for ap in active_procs:
            cid = ap["biomedical_concept_id"]
            concept_counts[cid] = concept_counts.get(cid, 0) + 1
            if cid not in dedup_by_concept or ap["effective_chair_hours"] > dedup_by_concept[cid]["effective_chair_hours"]:
                dedup_by_concept[cid] = ap

        unique_procs = list(dedup_by_concept.values())

        # Innovation #3: Log-Subadditive Intra-Concept Assessment Stacking Load
        # Captures multi-questionnaire / multi-panel stacking within the same concept category
        stacked_assessment_load = round(
            float(sum(1.0 + 0.35 * np.log(max(1, m_c)) for m_c in concept_counts.values())),
            2,
        )

        max_hrs = max(ap["effective_chair_hours"] for ap in unique_procs)
        raw_hours_sum = sum(ap["effective_chair_hours"] for ap in unique_procs)
        total_chair_hours = round(max_hrs + 0.65 * (raw_hours_sum - max_hrs), 2)

        max_invasiveness = max(ap["effective_invasiveness"] for ap in unique_procs)
        invasive_proc_count = sum(1 for ap in unique_procs if ap["effective_invasiveness"] >= 4.0)
        effective_pain_score = round(min(10.0, max_invasiveness + 0.6 * max(0, invasive_proc_count - 1)), 2)

        fasting_required = any(ap["fasting_required"] for ap in unique_procs)
        total_pro_mins = sum(ap["pro_cognitive_mins"] for ap in unique_procs)
        caregiver_required = any(ap["caregiver_required"] for ap in unique_procs)

        all_home_eligible = all(ap.get("home_health_eligible", False) for ap in unique_procs)
        is_decentralized_home_visit = all_home_eligible and any(
            ap.get("converted_to_home_health", False) for ap in unique_procs
        )

        visit_site_cost_usd = round(
            sum(
                float(ap.get("site_cost_usd_est", 100.0)) * (0.55 if ap.get("converted_to_home_health", False) else 1.0)
                for ap in unique_procs
            ),
            2,
        )

        friction_index = (
            min(total_chair_hours / 8.0, 1.2) * 30.0
            + (effective_pain_score / 10.0) * 35.0
            + (10.0 if fasting_required else 0.0)
            + min(total_pro_mins / 180.0, 1.0) * 10.0
            + (15.0 if caregiver_required else 0.0)
        )
        friction_index = round(min(100.0, friction_index), 1)

        if friction_index >= 65.0:
            risk_tier = "CRITICAL_DROPOUT_SHOCK"
        elif friction_index >= 35.0:
            risk_tier = "MODERATE_BURDEN"
        else:
            risk_tier = "LOW_ROUTINE_BURDEN"

        visit_rows.append(
            {
                "visit_id": vid,
                "visit_name": v["visit_name"],
                "week": wk,
                "study_day": v["study_day"],
                "procedure_count": len(unique_procs),
                "raw_procedure_count": len(active_procs),
                "stacked_assessment_load": stacked_assessment_load,
                "total_chair_hours": total_chair_hours,
                "effective_pain_0_10": effective_pain_score,
                "fasting_required": fasting_required,
                "pro_cognitive_mins": total_pro_mins,
                "caregiver_required": caregiver_required,
                "fully_decentralizable_visit": all_home_eligible,
                "is_decentralized_home_visit": is_decentralized_home_visit,
                "visit_site_cost_usd": visit_site_cost_usd,
                "visit_friction_index_0_100": friction_index,
                "risk_tier": risk_tier,
                "procedures_list": [ap["canonical_name"] for ap in unique_procs],
            }
        )

    df = pd.DataFrame(visit_rows).sort_values("week").reset_index(drop=True)

    parquet_path = out_dir / "weekly_burden_scores.parquet"
    json_path = out_dir / "weekly_burden_summary.json"

    df.to_parquet(parquet_path, index=False)
    json_path.write_text(df.to_json(orient="records", indent=2, force_ascii=False), encoding="utf-8")

    print(f"[Box 4: Burden Scorer] Scored {len(df)} calendar visits -> {parquet_path}")
    return df


if __name__ == "__main__":
    out_dir = TEMP_CACHE_DIR / "sample_mash_protocol_2026"
    raw_soa_path = out_dir / "raw_soa_grid.json"
    usdm_sched = run_usdm_normalizer(raw_soa_path)
    compute_weekly_burden_scores(usdm_sched, out_dir)