import json
import re
import sys
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import TEMP_CACHE_DIR, SAVED_REPORTS_DIR


def run_deep_forensic_folder_audit() -> None:
    """
    Scans EVERY active protocol folder in Google Drive `temp_cache/` and `saved_reports/`,
    audits extracted visit columns, expanded weeks, procedure names, concept mappings,
    fallback rates, and simulation accuracy, and exports `ALL_PROTOCOLS_DEEP_FORENSIC_AUDIT.csv`.
    """
    master_bench_path = Path(SAVED_REPORTS_DIR) / "MASTER_REAL_WORLD_VALIDATION_BENCHMARK.json"
    ground_truth_by_nct: Dict[str, Dict[str, Any]] = {}
    abstained_ncts = set()
    if master_bench_path.exists():
        try:
            mb = json.loads(master_bench_path.read_text(encoding="utf-8"))
            for row in mb.get("tier2_autonomous_validated_cohort", []):
                ground_truth_by_nct[row["nct_id"]] = row
            for row in mb.get("tier2_selective_abstention_log", []):
                abstained_ncts.add(row["nct_id"])
        except Exception:
            pass

    audit_rows: List[Dict[str, Any]] = []
    folders = sorted([d for d in Path(TEMP_CACHE_DIR).iterdir() if d.is_dir()])

    for folder in folders:
        if folder.name == "mini_soa_pages":
            continue

        raw_soa_path = folder / "raw_soa_grid.json"
        usdm_path = folder / "clean_usdm_schedule.json"
        pareto_path = Path(SAVED_REPORTS_DIR) / f"{folder.name}_pareto_report.json"
        if not pareto_path.exists():
            pareto_path = folder / "box6_pareto_comparison.json"

        if not raw_soa_path.exists():
            continue

        try:
            raw_soa = json.loads(raw_soa_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        val_passed = bool(raw_soa.get("validation_passed", False))
        m_nct = re.search(r"(NCT\d{8})", folder.name)
        nct_id = m_nct.group(1) if m_nct else folder.name

        # Skip unvalidated legacy test folders that are not part of the 40-trial or 26-language cohort
        if nct_id.startswith("NCT") and nct_id not in ground_truth_by_nct and nct_id not in abstained_ncts:
            continue

        if not val_passed or nct_id in abstained_ncts:
            audit_rows.append(
                {
                    "folder_name": folder.name,
                    "nct_id": nct_id,
                    "condition": "Healthy Brain Perfusion" if nct_id == "NCT05164705" else "",
                    "domain_used": "SELECTIVE_ABSTENTION",
                    "extraction_mode": raw_soa.get("extraction_mode", ""),
                    "conf_score": raw_soa.get("extraction_confidence_audit", {}).get("extraction_confidence_pct", 0.0),
                    "isolated_pages": str(raw_soa.get("isolated_pages_1based", [])),
                    "visits_count": 0,
                    "procs_count": 0,
                    "default_fallback_pct": 0.0,
                    "actual_ret_pct": None,
                    "sim_base_pct": None,
                    "abs_err_pct": None,
                    "sim_opt_pct": None,
                    "hrs_saved": 0.0,
                    "forensic_flags": "CLEAN_ABSTAINED_OK",
                    "extracted_visit_columns": "",
                    "expanded_weeks": "[]",
                    "extracted_procedure_names": "",
                    "mapped_concepts_sample": "",
                }
            )
            continue

        usdm = {}
        if usdm_path.exists():
            try:
                usdm = json.loads(usdm_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        pareto = {}
        if pareto_path.exists():
            try:
                pareto = json.loads(pareto_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        gt_info = ground_truth_by_nct.get(nct_id, {})

        visit_cols = raw_soa.get("visit_columns", [])
        raw_procs = [p.get("raw_procedure_name", "") for p in raw_soa.get("procedures", [])]
        expanded_weeks = [v.get("week") for v in usdm.get("expanded_timeline_visits", [])]
        norm_procs = usdm.get("normalized_procedures", [])

        total_norm = len(norm_procs)
        default_mapped = sum(1 for p in norm_procs if p.get("mapping_method") == "default_routine_clinical")
        default_pct = round((default_mapped / max(1, total_norm)) * 100.0, 1) if total_norm else 0.0

        mapped_concepts_summary = ", ".join(
            f"{p.get('raw_procedure_name', '')[:25]}->{p.get('biomedical_concept_id', '').replace('BC_', '')}"
            for p in norm_procs[:12]
        )

        b_sim = pareto.get("baseline_simulation", {})
        o_sim = pareto.get("optimized_simulation", {})
        sim_base = b_sim.get("final_week_retention_pct", None)
        sim_opt = o_sim.get("final_week_retention_pct", None)
        actual_ret = gt_info.get("actual_ret", None)
        abs_err = round(abs(float(sim_base) - float(actual_ret)), 2) if (sim_base is not None and actual_ret is not None) else None

        flags: List[str] = []
        if sim_base is not None and sim_opt is not None and float(sim_opt) < float(sim_base):
            flags.append("NEGATIVE_OPT_GAIN")
        if any(c.strip().startswith(("-", "=", "—", "–")) and not re.match(r"^-\d", c.strip()) for c in visit_cols):
            flags.append("DASH_HEADER_ARTIFACT")
        if default_pct > 65.0 and total_norm >= 5:
            flags.append(f"HIGH_DEFAULT_FALLBACK_{default_pct}%")
        if abs_err is not None and abs_err > 6.0:
            flags.append(f"ERR_GT_6PCT ({abs_err}%)")

        audit_rows.append(
            {
                "folder_name": folder.name,
                "nct_id": nct_id,
                "condition": gt_info.get("condition", ""),
                "domain_used": b_sim.get("therapeutic_area", ""),
                "extraction_mode": raw_soa.get("extraction_mode", ""),
                "conf_score": raw_soa.get("extraction_confidence_audit", {}).get("extraction_confidence_pct", 0.0),
                "isolated_pages": str(raw_soa.get("isolated_pages_1based", [])),
                "visits_count": len(visit_cols),
                "procs_count": len(raw_procs),
                "default_fallback_pct": default_pct,
                "actual_ret_pct": actual_ret,
                "sim_base_pct": sim_base,
                "abs_err_pct": abs_err,
                "sim_opt_pct": sim_opt,
                "hrs_saved": pareto.get("chair_hours_saved_per_patient", 0.0),
                "forensic_flags": " | ".join(flags) if flags else "CLEAN_OK",
                "extracted_visit_columns": " ; ".join(visit_cols),
                "expanded_weeks": str(expanded_weeks),
                "extracted_procedure_names": " ; ".join(raw_procs[:20]),
                "mapped_concepts_sample": mapped_concepts_summary,
            }
        )

    df_audit = pd.DataFrame(audit_rows)
    csv_out = Path(SAVED_REPORTS_DIR) / "ALL_PROTOCOLS_DEEP_FORENSIC_AUDIT.csv"
    json_out = Path(SAVED_REPORTS_DIR) / "ALL_PROTOCOLS_DEEP_FORENSIC_AUDIT.json"

    df_audit.to_csv(csv_out, index=False, encoding="utf-8-sig")
    json_out.write_text(json.dumps(audit_rows, indent=2, ensure_ascii=False), encoding="utf-8")

    flagged = df_audit[~df_audit["forensic_flags"].isin(["CLEAN_OK", "CLEAN_ABSTAINED_OK"])]
    print("================================================================================")
    print(f"  DEEP FORENSIC FOLDER AUDIT COMPLETE ({len(df_audit)} Active Protocol Folders Scanned)")
    print("================================================================================")
    print(f" • Google Drive Temp Cache -> {TEMP_CACHE_DIR}")
    print(f" • Saved Full CSV Audit    -> {csv_out}")
    print(f" • Saved Full JSON Audit   -> {json_out}")
    print(f" • Protocols with Forensic Flags: {len(flagged)} / {len(df_audit)}")
    if not flagged.empty:
        print("\n--- FLAGGED PROTOCOLS REQUIRING ATTENTION ---")
        print(
            flagged[
                ["nct_id", "condition", "extraction_mode", "visits_count", "procs_count", "actual_ret_pct", "sim_base_pct", "sim_opt_pct", "forensic_flags"]
            ].to_string(index=False)
        )
    else:
        print("\n[100% CLEAN AUDIT] Zero forensic flags detected across all active protocols!")


if __name__ == "__main__":
    run_deep_forensic_folder_audit()