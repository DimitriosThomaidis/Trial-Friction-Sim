import json
import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import PROTOCOLS_DIR, SAVED_REPORTS_DIR, TEMP_CACHE_DIR
from box1_page_sieve.sieve import extract_protocol_synopsis_and_title_text
from box2_table_extractor.extractor import run_table_extractor
from box5_dropout_simulator.aact_duckdb_loader import classify_clinical_domain_from_condition
from box6_optimizer_ui.pareto_simplifier import run_full_pareto_optimization

# Strictly exclude all 40 development/calibration trials + 2 legacy test trials
EXCLUDED_DEVELOPMENT_NCTS = {
    "NCT01615029", "NCT01720667", "NCT01724346", "NCT01876901", "NCT02059239",
    "NCT02224703", "NCT02371369", "NCT02412670", "NCT02532283", "NCT02617628",
    "NCT02623829", "NCT02648477", "NCT02708966", "NCT02709486", "NCT02720094",
    "NCT03013751", "NCT03125629", "NCT03173456", "NCT03247556", "NCT03260205",
    "NCT03332303", "NCT03334539", "NCT03377699", "NCT03390842", "NCT03537742",
    "NCT03645096", "NCT03802396", "NCT03895372", "NCT03931941", "NCT04063163",
    "NCT04169373", "NCT04206553", "NCT04251156", "NCT04540094", "NCT04682405",
    "NCT04880642", "NCT05164705", "NCT05196919", "NCT05486065", "NCT05645354",
    "NCT05707351", "NCT05890586",
}

# Pinned 20 Holdout Trials: 10 from Wave 1 + 10 from Wave 2 for 100% deterministic evaluation
PINNED_HOLDOUT_20_NCTS = [
    # Wave 1 (10 Unseen Trials)
    "NCT04449029", "NCT01434602", "NCT03576001", "NCT02951819", "NCT01617967",
    "NCT04286607", "NCT01570387", "NCT04399837", "NCT03385239", "NCT03065075",
    # Wave 2 (10 Unseen Trials)
    "NCT03296553", "NCT01987895", "NCT01857934", "NCT03449134", "NCT05579899",
    "NCT03075826", "NCT02978326", "NCT03939689", "NCT03489850", "NCT02157974",
]


def download_holdout_protocol_pdf(nct_id: str, pdf_url: str) -> Path:
    target = Path(PROTOCOLS_DIR) / f"HOLDOUT_{nct_id}_Protocol.pdf"
    if target.exists() and target.stat().st_size > 15_000:
        return target
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) TrialFrictionSim/1.0"}
    r = requests.get(pdf_url, headers=headers, timeout=60)
    r.raise_for_status()
    target.write_bytes(r.content)
    return target


def discover_blind_holdout_protocols(target_count: int = 20) -> List[Dict[str, Any]]:
    """
    Fetches the 20 unseen Holdout ClinicalTrials.gov protocols (Wave 1 + Wave 2)
    with uploaded Sponsor Protocol PDFs and posted participant flow results,
    strictly excluding all 42 development NCT IDs.
    """
    search_url = "https://clinicaltrials.gov/api/v2/studies"
    base_params: Dict[str, Any] = {
        "filter.advanced": (
            "AREA[LargeDocHasProtocol]true AND AREA[OverallStatus]COMPLETED "
            "AND AREA[StudyType]INTERVENTIONAL AND AREA[Phase](PHASE2 OR PHASE3)"
        ),
        "pageSize": 100,
    }

    print(
        f"[Blind Holdout Discovery] Loading {target_count} unseen holdout trials "
        f"(Wave 1 + Wave 2, excluding all {len(EXCLUDED_DEVELOPMENT_NCTS)} development NCT IDs)..."
    )

    discovered_holdout: List[Dict[str, Any]] = []
    seen_ncts = set(EXCLUDED_DEVELOPMENT_NCTS)
    next_page_token: Optional[str] = None

    for page_idx in range(5):
        if len(discovered_holdout) >= target_count:
            break

        studies: List[Dict[str, Any]] = []

        # On page 0, load the 20 pinned Wave 1 + Wave 2 holdout trials in exact deterministic order
        if page_idx == 0:
            for ref_nct in PINNED_HOLDOUT_20_NCTS:
                try:
                    r_ref = requests.get(f"https://clinicaltrials.gov/api/v2/studies/{ref_nct}", timeout=20)
                    if r_ref.status_code == 200:
                        studies.append(r_ref.json())
                except Exception:
                    pass

        if len(studies) < target_count:
            params = dict(base_params)
            if next_page_token:
                params["pageToken"] = next_page_token
            try:
                resp = requests.get(search_url, params=params, timeout=30)
                resp.raise_for_status()
                payload = resp.json()
                studies.extend(payload.get("studies", []))
                next_page_token = payload.get("nextPageToken")
            except Exception:
                next_page_token = None

        for st_data in studies:
            if len(discovered_holdout) >= target_count:
                break

            p_sec = st_data.get("protocolSection", {})
            r_sec = st_data.get("resultsSection", {})
            d_sec = st_data.get("documentSection", {})

            nct_id = p_sec.get("identificationModule", {}).get("nctId", "")
            if not nct_id or nct_id in seen_ncts:
                continue
            seen_ncts.add(nct_id)

            large_docs = d_sec.get("largeDocumentModule", {}).get("largeDocs", [])
            pdf_filename = None
            for d in large_docs:
                fname = d.get("filename", "")
                if d.get("hasProtocol") and fname.endswith(".pdf"):
                    pdf_filename = fname
                    break
            if not pdf_filename:
                continue

            best_started = 0
            best_completed = 0
            periods = r_sec.get("participantFlowModule", {}).get("periods", [])
            for period in periods:
                p_started = 0
                p_completed = 0
                for ms in period.get("milestones", []):
                    m_type = ms.get("type", "").upper()
                    c_sum = sum(
                        int(ach.get("numSubjects", 0))
                        for ach in ms.get("achievements", [])
                        if str(ach.get("numSubjects", "")).isdigit()
                    )
                    if m_type == "STARTED" and c_sum > p_started:
                        p_started = c_sum
                    elif m_type == "COMPLETED" and c_sum > p_completed:
                        p_completed = c_sum
                if p_started >= 20 and 0 < p_completed <= p_started:
                    if p_started > best_started:
                        best_started = p_started
                        best_completed = p_completed

            if best_started < 20 or best_completed <= 0 or best_completed > best_started:
                continue

            actual_dropout_pct = round(((best_started - best_completed) / best_started) * 100.0, 2)
            actual_retention_pct = round(100.0 - actual_dropout_pct, 2)

            if actual_retention_pct < 10.0:
                continue

            last_two = nct_id[-2:]
            pdf_url = f"https://clinicaltrials.gov/ProvidedDocs/{last_two}/{nct_id}/{pdf_filename}"

            try:
                pdf_path = download_holdout_protocol_pdf(nct_id, pdf_url)
            except Exception:
                continue

            title = p_sec.get("identificationModule", {}).get("briefTitle", nct_id)
            phases = p_sec.get("designModule", {}).get("phases", ["PHASE2"])
            conditions = p_sec.get("conditionsModule", {}).get("conditions", ["Clinical"])

            discovered_holdout.append(
                {
                    "nct_id": nct_id,
                    "brief_title": title[:80],
                    "phase": ",".join(phases),
                    "condition": conditions[0] if conditions else "Clinical",
                    "started_n": best_started,
                    "completed_n": best_completed,
                    "actual_dropout_pct": actual_dropout_pct,
                    "actual_retention_pct": actual_retention_pct,
                    "pdf_url": pdf_url,
                    "pdf_path": pdf_path,
                }
            )
            print(
                f"  [Blind Holdout #{len(discovered_holdout):02d}] {nct_id} "
                f"({conditions[0][:25]} | Actual Ret: {actual_retention_pct}%) -> Ready"
            )

        if not next_page_token:
            break

    return discovered_holdout


def run_blind_holdout_test(target_count: int = 20) -> None:
    print("================================================================================")
    print(f"  OUT-OF-SAMPLE BLIND HOLDOUT VALIDATION ({target_count} UNSEEN CLINICALTRIALS.GOV PDFs)")
    print("================================================================================")

    holdout_cohort = discover_blind_holdout_protocols(target_count=target_count)
    sim_rows: List[Dict[str, Any]] = []
    abstained_rows: List[Dict[str, Any]] = []
    forensic_rows: List[Dict[str, Any]] = []

    for study in holdout_cohort:
        nct_id = study["nct_id"]
        pdf_path = Path(study["pdf_path"])
        cache_dir = Path(TEMP_CACHE_DIR) / pdf_path.stem

        try:
            soa_extracted = run_table_extractor(pdf_path)
            val_passed = bool(soa_extracted.get("validation_passed", False))
            conf_pct = float(soa_extracted.get("extraction_confidence_audit", {}).get("extraction_confidence_pct", 0.0))
            ext_mode = str(soa_extracted.get("extraction_mode", "PATH_A_RULED_VECTOR_GRID"))
            short_mode = ext_mode.replace("PATH_", "")[:12]

            manifest_path = cache_dir / "box1_sieve_manifest.json"
            front_synopsis = ""
            if manifest_path.exists():
                try:
                    front_synopsis = json.loads(manifest_path.read_text(encoding="utf-8")).get("front_matter_synopsis_text", "")
                except Exception:
                    pass
            if not front_synopsis:
                front_synopsis = extract_protocol_synopsis_and_title_text(pdf_path)

            proc_text = " ".join(p.get("raw_procedure_name", "") for p in soa_extracted.get("procedures", []))
            clinical_domain = classify_clinical_domain_from_condition(
                condition=study["condition"],
                phase=study["phase"],
                brief_title=study["brief_title"],
                synopsis_fallback=f"{front_synopsis} {proc_text}",
            )

            if not val_passed:
                abstained_rows.append(
                    {
                        "nct_id": nct_id,
                        "condition": study["condition"][:22],
                        "status": "ABSTAINED_ROUTED_TO_UI_EDITOR",
                        "heuristic_score": conf_pct,
                        "actual_ret": study["actual_retention_pct"],
                        "reason": "No machine-verifiable SoA grid (>2 visits & >=3 procedures) in PDF.",
                    }
                )
                forensic_rows.append(
                    {
                        "folder_name": pdf_path.stem,
                        "nct_id": nct_id,
                        "condition": study["condition"],
                        "domain_used": "SELECTIVE_ABSTENTION",
                        "extraction_mode": ext_mode,
                        "conf_score": conf_pct,
                        "isolated_pages": str(soa_extracted.get("isolated_pages_1based", [])),
                        "visits_count": 0,
                        "procs_count": 0,
                        "default_fallback_pct": 0.0,
                        "actual_ret_pct": study["actual_retention_pct"],
                        "sim_base_pct": None,
                        "abs_err_pct": None,
                        "sim_opt_pct": None,
                        "hrs_saved": 0.0,
                        "extracted_visit_columns": "",
                        "expanded_weeks": "[]",
                        "extracted_procedure_names": "",
                        "mapped_concepts_sample": "",
                    }
                )
                print(
                    f"[Selective Abstention Gate] {nct_id} ({study['condition'][:20]}): "
                    f"Abstained safely (Score: {conf_pct}/100) -> Routed to Human-in-the-Loop UI."
                )
                continue

            bundle = run_full_pareto_optimization(
                cache_dir,
                therapeutic_area=clinical_domain,
                generate_llm_memo=False,
            )

            sim_base = float(bundle["baseline_simulation"]["final_week_retention_pct"])
            sim_opt = float(bundle["optimized_simulation"]["final_week_retention_pct"])
            actual_ret = float(study["actual_retention_pct"])
            abs_err = round(abs(sim_base - actual_ret), 2)

            usdm_data = json.loads((cache_dir / "clean_usdm_schedule.json").read_text(encoding="utf-8"))
            norm_procs = usdm_data.get("normalized_procedures", [])
            expanded_visits = usdm_data.get("expanded_timeline_visits", [])
            visit_cols = soa_extracted.get("visit_columns", [])
            raw_procs = [p.get("raw_procedure_name", "") for p in soa_extracted.get("procedures", [])]

            default_mapped = sum(1 for p in norm_procs if p.get("mapping_method") == "default_routine_clinical")
            default_pct = round((default_mapped / max(1, len(norm_procs))) * 100.0, 1) if norm_procs else 0.0

            mapped_sample = ", ".join(
                f"{p.get('raw_procedure_name', '')[:25]}->{p.get('biomedical_concept_id', '').replace('BC_', '')}"
                for p in norm_procs[:12]
            )

            sim_rows.append(
                {
                    "nct_id": nct_id,
                    "condition": study["condition"][:18],
                    "domain": bundle["baseline_simulation"]["therapeutic_area"][:16],
                    "mode": short_mode,
                    "conf": conf_pct,
                    "visits": len(expanded_visits),
                    "procs": len(norm_procs),
                    "fb_pct": default_pct,
                    "actual_ret": actual_ret,
                    "sim_base": sim_base,
                    "abs_err": abs_err,
                    "sim_opt": sim_opt,
                    "hrs_saved": bundle["chair_hours_saved_per_patient"],
                }
            )

            forensic_rows.append(
                {
                    "folder_name": pdf_path.stem,
                    "nct_id": nct_id,
                    "condition": study["condition"],
                    "domain_used": bundle["baseline_simulation"]["therapeutic_area"],
                    "extraction_mode": ext_mode,
                    "conf_score": conf_pct,
                    "isolated_pages": str(soa_extracted.get("isolated_pages_1based", [])),
                    "visits_count": len(visit_cols),
                    "procs_count": len(raw_procs),
                    "default_fallback_pct": default_pct,
                    "actual_ret_pct": actual_ret,
                    "sim_base_pct": sim_base,
                    "abs_err_pct": abs_err,
                    "sim_opt_pct": sim_opt,
                    "hrs_saved": bundle["chair_hours_saved_per_patient"],
                    "extracted_visit_columns": " ; ".join(visit_cols),
                    "expanded_weeks": str([v.get("week") for v in expanded_visits]),
                    "extracted_procedure_names": " ; ".join(raw_procs[:20]),
                    "mapped_concepts_sample": mapped_sample,
                }
            )
        except Exception as err:
            print(f"[Blind Holdout Notice] {nct_id} encountered notice: {err}")

    df_sim = pd.DataFrame(sim_rows)
    print("\n================ BLIND HOLDOUT REAL PROTOCOL PDFs TABLE (20 UNSEEN TRIALS) ================")
    if not df_sim.empty:
        print(df_sim.to_string(index=False))
        overall_mae = round(float(df_sim["abs_err"].mean()), 2)
        overall_median = round(float(df_sim["abs_err"].median()), 2)
        overall_rmse = round(float(np.sqrt((df_sim["abs_err"] ** 2).mean())), 2)

        tier_a_df = df_sim[df_sim["conf"] >= 75.0]
        tier_b_df = df_sim[df_sim["conf"] < 75.0]

        tier_a_mae = round(float(tier_a_df["abs_err"].mean()), 2) if not tier_a_df.empty else 0.0
        tier_b_mae = round(float(tier_b_df["abs_err"].mean()), 2) if not tier_b_df.empty else 0.0

        print(
            f"\n[BLIND HOLDOUT SCORECARD] Unseen PDFs Tested: {len(holdout_cohort)} | "
            f"Simulated: {len(df_sim)} | Selective Abstentions (Tier C): {len(abstained_rows)}\n"
            f" • Overall Blind Holdout MAE: {overall_mae}% (Median: {overall_median}% | RMSE: {overall_rmse}%)\n"
            f" • Tier A High-Confidence Autonomous MAE (n={len(tier_a_df)}): {tier_a_mae}%\n"
            f" • Tier B Model-Reconstructed MAE (n={len(tier_b_df)}, UI Recommends Human Review): {tier_b_mae}%"
        )

    if abstained_rows:
        print("\n================ BLIND HOLDOUT SELECTIVE ABSTENTION LOG ================")
        print(pd.DataFrame(abstained_rows).to_string(index=False))

    csv_out = Path(SAVED_REPORTS_DIR) / "BLIND_10_TRIAL_HOLDOUT_FORENSIC_AUDIT.csv"
    json_out = Path(SAVED_REPORTS_DIR) / "BLIND_10_TRIAL_HOLDOUT_BENCHMARK.json"

    pd.DataFrame(forensic_rows).to_csv(csv_out, index=False, encoding="utf-8-sig")
    json_out.write_text(
        json.dumps(
            {
                "holdout_simulated_trials": sim_rows,
                "holdout_abstained_trials": abstained_rows,
                "holdout_forensic_details": forensic_rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\n[COMPLETE] Saved Blind Holdout Forensic CSV  -> {csv_out}")
    print(f"[COMPLETE] Saved Blind Holdout Benchmark JSON -> {json_out}")


if __name__ == "__main__":
    run_blind_holdout_test(target_count=20)