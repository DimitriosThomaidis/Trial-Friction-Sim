import json
import re
import sys
from pathlib import Path
from typing import Dict, Any, List
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import PROTOCOLS_DIR, AACT_PARQUET_DIR, SAVED_REPORTS_DIR, TEMP_CACHE_DIR
from box2_table_extractor.extractor import run_table_extractor
from box5_dropout_simulator.aact_duckdb_loader import (
    query_trialbench_38k_summary,
    classify_clinical_domain_from_condition,
)
from box6_optimizer_ui.pareto_simplifier import run_full_pareto_optimization


def discover_real_protocols_from_clinicaltrials_api(target_count: int = 40) -> List[Dict[str, Any]]:
    """
    Discovers 40 real COMPLETED interventional trials with uploaded Sponsor Protocol PDFs
    and posted participant flow results (`STARTED` vs `COMPLETED`) from ClinicalTrials.gov API v2.
    Preserves the exact order of our 30 validated protocols (#01-#30) and appends 10 additional
    hard real-world protocols (#31-#40), while filtering out corrupt registry milestones (<5% completion).
    """
    search_url = "https://clinicaltrials.gov/api/v2/studies"
    params = {
        "filter.advanced": "AREA[LargeDocHasProtocol]true AND AREA[OverallStatus]COMPLETED AND AREA[StudyType]INTERVENTIONAL AND AREA[Phase](PHASE2 OR PHASE3)",
        "pageSize": 100,
    }
    print(f"[Tier 2: 40-Trial Multi-Wave Discovery] Querying live ClinicalTrials.gov API v2 for {target_count} completed trials with Protocol PDFs...")
    resp = requests.get(search_url, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    api_studies = payload.get("studies", [])

    next_token = payload.get("nextPageToken")
    if next_token and len(api_studies) < 150:
        try:
            params["pageToken"] = next_token
            r2 = requests.get(search_url, params=params, timeout=30)
            if r2.status_code == 200:
                api_studies.extend(r2.json().get("studies", []))
        except Exception:
            pass

    # Ordered list of 40 real ClinicalTrials.gov protocols (#01-#30 baseline cohort + #31-#40 new hard cohort)
    ordered_priority_ncts = [
        "NCT03895372", "NCT04682405", "NCT03645096", "NCT02224703",
        "NCT05645354", "NCT05164705", "NCT03260205", "NCT04251156",
        "NCT03247556", "NCT03013751", "NCT03334539", "NCT01720667",
        "NCT05486065", "NCT04063163", "NCT01876901", "NCT05196919",
        "NCT01724346", "NCT03173456", "NCT01615029", "NCT04880642",
        "NCT05890586", "NCT02709486", "NCT02412670", "NCT02371369",
        "NCT04540094", "NCT03802396", "NCT03332303", "NCT03537742",
        "NCT03390842", "NCT04169373",
        # 10 Additional Hard Real-World Tests (#31 to #40):
        "NCT02708966", "NCT02059239", "NCT02648477", "NCT02617628",
        "NCT02532283", "NCT03931941", "NCT05707351", "NCT02623829",
        "NCT03377699", "NCT04206553", "NCT03040414", "NCT00911508",
    ]

    priority_studies: List[Dict[str, Any]] = []
    for ref_nct in ordered_priority_ncts:
        try:
            r_ref = requests.get(f"https://clinicaltrials.gov/api/v2/studies/{ref_nct}", timeout=20)
            if r_ref.status_code == 200:
                priority_studies.append(r_ref.json())
        except Exception:
            pass

    studies = priority_studies + api_studies

    discovered_cohort: List[Dict[str, Any]] = []
    seen_ncts = set()

    for st_data in studies:
        if len(discovered_cohort) >= target_count:
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
            if p_started >= 20 and p_completed > 0 and p_completed <= p_started:
                if p_started > best_started:
                    best_started = p_started
                    best_completed = p_completed

        if best_started < 20 or best_completed <= 0 or best_completed > best_started:
            continue

        actual_dropout_pct = round(((best_started - best_completed) / best_started) * 100.0, 2)
        actual_retention_pct = round(100.0 - actual_dropout_pct, 2)

        if actual_retention_pct < 5.0:
            continue

        last_two = nct_id[-2:]
        pdf_url = f"https://clinicaltrials.gov/ProvidedDocs/{last_two}/{nct_id}/{pdf_filename}"

        try:
            pdf_path = download_protocol_pdf(nct_id, pdf_url)
        except Exception:
            continue

        title = p_sec.get("identificationModule", {}).get("briefTitle", nct_id)
        phases = p_sec.get("designModule", {}).get("phases", ["PHASE2"])
        conditions = p_sec.get("conditionsModule", {}).get("conditions", ["Clinical"])

        discovered_cohort.append(
            {
                "nct_id": nct_id,
                "brief_title": title[:65],
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
        print(f"  [Cohort Study #{len(discovered_cohort):02d}] {nct_id} ({conditions[0][:25]}) -> Ready for Boxes 1-6")

    return discovered_cohort


def download_protocol_pdf(nct_id: str, pdf_url: str) -> Path:
    target = PROTOCOLS_DIR / f"REAL_{nct_id}_Protocol.pdf"
    if target.exists() and target.stat().st_size > 15_000:
        return target
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) TrialFrictionSim/1.0"}
    r = requests.get(pdf_url, headers=headers, timeout=60)
    r.raise_for_status()
    target.write_bytes(r.content)
    return target


def create_multilingual_stress_test_pdfs() -> List[Dict[str, Any]]:
    """
    Tier 3: 26-Language US FDA, EU EMA CTIS & Global Multilingual Protocol Stress Test
    (16 existing languages + 10 additional EU/Global languages = 26 languages).
    """
    specs = [
        {
            "lang_code": "EL_GREEK",
            "title": "Πρόγραμμα Αξιολογήσεων Μελέτης (Schedule of Assessments)",
            "headers": ["Διαδικασία / Εξέταση", "Διαλογή (Εβδ -2)", "Έναρξη (Εβδ 1)", "Εβδ 4", "Εβδ 8", "Εβδ 12", "Εβδ 16 to 24 (q4w)", "Εβδ 52"],
            "rows": [
                ["Ενημερωμένη Συγκατάθεση", "X", "", "", "", "", "", ""],
                ["Κλινική Εξέταση & Ζωτικά Σημεία", "X", "X", "X", "X", "X", "X", "X"],
                ["Γενική Αίματος & Βιοχημικός Έλεγχος", "X", "X", "X", "X", "X", "X", "X"],
                ["Λιπιδαιμικό Προφίλ Νηστείας", "X a", "X a", "", "X a", "X a", "X a", "X a"],
                ["Ηλεκτροκαρδιογράφημα (ΗΚΓ)", "X", "X", "", "", "X", "", "X"],
                ["Σειριακή Αιμοληψία Φαρμακοκινητικής (PK)", "", "X b", "X b", "", "X b", "", ""],
                ["Μαγνητική Τομογραφία (MRI)", "X", "", "", "", "X", "X", "X"],
                ["Βιοψία Ήπατος", "X c", "", "", "", "X c", "", "X c"],
                ["Οσφυονωτιαία Παρακέντηση", "", "X d", "", "", "X d", "", ""],
                ["Ερωτηματολόγιο Ποιότητας Ζωής (PRO)", "", "X", "X", "X", "X", "X", "X"],
            ],
        },
        {
            "lang_code": "DE_GERMAN",
            "title": "Visitenplan und Studienabläufe (Schedule of Assessments)",
            "headers": ["Untersuchung / Prozedur", "Screening (Wk -2)", "Baseline (Wk 1)", "Wk 4", "Wk 8", "Wk 12", "Wk 16 to 24 (q4w)", "Wk 52"],
            "rows": [
                ["Einverständniserklärung", "X", "", "", "", "", "", ""],
                ["Körperliche Untersuchung & Vitalzeichen", "X", "X", "X", "X", "X", "X", "X"],
                ["Blutentnahme & Hämatologie", "X", "X", "X", "X", "X", "X", "X"],
                ["Nüchternblut Lipidprofil", "X a", "X a", "", "X a", "X a", "X a", "X a"],
                ["12-Kanal-EKG (Elektrokardiogramm)", "X", "X", "", "", "X", "", "X"],
                ["Serielle PK Pharmakokinetik", "", "X b", "X b", "", "X b", "", ""],
                ["MRT Computertomographie", "X", "", "", "", "X", "X", "X"],
                ["Leberbiopsie", "X c", "", "", "", "X c", "", "X c"],
                ["Lumbalpunktion", "", "X d", "", "", "X d", "", ""],
                ["Fragebogen Lebensqualität", "", "X", "X", "X", "X", "X", "X"],
            ],
        },
        {
            "lang_code": "FR_FRENCH",
            "title": "Calendrier des Évaluations de l'Étude",
            "headers": ["Procédure / Évaluation", "Screening (Wk -2)", "Baseline (Wk 1)", "Wk 4", "Wk 8", "Wk 12", "Wk 16 to 24 (q4w)", "Wk 52"],
            "rows": [
                ["Consentement éclairé", "X", "", "", "", "", "", ""],
                ["Examen physique & Signes vitaux", "X", "X", "X", "X", "X", "X", "X"],
                ["Prise de sang & Hématologie", "X", "X", "X", "X", "X", "X", "X"],
                ["Bilan lipidique à jeun", "X a", "X a", "", "X a", "X a", "X a", "X a"],
                ["Électrocardiogramme (ECG 12 dérivations)", "X", "X", "", "", "X", "", "X"],
                ["Pharmacocinétique (Échantillonnage PK)", "", "X b", "X b", "", "X b", "", ""],
                ["IRM Tomodensitométrie", "X", "", "", "", "X", "X", "X"],
                ["Biopsie hépatique", "X c", "", "", "", "X c", "", "X c"],
                ["Ponction lombaire", "", "X d", "", "", "X d", "", ""],
                ["Questionnaire Qualité de Vie", "", "X", "X", "X", "X", "X", "X"],
            ],
        },
        {
            "lang_code": "ES_SPANISH",
            "title": "Calendario de Evaluaciones del Estudio",
            "headers": ["Procedimiento / Evaluación", "Screening (Wk -2)", "Baseline (Wk 1)", "Wk 4", "Wk 8", "Wk 12", "Wk 16 to 24 (q4w)", "Wk 52"],
            "rows": [
                ["Consentimiento informado", "X", "", "", "", "", "", ""],
                ["Exploración física & Signos vitales", "X", "X", "X", "X", "X", "X", "X"],
                ["Extracción de sangre & Hematología", "X", "X", "X", "X", "X", "X", "X"],
                ["Perfil lipídico en ayunas", "X a", "X a", "", "X a", "X a", "X a", "X a"],
                ["Electrocardiograma (ECG)", "X", "X", "", "", "X", "", "X"],
                ["Farmacocinética (Muestreo PK)", "", "X b", "X b", "", "X b", "", ""],
                ["Resonancia magnética (MRI)", "X", "", "", "", "X", "X", "X"],
                ["Biopsia hepática", "X c", "", "", "", "X c", "", "X c"],
                ["Punción lumbar", "", "X d", "", "", "X d", "", ""],
                ["Cuestionario Calidad de Vida", "", "X", "X", "X", "X", "X", "X"],
            ],
        },
        {
            "lang_code": "IT_ITALIAN",
            "title": "Calendario delle Valutazioni dello Studio",
            "headers": ["Procedura / Valutazione", "Screening (Wk -2)", "Baseline (Wk 1)", "Wk 4", "Wk 8", "Wk 12", "Wk 16 to 24 (q4w)", "Wk 52"],
            "rows": [
                ["Consenso informato", "X", "", "", "", "", "", ""],
                ["Esame obiettivo & Segni vitali", "X", "X", "X", "X", "X", "X", "X"],
                ["Prelievo di sangue & Ematologia", "X", "X", "X", "X", "X", "X", "X"],
                ["Profilo lipidico a digiuno", "X a", "X a", "", "X a", "X a", "X a", "X a"],
                ["Elettrocardiogramma (ECG a 12 derivazioni)", "X", "X", "", "", "X", "", "X"],
                ["Farmacocinetica (Campionamento PK)", "", "X b", "X b", "", "X b", "", ""],
                ["Risonanza magnetica (MRI)", "X", "", "", "", "X", "X", "X"],
                ["Biopsia epatica", "X c", "", "", "", "X c", "", "X c"],
                ["Puntura lombare", "", "X d", "", "", "X d", "", ""],
                ["Questionario Qualità della Vita", "", "X", "X", "X", "X", "X", "X"],
            ],
        },
        {
            "lang_code": "PT_PORTUGUESE",
            "title": "Calendário de Avaliações do Estudo",
            "headers": ["Procedimento / Avaliação", "Screening (Wk -2)", "Baseline (Wk 1)", "Wk 4", "Wk 8", "Wk 12", "Wk 16 to 24 (q4w)", "Wk 52"],
            "rows": [
                ["Consentimento informado", "X", "", "", "", "", "", ""],
                ["Exame físico & Sinais vitais", "X", "X", "X", "X", "X", "X", "X"],
                ["Colheita de sangue & Hematologia", "X", "X", "X", "X", "X", "X", "X"],
                ["Perfil lipídico em jejum", "X a", "X a", "", "X a", "X a", "X a", "X a"],
                ["Eletrocardiograma (ECG de 12 derivações)", "X", "X", "", "", "X", "", "X"],
                ["Farmacocinética (Amostragem PK)", "", "X b", "X b", "", "X b", "", ""],
                ["Ressonância magnética (MRI)", "X", "", "", "", "X", "X", "X"],
                ["Biópsia hepática", "X c", "", "", "", "X c", "", "X c"],
                ["Punção lombar", "", "X d", "", "", "X d", "", ""],
                ["Questionário Qualidade de Vida", "", "X", "X", "X", "X", "X", "X"],
            ],
        },
        {
            "lang_code": "NL_DUTCH",
            "title": "Schema van Studiebeoordelingen",
            "headers": ["Procedure / Beoordeling", "Screening (Wk -2)", "Baseline (Wk 1)", "Wk 4", "Wk 8", "Wk 12", "Wk 16 to 24 (q4w)", "Wk 52"],
            "rows": [
                ["Geïnformeerde toestemming", "X", "", "", "", "", "", ""],
                ["Lichamelijk onderzoek & Vitale functies", "X", "X", "X", "X", "X", "X", "X"],
                ["Bloedafname & Hematologie", "X", "X", "X", "X", "X", "X", "X"],
                ["Nuchter lipidenprofiel", "X a", "X a", "", "X a", "X a", "X a", "X a"],
                ["Elektrocardiogram (12-afleidingen-ECG)", "X", "X", "", "", "X", "", "X"],
                ["Farmacokinetiek (Seriële PK)", "", "X b", "X b", "", "X b", "", ""],
                ["Magnetische resonantie (MRI)", "X", "", "", "", "X", "X", "X"],
                ["Leverbiopsie", "X c", "", "", "", "X c", "", "X c"],
                ["Lumbaalpunctie", "", "X d", "", "", "X d", "", ""],
                ["Vragenlijst Kwaliteit van Leven", "", "X", "X", "X", "X", "X", "X"],
            ],
        },
        {
            "lang_code": "PL_POLISH",
            "title": "Harmonogram Procedur i Ocen Badania (Schedule of Assessments)",
            "headers": ["Procedura / Ocena", "Screening (Wk -2)", "Baseline (Wk 1)", "Wk 4", "Wk 8", "Wk 12", "Wk 16 to 24 (q4w)", "Wk 52"],
            "rows": [
                ["Świadoma zgoda", "X", "", "", "", "", "", ""],
                ["Badanie fizykalne & Parametry życiowe", "X", "X", "X", "X", "X", "X", "X"],
                ["Pobranie krwi & Morfologia krwi", "X", "X", "X", "X", "X", "X", "X"],
                ["Profil lipidowy na czczo", "X a", "X a", "", "X a", "X a", "X a", "X a"],
                ["Elektrokardiogram (12-odprowadzeniowe EKG)", "X", "X", "", "", "X", "", "X"],
                ["Farmakokinetyka (PK)", "", "X b", "X b", "", "X b", "", ""],
                ["Rezonans magnetyczny (MRI)", "X", "", "", "", "X", "X", "X"],
                ["Biopsja wątroby", "X c", "", "", "", "X c", "", "X c"],
                ["Nakłucie lędźwiowe", "", "X d", "", "", "X d", "", ""],
                ["Kwestionariusz Jakość Życia", "", "X", "X", "X", "X", "X", "X"],
            ],
        },
        {
            "lang_code": "SV_SWEDISH",
            "title": "Schema över Studiebedömningar (Schedule of Assessments)",
            "headers": ["Procedur / Bedömning", "Screening (Wk -2)", "Baseline (Wk 1)", "Wk 4", "Wk 8", "Wk 12", "Wk 16 to 24 (q4w)", "Wk 52"],
            "rows": [
                ["Informerat samtycke", "X", "", "", "", "", "", ""],
                ["Kroppsundersökning & Vitalparametrar", "X", "X", "X", "X", "X", "X", "X"],
                ["Blodprov & Hematologi", "X", "X", "X", "X", "X", "X", "X"],
                ["Fastande lipidprofil", "X a", "X a", "", "X a", "X a", "X a", "X a"],
                ["Elektrokardiogram (EKG)", "X", "X", "", "", "X", "", "X"],
                ["Farmakokinetik (PK)", "", "X b", "X b", "", "X b", "", ""],
                ["Magnetkameraundersökning (MRI)", "X", "", "", "", "X", "X", "X"],
                ["Leverbiopsi", "X c", "", "", "", "X c", "", "X c"],
                ["Lumbalpunktion", "", "X d", "", "", "X d", "", ""],
                ["Frågeformulär Livskvalitet", "", "X", "X", "X", "X", "X", "X"],
            ],
        },
        {
            "lang_code": "DA_DANISH",
            "title": "Skema over Undersøgelser og Vurderinger (Schedule of Assessments)",
            "headers": ["Procedure / Vurdering", "Screening (Wk -2)", "Baseline (Wk 1)", "Wk 4", "Wk 8", "Wk 12", "Wk 16 to 24 (q4w)", "Wk 52"],
            "rows": [
                ["Informeret samtykke", "X", "", "", "", "", "", ""],
                ["Fysisk undersøgelse & Vitale tegn", "X", "X", "X", "X", "X", "X", "X"],
                ["Blodprøve & Hæmatologi", "X", "X", "X", "X", "X", "X", "X"],
                ["Fastende lipidprofil", "X a", "X a", "", "X a", "X a", "X a", "X a"],
                ["Elektrokardiogram (EKG)", "X", "X", "", "", "X", "", "X"],
                ["Farmakokinetik (PK)", "", "X b", "X b", "", "X b", "", ""],
                ["MR-scanning (MRI)", "X", "", "", "", "X", "X", "X"],
                ["Leverbiopsi", "X c", "", "", "", "X c", "", "X c"],
                ["Lumbalpunktur", "", "X d", "", "", "X d", "", ""],
                ["Spørgeskema Livskvalitet", "", "X", "X", "X", "X", "X", "X"],
            ],
        },
        {
            "lang_code": "FI_FINNISH",
            "title": "Tutkimustoimenpiteiden Aikataulu (Schedule of Assessments)",
            "headers": ["Toimenpide / Arviointi", "Screening (Wk -2)", "Baseline (Wk 1)", "Wk 4", "Wk 8", "Wk 12", "Wk 16 to 24 (q4w)", "Wk 52"],
            "rows": [
                ["Tietoinen suostumus", "X", "", "", "", "", "", ""],
                ["Lääkärintarkastus & Elintoiminnot", "X", "X", "X", "X", "X", "X", "X"],
                ["Verikoe & Hematologia", "X", "X", "X", "X", "X", "X", "X"],
                ["Paastolipidiprofiili", "X a", "X a", "", "X a", "X a", "X a", "X a"],
                ["Sydänfilmi (EKG)", "X", "X", "", "", "X", "", "X"],
                ["Farmakokinetiikka (PK)", "", "X b", "X b", "", "X b", "", ""],
                ["Magneettikuvaus (MRI)", "X", "", "", "", "X", "X", "X"],
                ["Maksabiopsia", "X c", "", "", "", "X c", "", "X c"],
                ["Lannepisto", "", "X d", "", "", "X d", "", ""],
                ["Kyselylomake Elämänlaatu", "", "X", "X", "X", "X", "X", "X"],
            ],
        },
        {
            "lang_code": "CS_CZECH",
            "title": "Harmonogram Hodnocení Studie (Schedule of Assessments)",
            "headers": ["Procedura / Hodnocení", "Screening (Wk -2)", "Baseline (Wk 1)", "Wk 4", "Wk 8", "Wk 12", "Wk 16 to 24 (q4w)", "Wk 52"],
            "rows": [
                ["Informovaný souhlas", "X", "", "", "", "", "", ""],
                ["Fyzikální vyšetření & Vitální funkce", "X", "X", "X", "X", "X", "X", "X"],
                ["Odběr krve & Hematologie", "X", "X", "X", "X", "X", "X", "X"],
                ["Lipidový profil nalačno", "X a", "X a", "", "X a", "X a", "X a", "X a"],
                ["Elektrokardiogram (EKG)", "X", "X", "", "", "X", "", "X"],
                ["Farmakokinetika (PK)", "", "X b", "X b", "", "X b", "", ""],
                ["Magnetická rezonance (MRI)", "X", "", "", "", "X", "X", "X"],
                ["Jaterní biopsie", "X c", "", "", "", "X c", "", "X c"],
                ["Lumbální punkce", "", "X d", "", "", "X d", "", ""],
                ["Dotazník Kvalita Života", "", "X", "X", "X", "X", "X", "X"],
            ],
        },
        {
            "lang_code": "HU_HUNGARIAN",
            "title": "Vizsgálati Eljárások Ütemterve (Schedule of Assessments)",
            "headers": ["Eljárás / Értékelés", "Screening (Wk -2)", "Baseline (Wk 1)", "Wk 4", "Wk 8", "Wk 12", "Wk 16 to 24 (q4w)", "Wk 52"],
            "rows": [
                ["Tájékoztatáson alapuló beleegyezés", "X", "", "", "", "", "", ""],
                ["Fizikális vizsgálat & Életjelek", "X", "X", "X", "X", "X", "X", "X"],
                ["Vérvétel & Hematológia", "X", "X", "X", "X", "X", "X", "X"],
                ["Éhgyomri lipidprofil", "X a", "X a", "", "X a", "X a", "X a", "X a"],
                ["Elektrokardiográfia (EKG)", "X", "X", "", "", "X", "", "X"],
                ["Farmakokinetika (PK)", "", "X b", "X b", "", "X b", "", ""],
                ["Mágneses rezonancia (MRI)", "X", "", "", "", "X", "X", "X"],
                ["Májbiopszia", "X c", "", "", "", "X c", "", "X c"],
                ["Gerinccsapolás", "", "X d", "", "", "X d", "", ""],
                ["Kérdőív Életminőség", "", "X", "X", "X", "X", "X", "X"],
            ],
        },
        {
            "lang_code": "RO_ROMANIAN",
            "title": "Calendarul Evaluărilor Studiului (Schedule of Assessments)",
            "headers": ["Procedură / Evaluare", "Screening (Wk -2)", "Baseline (Wk 1)", "Wk 4", "Wk 8", "Wk 12", "Wk 16 to 24 (q4w)", "Wk 52"],
            "rows": [
                ["Consimțământ informat", "X", "", "", "", "", "", ""],
                ["Examen fizic & Semne vitale", "X", "X", "X", "X", "X", "X", "X"],
                ["Recoltare de sânge & Hematologie", "X", "X", "X", "X", "X", "X", "X"],
                ["Profil lipidic à jeun", "X a", "X a", "", "X a", "X a", "X a", "X a"],
                ["Electrocardiogramă (ECG)", "X", "X", "", "", "X", "", "X"],
                ["Farmacocinetică (PK)", "", "X b", "X b", "", "X b", "", ""],
                ["Imagistică prin rezonanță magnetică (MRI)", "X", "", "", "", "X", "X", "X"],
                ["Biopsie hepatică", "X c", "", "", "", "X c", "", "X c"],
                ["Puncție lombară", "", "X d", "", "", "X d", "", ""],
                ["Chestionar Calitatea Vieții", "", "X", "X", "X", "X", "X", "X"],
            ],
        },
        {
            "lang_code": "TR_TURKISH",
            "title": "Çalışma Değerlendirmeleri Takvimi (Schedule of Assessments)",
            "headers": ["Prosedür / Değerlendirme", "Screening (Wk -2)", "Baseline (Wk 1)", "Wk 4", "Wk 8", "Wk 12", "Wk 16 to 24 (q4w)", "Wk 52"],
            "rows": [
                ["Bilgilendirilmiş onam", "X", "", "", "", "", "", ""],
                ["Fizik muayene & Yaşam bulguları", "X", "X", "X", "X", "X", "X", "X"],
                ["Kan alma & Hematoloji", "X", "X", "X", "X", "X", "X", "X"],
                ["Açlık lipid profili", "X a", "X a", "", "X a", "X a", "X a", "X a"],
                ["Elektrokardiyogram (EKG)", "X", "X", "", "", "X", "", "X"],
                ["Farmakokinetik örnekleme (PK)", "", "X b", "X b", "", "X b", "", ""],
                ["Manyetik rezonans (MRI)", "X", "", "", "", "X", "X", "X"],
                ["Karaciğer biyopsisi", "X c", "", "", "", "X c", "", "X c"],
                ["Lomber ponksiyon", "", "X d", "", "", "X d", "", ""],
                ["Anket Yaşam Kalitesi", "", "X", "X", "X", "X", "X", "X"],
            ],
        },
        {
            "lang_code": "EN_ENGLISH",
            "title": "Schedule of Assessments (SoA)",
            "headers": ["Procedure / Assessment", "Screening (Wk -2)", "Baseline (Wk 1)", "Wk 4", "Wk 8", "Wk 12", "Wk 16 to 24 (q4w)", "Wk 52"],
            "rows": [
                ["Informed Consent", "X", "", "", "", "", "", ""],
                ["Physical Exam & Vital Signs", "X", "X", "X", "X", "X", "X", "X"],
                ["Routine Safety Blood Draw (CBC/CMP)", "X", "X", "X", "X", "X", "X", "X"],
                ["Fasting Lipid & Metabolic Panel", "X a", "X a", "", "X a", "X a", "X a", "X a"],
                ["12-Lead Electrocardiogram (ECG)", "X", "X", "", "", "X", "", "X"],
                ["Serial 4-Hour PK Blood Sampling", "", "X b", "X b", "", "X b", "", ""],
                ["Contrast MRI-PDFF Liver Imaging", "X", "", "", "", "X", "X", "X"],
                ["Ultrasound-Guided Liver Biopsy", "X c", "", "", "", "X c", "", "X c"],
                ["Lumbar Puncture (Exploratory CSF)", "", "X d", "", "", "X d", "", ""],
                ["PRO Quality of Life Questionnaire", "", "X", "X", "X", "X", "X", "X"],
            ],
        },
        # 10 NEW LANGUAGES (#17 to #26):
        {
            "lang_code": "NO_NORWEGIAN",
            "title": "Tidsskjema for Studieundersøkelser (Schedule of Assessments)",
            "headers": ["Prosedyre / Vurdering", "Screening (Wk -2)", "Baseline (Wk 1)", "Wk 4", "Wk 8", "Wk 12", "Wk 16 to 24 (q4w)", "Wk 52"],
            "rows": [
                ["Informert samtykke", "X", "", "", "", "", "", ""],
                ["Fysisk undersøkelse & Vitale tegn", "X", "X", "X", "X", "X", "X", "X"],
                ["Blodprøve & Hematologi", "X", "X", "X", "X", "X", "X", "X"],
                ["Fastende lipidprofil", "X a", "X a", "", "X a", "X a", "X a", "X a"],
                ["Elektrokardiogram (EKG)", "X", "X", "", "", "X", "", "X"],
                ["Farmakokinetikk (Seriell PK)", "", "X b", "X b", "", "X b", "", ""],
                ["MR-undersøkelse (MRI)", "X", "", "", "", "X", "X", "X"],
                ["Leverbiopsi", "X c", "", "", "", "X c", "", "X c"],
                ["Lumbalpunksjon", "", "X d", "", "", "X d", "", ""],
                ["Spørreskjema Livskvalitet", "", "X", "X", "X", "X", "X", "X"],
            ],
        },
        {
            "lang_code": "BG_BULGARIAN",
            "title": "График на оценяванията в проучването (Schedule of Assessments)",
            "headers": ["Процедура / Оценка", "Screening (Wk -2)", "Baseline (Wk 1)", "Wk 4", "Wk 8", "Wk 12", "Wk 16 to 24 (q4w)", "Wk 52"],
            "rows": [
                ["Информирано съгласие", "X", "", "", "", "", "", ""],
                ["Физикален преглед и жизнени показатели", "X", "X", "X", "X", "X", "X", "X"],
                ["Вземане на кръв и хематология", "X", "X", "X", "X", "X", "X", "X"],
                ["Липиден профил на гладно", "X a", "X a", "", "X a", "X a", "X a", "X a"],
                ["Електрокардиограма (ECG)", "X", "X", "", "", "X", "", "X"],
                ["Фармакокинетика (PK sampling)", "", "X b", "X b", "", "X b", "", ""],
                ["Ядрено-магнитен резонанс (MRI)", "X", "", "", "", "X", "X", "X"],
                ["Чернодробна биопсия", "X c", "", "", "", "X c", "", "X c"],
                ["Лумбална пункция", "", "X d", "", "", "X d", "", ""],
                ["Въпросник за качество на живот", "", "X", "X", "X", "X", "X", "X"],
            ],
        },
        {
            "lang_code": "SK_SLOVAK",
            "title": "Harmonogram Hodnotení Štúdie (Schedule of Assessments)",
            "headers": ["Procedúra / Hodnotenie", "Screening (Wk -2)", "Baseline (Wk 1)", "Wk 4", "Wk 8", "Wk 12", "Wk 16 to 24 (q4w)", "Wk 52"],
            "rows": [
                ["Informovaný súhlas", "X", "", "", "", "", "", ""],
                ["Fyzikálne vyšetrenie & Vitálne funkcie", "X", "X", "X", "X", "X", "X", "X"],
                ["Odber krvi & Hematológia", "X", "X", "X", "X", "X", "X", "X"],
                ["Lipidový profil nalačno", "X a", "X a", "", "X a", "X a", "X a", "X a"],
                ["Elektrokardiogram (EKG)", "X", "X", "", "", "X", "", "X"],
                ["Farmakokinetika (PK)", "", "X b", "X b", "", "X b", "", ""],
                ["Magnetická rezonancia (MRI)", "X", "", "", "", "X", "X", "X"],
                ["Biopsia pečene", "X c", "", "", "", "X c", "", "X c"],
                ["Lumbálna punkcia", "", "X d", "", "", "X d", "", ""],
                ["Dotazník Kvalita života", "", "X", "X", "X", "X", "X", "X"],
            ],
        },
        {
            "lang_code": "HR_CROATIAN",
            "title": "Raspored Ispitivanja i Procjena Studije (Schedule of Assessments)",
            "headers": ["Postupak / Procjena", "Screening (Wk -2)", "Baseline (Wk 1)", "Wk 4", "Wk 8", "Wk 12", "Wk 16 to 24 (q4w)", "Wk 52"],
            "rows": [
                ["Informirani pristanak", "X", "", "", "", "", "", ""],
                ["Fizikalni pregled & Vitalni znakovi", "X", "X", "X", "X", "X", "X", "X"],
                ["Vađenje krvi & Hematologija", "X", "X", "X", "X", "X", "X", "X"],
                ["Lipidni profil natašte", "X a", "X a", "", "X a", "X a", "X a", "X a"],
                ["Elektrokardiogram (EKG)", "X", "X", "", "", "X", "", "X"],
                ["Farmakokinetika (PK)", "", "X b", "X b", "", "X b", "", ""],
                ["Magnetska rezonancija (MRI)", "X", "", "", "", "X", "X", "X"],
                ["Biopsija jetre", "X c", "", "", "", "X c", "", "X c"],
                ["Lumbalna punkcija", "", "X d", "", "", "X d", "", ""],
                ["Upitnik kvalitete života", "", "X", "X", "X", "X", "X", "X"],
            ],
        },
        {
            "lang_code": "SL_SLOVENIAN",
            "title": "Urnik Ocenjevanj in Postopkov Študije (Schedule of Assessments)",
            "headers": ["Postopek / Ocena", "Screening (Wk -2)", "Baseline (Wk 1)", "Wk 4", "Wk 8", "Wk 12", "Wk 16 to 24 (q4w)", "Wk 52"],
            "rows": [
                ["Informirano soglasje", "X", "", "", "", "", "", ""],
                ["Klinični pregled & Vitalni znaki", "X", "X", "X", "X", "X", "X", "X"],
                ["Odvzem krvi & Hematologija", "X", "X", "X", "X", "X", "X", "X"],
                ["Lipidni profil na tešče", "X a", "X a", "", "X a", "X a", "X a", "X a"],
                ["Elektrokardiogram (EKG)", "X", "X", "", "", "X", "", "X"],
                ["Farmakokinetika (PK)", "", "X b", "X b", "", "X b", "", ""],
                ["Magnetna resonanca (MRI)", "X", "", "", "", "X", "X", "X"],
                ["Biopsija jeter", "X c", "", "", "", "X c", "", "X c"],
                ["Lumbalna punkcija", "", "X d", "", "", "X d", "", ""],
                ["Vprašalnik kakovosti življenja", "", "X", "X", "X", "X", "X", "X"],
            ],
        },
        {
            "lang_code": "ET_ESTONIAN",
            "title": "Uuringu Protseduuride ja Hindamiste Ajakava (Schedule of Assessments)",
            "headers": ["Protseduur / Hindamine", "Screening (Wk -2)", "Baseline (Wk 1)", "Wk 4", "Wk 8", "Wk 12", "Wk 16 to 24 (q4w)", "Wk 52"],
            "rows": [
                ["Teadlik nõusolek", "X", "", "", "", "", "", ""],
                ["Füüsiline läbivaatus & Elulised näitajad", "X", "X", "X", "X", "X", "X", "X"],
                ["Vereproov & Hematoloogia", "X", "X", "X", "X", "X", "X", "X"],
                ["Tühja kõhu lipiidide profiil", "X a", "X a", "", "X a", "X a", "X a", "X a"],
                ["Elektrokardiogramm (EKG)", "X", "X", "", "", "X", "", "X"],
                ["Farmakokineetika (PK)", "", "X b", "X b", "", "X b", "", ""],
                ["Magnetresonantstomograafia (MRI)", "X", "", "", "", "X", "X", "X"],
                ["Maksabiopsia", "X c", "", "", "", "X c", "", "X c"],
                ["Lumbaalpunktsioon", "", "X d", "", "", "X d", "", ""],
                ["Elukvaliteedi küsimustik", "", "X", "X", "X", "X", "X", "X"],
            ],
        },
        {
            "lang_code": "LV_LATVIAN",
            "title": "Pētījuma Procedūru un Novērtējumu Grafiks (Schedule of Assessments)",
            "headers": ["Procedūra / Novērtējums", "Screening (Wk -2)", "Baseline (Wk 1)", "Wk 4", "Wk 8", "Wk 12", "Wk 16 to 24 (q4w)", "Wk 52"],
            "rows": [
                ["Informēta piekrišana", "X", "", "", "", "", "", ""],
                ["Fizikālā izmeklēšana & Dzīvības rādītāji", "X", "X", "X", "X", "X", "X", "X"],
                ["Asins analīze & Hematoloģija", "X", "X", "X", "X", "X", "X", "X"],
                ["Lipīdu profils tukšā dūšā", "X a", "X a", "", "X a", "X a", "X a", "X a"],
                ["Elektrokardiogramma (EKG)", "X", "X", "", "", "X", "", "X"],
                ["Farmakokinētika (PK)", "", "X b", "X b", "", "X b", "", ""],
                ["Magnētiskā rezonanse (MRI)", "X", "", "", "", "X", "X", "X"],
                ["Aknu biopsija", "X c", "", "", "", "X c", "", "X c"],
                ["Lumbālpunkcija", "", "X d", "", "", "X d", "", ""],
                ["Dzīves kvalitātes anketa", "", "X", "X", "X", "X", "X", "X"],
            ],
        },
        {
            "lang_code": "LT_LITHUANIAN",
            "title": "Tyrimo Procedūrų ir Vertinimų Grafikas (Schedule of Assessments)",
            "headers": ["Procedūra / Vertinimas", "Screening (Wk -2)", "Baseline (Wk 1)", "Wk 4", "Wk 8", "Wk 12", "Wk 16 to 24 (q4w)", "Wk 52"],
            "rows": [
                ["Informuoto asmens sutikimas", "X", "", "", "", "", "", ""],
                ["Fizinis ištyrimas & Gyvybiniai rodikliai", "X", "X", "X", "X", "X", "X", "X"],
                ["Kraujo tyrimas & Hematologija", "X", "X", "X", "X", "X", "X", "X"],
                ["Lipidų profilis nevalgius", "X a", "X a", "", "X a", "X a", "X a", "X a"],
                ["Elektrokardiograma (EKG)", "X", "X", "", "", "X", "", "X"],
                ["Farmakokinetika (PK)", "", "X b", "X b", "", "X b", "", ""],
                ["Magnetinio rezonanso tomografija (MRI)", "X", "", "", "", "X", "X", "X"],
                ["Kepenų biopsija", "X c", "", "", "", "X c", "", "X c"],
                ["Juosmens punkcija", "", "X d", "", "", "X d", "", ""],
                ["Gyvenimo kokybės klausimynas", "", "X", "X", "X", "X", "X", "X"],
            ],
        },
        {
            "lang_code": "ID_INDONESIAN",
            "title": "Jadwal Evaluasi dan Prosedur Studi (Schedule of Assessments)",
            "headers": ["Prosedur / Evaluasi", "Screening (Wk -2)", "Baseline (Wk 1)", "Wk 4", "Wk 8", "Wk 12", "Wk 16 to 24 (q4w)", "Wk 52"],
            "rows": [
                ["Persetujuan tindakan medis", "X", "", "", "", "", "", ""],
                ["Pemeriksaan fisik & Tanda vital", "X", "X", "X", "X", "X", "X", "X"],
                ["Pengambilan darah & Hematologi", "X", "X", "X", "X", "X", "X", "X"],
                ["Profil lipid puasa", "X a", "X a", "", "X a", "X a", "X a", "X a"],
                ["Elektrokardiogram (EKG)", "X", "X", "", "", "X", "", "X"],
                ["Farmakokinetik (PK sampling)", "", "X b", "X b", "", "X b", "", ""],
                ["Pencitraan resonansi magnetik (MRI)", "X", "", "", "", "X", "X", "X"],
                ["Biopsi hati", "X c", "", "", "", "X c", "", "X c"],
                ["Pungsi lumbal", "", "X d", "", "", "X d", "", ""],
                ["Kuesioner kualitas hidup", "", "X", "X", "X", "X", "X", "X"],
            ],
        },
        {
            "lang_code": "VI_VIETNAMESE",
            "title": "Lịch Trình Đánh Giá Nghiên Cứu (Schedule of Assessments)",
            "headers": ["Quy trình / Đánh giá", "Screening (Wk -2)", "Baseline (Wk 1)", "Wk 4", "Wk 8", "Wk 12", "Wk 16 to 24 (q4w)", "Wk 52"],
            "rows": [
                ["Chấp thuận tham gia nghiên cứu", "X", "", "", "", "", "", ""],
                ["Khám lâm sàng & Dấu hiệu sinh tồn", "X", "X", "X", "X", "X", "X", "X"],
                ["Lấy mẫu máu & Huyết học", "X", "X", "X", "X", "X", "X", "X"],
                ["Xét nghiệm lipid máu lúc đói", "X a", "X a", "", "X a", "X a", "X a", "X a"],
                ["Điện tâm đồ (ECG)", "X", "X", "", "", "X", "", "X"],
                ["Dược động học (PK sampling)", "", "X b", "X b", "", "X b", "", ""],
                ["Chụp cộng hưởng từ (MRI)", "X", "", "", "", "X", "X", "X"],
                ["Sinh thiết gan", "X c", "", "", "", "X c", "", "X c"],
                ["Chọc dò tủy sống", "", "X d", "", "", "X d", "", ""],
                ["Bảng câu hỏi chất lượng cuộc sống", "", "X", "X", "X", "X", "X", "X"],
            ],
        },
    ]

    arial_font = Path(r"C:\Windows\Fonts\arial.ttf")
    font_file = str(arial_font) if arial_font.exists() else None
    font_name = "F_UNI" if font_file else "helv"

    results = []
    for sp in specs:
        pdf_path = PROTOCOLS_DIR / f"MULTILINGUAL_{sp['lang_code']}_Protocol.pdf"
        doc = fitz.open()
        page = doc.new_page(width=612, height=792)
        if font_file:
            page.insert_font(fontname=font_name, fontfile=font_file)

        page.insert_text((40, 45), sp["title"], fontsize=11, fontname=font_name)
        x_cols = [40, 220, 275, 330, 370, 410, 455, 525, 575]
        y_top = 75
        row_h = 26

        for c_idx, h_text in enumerate(sp["headers"]):
            rect = fitz.Rect(x_cols[c_idx], y_top, x_cols[c_idx + 1], y_top + 32)
            page.draw_rect(rect)
            page.insert_textbox(rect, h_text, fontsize=7.5, fontname=font_name)

        y_curr = y_top + 32
        for r in sp["rows"]:
            for c_idx, cell_val in enumerate(r):
                rect = fitz.Rect(x_cols[c_idx], y_curr, x_cols[c_idx + 1], y_curr + row_h)
                page.draw_rect(rect)
                page.insert_textbox(rect, cell_val, fontsize=7.5, fontname=font_name)
            y_curr += row_h

        fn_text = (
            "a. Fasting required (12-hour overnight fasting).\n"
            "b. Serial PK 4.5-hour post-dose chair observation.\n"
            "c. Biopsy requires 6-hour post-procedure observation and caregiver driver.\n"
            "d. Optional exploratory CSF tap."
        )
        page.insert_textbox(fitz.Rect(40, y_curr + 12, 570, y_curr + 95), fn_text, fontsize=8, fontname=font_name)
        doc.save(str(pdf_path))
        doc.close()

        run_table_extractor(pdf_path)
        cache_dir = TEMP_CACHE_DIR / pdf_path.stem
        bundle = run_full_pareto_optimization(cache_dir, generate_llm_memo=False)
        usdm_data = json.loads((cache_dir / "clean_usdm_schedule.json").read_text(encoding="utf-8"))

        results.append(
            {
                "language": sp["lang_code"],
                "expected_procs": 10,
                "extracted_procs": len(usdm_data.get("normalized_procedures", [])),
                "expected_visits": 9,
                "expanded_visits": len(usdm_data.get("expanded_timeline_visits", [])),
                "base_ret_pct": bundle["baseline_simulation"]["final_week_retention_pct"],
                "opt_ret_pct": bundle["optimized_simulation"]["final_week_retention_pct"],
                "gain_pct": bundle["retention_gain_pct"],
                "hrs_saved": bundle["chair_hours_saved_per_patient"],
            }
        )
    return results


def run_verified_stress_test() -> None:
    print("================================================================================")
    print("  TIER 1: MACRO CALIBRATION ON 38,302 REAL CLINICAL TRIALS (TRIALBENCH + DUCKDB)")
    print("================================================================================")
    df_tb = query_trialbench_38k_summary()
    if not df_tb.empty:
        print(df_tb.to_string(index=False))

    print("\n================================================================================")
    print("  TIER 2: EXPANDED 40-TRIAL CLINICALTRIALS.GOV COHORT & ABSTENTION BENCHMARK")
    print("================================================================================")
    cohort = discover_real_protocols_from_clinicaltrials_api(target_count=40)
    cohort_results: List[Dict[str, Any]] = []
    abstained_protocols: List[Dict[str, Any]] = []

    for study in cohort:
        nct_id = study["nct_id"]
        pdf_path = study["pdf_path"]
        try:
            soa_extracted = run_table_extractor(pdf_path)
            val_passed = bool(soa_extracted.get("validation_passed", False))
            conf_pct = soa_extracted.get("extraction_confidence_audit", {}).get("extraction_confidence_pct", 0.0)
            ext_mode = soa_extracted.get("extraction_mode", "PATH_A").replace("PATH_", "")[:12]

            cache_dir = TEMP_CACHE_DIR / pdf_path.stem
            manifest_path = cache_dir / "box1_sieve_manifest.json"
            front_synopsis = ""
            if manifest_path.exists():
                try:
                    front_synopsis = json.loads(manifest_path.read_text(encoding="utf-8")).get("front_matter_synopsis_text", "")
                except Exception:
                    pass

            clinical_domain = classify_clinical_domain_from_condition(
                condition=study["condition"],
                phase=study["phase"],
                brief_title=study["brief_title"],
                synopsis_fallback=front_synopsis[:600],
            )

            if not val_passed:
                # Remove any stale simulation/pareto report if a protocol abstains
                for stale_p in [
                    SAVED_REPORTS_DIR / f"{pdf_path.stem}_pareto_report.json",
                    cache_dir / "box6_pareto_comparison.json",
                ]:
                    stale_p.unlink(missing_ok=True)
                abstained_protocols.append(
                    {
                        "nct_id": nct_id,
                        "condition": study["condition"][:22],
                        "status": "ABSTAINED_NEEDS_HUMAN_SOA_REVIEW",
                        "heuristic_score": conf_pct,
                        "reason": "No machine-verifiable SoA grid (>2 visits & >=3 procedures) in PDF.",
                    }
                )
                print(
                    f"[Selective Abstention Gate] {nct_id} ({study['condition'][:20]}): "
                    f"Flagged for Human Review (Score: {conf_pct}/100) — Skipping blind simulation."
                )
                continue

            bundle = run_full_pareto_optimization(
                cache_dir,
                therapeutic_area=clinical_domain,
                generate_llm_memo=False,
            )

            sim_base_ret = bundle["baseline_simulation"]["final_week_retention_pct"]
            sim_opt_ret = bundle["optimized_simulation"]["final_week_retention_pct"]
            actual_ret = study["actual_retention_pct"]
            abs_err = round(abs(sim_base_ret - actual_ret), 2)

            usdm_data = json.loads((cache_dir / "clean_usdm_schedule.json").read_text(encoding="utf-8"))

            cohort_results.append(
                {
                    "nct_id": nct_id,
                    "condition": study["condition"][:18],
                    "domain": bundle["baseline_simulation"]["therapeutic_area"][:16],
                    "mode": ext_mode,
                    "conf": conf_pct,
                    "visits": len(usdm_data.get("expanded_timeline_visits", [])),
                    "procs": len(usdm_data.get("normalized_procedures", [])),
                    "actual_ret": actual_ret,
                    "sim_base": sim_base_ret,
                    "abs_err": abs_err,
                    "sim_opt": sim_opt_ret,
                    "hrs_saved": bundle["chair_hours_saved_per_patient"],
                }
            )
        except Exception as err:
            print(f"[Tier 2 Notice] {nct_id} encountered notice: {err}")

    df_cohort = pd.DataFrame(cohort_results)
    print("\n================ STRUCTURALLY VALIDATED REAL PROTOCOL PDFs TABLE ================")
    if not df_cohort.empty:
        print(df_cohort.to_string(index=False))
        mae = round(float(df_cohort["abs_err"].mean()), 2)
        rmse = round(float(np.sqrt((df_cohort["abs_err"] ** 2).mean())), 2)
        high_conf_df = df_cohort[df_cohort["conf"] >= 70.0]
        hc_mae = round(float(high_conf_df["abs_err"].mean()), 2) if not high_conf_df.empty else mae
        print(
            f"\n[SELECTIVE PREDICTION SCORECARD] Total Cohort Tested: {len(cohort)} Real PDFs | "
            f"Autonomous Simulated: {len(df_cohort)} | Selective Abstentions (Routed to UI Review): {len(abstained_protocols)}\n"
            f" • Overall Autonomous MAE: {mae}% (RMSE: {rmse}%) | High-Confidence Tier A MAE: {hc_mae}%"
        )

    if abstained_protocols:
        print("\n================ SELECTIVE ABSTENTION LOG (ROUTED TO CLINICIAN UI) ================")
        print(pd.DataFrame(abstained_protocols).to_string(index=False))

    print("\n================================================================================")
    print("  TIER 3: 26-LANGUAGE US/EU/GLOBAL MULTILINGUAL STRESS TEST")
    print("================================================================================")
    ml_results = create_multilingual_stress_test_pdfs()
    df_ml = pd.DataFrame(ml_results)
    print(df_ml.to_string(index=False))

    master_validation = {
        "tier1_trialbench_38k_summary": df_tb.to_dict(orient="records") if not df_tb.empty else [],
        "tier2_autonomous_validated_cohort": cohort_results,
        "tier2_selective_abstention_log": abstained_protocols,
        "tier3_multilingual_26lang_stress_test": ml_results,
    }
    val_path = SAVED_REPORTS_DIR / "MASTER_REAL_WORLD_VALIDATION_BENCHMARK.json"
    val_path.write_text(json.dumps(master_validation, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[COMPLETE] Saved Master Validation Asset to Google Drive: {val_path}")


if __name__ == "__main__":
    run_verified_stress_test()