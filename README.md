# Trial-Friction-Sim: Clinical Trial SoA Burden & Competing-Risks Dropout Simulator

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-2563eb.svg)](https://www.python.org/)
[![Local Hardware](https://img.shields.io/badge/VRAM_Budget-%E2%89%A48GB_Local_GPU_Airlock-059669.svg)](#hardware-discipline--8-gb-vram-gpu-airlock)
[![Regulatory Alignment](https://img.shields.io/badge/Standards-FDA_%26_EMA_ICH_E6(R3)_%7C_ICH_M11_%7C_CDISC_USDM_v4.1-7c3aed.svg)](#regulatory-alignment-us-fda--eu-ema-20252026)
[![Real-World Benchmark](https://img.shields.io/badge/Validated_Cohort-60_Real_ClinicalTrials.gov_PDFs_(2.60%25_MAE)-d97706.svg)](#empirical-validation--60-trial-real-world-benchmark)
[![Macro Calibration](https://img.shields.io/badge/Macro_Calibration-38%2C302_Real_Trials_(DuckDB)-0891b2.svg)](#tier-1-macro-calibration-on-38302-historical-trials)
[![Multilingual Parity](https://img.shields.io/badge/Multilingual-26_US_%2F_EU_%2F_Global_Languages-dc2626.svg)](#tier-4-26-language-useuglobal-stress-test)

**Trial-Friction-Sim** is a 100% local, privacy-preserving, 6-box clinical trial protocol intelligence and competing-risks patient retention simulator engineered to run within an **8 GB VRAM consumer GPU envelope**. Given any raw Sponsor Protocol PDF (from 2-page summaries to 483-page multi-epoch Phase III documents), the pipeline:

1. **Isolates** the Schedule of Assessments (SoA) pages on CPU (`0.0 GB` VRAM) via topological grid scoring (`Box 1`),
2. **Extracts & fuses** multi-page and split-epoch SoA tables via a self-healing tri-modal tournament (`Path A` deterministic vector grid $\rightarrow$ `Path B` local `qwen3.5:4b` VLM OCR $\rightarrow$ `Path C` prose synthesis) with a **3-Tier Epistemic Readiness & Selective Abstention Gate** (`Box 2`),
3. **Normalizes** complex visit headers and clinical procedures across **26 languages** into **CDISC USDM v4.1**, **ICH M11 CeSHarP**, **LOINC v2.78**, and **SNOMED-CT** Biomedical Concepts using a two-tier deterministic + self-learning local LLM normalizer (`Box 3`),
4. **Quantifies** 5-dimensional patient friction, log-subadditive intra-concept assessment stacking, and estimated site cost per calendar visit (`Box 4`),
5. **Simulates** 1,000 virtual patient-caregiver dyads week-by-week under competing risks (*Modifiable Logistical/Pain Burden Dropout* vs. *Unavoidable Adverse Event Dropout*), anchored by **38,302 real completed clinical trials** via zero-copy `DuckDB` SQL (`Box 5`), and
6. **Optimizes** the protocol schedule counterfactually under **US FDA & EU EMA ICH E6(R3) Section 3 (Quality by Design)**, locking primary efficacy and mandatory safety endpoints while quantifying rescued statistical power (`%`), saved clinic chair hours, and avoided **Tufts CSDD protocol amendment risk (`$ USD`)** (`Box 6`).

---

## Table of Contents

- [Why Trial-Friction-Sim? (The Clinical Problem)](#why-trial-friction-sim-the-clinical-problem)
- [Scientific Literature & Theoretical Foundations](#scientific-literature--theoretical-foundations)
- [Key Architectural Innovations](#key-architectural-innovations)
- [6-Box System Architecture & Data Flow](#6-box-system-architecture--data-flow)
- [Mathematical Formulation (Boxes 4, 5 & 6)](#mathematical-formulation-boxes-4-5--6)
- [Interactive Streamlit Studio: Step-by-Step User Manual](#interactive-streamlit-studio-step-by-step-user-manual)
- [Empirical Validation & 60-Trial Real-World Benchmark](#empirical-validation--60-trial-real-world-benchmark)
  - [Summary Scorecard (40 Calibration + 20 Blind Holdout = 60 Real Trials)](#summary-scorecard-40-calibration--20-blind-holdout--60-real-trials)
  - [Tier 1: Macro Calibration on 38,302 Historical Trials](#tier-1-macro-calibration-on-38302-historical-trials)
  - [Tier 2: 40-Trial Calibration Cohort (`2.47%` MAE | `0/67` Audit Flags)](#tier-2-40-trial-calibration-cohort-247-mae--067-audit-flags)
  - [Tier 3: 20-Trial Out-of-Sample Blind Holdout Benchmark (`2.88%` MAE)](#tier-3-20-trial-out-of-sample-blind-holdout-benchmark-288-mae)
  - [Tier 4: 26-Language US/EU/Global Stress Test (`100%` Parity)](#tier-4-26-language-useuglobal-stress-test-100-parity)
- [Patient Safety, Epistemic Humility & Human-in-the-Loop Guardrails](#patient-safety-epistemic-humility--human-in-the-loop-guardrails)
- [Regulatory Alignment (US FDA & EU EMA 2025β€“2026)](#regulatory-alignment-us-fda--eu-ema-20252026)
- [Repository Structure](#repository-structure)
- [Installation, Hardware Requirements & Quickstart](#installation-hardware-requirements--quickstart)

---

## Why Trial-Friction-Sim? (The Clinical Problem)

Across global Phase II and Phase III clinical trials, **~30% of enrolled participants drop out prior to study completion**, and **57% to 76% of protocols require at least one substantial protocol amendment**β€”costing sponsors **$141,000 (Phase II) to $535,000 (Phase III)** in direct implementation fees and adding **130 to 260 days** of operational delay.

Existing clinical trial AI benchmarks operate on high-level registry metadata (study phase, disease keyword, target enrollment size) while ignoring the primary operational driver of preventable participant attrition: **the week-by-week Schedule of Assessments (SoA) table locked inside 100-to-480 page Sponsor Protocol PDFs**. When a protocol stacks a 4.5-hour serial pharmacokinetic (PK) draw, a contrast MRI, an exploratory lumbar puncture or tissue biopsy, and 90 minutes of psychiatric/PRO questionnaires into the same visit week, employed patients and caregivers drop out from logistical and biophysical exhaustion.

**Trial-Friction-Sim** bridges document AI, regulatory informatics (`CDISC USDM v4.1` / `ICH M11`), and competing-risks survival modeling so clinical trial designers can diagnose visit-level dropout shocks and test **ICH E6(R3)** schedule simplifications *before* the first patient is enrolled.

---

## Scientific Literature & Theoretical Foundations

Every mathematical equation, burden weight, and macro-calibration prior in `Trial-Friction-Sim` is grounded in peer-reviewed clinical trial research and biostatistical literature:

1. **Multi-Modal Clinical Trial Attrition Benchmarks (`Box 5A` β€” 38,302-Trial DuckDB Calibration):**
   - **Paper:** Chen, J., Hu, Y., Gao, C., Fu, T., Sun, J., *et al.* (2025). *"TrialBench: Multi-Modal AI-Ready Datasets for Clinical Trial Prediction."* **Scientific Data (Nature Portfolio)**, 12, 1564. [https://doi.org/10.1038/s41597-025-05680-8](https://doi.org/10.1038/s41597-025-05680-8) (`arXiv:2407.00631`).
   - **Application in Repo:** `aact_duckdb_loader.py` streams the `38,302` curated real-world clinical trial dropout records (`trialbench-patient-dropout`) via zero-copy `DuckDB` SQL to calibrate phase-specific and domain-specific baseline dropout hazards.
2. **Time Toxicity & Clinic Chair-Hour Burden (`Box 4` & `Box 6`):**
   - **Paper:** Gupta, A., Jensen, E. H., Virnig, B. A., & Booth, C. M. (2022). *"The 'Time Toxicity' of Cancer Treatment."* **Journal of Clinical Oncology (JCO)**, 40(15), 1611β€“1615. [https://doi.org/10.1200/JCO.21.02810](https://doi.org/10.1200/JCO.21.02810).
   - **Application in Repo:** `scorer.py` and `survival_sim.py` explicitly model **Time Toxicity**β€”quantifying overlap-adjusted clinic chair hours $H(w)$ plus round-trip patient commute hours $T_i(w)$ relative to employment inflexibility $J_i$.
3. **Protocol Complexity, Patient Burden & Substantial Amendment Cost (`Box 4` & `Box 6`):**
   - **Paper:** Getz, K. A., Stergiopoulos, S., Short, M., *et al.* (2016β€“2024). *"The Impact of Protocol Amendments on Clinical Trial Performance."* **Therapeutic Innovation & Regulatory Science (TIRS)**, 50(4), 436β€“441. [https://pubmed.ncbi.nlm.nih.gov/30227022/](https://pubmed.ncbi.nlm.nih.gov/30227022/).
   - **Application in Repo:** `compute_tufts_csdd_amendment_risk()` in `pareto_simplifier.py` models substantial amendment probability (`12%β€“94%`), expected direct implementation cost (`$141,000` Phase II to `$535,000` Phase III midpoint = `$338,000`), and expected implementation delay (`260` days).
4. **Competing-Risks Subdistribution Survival Modeling (`Box 5B`):**
   - **Paper:** Fine, J. P., & Gray, R. J. (1999). *"A Proportional Hazards Model for the Subdistribution of a Competing Risk."* **Journal of the American Statistical Association (JASA)**, 94(446), 496β€“509.
   - **Application in Repo:** `run_competing_risks_simulation()` in `survival_sim.py` decomposes trial attrition at each discrete visit week $w_k$ into two competing absorbing events: **Modifiable Burden Dropout** ($p_{i,k}^{(B)}$) and **Unavoidable Adverse Event / Toxicity Dropout** ($p_k^{(AE)}$).
5. **Aggregate Analysis of ClinicalTrials.gov (`AACT`) Participant Flow Schema (`Box 5A` & Validators):**
   - **Paper:** Tasneem, A., Aberle, L., Ananth, H., Califf, R. M., *et al.* (2012). *"The Database for Aggregate Analysis of ClinicalTrials.gov (AACT) and Subsequent Regrouping by Clinical Specialty."* **PLOS ONE**, 7(3), e33677.
   - **Application in Repo:** `real_data_validator.py` and `blind_holdout_validator.py` extract period-matched `participantFlowModule` (`STARTED` vs. `COMPLETED`) milestones from live `ClinicalTrials.gov` API v2 records.

---

## Key Architectural Innovations

1. **Multi-Cluster Split-Epoch SoA Fusion & 3-Tier Selective Abstention (`Box 1` & `Box 2`):**
   - Real-world protocols frequently split Screening/Baseline and Treatment/Follow-Up across disjoint tables or pages. `extractor.py` implements an **Epoch-Completeness Gate (`_is_epoch_complete_schedule`)** that detects pre-treatment-only sub-tables, scans Cluster #2 via zero-VRAM vector parsing, and fuses compatible adjacent tables (`_fuse_split_epoch_clusters`).
   - Rather than hallucinating on unparseable PDFs, Box 2 computes an explicit structural readiness audit (`Tier A >= 75.0`, `Tier B < 75.0`, `Tier C Selective Abstention`) and routes unverified schedules to an interactive clinician editor.
2. **Two-Tier Self-Learning Tri-Standard Ontology Normalizer (`CENT-Local` in `Box 3`):**
   - Maps raw clinical procedure strings across **26 languages** to **15 canonical Biomedical Concepts** (`1,051` curated synonyms) bound to **CDISC NCIt**, **LOINC v2.78**, **SNOMED-CT**, and **ICH M11** categories.
   - Any unseen procedure failing Tier-1 lexical matching triggers a constrained zero-shot local LLM classifier (`_classify_unmapped_procedure_with_llm` via `qwen2.5:7b-instruct`) that permanently caches learned mappings in `learned_concept_synonyms_cache.json`β€”achieving a **`0.07%` default fallback rate across 59 simulated real trials** (`58 / 59` trials at `0.00%` fallback).
3. **Log-Subadditive Intra-Concept Assessment Stacking (`Box 4`):**
   - Captures the cumulative fatigue of stacking multiple blood panels or multiple psychiatric questionnaires within the same visit week via $L_{\text{stack}}(w) = \sum_{c} [1 + 0.35 \ln(m_{w,c})]$, while tracking both unique concept count (`procedure_count`) and raw stacked row count (`raw_procedure_count`).
4. **Universal Continuous Competing-Risks Survival Law (`Box 5`):**
   - Replaces brittle disease-named `if/elif` rules with `compute_universal_effective_domain_shift()`, a continuous mathematical function of domain log-odds shift ($\Delta_d$), total visits ($K$), peak single-visit procedure density ($M_{\text{raw}}$), and peak biophysical pain ($P_{\max}$).
5. **Visit-Preserving Counterfactual Pareto Optimizer & Case-Based Vector Knowledge Base (`Box 6`):**
   - Applies 3 regulatory-safe **ICH E6(R3)** simplification levers while guaranteeing `week_proc_counts[wk] > 1` so simplifying a protocol never deletes a scheduled calendar visit.
   - Includes a zero-VRAM **Vector-Similarity Knowledge Base (`retrieve_nearest_kb_precedents`)** that queries all 60 verified `ClinicalTrials.gov` protocols in log-structural space $[\text{domain\_match}, \ln(1+K), \ln(1+P)]$ to display the Top-5 closest real-world trial precedents alongside every simulation.

---

## 6-Box System Architecture & Data Flow

```text
========================================================================================================================
                TRIAL-FRICTION-SIM: COMPLETE 6-BOX SYSTEM ARCHITECTURE & DATA FLOW (v3.1 β€” 2026)
========================================================================================================================

  β”β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”
  β”‚                                  INPUT LAYER: REAL CLINICAL PROTOCOL DOCUMENTS                                   β”‚
  β”‚  β€Ά Live API Stream : ClinicalTrials.gov REST API v2 (`ProvidedDocs` Sponsor PDFs + Period-Matched Flow Results)  β”‚
  β”‚  β€Ά Local Workspace : `./workspace/protocols_pdf/` (2-page to 483-page Sponsor Protocol PDFs)                     β”‚
  β”‚  β€Ά Empirical Scale : Validated on 60 Real Completed Protocols (2.60% Combined MAE | 2.50% Median | 3.08% RMSE)   β”‚
  β”‚                      - Development Cohort (N=40) : 2.47% MAE | 2.87% RMSE | 0.00% Fallback | 0/67 Audit Flags    β”‚
  β”‚                      - Blind Holdout (N=20)      : 2.88% MAE | 2.24% Median | 2.85% Tier A MAE (+1 Tier C Gate)  β”‚
  β”‚                      + 38,302 Historical Trials (TrialBench / AACT Parquet) + 26 Global & EU Languages           β”‚
  β””β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”¬β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”
                                                           β”‚
                                                           β–Ό
  β”β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”
  β”‚ BOX 1: CONTIGUOUS CLUSTER PAGE SIEVE & SYNOPSIS READER (`box1_page_sieve/sieve.py` β€” CPU | 0.0 GB VRAM)          β”‚
  β”‚ β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€ β”‚
  β”‚  1. Fast Page Scorer (`score_pdf_page`):                                                                         β”‚
  β”‚     β€Ά Rejects Table of Contents (`"..." > 2`) and pure prose Estimand / Inclusion / Exclusion pages.             β”‚
  β”‚     β€Ά Scans for `EXTENDED_SOA_HEADERS` ("schedule of activities", "study calendar", "assessment schedule").      β”‚
  β”‚     β€Ά Counts `TIME_HEADER_PATTERN` tokens (Week, Day, Cycle, Visit, Month, Screening, Baseline, EOS/EOT).        β”‚
  β”‚     β€Ά Counts standalone checkmarks (`X, β“, β”, β€Ά`) and tests `page.find_tables()` for ruled grids (+65.0 pts).   β”‚
  β”‚  2. Front-Matter Synopsis Extractor (`extract_protocol_synopsis_and_title_text`):                                β”‚
  β”‚     β€Ά Reads Pages 1β€“3 (`front_matter_synopsis_text`) so raw user-uploaded PDFs without API metadata still have   β”‚
  β”‚       their disease indication and study phase auto-detected.                                                    β”‚
  β”‚  3. Contiguous Cluster Isolator & Backtracking Memory (`run_page_sieve`):                                        β”‚
  β”‚     β€Ά Groups candidate pages into contiguous clusters (1 to 6 consecutive pages).                                β”‚
  β”‚     β€Ά Supports `excluded_pages_0based` so Box 2 can backtrack to Cluster #2 if Cluster #1 fails validation.      β”‚
  β””β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”¬β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”
                                                           β”‚
                                                           β–Ό
  β”β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”
  β”‚ BOX 2: SELF-HEALING TRI-MODAL EXTRACTOR & EPOCH FUSION (`box2_table_extractor/extractor.py`)                     β”‚
  β”‚ β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€ β”‚
  β”‚  β”β”€β”€β–Ί [MULTI-CLUSTER TOURNAMENT LOOP: Evaluates up to `MAX_PAGE_CLUSTERS = 2` Disjoint Page Clusters] β”€β”€β”€β”€β”€β”€β”€β”€β”  β”‚
  β”‚  β”‚                                                                                                            β”‚  β”‚
  β”‚  β”‚  β–Ί PATH A: Deterministic Ruled Vector Grid Parser (`_deterministic_grid_to_json` | 0.0 GB VRAM | n=34)     β”‚  β”‚
  β”‚  β”‚    β€Ά Extracts PyMuPDF vector tables (`find_tables`); carries headers across multi-page continuations.      β”‚  β”‚
  β”‚  β”‚    β€Ά Multi-Cluster Split-Epoch Fusion (`_is_epoch_complete_schedule` & `_fuse_split_epoch_clusters`):      β”‚  β”‚
  β”‚  β”‚      If Cluster #1 is a valid pre-treatment/screening sub-table (<4 visits), inspects Cluster #2 via       β”‚  β”‚
  β”‚  β”‚      0-VRAM Path A and fuses adjacent Screening + Treatment tables into a unified longitudinal schedule.   β”‚  β”‚
  β”‚  β”‚      β”β”€β”€β–Ί EPOCH-COMPLETE : Breaks loop immediately (`PATH_A_RULED_VECTOR_GRID` β€” 2.36% Dev / 2.99% Hold).  β”‚  β”‚
  β”‚  β”‚      β””β”€β”€β–Ί FAIL           : Saves candidate and triggers PATH B on current cluster's PNG images β”€β”€β–Ί         β”‚  β”‚
  β”‚  β”‚                                                                                                            β”‚  β”‚
  β”‚  β”‚  β–Ί PATH B: Local VLM Visual Image-to-Table OCR (`extract_table_from_image_with_vlm` | 3.4 GB VRAM | n=23)  β”‚  β”‚
  β”‚  β”‚    β€Ά Loads `qwen3.5:4b` in GPU Airlock Slot 1 (`think=False`, `seed=42`, SHA256-cached in `temp_cache`).   β”‚  β”‚
  β”‚  β”‚    β€Ά Self-Healing JSON Recovery (`_repair_truncated_json` via `JSONDecoder.raw_decode`).                   β”‚  β”‚
  β”‚  β”‚    β€Ά Immediately unloads `qwen3.5:4b` from 8 GB VRAM (`keep_alive=0`).                                     β”‚  β”‚
  β”‚  β”‚      β”β”€β”€β–Ί PASS : Breaks loop immediately (`PATH_B_VLM_VISUAL_IMAGE_OCR` β€” 2.50% Dev / 2.69% Holdout MAE).  β”‚  β”‚
  β”‚  β”‚      β””β”€β”€β–Ί FAIL : Adds cluster pages to `excluded_pages` and BACKTRACKS to Box 1 for Cluster #2! β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”  β”‚
  β”‚  β”‚                                                                                                               β”‚
  β”‚  β–Ό (If neither Cluster #1 nor Cluster #2 passed Validation Gate)                                                 β”‚
  β”‚  β–Ί PATH C: Prose Text-to-Table Synthesizer (`synthesize_table_from_prose_text_with_llm` | 4.7 GB VRAM | n=2)     β”‚
  β”‚    β€Ά Scans top 5 scheduling prose sections (`extract_scheduling_prose_from_pdf`).                                β”‚
  β”‚    β€Ά Loads `qwen2.5:7b-instruct` in GPU Airlock Slot 2 + Canonicalized Timepoint Lexical Co-Occurrence Fallback. β”‚
  β””β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”¬β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”
                                                           β”‚
                                                           β–Ό
  β”β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”
  β”‚ SELECTIVE ABSTENTION GATE & HUMAN-IN-THE-LOOP SCHEDULE EDITOR (Protecting Prospective / Unseen Trials)           β”‚
  β”‚ β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€ β”‚
  β”‚  β€Ά Tier A (`conf >= 75.0` & `validation_passed`, n=37, 2.62% MAE) -> Green Banner: Autonomous simulation.        β”‚
  β”‚  β€Ά Tier B (`conf <  75.0` & `validation_passed`, n=22, 2.58% MAE) -> Yellow Banner: Simulates + Schedule Editor. β”‚
  β”‚  β€Ά Tier C (`validation_passed == False`, n=1 `NCT03065075`)       -> Red Banner (`ABSTAINED_ROUTED_TO_UI`):      β”‚
  β”‚                                                                      Pauses blind simulation & opens UI Editor.  β”‚
  β””β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”¬β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”
                                                           β”‚
                                                           β–Ό
  β”β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”
  β”‚ BOX 3: 26-LANGUAGE TRI-STANDARD USDM v4.1 & CENT-LOCAL NORMALIZER (`box3_usdm_normalizer/normalizer.py`)         β”‚
  β”‚ β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€ β”‚
  β”‚  1. Stage 3A Timepoint Expander (`parse_and_expand_visit_header`):                                               β”‚
  β”‚     β€Ά Unpacks Slash-Lists (`18/20/24/28/32/36`), Negative Screening Windows (`-6 to -1`), Week Ranges (`4A`),    β”‚
  β”‚       Visit Ranges (`4B`), and Multi-Cycle Ranges (`4C`, e.g. `4 to 8 28-day cycles`).                           β”‚
  β”‚  2. Stage 3B Two-Tier Self-Learning Ontology Mapper (`map_procedure_to_cdisc_concept` + `CENT-Local`):           β”‚
  β”‚     β€Ά Tier 1: Deterministic lexical matching across 1,051 clinical & 26-language synonyms (`data_local/`).       β”‚
  β”‚     β€Ά Tier 2: Constrained zero-shot `qwen2.5:7b-instruct` normalizer cached in `learned_concept_synonyms_cache`. β”‚
  β””β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”¬β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”
                                                           β”‚
                                                           β–Ό
  β”β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”
  β”‚ BOX 4: 5D TUFTS CSDD BURDEN, STACKING LOAD & SITE COST SCORER (`box4_burden_scorer/scorer.py`)                   β”‚
  β”‚ β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€ β”‚
  β”‚  β€Ά Computes overlap-adjusted `total_chair_hours`, `effective_pain_0_10`, `fasting_required`,                     β”‚
  β”‚    `pro_cognitive_mins`, `caregiver_required`, `visit_site_cost_usd`, and Log-Subadditive Intra-Concept          β”‚
  β”‚    Assessment Stacking Load (`stacked_assessment_load = sum(1 + 0.35*ln(m_c))`).                                 β”‚
  β””β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”¬β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”
                                                           β”‚
                                                           β–Ό
  β”β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”
  β”‚ BOX 5: 38,302-TRIAL DUCKDB STREAM & UNIVERSAL SURVIVAL SIMULATOR (`box5_dropout_simulator/`)                     β”‚
  β”‚ β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€ β”‚
  β”‚  β€Ά Streams baseline hazards from `trialbench_38k_real_dropouts.parquet` (38,302 real completed trials).          β”‚
  β”‚  β€Ά Simulates 1,000 virtual patient-caregiver dyads on PyTorch tensors via `compute_universal_effective_domain_shift`.β”‚
  β””β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”¬β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”
                                                           β”‚
                                                           β–Ό
  β”β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”
  β”‚ BOX 6: VISIT-PRESERVING ICH E6(R3) PARETO SIMPLIFIER, VECTOR KB PRECEDENTS & TUFTS AMENDMENT AUDITOR             β”‚
  β”‚ β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€ β”‚
  β”‚  β€Ά `pareto_simplifier.py`: Locks Primary/Safety endpoints; applies 3 ICH E6(R3) Levers while guaranteeing        β”‚
  β”‚    `week_proc_counts[wk] > 1` so pruning never deletes a scheduled calendar visit.                               β”‚
  β”‚  β€Ά `app_dashboard.py`: Interactive Streamlit Studio with Schedule Editor, Plotly Survival & Time-Toxicity        β”‚
  β”‚    Charts, Top-5 Nearest ClinicalTrials.gov Precedents (Vector Knowledge Base), and Bilingual (EN/EL) AI Memo.   β”‚
  β””β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”
```

---

## Mathematical Formulation (Boxes 4, 5 & 6)

### 1. Overlap-Adjusted Clinic Chair Time & Biophysical Pain (`Box 4`)
For each discrete calendar visit week $w$ with deduplicated biomedical concepts $c \in \mathcal{C}_w$ (where each concept has effective chair hours $h_c$, invasiveness $p_c \in [0, 10]$, fasting indicator $f_c \in \{0,1\}$, cognitive minutes $q_c$, caregiver indicator $g_c \in \{0,1\}$, and multiplicity $m_{w,c} \ge 1$):

$$H(w) = h_{\max}(w) + 0.65 \sum_{c \in \mathcal{C}_w \setminus \{c_{\max}\}} h_c, \qquad P(w) = \min\Big(10.0,\; p_{\max}(w) + 0.60 \cdot \max\big(0, |\mathcal{C}_w^{\text{inv}}| - 1\big)\Big)$$

where $\mathcal{C}_w^{\text{inv}} = \{c \in \mathcal{C}_w : p_c \ge 4.0\}$. The composite **Visit Friction Index** $\text{VFI}(w) \in [0, 100]$ is:

$$\text{VFI}(w) = \min\!\left(100.0,\; 30 \min\!\left(\frac{H(w)}{8.0}, 1.2\right) + 35 \left(\frac{P(w)}{10.0}\right) + 10\,\mathbb{I}_{\text{fast}}(w) + 10 \min\!\left(\frac{Q(w)}{180}, 1.0\right) + 15\,\mathbb{I}_{\text{care}}(w)\right)$$

### 2. Competing-Risks Patient-Caregiver Survival Process (`Box 5`)
For virtual patient $i \in \{1, \dots, N\}$ ($N=1,000$) with age $A_i \sim \mathcal{N}(54, 12^2)_{[22,80]}$, job inflexibility $J_i$, log-normal commute distance $D_i \sim \text{LogNormal}(2.8, 0.75^2)_{[2,120]}$, pain aversion $\alpha_i \sim \text{Gamma}(2.5, 0.45)_{[0.4,2.2]}$, and caregiver availability $G_i \sim \text{Bernoulli}(0.68)$, the visit stress $S_i(w_k)$ and exponentially decaying cumulative fatigue $F_i(w_k)$ at visit $k \in \{1, \dots, K\}$ are:

$$S_i(w_k) = \left(\frac{H(w_k) + T_i(w_k)}{8.5}\right) J_i + \left(\frac{P(w_k)}{10.0}\right)^{1.8} \alpha_i + 0.55(1 - G_i)\mathbb{I}_{\text{care}}(w_k) + 0.12\,\mathbb{I}_{\text{fast}}(w_k)$$

$$F_i(w_k) = \min\!\Big(2.2,\; F_i(w_{k-1}) \exp\!\big(-0.12 \max(1, w_k - w_{k-1})\big) + 0.22\,S_i(w_k)\Big)$$

At each visit $w_k$, active patient $i$ faces two competing hazard processes:
1. **Modifiable Burden Dropout Probability** $p_{i,k}^{(B)} = \text{clip}\big(\sigma(\eta_{i,k}),\, 0.0005,\, 0.38\big)$, where:
   $$\eta_{i,k} = -4.85 + \delta_{\text{density}}(K) + \delta_{\text{inpatient}} + \Delta_{\text{eff}}(\Delta_d, K, M_{\text{raw}}, P_{\max}) + 0.58\,S_i(w_k) + 0.24\,F_i(w_k)$$
2. **Unavoidable Adverse Event (AE) Dropout Probability** $p_k^{(AE)} = 1 - (1 - r_{\text{AE, eff}})^{1/K}$, calibrated from `trialbench_38k_real_dropouts.parquet`.

---

## Interactive Streamlit Studio: Step-by-Step User Manual

Launch the interactive clinical protocol studio by running:
```powershell
streamlit run box6_optimizer_ui/app_dashboard.py
```

### What Happens When You Upload a Protocol PDF?

#### 1. Left Sidebar Controls (Input, Cohort & Optimization Levers)
* **Protocol Input:** Upload any Sponsor Protocol PDF (`Upload Any Protocol PDF β€” 26 Languages Supported`) or select any cached protocol from the dropdown menu.
* **Clinical Attrition Domain:** Leave on `AUTO_DETECT_FROM_PROTOCOL` (which reads the protocol's front-matter synopsis on Pages 1β€“3 and extracted procedures to classify the trial across 20 therapeutic domains) or manually select a specific domain to run sensitivity analysis.
* **Virtual Patient Cohort Size:** Slide between `500` and `5,000` virtual patient-caregiver dyads (`default = 1,000`).
* **ICH E6(R3) Optimization Levers:** Independently toggle:
  - **Lever 1:** Prune redundant interim exploratory biopsies / lumbar punctures & convert 4.5-hr Serial PK to 1-point Sparse PK.
  - **Lever 2:** Consolidate redundant interim MRIs & duplicate 120-min neurocognitive batteries.
  - **Lever 3:** Decentralize non-anchor routine check-in weeks to Home-Health Nursing / Mobile ePRO (`0 km` hospital commute).
* **Local AI Memo Settings:** Select `English` or `Greek (Ξ•Ξ»Ξ»Ξ·Ξ½ΞΉΞΊΞ¬)` and check `Generate Executive Memo via Local qwen2.5:7b-instruct` to generate a formal FDA/EMA regulatory justification memo.

#### 2. Top Epistemic Readiness Banner (3-Tier Traffic Light)
Immediately after extraction, the top banner displays the structural readiness tier of the uploaded PDF:
* πΆ **Tier A β€” Structurally Validated Schedule (`Score >= 75/100` | `PATH_A_RULED_VECTOR_GRID`):** High-confidence vector table extracted directly from the PDF; displays isolated 1-based page numbers, active visit count, procedure count, and detected clinical domain.
* π΅ **Tier B β€” Model-Reconstructed / Partial Schedule (`Score < 75/100` | `PATH_B_VLM` or `PATH_C_PROSE`):** Schedule reconstructed via local VLM OCR or prose synthesis; runs the simulation while recommending clinician verification via the expandable schedule editor.
* π”΄ **Tier C β€” Selective Abstention Triggered (`validation_passed == False`):** Triggered when a PDF contains no machine-verifiable SoA grid (`< 2` visits or `< 3` procedures). Automated blind simulation is paused, and the **Interactive Human-in-the-Loop Schedule Editor** opens automatically.

#### 3. Interactive Human-in-the-Loop Schedule Editor
Expanding **"Interactive Human-in-the-Loop Schedule Editor (Add / Edit Visits & Procedures)"** reveals:
* A text input for comma-separated **Study Visit Columns** (e.g., `Screening (Wk -2), Baseline (Wk 0), Wk 4, Wk 8, Wk 12, Follow-Up (Wk 24)`).
* An interactive spreadsheet (`st.data_editor`) where clinicians can add, edit, or remove procedures and their scheduled visits, then click **"Save Verified Schedule & Run Simulation"** (`95.0/100 CLINICIAN_VERIFIED_SCHEDULE`).

#### 4. Six Executive KPI Metric Cards
Across the top of the simulation workspace, 6 real-time KPI cards display:
1. **Baseline Cohort Retention (`%`):** Predicted end-of-study retention for the original protocol, with a red `FAILED (<80% Power)` or green `PASSED (>=80% Power)` badge.
2. **Optimized Cohort Retention (`%`):** Predicted retention after applying the selected ICH E6(R3) levers, showing `+{X}% Rescued` participants.
3. **Modifiable Burden Dropout (`%`):** Percentage of participants lost specifically to logistical/pain friction (excluding unavoidable adverse events) and the reduction achieved vs. baseline.
4. **Chair Hours Saved / Patient (`hrs`):** Total clinic chair hours eliminated per patient across the study timeline and total count of SoA modifications.
5. **Tufts Amendment Risk (`%`):** Predicted probability of requiring a substantial protocol amendment and the expected direct financial cost avoided (`$ USD`).
6. **Extraction Readiness Score (`0β€“100`):** Structural extraction score and active Box 2 extraction path.

#### 5. Dual Interactive Plotly Visualizations
* **Left Chart β€” Cohort Retention Curve (Baseline vs. Pareto-Optimized Protocol):** Plots week-by-week retained cohort percentage (`%`) for the original schedule (red curve) vs. the ICH E6(R3) optimized schedule (green curve) against the dashed orange **80% Minimum Statistical Power Threshold**.
* **Right Chart β€” Clinic Chair Hours per Visit (Time Toxicity Bar Chart):** Grouped bar chart comparing baseline vs. optimized clinic chair hours (`hrs`) at every scheduled visit week, visually pinpointing multi-procedure "dropout shock" weeks.

#### 6. Four Deep-Dive Audit & Regulatory Tabs
* **Tab 1 (`ICH E6(R3) Protocol Simplifications Log`):** Filterable table of every schedule modification (`week`, `lever`, `procedure [canonical concept]`, `action`, and `ich_e6_r3_principle`).
* **Tab 2 (`Visit-by-Visit Simulation Table β€” Before vs. After`):** Side-by-side weekly tables showing `week`, `visit_name`, `total_chair_hours`, `effective_pain_0_10`, `new_burden_dropouts`, and `retention_pct`.
* **Tab 3 (`Executive Regulatory Justification Memo`):** Displays the generated FDA & EMA ICH E6(R3) Executive Redesign Memo in English or Greek.
* **Tab 4 (`Knowledge Base Precedents & 60-Trial Validation Benchmark`):**
  1. **Top-5 Nearest Real-World `ClinicalTrials.gov` Precedents:** Queries the 60-trial Vector Knowledge Base in log-structural space $[\text{domain\_match}, \ln(1+K), \ln(1+P)]$ to display the 5 most structurally similar completed trials on `ClinicalTrials.gov` alongside their actual vs. simulated retention and vector similarity (`%`).
  2. **Verified 60-Trial Real-World Validation Scorecard:** Summary table across the 40-trial calibration cohort, 20-trial blind holdout cohort, and combined 60-trial benchmark.
  3. **Box 2 Multi-Cluster Extraction Attempts & Validation Gate Audit:** Full provenance log of every cluster and extraction path attempted on the PDF.
  4. **Tier 1 Macro Calibration Table:** Live DuckDB summary across all `38,302` real clinical trials in `trialbench_38k_real_dropouts.parquet`.

---

## Empirical Validation & 60-Trial Real-World Benchmark

### Summary Scorecard (40 Calibration + 20 Blind Holdout = 60 Real Trials)

| Validation Cohort (`ClinicalTrials.gov` Sponsor PDFs) | Protocols Tested | Simulated (`Tier A + B`) | Mean Abs Error (`MAE %`) | Std Dev (`%`) | Median Error (`%`) | Min / Max Error (`%`) | RMSE (`%`) | Default Fallback (`%`) | Trials $\le 6.0\%$ Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Development & Calibration Cohort (`N = 40`)** | `40` | `40 / 40` | **`2.47%`** | `1.48%` | **`2.51%`** | `0.06%` / `5.63%` | **`2.87%`** | **`0.00%`** | **`40 / 40` (`100.0%`)** |
| **2. Blind Holdout β€” Tier A (`conf >= 75.0`)** | `13` | `13 / 13` | **`2.85%`** | `1.87%` | **`2.51%`** | `0.01%` / `5.76%` | **`3.34%`** | `0.33%` | **`13 / 13` (`100.0%`)** |
| **3. Blind Holdout β€” Tier B (`conf < 75.0`)** | `6` | `6 / 6` | **`2.94%`** | `2.42%` | **`1.87%`** | `0.96%` / `7.03%` | **`3.65%`** | **`0.00%`** | `5 / 6` (`83.3%`) |
| **4. Blind Holdout β€” All (`19` Sim + `1` Tier C)** | `20` | `19 / 20` (+`1` Tier C) | **`2.88%`** | `1.99%` | **`2.24%`** | `0.01%` / `7.03%` | **`3.47%`** | **`0.23%`** | **`18 / 19` (`94.7%`)** |
| **5. Combined 60-Trial Benchmark (`N = 60`)** | **`60`** | **`59 / 60`** (+`1` Tier C) | **`2.60%`** | **`1.65%`** | **`2.50%`** | **`0.01%` / `7.03%`** | **`3.08%`** | **`0.07%`** | **`58 / 59` (`98.3%`)** |

---

### Tier 1: Macro Calibration on 38,302 Historical Trials

Executed via `box5_dropout_simulator/aact_duckdb_loader.py` on `trialbench_38k_real_dropouts.parquet`:

| Clinical Trial Phase | Real Completed Trials (`N`) | Mean All-Cause Dropout (`%`) | Median Dropout (`%`) | 75th Percentile High-Burden Dropout (`%`) |
| :--- | :---: | :---: | :---: | :---: |
| **Phase 1** | `4,204` | `18.15%` | `3.13%` | `16.67%` |
| **Phase 2** | `15,770` | `25.57%` | `11.11%` | `34.00%` |
| **Phase 3** | `11,461` | `23.72%` | `12.41%` | `29.35%` |
| **Phase 4** | `6,867` | `15.51%` | `7.69%` | `20.88%` |
| **Total / Weighted** | **`38,302`** | **`22.40%`** | β€” | β€” |

---

### Tier 2: 40-Trial Calibration Cohort (`2.47%` MAE | `0/67` Audit Flags)

Reproducible via `python real_data_validator.py` and `python forensic_folder_auditor.py`:

| # | NCT ID | Condition / Indication | Clinical Domain | Box 2 Mode | Score | Visits | Procs | Actual Ret (%) | Sim Base (%) | Abs Err (%) | Sim Opt (%) | Hrs Saved |
| :-: | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 01 | `NCT03895372` | Psoriasis | `AUTOIMMUNE_DERMATOLOGY` | `PATH_A_VECTOR` | `96.0` | `17` | `29` | `85.96%` | `87.6%` | **`1.64%`** | `88.4%` | `10.26` |
| 02 | `NCT04682405` | Multiple Myeloma | `INPATIENT_HEMATOLOGY` | `PATH_A_VECTOR` | `86.1` | `8` | `17` | `98.04%` | `95.7%` | **`2.34%`** | `97.6%` | `11.06` |
| 03 | `NCT03645096` | Major Depressive Disorder | `NEURO_PSYCHIATRY` | `PATH_B_VLM_OCR` | `66.6` | `5` | `9` | `82.76%` | `81.8%` | **`0.96%`** | `82.6%` | `0.40` |
| 04 | `NCT02224703` | Epilepsy | `PEDIATRIC_ACUTE` | `PATH_B_VLM_OCR` | `63.8` | `3` | `7` | `95.48%` | `97.8%` | **`2.32%`** | `97.8%` | `0.71` |
| 05 | `NCT05645354` | Smoking Cessation | `BEHAVIORAL_SMOKING` | `PATH_B_VLM_OCR` | `64.0` | `4` | `5` | `35.00%` | `35.7%` | **`0.70%`** | `36.2%` | `0.40` |
| 06 | `NCT05164705` | Healthy Brain Perfusion | `NEURO_PSYCHIATRY` | `PATH_B_VLM_OCR` | `70.0` | `4` | `20` | `72.00%` | `75.5%` | **`3.50%`** | `76.4%` | `0.98` |
| 07 | `NCT03260205` | Attention Deficit (ADHD) | `NEURO_PSYCHIATRY` | `PATH_A_VECTOR` | `87.5` | `5` | `27` | `75.38%` | `76.1%` | **`0.72%`** | `76.7%` | `1.14` |
| 08 | `NCT04251156` | Overweight | `METABOLIC_WEIGHT` | `PATH_B_VLM_OCR` | `61.6` | `2` | `4` | `97.33%` | `96.6%` | **`0.73%`** | `96.6%` | `0.00` |
| 09 | `NCT03247556` | ADHD | `NEURO_PSYCHIATRY` | `PATH_C_PROSE` | `58.0` | `8` | `5` | `85.81%` | `84.4%` | **`1.41%`** | `85.4%` | `2.71` |
| 10 | `NCT03013751` | Functional Single Ventricle | `PEDIATRIC_ACUTE` | `PATH_B_VLM_OCR` | `63.6` | `4` | `4` | `100.00%` | `97.4%` | **`2.60%`** | `97.4%` | `0.52` |
| 11 | `NCT03334539` | Dry Eye | `OPHTHALMOLOGY_TOPICAL` | `PATH_A_VECTOR` | `87.7` | `6` | `25` | `92.00%` | `94.1%` | **`2.10%`** | `94.1%` | `0.63` |
| 12 | `NCT01720667` | Neonatal Seizures | `PEDIATRIC_ACUTE` | `PATH_B_VLM_OCR` | `64.2` | `5` | `3` | `95.28%` | `97.5%` | **`2.22%`** | `97.6%` | `1.06` |
| 13 | `NCT05486065` | Diabetes Mellitus, Type 2 | `ENDOCRINE_DIABETES` | `PATH_B_VLM_OCR` | `78.0` | `13` | `28` | `89.39%` | `86.8%` | **`2.59%`** | `92.2%` | `54.17` |
| 14 | `NCT04063163` | Extensive Stage SCLC | `ADVANCED_METASTATIC` | `PATH_A_VECTOR` | `86.8` | `4` | `28` | `15.21%` | `12.6%` | **`2.61%`** | `14.2%` | `3.68` |
| 15 | `NCT01876901` | Medium & Lower Rectal Ca | `SOLID_TUMOR_ONCOLOGY` | `PATH_B_VLM_OCR` | `61.2` | `2` | `3` | `80.65%` | `84.5%` | **`3.85%`** | `84.5%` | `0.00` |
| 16 | `NCT05196919` | Facet Joint Pain | `MUSCULOSKELETAL_PAIN` | `PATH_A_VECTOR` | `78.8` | `4` | `12` | `85.33%` | `89.2%` | **`3.87%`** | `89.2%` | `0.58` |
| 17 | `NCT01724346` | Chronic Lymphocytic Leukemia | `HEMATOLOGIC_MALIGNANCY` | `PATH_B_VLM_OCR` | `62.2` | `3` | `3` | `44.61%` | `47.6%` | **`2.99%`** | `47.9%` | `0.50` |
| 18 | `NCT03173456` | Musculoskeletal Pain (ED) | `PEDIATRIC_ACUTE` | `PATH_B_VLM_OCR` | `64.2` | `3` | `8` | `99.50%` | `98.5%` | **`1.00%`** | `98.7%` | `2.62` |
| 19 | `NCT01615029` | Multiple Myeloma | `HEMATOLOGIC_MALIGNANCY` | `PATH_A_VECTOR` | `93.9` | `7` | `44` | `46.67%` | `50.4%` | **`3.73%`** | `65.5%` | `18.62` |
| 20 | `NCT04880642` | COVID-19 | `PEDIATRIC_ACUTE` | `PATH_A_VECTOR` | `90.1` | `8` | `25` | `98.16%` | `98.0%` | **`0.16%`** | `98.0%` | `5.88` |
| 21 | `NCT05890586` | COVID-19 | `PEDIATRIC_ACUTE` | `PATH_A_VECTOR` | `85.1` | `8` | `15` | `96.99%` | `98.3%` | **`1.31%`** | `98.3%` | `1.37` |
| 22 | `NCT02709486` | Osteoarthritis, Hip | `MUSCULOSKELETAL_PAIN` | `PATH_B_VLM_OCR` | `78.0` | `13` | `20` | `85.51%` | `90.8%` | **`5.29%`** | `91.1%` | `4.26` |
| 23 | `NCT02412670` | Upper Tract Urothelial Ca | `SOLID_TUMOR_ONCOLOGY` | `PATH_A_VECTOR` | `90.4` | `12` | `16` | `77.78%` | `80.9%` | **`3.12%`** | `82.5%` | `8.39` |
| 24 | `NCT02371369` | Pigmented Villonodular Syn | `MUSCULOSKELETAL_PAIN` | `PATH_A_VECTOR` | `96.0` | `23` | `39` | `83.33%` | `79.5%` | **`3.83%`** | `91.1%` | `100.78` |
| 25 | `NCT04540094` | Sepsis | `PEDIATRIC_ACUTE` | `PATH_A_VECTOR` | `85.1` | `8` | `15` | `100.00%` | `98.3%` | **`1.70%`** | `98.3%` | `1.61` |
| 26 | `NCT03802396` | Postoperative Delirium | `PEDIATRIC_ACUTE` | `PATH_A_VECTOR` | `86.7` | `6` | `23` | `98.41%` | `94.3%` | **`4.11%`** | `97.3%` | `16.02` |
| 27 | `NCT03332303` | Vulvar & Vaginal Atrophy | `OPHTHALMOLOGY_TOPICAL` | `PATH_A_VECTOR` | `78.4` | `2` | `16` | `91.85%` | `95.8%` | **`3.95%`** | `96.1%` | `2.23` |
| 28 | `NCT03537742` | Vasculopathy | `PEDIATRIC_ACUTE` | `PATH_B_VLM_OCR` | `67.8` | `7` | `7` | `100.00%` | `97.5%` | **`2.50%`** | `97.6%` | `1.29` |
| 29 | `NCT03390842` | Metabolic Acidosis | `ENDOCRINE_DIABETES` | `PATH_A_VECTOR` | `95.4` | `12` | `26` | `94.39%` | `91.5%` | **`2.89%`** | `92.5%` | `8.84` |
| 30 | `NCT04169373` | Spondyloarthritis | `MUSCULOSKELETAL_PAIN` | `PATH_B_VLM_OCR` | `62.6` | `3` | `4` | `92.78%` | `93.4%` | **`0.62%`** | `93.4%` | `0.12` |
| 31 | `NCT02708966` | Impaired Glucose Tolerance | `ENDOCRINE_DIABETES` | `PATH_C_PROSE` | `56.2` | `7` | `3` | `86.67%` | `92.3%` | **`5.63%`** | `93.3%` | `2.65` |
| 32 | `NCT02059239` | Hodgkin's Lymphoma | `LYMPHOMA_CHEMO` | `PATH_A_VECTOR` | `75.5` | `5` | `3` | `85.29%` | `85.9%` | **`0.61%`** | `86.2%` | `0.24` |
| 33 | `NCT02648477` | ER-Negative Breast Cancer | `SHORT_WINDOW_ONCOLOGY` | `PATH_A_VECTOR` | `85.0` | `5` | `22` | `100.00%` | `96.1%` | **`3.90%`** | `97.3%` | `6.02` |
| 34 | `NCT02617628` | Opiate Dependence | `OPIOID_DEPENDENCE` | `PATH_B_VLM_OCR` | `69.2` | `10` | `3` | `15.75%` | `21.2%` | **`5.45%`** | `23.1%` | `0.96` |
| 35 | `NCT02532283` | Influenza A Virus | `AUTOIMMUNE_DERMATOLOGY` | `PATH_A_VECTOR` | `95.5` | `12` | `31` | `85.86%` | `85.8%` | **`0.06%`** | `88.9%` | `19.88` |
| 36 | `NCT03931941` | Clostridium Difficile Inf | `AUTOIMMUNE_DERMATOLOGY` | `PATH_A_VECTOR` | `79.9` | `2` | `19` | `90.97%` | `93.5%` | **`2.53%`** | `93.5%` | `0.00` |
| 37 | `NCT05707351` | Hemophilia A | `ENDOCRINE_DIABETES` | `PATH_A_VECTOR` | `89.4` | `7` | `26` | `91.89%` | `87.6%` | **`4.29%`** | `92.7%` | `30.30` |
| 38 | `NCT02623829` | Post-operative Scarring | `PEDIATRIC_ACUTE` | `PATH_B_VLM_OCR` | `62.6` | `3` | `4` | `95.24%` | `97.9%` | **`2.66%`** | `97.9%` | `0.12` |
| 39 | `NCT03377699` | Diabetes | `ENDOCRINE_DIABETES` | `PATH_A_VECTOR` | `85.1` | `3` | `27` | `79.11%` | `77.9%` | **`1.21%`** | `80.3%` | `2.44` |
| 40 | `NCT04206553` | Bullous Pemphigoid | `FRAIL_DERMATOLOGY` | `PATH_A_VECTOR` | `96.0` | `13` | `27` | `72.64%` | `73.9%` | **`1.26%`** | `77.0%` | `21.44` |

---

### Tier 3: 20-Trial Out-of-Sample Blind Holdout Benchmark (`2.88%` MAE)

Reproducible via `python blind_holdout_validator.py`:

| # | NCT ID | Condition / Indication | Clinical Domain | Box 2 Mode | Score | Visits | Procs | Actual Ret (%) | Sim Base (%) | Abs Err (%) | Sim Opt (%) | Hrs Saved |
| :-: | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 01 | `NCT04449029` | Hepatitis B | `MASH_METABOLIC` | `PATH_A_VECTOR` | `96.0` | `23` | `41` | `95.19%` | `90.2%` | **`4.99%`** | `90.8%` | `19.00` |
| 02 | `NCT01434602` | Brain Tumor | `SOLID_TUMOR_ONCOLOGY` | `PATH_B_VLM_OCR` | `68.6` | `7` | `9` | `84.93%` | `77.9%` | **`7.03%`** | `78.5%` | `1.95` |
| 03 | `NCT03576001` | Spinal Cord Injuries | `NEURO_PSYCHIATRY` | `PATH_B_VLM_OCR` | `71.6` | `9` | `14` | `78.57%` | `83.3%` | **`4.73%`** | `83.5%` | `2.17` |
| 04 | `NCT02951819` | Multiple Myeloma | `HEMATOLOGIC_MALIGNANCY` | `PATH_A_VECTOR` | `83.1` | `8` | `23` | `65.35%` | `60.5%` | **`4.85%`** | `63.7%` | `4.21` |
| 05 | `NCT01617967` | TTR-Mediated Amyloidosis | `ENDOCRINE_DIABETES` | `PATH_B_VLM_OCR` | `66.0` | `6` | `5` | `89.66%` | `88.7%` | **`0.96%`** | `92.8%` | `21.33` |
| 06 | `NCT04286607` | Chronic Plaque Psoriasis | `AUTOIMMUNE_DERMATOLOGY` | `PATH_A_VECTOR` | `82.2` | `6` | `14` | `88.29%` | `90.2%` | **`1.91%`** | `90.3%` | `0.83` |
| 07 | `NCT01570387` | AL Amyloidosis | `ENDOCRINE_DIABETES` | `PATH_A_VECTOR` | `85.5` | `5` | `23` | `88.89%` | `88.9%` | **`0.01%`** | `89.6%` | `9.93` |
| 08 | `NCT04399837` | Generalized Pustular Psor | `AUTOIMMUNE_DERMATOLOGY` | `PATH_A_VECTOR` | `96.0` | `16` | `21` | `90.24%` | `85.7%` | **`4.54%`** | `88.5%` | `18.93` |
| 09 | `NCT03385239` | Hypertriglyceridemia | `ENDOCRINE_DIABETES` | `PATH_A_VECTOR` | `96.0` | `17` | `23` | `89.47%` | `91.4%` | **`1.93%`** | `92.4%` | `13.27` |
| 10 | `NCT03065075` | Urinary Retention Post-Op | `SELECTIVE_ABSTENTION` | `TIER_C_ABSTAIN` | `0.0` | `0` | `0` | `100.00%` | *Abstained* | *Routed UI* | *Abstained* | β€” |
| 11 | `NCT03296553` | Kaposi Sarcoma | `OPHTHALMOLOGY_TOPICAL` | `PATH_B_VLM_OCR` | `75.8` | `9` | `22` | `92.50%` | `93.7%` | **`1.20%`** | `93.9%` | `1.55` |
| 12 | `NCT01987895` | Clostridium Difficile Inf | `AUTOIMMUNE_DERMATOLOGY` | `PATH_A_VECTOR` | `83.7` | `6` | `17` | `90.51%` | `88.0%` | **`2.51%`** | `89.6%` | `5.05` |
| 13 | `NCT01857934` | Neuroblastoma | `SOLID_TUMOR_ONCOLOGY` | `PATH_B_VLM_OCR` | `72.0` | `4` | `34` | `84.38%` | `83.2%` | **`1.18%`** | `85.7%` | `4.19` |
| 14 | `NCT03449134` | Chronic Cough | `MASH_METABOLIC` | `PATH_A_VECTOR` | `96.0` | `11` | `37` | `79.64%` | `83.6%` | **`3.96%`** | `88.8%` | `27.23` |
| 15 | `NCT05579899` | Atopic Dermatitis Eczema | `AUTOIMMUNE_DERMATOLOGY` | `PATH_A_VECTOR` | `85.7` | `6` | `21` | `90.76%` | `85.0%` | **`5.76%`** | `89.4%` | `13.88` |
| 16 | `NCT03075826` | Myeloproliferative Neopl | `MASH_METABOLIC` | `PATH_A_VECTOR` | `81.7` | `6` | `13` | `72.73%` | `75.8%` | **`3.07%`** | `89.4%` | `20.93` |
| 17 | `NCT02978326` | Postpartum Depression | `METABOLIC_WEIGHT` | `PATH_B_VLM_OCR` | `65.6` | `4` | `9` | `92.81%` | `94.4%` | **`1.59%`** | `94.7%` | `0.46` |
| 18 | `NCT03939689` | Metastatic Prostate Ca | `ADVANCED_METASTATIC` | `PATH_A_VECTOR` | `84.0` | `5` | `20` | `36.52%` | `36.6%` | **`0.08%`** | `38.4%` | `4.87` |
| 19 | `NCT03489850` | Alcohol Use Disorder | `PEDIATRIC_ACUTE` | `PATH_B_VLM_OCR` | `67.2` | `6` | `8` | `96.15%` | `98.3%` | **`2.15%`** | `98.3%` | `0.86` |
| 20 | `NCT02157974` | Hepatic Steatosis | `METABOLIC_WEIGHT` | `PATH_A_VECTOR` | `80.1` | `3` | `17` | `96.74%` | `94.5%` | **`2.24%`** | `94.6%` | `0.46` |

---

### Tier 4: 26-Language US/EU/Global Stress Test (`100%` Parity)

Validated across **26 languages** (`EL_GREEK`, `DE_GERMAN`, `FR_FRENCH`, `ES_SPANISH`, `IT_ITALIAN`, `PT_PORTUGUESE`, `NL_DUTCH`, `PL_POLISH`, `SV_SWEDISH`, `DA_DANISH`, `FI_FINNISH`, `CS_CZECH`, `HU_HUNGARIAN`, `RO_ROMANIAN`, `TR_TURKISH`, `EN_ENGLISH`, `NO_NORWEGIAN`, `BG_BULGARIAN`, `SK_SLOVAK`, `HR_CROATIAN`, `SL_SLOVENIAN`, `ET_ESTONIAN`, `LV_LATVIAN`, `LT_LITHUANIAN`, `ID_INDONESIAN`, `VI_VIETNAMESE`):

- **Extracted Procedures:** `10 / 10` (`100%` across all 26 languages)
- **Expanded Calendar Visits:** `9 / 9` (`Weeks [-2, 1, 4, 8, 12, 16, 20, 24, 52]`)
- **Default Concept Fallback Rate:** `0.00%` across all 26 languages
- **Simulated Retention & Optimization Parity:** Identical `74.8%` Baseline $\rightarrow$ `85.1%` Optimized (`+10.3%` retention rescued, `30.15` clinic chair hours saved/patient) across all 26 languages.

---

## Patient Safety, Epistemic Humility & Human-in-the-Loop Guardrails

1. **Pre-Trial Operational Decision Support (Never Bedside Clinical Advice):**
   - `Trial-Friction-Sim` is an operational protocol design and feasibility simulator. It never alters drug dosages, clinical eligibility criteria, or patient-level medical care.
2. **Hard Endpoint Safety Locks (`optimize_usdm_protocol_schedule` in `Box 6`):**
   - Primary efficacy endpoints (`endpoint_tier == "primary"`), mandatory administrative procedures (`Informed Consent`), and routine safety monitoring (`Physical Exam & Vitals`, `Routine Safety Blood Draw`, `Urinalysis & Pregnancy Test`, `12-Lead ECG`, `Adverse Event & Concomitant Medication Check`) are **never deleted** by the Pareto optimizer.
   - Anchor clinic weeks (`first_wk`, `baseline_wk`, `Week 12`, `Week 24`, `Week 40`, `last_wk`) are locked at the hospital clinic, and `week_proc_counts[wk] > 1` guarantees no calendar visit is ever orphaned.
3. **3-Tier Epistemic Readiness & Selective Abstention Gate (`app_dashboard.py`):**
   - **Tier A (`conf >= 75.0`):** Green banner; autonomous simulation on verified ruled vector grids (`2.62%` Combined Tier A MAE).
   - **Tier B (`conf < 75.0`):** Yellow warning banner (`Human Verification Recommended`); exposes the **Interactive Human-in-the-Loop Schedule Editor (`st.data_editor`)** so clinicians can inspect or enrich VLM/prose-reconstructed schedules in seconds.
   - **Tier C (`validation_passed == False`):** Red banner (`Selective Abstention Triggered`); halts blind simulation completely when a PDF lacks a machine-readable SoA grid (such as `NCT03065075`) and opens the Interactive Schedule Editor.

---

## Regulatory Alignment (US FDA & EU EMA 2025β€“2026)

| Regulatory / Data Standard | Governing Body | Implementation in `Trial-Friction-Sim` |
| :--- | :--- | :--- |
| **ICH E6(R3) Good Clinical Practice (Section 3: Quality by Design)** | US FDA & EU EMA (2025β€“2026) | Quantifies participant time toxicity and biophysical burden per visit; tests proportionality levers in `pareto_simplifier.py`. |
| **ICH M11 CeSHarP (Clinical Electronic Structured Harmonised Protocol)** | US FDA (May 2026) & EU EMA (June 2026) | Tags every normalized procedure with `ich_m11_category` and structures epochs (`SCREENING_EPOCH`, `BASELINE_EPOCH`, `TREATMENT_EPOCH`, `FOLLOW_UP_EPOCH`). |
| **CDISC USDM v4.1 (Unified Study Definitions Model β€” DDF SoA)** | CDISC & TransCelerate | Outputs `clean_usdm_schedule.json` with ISO-8601 durations (`planned_duration_iso8601`), study days, weeks, and biomedical concepts. |
| **CDISC NCIt + LOINC v2.78 + SNOMED-CT** | US FDA & EU EHDS / EMA CTIS | Binds all 15 canonical concepts (`1,051` synonyms) to exact `cdisc_code`, `loinc_code`, and `snomed_code` identifiers. |

---

## Repository Structure

```text
Trial-Friction-Sim/
β”β”€β”€ config/
β”‚   β””β”€β”€ settings.py                        # Portable workspace routing & 8GB VRAM GPU Airlock (unload_ollama_model)
β”β”€β”€ data_local/
β”‚   β”β”€β”€ procedure_burden_dict.json         # 15 Canonical Biomedical Concepts (1,051 synonyms across 26 languages)
β”‚   β”β”€β”€ ALL_PROTOCOLS_DEEP_FORENSIC_AUDIT.json
β”‚   β””β”€β”€ BLIND_10_TRIAL_HOLDOUT_BENCHMARK.json
β”β”€β”€ box1_page_sieve/
β”‚   β””β”€β”€ sieve.py                           # Box 1: 0-VRAM topological SoA page cluster isolator & synopsis reader
β”β”€β”€ box2_table_extractor/
β”‚   β””β”€β”€ extractor.py                       # Box 2: Self-healing Path A/B/C extractor, epoch fusion & 3-tier auditor
β”β”€β”€ box3_usdm_normalizer/
β”‚   β””β”€β”€ normalizer.py                      # Box 3: Timepoint expander & two-tier CENT-Local CDISC/LOINC/SNOMED mapper
β”β”€β”€ box4_burden_scorer/
β”‚   β””β”€β”€ scorer.py                          # Box 4: 5D burden scorer, log-subadditive stacking & site cost estimator
β”β”€β”€ box5_dropout_simulator/
β”‚   β”β”€β”€ aact_duckdb_loader.py              # Box 5A: 38,302-trial DuckDB stream & negation-safe domain classifier
β”‚   β””β”€β”€ survival_sim.py                    # Box 5B: 1,000-patient PyTorch competing-risks survival simulator
β”β”€β”€ box6_optimizer_ui/
β”‚   β”β”€β”€ pareto_simplifier.py               # Box 6A: Visit-preserving ICH E6(R3) Pareto optimizer & Tufts CSDD auditor
β”‚   β””β”€β”€ app_dashboard.py                   # Box 6B: Streamlit studio, schedule editor & vector KB precedent search
β”β”€β”€ real_data_validator.py                 # Reproducible 38,302-trial macro + 40-trial real PDF + 26-lang benchmark
β”β”€β”€ blind_holdout_validator.py             # Reproducible 20-trial out-of-sample blind holdout benchmark
β”β”€β”€ forensic_folder_auditor.py             # Deep forensic folder auditor verifying 0/67 flags across cached protocols
β”β”€β”€ requirements.txt
β”β”€β”€ The Full Architecture in Simple Boxes & Arrows.md
β””β”€β”€ README.md
```

---

## Installation, Hardware Requirements & Quickstart

### Hardware Discipline & 8 GB VRAM GPU Airlock
- **CPU / RAM:** Standard 4+ core CPU, 16β€“32 GB RAM.
- **GPU (Optional / Local Airlock):** Designed for an **8 GB VRAM** local GPU running [Ollama](https://ollama.com/). Boxes 1, 3 (Tier 1), 4, and 5 use **`0.0 GB` VRAM**. When Box 2 (`Path B` / `Path C`) or Box 6 loads `qwen3.5:4b` (`~3.4 GB`) or `qwen2.5:7b-instruct` (`~4.7 GB`), `unload_ollama_model()` immediately evicts weights via `keep_alive=0` so two models never coexist in VRAM.

### 1. Environment Setup
```powershell
git clone [https://github.com/DimitriosThomaidis/Trial-Friction-Sim.git](https://github.com/DimitriosThomaidis/Trial-Friction-Sim.git)
cd Trial-Friction-Sim
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Pull Local Ollama Models (For VLM OCR Fallback & Executive Memo)
```powershell
ollama pull qwen3.5:4b
ollama pull qwen2.5:7b-instruct
```

### 3. Launch the Interactive Streamlit Studio (`Box 6`)
```powershell
streamlit run box6_optimizer_ui/app_dashboard.py
```

### 4. Reproduce the Full Empirical Benchmarks
```powershell
# Run Tier 1 (38,302 Trials) + Tier 2 (40 Real ClinicalTrials.gov PDFs) + Tier 4 (26 Languages)
python real_data_validator.py

# Run Tier 3 (20 Unseen Out-of-Sample Blind Holdout ClinicalTrials.gov PDFs)
python blind_holdout_validator.py

# Run the Deep Forensic Folder Auditor across all active protocol folders
python forensic_folder_auditor.py
```

---

## License & Citation

Released under the **MIT License**. If you use `Trial-Friction-Sim` in your research or clinical operations, please cite:

```bibtex
@software{thomaidis2026trialfrictionsim,
  author       = {Thomaidis, Dimitrios},
  title        = {Trial-Friction-Sim: A 6-Box Clinical Trial Schedule of Assessments (SoA) Burden & Competing-Risks Dropout Simulator},
  year         = {2026},
  url          = {[https://github.com/DimitriosThomaidis/Trial-Friction-Sim](https://github.com/DimitriosThomaidis/Trial-Friction-Sim)},
  note         = {Validated across 60 real ClinicalTrials.gov Sponsor Protocol PDFs (2.60\% MAE), 38,302 TrialBench/AACT historical trials, and 26 languages under US FDA & EU EMA ICH E6(R3), ICH M11 CeSHarP, and CDISC USDM v4.1}
}
```
