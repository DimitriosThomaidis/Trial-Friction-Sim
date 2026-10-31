========================================================================================================================
                TRIAL-FRICTION-SIM: COMPLETE 6-BOX SYSTEM ARCHITECTURE & DATA FLOW (v3.1 — 2026)
========================================================================================================================

  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │                                  INPUT LAYER: REAL CLINICAL PROTOCOL DOCUMENTS                                   │
  │  • Live API Stream : ClinicalTrials.gov REST API v2 (`ProvidedDocs` Sponsor PDFs + Period-Matched Flow Results)  │
  │  • Google Drive    : `G:\Το Drive μου\TrialFrictionSim\protocols_pdf\` (2-page to 483-page Sponsor PDFs)         │
  │  • Empirical Scale : Validated on 60 Real Completed Protocols (2.60% Combined MAE | 2.50% Median | 3.08% RMSE)   │
  │                      - Development Cohort (N=40) : 2.47% MAE | 2.87% RMSE | 0.00% Fallback | 0/67 Audit Flags    │
  │                      - Blind Holdout (N=20)      : 2.88% MAE | 2.24% Median | 2.85% Tier A MAE (+1 Tier C Gate)  │
  │                      + 38,302 Historical Trials (TrialBench / AACT Parquet) + 26 Global & EU Languages           │
  └────────────────────────────────────────────────────────┬─────────────────────────────────────────────────────────┘
                                                           │
                                                           ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ BOX 1: CONTIGUOUS CLUSTER PAGE SIEVE & SYNOPSIS READER (`box1_page_sieve/sieve.py` — CPU | 0.0 GB VRAM)          │
  │ ──────────────────────────────────────────────────────────────────────────────────────────────────────────────── │
  │  1. Fast Page Scorer (`score_pdf_page`):                                                                         │
  │     • Rejects Table of Contents (`"..." > 2`) and pure prose Estimand / Inclusion / Exclusion pages.             │
  │     • Scans for `EXTENDED_SOA_HEADERS` ("schedule of activities", "study calendar", "assessment schedule").      │
  │     • Counts `TIME_HEADER_PATTERN` tokens (Week, Day, Cycle, Visit, Month, Screening, Baseline, EOS/EOT).        │
  │     • Counts standalone checkmarks (`X, ✓, ✔, •`) and tests `page.find_tables()` for ruled grids (+65.0 pts).   │
  │  2. Front-Matter Synopsis Extractor (`extract_protocol_synopsis_and_title_text`):                                │
  │     • Reads Pages 1–3 (`front_matter_synopsis_text`) so raw user-uploaded PDFs without API metadata still have   │
  │       their disease indication and study phase auto-detected.                                                    │
  │  3. Contiguous Cluster Isolator & Backtracking Memory (`run_page_sieve`):                                        │
  │     • Groups candidate pages into contiguous clusters (1 to 6 consecutive pages).                                │
  │     • Supports `excluded_pages_0based` so Box 2 can backtrack to Cluster #2 if Cluster #1 fails validation.      │
  │  4. Artifacts Synced Directly to Google Drive (`G:\Το Drive μου\TrialFrictionSim\temp_cache\<protocol>\`):       │
  │     • `mini_soa_pages.pdf`, `soa_page_images/*.png` (150 DPI), and `box1_sieve_manifest.json`.                   │
  └────────────────────────────────────────────────────────┬─────────────────────────────────────────────────────────┘
                                                           │
                                                           ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ BOX 2: SELF-HEALING TRI-MODAL EXTRACTOR & EPOCH FUSION (`box2_table_extractor/extractor.py`)                     │
  │ ──────────────────────────────────────────────────────────────────────────────────────────────────────────────── │
  │  ┌──► [MULTI-CLUSTER TOURNAMENT LOOP: Evaluates up to `MAX_PAGE_CLUSTERS = 2` Disjoint Page Clusters] ────────┐  │
  │  │                                                                                                            │  │
  │  │  ► PATH A: Deterministic Ruled Vector Grid Parser (`_deterministic_grid_to_json` | 0.0 GB VRAM | n=34)     │  │
  │  │    • Extracts PyMuPDF vector tables (`find_tables`); carries headers across multi-page continuations.      │  │
  │  │    • Multi-Row Header & Watermark Cleaner (`_clean_superscript_from_header`): Strips vertical PDF          │  │
  │  │      watermark letters (`m m o c r e`), unmerges stacked cells (`"4\n2"` -> `Wk 2, Wk 4`), strips glued    │  │
  │  │      footnote suffixes (`"Week 8c"` -> `"Week 8"`), and blocks empty em-dash window headers (`"— (±2d)"`). │  │
  │  │    • Multi-Cluster Split-Epoch Fusion (`_is_epoch_complete_schedule` & `_fuse_split_epoch_clusters`):      │  │
  │  │      If Cluster #1 is a valid pre-treatment/screening sub-table (<4 visits), inspects Cluster #2 via       │  │
  │  │      0-VRAM Path A and fuses adjacent Screening + Treatment tables into a unified longitudinal schedule.   │  │
  │  │      ├──► EPOCH-COMPLETE : Breaks loop immediately (`PATH_A_RULED_VECTOR_GRID` — 2.36% Dev / 2.99% Hold).  │  │
  │  │      └──► FAIL           : Saves candidate and triggers PATH B on current cluster's PNG images ──►         │  │
  │  │                                                                                                            │  │
  │  │  ► PATH B: Local VLM Visual Image-to-Table OCR (`extract_table_from_image_with_vlm` | 3.4 GB VRAM | n=23)  │  │
  │  │    • Loads `qwen3.5:4b` in GPU Airlock Slot 1 (`think=False`, `seed=42`, SHA256-cached in `temp_cache`).   │  │
  │  │    • Self-Healing JSON Recovery (`_repair_truncated_json` via `JSONDecoder.raw_decode`).                   │  │
  │  │    • Cleans VLM visit headers via `_clean_superscript_from_header` and filters suffix loops (`-C-T-R`).    │  │
  │  │    • Immediately unloads `qwen3.5:4b` from 8 GB VRAM (`keep_alive=0`).                                     │  │
  │  │      ├──► PASS : Breaks loop immediately (`PATH_B_VLM_VISUAL_IMAGE_OCR` — 2.50% Dev / 2.69% Holdout MAE).  │  │
  │  │      └──► FAIL : Adds cluster pages to `excluded_pages` and BACKTRACKS to Box 1 for Cluster #2! ───────────┘  │
  │  │                                                                                                               │
  │  ▼ (If neither Cluster #1 nor Cluster #2 passed Validation Gate)                                                 │
  │  ► PATH C: Prose Text-to-Table Synthesizer (`synthesize_table_from_prose_text_with_llm` | 4.7 GB VRAM | n=2)     │
  │    • Scans top 5 scheduling prose sections (`extract_scheduling_prose_from_pdf`).                                │
  │    • Loads `qwen2.5:7b-instruct` in GPU Airlock Slot 2 + Canonicalized Timepoint Lexical Co-Occurrence Fallback. │
  │                                                                                                                  │
  │  ► CANDIDATE TOURNAMENT (`max(candidates, key=_candidate_rank)`) & CONFIDENCE AUDITOR:                           │
  │    • Selects highest-ranked candidate; computes `extraction_confidence_audit` (0–100 score; Tier A/B/C).         │
  │    • Writes `raw_soa_grid.json` with full `"extraction_attempts"` provenance to Google Drive `temp_cache`.       │
  └────────────────────────────────────────────────────────┬─────────────────────────────────────────────────────────┘
                                                           │
                                                           ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ SELECTIVE ABSTENTION GATE & HUMAN-IN-THE-LOOP SCHEDULE EDITOR (Protecting Prospective / Unseen Trials)           │
  │ ──────────────────────────────────────────────────────────────────────────────────────────────────────────────── │
  │  • Tier A (`conf >= 75.0` & `validation_passed`, n=37, 2.62% MAE) -> Green Banner: Autonomous simulation.        │
  │  • Tier B (`conf <  75.0` & `validation_passed`, n=22, 2.58% MAE) -> Yellow Banner: Simulates + Schedule Editor. │
  │  • Tier C (`validation_passed == False`, n=1 `NCT03065075`)       -> Red Banner (`ABSTAINED_ROUTED_TO_UI`):      │
  │                                                                      Pauses blind simulation & opens UI Editor.  │
  └────────────────────────────────────────────────────────┬─────────────────────────────────────────────────────────┘
                                                           │
                                                           ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ BOX 3: 26-LANGUAGE TRI-STANDARD USDM v4.1 & CENT-LOCAL NORMALIZER (`box3_usdm_normalizer/normalizer.py`)         │
  │ ──────────────────────────────────────────────────────────────────────────────────────────────────────────────── │
  │  1. Stage 3A Timepoint Expander (`parse_and_expand_visit_header`):                                               │
  │     • Strips `±` visit-window tolerances first; unpacks Slash-Lists (`18/20/24/28/32/36`), Negative Screening    │
  │       Ranges (`-6 to -1`, `Within 30 days`), Week Ranges (`4A`), Visit Ranges (`4B`), and Cycle Ranges (`4C`).   │
  │     • Evaluates prefix `Day -14` before suffix `+7d`; flags `Telephone Call` visits as `0.15 hr` remote calls.   │
  │  2. Stage 3B Two-Tier Self-Learning Ontology Mapper (`map_procedure_to_cdisc_concept` + `CENT-Local`):           │
  │     • Tier 1: Deterministic lexical matching across 1,051 clinical & 26-language synonyms (`data_local/`).       │
  │     • Tier 2: Constrained zero-shot `qwen2.5:7b-instruct` normalizer (`_classify_unmapped_procedure_with_llm`)   │
  │       cached permanently in `learned_concept_synonyms_cache.json` (`0.00%` Dev / `0.07%` Combined fallback).     │
  │     • Binds `cdisc_code` (NCIt) + `loinc_code` + `snomed_code` + `ich_m11_category` + `planned_duration_iso8601`.│
  └────────────────────────────────────────────────────────┬─────────────────────────────────────────────────────────┘
                                                           │
                                                           ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ BOX 4: 5D TUFTS CSDD BURDEN, STACKING LOAD & SITE COST SCORER (`box4_burden_scorer/scorer.py`)                   │
  │ ──────────────────────────────────────────────────────────────────────────────────────────────────────────────── │
  │  • Deduplicates same-visit concept sub-rows while tracking `procedure_count`, `raw_procedure_count`, and         │
  │    Log-Subadditive Intra-Concept Assessment Stacking Load (`stacked_assessment_load = sum(1 + 0.35*ln(m_c))`):   │
  │    1. `total_chair_hours` (Overlap-adjusted clinic duration)   4. `pro_cognitive_mins` (Questionnaire load)      │
  │    2. `effective_pain_0_10` (Biophysical invasiveness peak)    5. `caregiver_required` (Sedation/driver flag)    │
  │    3. `fasting_required` (Overnight fasting flag)              6. `visit_site_cost_usd` (Estimated site cost $)  │
  │  • Outputs: `weekly_burden_scores.parquet` & `weekly_burden_summary.json`                                        │
  └────────────────────────────────────────────────────────┬─────────────────────────────────────────────────────────┘
                                                           │
                                                           ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ BOX 5: 38,302-TRIAL DUCKDB STREAM & UNIVERSAL SURVIVAL SIMULATOR (`box5_dropout_simulator/`)                     │
  │ ──────────────────────────────────────────────────────────────────────────────────────────────────────────────── │
  │  1. Zero-Copy Cloud Database & Two-Stage Negation-Safe Classifier (`aact_duckdb_loader.py`):                     │
  │     • Streams baseline hazards from `trialbench_38k_real_dropouts.parquet` (38,302 real completed trials).       │
  │     • Classifies pre-trial metadata across 20 clinical domains (stripping negated exclusion prose).              │
  │  2. Competing-Risks Survival Engine (`survival_sim.py` — PyTorch Tensors):                                       │
  │     • Simulates 1,000 virtual patient-caregiver dyads via `compute_universal_effective_domain_shift()`           │
  │       (Zero disease-named `if/elif` rules: Empirical Bayes multi-visit damping, per-visit assessment-compression │
  │       scaling for `max_raw_procs >= 6` & `max_pain < 8.0`, and acute <=4-week daily supervision scaling).        │
  └────────────────────────────────────────────────────────┬─────────────────────────────────────────────────────────┘
                                                           │
                                                           ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ BOX 6: VISIT-PRESERVING ICH E6(R3) PARETO SIMPLIFIER, VECTOR KB PRECEDENTS & TUFTS AMENDMENT AUDITOR             │
  │ ──────────────────────────────────────────────────────────────────────────────────────────────────────────────── │
  │  • `pareto_simplifier.py`: Locks Primary/Safety endpoints; applies 3 ICH E6(R3) Levers while guaranteeing        │
  │    `week_proc_counts[wk] > 1` so pruning never deletes a scheduled calendar visit. Computes Tufts CSDD           │
  │    Substantial Amendment Probability (%) and Expected Direct Amendment Cost Saved ($ USD).                       │
  │  • `app_dashboard.py`: Displays Tier A/B/C Readiness Traffic Light, Interactive Schedule Editor, Survival &      │
  │    Time-Toxicity Charts, Top-5 Nearest ClinicalTrials.gov Precedents (Vector-Similarity Knowledge Base),         │
  │    60-Trial Validation Scorecard, and Bilingual (EN/EL) AI Regulatory Memo.                                      │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘