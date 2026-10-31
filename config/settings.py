import os
import shutil
from pathlib import Path
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_LOCAL_DIR = PROJECT_ROOT / "data_local"
BURDEN_DICT_PATH = DATA_LOCAL_DIR / "procedure_burden_dict.json"

# Workspace Resolution Order:
# 1. Explicit environment variable `TRIAL_FRICTION_WORKSPACE` (if set by user/Docker/CI).
# 2. Existing local Google Drive workspace (`G:\Το Drive μου\TrialFrictionSim`) if already present on disk.
# 3. Default self-contained local repository directory (`PROJECT_ROOT / "workspace"`) for all GitHub clones.
_env_workspace = os.environ.get("TRIAL_FRICTION_WORKSPACE", "").strip()
_existing_gdrive = Path(r"G:\Το Drive μου\TrialFrictionSim")

if _env_workspace:
    WORKSPACE_ROOT = Path(_env_workspace).expanduser().resolve()
elif _existing_gdrive.exists():
    WORKSPACE_ROOT = _existing_gdrive
else:
    WORKSPACE_ROOT = PROJECT_ROOT / "workspace"

GDRIVE_ROOT = WORKSPACE_ROOT

PROTOCOLS_DIR = WORKSPACE_ROOT / "protocols_pdf"
AACT_PARQUET_DIR = WORKSPACE_ROOT / "aact_parquet"
SAVED_REPORTS_DIR = WORKSPACE_ROOT / "saved_reports"
TEMP_CACHE_DIR = WORKSPACE_ROOT / "temp_cache"
LEGACY_LOCAL_CACHE_DIR = PROJECT_ROOT / "temp_cache"

for d in [WORKSPACE_ROOT, PROTOCOLS_DIR, AACT_PARQUET_DIR, SAVED_REPORTS_DIR, TEMP_CACHE_DIR, DATA_LOCAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

_legacy_vlm_cache = LEGACY_LOCAL_CACHE_DIR / "vlm_deterministic_cache.json"
_active_vlm_cache = TEMP_CACHE_DIR / "vlm_deterministic_cache.json"
if _legacy_vlm_cache.exists() and not _active_vlm_cache.exists():
    try:
        shutil.copy2(_legacy_vlm_cache, _active_vlm_cache)
    except Exception:
        pass

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

MODELS = {
    "vlm_extractor": "qwen3.5:4b",
    "json_normalizer": "qwen2.5:7b-instruct",
}

SOA_KEYWORDS = [
    "schedule of assessments",
    "schedule of activities",
    "schedule of events",
    "time and events schedule",
    "study procedures",
    "visit schedule",
    "clinical trial schedule",
    "πρόγραμμα αξιολογήσεων",
    "visitenplan",
    "calendrier des évaluations",
    "calendario de evaluaciones",
    "calendario delle valutazioni",
    "calendário de avaliações",
    "schema van studiebeoordelingen",
]


def unload_ollama_model(model_name: str) -> None:
    """Immediately evicts a loaded model from 8GB VRAM via keep_alive=0."""
    try:
        requests.post(
            f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate",
            json={"model": model_name, "keep_alive": 0},
            timeout=10,
        )
    except Exception:
        pass


def unload_all_models() -> None:
    """Evicts all models from VRAM to guarantee a clean GPU Airlock."""
    for m in MODELS.values():
        unload_ollama_model(m)


def verify_environment() -> dict:
    return {
        "project_root": str(PROJECT_ROOT),
        "workspace_root": str(WORKSPACE_ROOT),
        "temp_cache_dir": str(TEMP_CACHE_DIR),
        "saved_reports_dir": str(SAVED_REPORTS_DIR),
        "burden_dict_exists": BURDEN_DICT_PATH.exists(),
    }


if __name__ == "__main__":
    print(verify_environment())