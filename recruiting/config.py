from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path

def _app_root() -> Path:
    """
    Resolve the project root (directory that contains .env, app.py, secrets/).
    By default: two levels up from this file, e.g. D:/EcodiaOS/eco_local
    Override with ECO_LOCAL_APP_ROOT if needed.
    """
    override = os.getenv("ECO_LOCAL_APP_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    # If this file is recruiting/config.py, parents[1] = eco_local/
    return Path(__file__).resolve().parents[1]

APP_ROOT = _app_root()
SECRETS_DIR = Path(os.getenv("ECO_LOCAL_SECRETS_DIR", APP_ROOT / "secrets")).resolve()

@dataclass
class Settings:
    ENV: str = os.getenv("ENV", "prod")
    LOCAL_TZ: str = os.getenv("LOCAL_TZ", "Australia/Brisbane")

    # Outbound mail (SES)
    SES_REGION: str = os.getenv("SES_REGION", os.getenv("AWS_REGION", "ap-southeast-2"))
    SES_SENDER_EMAIL: str = os.getenv("SES_SENDER_EMAIL", "ECOLocal@ecodia.au")
    ECO_LOCAL_REPLY_TO: str = os.getenv("ECO_LOCAL_REPLY_TO", "ECOLocal@ecodia.au")
    UNSUB_BASE_URL: str = os.getenv("UNSUB_BASE_URL", "https://ecodia.au/eco_local/business/unsubscribe")

    # Gmail/Calendar SA impersonation (default to <APP_ROOT>/secrets/*.json)
    GSUITE_IMPERSONATED_USER: str = os.getenv("GSUITE_IMPERSONATED_USER", "ECOLocal@ecodia.au")
    ECO_LOCAL_GCAL_ID: str = os.getenv("ECO_LOCAL_GCAL_ID", "primary")

    # Allow absolute env paths, otherwise use project-relative defaults
    GMAIL_SA_PATH: str = os.getenv(
        "GMAIL_SERVICE_ACCOUNT_JSON_PATH",
        str(SECRETS_DIR / "eco_local-recruit-sa.json"),
    )
    GCAL_SA_PATH: str = os.getenv(
        "GCAL_SERVICE_ACCOUNT_JSON_PATH",
        str(SECRETS_DIR / "eco_local-recruit-sa.json"),
    )

    # Neo4j (used directly in store)
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "password")

    # Quotas / cadence
    ECO_LOCAL_FIRST_TOUCH_QUOTA: int = int(os.getenv("ECO_LOCAL_FIRST_TOUCH_QUOTA", "5"))
    ECO_LOCAL_FOLLOWUP_DAYS: str = os.getenv("ECO_LOCAL_FOLLOWUP_DAYS", "3,7,14")
    ECO_LOCAL_MAX_ATTEMPTS: int = int(os.getenv("ECO_LOCAL_MAX_ATTEMPTS", "3"))

    # Embeddings (Gemini)
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

settings = Settings()

# Optional startup diagnostics (safe to keep)
if os.getenv("ECO_LOCAL_DEBUG_PATHS", "1") == "1":
    try:
        print(f"APP_ROOT: {APP_ROOT}")
        print(f"SECRETS_DIR: {SECRETS_DIR}")
        print(f"GMAIL_SA_PATH: {settings.GMAIL_SA_PATH}")
        print(f"GCAL_SA_PATH:  {settings.GCAL_SA_PATH}")
        print(f"NEO4J_URI:  {settings.NEO4J_URI}")
        if SECRETS_DIR.exists():
            print("DEBUG secrets dir :", [p.name for p in SECRETS_DIR.iterdir()])
        else:
            print("DEBUG secrets dir: <missing>", SECRETS_DIR)
    except Exception as _e:
        print("Path debug failed:", _e)
