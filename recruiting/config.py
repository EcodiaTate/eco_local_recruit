from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path


def _app_root() -> Path:
    """
    Resolve the project root (directory that contains .env, app.py, secrets/).
    Default: two levels up from this file, e.g. D:/EcodiaOS/eco-local
    Override with ECO_LOCAL_APP_ROOT if needed.
    """
    override = os.getenv("ECO_LOCAL_APP_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    # If this file is recruiting/config.py, parents[1] = eco_local/
    return Path(__file__).resolve().parents[1]


def _detect_secrets_dir(app_root: Path) -> Path:
    """
    Cloud Run: prefer the mounted path /secrets if it exists.
    Otherwise, use ECO_LOCAL_SECRETS_DIR if provided.
    Otherwise, fall back to <APP_ROOT>/secrets for local dev.
    """
    if Path("/secrets").is_dir():
        return Path("/secrets")
    env_dir = os.getenv("ECO_LOCAL_SECRETS_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return (app_root / "secrets").resolve()


APP_ROOT = _app_root()
SECRETS_DIR = _detect_secrets_dir(APP_ROOT)


def _default_sa_basename() -> str:
    """
    Basename (without enforced extension) for the service account file.
    Cloud Run will often write the secret as a file named exactly after the secret.
    Your SA loader can try with and without '.json'.
    """
    return "eco_local-recruit-sa"


@dataclass
class Settings:
    ENV: str = os.getenv("ENV", "prod")
    LOCAL_TZ: str = os.getenv("LOCAL_TZ", "Australia/Brisbane")

    # Outbound mail (SES)
    SES_REGION: str = os.getenv("SES_REGION", os.getenv("AWS_REGION", "ap-southeast-2"))
    SES_SENDER_EMAIL: str = os.getenv("SES_SENDER_EMAIL", "ECOLocal@ecodia.au")
    ECO_LOCAL_REPLY_TO: str = os.getenv("ECO_LOCAL_REPLY_TO", "ECOLocal@ecodia.au")
    UNSUB_BASE_URL: str = os.getenv(
        "UNSUB_BASE_URL",
        "https://ecodia.au/eco-local/business/unsubscribe",
    )

    # Gmail/Calendar SA impersonation
    GSUITE_IMPERSONATED_USER: str = os.getenv("GSUITE_IMPERSONATED_USER", "ECOLocal@ecodia.au")
    ECO_LOCAL_GCAL_ID: str = os.getenv("ECO_LOCAL_GCAL_ID", "primary")

    # Allow absolute env paths; otherwise prefer /secrets if mounted.
    # Use basename without .json so the loader can try both.
    GMAIL_SA_PATH: str = os.getenv(
        "GMAIL_SERVICE_ACCOUNT_JSON_PATH",
        str((SECRETS_DIR / _default_sa_basename()).resolve()),
    )
    GCAL_SA_PATH: str = os.getenv(
        "GCAL_SERVICE_ACCOUNT_JSON_PATH",
        str((SECRETS_DIR / _default_sa_basename()).resolve()),
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


def _debug_path_existence(label: str, p: Path) -> str:
    try:
        return f"{label}: exists={p.exists()} is_dir={p.is_dir()} path={p}"
    except Exception as e:
        return f"{label}: <error checking> {p} ({e})"


# Optional startup diagnostics (safe to keep)
if os.getenv("ECO_LOCAL_DEBUG_PATHS", "1") == "1":
    try:
        print(f"APP_ROOT: {APP_ROOT}")
        print(_debug_path_existence("SECRETS_DIR", SECRETS_DIR))
        print(f"GMAIL_SA_PATH: {settings.GMAIL_SA_PATH}")
        print(f"GCAL_SA_PATH:  {settings.GCAL_SA_PATH}")
        print(f"NEO4J_URI:  {settings.NEO4J_URI}")

        # List secrets dir contents if present
        if SECRETS_DIR.exists():
            try:
                print("DEBUG secrets dir contents:", [p.name for p in SECRETS_DIR.iterdir()])
            except Exception as e:
                print("DEBUG secrets dir: unable to list contents:", e)
        else:
            print("DEBUG secrets dir: <missing>", SECRETS_DIR)

        # Also show whether SA candidates exist (basename and .json)
        base = Path(settings.GMAIL_SA_PATH)
        candidates = [base, base.with_suffix(base.suffix or ".json")]
        for c in candidates:
            # If the path includes directories that don't exist, .exists() is False; that's fine.
            print(_debug_path_existence("SA candidate", Path(c)))
    except Exception as _e:
        print("Path debug failed:", _e)
