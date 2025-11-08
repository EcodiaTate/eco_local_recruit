# recruiting/sa_loader.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Optional

from google.oauth2 import service_account
from .config import settings


@dataclass(frozen=True)
class AppPaths:
    app_root: str
    secrets_dir: str


def _detect_paths() -> AppPaths:
    """
    APP_ROOT is the repo/app root (where main.py lives).
    SECRETS_DIR prefers env if set; otherwise:
      - if /secrets exists in the container (Cloud Run secret mount), use that
      - else fall back to APP_ROOT/secrets (local dev)
    """
    app_root = os.environ.get("APP_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    env_secrets = os.environ.get("SECRETS_DIR")
    if env_secrets:
        secrets_dir = env_secrets
    elif os.path.isdir("/secrets"):
        secrets_dir = "/secrets"
    else:
        secrets_dir = os.path.join(app_root, "secrets")

    return AppPaths(app_root=app_root, secrets_dir=secrets_dir)


_PATHS = _detect_paths()


def _resolve_path(p: str) -> str:
    """
    Resolution rules:
      - Absolute paths are respected verbatim (no rewriting). This includes "/secrets/...".
      - Relative paths beginning with "secrets/" (or ./secrets, ../secrets) resolve under SECRETS_DIR.
      - Other relative/bare filenames try SECRETS_DIR first, then APP_ROOT.
    """
    if not p:
        return p

    p = p.strip()

    # Absolute: respect as-is (DO NOT rewrite /secrets to /app/secrets).
    if os.path.isabs(p):
        return p

    # Explicit secrets-relative forms
    if p.startswith(("secrets/", "secrets\\", "./secrets", ".\\secrets", "../secrets", "..\\secrets")):
        rel = p.replace("\\", "/")
        # strip leading ./ or .\ or ../ etc.
        while rel.startswith(("./", ".\\", "../", "..\\")):
            rel = rel[2:] if rel.startswith(("./", ".\\")) else rel[3:]
        # now rel should start with "secrets/"
        rel = rel.split("/", 1)[1] if "/" in rel else ""
        return os.path.join(_PATHS.secrets_dir, rel)

    # Bare filename or other relative: prefer secrets_dir, then app_root
    cand1 = os.path.join(_PATHS.secrets_dir, p)
    if os.path.exists(cand1):
        return os.path.abspath(cand1)
    return os.path.abspath(os.path.join(_PATHS.app_root, p))


def _debug_dump_missing(label: str, path: str) -> None:
    print(f"[sa_loader] {label} not found at: {path}")
    print(f"[sa_loader] APP_ROOT={_PATHS.app_root}")
    print(f"[sa_loader] SECRETS_DIR={_PATHS.secrets_dir}")
    try:
        print(f"[sa_loader] contents of SECRETS_DIR: {os.listdir(_PATHS.secrets_dir)}")
    except Exception as e:
        print(f"[sa_loader] unable to list SECRETS_DIR: {e}")


def _first_existing(paths: list[str]) -> Optional[str]:
    for x in paths:
        if x and os.path.exists(x):
            return x
    return None


def _ensure_exists_any(candidates: list[str], label: str) -> str:
    found = _first_existing(candidates)
    if found:
        return found
    # dump the first as the "primary" for error context
    _debug_dump_missing(label, candidates[0])
    raise FileNotFoundError(f"{label} not found: tried {candidates}")


def load_sa_credentials(*, scopes: Iterable[str], subject: Optional[str] = None):
    """
    Loads a Google service account from path(s) supplied in env via settings,
    resolving paths relative to SECRETS_DIR and supporting absolute '/secrets/...'.
    Also tries the same path with '.json' appended if the first candidate is missing.
    """
    # inside load_sa_credentials() just before _ensure_exists_any(...)

    # Prefer explicit GMAIL/GCAL paths if present; they can be the same file.
    raw_path = (subject and settings.GMAIL_SA_PATH) if subject else settings.GCAL_SA_PATH
    if not raw_path:
        raw_path = settings.GMAIL_SA_PATH or settings.GCAL_SA_PATH

    # Still not set? default to a sensible filename under secrets_dir
    if not raw_path:
        raw_path = "secrets/eco-local-recruit-sa"  # no suffix required; we'll try with and without .json

    resolved = _resolve_path(raw_path)

    # Try both exact and with '.json' suffix
    candidates = [resolved]
    if not resolved.endswith(".json"):
        candidates.append(resolved + ".json")

    path = _ensure_exists_any(candidates, "Service Account JSON")
    # inside load_sa_credentials() just before _ensure_exists_any(...)
    print(f"[sa_loader] raw_path={raw_path}")
    print(f"[sa_loader] resolved={resolved}")
    for c in candidates:
        print(f"[sa_loader] candidate={c} exists={os.path.exists(c)}")

    creds = service_account.Credentials.from_service_account_file(path, scopes=list(scopes))
    if subject:
        creds = creds.with_subject(subject)
    return creds
