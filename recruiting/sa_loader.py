# recruiting/sa_loader.py
from __future__ import annotations

import json
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
    # APP_ROOT is the directory containing your app (main.py lives in this folder)
    app_root = os.environ.get("APP_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    secrets_dir = os.environ.get("SECRETS_DIR") or os.path.join(app_root, "secrets")
    return AppPaths(app_root=app_root, secrets_dir=secrets_dir)


_PATHS = _detect_paths()


def _resolve_path(p: str) -> str:
   
    if not p:
        return p

    # Normalize weird whitespace
    p = p.strip()

    # If it's an absolute Windows path like "D:\foo\bar.json" or a POSIX abs path
    if os.path.isabs(p):
        # Special-case "/secrets/..." and "\secrets\..." to mean APP_ROOT/secrets/...
        if p.startswith(("/secrets", "\\secrets")):
            return os.path.join(_PATHS.app_root, p.lstrip("/\\"))
        return p

    # Relative: allow "secrets/..." or "./secrets/..." or "../secrets/..."
    # If it already starts with "secrets", just join to APP_ROOT
    if p.startswith(("secrets/", "secrets\\", "./", ".\\", "../", "..\\")):
        return os.path.abspath(os.path.join(_PATHS.app_root, p))

    # Bare filename: look in APP_ROOT first
    return os.path.abspath(os.path.join(_PATHS.app_root, p))


def _ensure_exists(path: str, label: str) -> str:
    if not os.path.exists(path):
        # Helpful debug dump
        print(f"[sa_loader] {label} not found at: {path}")
        print(f"[sa_loader] APP_ROOT={_PATHS.app_root}")
        print(f"[sa_loader] SECRETS_DIR={_PATHS.secrets_dir}")
        try:
            print(f"[sa_loader] contents of SECRETS_DIR: {os.listdir(_PATHS.secrets_dir)}")
        except Exception as e:
            print(f"[sa_loader] unable to list SECRETS_DIR: {e}")
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def load_sa_credentials(*, scopes: Iterable[str], subject: Optional[str] = None):
    """
    Loads a Google service account from path(s) supplied in env via settings,
    resolving paths relative to APP_ROOT and supporting '/secrets/...' shorthand.
    """
    # Prefer explicit GMAIL/GCAL paths if present; they can be the same file.
    raw_path = (subject and settings.GMAIL_SA_PATH) if subject else settings.GCAL_SA_PATH
    # Fallback to either setting if one is empty
    if not raw_path:
        raw_path = settings.GMAIL_SA_PATH or settings.GCAL_SA_PATH

    # If still not set, use default inside repo
    if not raw_path:
        raw_path = "secrets/eco_local-recruit-sa.json"

    resolved = _resolve_path(raw_path)
    resolved = _ensure_exists(resolved, "Service Account JSON")

    creds = service_account.Credentials.from_service_account_file(resolved, scopes=list(scopes))
    if subject:
        creds = creds.with_subject(subject)
    return creds
