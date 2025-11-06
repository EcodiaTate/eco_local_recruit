# recruiting/llm_client.py
from __future__ import annotations
import json
import os
import time
import logging
from typing import Any, Dict, Optional, List, Tuple

from openai import OpenAI
from openai.types.chat import ChatCompletion
from openai import BadRequestError, APIStatusError, APIError, RateLimitError

log = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)

_DEFAULT_MODEL = os.getenv("LLM_MODEL", "gpt-4o").strip()
_client = OpenAI()  # reads OPENAI_API_KEY / OPENAI_BASE_URL if set


# ─────────────────────────────────────────────────────────
# JSON helpers
# ─────────────────────────────────────────────────────────
def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Find the first top-level {...} JSON object in text with a tiny stack parser."""
    if not text:
        return None
    start = -1
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if start == -1:
            if ch == "{":
                start = i
                depth = 1
                in_string = False
                escape = False
        else:
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        frag = text[start : i + 1]
                        try:
                            return json.loads(frag)
                        except Exception:
                            start = -1  # keep scanning
    return None


def _flatten_output_text(resp: ChatCompletion | Any) -> Tuple[str, Optional[str]]:
    """
    Return (text, finish_reason). Works across common SDK shapes.
    """
    try:
        choice = resp.choices[0]
        content = getattr(choice.message, "content", "") or ""
        finish = getattr(choice, "finish_reason", None)
        return str(content), (str(finish) if finish else None)
    except Exception:
        return ("", None)

# recruiting/llm_client.py
def _to_messages(prompt: Any) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = []
    if isinstance(prompt, list):
        # Ensure at least ONE system message explicitly mentions 'json'
        has_json_hint = False
        for item in prompt:
            if isinstance(item, dict) and item.get("system") and ("json" in item["system"].lower()):
                has_json_hint = True
                break
        if not has_json_hint:
            msgs.append({"role": "system", "content": "return only a valid json object. no prose."})

        for item in prompt:
            if not isinstance(item, dict):
                continue
            sys = item.get("system")
            usr = item.get("user")
            if sys:
                msgs.append({"role": "system", "content": str(sys)})
            if usr is not None:
                msgs.append({
                    "role": "user",
                    "content": json.dumps(usr, separators=(",", ":"), ensure_ascii=False),
                })
        return msgs
    if isinstance(prompt, str):
        return [
            {"role": "system", "content": "Return ONLY a valid JSON object. No prose."},
            {"role": "user", "content": prompt},
        ]
    return [
        {"role": "system", "content": "Return ONLY a valid JSON object. No prose."},
        {"role": "user", "content": json.dumps(prompt, separators=(",", ":"), ensure_ascii=False)},
    ]


def _parse_json(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    obj = _extract_first_json_object(text)
    if obj is not None:
        return obj
    if text.lstrip().startswith("{"):
        try:
            return json.loads(text)
        except Exception:
            pass
    return {"text": text}


# ─────────────────────────────────────────────────────────
# Core
# ─────────────────────────────────────────────────────────
def _chat_once(
    *,
    model: str,
    messages: List[Dict[str, str]],
    json_mode: bool,
    max_output_tokens: int,
) -> Tuple[Dict[str, Any], Optional[str], Optional[str], Optional[int], Optional[str]]:
    """
    Single request.
    Returns (parsed_obj, finish_reason, request_id, http_status, error_kind)
      - http_status: int if we caught an HTTP error
      - error_kind:  short label e.g. 'bad_request', 'rate_limit', 'server'
    """
    req_id: Optional[str] = None
    finish: Optional[str] = None
    try:
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_output_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        chat: ChatCompletion = _client.chat.completions.create(**kwargs)  # type: ignore[arg-type]
        req_id = getattr(chat, "id", None)
        text, finish = _flatten_output_text(chat)
        obj = _parse_json(text)
        return obj, finish, req_id, None, None

    except BadRequestError as e:
        # True 4xx client error — do NOT retry this mode; pivot immediately.
        status = getattr(getattr(e, "response", None), "status_code", 400)
        detail = getattr(e, "message", str(e))
        log.info("[llm] 400 BadRequest json_mode=%s model=%s: %s", json_mode, model, detail)
        return {}, None, None, status, "bad_request"

    except RateLimitError as e:
        status = getattr(getattr(e, "response", None), "status_code", 429)
        log.warning("[llm] 429 rate limit: %s", getattr(e, "message", str(e)))
        return {}, None, None, status, "rate_limit"

    except APIStatusError as e:
        # Non-2xx from API with known status (often 5xx)
        status = getattr(e, "status_code", None)
        log.warning("[llm] APIStatusError %s json_mode=%s: %s", status, json_mode, getattr(e, "message", str(e)))
        return {}, None, None, status, "server" if status and status >= 500 else "other"

    except APIError as e:
        log.warning("[llm] APIError json_mode=%s: %s", json_mode, getattr(e, "message", str(e)))
        return {}, None, None, None, "api_error"

    except Exception as e:
        # Unknown/local error
        log.debug("[llm] chat_once failed json_mode=%s: %s", json_mode, e)
        return {}, None, None, None, "local"


def generate_json(
    prompt: Any,
    *,
    model: str | None = None,
    max_output_tokens: int = 800,
) -> Dict[str, Any]:
    """
    Deterministic JSON fetch with graceful fallbacks:
      1) Try Chat JSON mode (response_format=json_object) ONCE. If 4xx, skip further retries in that mode.
         - If truncated ('length'), retry once with a larger token budget.
      2) Fallback: Plain chat (no response_format) with light retry/backoff for transient errors.
      3) On hard failure, return {}.
    """
    mdl = (model or _DEFAULT_MODEL).strip()
    messages = _to_messages(prompt)

    def _try_json_mode(max_tokens: int) -> Tuple[Dict[str, Any], Optional[str], bool]:
        # returns (obj, finish, retriable)
        obj, finish, req_id, status, kind = _chat_once(
            model=mdl, messages=messages, json_mode=True, max_output_tokens=max_tokens
        )
        if obj:
            return obj, finish, False
        # If explicit 4xx → do not retry this mode
        if status and 400 <= status < 500:
            return {}, None, False
        # Otherwise (5xx/429/local) allow a single retry after a brief backoff
        time.sleep(0.4)
        obj2, finish2, _, status2, kind2 = _chat_once(
            model=mdl, messages=messages, json_mode=True, max_output_tokens=max_tokens
        )
        return obj2, finish2, False  # either we got it or we move on

    # 1) JSON mode first (no triple retry on 4xx)
    obj, finish, _ = _try_json_mode(max_output_tokens)
    if obj:
        if finish == "length":
            obj2, _, _ = _try_json_mode(min(max_output_tokens * 2, 4000))
            return obj2 or obj
        return obj

    # 2) Fallback: plain chat with small retry loop for transient issues
    attempts = 0
    backoff = 0.5
    last_obj: Dict[str, Any] = {}
    last_finish: Optional[str] = None
    while attempts < 3:
        o, f, _, status, kind = _chat_once(
            model=mdl, messages=messages, json_mode=False, max_output_tokens=max_output_tokens
        )
        if o:
            if f == "length":
                # one larger retry
                o2, f2, *_ = _chat_once(
                    model=mdl, messages=messages, json_mode=False, max_output_tokens=min(max_output_tokens * 2, 4000)
                )
                return o2 or o
            return o
        attempts += 1
        # Only back off for transient cases; 4xx in plain mode usually means prompt issue — but still bail quickly.
        if status and 400 <= status < 500:
            break
        time.sleep(backoff)
        backoff *= 1.7

    # 3) Hard failure
    return {}
