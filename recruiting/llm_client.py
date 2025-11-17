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

# ─────────────────────────────────────────────────────────
# Defaults / ENV
# ─────────────────────────────────────────────────────────
_DEFAULT_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()

# GPT-5 controls (honored only when model startswith "gpt-5")
_GPT5_REASONING = os.getenv("LLM_GPT5_REASONING", "minimal").strip()  # minimal|low|medium|high
_GPT5_VERBOSITY = os.getenv("LLM_GPT5_VERBOSITY", "low").strip()      # low|medium|high

_client = OpenAI()  # respects OPENAI_API_KEY/BASE_URL etc.

# ─────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────
def _is_gpt5(m: str) -> bool:
    m = (m or "").lower()
    return m.startswith("gpt-5")

def _responses_messages(messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Convert simple {role, content} messages to Responses API message objects with
    explicit text parts. This avoids the fragile 'single concatenated string' input.
    """
    out: List[Dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = str(msg.get("content", ""))
        out.append({
            "role": role,
            "content": [{"type": "input_text", "text": content}],
        })
    return out

def _extract_output_text(resp) -> str:
    """
    Robust extraction for Responses API:
      - prefer resp.output_text if present
      - else walk resp.output[*].content[*].text
    """
    t = getattr(resp, "output_text", None)
    if isinstance(t, str) and t.strip():
        return t.strip()

    collected: List[str] = []
    output = getattr(resp, "output", None) or []
    for item in output:
        segs = getattr(item, "content", None) or []
        if isinstance(segs, list):
            for seg in segs:
                txt = getattr(seg, "text", None)
                if txt:
                    collected.append(str(txt))
    return "".join(collected).strip()

def _extract_first_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    start = -1
    depth = 0
    in_string = False
    esc = False
    for i, ch in enumerate(text):
        if start < 0:
            if ch == "{":
                start = i; depth = 1; in_string = False; esc = False
        else:
            if in_string:
                if esc: esc = False
                elif ch == "\\": esc = True
                elif ch == '"': in_string = False
            else:
                if ch == '"': in_string = True
                elif ch == "{": depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        frag = text[start:i+1]
                        try:
                            return json.loads(frag)
                        except Exception:
                            start = -1
    return None

def _parse_json_maybe(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    obj = _extract_first_json(text)
    if obj is not None:
        return obj
    if text.lstrip().startswith("{"):
        try:
            return json.loads(text)
        except Exception:
            pass
    return {"text": text}

# ─────────────────────────────────────────────────────────
# Message shaping
# ─────────────────────────────────────────────────────────
def _to_messages_for_text(prompt: Any, system: Optional[str]) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = []
    if system:
        msgs.append({"role": "system", "content": system})
    if isinstance(prompt, list):
        for item in prompt:
            if isinstance(item, dict) and item.get("role") and item.get("content") is not None:
                msgs.append({"role": str(item["role"]), "content": str(item["content"])})
        return msgs
    if isinstance(prompt, str):
        msgs.append({"role": "user", "content": prompt})
        return msgs
    # object → JSON string
    msgs.append({"role": "user", "content": json.dumps(prompt, ensure_ascii=False, separators=(",", ":"))})
    return msgs
def _to_messages_for_json(prompt: Any) -> List[Dict[str, str]]:
    # Always add a JSON-only guard system msg
    msgs: List[Dict[str, str]] = [
        {"role": "system", "content": "Return ONLY a valid JSON object. No prose."}
    ]

    if isinstance(prompt, list):
        for item in prompt:
            if not isinstance(item, dict):
                continue

            # 1) Already-shaped chat messages
            if item.get("role") and item.get("content") is not None:
                content = item["content"]
                if not isinstance(content, str):
                    content = json.dumps(content, ensure_ascii=False, separators=(",", ":"))
                msgs.append({"role": str(item["role"]), "content": str(content)})
                continue

            # 2) Our {system, user} pattern from llm_flow
            if "system" in item or "user" in item:
                sys_txt = item.get("system")
                usr = item.get("user")
                if sys_txt:
                    msgs.append({"role": "system", "content": str(sys_txt)})
                if usr is not None:
                    if not isinstance(usr, str):
                        usr = json.dumps(usr, ensure_ascii=False, separators=(",", ":"))
                    msgs.append({"role": "user", "content": usr})
                continue

        return msgs

    # Non-list prompt: keep old behaviour
    if isinstance(prompt, str):
        msgs.append({"role": "user", "content": prompt})
        return msgs

    # Object → single JSON blob as user
    msgs.append({
        "role": "user",
        "content": json.dumps(prompt, ensure_ascii=False, separators=(",", ":")),
    })
    return msgs


# ─────────────────────────────────────────────────────────
# Core request shims
# ─────────────────────────────────────────────────────────
def _responses_request(
    *, model: str, messages: List[Dict[str, str]], json_mode: bool, max_output_tokens: int
) -> Tuple[str, Optional[str], Optional[int], Optional[str]]:
    """
    Call the Responses API (for GPT-5 models). Never send temperature/top_p/logprobs.
    Returns: (text, finish_reason, http_status, error_kind)
    """
    try:
        kwargs: Dict[str, Any] = {
            "model": model,
            "input": _responses_messages(messages),
            "max_output_tokens": max_output_tokens,
            "reasoning": {"effort": _GPT5_REASONING},
            "text": {"verbosity": _GPT5_VERBOSITY},
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        resp = _client.responses.create(**kwargs)
        text = _extract_output_text(resp)
        finish = getattr(resp, "finish_reason", None)

        if not text:
            # Make the failure visible during bring-up
            log.warning("[llm][responses] empty output (model=%s json=%s). raw_status=%s", model, json_mode, getattr(resp, "status", None))

        return text, finish, None, None

    except BadRequestError as e:
        status = getattr(getattr(e, "response", None), "status_code", 400)
        msg = getattr(e, "message", str(e))
        log.error("[llm][responses] %s BadRequest model=%s json=%s: %s", status, model, json_mode, msg)
        return "", None, status, "bad_request"
    except RateLimitError as e:
        status = getattr(getattr(e, "response", None), "status_code", 429)
        log.warning("[llm][responses] 429: %s", getattr(e, "message", str(e)))
        return "", None, status, "rate_limit"
    except APIStatusError as e:
        status = getattr(e, "status_code", None)
        log.warning("[llm][responses] APIStatus %s: %s", status, getattr(e, "message", str(e)))
        return "", None, status, "server" if status and status >= 500 else "other"
    except APIError as e:
        log.warning("[llm][responses] APIError: %s", getattr(e, "message", str(e)))
        return "", None, None, "api_error"
    except Exception as e:
        log.debug("[llm][responses] local error: %s", e)
        return "", None, None, "local"

def _chat_request(
    *, model: str, messages: List[Dict[str, str]], json_mode: bool, max_output_tokens: int, temperature: Optional[float]
) -> Tuple[str, Optional[str], Optional[int], Optional[str]]:
    """
    Chat Completions path - only for non-GPT-5 models. Supports max_completion_tokens and temperature.
    Returns: (text, finish_reason, http_status, error_kind)
    """
    assert not _is_gpt5(model), "Chat Completions should not be used for GPT-5 models."

    def build_kwargs(token_param: str, include_temp: bool) -> Dict[str, Any]:
        kw: Dict[str, Any] = {"model": model, "messages": messages}
        if token_param == "max_completion_tokens":
            kw["max_completion_tokens"] = max_output_tokens
        else:
            kw["max_tokens"] = max_output_tokens
        if include_temp and temperature is not None:
            kw["temperature"] = temperature
        if json_mode:
            kw["response_format"] = {"type": "json_object"}
        return kw

    for token_param in ("max_completion_tokens", "max_tokens"):
        for include_temp in ((temperature is not None), False):
            try:
                chat: ChatCompletion = _client.chat.completions.create(**build_kwargs(token_param, include_temp))  # type: ignore[arg-type]
                choice = chat.choices[0]
                text = (getattr(choice.message, "content", "") or "").strip()
                finish = getattr(choice, "finish_reason", None)
                return text, (str(finish) if finish else None), None, None
            except BadRequestError as e:
                msg = (getattr(e, "message", str(e)) or "").lower()
                status = getattr(getattr(e, "response", None), "status_code", 400)
                if "unsupported parameter" in msg and ("max_tokens" in msg or "max_completion_tokens" in msg):
                    break
                if "temperature" in msg:
                    continue
                log.error("[llm][chat] %s BadRequest model=%s json=%s: %s", status, model, json_mode, e)
                return "", None, status, "bad_request"
            except RateLimitError as e:
                status = getattr(getattr(e, "response", None), "status_code", 429)
                return "", None, status, "rate_limit"
            except APIStatusError as e:
                status = getattr(e, "status_code", None)
                return "", None, status, "server" if status and status >= 500 else "other"
            except APIError:
                return "", None, None, "api_error"
            except Exception:
                return "", None, None, "local"

    return "", None, None, "bad_request"

# ─────────────────────────────────────────────────────────
# Public: plain text
# ─────────────────────────────────────────────────────────
def generate_text(
    prompt: Any,
    *,
    model: Optional[str] = None,
    max_output_tokens: int = 512,
    system: Optional[str] = None,
    temperature: Optional[float] = None,   # ignored for GPT-5 per docs
    retries: int = 2,
    backoff_start: float = 0.4,
) -> str:
    mdl = (model or _DEFAULT_MODEL).strip()
    messages = _to_messages_for_text(prompt, system)

    attempt = 0
    backoff = backoff_start
    while True:
        if _is_gpt5(mdl):
            text, finish, status, kind = _responses_request(
                model=mdl, messages=messages, json_mode=False, max_output_tokens=max_output_tokens
            )
        else:
            text, finish, status, kind = _chat_request(
                model=mdl, messages=messages, json_mode=False, max_output_tokens=max_output_tokens, temperature=temperature
            )

        if text:
            if finish == "length" and max_output_tokens < 4000:
                if _is_gpt5(mdl):
                    t2, *_ = _responses_request(
                        model=mdl, messages=messages, json_mode=False, max_output_tokens=min(max_output_tokens * 2, 4000)
                    )
                else:
                    t2, *_ = _chat_request(
                        model=mdl, messages=messages, json_mode=False, max_output_tokens=min(max_output_tokens * 2, 4000), temperature=temperature
                    )
                return t2 or text
            return text

        attempt += 1
        if attempt > retries:
            return ""

        if status and 400 <= status < 500:
            return ""

        time.sleep(backoff)
        backoff *= 1.7

# ─────────────────────────────────────────────────────────
# Public: JSON
# ─────────────────────────────────────────────────────────
def generate_json(
    prompt: Any,
    *,
    model: Optional[str] = None,
    max_output_tokens: int = 800,
    temperature: Optional[float] = None,   # ignored for GPT-5 per docs
) -> Dict[str, Any]:
    mdl = (model or _DEFAULT_MODEL).strip()

    # IMPORTANT: do NOT double-encode your dict. Pass the dict directly.
    messages = _to_messages_for_json(prompt)

    if _is_gpt5(mdl):
        text, finish, status, kind = _responses_request(
            model=mdl, messages=messages, json_mode=True, max_output_tokens=max_output_tokens
        )
        if text:
            obj = _parse_json_maybe(text)
            if finish == "length" and max_output_tokens < 4000:
                t2, *_ = _responses_request(
                    model=mdl, messages=messages, json_mode=True, max_output_tokens=min(max_output_tokens * 2, 4000)
                )
                return _parse_json_maybe(t2) if t2 else obj
            return obj

        if status and 400 <= status < 500:
            # Harden: return error shape instead of {}
            return {"_error": {"kind": "client", "status": status}}

        # Fallback to plain text extraction, still on Responses API
        t, f, status2, _ = _responses_request(
            model=mdl, messages=messages, json_mode=False, max_output_tokens=max_output_tokens
        )
        if t:
            if f == "length" and max_output_tokens < 4000:
                t2, *_ = _responses_request(
                    model=mdl, messages=messages, json_mode=False, max_output_tokens=min(max_output_tokens * 2, 4000)
                )
                return _parse_json_maybe(t2) if t2 else _parse_json_maybe(t)
            return _parse_json_maybe(t)
        return {}

    # Non-GPT-5 path (legacy models): JSON mode via Chat Completions
    text, finish, status, kind = _chat_request(
        model=mdl, messages=messages, json_mode=True, max_output_tokens=max_output_tokens, temperature=temperature
    )
    if text:
        obj = _parse_json_maybe(text)
        if finish == "length" and max_output_tokens < 4000:
            t2, *_ = _chat_request(
                model=mdl, messages=messages, json_mode=True, max_output_tokens=min(max_output_tokens * 2, 4000), temperature=temperature
            )
            return _parse_json_maybe(t2) if t2 else obj
        return obj

    if status and 400 <= status < 500:
        return {"_error": {"kind": "client", "status": status}}

    # Last resort: plain chat and parse
    t, f, status2, _ = _chat_request(
        model=mdl, messages=messages, json_mode=False, max_output_tokens=max_output_tokens, temperature=temperature
    )
    if t:
        if f == "length" and max_output_tokens < 4000:
            t2, *_ = _chat_request(
                model=mdl, messages=messages, json_mode=False, max_output_tokens=min(max_output_tokens * 2, 4000), temperature=temperature
            )
            return _parse_json_maybe(t2) if t2 else _parse_json_maybe(t)
        return _parse_json_maybe(t)
    return {}
