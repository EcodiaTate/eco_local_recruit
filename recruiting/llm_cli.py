# recruiting/llm_cli.py
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Import your existing client helpers
from .llm_client import generate_json, _DEFAULT_MODEL  # type: ignore


def _read_stdin() -> str:
    try:
        data = sys.stdin.read()
        return data
    except KeyboardInterrupt:
        return ""


def _load_prompt(args: argparse.Namespace) -> Any:
    """
    Load the prompt from --prompt, --input, or STDIN.
    If --json-prompt is set, parse JSON into an object/list for llm_client.
    Otherwise keep as a plain string.
    """
    source = None

    if args.prompt is not None:
        source = args.prompt
    elif args.input is not None:
        if args.input == "-":
            source = _read_stdin()
        else:
            p = Path(args.input)
            if not p.exists():
                print(f"error: input file not found: {p}", file=sys.stderr)
                sys.exit(2)
            source = p.read_text(encoding="utf-8")
    else:
        # If no explicit prompt/file, read from stdin (if any)
        if not sys.stdin.isatty():
            source = _read_stdin()

    if source is None or str(source).strip() == "":
        print("error: no prompt provided (use --prompt, --input FILE, or pipe via STDIN)", file=sys.stderr)
        sys.exit(2)

    if args.json_prompt:
        try:
            return json.loads(source)
        except json.JSONDecodeError as e:
            print(f"error: --json-prompt provided but input is not valid JSON: {e}", file=sys.stderr)
            sys.exit(2)

    # Keep as a plain string
    return source


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m recruiting.llm_cli",
        description="Send a single chat completion request using recruiting.llm_client.generate_json().",
    )
    g_in = parser.add_argument_group("input")
    g_in.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Inline prompt text. If omitted, use --input or STDIN.",
    )
    g_in.add_argument(
        "--input",
        type=str,
        default=None,
        help="Read prompt from FILE or '-' for STDIN.",
    )
    g_in.add_argument(
        "--json-prompt",
        action="store_true",
        help="Treat the prompt as JSON (object/list). Useful for structured prompts.",
    )

    g_req = parser.add_argument_group("request")
    g_req.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Model name (overrides env LLM_MODEL, default in client is '{_DEFAULT_MODEL}').",
    )
    g_req.add_argument(
        "--max-tokens",
        type=int,
        default=800,
        help="Max output tokens. (Default: 800)",
    )

    g_out = parser.add_argument_group("output")
    g_out.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    g_out.add_argument(
        "--strict-exit",
        action="store_true",
        help="Exit with code 1 if the response is empty ({}).",
    )

    args = parser.parse_args(argv)

    prompt = _load_prompt(args)

    # Fire the request via your llm client
    obj = generate_json(prompt, model=args.model, max_output_tokens=args.max_tokens)

    # Print to stdout
    if args.pretty:
        print(json.dumps(obj, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))

    # Non-zero exit if requested and response is empty
    if args.strict_exit and (not isinstance(obj, dict) or len(obj) == 0):
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
