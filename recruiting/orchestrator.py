# recruiting/orchestrator.py
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List

from .scrape import scrape_all  # async: returns list[raw]
from .parse import parse_batch  # sync: List[raw] -> List[dict]
from .qualify import qualify    # sync: dict -> dict (adds score/flags)
from .profile import bulk_upsert  # sync: List[dict] -> None

log = logging.getLogger("orchestrator")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# --- helpers ---------------------------------------------------------------

async def _with_retry(coro_factory, *, retries: int = 3, base_delay: float = 0.5) -> Any:
    """Minimal async retry wrapper for transient scrape errors."""
    attempt = 0
    last_exc: BaseException | None = None
    while attempt <= retries:
        try:
            return await coro_factory()
        except Exception as e:
            last_exc = e
            if attempt == retries:
                break
            delay = base_delay * (2 ** attempt)
            log.warning("retrying after error (attempt %s/%s): %s", attempt + 1, retries, e)
            await asyncio.sleep(delay)
            attempt += 1
    assert last_exc is not None
    raise last_exc

# --- public orchestrations -------------------------------------------------

async def run_scrape_pipeline(*, max_concurrency: int = 8) -> Dict[str, int]:
    """
    1) scrape_all(): async aggregate fetch (internally can use semaphores)
    2) parse_batch(): normalize + extract fields
    3) qualify(): score/filter each profile
    4) bulk_upsert(): persist to your store (Neo4j/Postgres/etc.)
    Returns basic counts for observability.
    """
    log.info("scrape: starting")
    raw_list: List[Any] = await _with_retry(lambda: scrape_all())
    log.info("scrape: got %d raw items", len(raw_list))

    parsed: List[Dict[str, Any]] = parse_batch(raw_list)
    log.info("parse: produced %d parsed items", len(parsed))

    # If qualify is CPU-light, do it sync; otherwise, run in a small thread-pool
    qualified: List[Dict[str, Any]] = [qualify(p) for p in parsed]
    kept = [q for q in qualified if not q.get("disqualified")]
    log.info("qualify: kept=%d dropped=%d", len(kept), len(qualified) - len(kept))

    if kept:
        bulk_upsert(kept)
        log.info("upsert: wrote %d items", len(kept))
    else:
        log.info("upsert: nothing to write")

    return {"raw": len(raw_list), "parsed": len(parsed), "kept": len(kept)}

# --- CLI entrypoint --------------------------------------------------------

async def main() -> None:
    stats = await run_scrape_pipeline()
    log.info("DONE stats=%s", stats)

if __name__ == "__main__":
    asyncio.run(main())
