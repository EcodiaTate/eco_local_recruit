#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
from neo4j import GraphDatabase

# Example:
# python .\ecolocaldocs.py .\eco_local_docs.csv ^
#   --title-col title --text-col text --tags-col tags --id-col id ^
#   --limit 10

# ──────────────────────────────────────────────────────────────────────────────
# Defaults / Env
# ──────────────────────────────────────────────────────────────────────────────
DEF_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001").strip() or "gemini-embedding-001"
DEF_DIMS = int(os.getenv("EMBED_DIMENSIONS", "3072"))  # hard-require 3072 below
API_KEY = (os.getenv("GOOGLE_API_KEY", "AIzaSyCJWIow_5C1OZPklEhbdIVGcGYsgM2qsIg") or os.getenv("GEMINI_API_KEY") or "").strip()

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://9f31d51e.databases.neo4j.io")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "dLd8yjfY2ypQxojTqqINMkU3afiPBjzgGklgc6xBwLI")

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _truncate(s: str, n: int = 500) -> str:
    return s if len(s) <= n else s[:n] + f"... <+{len(s)-n} chars>"

def _hash_id(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _parse_tags(raw: str | None) -> List[str]:
    if not raw:
        return []
    raw = raw.strip()
    # accept JSON array
    if raw.startswith("[") and raw.endswith("]"):
        try:
            arr = json.loads(raw)
            return [str(x).strip() for x in arr if str(x).strip()]
        except Exception:
            pass
    # or comma/semicolon separated
    sep = "," if "," in raw else ";"
    return [t.strip() for t in raw.split(sep) if t.strip()]

def _require_api_key():
    if not API_KEY:
        raise OSError("No Google API key. Set GOOGLE_API_KEY or GEMINI_API_KEY.")

def _require_dims(d: int) -> int:
    if d != 3072:
        raise ValueError(f"outputDimensionality must be 3072; got {d}")
    return d

def _mk_driver(uri: str, user: str, pwd: str):
    return GraphDatabase.driver(uri, auth=(user, pwd))

def _neo_create_indexes_autocommit(driver):
    # Run schema ops as separate autocommit transactions
    try:
        with driver.session() as s:
            s.run("""
                CREATE INDEX eco_local_doc_id IF NOT EXISTS
                FOR (d:EcoLocalDoc) ON (d.id)
            """)
    except Exception as e:
        print(f"[neo4j] id-index note: {e}")

    try:
        with driver.session() as s:
            # Might error if exists; that's fine for idempotency
            s.run("""
                CALL db.index.fulltext.createNodeIndex(
                  'EcoLocalDocsText',
                  ['EcoLocalDoc'],
                  ['title','text'],
                  { analyzer: 'standard' }
                )
            """)
    except Exception as e:
        print(f"[neo4j] fulltext note: {e}")

def _neo_merge_docs(tx, rows: List[Dict[str, Any]]):
    """
    rows: [{id,title,text,tags,embedding,embedding_model,embedding_dims,updated_at}]
    """
    cy = """
    UNWIND $rows AS r
    MERGE (d:EcoLocalDoc {id: r.id})
      ON CREATE SET d.created_at = datetime()
    SET d.title = r.title,
        d.text = r.text,
        d.tags = r.tags,
        d.embedding = r.embedding,
        d.embedding_model = r.embedding_model,
        d.embedding_dims = r.embedding_dims,
        d.updated_at = r.updated_at
    RETURN count(d) AS upserted
    """
    tx.run(cy, rows=rows)

# ──────────────────────────────────────────────────────────────────────────────
# Embedding (REST batch)
# ──────────────────────────────────────────────────────────────────────────────
def _gemini_batch_embed(
    texts: List[str],
    *,
    model: str,
    dims: int,
    task_type: str = "RETRIEVAL_DOCUMENT",
    timeout: float = 60.0,
) -> List[List[float]]:
    _require_api_key()
    _require_dims(dims)

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents"
    body = {
        # Keep top-level for compatibility, but some backends require per-request model:
        "model": f"models/{model}",
        "requests": [
            {
                "model": f"models/{model}",  # <- important
                "content": {"parts": [{"text": t}]},
                "taskType": task_type,
                "outputDimensionality": dims,
            }
            for t in texts
        ],
    }
    headers = {"Content-Type": "application/json", "x-goog-api-key": API_KEY}

    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, headers=headers, json=body)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Gemini batchEmbed error {resp.status_code}: {_truncate(resp.text, 1200)}"
            )
        data = resp.json()
        embs = data.get("embeddings")
        if not isinstance(embs, list) or len(embs) != len(texts):
            raise RuntimeError(
                f"Embeddings length mismatch: expected {len(texts)}, got {len(embs) if isinstance(embs, list) else 'N/A'}"
            )
        out: List[List[float]] = []
        for i, e in enumerate(embs):
            vec = e.get("values") or e.get("embedding")
            if not isinstance(vec, list):
                raise RuntimeError(f"Bad embedding at index {i}: {type(vec)}")
            if len(vec) != dims:
                raise RuntimeError(f"Expected {dims} dims at index {i}, got {len(vec)}")
            out.append([float(x) for x in vec])
        return out

def _retry_embed(
    batch: List[str],
    *,
    model: str,
    dims: int,
    max_retries: int = 3,
    base_delay: float = 0.75,
) -> List[List[float]]:
    attempt = 0
    while True:
        try:
            return _gemini_batch_embed(batch, model=model, dims=dims)
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                raise
            sleep_s = base_delay * (2 ** (attempt - 1))
            print(f"[embed/retry] attempt {attempt}/{max_retries} failed: {e} -> sleeping {sleep_s:.2f}s")
            time.sleep(sleep_s)

# ──────────────────────────────────────────────────────────────────────────────
# CSV → records → embed → Neo4j
# ──────────────────────────────────────────────────────────────────────────────
def read_csv(
    path: str,
    *,
    id_col: Optional[str],
    title_col: str,
    text_col: str,
    tags_col: Optional[str],
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            if limit is not None and i >= limit:
                break
            title = (row.get(title_col) or "").strip()
            text = (row.get(text_col) or "").strip()
            if not text and not title:
                continue
            rid = (row.get(id_col).strip() if (id_col and row.get(id_col)) else None)
            if not rid:
                basis = text if text else title
                rid = _hash_id(basis)  # stable id if re-run on same content
            tags = _parse_tags(row.get(tags_col) if tags_col else None)
            out.append({"id": rid, "title": title, "text": text, "tags": tags})
    return out

def chunked(seq: List[Any], size: int) -> List[List[Any]]:
    return [seq[i : i + size] for i in range(0, len(seq), size)]

def ingest(
    csv_path: str,
    *,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    id_col: Optional[str],
    title_col: str,
    text_col: str,
    tags_col: Optional[str],
    batch_size: int,
    model: str,
    dims: int,
    dry_run: bool = False,
    limit: Optional[int] = None,
) -> None:
    docs = read_csv(
        csv_path,
        id_col=id_col,
        title_col=title_col,
        text_col=text_col,
        tags_col=tags_col,
        limit=limit,
    )
    if not docs:
        print("No rows found to import.")
        return

    print(f"Loaded {len(docs)} docs from CSV. Embedding model={model} dims={dims}.")
    driver = _mk_driver(neo4j_uri, neo4j_user, neo4j_password)

    # Prepare indexes (idempotent; each op in autocommit tx)
    _neo_create_indexes_autocommit(driver)

    ts_now = _now_iso()
    upsert_rows: List[Dict[str, Any]] = []

    for group in chunked(docs, batch_size):
        texts = [(g["text"] or g["title"] or "") for g in group]
        # ensure non-empty string to embed
        texts = [t if t.strip() else (g["title"] or " ") for t, g in zip(texts, group)]
        vecs = _retry_embed(texts, model=model, dims=dims)

        for g, v in zip(group, vecs):
            upsert_rows.append({
                "id": g["id"],
                "title": g["title"],
                "text": g["text"],
                "tags": g["tags"],
                "embedding": v,
                "embedding_model": model,
                "embedding_dims": dims,
                "updated_at": ts_now,
            })

        print(f"Embedded batch {len(group)} -> total {len(upsert_rows)}/{len(docs)}")

    if dry_run:
        print("[dry-run] Skipping Neo4j write.")
        driver.close()
        return

    with driver.session() as s:
        s.execute_write(_neo_merge_docs, upsert_rows)
        print(f"[neo4j] Upserted {len(upsert_rows)} EcoLocalDoc nodes.")

    driver.close()
    print("Done.")

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Import CSV into Neo4j as EcoLocalDoc with Gemini 3072 embeddings."
    )
    parser.add_argument("csv", help="Path to CSV file.")
    parser.add_argument("--neo4j-uri", default=NEO4J_URI)
    parser.add_argument("--neo4j-user", default=NEO4J_USER)
    parser.add_argument("--neo4j-password", default=NEO4J_PASSWORD)

    parser.add_argument("--id-col", default=None, help="CSV column for stable id (optional).")
    parser.add_argument("--title-col", default="title", help="CSV column for title.")
    parser.add_argument("--text-col", default="text", help="CSV column for main text.")
    parser.add_argument("--tags-col", default=None, help="CSV column for tags (JSON array or CSV/; list).")

    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size.")
    parser.add_argument("--model", default=DEF_MODEL, help="Gemini embedding model (default: gemini-embedding-001).")
    parser.add_argument("--dims", type=int, default=DEF_DIMS, help="Output dimensionality (must be 3072).")

    parser.add_argument("--limit", type=int, default=None, help="Limit rows for a trial run.")
    parser.add_argument("--dry-run", action="store_true", help="Do everything except write to Neo4j.")

    args = parser.parse_args()

    try:
        ingest(
            args.csv,
            neo4j_uri=args.neo4j_uri,
            neo4j_user=args.neo4j_user,
            neo4j_password=args.neo4j_password,
            id_col=args.id_col,
            title_col=args.title_col,
            text_col=args.text_col,
            tags_col=args.tags_col,
            batch_size=max(1, args.batch_size),
            model=args.model,
            dims=_require_dims(args.dims),
            dry_run=args.dry_run,
            limit=args.limit,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
