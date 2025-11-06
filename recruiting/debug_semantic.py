# recruiting/debug_semantic.py (edit)
from fastapi import APIRouter, Query
from typing import Optional, Dict, Any
from .tools import build_semantic_query, semantic_docs_raw, thread_context as _thread_context, semantic_status

router = APIRouter(prefix="/debug/semantic", tags=["debug-semantic"])

@router.get("/status")
def debug_semantic_status() -> Dict[str, Any]:
    return semantic_status()

@router.get("/query")
def debug_semantic_query(
    query: str = Query(..., description="Raw query text"),
    k: int = Query(5, ge=1, le=100),
    loose: int = Query(0, description="1 to enable loose fulltext fallback"),
) -> Dict[str, Any]:
    res = semantic_docs_raw(query, k=k, loose_fulltext=bool(loose))
    docs = [{
        "id": d.get("id"),
        "title": d.get("title"),
        "score": d.get("score"),
        "tags": d.get("tags") or [],
        "snippet_300": (d.get("text") or "")[:300],
    } for d in res.get("docs", [])]
    return {"mode": res.get("source"), "k": k, "query_used": res.get("query"), "results": docs}

@router.get("/thread")
def debug_semantic_thread(
    subject: Optional[str] = Query("", description="Email subject"),
    body: Optional[str] = Query("", description="Latest email/plain body"),
    thread_id: Optional[str] = Query(None, description="Optional thread id to pull context tail"),
    k: int = Query(5, ge=1, le=100),
    loose: int = Query(0),
) -> Dict[str, Any]:
    tail = _thread_context(thread_id)[-1200:] if thread_id else ""
    qb = build_semantic_query(subject=subject or "", body=body or "", thread_text=tail, max_thread_tail=1200)
    res = semantic_docs_raw(qb["query"], k=k, loose_fulltext=bool(loose))
    docs = [{
        "id": d.get("id"),
        "title": d.get("title"),
        "score": d.get("score"),
        "tags": d.get("tags") or [],
        "snippet_300": (d.get("text") or "")[:300],
    } for d in res.get("docs", [])]
    return {"mode": res.get("source"), "k": k, "query_parts": qb["parts"], "query_used": res.get("query"), "results": docs}
