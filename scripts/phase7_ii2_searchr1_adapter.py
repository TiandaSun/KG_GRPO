"""Phase 7 Action II2 / V14-A1: Search-R1 Freebase adapter.

Search-R1 (arXiv:2503.09516) ships with a single-tool `search(query)` interface
backed by BM25/dense retrieval over Wikipedia. We wrap our 4-tool Freebase API
as the same single-tool interface so Search-R1's unmodified training/eval code
can target CWQ/Freebase.

Wire protocol (matches Search-R1's retrieval_server.py and generation.py):

  POST /retrieve
  request  = {"queries": [str, ...], "topk": int, "return_scores": bool}
  response = {"result": [
      [  # per-query list of document dicts
          {"document": {"contents": "Title\nBody text ..."}, "score": 1.0},
          ...
      ],
      ...
  ]}

Strategy for mapping free-form Search-R1 queries onto our 4-tool Freebase API:
  - If the query contains a Freebase-style predicate (e.g. `people.person.place_of_birth`)
    treat "<entity> <relation>" as get_tail_entities(entity, relation).
  - Otherwise interpret as "give me facts about <entity>": call get_tail_relations
    and synthesize pseudo-documents enumerating outgoing relations.
  - Fallback: return a single document stating no results.

Reviewer purpose: prove that Search-R1's outcome-only recipe generalizes (or
fails to generalize) to KG multi-hop. If Search-R1 > E3 on CWQ, our process-
reward diagnostic is weakened; we need to know this NOW, not at rebuttal.

Usage:
  1. Start the Freebase KG server on port 18901 (as normal):
       python -m src_verl.kg_server.server --kg freebase --freebase_dir data/freebase/kg --port 18901
  2. Start this adapter on port 19001:
       python scripts/phase7_ii2_searchr1_adapter.py --freebase_port 18901 --port 19001
  3. Point Search-R1 training at http://localhost:19001/retrieve
"""
from __future__ import annotations

import argparse
import logging
import re
from typing import Any

import requests

logger = logging.getLogger(__name__)


def parse_query_to_call(query: str) -> tuple[str, str, str | None]:
    """Heuristic: map a free-form natural-language query to one of our 4 tools.

    Returns (action, entity, relation) where action is the KG-server action
    name: get_tail_entities | get_tail_relations | get_head_entities |
    get_head_relations.
    """
    q = query.strip()

    # Pattern 1: "relations of X" or "what can X do"
    m = re.match(
        r"(?:relations?\s+of|relations?\s+for|what\s+(?:are|can)\s+)\s*(.+)",
        q,
        re.IGNORECASE,
    )
    if m:
        return "get_tail_relations", m.group(1).strip(), None

    # Pattern 2: 'X "relation.dotted.path"' — quoted entity with relation
    m = re.match(r'^"?([^"]+?)"?\s+([a-z_.]+\.[a-z_.]+(?:\.[a-z_.]+)*)$', q)
    if m:
        return "get_tail_entities", m.group(1).strip(), m.group(2).strip()

    # Pattern 3: "entity relation" — best-effort split on last token
    tokens = q.split()
    if len(tokens) >= 2:
        ent = " ".join(tokens[:-1])
        rel = tokens[-1]
        if "." in rel:  # looks like a Freebase predicate
            return "get_tail_entities", ent, rel

    # Fallback: treat whole query as entity, return its outgoing relations
    return "get_tail_relations", q, None


def call_freebase(
    action: str,
    entity: str,
    relation: str | None,
    freebase_url: str,
    topk: int = 5,
) -> list[str]:
    """Call the local Freebase KG server (src_verl.kg_server.server).

    Our server expects {"action", "entity", "relation"} and returns
    {"action", "entity", "relation", "results": [...], "found": bool}.
    """
    try:
        r = requests.post(
            f"{freebase_url}/retrieve",
            json={"action": action, "entity": entity, "relation": relation},
            timeout=30,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        return [str(x) for x in results[:topk]]
    except Exception as e:  # noqa: BLE001 — retriever must never crash training
        logger.warning("Freebase call failed for query action=%s entity=%s: %s", action, entity, e)
        return []


def retrieve_one(query: str, topk: int, freebase_url: str) -> list[dict[str, Any]]:
    """Handle a single Search-R1 query. Returns a list of doc dicts.

    Each doc dict matches Search-R1's expected shape:
        {"document": {"contents": "Title\\n<body>"}, "score": float}
    """
    action, entity, relation = parse_query_to_call(query)
    results = call_freebase(action, entity, relation, freebase_url, topk=topk)

    docs: list[dict[str, Any]] = []
    if not results:
        # Return one "no results" document so the model still sees something.
        title = f"Freebase query: {query}"
        body = "No results found in Freebase for this query."
        docs.append({"document": {"contents": f"{title}\n{body}"}, "score": 0.0})
        return docs

    for i, res in enumerate(results):
        if action in ("get_tail_entities", "get_head_entities") and relation:
            if action == "get_tail_entities":
                title = f"Freebase triple: {entity} --{relation}--> {res}"
                body = f"{entity}\t{relation}\t{res}"
            else:
                title = f"Freebase triple: {res} --{relation}--> {entity}"
                body = f"{res}\t{relation}\t{entity}"
        elif action in ("get_tail_relations", "get_head_relations"):
            direction = "outgoing" if action == "get_tail_relations" else "incoming"
            title = f"Freebase relation of {entity}"
            body = f"{entity} has {direction} relation '{res}'."
        else:
            title = f"Freebase result for {entity}"
            body = f"{entity}\t?\t{res}"

        docs.append(
            {
                "document": {"contents": f"{title}\n{body}"},
                # Use rank-based pseudo-score (higher is better).
                "score": float(max(topk - i, 1)),
            }
        )
    return docs


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Search-R1 -> Freebase adapter")
    parser.add_argument("--freebase_url", type=str, default="http://localhost:18901")
    parser.add_argument(
        "--freebase_port",
        type=int,
        default=None,
        help="If set, overrides the port in --freebase_url (host stays localhost).",
    )
    parser.add_argument("--port", type=int, default=19001)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--default_topk", type=int, default=5)
    args = parser.parse_args()

    if args.freebase_port is not None:
        freebase_url = f"http://localhost:{args.freebase_port}"
    else:
        freebase_url = args.freebase_url

    # Lazy import so the module is importable without FastAPI installed
    from fastapi import FastAPI  # type: ignore
    from pydantic import BaseModel  # type: ignore
    import uvicorn  # type: ignore

    class QueryRequest(BaseModel):
        queries: list[str]
        topk: int | None = None
        return_scores: bool = False

    app = FastAPI()

    @app.post("/retrieve")
    def _retrieve(request: QueryRequest) -> dict[str, Any]:
        topk = request.topk if request.topk else args.default_topk
        resp: list[list[dict[str, Any]]] = []
        for q in request.queries:
            docs = retrieve_one(q, topk=topk, freebase_url=freebase_url)
            if request.return_scores:
                resp.append(docs)  # already include score key
            else:
                # Strip score to match Search-R1's non-scored format
                resp.append([{"document": d["document"]} for d in docs])
        return {"result": resp}

    @app.get("/health")
    def _health() -> dict[str, str]:
        return {"status": "ok"}

    logger.info(
        "Search-R1 adapter listening on %s:%d (freebase=%s)",
        args.host,
        args.port,
        freebase_url,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
