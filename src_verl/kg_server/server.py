"""FastAPI KG retrieval server for multi-turn agent interaction.

Provides a /retrieve endpoint that handles batch KG queries. The agent
sends tool calls like `get_tail_relations(entity)` and receives results.

Follows KG-R1's server pattern: POST /retrieve with JSON body.

Usage:
    # Start with ConceptNet:
    python src_verl/kg_server/server.py \
        --kg conceptnet \
        --assertions_path data/raw/conceptnet-assertions-5.7.0.csv.gz \
        --port 8001

    # Start with Freebase:
    python src_verl/kg_server/server.py \
        --kg freebase \
        --freebase_dir data_kg \
        --port 8001
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI(title="KG Retrieval Server", version="1.0")

# Global adapter reference (set during startup)
_adapter: Any = None


class RetrieveRequest(BaseModel):
    """Single KG query request."""

    action: str  # get_tail_relations, get_head_relations, get_tail_entities, get_head_entities
    entity: str
    relation: str | None = None


class BatchRetrieveRequest(BaseModel):
    """Batch of KG queries."""

    queries: list[RetrieveRequest]


class RetrieveResponse(BaseModel):
    """Response for a single query."""

    action: str
    entity: str
    relation: str | None = None
    results: list[str]
    found: bool


class BatchRetrieveResponse(BaseModel):
    """Response for a batch of queries."""

    responses: list[RetrieveResponse]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    kg_type: str
    num_entities: int


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health check endpoint."""
    if _adapter is None:
        raise HTTPException(status_code=503, detail="KG adapter not loaded")

    num_entities = 0
    kg_type = type(_adapter).__name__
    if hasattr(_adapter, "num_nodes"):
        num_entities = _adapter.num_nodes
    elif hasattr(_adapter, "num_entities"):
        num_entities = _adapter.num_entities

    return HealthResponse(
        status="ok",
        kg_type=kg_type,
        num_entities=num_entities,
    )


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(request: RetrieveRequest) -> RetrieveResponse:
    """Handle a single KG query."""
    if _adapter is None:
        raise HTTPException(status_code=503, detail="KG adapter not loaded")

    results = _execute_query(request)
    return RetrieveResponse(
        action=request.action,
        entity=request.entity,
        relation=request.relation,
        results=results,
        found=len(results) > 0,
    )


@app.post("/batch_retrieve", response_model=BatchRetrieveResponse)
def batch_retrieve(request: BatchRetrieveRequest) -> BatchRetrieveResponse:
    """Handle a batch of KG queries."""
    if _adapter is None:
        raise HTTPException(status_code=503, detail="KG adapter not loaded")

    responses: list[RetrieveResponse] = []
    for query in request.queries:
        results = _execute_query(query)
        responses.append(
            RetrieveResponse(
                action=query.action,
                entity=query.entity,
                relation=query.relation,
                results=results,
                found=len(results) > 0,
            )
        )

    return BatchRetrieveResponse(responses=responses)


def _execute_query(request: RetrieveRequest) -> list[str]:
    """Execute a single KG query against the adapter."""
    action = request.action.lower().strip()
    entity = request.entity.strip()
    relation = request.relation.strip() if request.relation else None

    if action == "get_tail_relations":
        return _adapter.get_tail_relations(entity)
    elif action == "get_head_relations":
        return _adapter.get_head_relations(entity)
    elif action == "get_tail_entities":
        if relation is None:
            return []
        return _adapter.get_tail_entities(entity, relation)
    elif action == "get_head_entities":
        if relation is None:
            return []
        return _adapter.get_head_entities(entity, relation)
    else:
        logger.warning("Unknown action: %s", action)
        return []


def load_adapter(
    kg_type: str,
    assertions_path: Path | None = None,
    freebase_dir: Path | None = None,
    min_weight: float = 2.0,
) -> Any:
    """Load the appropriate KG adapter."""
    if kg_type == "conceptnet":
        from src_verl.kg_server.conceptnet_adapter import ConceptNetAdapter

        if assertions_path is None:
            assertions_path = Path("data/raw/conceptnet-assertions-5.7.0.csv.gz")
        logger.info("Loading ConceptNet from %s ...", assertions_path)
        return ConceptNetAdapter.from_assertions(assertions_path, min_weight)
    elif kg_type == "freebase":
        from src_verl.kg_server.freebase_adapter import FreebaseAdapter

        if freebase_dir is None:
            freebase_dir = Path("data_kg")
        logger.info("Loading Freebase from %s ...", freebase_dir)
        return FreebaseAdapter.from_data_dir(freebase_dir)
    else:
        raise ValueError(f"Unknown KG type: {kg_type}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="KG Retrieval Server")
    parser.add_argument(
        "--kg",
        type=str,
        choices=["conceptnet", "freebase"],
        default="conceptnet",
        help="Which KG to serve.",
    )
    parser.add_argument(
        "--assertions_path",
        type=Path,
        default=Path("data/raw/conceptnet-assertions-5.7.0.csv.gz"),
        help="ConceptNet assertions path.",
    )
    parser.add_argument(
        "--freebase_dir",
        type=Path,
        default=Path("data_kg"),
        help="Freebase data directory.",
    )
    parser.add_argument(
        "--min_weight",
        type=float,
        default=2.0,
        help="Minimum weight for ConceptNet filtering.",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host.")
    parser.add_argument("--port", type=int, default=8001, help="Server port.")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers.")
    parser.add_argument("--log-level", type=str, default="info",
                        choices=["debug", "info", "warning", "error"],
                        help="Uvicorn log level (default: info). Use 'warning' to suppress access logs.")
    args = parser.parse_args()

    global _adapter
    start = time.time()
    _adapter = load_adapter(
        args.kg,
        assertions_path=args.assertions_path,
        freebase_dir=args.freebase_dir,
        min_weight=args.min_weight,
    )
    logger.info("KG loaded in %.1f seconds", time.time() - start)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level=args.log_level,
        access_log=args.log_level != "warning",
    )


if __name__ == "__main__":
    main()
