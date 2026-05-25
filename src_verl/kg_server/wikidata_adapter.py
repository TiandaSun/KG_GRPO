"""Wikidata adapter for KG retrieval server.

Reads pre-materialized Wikidata neighborhood cache produced by
`scripts/task35_prematerialize_wikidata.py`.

Cache format (per QID):
    data/wikidata_cache/{qid}.json
    {
        "qid": "Q12345",
        "label": "Some Entity",
        "outgoing": {"P31": [["Q5", "human"], ...], ...},
        "incoming": {"P57": [["Q8888", "Some Film"], ...], ...}
    }

Implements the same interface as FreebaseAdapter:
  - get_tail_relations(entity)
  - get_head_relations(entity)
  - get_tail_entities(entity, relation)
  - get_head_entities(entity, relation)
  - has_entity(entity)
  - get_all_entities()
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


class WikidataAdapter:
    """Wraps Wikidata 1-hop cache for KG server queries."""

    def __init__(
        self,
        entities: dict[str, str],  # qid -> label
        relations: dict[str, str],  # pid -> label (best-effort, derived from neighbors)
        outgoing: dict[str, dict[str, list[str]]],  # qid -> {pid -> [tail_qids]}
        incoming: dict[str, dict[str, list[str]]],  # qid -> {pid -> [head_qids]}
    ) -> None:
        self._entities = entities
        self._entity_name_to_id: dict[str, str] = {}
        for qid, label in entities.items():
            if label:
                self._entity_name_to_id[label.lower()] = qid

        self._relations = relations
        self._relation_name_to_id: dict[str, str] = {}
        for pid, label in relations.items():
            if label:
                self._relation_name_to_id[label.lower()] = pid

        self._outgoing = outgoing
        self._incoming = incoming

        logger.info(
            "WikidataAdapter initialized: %d entities, %d relations, %d outgoing entries",
            len(entities),
            len(relations),
            sum(len(rels) for rels in outgoing.values()),
        )

    @classmethod
    def from_cache_dir(cls, cache_dir: Path) -> WikidataAdapter:
        """Load Wikidata data from pre-materialized cache directory."""
        cache_dir = Path(cache_dir)

        entities: dict[str, str] = {}
        relations: dict[str, str] = {}
        outgoing: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        incoming: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))

        # Cache for (pid -> labels) collected from neighbor metadata
        # We don't have property labels in the cache directly, so derive them
        # from the proof triples in KGQAGen-10k or fall back to PIDs.
        # For now, we'll use just the PIDs and let the KG server return them.

        files = sorted(cache_dir.glob("Q*.json"))
        logger.info("Loading %d Wikidata cache files from %s", len(files), cache_dir)

        for f in files:
            try:
                with open(f) as fh:
                    data = json.load(fh)
            except Exception as e:
                logger.warning("Failed to read %s: %s", f.name, e)
                continue

            qid = data.get("qid", f.stem)
            label = data.get("label", "")
            entities[qid] = label

            # Outgoing edges
            for pid, neighbors in data.get("outgoing", {}).items():
                for neighbor in neighbors:
                    if isinstance(neighbor, list) and len(neighbor) >= 1:
                        target_qid = neighbor[0]
                        target_label = neighbor[1] if len(neighbor) >= 2 else ""
                        outgoing[qid][pid].append(target_qid)
                        # Add target label to entities (don't overwrite known labels)
                        if target_qid and target_qid not in entities:
                            entities[target_qid] = target_label
                # Track property ID; label not available in cache
                if pid not in relations:
                    relations[pid] = pid

            # Incoming edges
            for pid, neighbors in data.get("incoming", {}).items():
                for neighbor in neighbors:
                    if isinstance(neighbor, list) and len(neighbor) >= 1:
                        source_qid = neighbor[0]
                        source_label = neighbor[1] if len(neighbor) >= 2 else ""
                        incoming[qid][pid].append(source_qid)
                        if source_qid and source_qid not in entities:
                            entities[source_qid] = source_label
                if pid not in relations:
                    relations[pid] = pid

        # Convert defaultdicts to plain dicts for consistency
        outgoing = {k: dict(v) for k, v in outgoing.items()}
        incoming = {k: dict(v) for k, v in incoming.items()}

        logger.info(
            "Loaded Wikidata cache: %d entities (incl. neighbors), %d relations",
            len(entities), len(relations),
        )

        return cls(entities, relations, outgoing, incoming)

    def _resolve_entity(self, entity: str) -> str | None:
        """Resolve entity QID or label to QID."""
        if entity in self._outgoing or entity in self._incoming or entity in self._entities:
            return entity
        lower = entity.lower()
        if lower in self._entity_name_to_id:
            return self._entity_name_to_id[lower]
        return None

    def _resolve_relation(self, relation: str) -> str | None:
        """Resolve relation PID or label to PID."""
        if relation in self._relations:
            return relation
        lower = relation.lower()
        if lower in self._relation_name_to_id:
            return self._relation_name_to_id[lower]
        return None

    def _entity_name(self, qid: str) -> str:
        """Get human-readable label for QID, fall back to QID."""
        label = self._entities.get(qid, "")
        return label if label else qid

    def _relation_name(self, pid: str) -> str:
        """Get label for property PID, fall back to PID."""
        label = self._relations.get(pid, "")
        return label if label else pid

    def get_tail_relations(self, entity: str) -> list[str]:
        """Get all property IDs going out from entity."""
        eid = self._resolve_entity(entity)
        if eid is None or eid not in self._outgoing:
            return []
        return sorted(set(self._relation_name(p) for p in self._outgoing[eid].keys()))

    def get_head_relations(self, entity: str) -> list[str]:
        """Get all property IDs coming into entity."""
        eid = self._resolve_entity(entity)
        if eid is None or eid not in self._incoming:
            return []
        return sorted(set(self._relation_name(p) for p in self._incoming[eid].keys()))

    def get_tail_entities(self, entity: str, relation: str) -> list[str]:
        """Get entities reachable from entity via relation."""
        eid = self._resolve_entity(entity)
        pid = self._resolve_relation(relation)
        if eid is None or pid is None:
            return []
        if eid in self._outgoing and pid in self._outgoing[eid]:
            return sorted(set(self._entity_name(t) for t in self._outgoing[eid][pid]))
        return []

    def get_head_entities(self, entity: str, relation: str) -> list[str]:
        """Get entities that connect to entity via relation."""
        eid = self._resolve_entity(entity)
        pid = self._resolve_relation(relation)
        if eid is None or pid is None:
            return []
        if eid in self._incoming and pid in self._incoming[eid]:
            return sorted(set(self._entity_name(h) for h in self._incoming[eid][pid]))
        return []

    def has_entity(self, entity: str) -> bool:
        """Check if entity exists in the cache."""
        return self._resolve_entity(entity) is not None

    def get_all_entities(self) -> list[str]:
        """Get all entity labels (or QIDs if label missing)."""
        return [self._entity_name(qid) for qid in self._entities.keys()]

    @property
    def num_entities(self) -> int:
        return len(self._entities)

    @property
    def num_relations(self) -> int:
        return len(self._relations)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=Path, default=Path("data/wikidata_cache"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    adapter = WikidataAdapter.from_cache_dir(args.cache_dir)
    print(f"Loaded {adapter.num_entities} entities, {adapter.num_relations} relations")

    # Smoke test: pick first entity and query its relations
    sample_qid = next(iter(adapter._outgoing.keys()), None)
    if sample_qid:
        print(f"\nSample QID: {sample_qid} (label: {adapter._entity_name(sample_qid)})")
        print(f"  Tail relations: {adapter.get_tail_relations(sample_qid)[:5]}")
        print(f"  Head relations: {adapter.get_head_relations(sample_qid)[:5]}")
