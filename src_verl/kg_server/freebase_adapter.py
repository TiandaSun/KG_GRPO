"""Freebase adapter for KG retrieval server.

Loads KG-R1's pre-processed Freebase data (entity dict, relation dict, triples)
and provides the same 4 query methods as ConceptNetAdapter.

Expected KG-R1 data layout in data_kg/:
  data_kg/
    entities.txt       — entity_id \t entity_name (one per line)
    relations.txt      — relation_id \t relation_name (one per line)
    triples.txt        — head_id \t relation_id \t tail_id (one per line)

If the data uses a different format (e.g., JSON), this adapter will
attempt to auto-detect and handle it.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


class FreebaseAdapter:
    """Wraps Freebase triples for KG server queries."""

    def __init__(
        self,
        entities: dict[str, str],
        relations: dict[str, str],
        outgoing: dict[str, dict[str, list[str]]],
        incoming: dict[str, dict[str, list[str]]],
        informative_errors: bool = False,
    ) -> None:
        self._entities = entities  # id -> name
        self._entity_name_to_id: dict[str, str] = {
            v.lower(): k for k, v in entities.items()
        }
        self._relations = relations  # id -> name
        self._relation_name_to_id: dict[str, str] = {
            v.lower(): k for k, v in relations.items()
        }
        self._outgoing = outgoing  # entity_id -> {relation_id -> [tail_ids]}
        self._incoming = incoming  # entity_id -> {relation_id -> [head_ids]}
        # v19 W20: when True, empty `[]` responses are replaced with informative
        # ERROR codes disambiguating entity-not-in-subgraph vs relation-not-present.
        self._informative_errors = informative_errors

        logger.info(
            "FreebaseAdapter initialized: %d entities, %d relations, %d outgoing entries "
            "(informative_errors=%s)",
            len(entities),
            len(relations),
            sum(len(rels) for rels in outgoing.values()),
            informative_errors,
        )

    @classmethod
    def from_data_dir(cls, data_dir: Path, informative_errors: bool = False) -> FreebaseAdapter:
        """Load Freebase data from KG-R1 format directory.

        informative_errors=True replaces empty `[]` responses with ERROR codes
        disambiguating entity_not_in_subgraph vs relation_not_present (v19 W20).
        """
        data_dir = Path(data_dir)

        # Load entities
        entities = cls._load_mapping(data_dir / "entities.txt")
        if not entities:
            entities = cls._load_mapping_json(data_dir / "entities.json")

        # Load relations
        relations = cls._load_mapping(data_dir / "relations.txt")
        if not relations:
            relations = cls._load_mapping_json(data_dir / "relations.json")

        # Load triples
        outgoing: dict[str, dict[str, list[str]]] = defaultdict(
            lambda: defaultdict(list)
        )
        incoming: dict[str, dict[str, list[str]]] = defaultdict(
            lambda: defaultdict(list)
        )

        triples_path = data_dir / "triples.txt"
        if not triples_path.exists():
            triples_path = data_dir / "train.txt"  # KG-R1 alternate name

        if triples_path.exists():
            count = 0
            with open(triples_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 3:
                        h, r, t = parts[0], parts[1], parts[2]
                        outgoing[h][r].append(t)
                        incoming[t][r].append(h)
                        count += 1
            logger.info("Loaded %d Freebase triples from %s", count, triples_path)
        else:
            logger.warning("No triples file found in %s", data_dir)

        return cls(entities, relations, outgoing, incoming, informative_errors=informative_errors)

    @staticmethod
    def _load_mapping(path: Path) -> dict[str, str]:
        """Load id->name mapping from TSV file."""
        if not path.exists():
            return {}
        mapping: dict[str, str] = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    mapping[parts[0]] = parts[1]
                elif len(parts) == 1:
                    mapping[parts[0]] = parts[0]
        logger.info("Loaded %d entries from %s", len(mapping), path)
        return mapping

    @staticmethod
    def _load_mapping_json(path: Path) -> dict[str, str]:
        """Load id->name mapping from JSON file."""
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
        return {}

    def _resolve_entity(self, entity: str) -> str | None:
        """Resolve entity name or ID to entity ID."""
        if entity in self._outgoing or entity in self._incoming:
            return entity
        lower = entity.lower()
        if lower in self._entity_name_to_id:
            return self._entity_name_to_id[lower]
        return None

    def _resolve_relation(self, relation: str) -> str | None:
        """Resolve relation name or ID to relation ID."""
        if relation in self._relations:
            return relation
        lower = relation.lower()
        if lower in self._relation_name_to_id:
            return self._relation_name_to_id[lower]
        return None

    def _entity_name(self, entity_id: str) -> str:
        """Get human-readable name for entity ID."""
        return self._entities.get(entity_id, entity_id)

    def _relation_name(self, relation_id: str) -> str:
        """Get human-readable name for relation ID."""
        return self._relations.get(relation_id, relation_id)

    # ------------------------------------------------------------------
    # v19 W20 — informative-failure helpers (replace empty [] with disambiguating ERROR codes)
    # ------------------------------------------------------------------
    def _err_entity(self, entity: str) -> list[str]:
        if self._informative_errors:
            return [f"ERROR: entity_not_in_subgraph({entity})"]
        return []

    def _err_relation(self, entity: str, relation: str) -> list[str]:
        if self._informative_errors:
            return [f"ERROR: relation_not_present({entity}, {relation})"]
        return []

    def get_tail_relations(self, entity: str) -> list[str]:
        """Get all relation types going out from entity."""
        eid = self._resolve_entity(entity)
        if eid is None:
            return self._err_entity(entity)
        if eid not in self._outgoing:
            return self._err_entity(entity)
        return sorted(set(self._relation_name(r) for r in self._outgoing[eid].keys()))

    def get_head_relations(self, entity: str) -> list[str]:
        """Get all relation types coming into entity."""
        eid = self._resolve_entity(entity)
        if eid is None:
            return self._err_entity(entity)
        if eid not in self._incoming:
            return self._err_entity(entity)
        return sorted(set(self._relation_name(r) for r in self._incoming[eid].keys()))

    def get_tail_entities(self, entity: str, relation: str) -> list[str]:
        """Get entities reachable from entity via relation."""
        eid = self._resolve_entity(entity)
        if eid is None:
            return self._err_entity(entity)
        rid = self._resolve_relation(relation)
        if rid is None:
            return self._err_relation(entity, relation)
        if eid in self._outgoing and rid in self._outgoing[eid]:
            return sorted(set(self._entity_name(t) for t in self._outgoing[eid][rid]))
        return self._err_relation(entity, relation)

    def get_head_entities(self, entity: str, relation: str) -> list[str]:
        """Get entities that connect to entity via relation."""
        eid = self._resolve_entity(entity)
        if eid is None:
            return self._err_entity(entity)
        rid = self._resolve_relation(relation)
        if rid is None:
            return self._err_relation(entity, relation)
        if eid in self._incoming and rid in self._incoming[eid]:
            return sorted(set(self._entity_name(h) for h in self._incoming[eid][rid]))
        return self._err_relation(entity, relation)

    def has_entity(self, entity: str) -> bool:
        """Check if entity exists in the graph."""
        return self._resolve_entity(entity) is not None

    def get_all_entities(self) -> list[str]:
        """Get all entity names."""
        return list(self._entities.values())

    @property
    def num_entities(self) -> int:
        return len(self._entities)

    @property
    def num_relations(self) -> int:
        return len(self._relations)
