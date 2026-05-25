"""V15-Q8 synthetic-schema KG adapter wrapper.

Translates synthetic URIs (entity://safe_label__short_hash) to original Freebase
entity strings, delegates to the existing FreebaseAdapter for KG queries, then
translates results back to synthetic URIs.

Mapping is loaded from data/freebase/verl_cwq_q8_synthetic/entity_mapping.json
(produced by scripts/task_q8_synthetic_schema.py).

Relation strings are passed through unchanged — relations are NOT rewritten.

Used by src_verl.kg_server.server when --kg q8_synthetic is selected.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src_verl.kg_server.freebase_adapter import FreebaseAdapter

logger = logging.getLogger(__name__)


class Q8SyntheticAdapter:
    """Wrap a FreebaseAdapter with URI <-> original-string translation.

    All five public methods are duck-typed compatible with FreebaseAdapter so the
    KG server (server.py) can use this adapter without code changes.
    """

    def __init__(
        self,
        freebase_adapter: FreebaseAdapter,
        uri_to_original: dict[str, str],
    ) -> None:
        self._fb = freebase_adapter
        self._uri_to_original: dict[str, str] = dict(uri_to_original)
        self._original_to_uri: dict[str, str] = {v: k for k, v in self._uri_to_original.items()}
        logger.info(
            "Q8SyntheticAdapter loaded: %d URI mappings, wrapping FreebaseAdapter "
            "(%d entities, %d relations)",
            len(self._uri_to_original),
            self._fb.num_entities,
            self._fb.num_relations,
        )

    @classmethod
    def from_files(
        cls,
        freebase_dir: Path,
        mapping_path: Path,
    ) -> Q8SyntheticAdapter:
        """Load both the underlying Freebase KG and the URI mapping."""
        fb = FreebaseAdapter.from_data_dir(freebase_dir)
        with open(mapping_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        uri_to_original = data.get("uri_to_original", {})
        return cls(fb, uri_to_original)

    # --- internal translators ---
    def _uri_to_orig(self, s: str) -> str:
        """If s is a known synthetic URI, return its original string. Otherwise return s."""
        if not isinstance(s, str):
            return s
        if s.startswith("entity://"):
            return self._uri_to_original.get(s, s)
        return s

    def _orig_to_uri(self, s: str) -> str:
        """If s is a known original entity, return its URI. Otherwise return s unchanged
        (so unknown new entities surface as-is — the policy still sees something)."""
        if not isinstance(s, str):
            return s
        return self._original_to_uri.get(s, s)

    # --- 5-endpoint API (matches FreebaseAdapter signatures) ---
    def get_tail_relations(self, entity: str) -> list[str]:
        orig = self._uri_to_orig(entity)
        return self._fb.get_tail_relations(orig)

    def get_head_relations(self, entity: str) -> list[str]:
        orig = self._uri_to_orig(entity)
        return self._fb.get_head_relations(orig)

    def get_tail_entities(self, entity: str, relation: str) -> list[str]:
        orig = self._uri_to_orig(entity)
        results = self._fb.get_tail_entities(orig, relation)
        # Translate each returned entity back to a URI when known
        return [self._orig_to_uri(r) for r in results]

    def get_head_entities(self, entity: str, relation: str) -> list[str]:
        orig = self._uri_to_orig(entity)
        results = self._fb.get_head_entities(orig, relation)
        return [self._orig_to_uri(r) for r in results]

    def has_entity(self, entity: str) -> bool:
        orig = self._uri_to_orig(entity)
        return self._fb.has_entity(orig)

    def get_all_entities(self) -> list[str]:
        return [self._orig_to_uri(e) for e in self._fb.get_all_entities()]

    @property
    def num_entities(self) -> int:
        return self._fb.num_entities

    @property
    def num_relations(self) -> int:
        return self._fb.num_relations
