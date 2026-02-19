"""
GLiNER2 relation extraction  →  Cypher query generation.

Usage
-----
Low-level converter (no ML model needed)::

    from cypher_validator import RelationToCypherConverter

    results = {
        "relation_extraction": {
            "works_for": [("John", "Apple Inc.")],
            "lives_in":  [("John", "San Francisco")],
        }
    }
    converter = RelationToCypherConverter()
    print(converter.convert(results, mode="merge"))

End-to-end pipeline::

    from cypher_validator import NLToCypher

    pipeline = NLToCypher.from_pretrained("fastino/gliner2-large-v1", schema=schema)
    cypher = pipeline(
        "John works for Apple Inc. and lives in San Francisco.",
        ["works_for", "lives_in"],
        mode="merge",
    )
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_cypher_rel_type(rel_type: str) -> str:
    """Convert any-case relation type to UPPER_SNAKE_CASE (Cypher convention)."""
    return rel_type.upper()


# ---------------------------------------------------------------------------
# RelationToCypherConverter
# ---------------------------------------------------------------------------

class RelationToCypherConverter:
    """Convert GLiNER2 relation-extraction output to a Cypher query.

    Works with any dict matching the GLiNER2 output schema::

        {
            "relation_extraction": {
                "works_for": [("John", "Apple Inc.")],
                "lives_in":  [("John", "San Francisco")],
                "founded":   [],   # requested but not found
            }
        }

    Three Cypher generation modes are supported via :meth:`convert`:

    * ``"match"``  — ``MATCH … WHERE …`` (read, find existing nodes)
    * ``"merge"``  — ``MERGE …`` (upsert nodes and relationships)
    * ``"create"`` — ``CREATE …`` (insert new nodes and relationships)

    Parameters
    ----------
    schema:
        Optional ``cypher_validator.Schema`` instance.  When provided,
        source/target node labels are automatically added to ``MERGE``/``CREATE``
        clauses using the schema's relationship endpoint information.
    name_property:
        Node property key used to store entity text spans (default: ``"name"``).
    """

    def __init__(
        self,
        schema: Any = None,
        name_property: str = "name",
    ) -> None:
        self.schema = schema
        self.name_property = name_property

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_endpoints(self, cypher_rel: str) -> Tuple[str, str]:
        """Return ``(src_label, tgt_label)`` from schema, or ``("", "")``."""
        if self.schema is not None:
            endpoints = self.schema.rel_endpoints(cypher_rel)
            if endpoints is not None:
                return endpoints[0], endpoints[1]
        return "", ""

    def _clean_pairs(
        self, pairs: List[Any]
    ) -> List[Tuple[str, str]]:
        """Drop falsy entries and coerce to (str, str) tuples."""
        return [(str(s), str(o)) for s, o in pairs if s and o]

    # ------------------------------------------------------------------
    # MATCH
    # ------------------------------------------------------------------

    def to_match_query(
        self,
        relations: Dict[str, Any],
        return_clause: Optional[str] = None,
    ) -> str:
        """Build ``MATCH`` clauses for the extracted relations.

        Each (subject, object) pair produces its own ``MATCH`` clause using
        inline property maps—one clause per pair, regardless of how many pairs
        share the same relation type.

        Parameters
        ----------
        relations:
            GLiNER2 output dict.
        return_clause:
            Custom ``RETURN …`` tail.  Auto-generated from pattern variables
            when *None*.

        Returns
        -------
        str
            Cypher query, or ``""`` when nothing was extracted.
        """
        rel_data: Dict[str, List[Any]] = relations.get("relation_extraction", {})
        clauses: List[str] = []
        return_vars: List[str] = []
        np = self.name_property
        idx = 0
        seen: set = set()  # (subject, obj, cypher_rel) deduplication

        for rel_type, raw_pairs in rel_data.items():
            pairs = self._clean_pairs(raw_pairs)
            if not pairs:
                continue

            cypher_rel = _to_cypher_rel_type(rel_type)

            for subject, obj in pairs:
                key = (subject, obj, cypher_rel)
                if key in seen:
                    continue
                seen.add(key)
                a_var, b_var = f"a{idx}", f"b{idx}"
                clauses.append(
                    f'MATCH ({a_var} {{{np}: "{subject}"}})'
                    f"-[:{cypher_rel}]->"
                    f'({b_var} {{{np}: "{obj}"}})'
                )
                return_vars.extend([a_var, b_var])
                idx += 1

        if not clauses:
            return ""

        ret = (
            return_clause
            if return_clause is not None
            else f"RETURN {', '.join(return_vars)}"
        )
        return "\n".join(clauses) + f"\n{ret}"

    # ------------------------------------------------------------------
    # MERGE
    # ------------------------------------------------------------------

    def to_merge_query(
        self,
        relations: Dict[str, Any],
        return_clause: Optional[str] = None,
    ) -> str:
        """Build ``MERGE`` clauses to upsert extracted entities/relationships.

        When a ``schema`` is available and the relation type is known, node
        labels are added automatically (e.g. ``:Person``, ``:Company``).

        Parameters
        ----------
        relations:
            GLiNER2 output dict.
        return_clause:
            Custom ``RETURN …`` tail.

        Returns
        -------
        str
            Cypher query, or ``""`` when nothing was extracted.
        """
        return self._build_clause("MERGE", relations, return_clause)

    # ------------------------------------------------------------------
    # CREATE
    # ------------------------------------------------------------------

    def to_create_query(
        self,
        relations: Dict[str, Any],
        return_clause: Optional[str] = None,
    ) -> str:
        """Build ``CREATE`` clauses to insert extracted entities/relationships.

        Parameters
        ----------
        relations:
            GLiNER2 output dict.
        return_clause:
            Custom ``RETURN …`` tail.

        Returns
        -------
        str
            Cypher query, or ``""`` when nothing was extracted.
        """
        return self._build_clause("CREATE", relations, return_clause)

    # ------------------------------------------------------------------
    # Shared builder for MERGE / CREATE
    # ------------------------------------------------------------------

    def _build_clause(
        self,
        keyword: str,
        relations: Dict[str, Any],
        return_clause: Optional[str],
    ) -> str:
        rel_data: Dict[str, List[Any]] = relations.get("relation_extraction", {})
        clauses: List[str] = []
        return_vars: List[str] = []
        np = self.name_property
        idx = 0
        seen: set = set()  # (subject, obj, cypher_rel) deduplication

        for rel_type, raw_pairs in rel_data.items():
            pairs = self._clean_pairs(raw_pairs)
            if not pairs:
                continue

            cypher_rel = _to_cypher_rel_type(rel_type)
            src_label, tgt_label = self._get_endpoints(cypher_rel)
            src_l = f":{src_label}" if src_label else ""
            tgt_l = f":{tgt_label}" if tgt_label else ""

            for subject, obj in pairs:
                key = (subject, obj, cypher_rel)
                if key in seen:
                    continue
                seen.add(key)
                a_var, b_var = f"a{idx}", f"b{idx}"
                clauses.append(
                    f'{keyword} ({a_var}{src_l} {{{np}: "{subject}"}})'
                    f"-[:{cypher_rel}]->"
                    f'({b_var}{tgt_l} {{{np}: "{obj}"}})'
                )
                return_vars.extend([a_var, b_var])
                idx += 1

        if not clauses:
            return ""

        ret = (
            return_clause
            if return_clause is not None
            else f"RETURN {', '.join(return_vars)}"
        )
        return "\n".join(clauses) + f"\n{ret}"

    # ------------------------------------------------------------------
    # Unified entry point
    # ------------------------------------------------------------------

    def convert(
        self,
        relations: Dict[str, Any],
        mode: str = "match",
        **kwargs: Any,
    ) -> str:
        """Convert extracted relations to a Cypher query.

        Parameters
        ----------
        relations:
            Output from :meth:`GLiNER2RelationExtractor.extract_relations`.
        mode:
            ``"match"``, ``"merge"``, or ``"create"``.
        **kwargs:
            Forwarded to the chosen generator method
            (e.g. ``return_clause="RETURN *"``).

        Returns
        -------
        str
            Cypher query string, or ``""`` if nothing was extracted.

        Raises
        ------
        ValueError
            For an unrecognised *mode*.
        """
        if mode == "match":
            return self.to_match_query(relations, **kwargs)
        elif mode == "merge":
            return self.to_merge_query(relations, **kwargs)
        elif mode == "create":
            return self.to_create_query(relations, **kwargs)
        else:
            raise ValueError(
                f"Unknown mode '{mode}'.  Use 'match', 'merge', or 'create'."
            )

    def __repr__(self) -> str:
        return (
            f"RelationToCypherConverter("
            f"schema={self.schema!r}, name_property={self.name_property!r})"
        )


# ---------------------------------------------------------------------------
# GLiNER2RelationExtractor
# ---------------------------------------------------------------------------

class GLiNER2RelationExtractor:
    """Thin wrapper around the **GLiNER2** model for zero-shot relation extraction.

    Requires the ``gliner2`` Python package::

        pip install gliner2
        # or: pip install "cypher_validator[gliner2]"

    Example
    -------
    ::

        extractor = GLiNER2RelationExtractor.from_pretrained(
            "fastino/gliner2-large-v1"
        )
        results = extractor.extract_relations(
            "John works for Apple Inc. and lives in San Francisco.",
            ["works_for", "lives_in"],
        )
        # {
        #   "relation_extraction": {
        #     "works_for": [("John", "Apple Inc.")],
        #     "lives_in":  [("John", "San Francisco")],
        #   }
        # }
    """

    DEFAULT_MODEL: str = "fastino/gliner2-large-v1"

    def __init__(self, model: Any, threshold: float = 0.5) -> None:
        """
        Parameters
        ----------
        model:
            A loaded ``gliner2.GLiNER2`` instance.
        threshold:
            Default confidence threshold for accepting a predicted relation (0–1).
        """
        self._model = model
        self.threshold = threshold

    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = DEFAULT_MODEL,
        threshold: float = 0.5,
    ) -> "GLiNER2RelationExtractor":
        """Load a pretrained GLiNER2 model from the HuggingFace Hub or a local path.

        Parameters
        ----------
        model_name:
            HuggingFace model identifier or local directory.
            Defaults to ``"fastino/gliner2-large-v1"``.
        threshold:
            Default confidence threshold (0–1).

        Returns
        -------
        GLiNER2RelationExtractor

        Raises
        ------
        ImportError
            If the ``gliner2`` package is not installed.
        """
        try:
            from gliner2 import GLiNER2  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "The 'gliner2' package is required for relation extraction.\n"
                "Install it with:  pip install gliner2\n"
                "Or:               pip install 'cypher_validator[gliner2]'"
            ) from exc

        model = GLiNER2.from_pretrained(model_name)
        return cls(model, threshold=threshold)

    # ------------------------------------------------------------------

    def extract_relations(
        self,
        text: str,
        relation_types: List[str],
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Extract relations from *text* using the loaded GLiNER2 model.

        Parameters
        ----------
        text:
            Source sentence or passage.
        relation_types:
            Relation labels to look for (e.g. ``["works_for", "lives_in"]``).
            **All** requested types appear in the output—with an empty list when
            the relation was not found.
        threshold:
            Confidence threshold for this call; overrides the instance default.

        Returns
        -------
        dict
            ::

                {
                    "relation_extraction": {
                        "works_for": [("John", "Apple Inc.")],
                        "lives_in":  [],   # not found but was requested
                    }
                }
        """
        t = threshold if threshold is not None else self.threshold
        raw: Dict[str, Any] = self._model.extract_relations(
            text, relation_types, threshold=t
        )

        # Normalise: support both raw dict and wrapped dict
        rel_data: Dict[str, List[Tuple[str, str]]] = raw.get(
            "relation_extraction", raw
        )

        # Guarantee every requested type is present (empty list if absent)
        result: Dict[str, List[Tuple[str, str]]] = {
            rt: list(rel_data.get(rt, [])) for rt in relation_types
        }
        return {"relation_extraction": result}

    def __repr__(self) -> str:
        return f"GLiNER2RelationExtractor(threshold={self.threshold})"


# ---------------------------------------------------------------------------
# NLToCypher — end-to-end pipeline
# ---------------------------------------------------------------------------

class NLToCypher:
    """End-to-end pipeline: **natural language text → Cypher query**.

    Chains :class:`GLiNER2RelationExtractor` and :class:`RelationToCypherConverter`
    into a single callable object.

    Example
    -------
    ::

        from cypher_validator import Schema, NLToCypher

        schema = Schema(
            nodes={"Person": ["name"], "Company": ["name"]},
            relationships={"WORKS_FOR": ("Person", "Company", [])},
        )
        pipeline = NLToCypher.from_pretrained(
            "fastino/gliner2-large-v1",
            schema=schema,
        )

        # Single call
        cypher = pipeline(
            "John works for Apple Inc.",
            ["works_for"],
            mode="merge",
        )
        # MERGE (a0:Person {name: "John"})-[:WORKS_FOR]->(b0:Company {name: "Apple Inc."})
        # RETURN a0, b0

        # With both extraction result and Cypher
        relations, cypher = pipeline.extract_and_convert(
            "John works for Apple Inc.",
            ["works_for"],
        )
    """

    def __init__(
        self,
        extractor: GLiNER2RelationExtractor,
        schema: Any = None,
        name_property: str = "name",
    ) -> None:
        """
        Parameters
        ----------
        extractor:
            A :class:`GLiNER2RelationExtractor` instance.
        schema:
            Optional ``cypher_validator.Schema`` for label-aware Cypher generation.
        name_property:
            Node property key for entity names (default: ``"name"``).
        """
        self.extractor = extractor
        self.converter = RelationToCypherConverter(
            schema=schema, name_property=name_property
        )

    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = GLiNER2RelationExtractor.DEFAULT_MODEL,
        schema: Any = None,
        threshold: float = 0.5,
        name_property: str = "name",
    ) -> "NLToCypher":
        """Create a pipeline by loading a pretrained GLiNER2 model.

        Parameters
        ----------
        model_name:
            HuggingFace model name or local path.
            Defaults to ``"fastino/gliner2-large-v1"``.
        schema:
            Optional ``cypher_validator.Schema`` for label-aware queries.
        threshold:
            Confidence threshold for the relation extractor.
        name_property:
            Node property key for entity names.

        Returns
        -------
        NLToCypher
        """
        extractor = GLiNER2RelationExtractor.from_pretrained(
            model_name, threshold=threshold
        )
        return cls(extractor, schema=schema, name_property=name_property)

    # ------------------------------------------------------------------

    def __call__(
        self,
        text: str,
        relation_types: List[str],
        mode: str = "match",
        threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> str:
        """Extract relations from *text* and return a Cypher query.

        Parameters
        ----------
        text:
            Input sentence or passage.
        relation_types:
            Relation labels to extract.
        mode:
            Cypher generation mode: ``"match"``, ``"merge"``, or ``"create"``.
        threshold:
            Override the extractor's confidence threshold for this call.
        **kwargs:
            Extra keyword arguments forwarded to the Cypher converter
            (e.g. ``return_clause="RETURN *"``).

        Returns
        -------
        str
            Cypher query string, or ``""`` when no relations were found.
        """
        relations = self.extractor.extract_relations(
            text, relation_types, threshold=threshold
        )
        return self.converter.convert(relations, mode=mode, **kwargs)

    def extract_and_convert(
        self,
        text: str,
        relation_types: List[str],
        mode: str = "match",
        threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], str]:
        """Extract relations and return both the raw dict and the Cypher string.

        Returns
        -------
        tuple
            ``(relations_dict, cypher_query)``
        """
        relations = self.extractor.extract_relations(
            text, relation_types, threshold=threshold
        )
        cypher = self.converter.convert(relations, mode=mode, **kwargs)
        return relations, cypher

    def __repr__(self) -> str:
        return (
            f"NLToCypher(extractor={self.extractor!r}, "
            f"converter={self.converter!r})"
        )
