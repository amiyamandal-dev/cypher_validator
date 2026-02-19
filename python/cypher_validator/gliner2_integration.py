"""
GLiNER2 relation extraction  →  Cypher query generation  →  optional Neo4j execution.

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

End-to-end pipeline (Cypher generation only)::

    from cypher_validator import NLToCypher

    pipeline = NLToCypher.from_pretrained("fastino/gliner2-large-v1", schema=schema)
    cypher = pipeline(
        "John works for Apple Inc. and lives in San Francisco.",
        ["works_for", "lives_in"],
        mode="create",
    )

End-to-end pipeline with Neo4j execution::

    from cypher_validator import NLToCypher, Neo4jDatabase

    db = Neo4jDatabase("bolt://localhost:7687", "neo4j", "password")
    pipeline = NLToCypher.from_pretrained(
        "fastino/gliner2-large-v1",
        schema=schema,
        db=db,
    )
    cypher, results = pipeline(
        "John works for Apple Inc.",
        ["works_for"],
        mode="create",
        execute=True,
    )
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_cypher_rel_type(rel_type: str) -> str:
    """Convert any-case relation type to UPPER_SNAKE_CASE (Cypher convention)."""
    return rel_type.upper()


# ---------------------------------------------------------------------------
# Neo4jDatabase
# ---------------------------------------------------------------------------

class Neo4jDatabase:
    """Thin wrapper around the Neo4j Python driver for executing Cypher queries.

    Requires the ``neo4j`` package::

        pip install neo4j

    Can be used as a context manager::

        with Neo4jDatabase("bolt://localhost:7687", "neo4j", "password") as db:
            results = db.execute("MATCH (n:Person) RETURN n LIMIT 5")

    Or directly passed to :class:`NLToCypher` to enable database-aware execution::

        db = Neo4jDatabase("bolt://localhost:7687", "neo4j", "password")
        pipeline = NLToCypher.from_pretrained("fastino/gliner2-large-v1", db=db)
        cypher, results = pipeline(
            "John works for Apple Inc.",
            ["works_for"],
            mode="create",
            execute=True,
        )

    Parameters
    ----------
    uri:
        Bolt or neo4j URI, e.g. ``"bolt://localhost:7687"`` or
        ``"neo4j+s://xxx.databases.neo4j.io"``.
    username:
        Neo4j username (default: ``"neo4j"``).
    password:
        Neo4j password.
    database:
        Target database name (default: ``"neo4j"``).

    Raises
    ------
    ImportError
        If the ``neo4j`` package is not installed.
    """

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
    ) -> None:
        try:
            from neo4j import GraphDatabase  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "The 'neo4j' package is required for database execution.\n"
                "Install it with: pip install neo4j"
            ) from exc

        self._driver = GraphDatabase.driver(uri, auth=(username, password))
        self._database = database

    # ------------------------------------------------------------------

    def execute(
        self,
        cypher: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return results as a list of record dicts.

        Parameters
        ----------
        cypher:
            Cypher query string to execute.
        parameters:
            Optional query parameters (e.g. ``{"name": "Alice"}`` for
            ``WHERE n.name = $name``).

        Returns
        -------
        list[dict]
            One dict per returned record, keyed by the ``RETURN`` aliases.
            Empty list when the query returns no rows (e.g. a bare ``CREATE``).
        """
        with self._driver.session(database=self._database) as session:
            result = session.run(cypher, parameters or {})
            return [dict(record) for record in result]

    def execute_many(
        self,
        queries: List[str],
        parameters_list: Optional[List[Optional[Dict[str, Any]]]] = None,
    ) -> List[List[Dict[str, Any]]]:
        """Execute multiple Cypher queries in sequence and return all results.

        Parameters
        ----------
        queries:
            List of Cypher query strings to execute.
        parameters_list:
            Optional list of parameter dicts, one per query.  When *None*
            (or shorter than *queries*), missing entries default to no
            parameters.

        Returns
        -------
        list[list[dict]]
            One inner list per query, each containing the record dicts
            returned by that query.
        """
        params = parameters_list or []
        return [
            self.execute(q, params[i] if i < len(params) else None)
            for i, q in enumerate(queries)
        ]

    def close(self) -> None:
        """Close the underlying driver and release all connections."""
        self._driver.close()

    def __enter__(self) -> "Neo4jDatabase":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"Neo4jDatabase(database={self._database!r})"


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
                "Install it with:  pip install gliner2"
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
    into a single callable object.  Optionally executes the generated query
    directly against a Neo4j database when a :class:`Neo4jDatabase` instance
    is provided.

    Example — generation only
    -------------------------
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
        cypher = pipeline(
            "John works for Apple Inc.",
            ["works_for"],
            mode="create",
        )
        # CREATE (a0:Person {name: "John"})-[:WORKS_FOR]->(b0:Company {name: "Apple Inc."})
        # RETURN a0, b0

    Example — generation + execution (db-aware mode)
    -------------------------------------------------
    ::

        from cypher_validator import NLToCypher, Neo4jDatabase

        db = Neo4jDatabase("bolt://localhost:7687", "neo4j", "password")
        pipeline = NLToCypher.from_pretrained(
            "fastino/gliner2-large-v1",
            schema=schema,
            db=db,
        )
        cypher, results = pipeline(
            "John works for Apple Inc.",
            ["works_for"],
            mode="create",
            execute=True,           # ← db-aware flag
        )
        # cypher  → 'CREATE (a0:Person {name: "John"})-[:WORKS_FOR]->...'
        # results → [{"a0": {...}, "b0": {...}}]  (Neo4j records)
    """

    def __init__(
        self,
        extractor: GLiNER2RelationExtractor,
        schema: Any = None,
        name_property: str = "name",
        db: Optional[Neo4jDatabase] = None,
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
        db:
            Optional :class:`Neo4jDatabase` connection.  Required when
            ``execute=True`` is passed to :meth:`__call__` or
            :meth:`extract_and_convert`.
        """
        self.extractor = extractor
        self.converter = RelationToCypherConverter(
            schema=schema, name_property=name_property
        )
        self.db = db

    # ------------------------------------------------------------------

    @classmethod
    def from_env(
        cls,
        model_name: str = GLiNER2RelationExtractor.DEFAULT_MODEL,
        schema: Any = None,
        threshold: float = 0.5,
        name_property: str = "name",
        database: str = "neo4j",
    ) -> "NLToCypher":
        """Create a pipeline reading Neo4j credentials from environment variables.

        Reads ``NEO4J_URI``, ``NEO4J_USERNAME`` (default ``"neo4j"``), and
        ``NEO4J_PASSWORD`` from the process environment and creates a
        :class:`Neo4jDatabase` connection automatically.

        Parameters
        ----------
        model_name:
            HuggingFace model name or local path.
        schema:
            Optional ``cypher_validator.Schema`` for label-aware queries.
        threshold:
            Confidence threshold for the relation extractor.
        name_property:
            Node property key for entity names.
        database:
            Neo4j database name (default: ``"neo4j"``).

        Returns
        -------
        NLToCypher
            Pipeline with a live Neo4j connection ready for ``execute=True``.

        Raises
        ------
        KeyError
            If ``NEO4J_URI`` or ``NEO4J_PASSWORD`` are not set.
        ImportError
            If the ``gliner2`` or ``neo4j`` packages are not installed.
        """
        uri = os.environ["NEO4J_URI"]
        username = os.environ.get("NEO4J_USERNAME", "neo4j")
        password = os.environ["NEO4J_PASSWORD"]
        db = Neo4jDatabase(uri, username, password, database=database)
        return cls.from_pretrained(
            model_name,
            schema=schema,
            threshold=threshold,
            name_property=name_property,
            db=db,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = GLiNER2RelationExtractor.DEFAULT_MODEL,
        schema: Any = None,
        threshold: float = 0.5,
        name_property: str = "name",
        db: Optional[Neo4jDatabase] = None,
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
        db:
            Optional :class:`Neo4jDatabase` for direct query execution.
            Pass this to enable ``execute=True`` on :meth:`__call__`.

        Returns
        -------
        NLToCypher

        Raises
        ------
        ImportError
            If the ``gliner2`` package is not installed.
        """
        extractor = GLiNER2RelationExtractor.from_pretrained(
            model_name, threshold=threshold
        )
        return cls(extractor, schema=schema, name_property=name_property, db=db)

    # ------------------------------------------------------------------

    def __call__(
        self,
        text: str,
        relation_types: List[str],
        mode: str = "match",
        threshold: Optional[float] = None,
        execute: bool = False,
        **kwargs: Any,
    ) -> Union[str, Tuple[str, List[Dict[str, Any]]]]:
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
        execute:
            When ``True``, execute the generated query against the
            :class:`Neo4jDatabase` supplied at construction and return
            ``(cypher, results)`` instead of just ``cypher``.
            Requires ``db`` to be set.
        **kwargs:
            Extra keyword arguments forwarded to the Cypher converter
            (e.g. ``return_clause="RETURN *"``).

        Returns
        -------
        str
            Cypher query string when ``execute=False`` (default).
        tuple[str, list[dict]]
            ``(cypher, db_results)`` when ``execute=True``.

        Raises
        ------
        RuntimeError
            When ``execute=True`` but no ``db`` was provided.
        """
        relations = self.extractor.extract_relations(
            text, relation_types, threshold=threshold
        )
        cypher = self.converter.convert(relations, mode=mode, **kwargs)

        if execute:
            if self.db is None:
                raise RuntimeError(
                    "execute=True requires a database connection. "
                    "Pass db=Neo4jDatabase(...) when constructing NLToCypher "
                    "or via NLToCypher.from_pretrained(..., db=db)."
                )
            results = self.db.execute(cypher) if cypher else []
            return cypher, results

        return cypher

    # ------------------------------------------------------------------

    def extract_and_convert(
        self,
        text: str,
        relation_types: List[str],
        mode: str = "match",
        threshold: Optional[float] = None,
        execute: bool = False,
        **kwargs: Any,
    ) -> Union[Tuple[Dict[str, Any], str], Tuple[Dict[str, Any], str, List[Dict[str, Any]]]]:
        """Extract relations and return both the raw dict and the Cypher string.

        Parameters
        ----------
        text:
            Input sentence or passage.
        relation_types:
            Relation labels to extract.
        mode:
            Cypher generation mode.
        threshold:
            Override extractor confidence threshold.
        execute:
            When ``True``, also execute the query against the database.
            Requires ``db`` to be set.
        **kwargs:
            Forwarded to the Cypher converter.

        Returns
        -------
        tuple[dict, str]
            ``(relations_dict, cypher_query)`` when ``execute=False``.
        tuple[dict, str, list[dict]]
            ``(relations_dict, cypher_query, db_results)`` when ``execute=True``.

        Raises
        ------
        RuntimeError
            When ``execute=True`` but no ``db`` was provided.
        """
        relations = self.extractor.extract_relations(
            text, relation_types, threshold=threshold
        )
        cypher = self.converter.convert(relations, mode=mode, **kwargs)

        if execute:
            if self.db is None:
                raise RuntimeError(
                    "execute=True requires a database connection. "
                    "Pass db=Neo4jDatabase(...) when constructing NLToCypher."
                )
            results = self.db.execute(cypher) if cypher else []
            return relations, cypher, results

        return relations, cypher

    def __repr__(self) -> str:
        return (
            f"NLToCypher(extractor={self.extractor!r}, "
            f"converter={self.converter!r}, "
            f"db={self.db!r})"
        )
