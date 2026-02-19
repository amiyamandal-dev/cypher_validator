"""
Tests for cypher_validator.llm_utils — LLM integration helpers.

All tests are self-contained; no real LLM or Neo4j connection is needed.
"""
from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock

from cypher_validator import (
    Schema,
    CypherValidator,
    CypherGenerator,
    extract_cypher_from_text,
    format_records,
    repair_cypher,
    cypher_tool_spec,
    few_shot_examples,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def schema():
    return Schema(
        nodes={"Person": ["name", "age"], "Company": ["name", "founded"]},
        relationships={"WORKS_FOR": ("Person", "Company", ["since", "role"])},
    )


@pytest.fixture()
def validator(schema):
    return CypherValidator(schema)


@pytest.fixture()
def generator(schema):
    return CypherGenerator(schema, seed=42)


# ---------------------------------------------------------------------------
# extract_cypher_from_text
# ---------------------------------------------------------------------------

class TestExtractCypherFromText:
    def test_fenced_cypher_block(self):
        text = "Here you go:\n\n```cypher\nMATCH (n:Person) RETURN n\n```"
        assert extract_cypher_from_text(text) == "MATCH (n:Person) RETURN n"

    def test_fenced_sql_block(self):
        text = "```sql\nMATCH (n) RETURN n\n```"
        assert extract_cypher_from_text(text) == "MATCH (n) RETURN n"

    def test_fenced_plain_block(self):
        text = "```\nCREATE (n:Person {name: 'Alice'})\n```"
        assert extract_cypher_from_text(text) == "CREATE (n:Person {name: 'Alice'})"

    def test_inline_backtick(self):
        text = "Run `MATCH (n) RETURN n` on your database."
        assert extract_cypher_from_text(text) == "MATCH (n) RETURN n"

    def test_line_anchored_match(self):
        text = "Sure!\nMATCH (n:Person) RETURN n\n\nDone."
        result = extract_cypher_from_text(text)
        assert "MATCH" in result
        assert "RETURN" in result

    def test_line_anchored_create(self):
        text = "Here:\nCREATE (n:Person {name: 'Bob'}) RETURN n"
        result = extract_cypher_from_text(text)
        assert "CREATE" in result

    def test_fallback_returns_stripped_text(self):
        text = "   just some plain words   "
        assert extract_cypher_from_text(text) == "just some plain words"

    def test_empty_string(self):
        assert extract_cypher_from_text("") == ""

    def test_whitespace_only(self):
        assert extract_cypher_from_text("   \n\t  ") == ""

    def test_multiline_query_in_fence(self):
        text = (
            "```cypher\n"
            "MATCH (p:Person)-[:WORKS_FOR]->(c:Company)\n"
            "WHERE p.name = $name\n"
            "RETURN p, c\n"
            "```"
        )
        result = extract_cypher_from_text(text)
        assert "MATCH" in result
        assert "RETURN" in result
        assert "WHERE" in result

    def test_fenced_block_takes_priority_over_inline(self):
        text = "Run `MATCH (x) RETURN x` or:\n```cypher\nMATCH (n:Person) RETURN n\n```"
        result = extract_cypher_from_text(text)
        assert ":Person" in result  # fenced block wins

    def test_case_insensitive_fence_tag(self):
        text = "```CYPHER\nMATCH (n) RETURN n\n```"
        result = extract_cypher_from_text(text)
        assert "MATCH" in result


# ---------------------------------------------------------------------------
# format_records
# ---------------------------------------------------------------------------

class TestFormatRecords:
    def test_empty_records_returns_empty_string(self):
        assert format_records([]) == ""

    def test_markdown_basic(self):
        records = [{"name": "Alice", "age": "30"}]
        result = format_records(records, format="markdown")
        assert "name" in result
        assert "Alice" in result
        assert "|" in result

    def test_markdown_multiple_rows(self):
        records = [
            {"name": "Alice", "age": "30"},
            {"name": "Bob",   "age": "25"},
        ]
        result = format_records(records)
        assert "Alice" in result
        assert "Bob" in result
        lines = result.splitlines()
        assert len(lines) == 4   # header + separator + 2 data rows

    def test_csv_format(self):
        records = [{"x": "1", "y": "2"}, {"x": "3", "y": "4"}]
        result = format_records(records, format="csv")
        assert "x,y" in result
        assert "1,2" in result
        assert "3,4" in result

    def test_json_format(self):
        records = [{"key": "val"}]
        result = format_records(records, format="json")
        parsed = json.loads(result)
        assert parsed == [{"key": "val"}]

    def test_text_format(self):
        records = [{"a": "hello"}]
        result = format_records(records, format="text")
        assert "Record 1" in result
        assert "a: hello" in result

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError, match="Unknown format"):
            format_records([{"a": 1}], format="xml")

    def test_non_string_values_stringified(self):
        records = [{"count": 42, "flag": True}]
        result = format_records(records, format="markdown")
        assert "42" in result
        assert "True" in result

    def test_markdown_column_alignment(self):
        records = [{"short": "a", "a_longer_header": "value"}]
        result = format_records(records)
        # All three lines (header, sep, data) should have the same number of |
        lines = result.splitlines()
        pipe_counts = [line.count("|") for line in lines]
        assert len(set(pipe_counts)) == 1  # all equal


# ---------------------------------------------------------------------------
# repair_cypher
# ---------------------------------------------------------------------------

class TestRepairCypher:
    def test_valid_query_not_repaired(self, validator):
        query = "MATCH (n:Person) RETURN n"
        result_query, result = repair_cypher(validator, query, lambda q, e: "UNREACHABLE")
        assert result.is_valid
        assert result_query == query

    def test_invalid_query_calls_llm(self, validator):
        bad_query = "MATCH (n:Fake) RETURN n"
        fixed_query = "MATCH (n:Person) RETURN n"
        calls = []

        def llm_fn(q, errors):
            calls.append((q, errors))
            return fixed_query

        final_query, result = repair_cypher(validator, bad_query, llm_fn)
        assert final_query == fixed_query
        assert result.is_valid
        assert len(calls) == 1

    def test_repair_includes_error_messages(self, validator):
        bad_query = "MATCH (n:DoesNotExist) RETURN n"
        captured_errors = []

        def llm_fn(q, errors):
            captured_errors.extend(errors)
            return "MATCH (n:Person) RETURN n"

        repair_cypher(validator, bad_query, llm_fn)
        assert any("DoesNotExist" in e for e in captured_errors)

    def test_max_retries_respected(self, validator):
        calls = []

        def llm_fn(q, errors):
            calls.append(q)
            return "MATCH (n:StillFake) RETURN n"   # never valid

        final_query, result = repair_cypher(
            validator, "MATCH (n:Fake) RETURN n", llm_fn, max_retries=2
        )
        assert len(calls) == 2
        assert not result.is_valid

    def test_zero_retries_returns_original(self, validator):
        bad_query = "MATCH (n:Nope) RETURN n"
        final_query, result = repair_cypher(
            validator, bad_query, lambda q, e: "NEVER_CALLED", max_retries=0
        )
        assert final_query == bad_query
        assert not result.is_valid


# ---------------------------------------------------------------------------
# cypher_tool_spec
# ---------------------------------------------------------------------------

class TestCypherToolSpec:
    def test_anthropic_format_keys(self, schema):
        spec = cypher_tool_spec(schema, format="anthropic")
        assert "name" in spec
        assert spec["name"] == "execute_cypher"
        assert "input_schema" in spec
        assert "description" in spec

    def test_anthropic_schema_embedded(self, schema):
        spec = cypher_tool_spec(schema, format="anthropic")
        assert ":Person" in spec["description"] or "Person" in spec["description"]

    def test_openai_format_keys(self, schema):
        spec = cypher_tool_spec(schema, format="openai")
        assert spec["type"] == "function"
        assert "function" in spec
        assert spec["function"]["name"] == "execute_cypher"

    def test_no_schema_still_works(self):
        spec = cypher_tool_spec(format="anthropic")
        assert spec["name"] == "execute_cypher"

    def test_db_description_in_output(self, schema):
        spec = cypher_tool_spec(schema, db_description="employee knowledge graph")
        assert "employee knowledge graph" in spec["description"]

    def test_required_cypher_parameter(self, schema):
        spec = cypher_tool_spec(schema, format="anthropic")
        assert "cypher" in spec["input_schema"]["required"]

    def test_optional_parameters_field(self, schema):
        spec = cypher_tool_spec(schema, format="anthropic")
        assert "parameters" in spec["input_schema"]["properties"]


# ---------------------------------------------------------------------------
# few_shot_examples
# ---------------------------------------------------------------------------

class TestFewShotExamples:
    def test_returns_n_examples(self, generator):
        examples = few_shot_examples(generator, n=6)
        assert len(examples) == 6

    def test_each_example_is_two_tuple(self, generator):
        examples = few_shot_examples(generator, n=3)
        for item in examples:
            assert len(item) == 2
            desc, cypher = item
            assert isinstance(desc, str)
            assert isinstance(cypher, str)

    def test_descriptions_are_non_empty(self, generator):
        examples = few_shot_examples(generator, n=4)
        for desc, _ in examples:
            assert desc.strip() != ""

    def test_cypher_strings_are_non_empty(self, generator):
        examples = few_shot_examples(generator, n=4)
        for _, cypher in examples:
            assert cypher.strip() != ""

    def test_specific_query_type(self, generator):
        examples = few_shot_examples(generator, n=3, query_type="match_return")
        for _, cypher in examples:
            assert "MATCH" in cypher
            assert "RETURN" in cypher

    def test_invalid_query_type_raises(self, generator):
        with pytest.raises(ValueError, match="Unknown query_type"):
            few_shot_examples(generator, n=1, query_type="nonexistent_type")

    def test_spreads_across_all_types(self, generator):
        n_types = len(CypherGenerator.supported_types())
        examples = few_shot_examples(generator, n=n_types)
        # Every type should appear at least once
        types_seen = set()
        for _, cypher in examples:
            for qt in CypherGenerator.supported_types():
                if qt.split("_")[0].upper() in cypher:
                    types_seen.add(qt)
        assert len(examples) == n_types

    def test_reproducible_with_seed(self, schema):
        gen1 = CypherGenerator(schema, seed=7)
        gen2 = CypherGenerator(schema, seed=7)
        ex1 = few_shot_examples(gen1, n=4)
        ex2 = few_shot_examples(gen2, n=4)
        assert ex1 == ex2


# ---------------------------------------------------------------------------
# Schema prompt-format methods
# ---------------------------------------------------------------------------

class TestSchemaPromptMethods:
    def test_to_prompt_contains_labels(self, schema):
        p = schema.to_prompt()
        assert "Person" in p
        assert "Company" in p
        assert "WORKS_FOR" in p

    def test_to_prompt_contains_properties(self, schema):
        p = schema.to_prompt()
        assert "name" in p
        assert "since" in p

    def test_to_prompt_has_sections(self, schema):
        p = schema.to_prompt()
        assert "Nodes" in p
        assert "Relationships" in p

    def test_to_markdown_has_table_syntax(self, schema):
        md = schema.to_markdown()
        assert "|" in md
        assert "Person" in md
        assert "WORKS_FOR" in md

    def test_to_markdown_has_headers(self, schema):
        md = schema.to_markdown()
        assert "Label" in md
        assert "Type" in md

    def test_to_cypher_context_has_patterns(self, schema):
        ctx = schema.to_cypher_context()
        assert "(:Person" in ctx
        assert ":WORKS_FOR" in ctx
        assert "->(:Company)" in ctx

    def test_to_cypher_context_includes_properties(self, schema):
        ctx = schema.to_cypher_context()
        assert "name" in ctx
        assert "since" in ctx

    def test_empty_schema_no_crash(self):
        empty = Schema(nodes={}, relationships={})
        assert isinstance(empty.to_prompt(), str)
        assert isinstance(empty.to_markdown(), str)
        assert isinstance(empty.to_cypher_context(), str)
