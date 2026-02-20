"""
Example 06 — EntityNERExtractor: spaCy and HuggingFace backends.

spaCy backend requires Python < 3.14 (pydantic v1 limitation in current spaCy).
HuggingFace backend works on Python 3.14+.

Run:
    # spaCy:
    pip install "cypher_validator[ner-spacy]"
    python -m spacy download en_core_web_sm
    python examples/06_ner_extractors.py --backend spacy

    # HuggingFace:
    pip install "cypher_validator[ner-transformers]"
    python examples/06_ner_extractors.py --backend hf

    # Both (default):
    python examples/06_ner_extractors.py
"""
import sys

backend_arg = sys.argv[1] if len(sys.argv) > 1 else "both"

from cypher_validator import EntityNERExtractor

SENTENCES = [
    "John works for Apple Inc. and lives in San Francisco.",
    "Bob acquired TechCorp in 2019.",
    "Alice manages the Engineering team at Microsoft.",
    "Mary founded StartupCo in Berlin.",
]

DIVIDER = "-" * 60

# ── spaCy backend ─────────────────────────────────────────────────────────────
if backend_arg in ("spacy", "both"):
    print(f"\n{DIVIDER}\nspaCy NER (en_core_web_sm)\n{DIVIDER}")
    try:
        ner = EntityNERExtractor.from_spacy("en_core_web_sm")
        for text in SENTENCES:
            ents = ner.extract(text)
            print(f"\n  {text!r}")
            for e in ents:
                print(f"    {e['text']:30s}  → {e['label']}")
    except Exception as e:
        print(f"  spaCy unavailable: {e}")
        print("  (spaCy requires Python < 3.14 due to pydantic v1 dependency)")

# ── HuggingFace backend ───────────────────────────────────────────────────────
if backend_arg in ("hf", "both"):
    print(f"\n{DIVIDER}\nHuggingFace NER (dbmdz/bert-large-cased-finetuned-conll03-english)\n{DIVIDER}")
    try:
        import warnings; warnings.filterwarnings("ignore")
        import sys, io
        old_stderr = sys.stderr; sys.stderr = io.StringIO()
        ner_hf = EntityNERExtractor.from_transformers(
            "dbmdz/bert-large-cased-finetuned-conll03-english",
            label_map={"PER": "Person", "ORG": "Company", "LOC": "City"},
        )
        sys.stderr = old_stderr

        for text in SENTENCES:
            ents = ner_hf.extract(text)
            print(f"\n  {text!r}")
            for e in ents:
                print(f"    {e['text']:30s}  → {e['label']}")
    except Exception as e:
        print(f"  HuggingFace unavailable: {e}")

# ── Custom label_map ──────────────────────────────────────────────────────────
print(f"\n{DIVIDER}\nCustom label_map example\n{DIVIDER}")
print("""
    # Override built-in defaults to match your graph schema exactly:
    ner = EntityNERExtractor.from_transformers(
        "dbmdz/bert-large-cased-finetuned-conll03-english",
        label_map={
            "PER": "Employee",        # custom: PER → Employee
            "ORG": "Organization",
            "LOC": "GeographicArea",
        },
    )
""")
