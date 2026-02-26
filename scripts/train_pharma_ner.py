#!/usr/bin/env python3
"""
scripts/train_pharma_ner.py
===========================
Fine-tune a spaCy transformer NER model for pharma/medical entity extraction.

Datasets (open-license, no DUA required):
  BC5CDR       bigbio/bc5cdr            CHEMICAL, DISEASE
  DrugProt     bigbio/drugprot          CHEMICAL, GENE_OR_PROTEIN
  NCBI Disease bigbio/ncbi_disease      DISEASE
  BioRED       bigbio/biored            CHEMICAL, DISEASE, GENE_OR_PROTEIN,
                                         VARIANT, SPECIES, CELL_LINE

Backbone: microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext

Requirements (already in .venv):
  spacy  spacy-transformers  datasets  torch

Usage:
  python scripts/train_pharma_ner.py prepare                    # download datasets
  python scripts/train_pharma_ner.py config                     # write config
  python scripts/train_pharma_ner.py train --device cpu         # CPU
  python scripts/train_pharma_ner.py train --device mps         # Apple Silicon GPU
  python scripts/train_pharma_ner.py train --device cuda        # NVIDIA GPU 0
  python scripts/train_pharma_ner.py train --device cuda:1      # NVIDIA GPU 1
  python scripts/train_pharma_ner.py evaluate                   # eval on test set
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import spacy
from spacy.tokens import DocBin
from spacy.util import filter_spans

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT   = Path(__file__).resolve().parent.parent
DATA   = ROOT / "data" / "pharma_ner"
MODELS = ROOT / "models" / "pharma_ner"
CONFIG = ROOT / "config_pharma_ner.cfg"

# ---------------------------------------------------------------------------
# Unified label taxonomy
# ---------------------------------------------------------------------------

LABEL_MAP: dict[str, str] = {
    # ── Chemical / Drug ──────────────────────────────────────────────
    "Chemical":              "CHEMICAL",   # BC5CDR
    "CHEMICAL":              "CHEMICAL",   # DrugProt BigBio
    "ChemicalEntity":        "CHEMICAL",   # BioRED
    # ── Disease ──────────────────────────────────────────────────────
    "Disease":               "DISEASE",    # BC5CDR
    "DISEASE":               "DISEASE",
    "SpecificDisease":       "DISEASE",    # NCBI Disease subtypes
    "DiseaseClass":          "DISEASE",
    "Modifier":              "DISEASE",
    "CompositeMention":      "DISEASE",
    "DiseaseOrPhenotypicFeature": "DISEASE",  # BioRED
    # ── Gene / Protein ───────────────────────────────────────────────
    "Gene-or-Protein":       "GENE_OR_PROTEIN",   # (legacy BigBio name)
    "GENE-OR-GENE-PRODUCT":  "GENE_OR_PROTEIN",   # (legacy BigBio name)
    "GENE-Y":                "GENE_OR_PROTEIN",   # DrugProt: annotated gene
    "GENE-N":                "GENE_OR_PROTEIN",   # DrugProt: non-interacting gene
    "GeneOrGeneProduct":     "GENE_OR_PROTEIN",   # BioRED
    # ── Sequence Variant ─────────────────────────────────────────────
    "Sequence-Variant":      "VARIANT",    # (legacy BigBio name)
    "SequenceVariant":       "VARIANT",    # BioRED
    # ── Species ──────────────────────────────────────────────────────
    "Species":               "SPECIES",
    "OrganismTaxon":         "SPECIES",    # BioRED
    # ── Cell Line ────────────────────────────────────────────────────
    "Cell-Line":             "CELL_LINE",  # (legacy BigBio name)
    "CellLine":              "CELL_LINE",  # BioRED
}

ALL_LABELS: list[str] = sorted(set(LABEL_MAP.values()))

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASETS: list[dict] = [
    {
        "hf_id":  "bigbio/bc5cdr",
        "config": "bc5cdr_bigbio_kb",
        "splits": {"train": "train", "dev": "validation", "test": "test"},
        "notes":  "Chemical + Disease, 1,500 PubMed abstracts",
    },
    {
        "hf_id":  "bigbio/drugprot",
        "config": "drugprot_bigbio_kb",
        # DrugProt has no "test" split in BigBio; test_background is the
        # large unannotated background set used for the RE subtask — skip it.
        "splits": {"train": "train", "dev": "validation"},
        "notes":  "Drug + Gene/Protein, 5,000 abstracts (CC-BY 4.0)",
    },
    {
        "hf_id":  "bigbio/ncbi_disease",
        "config": "ncbi_disease_bigbio_kb",
        "splits": {"train": "train", "dev": "validation", "test": "test"},
        "notes":  "Disease, expert-annotated, CC0, 793 abstracts",
    },
    {
        "hf_id":  "bigbio/biored",
        "config": "biored_bigbio_kb",
        "splits": {"train": "train", "dev": "validation", "test": "test"},
        "notes":  "Chemical/Disease/Gene/Variant/Species/CellLine, 600 abstracts",
    },
]

# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def _parse_device(device: str) -> tuple[str, int]:
    """Return (device_type, cuda_index) from a --device string.

    Accepted forms: cpu | mps | cuda | cuda:N
    """
    d = device.strip().lower()
    if d == "cpu":
        return "cpu", -1
    if d == "mps":
        return "mps", 0
    if d == "cuda":
        return "cuda", 0
    if d.startswith("cuda:"):
        try:
            idx = int(d.split(":")[1])
        except (IndexError, ValueError):
            sys.exit(f"Invalid device '{device}'. Use cuda:0, cuda:1, …")
        return "cuda", idx
    sys.exit(f"Unknown device '{device}'. Use: cpu | mps | cuda | cuda:N")


def _device_env(device_type: str) -> dict[str, str]:
    """Return extra environment variables required for the chosen device."""
    env = os.environ.copy()

    # Suppress HuggingFace TRANSFORMERS_CACHE deprecation: forward to HF_HOME,
    # then remove the old var so transformers doesn't emit the FutureWarning.
    if "TRANSFORMERS_CACHE" in env:
        env.setdefault("HF_HOME", env["TRANSFORMERS_CACHE"])
        del env["TRANSFORMERS_CACHE"]

    # Prevent tokenizer parallelism after fork (spaCy spawns workers).
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    if device_type == "mps":
        # Allow CPU fallback for MPS-unsupported ops (e.g. float64 reductions).
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        # Silence the grad-layout UserWarning emitted by PyTorch MPS backend
        # (F-contiguous grad vs C-contiguous param — cosmetic, not a correctness issue).
        prev = env.get("PYTHONWARNINGS", "")
        filter_ = "ignore::UserWarning:torch"
        env["PYTHONWARNINGS"] = f"{prev},{filter_}" if prev else filter_

    return env


def _gpu_id_args(device_type: str, cuda_index: int) -> list[str]:
    """Return the --gpu-id CLI args to pass to spacy train / evaluate."""
    if device_type == "cpu":
        return []
    # Both MPS and CUDA use --gpu-id 0 / N.
    # thinc's prefer_gpu() with gpu_allocator=pytorch picks MPS automatically
    # on Apple Silicon when CUDA is unavailable.
    return ["--gpu-id", str(cuda_index)]


def _detect_available_devices() -> list[str]:
    """Return a list of available compute devices on this machine."""
    available = ["cpu"]
    try:
        import torch  # noqa: PLC0415
        if torch.backends.mps.is_available():
            available.append("mps")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                available.append(f"cuda:{i}")
    except ImportError:
        pass
    return available

# ---------------------------------------------------------------------------
# BigBio KB schema → spaCy Doc conversion
# ---------------------------------------------------------------------------

def _rebuild_text(passages: list[dict]) -> str:
    """Reconstruct full document text from BigBio passage offsets.

    BigBio stores each passage as (text_segment, [char_start, char_end]) in
    the original document coordinate space.  Placing each segment at its
    declared offset keeps entity char spans directly indexable.
    """
    if not passages:
        return ""
    max_end = max(end for p in passages for _s, end in p["offsets"])
    buf = [" "] * max_end
    for passage in passages:
        for seg_text, (start, _end) in zip(passage["text"], passage["offsets"]):
            for i, ch in enumerate(seg_text):
                if start + i < max_end:
                    buf[start + i] = ch
    return "".join(buf)


def bigbio_to_docs(
    examples,
    nlp: spacy.language.Language,
    label_map: dict[str, str] = LABEL_MAP,
) -> list[spacy.tokens.Doc]:
    """Convert BigBio KB schema examples to spaCy Docs with NER annotations."""
    docs: list[spacy.tokens.Doc] = []
    n_skipped = 0

    for ex in examples:
        text = _rebuild_text(ex.get("passages", []))
        if not text.strip():
            continue
        doc = nlp.make_doc(text)
        spans: list[spacy.tokens.Span] = []
        for ent in ex.get("entities", []):
            label = label_map.get(ent["type"])
            if label is None:
                continue
            for char_start, char_end in ent["offsets"]:
                span = doc.char_span(
                    char_start, char_end, label=label, alignment_mode="expand"
                )
                if span is not None:
                    spans.append(span)
                else:
                    n_skipped += 1
        doc.ents = filter_spans(spans)
        docs.append(doc)

    if n_skipped:
        print(f"    [{n_skipped} spans could not be aligned — skipped]")
    return docs

# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------

def prepare(args: argparse.Namespace) -> None:
    """Download all datasets from HuggingFace and write .spacy DocBin files."""
    try:
        from datasets import load_dataset  # noqa: PLC0415
    except ImportError:
        sys.exit("Missing dependency: pip install datasets")

    DATA.mkdir(parents=True, exist_ok=True)
    nlp  = spacy.blank("en")
    bins: dict[str, list[spacy.tokens.Doc]] = {"train": [], "dev": [], "test": []}

    for ds_cfg in DATASETS:
        hf_id  = ds_cfg["hf_id"]
        config = ds_cfg["config"]
        print(f"\n{'─' * 60}")
        print(f"  {hf_id}  [{ds_cfg.get('notes', '')}]")
        print(f"{'─' * 60}")
        try:
            dataset = load_dataset(hf_id, name=config, trust_remote_code=True)
        except Exception as exc:
            print(f"  WARNING: could not load — {exc}")
            continue
        for split_key, split_name in ds_cfg["splits"].items():
            if split_name not in dataset:
                print(f"  split '{split_name}' not found — skipping")
                continue
            docs = bigbio_to_docs(dataset[split_name], nlp)
            counts: dict[str, int] = {}
            for doc in docs:
                for ent in doc.ents:
                    counts[ent.label_] = counts.get(ent.label_, 0) + 1
            summary = "  ".join(f"{l}:{c}" for l, c in sorted(counts.items()))
            print(f"  {split_name:12s} → {len(docs):5d} docs  [{summary}]  → {split_key}")
            bins[split_key].extend(docs)

    print(f"\n{'═' * 60}")
    print("Writing .spacy files …")
    total = 0
    for split_key, docs in bins.items():
        if not docs:
            print(f"  {split_key:8s}:     0 docs — skipping")
            continue
        path = DATA / f"{split_key}.spacy"
        DocBin(docs=docs).to_disk(path)
        print(f"  {split_key:8s}: {len(docs):5d} docs → {path}")
        total += len(docs)

    print(f"\nDone. {total} total docs.  Labels: {ALL_LABELS}")
    print(f"\nNext step: python scripts/train_pharma_ner.py config")

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

# ${{...}} becomes ${...} after Python .format() — spaCy's interpolation.
_CONFIG_TEMPLATE = """\
# spaCy transformer NER config — pharma/medical NER
# Backbone : PubMedBERT (pre-trained on PubMed abstracts + PMC full-text)
# Generated: scripts/train_pharma_ner.py config

[paths]
train = "{train}"
dev   = "{dev}"
vectors = null
init_tok2vec = null

[system]
gpu_allocator = "pytorch"
seed = 42

[nlp]
lang = "en"
pipeline = ["transformer", "ner"]
batch_size = 128

[components]

[components.transformer]
factory = "transformer"

[components.transformer.model]
@architectures = "spacy-transformers.TransformerModel.v3"
name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
mixed_precision = false

[components.transformer.model.get_spans]
@span_getters = "spacy-transformers.strided_spans.v1"
window = 128
stride = 96

[components.ner]
factory = "ner"
moves = null
update_with_oracle_cut_size = 100

[components.ner.model]
@architectures = "spacy.TransitionBasedParser.v2"
state_type = "ner"
extra_state_tokens = false
hidden_width = 128
maxout_pieces = 3
use_upper = false
nO = null

[components.ner.model.tok2vec]
@architectures = "spacy-transformers.TransformerListener.v1"
grad_factor = 1.0

[components.ner.model.tok2vec.pooling]
@layers = "reduce_mean.v1"

[corpora]

[corpora.train]
@readers = "spacy.Corpus.v1"
path = ${{paths.train}}
max_length = 0
gold_preproc = false
limit = 0
augmenter = null

[corpora.dev]
@readers = "spacy.Corpus.v1"
path = ${{paths.dev}}
max_length = 0
gold_preproc = false
limit = 0
augmenter = null

[training]
# accumulate_gradient: effective batch = 4 × micro-batch (saves VRAM on MPS/CUDA)
accumulate_gradient = 4
dev_corpus = "corpora.dev"
train_corpus = "corpora.train"
seed = ${{system.seed}}
gpu_allocator = ${{system.gpu_allocator}}
dropout = 0.15
patience = 2400
max_epochs = 0
max_steps = 20000
eval_frequency = 400
frozen_components = []
annotating_components = []
before_to_disk = null
before_update = null

[training.batcher]
@batchers = "spacy.batch_by_padded.v1"
discard_oversize = true
size = 2000
buffer = 256
get_length = null

[training.logger]
@loggers = "spacy.ConsoleLogger.v3"
progress_bar = "train"

[training.optimizer]
@optimizers = "Adam.v1"
beta1 = 0.9
beta2 = 0.999
L2_is_weight_decay = true
L2 = 0.01
grad_clip = 1.0
use_averages = false
eps = 1e-8

[training.optimizer.learn_rate]
@schedules = "warmup_linear.v1"
warmup_steps = 1000
total_steps = 20000
initial_rate = 2e-5

[training.score_weights]
ents_per_type = null
ents_f = 1.0
ents_p = 0.0
ents_r = 0.0

[initialize]
vectors = ${{paths.vectors}}
init_tok2vec = ${{paths.init_tok2vec}}
vocab_data = null
lookups = null
before_init = null
after_init = null

[initialize.components]

[initialize.tokenizer]
"""


def config(args: argparse.Namespace) -> None:
    """Write config_pharma_ner.cfg to the project root."""
    text = _CONFIG_TEMPLATE.format(
        train = (DATA / "train.spacy").as_posix(),
        dev   = (DATA / "dev.spacy").as_posix(),
    )
    CONFIG.write_text(text)
    print(f"Config written → {CONFIG}")
    print(f"\nAvailable devices   : {_detect_available_devices()}")
    print(f"\nNext steps:")
    print(f"  python scripts/train_pharma_ner.py train --device cpu")
    print(f"  python scripts/train_pharma_ner.py train --device mps   # Apple Silicon")
    print(f"  python scripts/train_pharma_ner.py train --device cuda  # NVIDIA")

# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    """Run spacy train with MPS / CUDA / CPU device selection."""
    if not CONFIG.exists():
        sys.exit(
            f"Config not found: {CONFIG}\n"
            "Run: python scripts/train_pharma_ner.py config"
        )
    if not (DATA / "train.spacy").exists():
        sys.exit(
            "Training data not found.\n"
            "Run: python scripts/train_pharma_ner.py prepare"
        )

    device_type, cuda_index = _parse_device(args.device)
    env = _device_env(device_type)

    MODELS.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "spacy", "train",
        str(CONFIG),
        "--output", str(MODELS),
        *_gpu_id_args(device_type, cuda_index),
    ]

    print(f"Device  : {args.device}  (type={device_type}, gpu_id={cuda_index})")
    if device_type == "mps":
        print("MPS env : PYTORCH_ENABLE_MPS_FALLBACK=1")
    print(f"\nRunning: {' '.join(cmd)}\n")

    subprocess.run(cmd, check=True, env=env)
    print(f"\nBest model → {MODELS / 'model-best'}")
    print(f"Next step  : python scripts/train_pharma_ner.py evaluate --device {args.device}")

# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> None:
    """Evaluate model-best on the held-out test set."""
    best    = MODELS / "model-best"
    test    = DATA / "test.spacy"
    metrics = MODELS / "metrics.json"

    if not best.exists():
        sys.exit(f"model-best not found at {best}\nRun training first.")
    if not test.exists():
        sys.exit("test.spacy not found.\nRun: python scripts/train_pharma_ner.py prepare")

    device_type, cuda_index = _parse_device(args.device)
    env = _device_env(device_type)

    cmd = [
        sys.executable, "-m", "spacy", "evaluate",
        str(best), str(test),
        "--output", str(metrics),
        *_gpu_id_args(device_type, cuda_index),
    ]
    print(f"Device  : {args.device}\n")
    print(f"Running : {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True, env=env)

    if metrics.exists():
        import json
        data  = json.loads(metrics.read_text())
        ents  = data.get("ents_per_type", {})
        print(f"\n{'Label':20s}  {'P':>6}  {'R':>6}  {'F':>6}")
        print("─" * 44)
        for label, scores in sorted(ents.items()):
            print(
                f"  {label:18s}  {scores['p']:6.1f}  "
                f"{scores['r']:6.1f}  {scores['f']:6.1f}"
            )
        print(f"\n  Overall F1 : {data.get('ents_f', 0):.2f}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune a spaCy transformer NER model for pharma/medical text.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("prepare", help="Download HuggingFace datasets → .spacy files")

    sub.add_parser("config", help="Write spaCy training config")

    for cmd_name in ("train", "evaluate"):
        p = sub.add_parser(cmd_name, help=f"Run spacy {cmd_name}")
        p.add_argument(
            "--device", default="cpu",
            metavar="DEVICE",
            help="Compute device: cpu | mps | cuda | cuda:N  (default: cpu)",
        )

    args = parser.parse_args()
    {
        "prepare":  prepare,
        "config":   config,
        "train":    train,
        "evaluate": evaluate,
    }[args.command](args)


if __name__ == "__main__":
    main()
