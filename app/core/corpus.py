"""Corpus loading and deterministic chunking.

Chunking contract (v1): a chunk is one paragraph, produced by splitting a
document's text on blank lines. Chunk IDs are "{doc_id}::c{index}" and are
stable as long as the corpus file and this splitting rule do not change.
The eval dataset's ground-truth chunk IDs depend on this contract; any
change to it requires a new corpus/dataset version, not an edit in place.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str


class CorpusFormatError(ValueError):
    """Raised when the corpus file violates the expected JSONL schema."""


def load_chunks(corpus_path: Path) -> list[Chunk]:
    """Load a JSONL corpus and split each document into paragraph chunks."""
    chunks: list[Chunk] = []
    with corpus_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CorpusFormatError(
                    f"{corpus_path}:{line_no}: invalid JSON: {exc}"
                ) from exc
            missing = {"doc_id", "title", "text"} - doc.keys()
            if missing:
                raise CorpusFormatError(
                    f"{corpus_path}:{line_no}: missing fields {sorted(missing)}"
                )
            paragraphs = [p.strip() for p in doc["text"].split("\n\n") if p.strip()]
            if not paragraphs:
                raise CorpusFormatError(
                    f"{corpus_path}:{line_no}: document {doc['doc_id']!r} has no text"
                )
            for i, para in enumerate(paragraphs):
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc['doc_id']}::c{i}",
                        doc_id=doc["doc_id"],
                        title=doc["title"],
                        text=para,
                    )
                )
    if not chunks:
        raise CorpusFormatError(f"{corpus_path}: corpus produced zero chunks")
    return chunks
