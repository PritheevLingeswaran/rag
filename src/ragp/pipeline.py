"""Stage 0 skeleton pipeline: BM25 retrieval + extractive answer stub.

The "generator" here is deliberately trivial: it returns the first two
sentences of the top-ranked chunk. It exists so the eval harness exercises
a complete query -> retrieve -> answer path from day one. Because the
answer is extracted verbatim from retrieved context, its hallucination
rate should measure ~0; that is the honest baseline, and the metric
becomes meaningful once abstractive LLM generation lands.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .bm25 import BM25Index
from .corpus import Chunk, load_chunks

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class PipelineResult:
    query: str
    retrieved_chunk_ids: list[str]
    retrieved_texts: list[str]
    answer: str


class SkeletonPipeline:
    def __init__(self, corpus_path: Path, top_k: int = 10) -> None:
        self.top_k = top_k
        self._chunks: dict[str, Chunk] = {
            c.chunk_id: c for c in load_chunks(corpus_path)
        }
        self._index = BM25Index()
        self._index.build([(c.chunk_id, c.text) for c in self._chunks.values()])

    def run(self, query: str) -> PipelineResult:
        hits = self._index.search(query, top_k=self.top_k)
        chunk_ids = [cid for cid, _ in hits]
        texts = [self._chunks[cid].text for cid in chunk_ids]
        if texts:
            sentences = _SENTENCE_RE.split(texts[0])
            answer = " ".join(sentences[:2]).strip()
        else:
            answer = "No relevant documents found."
        return PipelineResult(
            query=query,
            retrieved_chunk_ids=chunk_ids,
            retrieved_texts=texts,
            answer=answer,
        )
