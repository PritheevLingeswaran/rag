"""Local ONNX text models: embedder + cross-encoder reranker.

Built directly on onnxruntime + tokenizers (not fastembed) because the
512MB Render cap requires session-level memory control that fastembed does
not expose:
  - enable_cpu_mem_arena = False  (arenas grew RSS ~50MB per burst)
  - session.disable_prepacking    (prepacking re-materializes int8 weights)
  - 1 intra-op thread             (matches Render's 0.1 CPU; keeps memory flat)
Measured numbers behind these choices: scripts/measure_memory.py.

Both classes take local model file paths and never touch the network;
download_model() is the only function that does, and it is called
explicitly at bootstrap, never implicitly at query time.

OnnxEmbedder implements the Stage 2 Embedder protocol (embedder_id, dim,
embed_batch) so ingestion can use it as a drop-in for HashingEmbedder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from app.errors import EmbeddingError

# Pinned model choices (see docs/loadtest_stage3.md for the RAM data
# behind them). Both are int8-quantized ONNX exports.
EMBED_MODEL_REPO = "Xenova/all-MiniLM-L6-v2"
EMBED_MODEL_FILE = "onnx/model_quantized.onnx"
EMBED_DIM = 384
EMBED_POOLING: Literal["mean", "cls"] = "mean"
RERANK_MODEL_REPO = "Xenova/ms-marco-MiniLM-L-6-v2"
RERANK_MODEL_FILE = "onnx/model_quantized.onnx"
TOKENIZER_FILE = "tokenizer.json"


def download_model(repo: str, model_file: str,
                   cache_dir: str | None = None) -> tuple[Path, Path]:
    """Fetch (model.onnx, tokenizer.json) into the local HF cache and
    return their paths. Network happens here and only here."""
    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download(repo, model_file, cache_dir=cache_dir)
    tok_path = hf_hub_download(repo, TOKENIZER_FILE, cache_dir=cache_dir)
    return Path(model_path), Path(tok_path)


def _make_session(model_path: Path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.enable_cpu_mem_arena = False
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.add_session_config_entry("session.disable_prepacking", "1")
    return ort.InferenceSession(
        str(model_path), sess_options=so, providers=["CPUExecutionProvider"]
    )


def _pad_batch(encodings) -> dict[str, np.ndarray]:
    maxlen = max(len(e.ids) for e in encodings)
    ids = np.zeros((len(encodings), maxlen), dtype=np.int64)
    mask = np.zeros_like(ids)
    types = np.zeros_like(ids)
    for i, e in enumerate(encodings):
        n = len(e.ids)
        ids[i, :n] = e.ids
        mask[i, :n] = e.attention_mask
        types[i, :n] = e.type_ids
    return {"input_ids": ids, "attention_mask": mask, "token_type_ids": types}


class OnnxEmbedder:
    """Sentence embedder over a quantized ONNX transformer.

    Satisfies the Embedder protocol from app.ingest.embedder.
    """

    def __init__(self, model_path: Path, tokenizer_path: Path,
                 dim: int = EMBED_DIM,
                 pooling: Literal["mean", "cls"] = EMBED_POOLING,
                 max_length: int = 256,
                 model_tag: str = "minilm-l6-q-v1") -> None:
        self._session = _make_session(model_path)
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._tokenizer.enable_truncation(max_length=max_length)
        self._dim = dim
        self._pooling = pooling
        self._model_tag = model_tag

    @property
    def embedder_id(self) -> str:
        return f"{self._model_tag}-{self._pooling}-d{self._dim}"

    @property
    def dim(self) -> int:
        return self._dim

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            raise EmbeddingError("embed_batch called with empty batch")
        try:
            encs = [self._tokenizer.encode(t) for t in texts]
            inputs = _pad_batch(encs)
            hidden = self._session.run(None, inputs)[0]
        except (RuntimeError, ValueError, ort.capi.onnxruntime_pybind11_state.Fail) as exc:
            raise EmbeddingError(f"onnx embedding failed: {exc}") from exc
        if self._pooling == "cls":
            pooled = hidden[:, 0]
        else:
            mask = inputs["attention_mask"][:, :, None].astype(np.float32)
            pooled = (hidden * mask).sum(axis=1) / np.clip(
                mask.sum(axis=1), 1e-9, None
            )
        if pooled.shape[1] != self._dim:
            raise EmbeddingError(
                f"model produced dim {pooled.shape[1]}, expected {self._dim}"
            )
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return (pooled / np.clip(norms, 1e-12, None)).astype(np.float32)


class OnnxReranker:
    """Cross-encoder relevance scorer over a quantized ONNX transformer.

    max_length=256 and micro_batch=5 are memory guards, not just tuning:
    transformer attention activations scale with batch x seq^2 (a 20x512
    batch transiently allocates ~250MB, blowing the 512MB container cap;
    5x256 stays ~16MB). Our chunks are ~100-150 tokens, so 256 loses
    nothing. Measured in scripts/measure_memory.py.

    Quantization caveat (measured, tests/integration/test_onnx_models.py):
    the model is dynamically int8-quantized, so activation scales depend
    on batch composition -- absolute logits shift ~0.1-0.3 when the same
    pair is scored in a different batch. Scores are therefore only
    comparable within one micro_batch configuration; micro_batch is fixed
    at construction and identical inputs always produce identical outputs.
    """

    def __init__(self, model_path: Path, tokenizer_path: Path,
                 max_length: int = 256, micro_batch: int = 5) -> None:
        if micro_batch <= 0:
            raise ValueError("micro_batch must be positive")
        self._session = _make_session(model_path)
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._tokenizer.enable_truncation(max_length=max_length)
        self._micro_batch = micro_batch

    @property
    def micro_batch(self) -> int:
        """Batch granularity; the hybrid pipeline's budget check runs
        between micro-batches, so this is also the budget's resolution."""
        return self._micro_batch

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        """Relevance logit for each (query, passage) pair; higher = better."""
        if not passages:
            raise ValueError("score called with no passages")
        chunks: list[np.ndarray] = []
        for start in range(0, len(passages), self._micro_batch):
            batch = passages[start:start + self._micro_batch]
            encs = [self._tokenizer.encode(query, p) for p in batch]
            inputs = _pad_batch(encs)
            logits = self._session.run(None, inputs)[0]
            chunks.append(logits.reshape(-1))
        return np.concatenate(chunks).astype(np.float32)

    def rerank(self, query: str, candidates: list[tuple[str, str]],
               top_k: int | None = None) -> list[tuple[str, float]]:
        """candidates: (chunk_id, text). Returns (chunk_id, score), best
        first; ties break by candidate input order (deterministic)."""
        scores = self.score(query, [text for _, text in candidates])
        order = sorted(
            range(len(candidates)), key=lambda i: (-float(scores[i]), i)
        )
        if top_k is not None:
            order = order[:top_k]
        return [(candidates[i][0], float(scores[i])) for i in order]
