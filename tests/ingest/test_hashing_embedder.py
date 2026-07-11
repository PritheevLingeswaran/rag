"""Isolated unit tests for the deterministic HashingEmbedder (Stage 2's
Embedder-protocol stub, still used by ingestion tests)."""

import numpy as np
import pytest

from app.errors import EmbeddingError
from app.ingest.embedder import HashingEmbedder


def test_deterministic_across_instances():
    a = HashingEmbedder(dim=64).embed_batch(["hello world", "bm25"])
    b = HashingEmbedder(dim=64).embed_batch(["hello world", "bm25"])
    assert np.array_equal(a, b)


def test_output_shape_dtype_and_normalization():
    out = HashingEmbedder(dim=32).embed_batch(["some text here"])
    assert out.shape == (1, 32)
    assert out.dtype == np.float32
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)


def test_different_texts_differ():
    out = HashingEmbedder(dim=64).embed_batch(["cats", "quorum replication"])
    assert not np.array_equal(out[0], out[1])


def test_embedder_id_encodes_config():
    assert HashingEmbedder(dim=128).embedder_id == "hashing-v1-d128"


def test_empty_batch_raises():
    with pytest.raises(EmbeddingError, match="empty batch"):
        HashingEmbedder().embed_batch([])


def test_invalid_dim_raises():
    with pytest.raises(ValueError):
        HashingEmbedder(dim=0)
