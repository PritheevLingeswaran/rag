import pytest

from app.core.corpus import CorpusFormatError, load_chunks


def test_load_chunks_splits_paragraphs(tmp_path):
    p = tmp_path / "corpus.jsonl"
    p.write_text(
        '{"doc_id": "d1", "title": "T", "text": "para one.\\n\\npara two."}\n',
        encoding="utf-8",
    )
    chunks = load_chunks(p)
    assert [c.chunk_id for c in chunks] == ["d1::c0", "d1::c1"]
    assert chunks[0].text == "para one."
    assert chunks[1].text == "para two."


def test_load_chunks_rejects_missing_field(tmp_path):
    p = tmp_path / "corpus.jsonl"
    p.write_text('{"doc_id": "d1", "title": "T"}\n', encoding="utf-8")
    with pytest.raises(CorpusFormatError):
        load_chunks(p)


def test_load_chunks_rejects_empty_corpus(tmp_path):
    p = tmp_path / "corpus.jsonl"
    p.write_text("", encoding="utf-8")
    with pytest.raises(CorpusFormatError):
        load_chunks(p)


def test_load_chunks_rejects_invalid_json(tmp_path):
    p = tmp_path / "corpus.jsonl"
    p.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(CorpusFormatError):
        load_chunks(p)
