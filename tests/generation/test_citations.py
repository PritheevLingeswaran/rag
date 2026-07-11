"""Citation validator tests, including the Stage 4 definition-of-done:
a deliberately fabricated claim must be caught and rejected."""

import pytest

from app.generation.citations import CitationValidator

RETRIEVED = [
    ("raft::c0",
     "Raft divides time into terms. A follower that hears no heartbeat "
     "becomes a candidate and requests votes; a candidate that receives "
     "votes from a majority becomes leader."),
    ("raft::c1",
     "The Raft leader replicates log entries to followers. An entry is "
     "committed once replicated on a majority of servers."),
]


@pytest.fixture()
def validator() -> CitationValidator:
    return CitationValidator()


def test_fabricated_claim_is_caught_and_rejected(validator):
    """DEFINITION OF DONE: the second sentence fabricates a specific,
    plausible-sounding claim (a '42-node hard limit') that appears
    nowhere in the retrieved chunks -- with a legitimate-looking
    citation attached. It must be rejected; the supported sentence
    must survive."""
    answer = (
        "A follower becomes a candidate and requests votes from other "
        "servers [1]. Raft clusters are hard-limited to a maximum of 42 "
        "nodes by the election protocol [2]."
    )
    result = validator.validate(answer, RETRIEVED)

    verdicts = {v.sentence[:20]: v.verdict for v in result.verdicts}
    assert result.rejected == 1
    assert result.kept == 1
    # the fabrication is gone from what the client would receive
    assert "42" not in result.validated_answer
    assert "hard-limited" not in result.validated_answer
    # the grounded sentence survived
    assert "requests votes" in result.validated_answer
    fabricated = [v for v in result.verdicts if "42" in v.sentence][0]
    assert fabricated.verdict == "unsupported"
    assert fabricated.coverage < 0.7
    assert verdicts[list(verdicts)[0]] == "supported"


def test_fully_supported_answer_passes_intact(validator):
    answer = ("A follower becomes a candidate and requests votes [1]. "
              "An entry is committed once replicated on a majority of "
              "servers [2].")
    result = validator.validate(answer, RETRIEVED)
    assert result.rejected == 0
    assert result.kept == 2
    assert result.validated_answer == answer
    assert result.citations == ["raft::c0", "raft::c1"]


def test_citation_of_nonexistent_source_is_rejected(validator):
    answer = "An entry is committed once replicated on a majority [7]."
    result = validator.validate(answer, RETRIEVED)
    assert result.all_rejected
    assert result.verdicts[0].verdict == "invalid_citation"


def test_supported_but_uncited_sentence_is_kept_and_flagged(validator):
    answer = "A candidate that receives votes from a majority becomes leader."
    result = validator.validate(answer, RETRIEVED)
    assert result.kept == 1
    assert result.verdicts[0].verdict == "supported_uncited"
    assert result.citations == []  # nothing explicitly cited


def test_sentence_graded_only_against_chunks_it_cites(validator):
    # This claim is supported by chunk 1 but cites only chunk 2:
    # per-citation validation must reject it (citing the wrong source is
    # a citation error even when the fact exists elsewhere).
    answer = "A follower that hears no heartbeat becomes a candidate [2]."
    result = validator.validate(answer, RETRIEVED)
    assert result.verdicts[0].verdict == "unsupported"


def test_all_fabricated_answer_rejects_everything(validator):
    answer = ("Raft was invented by aliens in 1847 [1]. Elections are "
              "decided by coin toss [2].")
    result = validator.validate(answer, RETRIEVED)
    assert result.all_rejected
    assert result.validated_answer == ""


def test_empty_retrieval_is_a_programming_error(validator):
    with pytest.raises(ValueError, match="no retrieved chunks"):
        validator.validate("anything", [])


def test_marker_only_fragments_are_ignored(validator):
    result = validator.validate("[1] [2]", RETRIEVED)
    assert result.kept == 0
    assert result.rejected == 0  # structural noise, not a claim
