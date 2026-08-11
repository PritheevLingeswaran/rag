# Stage 10 Report — Validating the grounding metric against RAGBench

Date: 2026-08-11. Since Stage 0 the README has carried an admission:

> The hallucination metric is a **deterministic lexical proxy**, not an
> LLM judge. It cannot detect fluent-but-wrong paraphrase, but it is
> reproducible, free, and strict against verbatim fabrication.

That was an argued limitation. This stage measures it against an external
reference standard, and the argument turns out to be right in a more
specific and more useful way than it was stated.

## Setup

**Reference standard.** RAGBench (`galileo-ai/ragbench`, CC-BY-4.0)
ships, per response, an `adherence_score` (is every sentence supported by
the provided documents?) and `unsupported_response_sentence_keys` (which
ones are not) — plus the scores three LLM-based judges gave the *same*
rows: RAGAS faithfulness, TruLens groundedness, and a GPT adherence
judge. One consistent standard, four methods, none of them ours.

**What the labels are — stated up front, because it bounds everything
below.** RAGBench's labels are **model-generated** (the dataset carries
`annotating_model_name`, `gpt-4o` where populated), not human
adjudication. So "agreement with RAGBench" means agreement with a
GPT-4o-defined standard. In particular `gpt3_adherence` shares model
lineage with the annotator and should be read as **favourably biased**,
not as a fair ceiling. What survives that caveat is the *relative*
ranking of the four methods against one external yardstick.

**Sample.** `eval/ragbench_sample_v1.jsonl` — 360 rows, 60 from each of
six deliberately dissimilar domains (biomedical, multi-hop web, finance,
technical support, consumer manuals, legal contracts), drawn only from
`test` splits with a seeded RNG. It is a versioned artifact in the same
sense as `data/corpus_v1.jsonl`: SHA-256 recorded in every results file,
`eval/fetch_ragbench.py --check` re-fetches and confirms reproducibility.
Class balance: 63/360 responses (17.5%) and 363/1851 sentences (19.6%)
are labelled unsupported.

**Fairness.** Every method is given its **best F1 over a sweep of its own
thresholds**, so none is handicapped by a badly chosen cut-off, and
threshold-free **AUC** is reported alongside. Per-domain figures are
computed for *all four* methods on the *same* slice — comparing our
per-domain number against a judge's overall number would have flattered
whichever method drew the easier rows.

The proxy is not re-implemented here: `eval/validate_grounding.py`
imports `content_tokens` and `GROUNDING_THRESHOLD` from
`app/core/grounding.py`, the module the citation validator enforces with,
so measurement cannot drift from serving.

## Result 1 — overall, the free metric is not beaten

Example level (the only level where per-response judges are comparable):

| method | best F1 | precision | recall | AUC | n |
|---|---|---|---|---|---|
| **ragp lexical proxy** | 0.383 | 0.274 | 0.635 | **0.639** | 360 |
| RAGAS faithfulness | 0.302 | 0.178 | 1.000 | 0.567 | 343 |
| TruLens groundedness | 0.384 | 0.262 | 0.717 | 0.635 | 348 |
| GPT adherence judge | 0.389 | 0.423 | 0.361 | 0.628 | 348 |

All four cluster in F1 0.30–0.39 and AUC 0.57–0.64. **A deterministic
bag-of-words overlap check, costing nothing and calling no API, lands
inside the same band as three LLM judges** — one of which shares lineage
with the annotator that produced the labels.

Read honestly, this says as much about the task as about the method:
adherence detection on this benchmark is hard for everyone. It does not
say the proxy is good in absolute terms. It does say the project's
free-tier constraint cost it nothing measurable here.

At its **shipped, untuned** threshold (0.7) the proxy scores F1 0.352 at
precision 0.230 / recall 0.746 — deliberately a high-recall screen, see
"operating point" below.

## Result 2 — where it works is exactly this project's register

Best F1 per domain, all methods on the same rows:

| domain | ragp | RAGAS | TruLens | GPT |
|---|---|---|---|---|
| techqa (technical support) | **0.758** | 0.696 | 0.753 | 0.696 |
| emanual (product manuals) | **0.483** | 0.364 | 0.400 | 0.261 |
| covidqa (biomedical) | 0.333 | 0.375 | 0.350 | **0.625** |
| cuad (legal contracts) | 0.333 | **0.500** | 0.217 | 0.189 |
| finqa (finance) | 0.130 | **0.444** | 0.195 | 0.125 |
| hotpotqa (multi-hop web) | 0.095 | **0.444** | 0.191 | 0.286 |

The spread is the finding. The proxy **beats every LLM judge on
technical documentation and product manuals**, and **collapses on
multi-hop web QA and finance**.

The mechanism is the one the README guessed, now with a shape: lexical
overlap detects fabrication when a faithful answer would *reuse the
source's vocabulary*. Technical support text answers extractively — the
words in the answer are the words in the manual. Multi-hop and financial
answers are *synthesised*: they compose facts across passages and state
computed figures, so a perfectly faithful sentence shares few tokens with
any single source, and the proxy flags it. `finqa` 0.130 and `hotpotqa`
0.095 are that failure, measured.

**Why this matters for ragp specifically:** its corpus is 30
systems-design documents — technical documentation, the `techqa`/`emanual`
register, where the proxy scores highest and outperforms the paid judges.
The cheap metric is *fit for the corpus it ships with*. It would be the
wrong metric for a paraphrase-heavy or numerical corpus, and that is now
a measured statement rather than a hope.

## Operating point: why low precision is the right trade here

Sentence level, at the shipped threshold: precision 0.310, recall 0.598,
AUC 0.666 (tp 217, fp 482, fn 146, tn 1006).

Precision that low would be alarming for a *reporting* metric. For the
role this check actually plays in serving it is the correct asymmetry:
`CitationValidator` **drops** sentences it cannot ground. A false
positive costs a true sentence (the answer gets shorter); a false
negative ships a fabrication to a user. The shipped threshold buys recall
0.746 at example level with precision 0.230 — it over-rejects, on
purpose, and `degraded_citation_rejected` exists precisely for when it
over-rejects everything.

The honest cost, now quantified: roughly three in four dropped sentences
would have been acceptable. That is a real quality tax, and it is the
strongest argument yet for adding an LLM-judge pass *alongside* the
lexical check — which is exactly the upgrade path the README has always
named, now with a number attached to the motivation.

## What this does NOT establish

- **Not human ground truth.** Model-generated labels; `gpt3_adherence` is
  favourably biased by shared lineage.
- **Not ragp's own distribution.** RAGBench responses come from other
  generators over other corpora. In production this check runs on *our*
  answers against *our* retrieved context, which is a different (and
  narrower) distribution than any row here.
- **Not a retrieval result.** Nothing in this stage touches retrieval,
  the serving corpus, or `eval/run_eval.py`'s baseline. The Stage 0
  metric contract is untouched; this is a new, separate harness.

## Reproducing

```
python eval/fetch_ragbench.py --check       # sample is byte-reproducible
python eval/validate_grounding.py --tag stage10-grounding
```

Pure function of the committed sample: no network, no LLM, no randomness.
Results: `eval/results/stage10-grounding_20260811T170727Z.json`.

## Verified

- 7 exact-value tests on the scoring maths (`tests/eval/`): confusion
  counts, AUC against perfect/inverted/tied separation, best-F1 threshold
  search, the coverage ratio matching the shipped definition, and an
  end-to-end pass over a hand-built two-row sample.
- The retrieval eval harness and its committed baseline are unchanged.
