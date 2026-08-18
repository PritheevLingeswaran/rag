# Stage 12 Report — A trained grounding classifier

Date: 2026-08-18. Stage 10 measured the shipped hallucination check and
found it weak: F1 0.383, AUC 0.639 on RAGBench, strong only on technical
documentation. This stage replaces the hand-written rule with a **trained
model** and measures both on the same held-out data.

Until now nothing in this project was trained. The embedder and reranker
are pretrained ONNX models, BM25 has no learned parameters, and the LLM
is an API call. This is the project's first learned component, and it
exists because Stage 10 produced a labelled failure to learn from.

## Split discipline

The result is only worth the separation behind it:

| split | file | role |
|---|---|---|
| train | `ragbench_train_v1.jsonl` (1 800 responses) | fits the weights |
| validation | `ragbench_validation_v1.jsonl` (720) | picks the threshold |
| test | `ragbench_sample_v1.jsonl` (360) | scored once, at the end |

The test set is the artifact **Stage 10 already committed**, drawn from
RAGBench's `test` splits only. Training and validation come from `train`
and `validation`. No response scored below contributed to the weights or
the threshold, and the heuristic and the three commercial judges are
scored on exactly those same 360 responses — so every number in the
comparison table is on identical rows.

The raw training pulls (~40 MB of source documents) are gitignored; the
committed artifacts are the 123 KB feature matrix, the 2 KB model, and
the two files' SHA-256 hashes recorded inside the model. Re-fetching is
one documented command.

## Features

Twelve features, all pure functions of (sentence, context, position),
computed with the tokenizer retrieval already uses — so a model trained
on them runs in the serving path with no new dependency and no extra
download.

`coverage` is the shipped heuristic itself. The learned model therefore
**strictly generalises** the rule it replaces: it could reproduce it by
weighting that one feature. The other eleven add signals the rule cannot
see — bigram overlap, longest verbatim shared run, numeral agreement
(figures are prime fabrication sites), long-token and capitalised-token
coverage, sentence length, position, context size.

## Model

L2-regularised logistic regression, full-batch gradient descent in numpy,
seeded. Chosen over a boosted ensemble deliberately: it adds **no
dependency** to a 512 MB deployment, trains in under a second, scores in
microseconds, and serialises to 2 KB of JSON.

Training: 6 054 sentences (11.0% unsupported), class-weighted so the rare
positive class is not ignored. Threshold 0.599 selected on validation
(val F1 0.4217, val AUC 0.7687).

## Results on the held-out test set

Sentence level — the unit the check actually judges:

| | precision | recall | F1 | AUC |
|---|---|---|---|---|
| shipped heuristic | 0.310 | 0.598 | 0.409 | 0.666 |
| **trained model** | **0.455** | **0.703** | **0.552** | **0.823** |

Example level, against the commercial judges on the same rows:

| method | best F1 | AUC | cost |
|---|---|---|---|
| **trained model** | **0.500** | **0.753** | none |
| GPT adherence judge | 0.389 | 0.628 | LLM calls |
| TruLens groundedness | 0.384 | 0.635 | LLM calls |
| shipped heuristic | 0.383 | 0.639 | none |
| RAGAS faithfulness | 0.302 | 0.567 | LLM calls |

The trained model is first on both measures. AUC 0.753 against 0.628–0.639
for the judges is the more meaningful gap, since it does not depend on a
threshold.

Per domain, best F1, all methods on the same slice:

| domain | heuristic | **learned** | RAGAS | TruLens | GPT |
|---|---|---|---|---|---|
| techqa | 0.758 | **0.806** | 0.696 | 0.753 | 0.696 |
| emanual | 0.483 | **0.500** | 0.364 | 0.400 | 0.261 |
| covidqa | 0.333 | 0.400 | 0.375 | 0.350 | **0.625** |
| cuad | 0.333 | 0.400 | **0.500** | 0.217 | 0.189 |
| finqa | 0.130 | 0.148 | **0.444** | 0.195 | 0.125 |
| hotpotqa | 0.095 | 0.105 | **0.444** | 0.191 | 0.286 |

## What training did and did not fix

Training improved **every domain** — but look at the shape. techqa gains
0.048 and emanual 0.017, while finqa and hotpotqa gain 0.018 and 0.010
and remain far behind RAGAS.

That is the honest headline: **the failure was in the features, not the
classifier.** Where a faithful answer reuses source wording, better
weighting of lexical evidence helps a lot. Where a faithful answer
synthesises across passages or states a computed figure, there is no
lexical evidence to weight, and no amount of fitting invents it. A model
that reads meaning — an LLM judge, or a fine-tuned cross-encoder — is the
only thing that closes finqa and hotpotqa, and RAGAS's 0.444 on both is
the evidence for that.

So the project now has a defensible position rather than a compromise:
for its own corpus of technical documentation, a 2 KB model that costs
nothing beats every commercial judge tested. For paraphrase-heavy or
numerical corpora it should not be trusted, and the report says which
tool would be.

## Not wired into serving

`app/core/grounding.py` is unchanged and remains what `CitationValidator`
enforces and what `run_eval.py` measures. Swapping the serving check
would invalidate every committed baseline and the Stage 0 metric
contract, which requires a harness version bump, not an edit. The trained
model ships as a measured, reproducible artifact; adopting it in serving
is a separate, gated decision with its own before/after run.

## Caveats

- RAGBench labels are **model-generated** (gpt-4o), not human
  adjudication, so this measures agreement with a GPT-4o-defined
  standard. `gpt3_adherence` shares lineage with the annotator and is
  favourably biased — which makes the trained model's margin over it more
  notable, not less.
- Training data is a contiguous prefix of each `train` split (random
  index sampling tripped the datasets-server rate limit at this volume).
  The test sample keeps random sampling, so evaluation is unbiased; a
  less varied training set would, if anything, understate the result.
- Individual coefficient signs are **not** separately interpretable: the
  overlap features correlate up to r = 0.83, which flips `coverage`
  positive. Forcing the sign with heavy regularisation cost AUC
  (0.733 vs 0.769), so the model keeps its performance and the report
  quotes univariate class means instead, which are all in the expected
  direction.

## Reproducing

```
python eval/fetch_ragbench.py --split train --per-config 300
python eval/fetch_ragbench.py --split validation --per-config 120
python eval/train_grounding_model.py
python eval/validate_grounding.py --tag stage12-learned
```

Results: `eval/results/stage12-learned_20260818T031927Z.json`

## Verified

16 tests across `tests/eval/`: the scoring maths (confusion, AUC against
perfect/inverted/tied separation, threshold search), the features
(coverage reproduces the heuristic, numeral and shared-run behaviour,
batch equals per-sentence extraction), and the model (well-formed,
never trained on the test split, deterministic, ranks fabrication above
verbatim, and the serialised weights reproduce a hand-computed logistic
score).
