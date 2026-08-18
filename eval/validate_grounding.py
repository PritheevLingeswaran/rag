"""Validate this project's grounding metric against external labels.

The README has always carried an admission: the hallucination metric is a
deterministic LEXICAL proxy, "strict against verbatim fabrication, blind
to fluent paraphrase". That was an argued limitation, never a measured
one. This harness measures it.

The reference standard is RAGBench (galileo-ai/ragbench, CC-BY-4.0),
which ships support labels per response AND per sentence, plus the scores
three LLM-based judges gave the same rows.

WHAT THE REFERENCE LABELS ARE, stated up front because it bounds every
conclusion below: RAGBench's labels are MODEL-generated (the dataset
carries an `annotating_model_name`, gpt-4o on the rows where it is
populated). They are not human adjudication, and this harness makes no
claim that they are. Two consequences that must travel with the numbers:

  - "Agreement with RAGBench" means agreement with a GPT-4o-defined
    standard, not with ground truth in the strict sense.
  - `gpt3_adherence` shares model lineage with the annotator, so its
    showing here should be read as favourably biased, not as a fair
    upper bound.

What the comparison still supports: a RELATIVE ranking of cheap lexical
overlap against paid LLM judges, all scored against one consistent
external standard that none of them is this project's own.

So the same examples answer two questions:

    1. How accurate is the lexical proxy against human-checked labels?
    2. How does it compare with RAGAS / TruLens / a GPT judge -- which
       cost API calls per evaluation, while this costs nothing?

Two tasks are scored, because they are genuinely different:

    sentence-level  each response sentence: unsupported or not. This is
                    the unit app/core/grounding.py actually judges, and
                    the unit the citation validator drops text on.
    example-level   does the response contain >=1 unsupported sentence?
                    This is `hallucination_rate` in run_eval.py, and the
                    only level at which the third-party judges (which
                    emit one score per response) can be compared.

Threshold fairness: every method is given its BEST F1 over a sweep of its
own thresholds, so no method is handicapped by a badly chosen cut-off.
Threshold-free AUC is reported alongside, since best-F1 is optimistic for
all methods equally but AUC does not depend on a cut-off at all.

    python eval/validate_grounding.py
    python eval/validate_grounding.py --tag stage10

Determinism: pure function of the committed sample; no network, no LLM,
no randomness. Re-running reproduces the numbers bit for bit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app import __version__ as app_version  # noqa: E402
from app.core.grounding import (  # noqa: E402
    GROUNDING_THRESHOLD,
    content_tokens,
)

SAMPLE_PATH = REPO_ROOT / "eval" / "ragbench_sample_v1.jsonl"
RESULTS_DIR = REPO_ROOT / "eval" / "results"
MODEL_PATH = REPO_ROOT / "eval" / "grounding_model_v1.json"
VALIDATION_VERSION = "1.1"


def load_model() -> dict | None:
    if not MODEL_PATH.exists():
        return None
    return json.loads(MODEL_PATH.read_text(encoding="utf-8"))


def score_learned(model: dict, sentences: list[str],
                  context: str) -> list[float]:
    """P(unsupported) per sentence from the trained logistic model."""
    import math

    from app.core.grounding_features import extract_response

    mu = model["mean"]
    sd = model["std"]
    w = model["coefficients"]
    b = model["intercept"]
    out = []
    for vec in extract_response(sentences, context):
        z = sum(((v - m) / s) * c for v, m, s, c in zip(vec, mu, sd, w)) + b
        out.append(1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, z)))))
    return out

# Judges shipped with RAGBench. Each scores GROUNDEDNESS in [0, 1] where
# HIGHER means better supported, so the "unsupported" score is 1 - value.
BASELINES = {
    "ragas_faithfulness": "RAGAS faithfulness (LLM judge)",
    "trulens_groundedness": "TruLens groundedness (LLM judge)",
    "gpt3_adherence": "GPT adherence (LLM judge)",
}


# ---------------------------------------------------------------- scoring

def sentence_coverage_ratio(sentence: str, ctx_tokens: set[str]) -> float | None:
    """Fraction of the sentence's content tokens present in context, or
    None when the sentence carries no content tokens to check. This is
    the identical computation run_eval.py performs -- imported pieces,
    not a re-implementation, so measurement cannot drift from serving."""
    toks = content_tokens(sentence)
    if not toks:
        return None
    present = sum(1 for t in toks if t in ctx_tokens)
    return present / len(toks)


def confusion(pairs: list[tuple[bool, bool]]) -> dict:
    """pairs of (predicted_positive, actual_positive); positive = the
    UNSUPPORTED class, because that is the event worth catching."""
    tp = sum(1 for p, a in pairs if p and a)
    fp = sum(1 for p, a in pairs if p and not a)
    fn = sum(1 for p, a in pairs if not p and a)
    tn = sum(1 for p, a in pairs if not p and not a)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round((tp + tn) / len(pairs), 4) if pairs else 0.0,
    }


def auc(scored: list[tuple[float, bool]]) -> float | None:
    """ROC-AUC via the Mann-Whitney U identity, ties counted as half.
    Threshold-free: the probability a random unsupported item is scored
    above a random supported one."""
    pos = [s for s, a in scored if a]
    neg = [s for s, a in scored if not a]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return round(wins / (len(pos) * len(neg)), 4)


def best_f1(scored: list[tuple[float, bool]]) -> dict:
    """Best F1 over every threshold the data actually distinguishes."""
    if not scored:
        return {"f1": 0.0}
    best = None
    for cut in sorted({s for s, _ in scored} | {-1.0}):
        stats = confusion([(s > cut, a) for s, a in scored])
        if best is None or stats["f1"] > best["f1"]:
            best = {**stats, "threshold": round(cut, 4)}
    return best


# ---------------------------------------------------------------- harness

def evaluate(rows: list[dict], model: dict | None = None) -> dict:
    sentence_pairs: list[tuple[bool, bool]] = []
    sentence_scored: list[tuple[float, bool]] = []
    learned_sentence_pairs: list[tuple[bool, bool]] = []
    learned_sentence_scored: list[tuple[float, bool]] = []
    learned_example_pairs: list[tuple[bool, bool]] = []
    learned_example_scored: list[tuple[float, bool]] = []
    example_pairs: list[tuple[bool, bool]] = []
    example_scored: list[tuple[float, bool]] = []
    baseline_scored: dict[str, list[tuple[float, bool]]] = {
        k: [] for k in BASELINES
    }
    per_domain: dict[str, list[tuple[bool, bool]]] = {}
    # Per-domain scores for EVERY method: comparing ragp's per-domain
    # figure against a judge's overall figure would flatter whichever
    # method happens to be measured on the easier slice.
    per_domain_scored: dict[str, dict[str, list[tuple[float, bool]]]] = {}
    skipped_no_content = 0

    for row in rows:
        learned_row_score = None
        context = "\n\n".join(row.get("documents") or [])
        ctx_tokens = set(content_tokens(context))
        sentences = row.get("response_sentences") or []
        unsupported_keys = set(row.get("unsupported_response_sentence_keys") or [])

        if model is not None:
            texts = [it[1] for it in sentences
                     if isinstance(it, (list, tuple)) and len(it) >= 2]
            keys = [it[0] for it in sentences
                    if isinstance(it, (list, tuple)) and len(it) >= 2]
            if texts:
                probs = score_learned(model, texts, context)
                thr = model["threshold"]
                worst_learned = 0.0
                any_learned = False
                for key, prob in zip(keys, probs):
                    actual_s = key in unsupported_keys
                    learned_sentence_pairs.append((prob >= thr, actual_s))
                    learned_sentence_scored.append((prob, actual_s))
                    worst_learned = max(worst_learned, prob)
                    any_learned = any_learned or prob >= thr
                actual_e = row.get("adherence_score") is False
                learned_example_pairs.append((any_learned, actual_e))
                learned_example_scored.append((worst_learned, actual_e))
                learned_row_score = worst_learned

        worst = 1.0          # lowest coverage seen in this response
        any_predicted = False
        for item in sentences:
            # RAGBench stores each sentence as [key, text].
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            key, text = item[0], item[1]
            cov = sentence_coverage_ratio(text, ctx_tokens)
            if cov is None:
                skipped_no_content += 1
                continue
            predicted = cov < GROUNDING_THRESHOLD
            actual = key in unsupported_keys
            sentence_pairs.append((predicted, actual))
            sentence_scored.append((1.0 - cov, actual))
            any_predicted = any_predicted or predicted
            worst = min(worst, cov)

        # adherence_score True = fully supported, so the positive
        # (unsupported) class is `not adherence_score`.
        actual_ex = row.get("adherence_score") is False
        example_pairs.append((any_predicted, actual_ex))
        example_scored.append((1.0 - worst, actual_ex))
        domain = row.get("dataset_name") or "?"
        per_domain.setdefault(domain, []).append((any_predicted, actual_ex))
        slot = per_domain_scored.setdefault(domain, {})
        slot.setdefault("ragp_lexical_proxy", []).append((1.0 - worst, actual_ex))
        if learned_row_score is not None:
            slot.setdefault("ragp_learned_model", []).append(
                (learned_row_score, actual_ex))

        for name in BASELINES:
            val = row.get(name)
            if val is not None:
                baseline_scored[name].append((1.0 - float(val), actual_ex))
                slot.setdefault(name, []).append((1.0 - float(val), actual_ex))

    methods = {
        "ragp_lexical_proxy": {
            "description": (
                "app/core/grounding.py, threshold "
                f"{GROUNDING_THRESHOLD} -- deterministic, no API calls"
            ),
            "n": len(example_scored),
            "at_shipped_threshold": confusion(example_pairs),
            "best_f1": best_f1(example_scored),
            "auc": auc(example_scored),
        }
    }
    if model is not None and learned_example_scored:
        methods["ragp_learned_model"] = {
            "description": (
                "logistic regression on 12 lexical features, trained on "
                "RAGBench train/validation splits -- deterministic, no API calls"
            ),
            "n": len(learned_example_scored),
            "at_shipped_threshold": confusion(learned_example_pairs),
            "best_f1": best_f1(learned_example_scored),
            "auc": auc(learned_example_scored),
        }
    for name, label in BASELINES.items():
        scored = baseline_scored[name]
        methods[name] = {
            "description": label,
            "n": len(scored),
            "at_shipped_threshold": None,   # no shipped cut-off to honour
            "best_f1": best_f1(scored),
            "auc": auc(scored),
        }

    return {
        "sentence_level": {
            "n": len(sentence_pairs),
            "positives": sum(1 for _, a in sentence_pairs if a),
            "at_shipped_threshold": confusion(sentence_pairs),
            "best_f1": best_f1(sentence_scored),
            "auc": auc(sentence_scored),
            "skipped_sentences_without_content_tokens": skipped_no_content,
            "learned_model": ({
                "at_trained_threshold": confusion(learned_sentence_pairs),
                "best_f1": best_f1(learned_sentence_scored),
                "auc": auc(learned_sentence_scored),
            } if learned_sentence_scored else None),
        },
        "example_level": {
            "n": len(example_pairs),
            "positives": sum(1 for _, a in example_pairs if a),
            "methods": methods,
            "per_domain_at_shipped_threshold": {
                d: confusion(p) for d, p in sorted(per_domain.items())
            },
            "per_domain_best_f1_all_methods": {
                d: {
                    name: {**best_f1(scored), "n": len(scored),
                           "auc": auc(scored)}
                    for name, scored in sorted(methods_here.items())
                }
                for d, methods_here in sorted(per_domain_scored.items())
            },
        },
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def print_report(report: dict) -> None:
    s = report["sentence_level"]
    e = report["example_level"]
    line = "=" * 74
    print(line)
    print(f"GROUNDING VALIDATION v{VALIDATION_VERSION}  "
          f"vs RAGBench reference labels (model-generated)")
    print(f"sentences: {s['n']} ({s['positives']} unsupported) | "
          f"responses: {e['n']} ({e['positives']} not adherent)")
    print("-" * 74)

    lm = s.get("learned_model")
    if lm:
        lc = lm["at_trained_threshold"]
        print("SENTENCE LEVEL — learned model at its trained threshold")
        print(f"  precision {lc['precision']:.3f}  recall {lc['recall']:.3f}  "
              f"F1 {lc['f1']:.3f}  accuracy {lc['accuracy']:.3f}"
              f"   | AUC {lm['auc']}")
        print("-" * 74)

    c = s["at_shipped_threshold"]
    print(f"SENTENCE LEVEL — the unit the shipped metric judges "
          f"(threshold {GROUNDING_THRESHOLD})")
    print(f"  precision {c['precision']:.3f}  recall {c['recall']:.3f}  "
          f"F1 {c['f1']:.3f}  accuracy {c['accuracy']:.3f}")
    print(f"  tp {c['tp']}  fp {c['fp']}  fn {c['fn']}  tn {c['tn']}"
          f"   | AUC {s['auc']}")
    print("-" * 74)

    print("EXAMPLE LEVEL — comparable with judges that emit one score/response")
    print(f"  {'method':<26} {'best F1':>8} {'prec':>7} {'recall':>7} {'AUC':>7} {'n':>5}")
    for name, m in e["methods"].items():
        b = m["best_f1"]
        a = "  n/a" if m["auc"] is None else f"{m['auc']:.3f}"
        print(f"  {name:<26} {b['f1']:>8.3f} {b['precision']:>7.3f} "
              f"{b['recall']:>7.3f} {a:>7} {m['n']:>5}")
    shipped = e["methods"]["ragp_lexical_proxy"]["at_shipped_threshold"]
    print(f"\n  ragp at its SHIPPED threshold (not tuned): "
          f"F1 {shipped['f1']:.3f}  precision {shipped['precision']:.3f}  "
          f"recall {shipped['recall']:.3f}")
    print("-" * 74)

    print("PER DOMAIN — best F1, every method on the SAME slice")
    names = ["ragp_lexical_proxy", "ragp_learned_model", *BASELINES]
    labels = {"ragp_lexical_proxy": "heuristic", "ragp_learned_model": "learned",
              "ragas_faithfulness": "ragas", "trulens_groundedness": "trulens",
              "gpt3_adherence": "gpt"}
    print(f"  {'domain':<16}" + "".join(f"{labels[n]:>10}" for n in names))
    for d, methods_here in e["per_domain_best_f1_all_methods"].items():
        cells = "".join(
            f"{methods_here[n]['f1']:>10.3f}" if n in methods_here else f"{'—':>10}"
            for n in names
        )
        print(f"  {d:<16}{cells}")
    print(line)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="grounding-validation",
                    help="label for the results filename")
    args = ap.parse_args()

    if not SAMPLE_PATH.exists():
        print(f"missing {SAMPLE_PATH.name}; run eval/fetch_ragbench.py first",
              file=sys.stderr)
        return 1

    rows = [json.loads(line) for line in
            SAMPLE_PATH.open(encoding="utf-8") if line.strip()]
    model = load_model()
    report = evaluate(rows, model)
    report.update({
        "validation_version": VALIDATION_VERSION,
        "grounding_threshold": GROUNDING_THRESHOLD,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "app_version": app_version,
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "sample_file": SAMPLE_PATH.name,
        "sample_sha256": sha256(SAMPLE_PATH),
        "learned_model": ({"file": MODEL_PATH.name,
                           "sha256": sha256(MODEL_PATH),
                           "training": model.get("training")}
                          if model else None),
        "source": "galileo-ai/ragbench (CC-BY-4.0), test splits only",
        "reference_label_provenance": (
            "model-generated (RAGBench annotating_model_name, gpt-4o where "
            "populated); NOT human adjudication. gpt3_adherence shares "
            "lineage with the annotator and is favourably biased."
        ),
    })
    print_report(report)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS_DIR / f"{args.tag}_{stamp}.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"results written: {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
