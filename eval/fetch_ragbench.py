"""Build a versioned RAGBench sample for grounding-metric validation.

RAGBench (galileo-ai/ragbench, CC-BY-4.0) ships support labels:
`adherence_score` (is the whole response supported by its documents?) and
`unsupported_response_sentence_keys` (which sentences are not). That is an
EXTERNAL reference standard for exactly the judgement
app/core/grounding.py makes with a lexical proxy.

The labels are model-generated, not human adjudication -- see the
provenance note in eval/validate_grounding.py, which bounds what the
comparison can claim.

This script pulls a FIXED, SEEDED sample through the HF datasets-server
REST API (no new runtime dependency) and writes one JSONL carrying only
the fields the validation needs. The output is a versioned artifact in
the same sense as data/corpus_v1.jsonl: its SHA-256 is recorded in every
results file, and re-running this script must reproduce it byte for byte.
Changing the configs, the per-config count, or the seed produces a NEW
sample version -- it is not an edit in place.

    python eval/fetch_ragbench.py                 # writes sample_v1.jsonl
    python eval/fetch_ragbench.py --check         # verify committed hash

Rows are taken from each config's `test` split only, so nothing here is
drawn from data any model in the comparison was tuned on.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "eval" / "ragbench_sample_v1.jsonl"

DATASET = "galileo-ai/ragbench"
SPLIT = "test"
# Six domains, deliberately diverse: biomedical, multi-hop web, finance,
# technical support, consumer manuals, legal contracts. A proxy that only
# works on one register would show up as variance across these.
CONFIGS = ["covidqa", "hotpotqa", "finqa", "techqa", "emanual", "cuad"]
PER_CONFIG = 60
RANDOM_SEED = 42
PAGE = 100                      # datasets-server per-request row cap

KEEP = (
    "id", "dataset_name", "question", "documents", "response",
    "response_sentences", "unsupported_response_sentence_keys",
    "adherence_score", "trulens_groundedness", "ragas_faithfulness",
    "gpt3_adherence",
)


def _get(url: str, attempts: int = 6) -> dict:
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            if i == attempts - 1:
                raise
            throttled = getattr(exc, "code", None) == 429
            delay = (20 * (i + 1)) if throttled else (2 * (i + 1))
            print(f"  retry {i + 1} in {delay}s after {exc}", file=sys.stderr)
            time.sleep(delay)
    raise AssertionError("unreachable")


def _rows_url(config: str, offset: int, length: int,
              split: str = SPLIT) -> str:
    q = urllib.parse.urlencode({
        "dataset": DATASET, "config": config, "split": split,
        "offset": offset, "length": length,
    })
    return f"https://datasets-server.huggingface.co/rows?{q}"


def fetch_config(config: str, split: str = SPLIT,
                 per_config: int = PER_CONFIG) -> list[dict]:
    """Deterministic sample: read the split's size, pick per_config row
    indices with a seeded RNG, then fetch only the pages those fall in."""
    head = _get(_rows_url(config, 0, 1, split))
    total = head.get("num_rows_total", 0)
    if not total:
        raise RuntimeError(f"{config}: datasets-server reported no rows")

    rng = random.Random(f"{RANDOM_SEED}:{config}"
                        if split == SPLIT else f"{RANDOM_SEED}:{config}:{split}")
    wanted = sorted(rng.sample(range(total), min(per_config, total)))

    out: list[dict] = []
    page_start = None
    page_rows: list[dict] = []
    for idx in wanted:
        start = (idx // PAGE) * PAGE
        if start != page_start:
            page_rows = _get(_rows_url(config, start, PAGE, split))["rows"]
            page_start = start
        row = page_rows[idx - start]["row"]
        out.append({k: row.get(k) for k in KEEP})
    print(f"  {config:9s} {len(out):3d} of {total} rows")
    return out


def fetch_config_sequential(config: str, split: str,
                            per_config: int) -> list[dict]:
    """Contiguous prefix of a split, one page request per 100 rows.

    Used for TRAINING pulls only. Random index sampling costs one request
    per distinct page and trips the datasets-server rate limit at this
    volume. The held-out test sample keeps random sampling, so evaluation
    is unbiased; a contiguous training prefix only risks a less varied
    training set, which the per-domain breakdown would expose.
    """
    head = _get(_rows_url(config, 0, 1, split))
    total = head.get("num_rows_total", 0)
    if not total:
        raise RuntimeError(f"{config}/{split}: no rows")
    want = min(per_config, total)
    out: list[dict] = []
    offset = 0
    while len(out) < want:
        rows = _get(_rows_url(config, offset, min(PAGE, want - len(out)), split))
        batch = rows.get("rows", [])
        if not batch:
            break
        out.extend({k: r["row"].get(k) for k in KEEP} for r in batch)
        offset += len(batch)
        time.sleep(1.5)
    print(f"  {config:9s} {len(out):4d} of {total} rows ({split})")
    return out


def write_sample(split: str = SPLIT, per_config: int = PER_CONFIG,
                 out_path: Path | None = None) -> Path:
    out_path = out_path or OUT_PATH
    records: list[dict] = []
    for config in CONFIGS:
        if split == SPLIT:
            records.extend(fetch_config(config, split, per_config))
        else:
            records.extend(fetch_config_sequential(config, split, per_config))
    # Stable order and stable key order => byte-reproducible file.
    records.sort(key=lambda r: (r["dataset_name"] or "", str(r["id"])))
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    return out_path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="re-fetch and confirm the committed file is reproduced")
    ap.add_argument("--split", default=SPLIT,
                    help="train / validation / test. The committed eval sample "
                         "is test-only; training data MUST come from another "
                         "split or the Stage 10 comparison is contaminated.")
    ap.add_argument("--per-config", type=int, default=PER_CONFIG)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.split != SPLIT or args.out is not None:
        out = args.out or (REPO_ROOT / "eval" /
                           f"ragbench_{args.split}_v1.jsonl")
        write_sample(args.split, args.per_config, out)
        n = sum(1 for _ in out.open(encoding="utf-8"))
        print(f"\nwrote {out.name}  split={args.split}  rows={n}  "
              f"bytes={out.stat().st_size}\nsha256={sha256(out)}")
        return 0

    if args.check:
        if not OUT_PATH.exists():
            print("no committed sample to check", file=sys.stderr)
            return 1
        before = sha256(OUT_PATH)
        write_sample()
        after = sha256(OUT_PATH)
        print(f"committed: {before}\nrefetched: {after}")
        if before != after:
            print("MISMATCH: the upstream split changed under the same seed",
                  file=sys.stderr)
            return 2
        print("reproducible")
        return 0

    path = write_sample()
    n = sum(1 for _ in path.open(encoding="utf-8"))
    print(f"\nwrote {path.relative_to(REPO_ROOT)}  rows={n}  "
          f"bytes={path.stat().st_size}\nsha256={sha256(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
