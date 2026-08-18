"""Train a sentence-level grounding classifier on RAGBench.

Stage 10 measured the shipped lexical heuristic and found it weak outside
technical documentation (F1 0.383 overall). This trains a learned
replacement on the same kind of labels and evaluates it on the SAME
held-out test sample, so the comparison against the heuristic and the
three commercial judges is like for like.

Split discipline, which is the part worth getting right:

    train        ragbench_train_v1.jsonl        fits the weights
    validation   ragbench_validation_v1.jsonl   picks the threshold
    test         ragbench_sample_v1.jsonl       never seen until scoring

The test sample is the artifact Stage 10 already committed, drawn from
RAGBench's `test` splits only. Training and validation come from `train`
and `validation`, so no response scored at the end contributed to either
the weights or the threshold.

Model: L2-regularised logistic regression, fit by full-batch gradient
descent in numpy. Chosen over a boosted ensemble deliberately -- it adds
no dependency to a 512MB deployment, trains in under a second, runs in
microseconds, and its coefficients are readable, which is worth more here
than a fraction of a point of F1.

    python eval/train_grounding_model.py
    python eval/train_grounding_model.py --epochs 4000 --l2 0.5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.core.grounding_features import FEATURE_NAMES, extract_response  # noqa: E402

TRAIN_PATH = REPO_ROOT / "eval" / "ragbench_train_v1.jsonl"
VAL_PATH = REPO_ROOT / "eval" / "ragbench_validation_v1.jsonl"
MODEL_PATH = REPO_ROOT / "eval" / "grounding_model_v1.json"
FEATURES_DIR = REPO_ROOT / "eval"
SEED = 42


def load_examples(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Returns (X, y, domains) at SENTENCE level. y=1 means unsupported."""
    X: list[list[float]] = []
    y: list[int] = []
    domains: list[str] = []
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        pairs = row.get("response_sentences") or []
        sents = [p[1] for p in pairs
                 if isinstance(p, (list, tuple)) and len(p) >= 2]
        keys = [p[0] for p in pairs
                if isinstance(p, (list, tuple)) and len(p) >= 2]
        if not sents:
            continue
        context = "\n\n".join(row.get("documents") or [])
        unsupported = set(row.get("unsupported_response_sentence_keys") or [])
        for vec, key in zip(extract_response(sents, context), keys):
            X.append(vec)
            y.append(1 if key in unsupported else 0)
            domains.append(row.get("dataset_name") or "?")
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64), domains


def fit(X: np.ndarray, y: np.ndarray, epochs: int, lr: float,
        l2: float) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0
    Z = (X - mu) / sigma

    n, d = Z.shape
    w = rng.normal(0.0, 0.01, size=d)
    b = 0.0

    pos = float(y.sum())
    neg = float(n - pos)
    # Class weights: unsupported sentences are ~20% of the data, and the
    # asymmetry that matters is missing one, not over-flagging.
    w_pos = neg / pos if pos else 1.0
    sample_w = np.where(y == 1.0, w_pos, 1.0)
    sample_w = sample_w / sample_w.mean()

    for _ in range(epochs):
        p = 1.0 / (1.0 + np.exp(-(Z @ w + b)))
        err = (p - y) * sample_w
        grad_w = Z.T @ err / n + l2 * w / n
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b, mu, sigma


def predict_proba(X: np.ndarray, w: np.ndarray, b: float,
                  mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    Z = (X - mu) / sigma
    return 1.0 / (1.0 + np.exp(-(Z @ w + b)))


def prf(pred: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def pick_threshold(scores: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    best_t, best_f = 0.5, -1.0
    for t in np.unique(np.round(scores, 3)):
        _, _, f = prf((scores >= t).astype(float), y)
        if f > best_f:
            best_t, best_f = float(t), f
    return best_t, best_f


def auc(scores: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = y == 1
    n_pos, n_neg = float(pos.sum()), float((~pos).sum())
    if not n_pos or not n_neg:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--l2", type=float, default=50.0)
    args = ap.parse_args()

    for p in (TRAIN_PATH, VAL_PATH):
        if not p.exists():
            print(f"missing {p.name}. Fetch it first:\n"
                  f"  python eval/fetch_ragbench.py --split "
                  f"{'train' if 'train' in p.name else 'validation'} "
                  f"--per-config {'300' if 'train' in p.name else '120'}",
                  file=sys.stderr)
            return 1

    Xtr, ytr, _ = load_examples(TRAIN_PATH)
    Xva, yva, _ = load_examples(VAL_PATH)
    print(f"train sentences {len(ytr)} ({int(ytr.sum())} unsupported, "
          f"{ytr.mean():.1%})")
    print(f"val   sentences {len(yva)} ({int(yva.sum())} unsupported, "
          f"{yva.mean():.1%})")

    w, b, mu, sigma = fit(Xtr, ytr, args.epochs, args.lr, args.l2)

    s_tr = predict_proba(Xtr, w, b, mu, sigma)
    s_va = predict_proba(Xva, w, b, mu, sigma)
    threshold, val_f1 = pick_threshold(s_va, yva)
    p_va, r_va, _ = prf((s_va >= threshold).astype(float), yva)

    print(f"\ntrain AUC {auc(s_tr, ytr):.4f}   "
          f"val AUC {auc(s_va, yva):.4f}")
    print(f"threshold {threshold:.3f} chosen on VALIDATION -> "
          f"val F1 {val_f1:.4f} (P {p_va:.3f} R {r_va:.3f})")

    print("\ncoefficients (standardised; + pushes toward UNSUPPORTED)")
    for name, coef in sorted(zip(FEATURE_NAMES, w), key=lambda kv: -abs(kv[1])):
        print(f"  {name:<22} {coef:+.4f}")
    print("  NOTE: the overlap features correlate up to r=0.83, so individual"
          "\n  signs are not separately interpretable. Univariate class means"
          "\n  below are the honest reading of feature direction.")
    print("\nunivariate direction (train)")
    for i, name in enumerate(FEATURE_NAMES):
        m0 = Xtr[ytr == 0, i].mean()
        m1 = Xtr[ytr == 1, i].mean()
        print(f"  {name:<22} supported {m0:6.3f}   unsupported {m1:6.3f}")

    model = {
        "model": "logistic_regression",
        "version": "1.0",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "hyperparameters": {"epochs": args.epochs, "lr": args.lr,
                            "l2": args.l2},
        "feature_names": list(FEATURE_NAMES),
        "mean": mu.tolist(),
        "std": sigma.tolist(),
        "coefficients": w.tolist(),
        "intercept": float(b),
        "threshold": threshold,
        "training": {
            "train_file": TRAIN_PATH.name,
            "train_sha256": sha256(TRAIN_PATH),
            "train_sentences": int(len(ytr)),
            "validation_file": VAL_PATH.name,
            "validation_sha256": sha256(VAL_PATH),
            "validation_sentences": int(len(yva)),
            "val_auc": round(auc(s_va, yva), 4),
            "val_f1_at_threshold": round(val_f1, 4),
        },
        "note": ("Trained on RAGBench train/validation splits only. The "
                 "Stage 10 test sample was not used for weights or "
                 "threshold selection."),
    }
    MODEL_PATH.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    print(f"\nmodel written: {MODEL_PATH.relative_to(REPO_ROOT)} "
          f"({MODEL_PATH.stat().st_size} bytes)")

    np.savez_compressed(FEATURES_DIR / "grounding_features_v1.npz",
                        X_train=Xtr, y_train=ytr, X_val=Xva, y_val=yva)
    print("features cached: eval/grounding_features_v1.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
