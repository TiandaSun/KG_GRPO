"""Task 35: Cross-benchmark comparison between CWQ (Task 14) and KGQAGen-10k (Task 35).

Reads:
  - results/task14_full_*.json   (CWQ)
  - results/task14_full_*chunk*.json (CWQ chunks; merged)
  - results/task35_kgqagen_*.json (KGQAGen-10k)

Computes:
  1. Side-by-side EM/ContEM/F1/Tools per model on both benchmarks
  2. Rank correlation (Spearman) between CWQ and KGQAGen rankings
  3. Per-model delta (CWQ - KGQAGen) — positive means CWQ easier
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

# Match task14_full_{model}.json or task14_full_{model}_chunk{n}.json
T14_PATTERN = re.compile(r"task14_full_(.+?)(?:_chunk\d+)?\.json$")
T35_PATTERN = re.compile(r"task35_kgqagen_(.+)\.json$")


def load_task14() -> dict[str, dict]:
    """Load CWQ results, merging chunked outputs.

    Each file is keyed by step. We pick the first/only step.
    For chunked results, we weighted-average across chunks (by n_samples).
    """
    results: dict[str, list[dict]] = {}
    for f in sorted(Path("results").glob("task14_full_*.json")):
        m = T14_PATTERN.search(f.name)
        if not m:
            continue
        model_key = m.group(1)
        # Strip llama_ prefix to keep grouping clean
        if model_key.startswith("llama_"):
            # keep as-is, llama models are separate
            pass
        try:
            data = json.load(open(f))
        except Exception:
            continue
        for step_key, step_data in data.items():
            results.setdefault(model_key, []).append(step_data)

    # Merge per model: weighted by n_samples
    merged: dict[str, dict] = {}
    for model_key, chunks in results.items():
        total_n = sum(c["n_samples"] for c in chunks)
        if total_n == 0:
            continue
        merged[model_key] = {
            "em": sum(c["em"] * c["n_samples"] for c in chunks) / total_n,
            "contains_em": sum(c["contains_em"] * c["n_samples"] for c in chunks) / total_n,
            "f1": sum(c["f1"] * c["n_samples"] for c in chunks) / total_n,
            "avg_tool_calls": sum(c["avg_tool_calls"] * c["n_samples"] for c in chunks) / total_n,
            "n_samples": total_n,
        }
    return merged


def load_task35() -> dict[str, dict]:
    """Load KGQAGen-10k results."""
    merged: dict[str, dict] = {}
    for f in sorted(Path("results").glob("task35_kgqagen_*.json")):
        m = T35_PATTERN.search(f.name)
        if not m:
            continue
        model_key = m.group(1)
        try:
            data = json.load(open(f))
        except Exception:
            continue
        for step_key, step_data in data.items():
            merged[model_key] = {
                "em": step_data["em"],
                "contains_em": step_data["contains_em"],
                "f1": step_data["f1"],
                "avg_tool_calls": step_data["avg_tool_calls"],
                "n_samples": step_data["n_samples"],
            }
    return merged


def spearman(a: list[float], b: list[float]) -> float:
    """Compute Spearman rank correlation."""
    if len(a) < 2:
        return 0.0
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    if denom == 0:
        return 0.0
    return float((ra * rb).sum() / denom)


def main() -> None:
    cwq = load_task14()
    kgqa = load_task35()

    common = sorted(set(cwq) & set(kgqa))
    print(f"\n=== Cross-benchmark: CWQ (Task 14) vs KGQAGen-10k (Task 35) ===\n")
    print(f"Models with results on both: {len(common)} of {len(cwq)} CWQ / {len(kgqa)} KGQAGen")
    print()

    print(f"{'Model':<20} | {'CWQ EM':>8} {'CWQ F1':>8} {'CWQ T':>6} {'CWQ n':>6}"
          f" | {'KG EM':>8} {'KG F1':>8} {'KG T':>6} {'KG n':>6}"
          f" | {'EM Δ':>8}")
    print("-" * 110)
    rows = []
    for k in common:
        c = cwq[k]
        g = kgqa[k]
        delta = c["em"] - g["em"]
        rows.append({
            "model": k,
            "cwq": c,
            "kgqa": g,
            "em_delta": delta,
        })
        print(f"{k:<20} | {c['em']:>8.3f} {c['f1']:>8.3f} {c['avg_tool_calls']:>6.1f} {c['n_samples']:>6d}"
              f" | {g['em']:>8.3f} {g['f1']:>8.3f} {g['avg_tool_calls']:>6.1f} {g['n_samples']:>6d}"
              f" | {delta:>+8.3f}")

    # Spearman correlation between EM rankings
    cwq_em = [r["cwq"]["em"] for r in rows]
    kg_em = [r["kgqa"]["em"] for r in rows]
    cwq_f1 = [r["cwq"]["f1"] for r in rows]
    kg_f1 = [r["kgqa"]["f1"] for r in rows]
    rho_em = spearman(cwq_em, kg_em)
    rho_f1 = spearman(cwq_f1, kg_f1)

    print()
    print(f"Spearman ρ (EM rankings):  {rho_em:+.3f}")
    print(f"Spearman ρ (F1 rankings):  {rho_f1:+.3f}")

    # Identify models where the picture differs
    print("\n=== Per-model takeaway ===")
    for r in sorted(rows, key=lambda x: -x["cwq"]["em"]):
        m = r["model"]
        c = r["cwq"]
        g = r["kgqa"]
        if c["em"] >= 0.20 and g["em"] >= 0.15:
            verdict = "STRONG on both"
        elif c["em"] < 0.05 and g["em"] < 0.05:
            verdict = "COLLAPSED on both"
        elif c["em"] < 0.05 or g["em"] < 0.05:
            verdict = "SPLIT (works on one)"
        else:
            verdict = "moderate"
        print(f"  {m:<20} CWQ={c['em']:.3f}  KGQAGen={g['em']:.3f}  Δ={r['em_delta']:+.3f}  → {verdict}")

    out = {
        "n_common": len(common),
        "spearman_em": rho_em,
        "spearman_f1": rho_f1,
        "rows": rows,
    }
    out_file = Path("results/task35_cross_benchmark.json")
    with open(out_file, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
