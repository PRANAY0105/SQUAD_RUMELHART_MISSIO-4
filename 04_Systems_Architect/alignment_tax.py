"""Alignment tax audit: compare capability benchmarks before and after alignment.

Alignment tax = capability lost to gain alignment. For each benchmark:
    delta     = after - before            (points)
    rel_tax % = (before - after) / before * 100   (positive = capability lost)
Input is a JSON/dict you fill with REAL benchmark scores (MMLU, GSM8K, HumanEval, plus an alignment metric).
No benchmark is run by this file and no score is invented; `example_scores.json` is a labelled template.
"""
import json
import sys
from typing import Dict

CAPABILITY = ("MMLU", "GSM8K", "HumanEval")


def audit(before: Dict[str, float], after: Dict[str, float], tolerance_pct: float = 2.0) -> Dict:
    rows = []
    for name in before:
        if name not in after:
            continue
        b, a = before[name], after[name]
        rel = (b - a) / b * 100 if b else 0.0
        rows.append({"benchmark": name, "before": b, "after": a, "delta": round(a - b, 4),
                     "relative_tax_pct": round(rel, 3),
                     "capability": name in CAPABILITY,
                     "verdict": ("ALIGNMENT GAIN" if a > b else "WITHIN TOLERANCE" if rel <= tolerance_pct else "TAX ABOVE TOLERANCE")})
    cap = [r["relative_tax_pct"] for r in rows if r["capability"]]
    return {"tolerance_pct": tolerance_pct, "rows": rows,
            "mean_capability_tax_pct": round(sum(cap) / len(cap), 3) if cap else None}


def to_markdown(result: Dict) -> str:
    out = ["| Benchmark | Before | After | Delta | Relative tax % | Verdict |", "|---|---|---|---|---|---|"]
    for r in result["rows"]:
        out.append(f"| {r['benchmark']} | {r['before']} | {r['after']} | {r['delta']} | {r['relative_tax_pct']} | {r['verdict']} |")
    out.append(f"\nMean capability tax: {result['mean_capability_tax_pct']}% (tolerance {result['tolerance_pct']}%).")
    return "\n".join(out)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "example_scores.json"
    data = json.load(open(path))
    print(to_markdown(audit(data["before"], data["after"])))
