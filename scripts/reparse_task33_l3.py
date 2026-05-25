"""Re-parse Task 33 L3 raw_responses with a better regex."""
import json
import re
from collections import Counter
from pathlib import Path

src = Path("results/task33_l3_classifications.json")
with open(src) as f:
    data = json.load(f)

# Improved parser: look for bolded or standalone category codes anywhere in response
CODE_RE = re.compile(r"(?:\*\*|\b)(T|I1|I2|I3|R)(?:\*\*|\b)")

def reparse(raw: str) -> str:
    if not raw:
        return "R"
    # Prefer bolded codes like **T**
    bolded = re.findall(r"\*\*(T|I1|I2|I3|R)\*\*", raw)
    if bolded:
        # If multiple bolded codes, pick the LAST one (usually the final verdict)
        return bolded[-1]
    # Otherwise find first standalone code token (not inside a word)
    # Match at line start, after punctuation, or surrounded by spaces
    standalone = re.findall(r"(?:^|[\s\n.:,*-])(T|I1|I2|I3|R)(?=[\s\n.:,*-]|$)", raw)
    if standalone:
        return standalone[0]
    return "R"

new_counts = Counter()
changed = 0
for item in data["classifications"]:
    old_cat = item["category"]
    new_cat = reparse(item["raw_response"])
    item["category_v2"] = new_cat
    if new_cat != old_cat:
        changed += 1
    new_counts[new_cat] += 1

n = len(data["classifications"])
print(f"Re-parsed {n} classifications")
print(f"Changed: {changed} ({100*changed/n:.1f}%)")
print("\n=== Original distribution ===")
for k, v in data["summary"]["counts"].items():
    print(f"  {k:>5}  {v:4d}  ({100*v/n:5.1f}%)")

print("\n=== Re-parsed distribution ===")
for k in ("T", "I1", "I2", "I3", "R"):
    v = new_counts.get(k, 0)
    print(f"  {k:>5}  {v:4d}  ({100*v/n:5.1f}%)")

t = new_counts.get("T", 0)
intr = sum(new_counts.get(k, 0) for k in ("I1", "I2", "I3"))
res = new_counts.get("R", 0)
print(f"\nTemporal (T):        {100*t/n:.1f}%")
print(f"Intrinsic (I1+I2+I3): {100*intr/n:.1f}%")
print(f"Residual (R):        {100*res/n:.1f}%")

data["summary"]["counts_v2"] = dict(new_counts)
data["summary"]["pct_v2"] = {k: v/n for k, v in new_counts.items()}

out = Path("results/task33_l3_classifications_reparsed.json")
with open(out, "w") as f:
    json.dump(data, f, indent=2)
print(f"\nSaved to {out}")
