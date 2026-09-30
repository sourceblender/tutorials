# bootstrap.py
"""Optional: how uncertain is the held-out gain? Paired bootstrap over the 300 rows.

Both models are scored on the SAME resampled rows each round (that is what
"paired" means), so the interval is for the difference, not two separate scores.
"""
import json
from pathlib import Path
import joblib
import numpy as np

held = [json.loads(l) for l in Path("evals/heldout.jsonl").read_text().splitlines()]
texts, labels = [r["text"] for r in held], np.array([r["label"] for r in held])
right = {n: joblib.load(f"{n}.joblib").predict(texts) == labels for n in ("incumbent", "candidate")}
diff = right["candidate"].astype(int) - right["incumbent"].astype(int)   # +1, 0 or -1 per row

rng = np.random.default_rng(0)
rows = rng.integers(0, len(diff), size=(20_000, len(diff)))              # 20,000 resamples of row indices
gains = diff[rows].mean(axis=1)
low, high = np.percentile(gains, [2.5, 97.5])
print(f"observed gain on these 300 rows: {diff.mean():+.3f}")
print(f"rows only the candidate got right: {(diff == 1).sum()}, only the incumbent: {(diff == -1).sum()}")
print(f"95% paired bootstrap interval: [{low:+.3f}, {high:+.3f}]")
