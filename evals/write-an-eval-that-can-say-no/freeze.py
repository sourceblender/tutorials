# freeze.py
"""Freeze the eval: copy the held-out rows out of the dataset and hash every
file the gate reads. Run once, before training anything. run_eval.py refuses
to score if any frozen file changed afterwards.
"""
import hashlib, json, random
from pathlib import Path
from datasets import load_dataset

REVISION = "8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb"  # exact stanfordnlp/sst2 commit
SEED, N_HELDOUT = 1234, 300

val = load_dataset("stanfordnlp/sst2", split="validation", revision=REVISION)
idx = sorted(random.Random(SEED).sample(range(len(val)), N_HELDOUT))
with open("evals/heldout.jsonl", "w") as f:           # the rows themselves, not pointers
    for row in val.select(idx):
        f.write(json.dumps({"text": row["sentence"], "label": row["label"]}) + "\n")

h = hashlib.sha256()
for name in ("evals/cases.jsonl", "evals/heldout.jsonl", "evals/gate.toml"):
    h.update(Path(name).read_bytes())
Path("evals/FROZEN.sha256").write_text(h.hexdigest() + "\n")
print("frozen", h.hexdigest()[:16], f"({N_HELDOUT} held-out rows, 20 authored cases)")
