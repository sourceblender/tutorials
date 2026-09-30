# freeze.py
"""Freeze the eval: copy the held-out rows out of the dataset and hash every
file the gate reads. Run once, before training anything. run_eval.py refuses
to score if any frozen file changed afterwards.
"""
import hashlib, json, random
from pathlib import Path

REVISION = "8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb"  # exact stanfordnlp/sst2 commit
SEED, N_HELDOUT = 1234, 300
FROZEN = ("evals/cases.jsonl", "evals/heldout.jsonl", "evals/gate.toml")


def frozen_digest(names=FROZEN):
    """SHA-256 over each file's name, byte length and bytes. Framing each file
    means bytes cannot move from one file into the next without changing it."""
    h = hashlib.sha256()
    for name in names:
        data = Path(name).read_bytes()
        h.update(name.encode() + b"\0" + len(data).to_bytes(8, "big") + data)
    return h.hexdigest()


def main():
    from datasets import load_dataset
    val = load_dataset("stanfordnlp/sst2", split="validation", revision=REVISION)
    idx = sorted(random.Random(SEED).sample(range(len(val)), N_HELDOUT))
    rows = [json.dumps({"text": r["sentence"], "label": r["label"]}) + "\n" for r in val.select(idx)]
    Path("evals/heldout.jsonl").write_bytes("".join(rows).encode("utf-8"))  # the rows themselves, LF on every OS

    digest = frozen_digest()
    Path("evals/FROZEN.sha256").write_text(digest + "\n")
    rows_hash = hashlib.sha256(Path("evals/heldout.jsonl").read_bytes()).hexdigest()
    print(f"held-out rows {rows_hash[:16]} ({N_HELDOUT} of {len(val)} validation rows)")
    print(f"frozen        {digest[:16]} (cases + held-out rows + gate)")


if __name__ == "__main__":
    main()
