"""Monday, last step: look at what the tokenizer produces, check it, and freeze the day's work.

1. Shows how a few sentences are cut into pieces and ids.
2. Round-trips the first 100 validation stories: decode(encode(text)) must give back the exact text.
3. Counts tokens in both splits and the average tokens per word (whitespace-separated) on validation.
4. Writes data/FROZEN.txt: the hashes and counts every later day checks before it starts.
Checks its inputs first: train.txt, val.txt and tok4096.json must match the pinned hashes from day1_data.py and
day1_tokenizer.py, and validation must hold at least 100 stories. FROZEN.txt is written only after every check
passes, and atomically, so a failed run never replaces a good one. Exits 1 on any failure.
"""
import hashlib, json, os, sys

import morpheme

import day1_data
import day1_tokenizer

SEP = "<|endoftext|>"
DATA = "data"


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    pinned = {os.path.join(DATA, "stories", "train.txt"): day1_data.EXPECT["train.txt"],
              os.path.join(DATA, "stories", "val.txt"): day1_data.EXPECT["val.txt"],
              os.path.join(DATA, "tok4096.json"): day1_tokenizer.EXPECT}
    for path, want in pinned.items():
        if not os.path.exists(path):
            sys.exit(f"{path} is missing; run day1_data.py and day1_tokenizer.py first")
        if sha(path) != want:
            sys.exit(f"{path} does not match the pinned hash; rerun day1_data.py / day1_tokenizer.py")
    tok = morpheme.Tokenizer.from_file(os.path.join(DATA, "tok4096.json"))
    for sentence in ["Once upon a time, there was a little dog named Max.",
                     "The dragon's spaghetti was unbelievably wobbly!",
                     "Lily and Ben went to the park."]:
        enc = tok.encode(sentence)
        print(f"\n{sentence}\n  {len(enc.ids)} tokens: {enc.tokens}\n  ids: {enc.ids}")

    val_text = open(os.path.join(DATA, "stories", "val.txt"), encoding="utf-8").read()
    stories = [s.strip() for s in val_text.split(SEP) if s.strip()]
    if len(stories) < 100:
        sys.exit(f"validation has {len(stories)} stories; at least 100 are needed for the round-trip check")
    checked = stories[:100]
    failed = [i for i, s in enumerate(checked) if tok.decode(tok.encode(s).ids) != s]
    print(f"\nround-trip: {len(checked) - len(failed)}/{len(checked)} validation stories came back exactly")
    if failed:
        sys.exit(f"round-trip failed for validation stories {failed[:5]}; FROZEN.txt not written")

    counts = {}
    for name in ("train", "val"):
        text = open(os.path.join(DATA, "stories", f"{name}.txt"), encoding="utf-8").read()
        counts[name] = len(tok.encode(text).ids)
    words = len(val_text.split())
    per_word = counts["val"] / words
    print(f"tokens: train {counts['train']:,}  val {counts['val']:,}   ({per_word:.2f} tokens per word on validation)")

    frozen = {
        "train.txt": sha(os.path.join(DATA, "stories", "train.txt")),
        "val.txt": sha(os.path.join(DATA, "stories", "val.txt")),
        "tok4096.json": sha(os.path.join(DATA, "tok4096.json")),
        "train_tokens": counts["train"], "val_tokens": counts["val"],
    }
    tmp = os.path.join(DATA, "FROZEN.txt.tmp")
    with open(tmp, "w") as f:
        json.dump(frozen, f, indent=2)
        f.write("\n")
    os.replace(tmp, os.path.join(DATA, "FROZEN.txt"))  # atomic: readers see the old file or the new one
    print(f"wrote {os.path.join(DATA, 'FROZEN.txt')}")


if __name__ == "__main__":
    main()
