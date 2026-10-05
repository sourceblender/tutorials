"""Monday, last step: look at what the tokenizer produces, check it, and freeze the day's work.

1. Shows how a few sentences are cut into pieces and ids.
2. Round-trips the first 100 validation stories: decode(encode(text)) must give back the exact text.
3. Counts tokens in both splits and the average tokens per word (whitespace-separated) on validation.
4. Writes data/FROZEN.txt: the hashes and counts every later day checks before it starts.
Exits 1 if any round-trip fails.
"""
import hashlib, json, os, sys

import morpheme

SEP = "<|endoftext|>"
DATA = "data"


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    tok = morpheme.Tokenizer.from_file(os.path.join(DATA, "tok4096.json"))
    for sentence in ["Once upon a time, there was a little dog named Max.",
                     "The dragon's spaghetti was unbelievably wobbly!",
                     "Lily and Ben went to the park."]:
        enc = tok.encode(sentence)
        print(f"\n{sentence}\n  {len(enc.ids)} tokens: {enc.tokens}\n  ids: {enc.ids}")

    val_text = open(os.path.join(DATA, "stories", "val.txt"), encoding="utf-8").read()
    stories = [s.strip() for s in val_text.split(SEP) if s.strip()]
    failed = [i for i, s in enumerate(stories[:100]) if tok.decode(tok.encode(s).ids) != s]
    print(f"\nround-trip: {100 - len(failed)}/100 validation stories came back exactly")

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
    with open(os.path.join(DATA, "FROZEN.txt"), "w") as f:
        json.dump(frozen, f, indent=2)
        f.write("\n")
    print(f"wrote {os.path.join(DATA, 'FROZEN.txt')}")
    if failed:
        sys.exit(f"round-trip failed for validation stories {failed[:5]}")


if __name__ == "__main__":
    main()
