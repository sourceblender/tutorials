"""Day 1: fetch the story data and split it into train/val.

Downloads the first 40 MiB of TinyStoriesV2-GPT4-train.txt from a pinned dataset revision (roneneldan/TinyStories,
CDLA-Sharing-1.0) and splits it into stories on "<|endoftext|>", dropping the cut-off last piece. This is our own
held-out split of that one training file, not TinyStories' published validation set.

Each story goes to validation if a hash of its text (whitespace collapsed) falls in the lowest 5 of 100 buckets,
otherwise to training. The hash, not the story's position in the file, decides, so file order can't sneak into
the split, and identical stories always land on the same side. Stories keep their file order within each split
and are written as "\n<|endoftext|>\n"-joined text to data/stories/{train,val}.txt. Every file is checked against
the hash the tutorial was built with; a mismatch exits 1. The data is downloaded, not redistributed.
"""
import hashlib, os, sys, urllib.request

REV = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
URL = f"https://huggingface.co/datasets/roneneldan/TinyStories/resolve/{REV}/TinyStoriesV2-GPT4-train.txt"
NBYTES = 40 * 1024 * 1024
EXPECT = {
    "slice": "ee1f1386743430ef2f1fcddcd29dc004b3716517dd1cfc72d95b3fa9287c9919",
    "train.txt": "c28be7b24679db8ecf4b8d3913963a96a1a8dd932edf4fb9e2cf8e84b17f4863",
    "val.txt": "30d1a849bb745ea455413ab82c7d876f486f52bad69589d45d68143f9cd5cbb3",
}
SEP = "<|endoftext|>"
OUT = os.path.join("data", "stories")


def is_val(story):
    """The split rule: 5 of 100 hash buckets go to validation. Whitespace is collapsed first, so two copies of a story
    that differ only in spacing still share a bucket."""
    key = " ".join(story.split())
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % 100 < 5


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    os.makedirs(OUT, exist_ok=True)
    req = urllib.request.Request(URL, headers={"Range": f"bytes=0-{NBYTES - 1}"})
    raw = urllib.request.urlopen(req).read()
    if sha(raw) != EXPECT["slice"]:
        sys.exit(f"downloaded slice hash mismatch: {sha(raw)}")
    text = raw.decode("utf-8", errors="ignore")
    stories = [s.strip() for s in text.split(SEP)]
    stories = [s for s in stories[:-1] if s]  # the last piece is cut off by the byte range
    val = [s for s in stories if is_val(s)]
    train = [s for s in stories if not is_val(s)]
    joiner = "\n" + SEP + "\n"
    parts = {"train.txt": train, "val.txt": val}
    for name, part in parts.items():
        data = (joiner.join(part) + joiner).encode("utf-8")
        if sha(data) != EXPECT[name]:
            sys.exit(f"{name} hash mismatch: {sha(data)}")
        with open(os.path.join(OUT, name), "wb") as f:
            f.write(data)
    print(f"stories: {len(stories)} (train {len(train)}, val {len(val)}); hashes match")


if __name__ == "__main__":
    main()
