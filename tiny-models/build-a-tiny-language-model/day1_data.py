"""Monday: fetch the story data and split it into train/val.

Downloads the first 40 MiB of TinyStoriesV2-GPT4-train.txt from a pinned dataset revision (roneneldan/TinyStories,
CDLA-Sharing-1.0), splits it into stories on "<|endoftext|>", drops the cut-off last piece, keeps the first 95% of
stories for training and the rest for validation, and writes data/stories/{train,val}.txt with stories joined by
"\n<|endoftext|>\n". Every file is checked against the hash the tutorial was built with; a mismatch exits 1.
The data is downloaded, not redistributed.
"""
import hashlib, os, sys, urllib.request

REV = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
URL = f"https://huggingface.co/datasets/roneneldan/TinyStories/resolve/{REV}/TinyStoriesV2-GPT4-train.txt"
NBYTES = 40 * 1024 * 1024
EXPECT = {
    "slice": "ee1f1386743430ef2f1fcddcd29dc004b3716517dd1cfc72d95b3fa9287c9919",
    "train.txt": "1df1a31eb15a0372f66539a373796cd3890ce4c9d456a9e070d38728e9333acb",
    "val.txt": "95d81feaf209229d1dff5e97b14341b76a0e6df16b556b0b34b47d25c8c13739",
}
SEP = "<|endoftext|>"
OUT = os.path.join("data", "stories")


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
    k = int(len(stories) * 0.95)
    joiner = "\n" + SEP + "\n"
    parts = {"train.txt": stories[:k], "val.txt": stories[k:]}
    for name, part in parts.items():
        data = (joiner.join(part) + joiner).encode("utf-8")
        if sha(data) != EXPECT[name]:
            sys.exit(f"{name} hash mismatch: {sha(data)}")
        with open(os.path.join(OUT, name), "wb") as f:
            f.write(data)
    print(f"stories: {len(stories)} (train {k}, val {len(stories) - k}); hashes match")


if __name__ == "__main__":
    main()
