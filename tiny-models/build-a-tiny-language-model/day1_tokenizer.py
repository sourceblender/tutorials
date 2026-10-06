"""Day 1: train the tokenizer with morpheme, on the training split only.

Runs the morpheme CLI (v0.5.0; install from https://github.com/sourceblender/morpheme/releases or
`cargo install morpheme-cli`) on the training stories only, never on validation:

    morpheme train --model bpe --preset byte-level --vocab-size 4096 --special-token '<|endoftext|>' \
        --out data/tok4096.json data/stories/train.txt

The resulting tokenizer.json is compared with the hash the tutorial used. A mismatch is reported (exit 1) rather than
silently continued, because every later number depends on these exact token ids.
"""
import hashlib, os, shutil, subprocess, sys

EXPECT = "c9ced905c3aba62a08acf63e67575f962fb87465407d18ea2da10090639fa1b9"
OUT = os.path.join("data", "tok4096.json")


def main():
    cli = shutil.which("morpheme")
    if not cli:
        sys.exit("morpheme CLI not found on PATH; see the README for the install command")
    subprocess.run([cli, "train", "--model", "bpe", "--preset", "byte-level", "--vocab-size", "4096",
                    "--special-token", "<|endoftext|>", "--out", OUT, os.path.join("data", "stories", "train.txt")], check=True)
    got = hashlib.sha256(open(OUT, "rb").read()).hexdigest()
    if got != EXPECT:
        sys.exit(f"tokenizer hash mismatch: {got} (expected {EXPECT}); check `morpheme --version` is 0.5.0")
    print("tokenizer matches the tutorial's tok4096.json")


if __name__ == "__main__":
    main()
