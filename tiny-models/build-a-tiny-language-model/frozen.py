"""Monday's freeze, enforced: every later entry point calls check() before touching the data.

Confirms data/FROZEN.txt exists, that it records the pinned hashes from day1_data.py and day1_tokenizer.py, and that
train.txt, val.txt and tok4096.json on disk still match it. Exits with a message instead of letting a later day run
on changed inputs. Read-only: it never rewrites anything.
"""
import hashlib, json, os, sys

import day1_data
import day1_tokenizer

FILES = {"train.txt": os.path.join("data", "stories", "train.txt"),
         "val.txt": os.path.join("data", "stories", "val.txt"),
         "tok4096.json": os.path.join("data", "tok4096.json")}
PINNED = {"train.txt": day1_data.EXPECT["train.txt"], "val.txt": day1_data.EXPECT["val.txt"],
          "tok4096.json": day1_tokenizer.EXPECT}


def check():
    path = os.path.join("data", "FROZEN.txt")
    if not os.path.exists(path):
        sys.exit("data/FROZEN.txt missing: finish Monday (day1_look.py) first")
    fz = json.load(open(path))
    for name, file in FILES.items():
        if fz.get(name) != PINNED[name]:
            sys.exit(f"FROZEN.txt records a different {name} than the tutorial's pinned one; rerun Monday")
        if not os.path.exists(file) or hashlib.sha256(open(file, "rb").read()).hexdigest() != fz[name]:
            sys.exit(f"{file} changed since Monday's freeze; rerun Monday's scripts")
    return fz
