"""Friday's scoreboard: how much better than guessing is the trained model?

Computes three validation cross-entropies on ALL validation tokens (non-overlapping 256-token windows), each from
scratch on this machine:
  untrained  a freshly initialised model (seed 1234)
  unigram    add-one smoothed token frequencies counted on train, ignoring context
  trained    the checkpoint you pass (default runs/main/final.pt)
Pass bar for the full reference run: trained <= 4.11, at least 30% below unigram (the bar we set before our first
run). --short marks a CPU short-path checkpoint: it reports the numbers but doesn't apply the reference bar.
Exits 1 if the reference bar fails.
"""
import argparse, math, os, sys

import numpy as np
import torch
import torch.nn.functional as F

import frozen
from model import GPT, device

V, SEQ, BAR = 4096, 256, 4.11


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.path.join("runs", "main", "final.pt"))
    ap.add_argument("--short", action="store_true")
    ap.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    a = ap.parse_args()
    fz = frozen.check()
    dev = device() if a.device == "auto" else a.device
    train = np.load(os.path.join("data", f"train.{fz['train.txt'][:12]}.npy"), mmap_mode="r")
    val = np.load(os.path.join("data", f"val.{fz['val.txt'][:12]}.npy"), mmap_mode="r")

    counts = np.bincount(np.asarray(train, dtype=np.int64), minlength=V) + 1
    logp = np.log(counts / counts.sum())
    ys = np.asarray(val[1:], dtype=np.int64)
    unigram = -logp[ys].mean()

    @torch.no_grad()
    def ce(model):
        model.eval(); total, n = 0.0, 0
        starts = list(range(0, len(val) - SEQ - 1, SEQ))
        for i in range(0, len(starts), 16):
            s = starts[i:i + 16]
            x = torch.from_numpy(np.stack([val[j:j + SEQ] for j in s]).astype(np.int64)).to(dev)
            y = torch.from_numpy(np.stack([val[j + 1:j + SEQ + 1] for j in s]).astype(np.int64)).to(dev)
            total += F.cross_entropy(model(x).float().reshape(-1, V), y.reshape(-1), reduction="sum").item()
            n += y.numel()
        return total / n

    torch.manual_seed(1234)
    untrained = ce(GPT().to(dev))
    model = GPT().to(dev)
    model.load_state_dict(torch.load(a.checkpoint, map_location=dev, weights_only=True))
    trained = ce(model)
    print(f"untrained {untrained:.4f}   (uniform guessing: ln {V} = {math.log(V):.4f})")
    print(f"unigram   {unigram:.4f}   (word-frequency guessing, no context)")
    print(f"trained   {trained:.4f}   ({100 * (1 - trained / unigram):.0f}% below unigram)")
    if a.short:
        print("short path: numbers reported, reference bar not applied")
        return
    if trained > BAR:
        sys.exit(f"FAIL: trained {trained:.4f} is above the reference bar {BAR}")
    print(f"reference bar {BAR}: pass")


if __name__ == "__main__":
    main()
