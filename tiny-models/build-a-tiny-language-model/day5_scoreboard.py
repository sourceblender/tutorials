"""Friday's scoreboard: how much better than guessing is the trained model?

Computes three validation cross-entropies over exactly the same targets: every validation token after the first,
each scored once (non-overlapping 256-token windows, then the shorter leftover tail). All three are recomputed on
this machine:
  untrained  a freshly initialised model (seed 1234)
  unigram    add-one smoothed TOKEN frequencies counted on train, ignoring context
  trained    the checkpoint you pass (default runs/main/final.pt)
The scored target count is printed and checked to be identical for all three. A non-finite score or a checkpoint
with non-finite weights fails.
Pass rule for the full reference run, a tutorial smoke-quality criterion ("learned clearly more than which tokens
are common"): trained cross-entropy at least 30% lower than unigram, i.e. trained <= 0.70 x unigram, both scored on
the same targets in this same run. --short marks a CPU short-path checkpoint: it reports the numbers but doesn't
apply the rule.
Exits 1 if the reference bar fails.
"""
import argparse, math, os, sys

import numpy as np
import torch
import torch.nn.functional as F

import frozen
from model import GPT, device

V, SEQ, RATIO = 4096, 256, 0.70


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
    unigram, n_targets = -logp[ys].mean(), len(ys)

    @torch.no_grad()
    def ce(model):
        """Every target val[1:] exactly once: full windows in batches of 16, then the shorter tail window."""
        model.eval(); total, n = 0.0, 0
        last = len(val) - 1                      # inputs val[0:last], targets val[1:last + 1]
        full = list(range(0, last - SEQ + 1, SEQ))
        for i in range(0, len(full), 16):
            s = full[i:i + 16]
            x = torch.from_numpy(np.stack([val[j:j + SEQ] for j in s]).astype(np.int64)).to(dev)
            y = torch.from_numpy(np.stack([val[j + 1:j + SEQ + 1] for j in s]).astype(np.int64)).to(dev)
            total += F.cross_entropy(model(x).float().reshape(-1, V), y.reshape(-1), reduction="sum").item()
            n += y.numel()
        tail = (full[-1] + SEQ) if full else 0
        if tail < last:
            x = torch.from_numpy(np.asarray(val[tail:last], dtype=np.int64))[None].to(dev)
            y = torch.from_numpy(np.asarray(val[tail + 1:last + 1], dtype=np.int64))[None].to(dev)
            total += F.cross_entropy(model(x).float().reshape(-1, V), y.reshape(-1), reduction="sum").item()
            n += y.numel()
        if n != n_targets:
            sys.exit(f"scored {n} targets, expected {n_targets}")
        return total / n

    torch.manual_seed(1234)
    untrained = ce(GPT().to(dev))
    model = GPT().to(dev)
    model.load_state_dict(torch.load(a.checkpoint, map_location=dev, weights_only=True))
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        sys.exit("FAIL: the checkpoint has non-finite weights")
    trained = ce(model)
    if not all(map(math.isfinite, (untrained, unigram, trained))):
        sys.exit(f"FAIL: non-finite score (untrained {untrained}, unigram {unigram}, trained {trained})")
    print(f"scored targets: {n_targets:,} (every validation token after the first, the same set for all three)")
    print(f"untrained {untrained:.4f}   perplexity {math.exp(untrained):7,.1f}   (uniform guessing: ln {V} = {math.log(V):.4f})")
    print(f"unigram   {unigram:.4f}   perplexity {math.exp(unigram):7,.1f}   (token-frequency guessing, no context)")
    print(f"trained   {trained:.4f}   perplexity {math.exp(trained):7,.1f}   "
          f"({100 * (1 - trained / unigram):.0f}% lower cross-entropy than unigram)")
    if a.short:
        print("short path: numbers reported, reference bar not applied")
        return
    bar = RATIO * unigram
    print(f"rule: trained <= {RATIO:.2f} x unigram = {bar:.4f}")
    if trained > bar:
        sys.exit(f"FAIL: trained {trained:.4f} is above {bar:.4f}")
    print("rule: pass")


if __name__ == "__main__":
    main()
