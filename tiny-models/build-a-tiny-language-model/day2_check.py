"""Day 2's checks: shift, size, shape, starting loss and the causal mask.

1. The training objective in one picture: a 16-token window x and its target y = x shifted left by one.
2. Parameter count of the week's model (expect 5,051,904).
3. A forward pass on one tiny batch (B=1, T=16): logits (1, 16, 4096). Then the training shape (32, 256, 4096).
4. Starting loss on 8 validation windows, near ln(4096) = 8.318: an untrained model should be nearly unsure.
5. Causal mask, on its own full-length input: changing token 200 must not change any logit at positions 0-199.
Runs on CPU with fixed seeds to keep numbers close across machines (not guaranteed identical); each check has a
tolerance, and that tolerance is the criterion. Exits 1 if a check fails.
--default-init skips the careful initialisation, to show the bug it prevents.
"""
import argparse, math, os, sys

import morpheme
import numpy as np
import torch
import torch.nn.functional as F

import frozen
from model import GPT

V, SEQ = 4096, 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--default-init", action="store_true", help="use PyTorch's default init (shows the bug)")
    a = ap.parse_args()
    frozen.check()
    torch.manual_seed(1234)
    tok = morpheme.Tokenizer.from_file(os.path.join("data", "tok4096.json"))
    val = np.array(tok.encode(open(os.path.join("data", "stories", "val.txt"), encoding="utf-8").read()).ids)
    fails = []

    x = torch.tensor(val[:16][None]); y = torch.tensor(val[1:17][None])
    print("1. input  x:", repr(tok.decode(x[0].tolist())))
    print("   target y:", repr(tok.decode(y[0].tolist())))
    print("   (y is x moved one token left: at every position the model must guess the NEXT token)")

    model = GPT(careful_init=not a.default_init)
    n = sum(p.numel() for p in model.parameters())
    print(f"2. parameters: {n:,}")
    groups = {"token embedding": 0, "position embedding": 0, "attention (5 blocks)": 0, "MLP (5 blocks)": 0,
              "LayerNorms": 0}
    for name, p in model.named_parameters():  # the tied head shares tok.weight, so it is listed once
        key = ("token embedding" if name.startswith("tok.") else "position embedding" if name.startswith("pos.")
               else "attention (5 blocks)" if ".qkv." in name or ".proj." in name
               else "MLP (5 blocks)" if ".mlp." in name else "LayerNorms")
        groups[key] += p.numel()
    for key, count in groups.items():
        print(f"     {key:22s} {count:>10,}")
    print(f"     {'output head (tied)':22s} {0:>10,}   (reuses the token embedding)")
    if sum(groups.values()) != n:
        fails.append("parameter breakdown does not sum to the total")
    if n != 5_051_904:
        fails.append(f"parameter count {n:,}, expected 5,051,904")

    with torch.no_grad():
        logits = model(x)
        print(f"3. tiny batch logits: {tuple(logits.shape)}   training batches will be (32, {SEQ}, {V})")
        if tuple(logits.shape) != (1, 16, V):
            fails.append(f"logits shape {tuple(logits.shape)}")

        g = torch.Generator().manual_seed(0)
        starts = torch.randint(0, len(val) - SEQ - 1, (8,), generator=g).tolist()
        xb = torch.tensor(np.stack([val[s:s + SEQ] for s in starts]))
        yb = torch.tensor(np.stack([val[s + 1:s + SEQ + 1] for s in starts]))
        loss = F.cross_entropy(model(xb).reshape(-1, V), yb.reshape(-1)).item()
        print(f"4. starting loss: {loss:.4f}   (uniform guessing would be ln({V}) = {math.log(V):.4f})")
        if abs(loss - math.log(V)) > 0.5:
            fails.append(f"starting loss {loss:.2f} is far from ln({V}) = {math.log(V):.2f}; check the initialisation")

        seq = torch.randint(0, V, (1, SEQ), generator=g)
        before = model(seq)
        changed = seq.clone(); changed[0, 200] = (changed[0, 200] + 1) % V
        after = model(changed)
        past_same = torch.allclose(before[0, :200], after[0, :200], atol=1e-5)
        future_moved = not torch.allclose(before[0, 200:], after[0, 200:], atol=1e-5)
        print(f"5. causal mask: positions 0-199 unchanged = {past_same}; positions 200+ changed = {future_moved}")
        if not (past_same and future_moved):
            fails.append("causal mask check failed")

    for f in fails:
        print(f"FAIL: {f}")
    if fails:
        sys.exit(1)
    print("Day 2 checks pass")


if __name__ == "__main__":
    main()
