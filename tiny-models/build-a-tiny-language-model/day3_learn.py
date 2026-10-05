"""Wednesday: the learning step, small enough to watch.

--overfit  Train on ONE batch of 4 windows, over and over. A working learning step memorises it: the loss must
           fall below 1.0 within 100 steps (we measured 0.005 at step 100 on an M5 Air). If it can't memorise
           one batch, something in loss / backward / optimizer is broken. Exits 1 on failure.
(default)  A short real run: 300 steps on random training windows, with the same recipe as Friday's full run
           (AdamW lr 1e-3, betas 0.9/0.95, weight decay 0.1, gradient clip 1.0). The learning rate warms up
           linearly to 1e-3 over 200 steps, then decays along a cosine to 10% at the last step (schedule.py).
           Writes the whole schedule to runs/lr_schedule.csv for plotting and prints a few points, then the loss
           (and perplexity = exp(loss)) every 25 steps.
--cpu-short  Smaller batches for machines without a GPU.

Both modes stop on a non-finite loss (NaN or inf) and save what they have to runs/emergency.pt first, so a
blown-up run is something you can inspect rather than lose.
"""
import argparse, math, os, sys, time

import morpheme
import numpy as np
import torch
import torch.nn.functional as F

import frozen
import schedule
from model import GPT, device

V, SEQ, LR, WARMUP = 4096, 256, 1e-3, 200


def lr_at(step, total):
    return schedule.lr_at(step, total, LR, WARMUP)


def autocast(dev):
    """bf16 where the device supports it (Monday's setup_check reports this); otherwise plain float32."""
    if dev != "cpu":
        try:
            with torch.autocast(device_type=dev, dtype=torch.bfloat16):
                (torch.ones(2, 2, device=dev) @ torch.ones(2, 2, device=dev)).sum().item()
            return torch.autocast(device_type=dev, dtype=torch.bfloat16)
        except Exception:
            pass
    return torch.autocast(device_type="cpu", enabled=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overfit", action="store_true")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--cpu-short", action="store_true", help="batch 8 instead of 32 (for CPU-only machines)")
    ap.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    ap.add_argument("--simulate-nan-at", type=int, default=0, help="force a non-finite loss at this step (shows the guard)")
    a = ap.parse_args()
    dev = device() if a.device == "auto" else a.device
    frozen.check()
    torch.manual_seed(1234)
    tok = morpheme.Tokenizer.from_file(os.path.join("data", "tok4096.json"))
    train = np.array(tok.encode(open(os.path.join("data", "stories", "train.txt"), encoding="utf-8").read()).ids,
                     dtype=np.int64)
    model = GPT().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
    gen = torch.Generator().manual_seed(0)

    def windows(b):
        starts = torch.randint(0, len(train) - SEQ - 1, (b,), generator=gen).tolist()
        x = torch.tensor(np.stack([train[s:s + SEQ] for s in starts]))
        y = torch.tensor(np.stack([train[s + 1:s + SEQ + 1] for s in starts]))
        return x.to(dev), y.to(dev)

    if a.overfit:
        steps, batch = 100, windows(4)
        print(f"overfit: one batch of 4 windows, {steps} steps, constant lr {LR} on {dev}")
    else:
        steps, b = a.steps, (8 if a.cpu_short else 32)
        print(f"short run: {steps} steps, batch {b} x {SEQ} tokens on {dev}")
        if not a.simulate_nan_at:  # a deliberate blow-up test must not replace the real run's schedule file
            os.makedirs("runs", exist_ok=True)
            with open(os.path.join("runs", "lr_schedule.csv"), "w") as f:
                f.write("step,lr\n" + "".join(f"{s},{lr_at(s, steps):.8f}\n" for s in range(1, steps + 1)))
        # start, halfway through warm-up, the peak, halfway through decay, the last step (only those inside this run)
        points = sorted({p for p in (1, WARMUP // 2, WARMUP, (WARMUP + steps) // 2, steps) if 1 <= p <= steps})
        where = "all steps in runs/lr_schedule.csv" if not a.simulate_nan_at else "file not written for a NaN test"
        print(f"learning-rate schedule ({where}):", "  ".join(f"step {p}: {lr_at(p, steps):.2e}" for p in points))
    t0, loss = time.perf_counter(), None
    for step in range(1, steps + 1):
        if not a.overfit:
            for group in opt.param_groups:
                group["lr"] = lr_at(step, steps)
        x, y = batch if a.overfit else windows(b)
        with autocast(dev):
            logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, V).float(), y.reshape(-1))
        if step == a.simulate_nan_at:
            loss = loss * float("nan")  # a stand-in for a real blow-up, so the guard below can be seen working
        if not torch.isfinite(loss):
            os.makedirs("runs", exist_ok=True)
            torch.save({"model": model.state_dict(), "step": step - 1}, os.path.join("runs", "emergency.pt"))  # weights after step - 1 updates
            sys.exit(f"step {step}: loss is {loss.item()}; saved runs/emergency.pt and stopped")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % 25 == 0:
            print(f"step {step:4d}  loss {loss.item():.3f}  perplexity {math.exp(loss.item()):,.1f}")
    print(f"done in {time.perf_counter() - t0:.1f}s")
    if a.overfit and loss.item() >= 1.0:
        sys.exit(f"FAIL: could not memorise one batch (loss {loss.item():.3f} after {steps} steps)")
    if not a.overfit and loss.item() >= math.log(V) - 1.0:
        sys.exit(f"FAIL: loss {loss.item():.3f} has not moved clearly below the starting ~{math.log(V):.1f}")
    print("Wednesday check passes")


if __name__ == "__main__":
    main()
