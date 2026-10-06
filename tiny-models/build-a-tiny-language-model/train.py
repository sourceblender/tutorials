"""Days 4 and 5: the full trainer, built so it can run unattended and resume where it stopped.

The recipe is the one we measured: 6,000 steps of 32 random windows of 256 tokens, AdamW lr 1e-3 betas (0.9, 0.95)
weight decay 0.1, gradient clip 1.0, bf16 autocast where supported. Learning rate: linear warm-up to 1e-3 over 200
steps, then cosine decay to 10% at the last step (schedule.py; its id is stored in every checkpoint).

What makes it reliable (Day 4):
- Tokens are encoded once and cached as a NumPy file next to the data, keyed by Day 1's FROZEN.txt.
- Every --eval-every steps: validation loss on a fixed set of windows (reported, never used to pick a checkpoint;
  for this teaching run the final step is the model), one JSON line in log.jsonl, and a checkpoint.
- A checkpoint holds everything a resume needs: model, optimizer, step, the window-sampling generator, PyTorch's
  RNG, the run's config, and Day 1's input hashes. It is written to a temporary file and then renamed, so a crash
  mid-write never leaves a half checkpoint.
- --resume refuses to continue if the config or Day 1's inputs changed.
- A non-finite loss stops the run after an emergency save.
- --stop-after N saves a checkpoint and exits cleanly after step N: a cooperative stop for testing resume. An abrupt
  interruption (Ctrl-C, a crash, a closed lid) has no handler here; it resumes from the latest periodic checkpoint
  and loses the steps since it.

Usage:
    uv run python train.py --out runs/friday              # Day 5's full run
    uv run python train.py --out runs/friday --resume     # continue from runs/friday/checkpoint.pt
    uv run python train.py --out runs/cpu --cpu-short     # smaller batch for CPU-only machines
"""
import argparse, hashlib, json, math, os, sys, time

import morpheme
import numpy as np
import torch
import torch.nn.functional as F

import frozen
import schedule
from model import GPT, device

V, SEQ = 4096, 256


def tokens(name, fz, tok):
    """Encode a split once; later runs load the cached ids. The cache name carries the split's hash."""
    cache = os.path.join("data", f"{name}.{fz[name + '.txt'][:12]}.npy")
    if not os.path.exists(cache):
        ids = tok.encode(open(os.path.join("data", "stories", f"{name}.txt"), encoding="utf-8").read()).ids
        np.save(cache + ".tmp.npy", np.array(ids, dtype=np.uint16))  # 4,096 ids fit in 16 unsigned bits (max 65,535)
        os.replace(cache + ".tmp.npy", cache)
    arr = np.load(cache, mmap_mode="r")
    if len(arr) != fz[f"{name}_tokens"]:
        sys.exit(f"{cache} has {len(arr)} tokens, FROZEN.txt says {fz[name + '_tokens']}")
    return arr


def amp(dev):
    if dev != "cpu":
        try:
            with torch.autocast(device_type=dev, dtype=torch.bfloat16):
                (torch.ones(2, 2, device=dev) @ torch.ones(2, 2, device=dev)).sum().item()
            return lambda: torch.autocast(device_type=dev, dtype=torch.bfloat16)
        except Exception:
            pass
    return lambda: torch.autocast(device_type="cpu", enabled=False)


def save(path, state):
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic: the old checkpoint survives until the new one is complete


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("runs", "friday"))
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--cpu-short", action="store_true", help="batch 8 and 1,000 steps, for CPU-only machines")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--lr", type=float, default=1e-3, help="peak learning rate (the recipe uses 1e-3)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--resume-from", default="", help="resume from this file instead of OUT/checkpoint.pt "
                    "(for example OUT/emergency.pt after a blow-up)")
    ap.add_argument("--stop-after", type=int, default=0, help="exit cleanly after this step (to test resuming)")
    ap.add_argument("--simulate-nan-at", type=int, default=0, help="force a non-finite loss at this step (shows the guard)")
    a = ap.parse_args()
    if a.cpu_short:
        a.batch, a.steps = 8, min(a.steps, 1000)
    dev = device() if a.device == "auto" else a.device
    config = {"steps": a.steps, "batch": a.batch, "seq": SEQ, "lr": a.lr, "warmup": 200, "schedule": schedule.SCHEDULE_ID,
              "eval_every": a.eval_every,
              "seed": a.seed, "model": {"vocab": V, "seq": SEQ, "d": 256, "layers": 5, "heads": 4}}
    fz = frozen.check()
    inputs = {k: fz[k] for k in ("train.txt", "val.txt", "tok4096.json")}
    tok = morpheme.Tokenizer.from_file(os.path.join("data", "tok4096.json"))
    train, val = tokens("train", fz, tok), tokens("val", fz, tok)

    torch.manual_seed(a.seed)
    model = GPT(**config["model"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=config["lr"], betas=(0.9, 0.95), weight_decay=0.1)
    gen = torch.Generator().manual_seed(a.seed)
    start = 1
    os.makedirs(a.out, exist_ok=True)
    ckpt_path, log_path = os.path.join(a.out, "checkpoint.pt"), os.path.join(a.out, "log.jsonl")
    if a.resume_from:
        a.resume = True
    if a.resume:
        source = a.resume_from or ckpt_path
        if not os.path.exists(source):
            sys.exit(f"--resume: no checkpoint at {source}")
        # Load on CPU: the sampler and RNG states must stay CPU ByteTensors (map_location=dev moved them onto the
        # GPU and set_state refused them on MPS). load_state_dict copies weights and optimizer state to the model's
        # device itself.
        ck = torch.load(source, map_location="cpu", weights_only=False)
        if ck["config"] != config:
            sys.exit("--resume refused: this run's settings differ from the checkpoint's")
        if ck["inputs"] != inputs:
            sys.exit("--resume refused: Day 1's data or tokenizer changed since the checkpoint")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        gen.set_state(ck["gen"])
        torch.set_rng_state(ck["torch_rng"])
        start = ck["step"] + 1
        print(f"resumed at step {start} from {source}")
    elif os.path.exists(log_path):
        sys.exit(f"{a.out} already has a run; use --resume or a new --out")

    val_gen = torch.Generator().manual_seed(0)  # a fixed validation set, the same every eval and every run
    val_starts = torch.randint(0, len(val) - SEQ - 1, (64,), generator=val_gen).tolist()
    run_amp = amp(dev)

    def windows(arr, starts):
        x = torch.from_numpy(np.stack([arr[s:s + SEQ] for s in starts]).astype(np.int64))
        y = torch.from_numpy(np.stack([arr[s + 1:s + SEQ + 1] for s in starts]).astype(np.int64))
        return x.to(dev), y.to(dev)

    @torch.no_grad()
    def val_ce():
        model.eval(); total = 0.0
        for i in range(0, len(val_starts), 16):
            x, y = windows(val, val_starts[i:i + 16])
            with run_amp():
                total += F.cross_entropy(model(x).reshape(-1, V).float(), y.reshape(-1)).item()
        model.train()
        return total / (len(val_starts) / 16)

    def state(step, gen_state=None, rng_state=None):
        """Everything a resume needs after `step` completed updates. gen_state/rng_state default to the current
        ones; the emergency save passes the states from BEFORE the failed step drew its batch, so a resume
        retries that same batch instead of skipping it."""
        return {"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                "gen": gen.get_state() if gen_state is None else gen_state,
                "torch_rng": torch.get_rng_state() if rng_state is None else rng_state,
                "config": config, "inputs": inputs}

    def log(**row):
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    if start == 1:
        log(event="start", device=dev, config=config, inputs=inputs, val_ce=round(val_ce(), 4))
    print(f"training on {dev}: steps {start}-{a.steps}, batch {a.batch} x {SEQ}")
    t0, seen = time.perf_counter(), 0
    for step in range(start, a.steps + 1):
        for g in opt.param_groups:
            g["lr"] = schedule.lr_at(step, a.steps, config["lr"], config["warmup"])
        gen_before, rng_before = gen.get_state(), torch.get_rng_state()
        starts = torch.randint(0, len(train) - SEQ - 1, (a.batch,), generator=gen).tolist()
        x, y = windows(train, starts)
        with run_amp():
            logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, V).float(), y.reshape(-1))
        if step == a.simulate_nan_at:
            loss = loss * float("nan")  # a stand-in for a real blow-up, so the guard below can be seen working
        if not torch.isfinite(loss):
            save(os.path.join(a.out, "emergency.pt"), state(step - 1, gen_before, rng_before))
            sys.exit(f"step {step}: loss is {loss.item()}; saved emergency.pt and stopped")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        seen += a.batch * SEQ
        if step % a.eval_every == 0 or step == a.steps:
            ce = val_ce()
            rate = seen / (time.perf_counter() - t0)
            log(event="eval", step=step, lr=round(opt.param_groups[0]["lr"], 8), train_loss=round(loss.item(), 4),
                val_ce=round(ce, 4), tokens_per_s=round(rate), starts_sha=hashlib.sha256(str(starts).encode()).hexdigest()[:12])
            save(ckpt_path, state(step))
            print(f"step {step:5d}  train {loss.item():.3f}  val {ce:.3f}  {rate:,.0f} tok/s")
        if a.stop_after and step == a.stop_after:
            save(ckpt_path, state(step))
            print(f"stopped after step {step} (checkpoint saved); resume with --resume")
            return
    torch.save(model.state_dict(), os.path.join(a.out, "final.pt"))
    log(event="final", step=a.steps, minutes=round((time.perf_counter() - t0) / 60, 2))
    print(f"done: {os.path.join(a.out, 'final.pt')}")


if __name__ == "__main__":
    main()
