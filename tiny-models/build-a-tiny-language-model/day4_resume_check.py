"""Day 4's check: does resuming really continue the same run?

Runs short trainings into a temporary folder on --device (default: this machine's training device; use --device cpu
for the tightest comparison):
  A  120 steps straight through.
  B  the same settings, stopped after step 50 (not an eval step), then resumed with --resume.
  C  the same settings with a simulated blow-up at step 30, then resumed from its emergency.pt.
All runs happen in a temporary copy of data/, so the check never touches your real FROZEN.txt, even if it is
killed halfway. Every 10 steps the runs log train loss, validation loss and a fingerprint of the windows that step trained on.
After a resume, B and C must train on exactly the same windows as A (the sampler's position was restored, and C
retries the failed step's batch instead of skipping it), with losses within a tolerance: 1e-4 on CPU, where runs
repeat closely, and a wider measured band on GPUs, which are not bit-reproducible. Then two refusals: resuming with
different settings, and resuming after the data changed. Exits 1 if anything disagrees.
"""
import argparse, json, os, shutil, subprocess, sys, tempfile

from model import device

GPU_TOL = 0.05  # GPU runs drift; the exact-window match is the strict part there


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    a = ap.parse_args()
    dev = device() if a.device == "auto" else a.device
    tol = 1e-4 if dev == "cpu" else GPU_TOL
    base = ["--device", dev, "--batch", "8", "--steps", "120", "--eval-every", "10"]

    here = os.path.dirname(os.path.abspath(__file__))
    tmp = tempfile.mkdtemp(prefix="resume-check-")
    work = os.path.join(tmp, "work")
    shutil.copytree(os.path.join(here, "data"), os.path.join(work, "data"))  # isolated: the planted hash stays here

    def run(args, extra=()):
        return subprocess.run([sys.executable, os.path.join(here, "train.py"), *base, *extra, *args],
                              capture_output=True, text=True, cwd=work)

    def evals(out):
        return {r["step"]: r for r in map(json.loads, open(os.path.join(out, "log.jsonl"))) if r["event"] == "eval"}

    fails = []
    try:
        A, B, C = (os.path.join(tmp, n) for n in "ABC")
        steps = [("A", ["--out", A], 0), ("B stop", ["--out", B, "--stop-after", "50"], 0),
                 ("B resume", ["--out", B, "--resume"], 0), ("C blow-up", ["--out", C, "--simulate-nan-at", "30"], 1),
                 ("C resume", ["--out", C, "--resume-from", os.path.join(C, "emergency.pt")], 0)]
        for label, args, want in steps:
            res = run(args)
            if res.returncode != want:
                sys.exit(f"{label}: exit {res.returncode}, expected {want}\n{res.stdout[-1500:]}{res.stderr[-1500:]}")
        a_rows = evals(A)
        print(f"device {dev}, loss tolerance {tol}")
        for name, out, first in (("B", B, 60), ("C", C, 30)):
            rows = evals(out)
            for step in sorted(s for s in a_rows if s >= first):
                x, y = a_rows[step], rows.get(step)
                if y is None:
                    fails.append(f"{name} step {step}: missing"); continue
                same = x["starts_sha"] == y["starts_sha"]
                dt, dv = abs(x["train_loss"] - y["train_loss"]), abs(x["val_ce"] - y["val_ce"])
                if step in (first, 120):
                    print(f"{name} step {step:3d}  windows same {same}   train diff {dt:.4f}   val diff {dv:.4f}")
                if not same or dt > tol or dv > tol:
                    fails.append(f"{name} step {step}: windows same {same}, train diff {dt:.4f}, val diff {dv:.4f}")
        refused = subprocess.run([sys.executable, os.path.join(here, "train.py"), "--device", dev, "--batch", "8",
                                  "--steps", "121", "--eval-every", "10", "--out", B, "--resume"],
                                 capture_output=True, text=True, cwd=work)
        print(f"resume with changed settings refused: {refused.returncode != 0}")
        if refused.returncode == 0:
            fails.append("a resume with different settings was accepted")
        frozen_path = os.path.join(work, "data", "FROZEN.txt")  # the isolated copy, never the real one
        fz = json.load(open(frozen_path)); fz["val.txt"] = "0" * 64
        json.dump(fz, open(frozen_path, "w"))
        refused = run(["--out", B, "--resume"])
        print(f"resume after the data changed refused: {refused.returncode != 0}")
        if refused.returncode == 0:
            fails.append("a resume after the data changed was accepted")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for f in fails:
        print(f"FAIL: {f}")
    if fails:
        sys.exit(1)
    print("Day 4 check passes: resumed runs continued the same run")


if __name__ == "__main__":
    main()
