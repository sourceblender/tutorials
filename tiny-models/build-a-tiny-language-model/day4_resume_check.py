"""Thursday's check: does resuming really continue the same run?

Runs two short CPU trainings into a temporary folder:
  A: 120 steps straight through.
  B: the same settings, stopped after step 50 (not an eval step), then resumed with --resume.
Every 20 steps both log train loss, validation loss and a fingerprint of which windows that step trained on. After
the resume, B must match A: the same windows (so the sampler's position was restored), and losses within a small
tolerance. Then two refusals: resuming with different settings, and resuming after the data changed, must both
fail. CPU keeps the comparison tight; GPU runs are not guaranteed identical. Exits 1 if anything disagrees.
"""
import json, os, shutil, subprocess, sys, tempfile

TOL = 1e-4


def run(args):
    return subprocess.run([sys.executable, "train.py", "--device", "cpu", "--batch", "8", "--steps", "120",
                           "--eval-every", "20", *args], capture_output=True, text=True)


def evals(out):
    return {r["step"]: r for r in map(json.loads, open(os.path.join(out, "log.jsonl"))) if r["event"] == "eval"}


def main():
    tmp = tempfile.mkdtemp(prefix="resume-check-")
    fails = []
    try:
        a_out, b_out = os.path.join(tmp, "A"), os.path.join(tmp, "B")
        for label, res in (("A", run(["--out", a_out])), ("B stop", run(["--out", b_out, "--stop-after", "50"]))):
            if res.returncode:
                sys.exit(f"{label} failed:\n{res.stdout}{res.stderr}")
        res = run(["--out", b_out, "--resume"])
        if res.returncode or "resumed at step 51" not in res.stdout:
            sys.exit(f"resume failed:\n{res.stdout}{res.stderr}")
        A, B = evals(a_out), evals(b_out)
        for step in sorted(A):
            a, b = A[step], B.get(step)
            if b is None:
                fails.append(f"step {step}: missing in the resumed run"); continue
            same_windows = a["starts_sha"] == b["starts_sha"]
            dtrain, dval = abs(a["train_loss"] - b["train_loss"]), abs(a["val_ce"] - b["val_ce"])
            print(f"step {step:3d}  windows same {same_windows}   train {a['train_loss']:.4f} vs {b['train_loss']:.4f}"
                  f"   val {a['val_ce']:.4f} vs {b['val_ce']:.4f}")
            if not same_windows or dtrain > TOL or dval > TOL:
                fails.append(f"step {step}: resumed run diverged")
        refused = run(["--out", b_out, "--resume", "--steps", "121"])
        print(f"resume with changed settings refused: {refused.returncode != 0}")
        if refused.returncode == 0:
            fails.append("a resume with different settings was accepted")
        frozen = os.path.join("data", "FROZEN.txt")
        shutil.copy(frozen, os.path.join(tmp, "FROZEN.backup"))
        try:
            fz = json.load(open(frozen)); fz["val.txt"] = "0" * 64
            json.dump(fz, open(frozen, "w"))
            refused = run(["--out", b_out, "--resume"])
        finally:
            shutil.copy(os.path.join(tmp, "FROZEN.backup"), frozen)
        print(f"resume after the data changed refused: {refused.returncode != 0}")
        if refused.returncode == 0:
            fails.append("a resume after the data changed was accepted")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for f in fails:
        print(f"FAIL: {f}")
    if fails:
        sys.exit(1)
    print("Thursday check passes: the resumed run continued the same run")


if __name__ == "__main__":
    main()
