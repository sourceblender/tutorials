"""Friday, before the full run: every earlier day's check, in order, stopping at the first failure.

setup_check -> frozen inputs -> day2_check (shape, starting loss, causal mask) -> day3_learn --overfit (the learning
step works) -> day4_resume_check (a stopped run resumes as the same run). About a minute and a half on our Air;
the resume check is most of it.
"""
import subprocess, sys

import frozen

STEPS = [["setup_check.py", "--no-cli"], ["day2_check.py"], ["day3_learn.py", "--overfit"], ["day4_resume_check.py"]]


def main():
    frozen.check()
    print("frozen inputs: match Monday's freeze")
    for step in STEPS:
        res = subprocess.run([sys.executable, *step], capture_output=True, text=True)
        last = (res.stdout.strip().splitlines() or [""])[-1]
        print(f"{' '.join(step):28s} {'ok' if res.returncode == 0 else 'FAILED'}   {last}")
        if res.returncode:
            print(res.stdout[-2000:], res.stderr[-2000:], sep="\n")
            sys.exit(f"preflight stopped at {' '.join(step)}; fix it before launching the full run")
    print("preflight passes: launch with  uv run python train.py --out runs/main")


if __name__ == "__main__":
    main()
