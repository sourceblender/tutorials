"""Monday, first thing: check that this machine can run the week.

Reports the Python version, which PyTorch device the week will use (Apple GPU "mps", NVIDIA "cuda" or "cpu"),
whether that device can train in bfloat16, and whether the morpheme CLI 0.5.0 is on PATH. Exits 1 if
something required is missing, so you fix setup before Monday's data work rather than halfway through it.
--no-cli skips the morpheme CLI check: only Monday's tokenizer training needs it.
"""
import platform, shutil, subprocess, sys

import torch


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def bf16_ok(dev):
    """Try one tiny bfloat16 matmul under autocast; a device that can't do it trains in float32 instead."""
    if dev == "cpu":
        return False
    try:
        with torch.autocast(device_type=dev, dtype=torch.bfloat16):
            a = torch.randn(8, 8, device=dev)
            (a @ a).sum().item()
        return True
    except Exception:
        return False


def main():
    no_cli = "--no-cli" in sys.argv[1:]
    problems = []
    print(f"python   {platform.python_version()}")
    if sys.version_info < (3, 12):
        problems.append("Python 3.12 or newer is required")
    dev = pick_device()
    print(f"torch    {torch.__version__}")
    print(f"device   {dev}" + ("   (CPU: use the short-run flags all week)" if dev == "cpu" else ""))
    print(f"bf16     {'yes' if bf16_ok(dev) else 'no (training will use float32)'}")
    cli = shutil.which("morpheme")
    if no_cli:
        print("morpheme CLI check skipped (--no-cli)")
    elif not cli:
        problems.append("morpheme CLI not on PATH (the Python package does not include it; see the README)")
        print("morpheme missing")
    else:
        version = subprocess.run([cli, "--version"], capture_output=True, text=True).stdout.strip()
        print(f"morpheme {version}")
        if version != "morpheme 0.5.0":
            problems.append(f"expected morpheme 0.5.0, found {version!r}")
    for p in problems:
        print(f"PROBLEM: {p}")
    if problems:
        sys.exit(1)
    print("setup OK")


if __name__ == "__main__":
    main()
