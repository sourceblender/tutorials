# Write an Eval That Can Say No

Runnable companion to Eric Mey's Hugging Face tutorial. The article explains how to choose the release decision, write cases, interpret failures, and carry this method into an application. **Article link will be added when it is published.**

This small sentiment classifier makes the failure visible: the candidate improves held-out accuracy from **0.710 to 0.780**, yet breaks three authored cases the incumbent got right. The gate rejects it. The deliberately broken model is a negative control; synthetic gate tests prove the gate can also pass. These are demonstration results on SST-2, not evidence about a production system.

## Run it

You need Python 3.11 or newer, [uv](https://docs.astral.sh/uv/getting-started/installation/), and internet access for the first [SST-2](https://huggingface.co/datasets/stanfordnlp/sst2) download. No GPU is needed.

```bash
git clone https://github.com/sourceblender/tutorials.git
cd tutorials/evals/write-an-eval-that-can-say-no
uv sync --locked
uv run python freeze.py
git diff --exit-code -- evals/FROZEN.sha256
uv run python train.py
uv run python -m unittest test_gate
```

The freeze step regenerates `evals/heldout.jsonl` from the dataset revision pinned in `freeze.py`. That file is intentionally untracked because the Hub lists the dataset license as unknown. The committed hash is the expected value; the `git diff` command above checks that the regenerated rows, authored cases, and gate policy still produce it. `freeze.py` also prints a hash of the held-out rows alone: `a3a3519a046a328d` means your data matches ours, independent of how the other files were saved.

Now run the candidate and the known-bad control. **Exit 1 is expected for both**: it means the gate rejected the model.

```bash
uv run python run_eval.py candidate
uv run python run_eval.py broken
```

The scorer uses distinct exits: `0` eligible for the next release step, `1` rejected, `2` frozen inputs changed, `3` evaluation could not run. A rejected model is a result; an unavailable model is a broken evaluation job.

For the optional uncertainty calculation, run `uv run python bootstrap.py`. It resamples the same 300 held-out rows for both models and reports the paired interval printed in the article.

## Files

- `evals/gate.toml` — three illustrative release conditions, written before training.
- `evals/cases.jsonl` — 20 visible regression and development cases, each with a reason.
- `evals/FROZEN.sha256` — expected hash of the cases, held-out rows, and gate policy.
- `freeze.py` — downloads a pinned SST-2 revision and regenerates the held-out rows.
- `train.py` — trains the incumbent, candidate, and shuffled-label control.
- `run_eval.py` — scores them and applies the gate.
- `test_gate.py` — a passing fixture, the exact threshold boundary, independent failures for each condition, and malformed input that must raise an error rather than produce a verdict.
- `bootstrap.py` — optional paired bootstrap over the held-out rows.
- `sensitivity.py` — optional check run after the result: retrains both models on training shuffle seeds 0–19 and applies the same gate, reproducing the article's 20-run counts. It is not part of the gate.

The visible cases are meant for iteration. In a real release process, keep an independent promotion set away from the jobs and people tuning the model, and combine the offline gate with review and live checks. A pass here only means the declared offline conditions were met.

Code: MIT license in the repository root. SST-2 remains with its publisher and is downloaded at runtime.
