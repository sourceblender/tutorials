# Build a tiny language model in five days

Over one week you build a small story-writing language model from scratch, train it on your own computer, save
it, load it back in a fresh process, and read what it writes. Each day ends with something on disk and one check
that can fail.

| Day | Lesson | You finish with |
| --- | --- | --- |
| Day 1 | [Give your model something to read](lessons/01-MONDAY.md) | Frozen story data, a trained tokenizer, a sentence round-tripped |
| Day 2 | [Turn token IDs into guesses](lessons/02-TUESDAY.md) | A 5,051,904-parameter model whose forward pass, starting loss and causal mask check out |
| Day 3 | [Make a guess less wrong](lessons/03-WEDNESDAY.md) | A learning step that memorises one batch and moves the loss on real stories |
| Day 4 | [Save the experiment, then prove it resumes](lessons/04-THURSDAY.md) | Checkpoints that continue the same run, checked against an uninterrupted one |
| Day 5 | [Read what your saved model writes](lessons/05-FRIDAY.md) | A trained model, a scoreboard against two baselines, and five stories |

## Before you start

You need basic Python and terminal experience, [uv](https://docs.astral.sh/uv/), and the morpheme CLI 0.5.0
(Day 1 installs it). The reference machine is an Apple-silicon Mac. The CPU path was also run on an M5 Air's CPU
and on Ubuntu 26.04 x86_64 (CPU only); Windows has not been checked.

Run every command from this folder. Day 1 downloads a pinned 40 MiB slice of
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (CDLA-Sharing-1.0); the data is downloaded,
not redistributed.

## Files

`setup_check.py`, `day1_*.py` (Day 1) · `model.py`, `day2_*.py` (Day 2) · `day3_learn.py`, `schedule.py`
(Day 3) · `train.py`, `day4_resume_check.py` (Day 4) · `day5_preflight.py`, `day5_scoreboard.py`,
`generate.py` (Day 5) · `frozen.py` checks Day 1's frozen inputs before every later day.
