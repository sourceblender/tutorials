# Build a tiny language model in five days

Over one week you build a small story-writing language model from scratch, train it on your own computer, save
it, load it back in a fresh process, and read what it writes. Each day ends with something on disk and one check
that can fail.

| Day | Lesson | You finish with |
| --- | --- | --- |
| Monday | [Give your model something to read](lessons/01-MONDAY.md) | Frozen story data, a trained tokenizer, a sentence round-tripped |
| Tuesday | [Turn token IDs into guesses](lessons/02-TUESDAY.md) | A 5,051,904-parameter model whose forward pass, starting loss and causal mask check out |
| Wednesday | [Make a guess less wrong](lessons/03-WEDNESDAY.md) | A learning step that memorises one batch and moves the loss on real stories |
| Thursday | [Save the experiment, then prove it resumes](lessons/04-THURSDAY.md) | Checkpoints that continue the same run, checked against an uninterrupted one |
| Friday | [Read what your saved model writes](lessons/05-FRIDAY.md) | A trained model, a scoreboard against two baselines, and five stories |

## Before you start

You need basic Python and terminal experience, [uv](https://docs.astral.sh/uv/), and the morpheme CLI 0.5.0
(Monday's lesson installs it). The reference machine is an Apple-silicon Mac. A CPU path is included and was run
on an M5 Air's CPU; Linux and Windows have not been checked for this folder.

Run every command from this folder. Monday downloads a pinned 40 MiB slice of
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (CDLA-Sharing-1.0); the data is downloaded,
not redistributed.

## Files

`setup_check.py`, `day1_*.py` (Monday) · `model.py`, `day2_*.py` (Tuesday) · `day3_learn.py`, `schedule.py`
(Wednesday) · `train.py`, `day4_resume_check.py` (Thursday) · `day5_preflight.py`, `day5_scoreboard.py`,
`generate.py` (Friday) · `frozen.py` checks Monday's frozen inputs before every later day.
