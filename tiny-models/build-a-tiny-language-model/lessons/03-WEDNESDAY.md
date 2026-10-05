
# Wednesday: Make a guess less wrong

Today the ignorance gets gradients. Yesterday the model returned scores for the next token. Today those scores
will become a loss, and the loss will guide updates to the weights. We will
first check whether the model can learn one batch, then train on changing
windows from the stories.

## Read one training step

Open `day3_learn.py` and follow the loop. It takes input windows and their
one-token-shifted targets, runs the model and calculates cross-entropy. This
loss is smaller when the model assigns more probability to the actual target.
The code passes logits to cross-entropy directly; you do not need to apply
softmax first.

One way to interpret this number is perplexity: `exp(loss)`, using natural-log
cross-entropy. Uniform guessing among 4,096 tokens has loss `ln(4096)` and
perplexity 4,096. A loss of 3.97 corresponds to perplexity about 53. Think of
this as an effective uncertainty over next-token choices, not 53 literal
options at every position or a measure of story intelligence.

The update has a few separate jobs:

1. Check that the loss is finite. NaN or infinity means this step cannot be
   treated as a successful update.
2. Clear accumulated gradients with `zero_grad`.
3. Call `loss.backward()` to compute gradients: how changes in the weights
   would affect this loss.
4. Clip the gradient norm at 1.0 to limit unusually large gradients.
5. Call the AdamW optimizer's `step()` to update weights.

This recipe uses AdamW with a base learning rate of 0.001, betas 0.9 and 0.95,
and weight decay 0.1. These are the supplied starting settings, not universally
best values for language models.

## Can it learn one batch?

```bash
uv run python day3_learn.py --overfit
```

The script selects four training windows and repeats that same batch for 100
updates with a constant learning rate. This deliberately measures memorization.
If this small exercise cannot learn, investigate the loop before spending time
on a larger run. It is not a test of unseen stories.

Our CPU check on the Air produced:

```text
step    1  loss 8.379  perplexity 4,355.7
step   25  loss 3.568  perplexity 35.4
step   50  loss 0.391  perplexity 1.5
step   75  loss 0.015  perplexity 1.0
step  100  loss 0.005  perplexity 1.0
Wednesday check passes
```

The implemented criterion is loss below 1.0 after 100 updates. Your device may
give slightly different values; the entire printed curve need not match.
In our testing, removing the optimizer update made this exercise fail rather
than reporting that it learned.

## Train on changing windows

For the reference path:

```bash
uv run python day3_learn.py
```

For a smaller CPU batch:

```bash
uv run python day3_learn.py --device cpu --cpu-short
```

Both run 300 updates. The reference batch contains 32 windows of 256 tokens;
the CPU short option uses eight. `--device auto` chooses the available device,
while the explicit CPU flag makes the second command use CPU even on a Mac
with an available GPU. The script uses bfloat16 where its capability probe
succeeds and otherwise uses float32.

The learning rate changes during this exercise. Both this loop and Friday’s
trainer use `schedule.py`: linear warmup for 200 updates, then cosine decay to
one tenth of the peak. The ramp reaches 0.001 at update 200. The remaining
100 updates of this exercise bring it down smoothly:

```text
step 1: 5.00e-06  step 100: 5.00e-04  step 200: 1.00e-03  step 250: 5.50e-04  step 300: 1.00e-04
```

The phases happen in order; decay begins after warmup. A diagnostic shorter
than 200 updates stays entirely in warmup and never reaches the peak. Changing
`--steps` changes the decay length, so a shorter run is a different schedule,
not simply a prefix of the longer one. The script writes every scheduled
value to `runs/lr_schedule.csv` for inspection and plotting.

Our 300-update CPU short run moved from loss 8.364 to 3.916. Individual batches
produced occasional increases; random windows do not have equal difficulty.
Its final perplexity was about 50.2. Training cross-entropy on the last batch
is not a held-out validation score.

The end-of-run criterion is loss below ln(4096) - 1, about 7.32. It checks a
clear move below uniform guessing. Our CPU short process took 58.55 seconds
on the M5 Air, including startup and encoding; the loop reported 47.4 seconds.
Those are observations on this machine, not a typical-laptop budget.

> **What this proves:** the fixed batch can be memorized, and the short run
> improves next-token prediction on training batches.
> **What it doesn’t prove:** generalization or story quality. Thursday adds
> held-out validation; Friday lets you read the saved model’s output.

## When learning goes wrong

The supplied loop checks loss before backward and the optimizer update. On
a non-finite loss it stops and writes `runs/emergency.pt` containing the
weights and last completed step for inspection. If loss fails at step 30,
the checkpoint records 29 completed updates. That file is not a successful model
or a full resumable checkpoint. Thursday supplies the complete checkpoint.

You can exercise the guard in a separate test run:

```bash
uv run python day3_learn.py --device cpu --cpu-short --steps 6 --simulate-nan-at 3
```

The intentional failure exits with code 1 and saves an inspection artifact
after two completed updates. Do not use the simulation flag on a run you want
to complete, or describe the resulting file as known-good trained weights.
This entry point now checks Monday's frozen inputs before learning begins.

By today's finish, you have seen weights learn both a fixed batch and changing
story windows. Tomorrow you will make a longer run resumable and measure it
on the held-out split.
