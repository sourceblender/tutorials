
# Day 3: Make a guess less wrong

[Day 1](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/01-MONDAY.md) · [Day 2](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/02-TUESDAY.md) · **Day 3** · [Day 4](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/04-THURSDAY.md) · [Day 5](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/05-FRIDAY.md)

> **Today’s question:** Can the weights actually learn?

> **Plan:** about an hour.
> **Bring:** Day 2’s working model and Day 1’s frozen inputs.
> **Finish with:** a learning curve, a memorized batch and a debugging snapshot.

Today the gremlin gets its first lesson. In Day 2 the model returned scores for the next token. Today those scores
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
perplexity 4,096. Our short run’s loss of 3.916 corresponds to perplexity 50.2. Think of
this as effective uncertainty about the next token. Lower means the model
is less surprised by the targets. It is a summary of uncertainty, rather
than a literal count of options at each position.

The update has a few separate jobs:

1. Check that the loss is finite. NaN or infinity means this step cannot be
   treated as a successful update.
2. Clear accumulated gradients with `zero_grad`.
3. Call `loss.backward()` to compute gradients: how changes in the weights
   would affect this loss.
4. Clip the gradient norm at 1.0 to limit unusually large gradients.
5. Call the AdamW optimizer's `step()` to update weights.

The learning rate controls the size of each update. AdamW also remembers
recent gradients and uses that memory to adjust the updates. Our recipe sets
the peak learning rate to 0.001, its two memory settings (betas) to 0.9 and
0.95, and weight decay to 0.1. Weight decay gently shrinks the weights during
updates. Keep these settings fixed while exploring the loop.

## Can it learn one batch?

```bash
uv run python day3_learn.py --overfit
```

The script selects four training windows and repeats that same batch for 100
updates with a constant learning rate. This deliberately measures memorization.
If this small exercise cannot learn, investigate the loop before spending time
on a larger run.

Our CPU check on the Air produced:

```text
step    1  loss 8.379  perplexity 4,355.7
step   25  loss 3.568  perplexity 35.4
step   50  loss 0.391  perplexity 1.5
step   75  loss 0.015  perplexity 1.0
step  100  loss 0.005  perplexity 1.0
Day 3 check passes
```

Look for loss below 1.0 after 100 updates. Small differences in the printed
curve are normal.
In our testing, removing the optimizer update made this exercise fail rather
than reporting that it learned.

## Train on changing windows

Predict first: will the gremlin’s loss fall at every printed step when the
training windows change? Look for a bump as well as a downward trend.

Choose one of the next two commands. On an Apple-silicon Mac, use:

```bash
uv run python day3_learn.py
```

On Linux, or for a smaller batch on another machine, use this instead:

```bash
uv run python day3_learn.py --device cpu --cpu-short
```

Both run 300 updates. The reference batch contains 32 windows of 256 tokens;
the CPU short option uses eight. `--device auto` chooses the available device,
while the explicit CPU flag makes the second command use CPU even on a Mac
with an available GPU. The script uses bfloat16 where its capability probe
succeeds and otherwise uses float32.

We start with small updates so the random weights can settle in, then
make larger updates while learning, and ease off toward the end. That is
what the learning-rate schedule controls. Both this loop and Day 5’s
trainer use `schedule.py`: linear warmup for 200 updates, then cosine decay to
one tenth of the peak. The ramp reaches 0.001 at update 200. The remaining
100 updates of this exercise bring it down smoothly:

```text
step 1: 5.00e-06  step 100: 5.00e-04  step 200: 1.00e-03  step 250: 5.50e-04  step 300: 1.00e-04
```

![Two distinct charts: training loss on the Air GPU falls from 8.379 to 3.372 over 300 steps; learning rate warms to 0.001 at step 200, then decays to 0.0001 at step 300.](https://raw.githubusercontent.com/sourceblender/tutorials/main/tiny-models/build-a-tiny-language-model/assets/day3-loss-and-schedule.png)

Left: training-batch loss from the Air GPU run. The CPU example below ends
at 3.916. Right: the learning-rate schedule, with its own axis.

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
Use the shorter batch if your machine needs more breathing room.

## When learning goes wrong

The supplied loop checks loss before backward and the optimizer update. On
a non-finite loss it stops and writes `runs/emergency.pt` containing the
weights and last completed step for inspection. If loss fails at step 30,
the checkpoint records 29 completed updates. Treat it as a debugging snapshot.
Day 4 adds the extra state needed to resume training.

You can exercise the guard in a separate test run:

```bash
uv run python day3_learn.py --device cpu --cpu-short --steps 6 --simulate-nan-at 3
```

The intentional failure exits with code 1 and saves an inspection artifact
after two completed updates. The gremlin has tripped over a banana peel we
put there on purpose. Remove the simulation flag for your real run.
This entry point checks Day 1's frozen inputs before learning begins.

By today's finish, you have seen weights learn both a fixed batch and changing
story windows. In Day 4 you will make a longer run resumable and measure it
on the held-out split.

## If you get stuck

- **Fixed-batch loss stays high:** inspect the input/target shift, logits-to-loss call and optimizer update before increasing the budget.
- **MPS or CUDA error:** retry the short exercise with --device cpu --cpu-short.
- **NaN exit:** the deliberate simulation should exit 1. For an unexpected NaN, keep emergency.pt for inspection and investigate the last finite steps.

> **What this proves:** the fixed batch can be memorized, and the short run
> improves next-token prediction on training batches.
> **What it doesn’t prove:** generalization or story quality. Day 4 adds
> held-out validation; Day 5 lets you read the saved model’s output.

[Day 1](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/01-MONDAY.md) · [Day 2](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/02-TUESDAY.md) · **Day 3** · [Day 4](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/04-THURSDAY.md) · [Day 5](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/05-FRIDAY.md)
