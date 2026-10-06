
# Day 3: Make a guess less wrong

[Day 1](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/day-1.md) · [Day 2](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/day-2.md) · **Day 3** · [Day 4](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/day-4.md) · [Day 5](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/day-5.md)

> **Today’s question:** Can the weights actually learn?

> **Time:** allow 20–30 minutes to read and explore, plus the short training runs.
> **You need:** Day 2’s model code and Day 1’s frozen inputs.
> **You’ll have:** a learning curve, a memorized batch, a rough story and a debugging snapshot.

Today the gremlin gets its first lesson. In Day 2 the model returned scores for the next token. Today those scores
will become a loss, and the loss will guide updates to the weights. We will
first check whether the model can learn one batch, then train on changing
windows from the stories. Each run starts with fresh random weights. The script checks Day 1’s fingerprints first so it knows which text it is learning from.

## Read one training step

Open `day3_learn.py` and follow the loop. It takes input windows and their
one-token-shifted targets, runs the model and calculates cross-entropy. This
loss is smaller when the model assigns more probability to the actual target.
The code passes logits to cross-entropy directly; you do not need to apply
softmax first.

The update has a few separate jobs:

1. Check that the loss is finite. NaN ("not a number") or infinity means this step cannot be
   treated as a successful update.
2. Clear old gradients, the nudges calculated for the weights, with `zero_grad`.
   PyTorch adds each new gradient to
   the old one, which lets you combine small batches. Here we want a fresh
   update, rather than accidentally carrying every previous batch along.
3. Call `loss.backward()` to compute gradients: how changes in the weights
   would affect this loss.
4. Clip the gradient norm at 1.0. The gradient is one proposed nudge per
   weight; its norm measures the length of the whole nudge. If it is longer
   than 1.0, scale it down so a strange batch cannot kick the weights too far.
5. Call the AdamW optimizer's `step()` to update weights.

The learning rate controls the size of each update. AdamW also remembers
recent gradients and uses that memory to adjust the updates. Our recipe sets
the peak learning rate to 0.001, its two memory settings (betas) to 0.9 and
0.95, and weight decay to 0.1. Weight decay gently shrinks the weights during
updates. Keep these recipe settings fixed for the first run. Focus on what
loss, gradients and learning rate do; you can investigate the optimizer’s
memory settings later.

## Can it learn one batch?

```bash
uv run python day3_learn.py --overfit
```

The script selects four training windows and repeats that same batch for 100
updates with a constant learning rate. This deliberately measures memorization.
Overfitting usually means memorizing examples without learning to handle
unseen ones. Here memorization is the test: if the loop cannot learn even one
batch, fix it before spending time on a larger run. Reaching 0.005 does not
mean it learned to write stories.

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

Before you run it, guess: will loss fall on every update when each batch
contains different windows? Or can it rise while the overall trend falls?

Choose one command. On an Apple-silicon Mac the plain command uses the GPU
and our run ended near 3.37. The CPU short command ended near 3.92. Two
recipes, two numbers; the chart shows the GPU run.

On an Apple-silicon Mac, use:

```bash
uv run python day3_learn.py
```

On Linux, or for a smaller batch on another machine, use this instead:

```bash
uv run python day3_learn.py --device cpu --cpu-short
```

Both run 300 updates, but use different batch sizes:

| Exercise | Updates | Batch | Learning rate | Purpose |
|---|---:|---|---|---|
| Memorize one batch | 100 | 4 fixed windows | Constant 0.001 | Check that updates can learn |
| Short training run | 300 | 32 changing windows; 8 on CPU | Warm up, then decay | Learn from changing story windows |

At the start, the weights are random and the optimizer has little history.
Small steps avoid a huge early correction. Then we make larger updates and
ease off toward the end: start small, grow, ease off. This is the
**learning-rate schedule**, the update size planned for each step. Both this loop and Day 5’s
trainer use `schedule.py`: linear warmup for 200 updates, then cosine decay to
one tenth of the peak. The ramp reaches 0.001 at update 200. The remaining
100 updates of this exercise bring it down smoothly:

```text
step 1: 5.00e-06  step 100: 5.00e-04  step 200: 1.00e-03  step 250: 5.50e-04  step 300: 1.00e-04
```

![Two distinct charts: training loss on the Air GPU falls from 8.379 to 3.372 over 300 steps; learning rate warms to 0.001 at step 200, then decays to 0.0001 at step 300.](https://raw.githubusercontent.com/sourceblender/tutorials/main/tiny-models/build-a-tiny-language-model/assets/day3-loss-and-schedule.png)

Left: every update of the Air GPU run, including a labelled rise near the end. Right: the learning-rate schedule, with its own axis.

The loss rose on 110 of the GPU run’s 299 transitions; the overall curve still fell. Some batches are harder than others. A rising step is a reason to inspect the trend, rather than immediately declaring the loop broken. Our CPU run went from 8.364 to 3.916 overall.

Here is a friendlier scale for the same information: **perplexity** is
`exp(loss)`. A loss of 3.916 gives perplexity about 50.2, compared with 4,096
for uniform guessing. You can think of 50.2 as the effective uncertainty of
an even choice among about 50 tokens, averaged over these targets. Today it
falls from thousands toward tens. Day 5’s full run reaches single digits.

The script prints perplexity beside loss and writes the planned learning
rates to `runs/lr_schedule.csv`. The last training batch is practice data;
Day 4 will also score stories the model has not trained on.

The end-of-run criterion is loss below ln(4096) - 1, about 7.32. It checks a
clear move below uniform guessing. Our CPU short process took 58.55 seconds
on the M5 Air, including startup and encoding; the loop reported 47.4 seconds.
Use the shorter batch if your machine needs more breathing room.

## What if the steps are too big?

Optional: on the Apple-silicon reference path, try a separate short run with
a larger peak learning rate. If you want to keep your first curve, copy
`runs/day3_loss.csv` and `runs/lr_schedule.csv` first: each run replaces them.

```bash
uv run python day3_learn.py --lr 0.1
```

On the Air GPU this did not explode. It improved faster at first, then
struggled near the peak and finished at loss 4.681, worse than the recipe’s
3.372. Its opening was “Once upon a time. Tom.” Bigger steps changed the
learning, but not in the way we wanted.

We also tried 0.01 and got 2.933 on this short run. The supplied 0.001 is a
recipe we have not tuned. We haven’t tested whether the larger rate also
helps the full 6,000-update run.

![Training loss over 300 Air GPU updates with three peak learning rates. The default 0.001 ends at 3.372, 0.1 at 4.681 and 0.01 at 2.933.](https://raw.githubusercontent.com/sourceblender/tutorials/main/tiny-models/build-a-tiny-language-model/assets/day3-learning-rates.png)

## When learning goes wrong

The supplied loop checks loss before backward and the optimizer update. On
a non-finite loss it stops and writes `runs/emergency.pt` containing the
weights and last completed step for inspection. If loss fails at step 30,
the checkpoint records 29 completed updates. Treat it as a debugging snapshot.
Day 4 adds the extra state needed to resume training.

Plant a NaN in a separate test run. The next command should stop with exit code 1:

```bash
uv run python day3_learn.py --device cpu --cpu-short --steps 6 --simulate-nan-at 3
```

The intentional failure exits with code 1 and saves an inspection artifact
after two completed updates. The gremlin has tripped over a banana peel we
put there on purpose. Remove the simulation flag for your real run.

## Hear its first attempt

The short training command ends by asking the model to continue “Once upon
a time”. It writes while the trained weights are still loaded; no normal
checkpoint is saved. This is the GPU run after 300 updates, with final training-batch loss
3.372:

```text
Once upon a time, there was a little girl named Sam. Ben liked to play with her dad. Then, Lily was playing near a big dog named Amy.
While Lily came to a lot of fun to the pond and saw the park. But they were playing with it was playing with it on, so happy
```

The CPU short run, with training-batch loss 3.916, wrote:

```text
Once upon a time, there was a little girl named Sam. Tim. Spot and he was a little bug was walking for a big tree, there was a little man with all day with a lot of fun.
```

Names wander and the grammar gets tangled. But it has started writing
story-shaped language. The gremlin has a voice; it just needs practice.
The samples give those training-batch scores a voice. Next we’ll score
stories the model hasn’t learned from.

By today's finish, you have seen weights learn both a fixed batch and changing
story windows. In Day 4 we give the gremlin a save game and measure it on held-out stories.
A closed laptop lid should not send the whole experiment back to the beginning.

## If you get stuck

- **Fixed-batch loss stays high:** inspect the input/target shift, logits-to-loss call and optimizer update before increasing the budget.
- **MPS or CUDA error:** retry the short exercise with --device cpu --cpu-short.
- **NaN exit:** the deliberate simulation should exit 1. For an unexpected NaN, keep emergency.pt for inspection and investigate the last finite steps.

> **What this proves:** the fixed batch can be memorized, and the short run
> improves next-token prediction on training batches.
> **What it doesn’t prove:** generalization or story quality. Day 4 adds
> held-out validation; today’s rough sample is a first look, not a quality benchmark.

[Day 1](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/day-1.md) · [Day 2](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/day-2.md) · **Day 3** · [Day 4](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/day-4.md) · [Day 5](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/day-5.md)
