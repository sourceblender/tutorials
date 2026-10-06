
# Day 4: Save the Experiment

[Day 1](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/day-1.md) · [Day 2](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/day-2.md) · [Day 3](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/day-3.md) · **Day 4** · [Day 5](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/day-5.md)

A checkpoint is a save game for the experiment, not just a bag of weights.
Today the gremlin gets that save game, and we test whether loading it really
continues where it left off.

Our full run takes minutes. Larger runs take hours or days, laptops sleep,
and sometimes someone else needs the GPU. A working save means you can stop
without sending the experiment back to the beginning.

> **Today’s question:** Can we stop and continue the same experiment?

> **Time:** allow 20–30 minutes to read and explore, plus the stop/resume checks.
> **You need:** the model and learning code, plus Day 1’s frozen inputs.
> **You’ll have:** a saved experiment that continues the same sequence of training windows.

Today starts a fresh model. Day 3’s story came from the practice model in
memory; we are not continuing those learned weights. These exercises, and
Day 5’s final run, each start from random weights. We carry the project
forward through the week, rather than one trained model.

Before starting, jot down your guess: what must a save remember besides the
model’s weights? We’ll open the save after trying the restart.

## Stop at 50, continue at 51

Run these commands from the same project folder as the other days. Choose an
unused output folder; if you repeat the exercise, give `--out` another name.

```bash
uv run python train.py --out runs/lesson-resume --device cpu --batch 8 --steps 120 --eval-every 20 --stop-after 50
```

An Air CPU run printed the following. The `tok/s` column is speed, not a pass criterion; yours will vary:

```text
training on cpu: steps 1-120, batch 8 x 256
step    20  train 7.604  val 7.591  17,958 tok/s
step    40  train 6.801  val 6.783  18,877 tok/s
stopped after step 50 (checkpoint saved); resume with --resume
```

The first run trains on CPU, evaluates every 20 updates and stops at 50.
It writes a checkpoint at the stop even though 50 is between evaluations.
Watch for the train and val columns. **Train loss** scores a batch the model
learns from; **validation loss** scores reserved stories. Improving on those
stories is more useful than merely memorizing the practice batch.

Continue with the same settings:

```bash
uv run python train.py --out runs/lesson-resume --device cpu --batch 8 --steps 120 --eval-every 20 --resume
```

The restart printed:

```text
resumed at step 51 from runs/lesson-resume/checkpoint.pt
training on cpu: steps 51-120, batch 8 x 256
step    60  train 5.959  val 5.829  17,691 tok/s
step    80  train 5.121  val 5.204  19,075 tok/s
step   100  train 4.858  val 4.817  19,372 tok/s
step   120  train 4.589  val 4.559  19,454 tok/s
done: runs/lesson-resume/final.pt
```

Day 5’s training chart puts the `train` and `val` curves side by side over
the full run. Today’s rows are a first look at that comparison.

The restart should begin at update 51 and finish at 120. The learning rate
keeps rising throughout this little run: warmup is fixed at 200 updates,
while only the later decay stretches to fit `--steps`. We are testing the
save, so a run entirely within warmup is fine.

If you hit Ctrl-C instead of planning a stop, there is no special final save.
You can resume from the last regular checkpoint, written at each evaluation,
and redo the updates since that checkpoint. Think of a game that reloads your last save.

## Open the save game

How did your guess compare? The checkpoint contains:

- **Weights:** the model’s learned numbers.
- **Optimizer state:** AdamW’s memory of previous gradients. Reloading only
  weights preserves its current predictions but changes the next update.
- **Progress and settings:** completed updates and the learning-rate schedule.
- **Random-number state:** especially the generator that chooses training
  windows. Different windows mean a different next lesson for the gremlin.
- **Input fingerprints:** the training text, validation text and tokenizer.

That window generator is easy to miss. A restart that chooses different
examples has resumed the model, but it has changed the experiment.

The trainer writes a temporary file, then atomically renames it into place.
Until the rename succeeds, the previous save stays intact. A half-written
replacement does not become your next save game.

## Compare with a run that never stopped

Loading a file is a start. Comparing against an uninterrupted control tells
us whether it continued correctly:

```bash
uv run python day4_resume_check.py
```

The diagnostic uses the available training device; add `--device cpu` to
choose CPU. It compares a continuous run, a stop/restart and recovery after
a planted non-finite loss. A **non-finite loss** is an invalid number such as
NaN or infinity, the failure we planted in Day 3.

At each compared evaluation, the selected training windows must match
exactly. The losses must be close. On our CPU they matched to four decimal
places. Parallel GPU calculations can add numbers in different orders,
changing the rounding slightly. The GPU check allows a little more numerical
drift while still requiring the same windows.

![CPU training timelines for 120 updates: uninterrupted and stopped at update 50 then resumed. At each of 12 evaluations, the sampled-window hashes and logged losses match.](https://raw.githubusercontent.com/sourceblender/tutorials/main/tiny-models/build-a-tiny-language-model/assets/day4-resume-timeline.png)

This CPU comparison checks every 10 updates; our manual exercise checks every
20. Each equals sign means both the sampled windows and losses matched.

Look for these last lines, with each refusal check reporting `True`:

```text
resume with changed settings refused: True
resume after the data changed refused: True
Day 4 check passes: resumed runs continued the same run
```

To see a refusal yourself, deliberately change just the total step count:

```bash
uv run python train.py --out runs/lesson-resume --device cpu --batch 8 --steps 200 --eval-every 20 --resume
```

This is an expected exit with code 1:

```text
--resume refused: this run's settings differ from the checkpoint's
```

The diagnostic also tests refusing changed settings and changed data. That
is useful resistance: a save from a different recipe should not quietly pass
as a continuation of this one. It does those checks in a temporary workspace,
leaving your Day 1 files alone.

## What happens before the loop?

Open `train.py` after the experiment. It checks Day 1’s fingerprints, then
encodes the stories once and caches the IDs for later runs. The cache uses
`uint16`, a two-byte integer format: all 4,096 vocabulary IDs fit in it.

Validation uses 64 fixed windows and its own random generator. Checking the
score should not use up the random choices meant for training; that would
change which batch comes next. Day 5’s scoreboard will score the whole
validation file after training, rather than this quick sample.

`log.jsonl` stores the progress records: step, learning rate, train loss,
validation loss and throughput. A stopped run has a checkpoint. A completed
run also has `final.pt`, the weights we’ll use to write stories.

## A real failure needs investigation

The diagnostic already tests recovery from an emergency save; there is no
extra recovery command to run in today’s manual exercise. The planned
stop at update 50 wrote `checkpoint.pt`, not `emergency.pt`.

A real NaN deserves investigation. The emergency save remembers the state
before the failed batch, so resume retries that batch. Retrying may repeat the
same failure. Inspect the file and the recipe; if you change the recipe,
start a separate run. A save preserves evidence, not a cure.

Tomorrow we train the final gremlin, load its saved weights in a fresh process
and let it talk. Today made sure those weights can come from an experiment
that survives an interruption.

## If you get stuck

- **Run folder already exists:** use `--resume` with the same settings, or an unused `--out` folder for a separate experiment.
- **Resume refuses settings or data:** restore the matching recipe; start a separate run if you intended to change it.
- **Device error:** run the isolated resume diagnostic with `--device cpu`.

> **What this proves:** restarting picks the same training windows and keeps
> the losses close. Changing the recipe or data is caught before training resumes. Loss tolerance
> is 0.0001 on CPU and 0.05 on GPU; both require identical sampled windows.
> **What it doesn’t prove:** that an abrupt interruption saves the last update.
> We also tested stopping on the Air’s CPU at update 50 and completing on its
> MPS GPU. That run completed, but cross-device continuation is not a promise
> of identical arithmetic or support for every GPU.

[Day 1](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/day-1.md) · [Day 2](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/day-2.md) · [Day 3](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/day-3.md) · **Day 4** · [Day 5](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/day-5.md)
