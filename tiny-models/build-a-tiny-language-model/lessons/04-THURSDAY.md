
# Thursday: Save the experiment, then prove it resumes

A checkpoint is a save game for the experiment, not just a bag of weights.
Today turns the short learning loop into a run you can interrupt and continue.
Weights are part of a checkpoint, but training also has an optimizer, a current
step, a learning-rate schedule and a sampler that chooses the next windows.
Saving weights alone cannot continue the same experiment.

## Follow the trainer's inputs

Open `train.py`. Before training, it checks Monday's frozen data and tokenizer.
It encodes each split once and caches token IDs in NumPy files, using the split
hash in the cache filename. Later runs reuse the cache instead of repeatedly
encoding the stories. Token IDs are stored compactly and converted to the
integer type expected by the embedding when a batch is assembled.

Validation uses 64 fixed windows from the held-out split. They are chosen with
a separate seeded generator, so evaluating does not consume training-sampler
draws. Validation makes no gradient updates. This run reports validation at
regular intervals and keeps the fixed final step, rather than selecting a
checkpoint by the lowest observed validation loss.

This is a sampled validation score, not a loss over every token in the file.
The final scoreboard must compare baselines on this same sample or clearly
label a separate whole-validation evaluation.

## What belongs in the save file?

The trainer's checkpoint contains:

- Model weights and AdamW optimizer state.
- The number of completed updates and the run configuration, including the
  total schedule length, schedule identifier and evaluation interval.
- The training-window generator's state and PyTorch's CPU random-number state.
- The hashes of the training text, validation text and tokenizer.

AdamW’s moving averages are the optimizer’s memory of previous gradients.
Reloading weights without that memory preserves the model’s current predictions
but changes its next update. The window generator matters for a different
reason: different windows mean a different training trajectory.

This supplied model has no dropout; its attention call uses dropout probability
zero. Training-window randomness comes from the saved CPU generator. If you add
stochastic operations on a GPU, revisit the checkpoint’s device RNG state too.
This claim belongs to the supplied code, not every transformer trainer.

The next training step's learning rate can be recomputed from the stored
configuration and completed step. This sampler has no sequential data cursor;
its generator state determines the next random windows.

A checkpoint is written to a temporary file and atomically renamed. Until
that rename, the previous checkpoint remains in place. This prevents a
partially written replacement from becoming the file that resume loads.

## Try a small stop and restart

Use a separate output folder for this exercise. If you repeat the exercise,
choose a new `--out` folder; the trainer refuses to overwrite an existing run:

```bash
uv run python train.py --out runs/lesson-resume --device cpu --batch 8 --steps 120 --eval-every 20 --stop-after 50
uv run python train.py --out runs/lesson-resume --device cpu --batch 8 --steps 120 --eval-every 20 --resume
```

The first command stops after update 50 and saves a checkpoint even though 50
is not an evaluation step. The second should start at update 51. Keep the same
settings on both commands. Changing the total number of steps changes the
schedule and is intentionally refused when resuming.

This is a cooperative stop. An abrupt interruption may leave only the latest
periodic checkpoint, so work done since that checkpoint can be lost. We do not
promise an exact last-step save when a process or machine is killed.

## Compare with an uninterrupted control

```bash
uv run python day4_resume_check.py
```

This command uses the available training device. To require CPU, add
`--device cpu`. It copies data into a temporary workspace, leaving your real
frozen manifest untouched even when it plants a bad hash for a refusal test.

The diagnostic compares three 120-update runs: uninterrupted; stop at 50 and
resume; and a simulated non-finite loss at 30 followed by emergency resume.
At each compared evaluation it checks the chosen training windows and the
logged losses. It also checks that changed settings and inputs are refused.

The selected windows must match exactly. Losses use a tolerance of 0.0001 on
CPU and 0.05 on GPU. In our testing on the Air’s CPU, both resume paths
matched windows and losses within 0.0001. The corresponding GPU check uses
the wider tolerance; it does not promise bit-identical GPU arithmetic.

> **What this proves:** the saved state continues the sampled training trajectory
> on the checked device, and changed settings or inputs are refused.
> **What it doesn’t prove:** that an abrupt power cut saves the last update, or
> that this test establishes behavior on an untested GPU backend.

## Read the logs and status

`log.jsonl` records start, evaluation and final events. Evaluation rows include
step, learning rate, train loss, validation loss and token throughput. A stopped
run has a checkpoint but is not a completed run. Friday's successful full run
writes `final.pt` and a final event.

The non-finite-loss guard stops before accepting the failed update. Its
emergency checkpoint is distinct from the regular checkpoint. It stores the
sampler state from before the failed batch, so a restart retries that batch
rather than skipping it. Use `--resume-from` to select the emergency file,
with the same run settings and without the simulation flag:

```bash
uv run python train.py --out runs/lesson-resume --device cpu --batch 8 --steps 120 --eval-every 20 --resume-from runs/lesson-resume/emergency.pt
```

Use that command only if this output folder actually has an emergency
checkpoint; the cooperative-stop example produces `checkpoint.pt` instead.
The automatic diagnostic exercises emergency resume in its own temporary run.
Do not mistake the presence of a save file for a successful training result.

At the end of today, the required result is a demonstrated resume on the
reference device and a refusal of mismatched inputs, not just a file that loads.
