
# Friday: Read what your saved model writes

Now we wake the tiny thing up and let it tell stories. You have data, a tokenizer, a model, a learning step and a checkpoint
protocol. Today you will run the supplied training budget and load the saved
result in a fresh process. The finish line is text you can read, not a progress
bar that reached the end.

## Check the project before launching

Monday's inputs must still match their frozen hashes. Tuesday's forward-pass
and causal checks must pass. Wednesday's fixed-batch test must learn. Thursday's
resume check must pass on the device you will use. Fix a failure before launching
the larger run; it is cheaper to discover a broken target shift on one batch
than after training has finished.

Run the combined sanity entry point:

```bash
uv run python day5_preflight.py
```

It checks frozen inputs, the environment, Tuesday's model checks, Wednesday's
fixed-batch learning and Thursday's resume paths, stopping at the first failure.
It does not require the tokenizer CLI again: only Monday's tokenizer training
uses that executable.

## Train the reference recipe

Use a new output folder so that an earlier experiment is not overwritten:

```bash
uv run python train.py --out runs/friday
```

The defaults are 6,000 updates, 32 windows per update and 256 tokens per
window. That is 49,152,000 token presentations. The windows are randomly
sampled from roughly ten million training tokens and can overlap. This is
not five complete passes through every training token.

For a smaller CPU run:

```bash
uv run python train.py --out runs/friday-cpu --device cpu --cpu-short
```

This option uses eight windows per update and at most 1,000 updates. It
demonstrates the same training and checkpoint workflow with a smaller budget;
it need not produce stories as good as the reference run's.

We observed 12.45 minutes for this new trainer's full reference run on the
M5 Air, with reported throughput around 65 thousand tokens per second.
That is one machine's observation. Installation, reading and debugging are
separate from training time. The CPU exercises in earlier lessons demonstrate
the loop and recovery; we are not quoting a full CPU training-time estimate
for this revised recipe.

If you deliberately stop at a saved step, resume with the same settings and
`--resume`. If the machine is interrupted abruptly, recovery starts at the
latest usable checkpoint and may repeat work since that checkpoint. A final
event and `final.pt` mark completion; a checkpoint alone does not.

## Compare the scoreboard on equal terms

Compare the trained model with an untrained model and a unigram baseline.
The unigram baseline guesses from training-token frequencies without using
the preceding context. It is a useful answer to “did this learn more than
which tokens are common?”

All three scores need the same tokenizer and evaluation targets. During
training, the 64 fixed validation windows ended at cross-entropy 1.7833.
That is a sampled score. The scoreboard below evaluates every target in the
validation file, so its 1.7770 is a separate measurement, not a relabeled
training-log value.

Run the separate scoreboard after training:

```bash
uv run python day5_scoreboard.py --checkpoint runs/friday/final.pt
```

For a CPU short-path checkpoint, use:

```bash
uv run python day5_scoreboard.py --checkpoint runs/friday-cpu/final.pt --short --device cpu
```

The separate scoreboard recomputes all three scores on every validation token
after the first, using full windows plus the shorter final window. All three
cover 529,455 target tokens exactly once. Our checked output is:

```text
scored targets: 529,455 (every validation token after the first, the same set for all three)
untrained 8.3802   perplexity 4,360.0   (uniform guessing: ln 4096 = 8.3178)
unigram   5.8673   perplexity   353.3   (token-frequency guessing, no context)
trained   1.7770   perplexity     5.9   (70% lower cross-entropy than unigram)
rule: trained <= 0.70 x unigram = 4.1071
rule: pass
```

The criterion is a tutorial smoke check: the full run must reduce cross-entropy
by at least 30% relative to the unigram baseline. The scoreboard computes
`0.70 × unigram` on the current targets, which gives 4.1071 here. It is a
rule in the recipe, not a threshold selected from this model’s final score,
and it does not define a useful assistant or a good storyteller.

For this revised recipe, the rule was recorded before the reference run.
A short-path run reports its scores without applying the criterion. Both
modes reject non-finite weights or scores.

> **What this proves:** the trained model predicts these held-out next tokens
> better than guessing from training-token frequencies.
> **What it doesn’t prove:** reliable facts or coherent plots. The 70% figure
> describes lower cross-entropy here, not 70% better story quality.

The scoreboard and generator default to `runs/main/final.pt`, so commands here
pass the checkpoint path explicitly to match our chosen `runs/friday` folder.

## Load the file in a new process

End the training process before generating. The generation entry point must
load the saved weights with the architecture and tokenizer that produced them.
This proves you can use the checkpoint independently of the live training
process and its in-memory model.

```bash
uv run python generate.py --checkpoint runs/friday/final.pt
```

Defaults are five supplied story openings, seed 1234, temperature 0.8,
top-k 40 and at most 120 new tokens per opening. The script stops early if
it samples the end-of-text token. Our fresh CPU process loaded the full Air
checkpoint and printed all five samples successfully. This verifies CPU
generation from that checkpoint, not CPU training performance.

Generate five fixed prompts using stated seeds and settings, and show all five
results. Ask a small set of questions: is the text on topic, is it grammatical,
and does it make sense? A readable sentence can still be factually wrong. A low
validation loss is not a guarantee that each generation will pass these checks.

## Turn the sampling knobs

Temperature changes the distribution used to choose the next token. Lower
values favor the highest-scoring tokens more strongly; higher values spread
probability more widely. Top-k limits the candidates to the k highest-scoring
tokens. Neither setting repairs a model that did not learn.

Use the same saved checkpoint and prompt for a controlled comparison, changing
one sampling setting at a time. Keep the displayed settings beside the text.
The generated examples, including repetitions and misses, belong in the final
article rather than a selected best-only sample.

By Friday's finish you should have a trained checkpoint, a fresh-process
generation run and a page of stories. Next week can ask how well the model
generalizes and whether it can learn a bounded job. This week establishes the
whole path from text to a model you can save, reload and hear from.

## All five samples from the reference checkpoint

These were loaded in a fresh CPU process from the full Air checkpoint, with
seed 1234, temperature 0.8, top-k 40 and a 120-new-token limit. The samples
are shown together, including the failures and the budget-truncated ending.

### Once upon a time

```text
Once upon a time, there was a little girl named Lucy. She had a big, soft bear named Mr. Bear. They were good friends. Lucy loved Mr. Bear because she liked him a lot.
One day, Lucy and Mr. Bear went to a party. They told their friend, "Mom, I am going to celebrate the party!" Lucy and her bear laughed. Lucy was very happy. She said, "We will celebrating for your friends to celebrate!" From that day on, Lucy and Mr. Bear were the best of friends.
```

### Lily wanted to

```text
Lily wanted to use the red hat to make a hat for her mom. She put the hat on her shelf and started to look for the hat. She looked for the hat, but she could not find it either. She felt very sad.
Just then, her mom came into the room and saw the hat. She said, "What's wrong, Lily?" Lily told her mom about how she made her coat. Her mom helped her get the hat down from the shelf. Lily was so happy! She put on the hat and went to sleep with her mom. She was very good at her new hat
```

### The big dog

```text
The big dog was nice. He wagged his tail and licked Lily's face. Lily laughed and said, "Thank you, Max. You are a good friend. Let's play!" Max wagged his tail and barked happily. They played hide and seek all day long.
```

### One day, Tom found

```text
One day, Tom found a big box under the bed. He was very excited and wanted to see what was inside.
Tom opened the box and saw a big cake on the table. He wanted to eat the cake, but his mom said, "No, Tom! You can't eat your cake." Tom was sad and did not understand what his mom said. He wanted to see what he had to eat. So, he went to his room and started to eat the cake.
When Tom's mom came back, she was very angry. She told him, "Tom, you know you ate a big cake
```

### In the garden, there was

```text
In the garden, there was a little girl named Sue. Sue loved to paint. She had many colors and colorful colors. One day, Sue saw her friend Tom playing with a ball. Sue wanted to play with Tom's football too.
Sue said to Tom, "Do you want to play with me?" Tom did not know. He said, "I don't know. Let's play together." They played with the ball and had lots of fun. They were very happy.
But then, a big wind came. It took Sue and Tom's football. Sue did not want to share. She started
```

Lucy and her bear stay on topic, but “We will celebrating” breaks the grammar.
Lily’s hat becomes a coat mid-explanation. The dog story is a readable short
scene. Tom opens a box under the bed and finds a cake on a table; the scene
does not maintain its spatial setup. Sue’s story repeats “colors” and ends
mid-action. Lily, Tom and Sue all reach the 120-new-token budget; the other
two stop at the end-of-text token. We do not hide those incomplete endings.

Next-token cross-entropy rewards predictive accuracy at each token; it does not
directly enforce character identity, plot consistency or world knowledge. The
model learned the shape and language of simple stories; it has not acquired
reliable world knowledge or consistently controlled their logic.

## Your turn: let the gremlin loose

Keep the five fixed samples above as the evaluation record. Now change the
prompt: ask for dragons, spaceships or a caffeinated duck:

```bash
uv run python generate.py --checkpoint runs/friday/final.pt --prompt "The caffeinated duck borrowed a spaceship"
```

This is exploration,
so have fun. Keep the checkpoint fixed and change one sampling knob at a time
if you want to learn which choice caused the difference. Your funniest story
does not replace the fixed samples; it gets its own page.
