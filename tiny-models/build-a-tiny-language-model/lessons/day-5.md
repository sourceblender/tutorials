
# Day 5: Read what your saved model writes

[Day 1](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/01-MONDAY.md) · [Day 2](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/02-TUESDAY.md) · [Day 3](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/03-WEDNESDAY.md) · [Day 4](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/04-THURSDAY.md) · **Day 5**

> **Today’s question:** Did the trained model learn more than token frequency, and can we use the saved result?

> **Plan:** about an hour on the reference GPU path; allow more on CPU.
> **Bring:** the checks from Days 1–4 and the frozen inputs.
> **Finish with:** a trained checkpoint, its scoreboard and five raw stories.

Now we wake the tiny thing up and let it tell stories. You have data, a tokenizer, a model, a learning step and a checkpoint
protocol. Today you will run the supplied training budget and load the saved
result in a fresh process. The payoff is a page of stories written by the model you trained.

## Check the project before launching

Day 1's inputs must still match their frozen hashes. Day 2's forward-pass
and causal checks must pass. Day 3's fixed-batch test must learn. Day 4's
resume check must pass on the device you will use. Fix a failure before launching
the larger run; it is cheaper to discover a broken target shift on one batch
than after training has finished.

Run the combined sanity entry point:

```bash
uv run python day5_preflight.py
```

It checks frozen inputs, the environment, Day 2's model checks, Day 3's
fixed-batch learning and Day 4's resume paths, stopping at the first failure.
It does not require the tokenizer CLI again: only Day 1's tokenizer training
uses that executable.

## Train the reference recipe

Choose one training path and an unused output folder. On an Apple-silicon
Mac, run the full reference recipe:

```bash
uv run python train.py --out runs/friday
```

The defaults are 6,000 updates, 32 windows per update and 256 tokens per
window. That is 49,152,000 token presentations. The windows are randomly
sampled from roughly ten million training tokens and can overlap. Because sampling is random, some tokens are seen more often than others.

On Linux, or for a smaller CPU run on another machine, use this instead:

```bash
uv run python train.py --out runs/friday-cpu --device cpu --cpu-short
```

This option uses eight windows per update and at most 1,000 updates. It
demonstrates the same training and checkpoint workflow with a smaller budget;
expect rougher stories with that smaller budget.

We observed 12.45 minutes for the trainer's full reference run on the
M5 Air, with reported throughput around 65 thousand tokens per second.
On our Ubuntu x86_64 machine, the CPU short trainer took 144 seconds. Its
whole-file score was 3.0865, or perplexity 21.9, compared with the full Air
run’s 1.7770. Both learned; the longer run had more practice.

![Training-batch and sampled-validation cross-entropy through 6000 updates on the Air GPU. The dashed line shows the separate whole-file scoreboard value of 1.7770.](https://raw.githubusercontent.com/sourceblender/tutorials/main/tiny-models/build-a-tiny-language-model/assets/day5-training-curve.png)

The gray line follows the training batch at each evaluation. The blue line
uses the same 64 validation windows each time. The dashed line comes from
the whole-file scoreboard in the next section. Watch the early drop: the
gremlin is finding patterns, though the next batch can still surprise it.

If you deliberately stop at a saved step, resume with the same settings and
`--resume`. If the machine is interrupted abruptly, recovery starts at the
latest usable checkpoint and may repeat work since that checkpoint. A final
event and `final.pt` mark completion; a checkpoint alone does not.

## Compare the scoreboard on equal terms

Compare the trained model with an untrained model and a unigram baseline.
The unigram baseline guesses from training-token frequencies without using
the preceding context. It adds one to the training count of each of the 4,096
vocabulary IDs, normalizes those smoothed counts into probabilities, and
averages the negative log probability of each validation target. It is a useful
answer to “did this learn more than which tokens are common?”

All three scores need the same tokenizer and evaluation targets. During
training, the 64 fixed validation windows ended at cross-entropy 1.7833.
The scoreboard below evaluates every target in the validation file. Its
1.7770 comes from that larger measurement.

Before running the scoreboard, guess the unigram perplexity. How much can
frequency alone improve on random initialization?

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

![Perplexity bars on a logarithmic axis: random initialization 4360.0, unigram 353.3, trained model 5.9. All use the same 529455 validation targets.](https://raw.githubusercontent.com/sourceblender/tutorials/main/tiny-models/build-a-tiny-language-model/assets/day5-perplexity.png)

The vertical axis is logarithmic: each equally spaced step represents the
same multiplying factor. These values come from the whole-file scoreboard above.

The criterion is a tutorial smoke check: the full run must reduce cross-entropy
by at least 30% relative to the unigram baseline. The scoreboard computes
`0.70 × unigram` on the current targets, which gives 4.1071 here. It is a
rule fixed in the recipe before the reference run. It asks the model to beat
a simple baseline by a clear margin.

A short-path run reports its scores without applying the criterion. Both
modes reject non-finite weights or scores.

## Load the file in a new process

Once training finishes, start a separate generation process. It loads the
saved weights with the matching architecture and tokenizer: your model can
write stories from its save file.

```bash
uv run python generate.py --checkpoint runs/friday/final.pt
```

For the CPU short path, use this checkpoint instead:

```bash
uv run python generate.py --checkpoint runs/friday-cpu/final.pt --device cpu
```

The script starts from five supplied story openings. Its random seed is
1234; temperature 0.8 and top-k 40 control how it chooses each next token.
We will explore those knobs after reading the stories. It allows at most
120 new tokens per opening. The script stops early if
it samples the end-of-text token. Our fresh CPU process loaded the full Air
checkpoint and printed all five samples. The saved gremlin is awake.

Keep all five outputs with their settings, including the failures. Ask: is the text on topic, is it grammatical,
and does it make sense? Read the stories as well as the score. The gremlin can learn the rhythm of
a sentence and still lose track of its cake.

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
two stop at the end-of-text token. The unfinished sentences stay in the record.

Next-token cross-entropy rewards predictive accuracy at each token; it does not
directly enforce character identity, plot consistency or world knowledge. The
gremlin has learned story-shaped language. Keeping the cake in the same
room is still a work in progress.

## Turn the sampling knobs

Temperature changes the distribution used to choose the next token. Lower
values favor the highest-scoring tokens more strongly; higher values spread
probability more widely. Top-k limits the candidates to the k highest-scoring
tokens. Neither setting repairs a model that did not learn.

Use the same saved checkpoint and prompt for a controlled comparison, changing
one sampling setting at a time. Keep the displayed settings beside the text.
Keep the repetitions and misses beside the good samples. Together they show
what changing the knob actually did.

By Day 5's finish you should have a trained checkpoint, a fresh-process
generation run and a page of stories. Next week, we can try a different job: sorting requests such as “search the
web” and “make a picture,” and checking when a model should say “unsure.” This week establishes the
whole path from text to a model you can save, reload and hear from.

## Your turn: let the gremlin loose

Your five fixed samples are saved. Time to play: ask for dragons, spaceships
or a caffeinated duck:

```bash
uv run python generate.py --checkpoint runs/friday/final.pt --prompt "The caffeinated duck borrowed a spaceship"
```

For the CPU short path, replace `runs/friday/final.pt` with
`runs/friday-cpu/final.pt` in that command. Try your own opening, then play
with one sampling knob at a time. Give the funniest story its own page.
The duck has waited five lessons for a spaceship.

## If you get stuck

- **Checkpoint missing:** confirm training wrote final.pt, then pass that exact path with --checkpoint.
- **Scoreboard rejects non-finite weights:** preserve the failed file for debugging and return to the training failure; a loadable checkpoint alone is insufficient.
- **Out of memory or device error:** use the CPU short training path and score/generate from runs/friday-cpu/final.pt.

> **What this proves:** the trained model predicts these held-out next tokens
> better than guessing from training-token frequencies.
> **What it doesn’t prove:** reliable facts or coherent plots. The 70% figure
> describes lower cross-entropy here, not 70% better story quality.

[Day 1](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/01-MONDAY.md) · [Day 2](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/02-TUESDAY.md) · [Day 3](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/03-WEDNESDAY.md) · [Day 4](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/04-THURSDAY.md) · **Day 5**
