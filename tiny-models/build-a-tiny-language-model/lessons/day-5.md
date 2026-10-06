
# Day 5: Read what your saved model writes

[Day 1](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/day-1.md) · [Day 2](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/day-2.md) · [Day 3](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/day-3.md) · [Day 4](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/day-4.md) · **Day 5**

Now we wake the tiny thing up and let it tell stories. Today you’ll train the
final gremlin, load its saved weights in a fresh process and read what it
wrote. The payoff is a page of stories from the model you trained.

> **Time:** allow 20–30 minutes to read and explore, plus training. Our full Air run took about 12 minutes; your machine’s time will vary.
> **You need:** the code and checks from Days 1–4, plus Day 1’s frozen inputs.
> **You’ll have:** a trained checkpoint, five stories and a score against a simple baseline.

## Check the project before launching

One command checks that the whole project still works before you spend time
on the full run. Catching a broken target shift on one batch is cheaper than
finding it after training:

```bash
uv run python day5_preflight.py
```

Look for this final line:

```text
preflight passes: launch the full run with  uv run python train.py --out <a new run folder>
```

The preflight combines the earlier data, model, learning and resume checks.
It stops at the first failure. Only Day 1 needs the tokenizer CLI.

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
smaller budget will show up in the stories below. Both paths start fresh; they do not continue Day 4’s test model.

![Training-batch and sampled-validation cross-entropy through 6000 updates on the Air GPU. The dashed line shows the separate whole-file scoreboard value of 1.7770.](https://raw.githubusercontent.com/sourceblender/tutorials/main/tiny-models/build-a-tiny-language-model/assets/day5-training-curve.png)

The gray line follows the training batch at each evaluation. The blue line
uses the same 64 validation windows each time. The dashed line comes from
the whole-file scoreboard in the next section. Watch the early drop: the
gremlin is finding patterns, though the next batch can still surprise it.

If you deliberately stop at a saved step, resume with the same settings and
`--resume`. If the machine is interrupted abruptly, recovery starts at the
latest usable checkpoint and may repeat work since that checkpoint. The run
is finished when it writes `final.pt`; a checkpoint alone does not mark completion.

## Compare the scoreboard on equal terms

Compare the trained model with an untrained model and a unigram baseline.
The unigram baseline guesses from training-token frequencies without using
the preceding context. It adds one to the training count of each of the 4,096
vocabulary IDs, normalizes those smoothed counts into probabilities, and
averages the negative log probability of each validation target. It is a useful
answer to “did this learn more than which tokens are common?”

All three scores need the same tokenizer and evaluation targets. The trainer’s quick score uses 64 fixed validation windows;
the scoreboard uses the whole file. On the full Air run those were 1.7833
and 1.7770 respectively. They are close, but come from different measurements.
The CPU short run’s whole-file score was 3.0865. Use the result for the recipe
you chose.

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
cover 529,455 target tokens exactly once. To pass, the full model has to
score at least 30% lower than the frequency guesser. Ours scored about 70% lower.

Here is the full Air checkpoint’s scoreboard:

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

Keep all five outputs with their settings, including the failures. Ask whether
they stay on topic, make grammatical sentences and keep the story straight.
The gremlin can learn the rhythm of a sentence and still lose track of its cake.

## All five samples from the reference checkpoint

These were loaded in a fresh CPU process from the full Air checkpoint, with
seed 1234, temperature 0.8, top-k 40 and a 120-new-token limit. All five appear below; the same token budget can leave an ending unfinished.

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
two stop at the end-of-text token. Those are token-budget stops, rather than conclusions the model chose.

Training rewarded guessing the next token. Nothing told it to keep names,
plots or rooms straight. The
gremlin has learned story-shaped language. Keeping the cake in the same
room is still a work in progress.

## What the CPU short path sounds like

The same five openings also ran on the Linux CPU checkpoint. Its whole-file
loss was 3.0865, versus 1.7770 for the full Air checkpoint. Here are two of
those CPU outputs with the same seed, temperature, top-k and token budget.
They are rougher, but the rhythm of a story is already there.

### The big dog: CPU short path

```text
The big dog. The cat and the cat did not have a cat. The cat was sad and wanted to help. The dog was sad. The cat had to have the dog get off the cat. The cat and the cat loved the dog, and made the dog. The dog became the cat and became big friends.
```

### Lily wanted to: CPU short path

```text
Lily wanted to share her toys.
The next day, they all played and her toys. One day, Lily went for a long time to the farm. Lily was very happy. Sue had a big tree. She had a lot of pretty stick and liked her ball.
As Mia heard her friends, Sue's tail. She looked at the tree and saw her. Sue said, "Hi, I want to play with me?" Her mom asked her mom, "No, Lily, Tim!" Sue saw the box and Tim was happy. But then, a funny dog flew away. But then, Sue
```

At 3.09 the dog becomes the cat. At 1.78 the cake only changes rooms. That
comparison gives these scores a face; two checkpoints and two training
budgets produced different kinds of mistakes. Your own run will give you
its own page of oddities.

## Turn the sampling knobs

**Temperature** changes how strongly generation favors its highest-scoring
next tokens. Lower values make them more likely; higher values spread the
probability more widely. **Top-k** keeps only the `k` highest-scoring candidates.
These change how the model chooses, rather than teaching it anything new.

The controls are `--temperature` (default 0.8), `--top-k` (40), `--max-tokens`
(120 new tokens), `--prompt` and `--seed` (1234). Start with one knob and the
same opening:

```bash
uv run python generate.py --checkpoint runs/friday/final.pt --prompt "The caffeinated duck borrowed a spaceship." --temperature 0.3 --seed 1234 --device cpu
uv run python generate.py --checkpoint runs/friday/final.pt --prompt "The caffeinated duck borrowed a spaceship." --temperature 1.3 --seed 1234 --device cpu
```

These commands generate on CPU so we can compare the same sampled output.
For the CPU short path, use `runs/friday-cpu/final.pt`. Keep the two outputs side by side. Which is more repetitive? Which wanders
farther from the opening? Let the text answer, rather than assuming the
higher temperature must ruin it.

Our reference checkpoint at temperature 0.3 wrote:

```text
The caffeinated duck borrowed a spaceship.
The duck was so happy and he thanked the curledge. The curled up in the sunshine and smiled.
```

At 1.3, this passage appeared:

```text
Jonny wanted to join them, but he was so clumsy, he shouted out of the hole as he pushed. In it came lots of crazy spinning in shorgzen their games they played a great job together. They were all having lots of fun, but soon the resuccessorted their friendship.
```

“Curledge” is already odd at the lower temperature. At the higher one, the
story wanders farther and invents “shorgzen” and “resuccessorted”. Try it
yourself: other samples can turn out differently. The 0.3 output is complete; the 1.3 passage is
an excerpt from the longer result.

Keep the full stop after “spaceship” for this comparison. Without it, our
model continued the word as “spaceshiping”. It sees token pieces, so it can
finish a word as well as continue a sentence.

To write longer stories, raise `--max-tokens`. Once prompt and generated text
exceed 256 tokens, the model sees only the most recent 256. A longer output
budget does not give it a longer memory.

## Fixing the cake

Want to improve the hat/coat confusion or keep the cake under the bed? The
levers include more training, a bigger model and more data. We have not run
that comparison, so we cannot call a winner. Change one at a time and keep
the same five openings and sampling settings. Then compare what actually
changed in the stories as well as the score.

You can also search the training text for a sample’s oddest phrase. Try “very
good at her new hat”, rather than a common name such as “Mr. Bear”. Finding
familiar fragments is expected; not finding one phrase does not rule out
memorization elsewhere. Here’s a literal phrase search:

```bash
uv run python - <<'PYCODE'
from pathlib import Path
text = Path("data/stories/train.txt").read_text()
print(text.count("very good at her new hat"))
PYCODE
```

Our training file prints `0`. That answers this one phrase search; it doesn’t
prove the whole story is new.

## Your turn: let the gremlin loose

You’ve read your five fixed samples. Time to play: ask for dragons, spaceships
or a caffeinated duck:

```bash
uv run python generate.py --checkpoint runs/friday/final.pt --prompt "The caffeinated duck borrowed a spaceship."
```

For the CPU short path, replace `runs/friday/final.pt` with
`runs/friday-cpu/final.pt` in that command. Try your own opening, then play
with one sampling knob at a time. Give the funniest story its own page.
The duck has waited five lessons for a spaceship.

You’ve taken the whole path from text to a model you can save, reload and
hear from. If you want to use your own text next, treat that as a separate
recipe: rebuild the split and tokenizer, and record its input fingerprints.
Do not change today’s expected hashes merely to make a refusal disappear.

## If you get stuck

- **Checkpoint missing:** confirm training wrote final.pt, then pass that exact path with --checkpoint.
- **Scoreboard rejects non-finite weights:** preserve the failed file for debugging and return to the training failure; a loadable checkpoint alone is insufficient.
- **Out of memory or device error:** use the CPU short training path and score/generate from runs/friday-cpu/final.pt.

> **What this proves:** the trained model predicts these held-out next tokens
> better than guessing from training-token frequencies.
> **What it doesn’t prove:** reliable facts or coherent plots. The 70% figure
> describes lower cross-entropy here, not 70% better story quality. The
> 30% pass rule was fixed in the recipe before this reference run.

Next week, we can try another job for a small model: sorting requests such
as “search the web” and “make a picture”, and checking when it should say
“unsure”. For today, let the duck enjoy its spaceship.

[Day 1](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/day-1.md) · [Day 2](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/day-2.md) · [Day 3](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/day-3.md) · [Day 4](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/day-4.md) · **Day 5**
