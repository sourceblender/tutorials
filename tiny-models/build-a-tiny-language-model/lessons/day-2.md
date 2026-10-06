
# Day 2: Turn token IDs into guesses

[Day 1](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/day-1.md) · **Day 2** · [Day 3](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/day-3.md) · [Day 4](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/day-4.md) · [Day 5](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/day-5.md)

> **Today’s question:** Can our model predict without cheating?

> **Time:** allow 25–40 minutes to read and explore; the checks are short.
> **You need:** Day 1’s text files, tokenizer and FROZEN.txt.
> **You’ll have:** a model that makes guesses, plus checks that catch peeking and bad initialization.

Today the gremlin gets a body: five million parameters of carefully initialized
ignorance. A **parameter** is a number training will change. You’ll read the
supplied model, check that it does the right thing, then break it on purpose.
You do not need to implement every line from an empty file.

Each token ID becomes a list of 256 numbers, its **embedding**. The length of
that list is the model’s **width**. The model reads and rewrites those numbers
to make its next-token guess. It can also see at most 256 tokens at once.
That’s a different 256: the length of the text window, rather than each token’s list.

A **forward pass** is one trip through the model to produce guesses. **Loss**
measures how wrong those guesses are; smaller is better. Today’s goal is to
get a forward pass that works and cannot sneak a look at the answer.

Run today’s commands from the same project folder as Day 1. The Python
package is already installed by uv; today needs no extra CLI setup.

## Start with the question the model will answer

The training input is a window of tokens. The target is the window starting
one token later. At each position, we ask the model to guess the next token.
For example, today's diagnostic prints:

```text
input  x: 'Once upon a time there was a little girl named Kate. She had a'
target y: ' upon a time there was a little girl named Kate. She had a wonderful'
```

The target drops the initial `Once` and adds `wonderful` at the end. These
decoded strings illustrate the offset; the actual training pairs are integer
IDs. The diagnostic uses the beginning of validation text for this display,
and makes no optimizer updates.

## How does a token look back?

Consider “Lily dropped her hat. She went back for it.” To guess what comes
next, it helps to connect “She” with “Lily”. **Attention** lets a token gather
useful information from the words it can see.

Each token’s 256 numbers pass through three learned tables to make:

- A **query**: what this position is looking for.
- A **key**: what a position offers.
- A **value**: what information it will hand over.

Compare this token’s query with each earlier token’s key. The comparison is a
**dot product**: multiply matching entries in the two lists and add the results.
That gives one score per pair. The value is what gets mixed into the answer;
it is not one of the two lists being compared.

Our four heads each work with a 64-number slice of the token’s 256 numbers.
A head compares lists of 64 numbers. We divide the score by `sqrt(64)` to keep
its scale manageable. **Softmax** turns the scores into positive weights that
sum to one. Very large score gaps would make that mixture nearly all-or-nothing;
scaling helps keep it useful for learning. A fourth learned table mixes the
heads’ results back together. Different learned slices let heads develop
different habits.

See this on a tiny four-position example:

```bash
uv run python day2_attention.py
```

Before softmax, we apply a **causal mask**, the no-reading-ahead rule. It replaces scores for later positions with negative infinity, making their weights zero. In the printed matrix, a row is
the reading position and a column is a position it can read:

```text
tensor([[1.00, 0.00, 0.00, 0.00],
        [0.49, 0.51, 0.00, 0.00],
        [0.62, 0.03, 0.35, 0.00],
        [0.20, 0.62, 0.10, 0.07]])
by hand == PyTorch: True
```

The zeros above the diagonal are the important pattern. The rows sum to one
before display rounding. The last line checks that our calculation agrees with PyTorch’s attention
function within a small numerical tolerance. Two ways to do the same job.
The future is still off-limits.

![A four-by-four attention heatmap. Every cell above the diagonal is zero; each row reads only its own position and earlier positions.](https://raw.githubusercontent.com/sourceblender/tutorials/main/tiny-models/build-a-tiny-language-model/assets/day2-causal-attention.png)

This is the small seed-0 example printed by the script, rounded to two decimals.

Before running the next check, predict the answer: if token 200 changes,
which earlier positions are allowed to change? Write your prediction down.
The causal check below turns that guess into an experiment.

## Read the supplied model

Open `model.py`. Follow its forward path rather than implementing every line
from an empty file:

1. Token embeddings turn IDs into 256-number lists. Position embeddings add
   where each token sits. There are 256 positions: training uses windows of
   that many tokens, and generation keeps only the last 256. A longer story
   eventually loses its opening from view.
2. Each block uses LayerNorm to keep the scale of its numbers manageable,
   then computes four attention heads, four ways to look at the context. Their
   outputs are mixed back into 256 numbers and added to what was there. This
   running list is the **residual stream**: a shared notepad each block adds
   a correction to, instead of erasing the previous work.
3. A second LayerNorm feeds an MLP, a small feed-forward network. It expands
   the width fourfold, applies a curved function called GELU so the network
   can learn more than straight-line transformations, and projects back.
   Its output is also added to the residual stream.
4. Five blocks are followed by a final LayerNorm and a vocabulary projection.
   The output compares the final 256 numbers with each token’s embedding to
   score possible next tokens. It reuses the input table for that comparison,
   a **tied output head**. One table does both jobs, saving about a million
   parameters.

These final vocabulary scores are called logits. They are not probabilities
until a softmax converts them. In Day 3, cross-entropy will use them to measure
how much probability the model gives the true next token.

![For one sequence, token IDs flow through token plus position embeddings, five residual attention-and-MLP blocks, a final LayerNorm, and 4096 vocabulary logits per position. The output weights are tied to token embeddings.](https://raw.githubusercontent.com/sourceblender/tutorials/main/tiny-models/build-a-tiny-language-model/assets/day2-model-path.png)

The diagram follows one sequence. A batch is several sequences processed together; the output has one set of vocabulary scores for each position in each sequence.

## Where do five million parameters go?

The MLPs hold more than half the weights. Use the table as a map of the budget; you can look up the arithmetic when you need it. We chose width 256, four heads
and five blocks to make a roughly five-million-parameter model that could
train on our laptop. We did not search for the best architecture.

Count shared weights once. Here is where the numbers go:

| Component | Calculation | Parameters |
|---|---|---:|
| Token embedding | 4,096 × 256 | 1,048,576 |
| Position embedding | 256 × 256 | 65,536 |
| Attention projections, five blocks | 5 × (3 × 256² + 256²) | 1,310,720 |
| MLPs, five blocks | 5 × (256 × 1,024 + 1,024 × 256) | 2,621,440 |
| Two LayerNorms per block, plus final norm | 11 × (256 weights + 256 biases) | 5,632 |
| Tied output head | Reuses the token embedding | 0 extra |
| **Total** | | **5,051,904** |

The attention row counts three tables for query, key and value, plus one to mix their results: `3 × 256² + 256²` in each block.

## Check the forward pass

```bash
uv run python day2_check.py
```

The checked output includes:

```text
2. parameters: 5,051,904
3. tiny batch logits: (1, 16, 4096)   training batches will be (32, 256, 4096)
4. starting loss: 8.3857   (uniform guessing would be ln(4096) = 8.3178)
5. causal mask: positions 0-199 unchanged = True; positions 200+ changed = True
Day 2 checks pass
```

The tiny forward pass has one sequence, 16 positions and 4,096 vocabulary scores per position. The printed larger shape shows what a training batch will look like.

Why about 8.32? Loss is minus the natural logarithm of the probability given
to the right next token. Guess evenly among 4,096 tokens and the right one
always gets `1/4096`: `-ln(1/4096) = ln(4096) ≈ 8.32`. A confident correct
guess has loss near zero. Random weights should begin near the even-guessing
number, rather than being wildly confident in wrong answers.

The causal check uses its own full-length input. It changes token 200 and
requires earlier logits to stay unchanged while later logits change. If
attention can see future tokens, training can look impressive by reading the
answer. In our testing, disabling the causal mask made this check fail.

## See why initialization matters

The next command should fail. It deliberately swaps our small starting weights for the library defaults:

```bash
uv run python day2_check.py --default-init
```

It keeps the architecture, but changes the scale of the starting weights. The checked run printed:

```text
4. starting loss: 171.2680   (uniform guessing would be ln(4096) = 8.3178)
FAIL: starting loss 171.27 is far from ln(4096) = 8.32; check the initialisation
```

Why the explosion? PyTorch’s Embedding defaults to a normal distribution with
standard deviation 1, versus our 0.02. The same table also turns the model’s
final numbers into next-token scores. Large starting weights can make those scores enormous
and confidently wrong. Initialization is controlling that scale, not teaching
the model facts. [PyTorch’s Embedding documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Embedding.html) specifies the default.

An exit code of 1 here is expected. The model still produces logits of the
right shape; shape alone did not catch the problem. Rerun without the flag to
return to the supplied initialization. You have not saved or trained bad
weights by running this diagnostic.

The gremlin can make guesses without peeking at the answer. In Day 3
you will change its weights so those guesses improve.

## If you get stuck

- **Starting loss near 171:** check whether you kept the deliberate --default-init flag. Remove it and rerun the normal check.
- **Frozen-input refusal:** return to Day 1 and rebuild the affected artifact before inspecting the model.
- **Causal check fails:** check the attention call’s causal mask. Fix that boundary before training.

> **What this proves:** the supplied model produces the intended shape, starts
> at a sensible loss and passes the tested causal boundary. The CPU diagnostic
> checks eight validation windows and allows loss within 0.5 of uniform guessing;
> your last decimal does not have to match ours.
> **What it doesn’t prove:** that it has learned anything. Today’s weights are
> still random; learning begins in Day 3.

[Day 1](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/day-1.md) · **Day 2** · [Day 3](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/day-3.md) · [Day 4](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/day-4.md) · [Day 5](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/day-5.md)
