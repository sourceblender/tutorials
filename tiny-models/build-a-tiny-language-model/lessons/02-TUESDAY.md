
# Day 2: Turn token IDs into guesses

[Day 1](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/01-MONDAY.md) · **Day 2** · [Day 3](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/03-WEDNESDAY.md) · [Day 4](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/04-THURSDAY.md) · [Day 5](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/05-FRIDAY.md)

> **Today’s question:** Can our model predict without cheating?

> **Plan:** about an hour.
> **Bring:** Day 1’s FROZEN.txt, text files and tokenizer.
> **Finish with:** a 5,051,904-parameter model that passes forward, initialization and causal checks.

In Day 1 you made a tokenizer. Today you will assemble and understand a model that accepts its
IDs and returns scores for the next token. It will not know how to write yet.
Our finish line is a working forward pass with the right shape, a reasonable
starting loss and attention that cannot read the future. Loss is a number
that measures how wrong the model’s next-token guesses are; smaller is better.

By the end, you own five million parameters of carefully initialized ignorance.

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

## Work through one attention head

```bash
uv run python day2_attention.py
```

The script makes four tiny query, key and value vectors. A query describes what
a position is looking for; a key describes information a position offers;
a value holds the information it can pass along. Their
dot product gives a score. Dividing by the square root of the head dimension
controls its scale. Our model divides width 256 across four heads, so each head
has dimension 64 and uses a scale of 1 / sqrt(64). Softmax turns a row of scores into weights, and multiplying
by the values makes a weighted mixture.

Before softmax, the causal mask replaces scores for later positions with
negative infinity. Their weights become zero. In the printed matrix, a row is
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

1. Token embeddings turn IDs into 256-number vectors. Learned position
   embeddings add where each token sits in the sequence.
2. Each block uses LayerNorm to keep the scale of its numbers manageable,
   then computes four attention heads, four ways to look at the context. Their
   outputs are projected back to the model width and added to the residual
   stream, the representation that runs through the blocks.
3. A second LayerNorm feeds an MLP, a small feed-forward network. It expands
   the width fourfold, applies a curved function called GELU so the network
   can learn more than straight-line transformations, and projects back.
   Its output is also added to the residual stream.
4. Five blocks are followed by a final LayerNorm and a vocabulary projection.
   The projection shares the token embedding's weight matrix. We count that
   shared matrix once when counting parameters.

These final vocabulary scores are called logits. They are not probabilities
until a softmax converts them. In Day 3, cross-entropy will use them to measure
how much probability the model gives the true next token.

![For one sequence, token IDs flow through token plus position embeddings, five residual attention-and-MLP blocks, a final LayerNorm, and 4096 vocabulary logits per position. The output weights are tied to token embeddings.](https://raw.githubusercontent.com/sourceblender/tutorials/main/tiny-models/build-a-tiny-language-model/assets/day2-model-path.png)

One sequence follows this path. The diagram suppresses the batch dimension after the input; the script retains it, so its output shape is batch × time × vocabulary.

## Where do five million parameters go?

Count shared weights once. With width 256, five blocks and no linear biases,
the budget is:

| Component | Calculation | Parameters |
|---|---|---:|
| Token embedding | 4,096 × 256 | 1,048,576 |
| Position embedding | 256 × 256 | 65,536 |
| Attention projections, five blocks | 5 × (3 × 256² + 256²) | 1,310,720 |
| MLPs, five blocks | 5 × (256 × 1,024 + 1,024 × 256) | 2,621,440 |
| Two LayerNorms per block, plus final norm | 11 × (256 weights + 256 biases) | 5,632 |
| Tied output head | Reuses the token embedding | 0 extra |
| **Total** | | **5,051,904** |

The MLPs hold more than half the weights. Attention is the memorable mechanism,
while the MLPs account for over half the parameters.

We’ll call this small model the gremlin: five million parameters, all of them
still guessing. Training comes next.

## Check the forward pass

```bash
uv run python day2_check.py
```

This diagnostic runs on CPU, even on the Air, and seeds its random inputs. The
checked output includes:

```text
2. parameters: 5,051,904
3. tiny batch logits: (1, 16, 4096)   training batches will be (32, 256, 4096)
4. starting loss: 8.3857   (uniform guessing would be ln(4096) = 8.3178)
5. causal mask: positions 0-199 unchanged = True; positions 200+ changed = True
Day 2 checks pass
```

The tiny forward pass has one sequence, 16 positions and 4,096 vocabulary
scores per position. The larger training shape is printed for reference; the
shape check does not allocate that training batch.

The loss check uses eight 256-token validation windows. A model that gives
every token equal probability has loss ln(4096). Random initialization is not
perfectly uniform, so the diagnostic accepts a difference below 0.5 rather
than requiring the printed 8.3857 exactly. We got 8.3857 on CPU. Day 5 will evaluate the untrained model over the whole
validation file and report a slightly different value.

The causal check uses its own full-length input. It changes token 200 and
requires earlier logits to stay unchanged while later logits change. If
attention can see future tokens, training can look impressive by reading the
answer. In our testing, disabling the causal mask made this check fail.

## See why initialization matters

```bash
uv run python day2_check.py --default-init
```

This is an intentional failing experiment. It keeps the architecture but skips
our initialization of Linear and Embedding weights to mean 0 and standard
deviation 0.02. LayerNorm keeps its defaults in both cases. With the library's
default weights and the tied vocabulary head, the checked run printed:

```text
4. starting loss: 171.2680   (uniform guessing would be ln(4096) = 8.3178)
FAIL: starting loss 171.27 is far from ln(4096) = 8.32; check the initialisation
```

Why the explosion? PyTorch’s Embedding defaults to a normal distribution with
standard deviation 1, versus our 0.02. The tied matrix also projects the hidden
state into output scores; large initial weights can make those scores enormous
and confidently wrong. Initialization is controlling that scale, not teaching
the model facts. [PyTorch’s Embedding documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Embedding.html) specifies the default.

An exit code of 1 here is expected. The model still produces logits of the
right shape; shape alone did not catch the problem. Rerun without the flag to
return to the supplied initialization. You have not saved or trained bad
weights by running this diagnostic.

The gremlin can now make guesses without peeking at the answer. In Day 3
you will change its weights so those guesses improve.

## If you get stuck

- **Starting loss near 171:** check whether you kept the deliberate --default-init flag. Remove it and rerun the normal check.
- **Frozen-input refusal:** return to Day 1 and rebuild the affected artifact before inspecting the model.
- **Causal check fails:** check the attention call’s causal mask. Fix that boundary before training.

> **What this proves:** the supplied model produces the intended shape, starts
> at a sensible loss and passes the tested causal boundary.
> **What it doesn’t prove:** that it has learned anything. Today’s weights are
> still random; learning begins in Day 3.

[Day 1](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/01-MONDAY.md) · **Day 2** · [Day 3](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/03-WEDNESDAY.md) · [Day 4](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/04-THURSDAY.md) · [Day 5](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/05-FRIDAY.md)
