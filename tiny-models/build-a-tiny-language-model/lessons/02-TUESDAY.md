
# Tuesday: Turn token IDs into guesses

[Day 1](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/01-MONDAY.md) · **Day 2** · [Day 3](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/03-WEDNESDAY.md) · [Day 4](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/04-THURSDAY.md) · [Day 5](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/05-FRIDAY.md)

> **Today’s question:** Can our model predict without cheating?

Yesterday you made a tokenizer. Today you will assemble and understand a model that accepts its
IDs and returns scores for the next token. It will not know how to write yet.
Our finish line is a working forward pass with the right shape, a reasonable
starting loss and attention that cannot read the future.

By the end, you own five million parameters of carefully initialized ignorance.

Run today's commands from the same companion folder as Monday.

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
a position is looking for; a key describes information a position offers. Their
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
before display rounding. The last line compares the manual calculation with
PyTorch's attention function using a numerical tolerance, not exact equality.

Before running the next check, predict the answer: if token 200 changes,
which earlier positions are allowed to change? Write your prediction down.
The causal check below turns that guess into an experiment.

## Read the supplied model

Open `model.py`. Follow its forward path rather than implementing every line
from an empty file:

1. Token embeddings turn IDs into 256-number vectors. Learned position
   embeddings add where each token sits in the sequence.
2. Each block applies LayerNorm and computes four attention heads. Their
   outputs are projected back to the model width and added to the residual
   stream, the representation that runs through the blocks.
3. A second LayerNorm feeds an MLP that expands the width fourfold, applies
   GELU and projects back. Its output is also added to the residual stream.
4. Five blocks are followed by a final LayerNorm and a vocabulary projection.
   The projection shares the token embedding's weight matrix. We count that
   shared matrix once when counting parameters.

These final vocabulary scores are called logits. They are not probabilities
until a softmax converts them. Tomorrow cross-entropy will use them to measure
how much probability the model gives the true next token.

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
but it does not own the entire parameter budget.

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
Tuesday checks pass
```

The tiny forward pass has one sequence, 16 positions and 4,096 vocabulary
scores per position. The larger training shape is printed for reference; the
shape check does not allocate that training batch.

The loss check uses eight 256-token validation windows. A model that gives
every token equal probability has loss ln(4096). Random initialization is not
perfectly uniform, so the diagnostic accepts a difference below 0.5 rather
than requiring the printed 8.3857 exactly. That number is this diagnostic's
CPU observation; the earlier whole-validation baseline used a different sample.

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

At today's finish, you have a model that asks the right question and obeys the
time boundary. Tomorrow you will change its weights so its guesses improve.

> **What this proves:** the supplied model produces the intended shape, starts
> at a sensible loss and passes the tested causal boundary.
> **What it doesn’t prove:** that it has learned anything. Today’s weights are
> still random; learning begins tomorrow.

[Day 1](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/01-MONDAY.md) · **Day 2** · [Day 3](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/03-WEDNESDAY.md) · [Day 4](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/04-THURSDAY.md) · [Day 5](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/05-FRIDAY.md)
