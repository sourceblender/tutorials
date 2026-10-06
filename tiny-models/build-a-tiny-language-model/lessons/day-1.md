
# Day 1: Give your model something to read

**Day 1** · [Day 2](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/day-2.md) · [Day 3](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/day-3.md) · [Day 4](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/day-4.md) · [Day 5](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/day-5.md)

This week you will build a small language model that writes short stories.
Here is a bite-sized preview from the model we trained:

```text
The big dog was nice. He wagged his tail and licked Lily's face. Lily laughed and said, "Thank you, Max. You are a good friend. Let's play!" Max wagged his tail and barked happily. They played hide and seek all day long.
```

That is the gremlin you’re working toward: a tiny model with a taste for dogs,
parties and occasionally very confused cakes. We’ll meet its mistakes too.
Today, dragons and spaghetti become integers. You’ll train a tokenizer,
the part that turns text into numbered pieces called tokens.

> **Today’s question:** What does a model see when it reads?

> **Time:** allow 20–30 minutes to read and explore; setup and downloads vary.
> **You’ll have:** text files, a trained tokenizer and a sentence you can turn into IDs and back.
> **You need:** basic Python, Git and a terminal.

We will supply the model code
and explain it as we go; you do not need to know calculus to start. The reference
machine is an Apple-silicon Mac. We tested on the M5 Air and on Ubuntu Linux
x86_64 using its CPU. Linux installs CPU-only PyTorch to keep setup simple.
For Linux, follow the smaller CPU training option in this course. It gives
you a working model sooner, with rougher stories; Day 5 shows examples.
There is no Intel Mac CLI build. Windows via WSL has not been tested.

Most of today’s time goes into setup, reading the outputs and trying your own sentences. The data scripts themselves take seconds on our reference machine.

The week’s path is short: prepare the reading material, assemble a model,
watch it learn, save its progress, then read the stories it writes.

## Prepare the environment

Install Git and [uv](https://docs.astral.sh/uv/getting-started/installation/),
then clone the companion repository and open the project folder:

```bash
git clone https://github.com/sourceblender/tutorials.git
cd tutorials/tiny-models/build-a-tiny-language-model
uv sync --locked
```

All commands for the rest of the week run from this project folder. If you
open a new terminal, return here before running the next lesson’s commands.

The folder pins Python 3.12 and the package versions used by the recipe. Keep
the lockfile: updating dependencies halfway through the week changes the
experiment you are following.

[morpheme](https://github.com/sourceblender/morpheme) is our open-source Rust
subword tokenizer, compatible with Hugging Face tokenizers. We use the project
we built so the training and encoding internals are available alongside the
exercise. There is one separate installation: the Python package encodes text,
but does not supply the command-line executable we use to train the tokenizer.
Install the pinned release with the same command on Apple silicon or Linux
x86_64. This is the standard cargo-dist installer; it chooses the build and
verifies the binary it downloads:

```bash
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/sourceblender/morpheme/releases/download/v0.5.0/morpheme-cli-installer.sh | sh
source "$HOME/.cargo/env"
morpheme --version
```

The final command should print:

```text
morpheme 0.5.0
```

The executable goes in `~/.cargo/bin`. The installer also sets up your shell
so later terminals can find it. The `source` line makes it available in this
terminal right away. Only Day 1 needs this CLI; the other days use Python.

Now check the environment before doing any data work:

```bash
uv run python setup_check.py
```

Look for `setup OK` at the end. The device is where the calculations run:
`mps` means the Mac’s GPU, while `cpu` means the processor. `bf16` is a smaller
number format that a GPU may use for speed. `bf16 no` is fine: the scripts
use ordinary 32-bit numbers instead. If you see `PROBLEM`, fix the named item.

Later lessons offer a **CPU short path**: smaller batches and a shorter final
run for machines without an Apple GPU. You’ll meet that choice in Day 3.

## Give it some stories

A **corpus** is a collection of text. We’ll split ours into training stories
to learn from and **validation** stories to score without learning from them.
That reserved text helps us ask whether the model learned useful patterns,
rather than only remembering its examples.

Our source is [TinyStories, by Ronen Eldan and Yuanzhi Li (2023)](https://arxiv.org/abs/2305.07759):
synthetic short stories designed around vocabulary understood by young children.
That constrained language makes a tiny model’s learning visible within a small
training budget. The [dataset card](https://huggingface.co/datasets/roneneldan/TinyStories)
lists its license as CDLA-Sharing-1.0.

We use a pinned 40 MiB slice of `TinyStoriesV2-GPT4-train.txt`. Our validation
set is held out from that slice, separately from TinyStories’ published
validation set. A small corpus keeps today focused on the pipeline. The data is downloaded from its source rather
than bundled with the tutorial.

```bash
uv run python day1_data.py
```

Expected output:

```text
stories: 51187 (train 48620, val 2567); hashes match
```

Identical stories should stay on the same side of the split. Otherwise,
validation could grade the model on its own homework. We decide the side from
the story’s text, rather than its position in the file: collapse whitespace,
calculate a SHA256 fingerprint, then put about five percent of fingerprints
in validation. Stories keep their original order within each side.

The script checks the download’s fingerprint, separates stories at
`<|endoftext|>` and drops the last partial story. That marker says where a
story ends. We’ll make it a **special token**, meaning it gets one ID and is
never split into pieces.

Open `data/stories/train.txt` and read a few stories. Notice the separator and
the simple language. Those are properties of what our model will learn from.

The split happens before tokenizer training so that validation text cannot influence the
tokenizer's vocabulary. Later, we will encode validation using the already
trained tokenizer. Using it later doesn’t change it; only training does.

## Why bytes, and why 4,096 tokens?

Computers store text as bytes. Byte-pair encoding (BPE) begins with small
pieces and repeatedly joins pairs that appear together often. Our byte-level
version starts from bytes. A rare spelling can fall back to smaller pieces instead
of needing its own vocabulary entry. Round-trip inspection lets you see whether
the complete encoding and decoding pipeline preserves the text you supplied.

Vocabulary size is a budget choice. More entries can shorten common sequences,
but need a larger lookup table to turn each token into numbers the model can
work with. We chose 4,096 entries to keep that table small enough for a laptop-sized model. Day 2 will show the table and count the learned numbers inside it.

## Train the tokenizer

A token is a piece of text with an integer ID. It can be a whole word, part of
a word or punctuation. Our tokenizer learns a byte-level BPE vocabulary with
4,096 entries from training text only. It starts from bytes and repeatedly
merges frequent neighbouring pairs into bigger pieces. An illustration:
`t` + `h` becomes `th`, then `th` + `e` becomes `the`. Those are example merges,
not a transcript of our tokenizer’s training. After thousands of merges,
common sequences have their own entries.

```bash
uv run python day1_tokenizer.py
```

The script calls the morpheme CLI with `--model bpe`, `--preset byte-level`,
`--vocab-size 4096` and `<|endoftext|>` as a special token. Its input is
`data/stories/train.txt`, and its output is `data/tok4096.json`.

The final line should be:

```text
tokenizer matches the tutorial's tok4096.json
```

If it reports a mismatch, check the CLI version and the preceding data step.
Don’t edit the expected hash to make the check pass. Fix the input or CLI version that caused the mismatch.

## Look at the pieces

Before running the inspection, predict how many pieces “unbelievably” will
become. The tokenizer will have the last word.

```bash
uv run python day1_look.py
```

The script prints pieces and IDs for three sentences. One is:

```text
The dragon's spaghetti was unbelievably wobbly!
```

The inspection also encodes and decodes 100 validation stories, checks that
they return unchanged and counts tokens in both files:

```text
round-trip: 100/100 validation stories came back exactly
tokens: train 9,975,703  val 529,456   (1.27 tokens per word on validation)
```

That is about 1.27 tokens per whitespace-separated word in our validation file.

After inspecting your output, compare it with ours:

In this tokenizer, `spaghetti` is one piece, while `unbelievably` becomes seven:
`un`, `b`, `el`, `ie`, `v`, `ab`, `ly`. The printed `Ġ` marker represents a
leading space in the byte-level token display; it is not an extra character
inserted into the decoded sentence. Our training stories contain the whole
word “spaghetti” 289 times and “unbelievably” zero times, counting without
regard to case. BPE learns frequent pairs, so this corpus gives those two
spellings very different opportunities to become a single piece. It learned
these stories, not a dictionary of English.

## One sentence that is yours

Give the tokenizer a sentence of your own. You can inspect it
without changing the training data or fitting another vocabulary:

```bash
uv run python - <<'PYCODE'
from morpheme import Tokenizer
tok = Tokenizer.from_file("data/tok4096.json")
sentence = "A caffeinated duck borrowed my spaceship."
enc = tok.encode(sentence)
print(enc.tokens)
print(tok.decode(enc.ids))
PYCODE
```

Change the sentence and look at the pieces. The tokenizer is not judging the
duck’s life choices; it is deciding how to spell them in its vocabulary.

![The caffeinated-duck sentence split into 13 colored byte-level BPE pieces, with the token ID below each piece. A leading-space symbol marks whitespace.](https://raw.githubusercontent.com/sourceblender/tutorials/main/tiny-models/build-a-tiny-language-model/assets/day1-token-pieces.png)

The same sentence becomes 13 IDs with our frozen tokenizer. Spaces are part of its representation, not separators added after tokenization.

## Fingerprint today’s outputs

Frozen means fingerprinted, not read-only. The look script already wrote
`data/FROZEN.txt`; there is nothing else to run here. It records fingerprints
for `data/stories/train.txt`, `data/stories/val.txt` and `data/tok4096.json`,
along with the token counts. Every later lesson checks them before it starts.
That catches a file quietly changing underneath the experiment.

If a check fails, rebuild the supplied data and tokenizer. The script keeps
an existing valid freeze file intact until the replacement passes its checks.

On our warmed-up M5 Air, data preparation took 3.08 seconds, tokenizer
training 1.31 seconds and inspection 10.97 seconds. Reading the outputs and
trying your own sentences is where you will spend most of this lesson.

## If you get stuck

- **Checksum mismatch:** stop and re-download the pinned artifact. Keep the expected hash unchanged.
- **morpheme missing:** run `source "$HOME/.cargo/env"`, then check `morpheme --version`. The Python package and CLI are separate installs.
- **bf16 no:** nothing to fix. The scripts use ordinary 32-bit numbers.

> **What this proves:** we can rebuild the same files and turn the checked
> stories into IDs and back without changing their text.
> **What it doesn’t prove:** that every word gets a tidy token, or that this
> small collection covers every kind of story.

**Day 1** · [Day 2](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/day-2.md) · [Day 3](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/day-3.md) · [Day 4](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/day-4.md) · [Day 5](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/day-5.md)
