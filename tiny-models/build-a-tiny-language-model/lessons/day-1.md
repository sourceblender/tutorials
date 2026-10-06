
# Day 1: Give your model something to read

**Day 1** · [Day 2](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/02-TUESDAY.md) · [Day 3](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/03-WEDNESDAY.md) · [Day 4](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/04-THURSDAY.md) · [Day 5](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/05-FRIDAY.md)

> **Today’s question:** Can the same stories become the same numbered pieces of text?

> **Plan:** about an hour, plus environment setup.
> **Bring:** basic Python, Git and a terminal.
> **Finish with:** frozen data, a trained tokenizer and reproducible token IDs.

This week you will build a small language model that writes short stories.
Today dragons and spaghetti become integers. You will prepare its reading material and train its tokenizer: the part
that turns text into numbered pieces called tokens. By the end, you should have frozen training
and validation files, a tokenizer and a sentence you can encode and decode.

You need basic Python and terminal experience. We will supply the model code
and explain it as we go; you do not need to know calculus to start. The reference
machine is an Apple-silicon Mac. This companion folder has been exercised
on an M5 Air, including its CPU training and generation path, and on
Ubuntu 26.04.1 x86_64 through the CPU short path. Linux installs CPU-only
PyTorch to keep setup simple. Windows is outside the tested path for this edition.

The time boxes are planning suggestions: leave room to explore, and take
a break when you need one. The scripts’ measured run times appear beside
the exercises.

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
For Apple Silicon, install the checksum-checked 0.5.0 release inside this folder:

```bash
mkdir -p tools/morpheme
cd tools/morpheme
curl --fail --location --silent --show-error https://github.com/sourceblender/morpheme/releases/download/v0.5.0/morpheme-cli-aarch64-apple-darwin.tar.xz --output morpheme-cli-aarch64-apple-darwin.tar.xz
curl --fail --location --silent --show-error https://github.com/sourceblender/morpheme/releases/download/v0.5.0/morpheme-cli-aarch64-apple-darwin.tar.xz.sha256 --output morpheme-cli-aarch64-apple-darwin.tar.xz.sha256
shasum -a 256 -c morpheme-cli-aarch64-apple-darwin.tar.xz.sha256
tar -xf morpheme-cli-aarch64-apple-darwin.tar.xz
cd ../..
export PATH="$PWD/tools/morpheme/morpheme-cli-aarch64-apple-darwin:$PATH"
morpheme --version
```

For Linux x86_64, use this block instead of the Apple-silicon block:

```bash
mkdir -p tools/morpheme
cd tools/morpheme
curl --fail --location --silent --show-error https://github.com/sourceblender/morpheme/releases/download/v0.5.0/morpheme-cli-x86_64-unknown-linux-gnu.tar.xz --output morpheme-cli-x86_64-unknown-linux-gnu.tar.xz
curl --fail --location --silent --show-error https://github.com/sourceblender/morpheme/releases/download/v0.5.0/morpheme-cli-x86_64-unknown-linux-gnu.tar.xz.sha256 --output morpheme-cli-x86_64-unknown-linux-gnu.tar.xz.sha256
sha256sum -c morpheme-cli-x86_64-unknown-linux-gnu.tar.xz.sha256
tar -xf morpheme-cli-x86_64-unknown-linux-gnu.tar.xz
cd ../..
export PATH="$PWD/tools/morpheme/morpheme-cli-x86_64-unknown-linux-gnu:$PATH"
morpheme --version
```

Stop if the checksum check fails. The last command should print
`morpheme 0.5.0`. The PATH change lasts for the current shell; if you open a new
terminal, set it again before continuing. Choose the block for your platform;
the two archives contain different executables.

Now check the environment before doing any data work:

```bash
uv run python setup_check.py
```

It prints Python, PyTorch, the selected device, bfloat16 support and the CLI
version, ending with `setup OK`. On our Air the device is `mps`; a CPU machine
uses `cpu`. A `bf16 no` result means training should use float32, not that setup
failed. If the script prints `PROBLEM`, fix that item before proceeding.

## Download a small, fixed corpus

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

The script requests the pinned byte range, checks its hash and separates stories
at `<|endoftext|>`. It strips surrounding whitespace and drops the last piece,
which was cut off by the download boundary. It collapses whitespace in each
story to make an assignment key, then hashes that key with SHA256. Dividing
that hash by 100 gives a remainder from 0 to 99: our bucket number. Buckets
0–4 go to validation; the others go to training. Identical normalized stories always share a side.

This assignment does not depend on where a story appeared in the source file.
Within each split, stories keep their original order and stripped text; the
normalized key is only for assignment. We aim for about five percent in validation; this slice gives 48,620 training and 2,567 validation stories.

Expected output:

```text
stories: 51187 (train 48620, val 2567); hashes match
```

Open `data/stories/train.txt` and read a few stories. Notice the separator and
the simple language. Those are properties of what our model will learn from.

Validation is the portion we reserve for measuring the result. The split
happens before tokenizer training so that validation text cannot influence the
tokenizer's vocabulary. Later, we will encode validation using the already
trained tokenizer; encoding text does not fit a new vocabulary.

## Why bytes, and why 4,096 tokens?

Computers store text as bytes. Byte-pair encoding (BPE) begins with small
pieces and repeatedly joins pairs that appear together often. Our byte-level
version starts from bytes. A rare spelling can fall back to smaller pieces instead
of needing its own vocabulary entry. Round-trip inspection lets you see whether
the complete encoding and decoding pipeline preserves the text you supplied.

Vocabulary size is a budget choice. More entries can shorten common sequences,
but need a larger lookup table to turn each token into numbers the model can
work with. At width 256, a 4,096-entry table contains 1,048,576 learned numbers,
or parameters. The model also reuses that table when scoring its next-token
guesses; Day 2 shows how. That leaves room in our five-million-parameter
budget for the layers that learn context.

## Train the tokenizer

A token is a piece of text with an integer ID. It can be a whole word, part of
a word or punctuation. Our tokenizer learns a byte-level BPE vocabulary with
4,096 entries from training text only.

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
Keep the expected hash while finding the cause. That hash connects your
tokenizer to the later results; changing it would hide the mismatch.

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

In this tokenizer, `spaghetti` is one piece, while `unbelievably` becomes seven:
`un`, `b`, `el`, `ie`, `v`, `ab`, `ly`. The printed `Ġ` marker represents a
leading space in the byte-level token display; it is not an extra character
inserted into the decoded sentence. Pieces reflect this corpus and vocabulary,
so word length alone will not tell you how many pieces to expect.

The inspection also encodes and decodes 100 validation stories, checks that
they return unchanged and counts tokens in both files. On the checked inputs:

```text
round-trip: 100/100 validation stories came back exactly
tokens: train 9,975,703  val 529,456   (1.27 tokens per word on validation)
```

Here, “word” means a whitespace-separated item in the serialized validation
file, including its story separators. The 1.27 ratio describes this file and
this tokenizer.

## One sentence that is yours

Before freezing, give the tokenizer a sentence of your own. You can inspect it
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

## Freeze today's outputs

The `day1_look.py` command you just ran records the three file hashes and the train/validation
token counts in `data/FROZEN.txt`. That is the identity of today's inputs for
the later lessons. In Day 2 you will turn their IDs into a model's guesses.

The script verifies the pinned input hashes first, requires at least 100
validation stories and tests those 100 round trips. Only after successful
checks does it replace the manifest. If an input changed or a check fails,
the script exits without replacing an existing valid manifest. Rerun the
data and tokenizer steps to recover the supplied inputs.

On our warmed-up M5 Air, data preparation took 3.08 seconds, tokenizer
training 1.31 seconds and inspection 10.97 seconds. Reading the outputs and
trying your own sentences is where you will spend most of this lesson.

## If you get stuck

- **Checksum mismatch:** stop and re-download the pinned artifact. Keep the expected hash unchanged.
- **morpheme missing:** return to the project folder and repeat the PATH export for your platform. The Python package and CLI are separate installs.
- **bf16 no:** the selected device uses float32. You can still continue with the CPU path.

> **What this proves:** the pinned slice rebuilds into the specified files, and
> the trained tokenizer round-trips the checked stories.
> **What it doesn’t prove:** every input will be equally compact or that this
> held-out slice represents all the stories you might want to write.

**Day 1** · [Day 2](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/02-TUESDAY.md) · [Day 3](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/03-WEDNESDAY.md) · [Day 4](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/04-THURSDAY.md) · [Day 5](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/05-FRIDAY.md)
