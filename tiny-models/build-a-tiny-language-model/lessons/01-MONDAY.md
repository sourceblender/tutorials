
# Monday: Give your model something to read

**Day 1** · [Day 2](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/02-TUESDAY.md) · [Day 3](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/03-WEDNESDAY.md) · [Day 4](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/04-THURSDAY.md) · [Day 5](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/05-FRIDAY.md)

> **Today’s question:** Can we turn a fixed corpus into reproducible token IDs?

This week you will build a small language model that writes short stories.
Today dragons and spaghetti become integers. You will prepare its reading material and train its tokenizer: the part
that turns text into integer IDs. By the end, you should have frozen training
and validation files, a tokenizer and a sentence you can encode and decode.

You need basic Python and terminal experience. We will supply the model code
and explain it as we go; you do not need to know calculus to start. The reference
machine is an Apple-silicon Mac. This new companion folder has been exercised
on an M5 Air, including its CPU training and generation path. Linux and Windows
have not yet been checked, nor has a separate CPU-only machine.

The week’s path is short: Monday turns text into tokens; Tuesday turns tokens
into logits; Wednesday makes the weights learn; Thursday makes the experiment
resumable; Friday turns a checkpoint into stories.

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

There is one separate installation. The Python morpheme package encodes text,
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

Stop if the checksum check fails. The last command should print
`morpheme 0.5.0`. The PATH change lasts for the current shell; if you open a new
terminal, set it again before continuing. Other platforms require their own
release binary; the Apple archive cannot run on them.

Now check the environment before doing any data work:

```bash
uv run python setup_check.py
```

It prints Python, PyTorch, the selected device, bfloat16 support and the CLI
version, ending with `setup OK`. On our Air the device is `mps`; a CPU machine
uses `cpu`. A `bf16 no` result means training should use float32, not that setup
failed. If the script prints `PROBLEM`, fix that item before proceeding.

## Download a small, fixed corpus

We use a pinned 40 MiB slice of `TinyStoriesV2-GPT4-train.txt`. Our validation
set is held out from that slice; it is not TinyStories’ published validation set. A small supplied corpus keeps
today focused on the pipeline. The data is downloaded from its source rather
than bundled with the tutorial.

```bash
uv run python day1_data.py
```

The script requests the pinned byte range, checks its hash and separates stories
at `<|endoftext|>`. It strips surrounding whitespace and drops the last piece,
which was cut off by the download boundary. It collapses whitespace in each
story to make an assignment key, hashes that key with SHA256 and uses the
integer hash modulo 100 as a bucket. Buckets 0–4 go to validation; the others
go to training. Identical normalized stories always share a side.

This assignment does not depend on where a story appeared in the source file.
Within each split, stories keep their original order and stripped text; the
normalized key is only for assignment. Five percent is the target proportion,
not an exact quota: this slice gives 48,620 training and 2,567 validation stories.
This is our custom held-out split, not a benchmark of every possible story.

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

Byte-level BPE starts with a representation of input bytes, then learns merges
for frequent sequences. A rare spelling can fall back to smaller pieces instead
of needing its own vocabulary entry. Round-trip inspection lets you see whether
the complete encoding and decoding pipeline preserves the text you supplied.

Vocabulary size is a budget choice. More entries can shorten common sequences,
but enlarge the embedding matrix and the scores computed for every next-token
guess. At width 256, a 4,096-entry token matrix has 1,048,576 parameters. We
share it with the output head. This is a manageable teaching budget, not a claim
that 4,096 is best for every corpus.

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
Do not replace the expected hash with the new one merely to continue. Later
results depend on the token IDs produced by this particular tokenizer.

## Look at the pieces

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
not a rule that long words must split and short words must stay whole.

The inspection also encodes and decodes 100 validation stories, checks that
they return unchanged and counts tokens in both files. On the checked inputs:

```text
round-trip: 100/100 validation stories came back exactly
tokens: train 9,975,703  val 529,456   (1.27 tokens per word on validation)
```

Here, “word” means a whitespace-separated item in the serialized validation
file, including its story separators. The 1.27 ratio describes this file and
this tokenizer; it is not an estimate for all English text.

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

## Freeze today's outputs

The final step records the three file hashes and the train/validation
token counts in `data/FROZEN.txt`. That is the identity of today's inputs for
the later lessons. Tomorrow you will turn their IDs into a model's guesses.

The script verifies the pinned input hashes first, requires at least 100
validation stories and tests those 100 round trips. Only after successful
checks does it replace the manifest. If an input changed or a check fails,
the script exits without replacing an existing valid manifest. Rerun the
data and tokenizer steps to recover the supplied inputs.

Timing note: we observed data preparation at 3.08 s, tokenizer training at
1.31 s and inspection at 10.97 s on the M5 Air. These are warm-machine observations,
not installation or reader completion times. A cold install remains unmeasured.

> **What this proves:** the pinned slice rebuilds into the specified files, and
> the trained tokenizer round-trips the checked stories.
> **What it doesn’t prove:** every input will be equally compact or that this
> held-out slice represents all the stories you might want to write.

**Day 1** · [Day 2](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/02-TUESDAY.md) · [Day 3](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/03-WEDNESDAY.md) · [Day 4](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/04-THURSDAY.md) · [Day 5](https://github.com/sourceblender/tutorials/blob/main/tiny-models/build-a-tiny-language-model/lessons/05-FRIDAY.md)
