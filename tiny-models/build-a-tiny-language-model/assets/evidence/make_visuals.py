"""The week's evidence figures, drawn only from real outputs. Run from the tutorial folder after Monday:
    uv run --with matplotlib python assets/evidence/make_visuals.py assets/evidence assets
Inputs in assets/evidence/: day3_short.txt (day3_learn.py on the Air GPU), lr_schedule.csv, resume-A/B-log.jsonl
(two 120-update CPU runs, B stopped at 50 and resumed), scoreboard.txt (day5_scoreboard.py) and
full-run-log.jsonl (train.py's log of the 6,000-update reference run). Day 1 reads the
tokenizer Monday builds; Day 2 recomputes day2_attention.py's seed-0 matrix. Every plotted value is also written
to plotted-data.json.
"""
import json, math, os, re, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import morpheme
import torch
import torch.nn.functional as F

EV, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
INK, MUTED, ACCENT = "#1f2933", "#7b8794", "#2f6fdf"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
PLOTTED = {}  # every value drawn, written to plotted-data.json so pixels can be checked against data

# 1. Monday: a sentence becomes token pieces and ids (the real v2 tokenizer).
tok = morpheme.Tokenizer.from_file(os.path.join("data", "tok4096.json"))
sentence = "A caffeinated duck borrowed my spaceship."
enc = tok.encode(sentence)
pieces = [t.replace("Ġ", "␣") for t in enc.tokens]
PLOTTED["day1"] = {"sentence": sentence, "pieces": enc.tokens, "ids": enc.ids}
palette = ["#dbeafe", "#fde68a", "#bbf7d0", "#fecaca", "#e9d5ff", "#fed7aa"]
fig, ax = plt.subplots(figsize=(11, 2.6)); ax.axis("off")
x = 0.01
for i, (p, n) in enumerate(zip(pieces, enc.ids)):
    w = 0.012 + 0.0115 * len(p)
    ax.add_patch(plt.Rectangle((x, 0.45), w, 0.32, color=palette[i % len(palette)], ec=INK, lw=0.8))
    ax.text(x + w / 2, 0.61, p, ha="center", va="center", fontsize=12, family="monospace")
    ax.text(x + w / 2, 0.30, str(n), ha="center", va="center", fontsize=9, color=MUTED, family="monospace")
    x += w + 0.006
ax.text(0.01, 0.92, f"\"{sentence}\"  ->  {len(enc.ids)} tokens", fontsize=12, color=INK)
ax.text(0.01, 0.08, "␣ marks a leading space. Grey numbers are the token ids the model actually sees.", fontsize=9, color=MUTED)
fig.savefig(os.path.join(OUT, "day1-token-pieces.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

# 2. Tuesday: the hand-worked attention weights (day2_attention.py, seed 0) as a heatmap.
torch.manual_seed(0)
T, D = 4, 8
q, k, v = torch.randn(T, D), torch.randn(T, D), torch.randn(T, D)
scores = (q @ k.T / math.sqrt(D)).masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf"))
w = scores.softmax(-1).numpy()
PLOTTED["day2_attention"] = [[round(float(c), 6) for c in row] for row in w]
fig, ax = plt.subplots(figsize=(4.6, 4.2))
ax.imshow(w, cmap="Blues", vmin=0, vmax=1)
for i in range(T):
    for j in range(T):
        ax.text(j, i, f"{w[i, j]:.2f}", ha="center", va="center", color="white" if w[i, j] > 0.55 else INK)
ax.set_xticks(range(T), [f"token {j}" for j in range(T)], rotation=30)
ax.set_yticks(range(T), [f"token {i}" for i in range(T)])
ax.set_xlabel("position being read"); ax.set_ylabel("position reading")
ax.set_title("Causal attention: zeros above the diagonal\n(nothing reads the future)", fontsize=11)
fig.savefig(os.path.join(OUT, "day2-causal-attention.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

# 2b. Tuesday: the model's path, as a simple block diagram (separate image so the heatmap stays full size).
fig, ax = plt.subplots(figsize=(11, 3.0)); ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(0, 3)
stages = [("token IDs", "1 x T"), ("token + position\nembeddings", "T x 256"), ("block x 5", None),
          ("final\nLayerNorm", "T x 256"), ("vocabulary\nlogits", "T x 4,096")]
xs = [0.2, 2.1, 4.3, 7.4, 9.2]; widths = [1.5, 1.8, 2.7, 1.4, 1.6]
for (label, shape), x0, wd in zip(stages, xs, widths):
    face = "#eef2ff" if label != "block x 5" else "#f8fafc"
    ax.add_patch(plt.Rectangle((x0, 0.9), wd, 1.3, fc=face, ec=INK, lw=1))
    if label == "block x 5":
        ax.text(x0 + wd / 2, 2.0, "block  (x 5)", ha="center", fontsize=10, weight="bold")
        ax.text(x0 + wd / 2, 1.55, "x + attention(LayerNorm(x))", ha="center", fontsize=9, family="monospace")
        ax.text(x0 + wd / 2, 1.15, "x + MLP(LayerNorm(x))", ha="center", fontsize=9, family="monospace")
    else:
        ax.text(x0 + wd / 2, 1.6, label, ha="center", va="center", fontsize=10)
    if shape:
        ax.text(x0 + wd / 2, 0.6, shape, ha="center", fontsize=9, color=MUTED, family="monospace")
for (x0, wd), x1 in zip(zip(xs, widths), xs[1:]):
    ax.annotate("", xy=(x1 - 0.05, 1.55), xytext=(x0 + wd + 0.05, 1.55), arrowprops=dict(arrowstyle="->", color=INK))
ax.text(5.5, 2.7, "The output layer reuses the token-embedding matrix (tied weights).", ha="center", fontsize=9, color=MUTED)
fig.savefig(os.path.join(OUT, "day2-model-path.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

# 3. Wednesday: the 300-step run's loss and its learning-rate schedule, as two separate charts.
rows = [l for l in open(os.path.join(EV, "day3_short.txt")) if l.startswith("step ")]
steps = [int(re.search(r"step\s+(\d+)", l).group(1)) for l in rows]
loss = [float(re.search(r"loss ([\d.]+)", l).group(1)) for l in rows]
lr = [tuple(map(float, l.split(","))) for l in open(os.path.join(EV, "lr_schedule.csv")).read().split("\n")[1:] if l]
PLOTTED["day3"] = {"loss_steps": steps, "loss": loss, "lr_source": "lr_schedule.csv (all 300 steps)"}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6))
a1.plot(steps, loss, marker="o", color=ACCENT); a1.axhline(math.log(4096), ls="--", color=MUTED)
a1.text(300, math.log(4096) + 0.12, "uniform guessing (ln 4096)", ha="right", fontsize=9, color=MUTED)
a1.set_xlabel("update"); a1.set_ylabel("training loss"); a1.set_title("Loss on changing story windows (Air GPU)", fontsize=11)
a2.plot([s for s, _ in lr], [r for _, r in lr], color=INK)
a2.axvline(200, ls=":", color=MUTED); a2.text(203, 1e-3 * 0.95, "peak at 200", fontsize=9, color=MUTED, va="top")
a2.set_xlabel("update"); a2.set_ylabel("learning rate"); a2.set_title("Warm-up, then cosine decay", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "day3-loss-and-schedule.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

# 4. Thursday: uninterrupted vs stopped-at-50-and-resumed, same sampled windows at every eval.
def evals(name):
    return {r["step"]: r for r in map(json.loads, open(os.path.join(EV, name))) if r["event"] == "eval"}
A, B = evals("resume-A-log.jsonl"), evals("resume-B-log.jsonl")
PLOTTED["day4"] = [{"update": s_, "A_windows": A[s_]["starts_sha"], "B_windows": B[s_]["starts_sha"],
                    "A_train": A[s_]["train_loss"], "B_train": B[s_]["train_loss"],
                    "A_val": A[s_]["val_ce"], "B_val": B[s_]["val_ce"]} for s_ in sorted(A)]
fig, ax = plt.subplots(figsize=(11, 3.4)); ax.set_xlim(-2, 125); ax.set_ylim(-0.4, 3.2); ax.axis("off")
ax.set_title("Stop and resume vs an uninterrupted run (CPU, batch 8, 120 updates, evaluated every 10)", fontsize=11)
ax.plot([0, 120], [2.2, 2.2], color=INK, lw=3); ax.text(-3, 2.2, "uninterrupted", ha="right", va="center")
ax.plot([0, 50], [1.2, 1.2], color=INK, lw=3); ax.plot([50, 120], [1.2, 1.2], color=ACCENT, lw=3)
ax.text(-3, 1.2, "stopped at 50, resumed", ha="right", va="center")
ax.plot([50], [1.2], "o", color=ACCENT, ms=9); ax.text(50, 1.55, "stop + checkpoint at update 50", ha="center", fontsize=9, color=ACCENT)
for s_ in sorted(A):
    same = A[s_]["starts_sha"] == B[s_]["starts_sha"] and A[s_]["train_loss"] == B[s_]["train_loss"]
    ax.plot([s_], [2.2], "|", color=INK, ms=12); ax.plot([s_], [1.2], "|", color=ACCENT if s_ > 50 else INK, ms=12)
    ax.text(s_, 2.55, str(s_), ha="center", fontsize=8, color=MUTED)
    ax.text(s_, 0.62, "=" if same else "x", ha="center", fontsize=13, color="#15803d" if same else "#b91c1c")
ax.text(0, 0.05, "= : at that evaluation both runs had trained on the same sampled windows and logged the same losses.",
        fontsize=9, color=MUTED)
fig.savefig(os.path.join(OUT, "day4-resume-timeline.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

# 5. Friday: perplexity from the scoreboard, untrained -> unigram -> trained (log scale).
sb = open(os.path.join(EV, "scoreboard.txt")).read()
ppl = {k: float(re.search(rf"^{k}\s+[\d.]+\s+perplexity\s+([\d,.]+)", sb, re.M).group(1).replace(",", ""))
       for k in ("untrained", "unigram", "trained")}
PLOTTED["day5_perplexity"] = ppl
fig, ax = plt.subplots(figsize=(6.4, 3.6))
names = ["untrained\n(random weights)", "unigram\n(token frequency)", "trained\n(your model)"]
bars = ax.bar(names, list(ppl.values()), color=[MUTED, "#93c5fd", ACCENT]); ax.set_yscale("log")
for b, val in zip(bars, ppl.values()):
    ax.text(b.get_x() + b.get_width() / 2, val * 1.15, f"{val:,.1f}", ha="center", fontsize=10)
ax.set_ylabel("perplexity (log scale, lower is better)")
ax.set_title("Same 529,455 validation targets for all three", fontsize=11)
fig.savefig(os.path.join(OUT, "day5-perplexity.png"), dpi=200, bbox_inches="tight"); plt.close(fig)
# 6. Day 5: the full reference run's curves, from its own log (Air GPU, 6,000 updates).
rows = [r for r in map(json.loads, open(os.path.join(EV, "full-run-log.jsonl")))]
start = next(r for r in rows if r["event"] == "start")
ev = [r for r in rows if r["event"] == "eval"]
trained_whole = float(re.search(r"^trained\s+([\d.]+)", sb, re.M).group(1))
PLOTTED["day5_curve"] = {"steps": [0] + [r["step"] for r in ev], "train_loss": [None] + [r["train_loss"] for r in ev],
                         "val_64_windows": [start["val_ce"]] + [r["val_ce"] for r in ev], "scoreboard_whole_file": trained_whole}
fig, ax = plt.subplots(figsize=(8.5, 3.8))
ax.plot([r["step"] for r in ev], [r["train_loss"] for r in ev], marker="o", color=MUTED,
        label="training loss (the batch at that update)")
ax.plot([0] + [r["step"] for r in ev], [start["val_ce"]] + [r["val_ce"] for r in ev], marker="o", color=ACCENT,
        label="validation loss (64 fixed windows, sampled)")
ax.axhline(trained_whole, ls="--", color=INK, lw=1)
ax.text(3000, 0.6, f"dashed line: Day 5 scoreboard on the whole validation file, {trained_whole:.4f}", ha="center", fontsize=9)
ax.set_xlabel("update"); ax.set_ylabel("cross-entropy"); ax.set_ylim(0, 9)
ax.set_title("The full reference run (Air GPU, 6,000 updates)", fontsize=11); ax.legend(frameon=False, fontsize=9)
fig.savefig(os.path.join(OUT, "day5-training-curve.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

json.dump(PLOTTED, open(os.path.join(OUT, "plotted-data.json"), "w"), indent=1)
print("wrote", sorted(os.listdir(OUT)), "| ppl", ppl, "| day4 all same:", all(A[s]["starts_sha"] == B[s]["starts_sha"] for s in A))
