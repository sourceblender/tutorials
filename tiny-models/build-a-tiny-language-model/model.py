"""Day 2: the small decoder language model used all week.

Token + learned position embeddings, pre-norm transformer blocks (causal multi-head attention + a 4x MLP), a
final LayerNorm, and an output head tied to the token embedding (one matrix both reads and writes tokens).

Initialisation: every Linear and Embedding weight starts at mean 0, std 0.02; LayerNorm keeps PyTorch's defaults.
Pass careful_init=False to see why that matters (day2_check.py --default-init): with PyTorch's default inits
(Embedding std 1) and the tied head, the starting loss on our validation windows measured about 171 to 173 (171.27
on the current data) rather than about 8.4.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    """One transformer block. Attention lets each position read earlier positions; the MLP thinks per position.
    Both ADD to the residual stream x rather than replacing it."""

    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d, bias=False), nn.GELU(), nn.Linear(4 * d, d, bias=False))

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, dim=2)
        q, k, v = (z.view(b, t, self.h, d // self.h).transpose(1, 2) for z in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # day2_attention.py does this by hand
        x = x + self.proj(a.transpose(1, 2).reshape(b, t, d))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab=4096, seq=256, d=256, layers=5, heads=4, careful_init=True):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(seq, d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight  # tied embeddings
        if careful_init:
            self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln(x))


def device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
