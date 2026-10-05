"""Tuesday: one attention head, by hand, on four tokens.

Builds tiny random queries, keys and values, computes softmax(Q K^T / sqrt(d) + causal mask) V step by step,
prints the attention weights (zeros above the diagonal: no token reads the future), and checks the result
equals PyTorch's scaled_dot_product_attention, which model.py uses. Exits 1 if they differ.
"""
import math, sys

import torch
import torch.nn.functional as F

torch.manual_seed(0)
T, D = 4, 8
q, k, v = torch.randn(T, D), torch.randn(T, D), torch.randn(T, D)

scores = q @ k.T / math.sqrt(D)                               # how much each token "wants" each other token
mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)       # True above the diagonal = the future
scores = scores.masked_fill(mask, float("-inf"))
weights = scores.softmax(dim=-1)                               # each row sums to 1
by_hand = weights @ v

torch.set_printoptions(precision=2, sci_mode=False)
print("attention weights (row = reading token, column = token being read):")
print(weights)
library = F.scaled_dot_product_attention(q[None], k[None], v[None], is_causal=True)[0]
ok = torch.allclose(by_hand, library, atol=1e-6)
print(f"by hand == PyTorch: {ok}")
if not ok:
    sys.exit(1)
