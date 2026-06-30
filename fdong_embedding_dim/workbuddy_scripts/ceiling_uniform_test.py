#!/usr/bin/env python3
"""
Ceiling experiment: does uniform data eliminate common directions?

Question: When data is uniform (all tokens equally frequent), what does
the embedding space look like? Do tokens still share a common subspace,
or does each token occupy an independent direction?

Setup: 12 tokens (4 groups × 3), simple Markov transitions.
Compare uniform (each group 25%) vs imbalanced (A:70%, B/C/D:10% each).
Model: d=4 linear tied-embedding.
"""
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import math

N_TOKENS = 12  # 4 groups × 3 tokens each
N_GROUPS = 4
TOKENS_PER_GROUP = 3

# Build per-group transitions: G0→G1, G1→G2, G2→G0
patterns = []
for g in range(N_GROUPS):
    base = g * TOKENS_PER_GROUP
    patterns.append((base, base+1))
    patterns.append((base+1, base+2))
    patterns.append((base+2, base))

ALL_PATTERNS = patterns  # 12 patterns total

class LinearTied(nn.Module):
    def __init__(self, d=4):
        super().__init__()
        self.E = nn.Embedding(N_TOKENS, d)
        nn.init.normal_(self.E.weight, std=0.02)
    def forward(self, x):
        h = self.E(x)
        return h @ self.E.weight.T


def analyze_embedding(model, label=""):
    """Collect embedding vectors and compute PCA / pairwise cos."""
    H = torch.stack([model.E.weight[i].detach() for i in range(N_TOKENS)])  # (12, d)
    Hc = H - H.mean(0)
    U, S, V = torch.svd(Hc)
    total = (S**2).sum()
    effrank = total**2 / max((S**4).sum(), 1e-30)
    # Pairwise cos sim per group
    cos_all = []
    for g in range(N_GROUPS):
        base = g * TOKENS_PER_GROUP
        for i in range(TOKENS_PER_GROUP):
            for j in range(i+1, TOKENS_PER_GROUP):
                a, b = H[base+i], H[base+j]
                cos_all.append((a @ b).item() / (a.norm().item() * b.norm().item() + 1e-8))
    # Cross-group cos
    cos_cross = []
    for g1 in range(N_GROUPS):
        for g2 in range(g1+1, N_GROUPS):
            for i in range(TOKENS_PER_GROUP):
                for j in range(TOKENS_PER_GROUP):
                    a = H[g1*TOKENS_PER_GROUP + i]
                    b = H[g2*TOKENS_PER_GROUP + j]
                    cos_cross.append((a @ b).item() / (a.norm().item() * b.norm().item() + 1e-8))

    print(f"\n{label}:")
    print(f"  SVD: σ = {S.numpy().round(3)}, effrank = {effrank:.1f}/{H.shape[1]}")
    print(f"  Σσ² ratios: {[round((s**2/total).item(),3) for s in S]}")
    print(f"  intra-group cos mean = {np.mean(cos_all):.3f}")
    print(f"  cross-group cos mean = {np.mean(cos_cross):.3f}")


def train(model, group_freqs, steps=1000):
    """group_freqs: weights per group, e.g. [0.7, 0.1, 0.1, 0.1]"""
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    group_weights = torch.tensor(group_freqs)
    for s in range(steps):
        g = torch.multinomial(group_weights, 1).item()
        pat_idx = np.random.RandomState(s).choice(3)  # 3 patterns per group
        seq = ALL_PATTERNS[g * 3 + pat_idx]
        x = torch.tensor([seq[0]])
        y = torch.tensor([seq[1]])
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()


if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)

    for uniform in [True, False]:
        freqs = [0.25, 0.25, 0.25, 0.25] if uniform else [0.7, 0.1, 0.1, 0.1]
        label = "UNIFORM" if uniform else "IMBALANCED (70/10/10/10)"
        model = LinearTied(d=4)
        for i in range(N_TOKENS):
            nn.init.normal_(model.E.weight[i:i+1], std=0.02)
        train(model, freqs)
        analyze_embedding(model, label)
