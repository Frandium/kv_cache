import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

# ===== Experiment 1: shared tokens (nested) =====
# the→sun, the→moon, moon→cake, banana→cake, fruit→cake (5 patterns, 6 tokens)
THE,SUN,MOON,BANANA,FRUIT,CAKE = range(6)
PAT_NESTED = [(THE,SUN),(THE,MOON),(MOON,CAKE),(BANANA,CAKE),(FRUIT,CAKE)]
D1 = 6

# ===== Experiment 2: split tokens (no sharing) =====
# the_1→sun, the_2→moon_1, moon_2→cake_1, banana→cake_2, fruit→cake_3
# 5 patterns, 10 independent tokens (each pattern has its own tokens)
D2 = 10  # tokens

# Zipf distribution: [6,6,1,1,1], same for both
def train(N_TOKENS, patterns, freqs, d=8, steps=4000):
    class M(nn.Module):
        def __init__(self): super().__init__(); self.E = nn.Embedding(N_TOKENS, d)
        def forward(self, x): return self.E(x) @ self.E.weight.T
    model = M()
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    probs = torch.tensor([f/sum(freqs) for f in freqs])
    for s in range(steps):
        p = torch.multinomial(probs, 1).item()
        loss = F.cross_entropy(model(torch.tensor([patterns[p][0]])),
                               torch.tensor([patterns[p][1]]))
        opt.zero_grad(); loss.backward(); opt.step()
    E = model.E.weight.detach()
    Ec = E - E.mean(0)
    _, S, _ = torch.svd(Ec)
    return S.cpu().numpy()

# Nested (6 tokens)
S_nest = train(D1, PAT_NESTED, [6,6,1,1,1])

# Split (10 tokens, no sharing)
PAT_SPLIT = [(0,1),(2,3),(4,5),(6,7),(8,9)]  # each pattern has unique token pair
S_split = train(D2, PAT_SPLIT, [6,6,1,1,1])

print(f"  {'Idx':>4s}  {'Nested(6t) σ':>12s}  {'σ²%':>7s}  {'Split(10t) σ':>12s}  {'σ²%':>7s}")
print(f"  {'-'*52}")
for i in range(6):
    n = S_nest[i] if i < len(S_nest) else 0
    s = S_split[i] if i < len(S_split) else 0
    n2 = n**2 / (S_nest**2).sum() * 100
    s2 = s**2 / (S_split**2).sum() * 100
    print(f"  {i:>4d}  {n:>12.4f}  {n2:>6.1f}%  {s:>12.4f}  {s2:>6.1f}%")

eff_n = (S_nest**2).sum()**2 / max((S_nest**4).sum(), 1e-30)
eff_s = (S_split**2).sum()**2 / max((S_split**4).sum(), 1e-30)
print(f"\n  Nested ||E||_F = {(S_nest**2).sum()**0.5:.3f}  effR = {eff_n:.1f}/8")
print(f"  Split  ||E||_F = {(S_split**2).sum()**0.5:.3f}  effR = {eff_s:.1f}/8")
