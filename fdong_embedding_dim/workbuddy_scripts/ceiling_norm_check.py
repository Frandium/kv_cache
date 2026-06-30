#!/usr/bin/env python3
"""Check: if intra-group cos=1.0, how are group-internal tokens distinguished?"""
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

N_TOKENS, N_GROUPS, TPG = 12, 4, 3
patterns = []
for g in range(N_GROUPS):
    base = g * TPG
    patterns.append((base, base+1)); patterns.append((base+1, base+2)); patterns.append((base+2, base))

class M(nn.Module):
    def __init__(self, d=4): super().__init__(); self.E = nn.Embedding(N_TOKENS,d)
    def forward(self,x): h=self.E(x); return h@self.E.weight.T

def analyze(model, label):
    H = model.E.weight.detach()
    print(f"\n{label}:")
    for g in range(N_GROUPS):
        base = g * TPG
        ns = [H[base+i].norm().item() for i in range(TPG)]
        cs = []
        for i in range(TPG):
            for j in range(i+1, TPG):
                cs.append((H[base+i]@H[base+j]).item() / (H[base+i].norm().item()*H[base+j].norm().item()+1e-8))
        print(f"  Group {g}: norms={[round(n,3) for n in ns]}, cos={[round(c,4) for c in cs]}")
        # What distinguishes them? Distance between G0→G1 and G1→G2
        print(f"    G0-G1 emb diff norm = {(H[base+0]-H[base+1]).norm():.4f}")
        print(f"    G1-G2 emb diff norm = {(H[base+1]-H[base+2]).norm():.4f}")
        print(f"    G2-G0 emb diff norm = {(H[base+2]-H[base+0]).norm():.4f}")

for uniform in [True, False]:
    model = M(4)
    freqs = [0.25]*4 if uniform else [0.7,0.1,0.1,0.1]
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    for s in range(1000):
        g = torch.multinomial(torch.tensor(freqs), 1).item()
        seq = patterns[g*3 + np.random.RandomState(s).choice(3)]
        loss = F.cross_entropy(model(torch.tensor([seq[0]])), torch.tensor([seq[1]]))
        opt.zero_grad(); loss.backward(); opt.step()
    analyze(model, "UNIFORM" if uniform else "IMBALANCED")
