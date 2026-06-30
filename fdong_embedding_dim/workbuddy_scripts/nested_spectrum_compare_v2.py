import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

THE, SUN, MOON, BANANA, FRUIT, CAKE = range(6)
N_TOKENS = 6
PAT = [(THE,SUN),(THE,MOON),(MOON,CAKE),(BANANA,CAKE),(FRUIT,CAKE)]

def train(freqs, d=8, steps=4000):
    class M(nn.Module):
        def __init__(self): super().__init__(); self.E=nn.Embedding(N_TOKENS,d)
        def forward(self,x): return self.E(x)@self.E.weight.T
    model = M()
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    probs = torch.tensor([f/sum(freqs) for f in freqs])
    for s in range(steps):
        p = torch.multinomial(probs,1).item()
        loss = F.cross_entropy(model(torch.tensor([PAT[p][0]])),torch.tensor([PAT[p][1]]))
        opt.zero_grad();loss.backward();opt.step()
    return model

results = {}
for nm, fr in [("Zipf",[6,6,1,1,1]),("Uniform",[3,3,3,3,3])]:
    model = train(fr, d=8)
    E = model.E.weight.detach()
    Ec = E - E.mean(0)
    _, S, _ = torch.svd(Ec)
    results[nm] = S.cpu().numpy()

print(f"  {'Idx':>4s}  {'Zipf σ':>10s}  {'σ²%':>7s}  {'Uniform σ':>10s}  {'σ²%':>7s}  {'U/Z':>8s}")
print(f"  {'-'*55}")
for i in range(6):
    z = results['Zipf'][i]; u = results['Uniform'][i]
    z2 = z**2/(results['Zipf']**2).sum()*100
    u2 = u**2/(results['Uniform']**2).sum()*100
    print(f"  {i:>4d}  {z:>10.4f}  {z2:>6.1f}%  {u:>10.4f}  {u2:>6.1f}%  {u/z:>8.3f}")

# Frobenius norm
for nm in ['Zipf','Uniform']:
    fn = (results[nm]**2).sum()**0.5
    print(f"  {nm} ||E||_F = {fn:.4f}")
