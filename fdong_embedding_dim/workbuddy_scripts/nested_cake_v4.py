import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

THE, SUN, MOON, BANANA, FRUIT, CAKE = range(6)
N_TOKENS = 6
PAT = [(THE,SUN),(THE,MOON),(MOON,CAKE),(BANANA,CAKE),(FRUIT,CAKE)]
PN = ["the→sun","the→moon","moon→cake","banana→cake","fruit→cake"]

def run(freqs, d=8, steps=4000):
    class M(nn.Module):
        def __init__(self): super().__init__(); self.E=nn.Embedding(N_TOKENS,d)
        def forward(self,x): return self.E(x)@self.E.weight.T
    model = M()
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    probs = torch.tensor([f/sum(freqs) for f in freqs])
    hist, grads = {p:[] for p in range(5)}, {}
    snap_steps = set(range(0,steps,200)) | {steps-1}
    for s in range(steps):
        p = torch.multinomial(probs,1).item()
        c,t = PAT[p]
        loss = F.cross_entropy(model(torch.tensor([c])),torch.tensor([t]))
        opt.zero_grad();loss.backward()
        if s in snap_steps:
            Eg = model.E.weight.grad.detach().clone()
            with torch.no_grad():
                H=model.E.weight.detach();Hc=H-H.mean(0);_,_,V=torch.svd(Hc)
            v0=V[:,0]
            grads[s]={}
            for pp,(cc,tt) in enumerate(PAT):
                g = Eg[cc]+Eg[tt]
                grads[s][pp]=(abs((g@v0).item()),g.norm().item())
        opt.step()
        if s%20==0:
            model.eval()
            for pi,(c,t) in enumerate(PAT):
                log=model(torch.tensor([c]));l=F.cross_entropy(log,torch.tensor([t])).item()
                pred=log.argmax().item();acc=1.0 if pred==t else 0.0
                hist[pi].append((s,l,acc))
            model.train()
    H=model.E.weight.detach();Hc=H-H.mean(0);_,S,V=torch.svd(Hc)
    return hist,H,S,V,grads,model

def conv(hist,th=0.9):
    for i,(_,_,a) in enumerate(hist):
        if np.mean([hist[j][2] for j in range(max(0,i-4),i+1)])>=th:
            return hist[i][0]
    return 9999

for nm,fr in [("Zipf",[6,6,1,1,1]),("Uniform",[3,3,3,3,3])]:
    hist,H,S,V,grads,model = run(fr)
    
    # ---- 1. Accuracy convergence ----
    print(f"\n{'='*70}")
    print(f"  {nm}: ACCURACY CONVERGENCE (running avg > 90%)")
    print(f"{'='*70}")
    print(f"  {'Pattern':<16s} {'step>90%':>10s} {'final acc':>10s}  {'the→? distrib':>14s}")
    print(f"  {'-'*54}")
    # the→sun vs the→moon distribution
    with torch.no_grad():
        the_log = model.E(torch.tensor([THE])) @ model.E.weight.T
        sun_p = torch.softmax(the_log, -1)[0, SUN].item()
        moon_p = torch.softmax(the_log, -1)[0, MOON].item()
    for pi in range(5):
        s90 = conv(hist[pi], 0.9)
        facc = np.mean([hist[pi][j][2] for j in range(-10,0)])
        extra = f"P(sun)={sun_p:.3f} P(moon)={moon_p:.3f}" if pi==0 else ""
        print(f"  {PN[pi]:<16s} {s90:>10d} {facc:>10.2f}  {extra:>14s}")
    all90 = max(conv(hist[p],0.9) for p in range(5))
    print(f"  {'ALL converge':<16s} {all90:>10d}")
    
    # ---- 2. Separability: trained on THIS distribution ----
    v0 = V[:,0]
    print(f"\n  SEPARABILITY (model trained on {nm} data)")
    print(f"  {'Pattern':<16s} {'||E(c)||':>8s} {'||E(t)||':>8s} {'common score':>12s} {'total score':>12s} {'common%':>9s}")
    print(f"  {'-'*68}")
    for pi,(c,t) in enumerate(PAT):
        cn = H[c].norm().item(); tn = H[t].norm().item()
        cs = (H[c]@v0).item() * (H[t]@v0).item()
        ts = (H[c]@H[t]).item()
        cpct = cs/max(abs(ts),1e-8)*100
        print(f"  {PN[pi]:<16s} {cn:>8.3f} {tn:>8.3f} {cs:>+12.2f} {ts:>+12.2f} {cpct:>8.1f}%")
    
    # ---- 3. Gradient ----
    print(f"\n  GRADIENT PROJECTION onto v0 (avg over training)")
    print(f"  {'Pattern':<16s} {'|grad·v0|':>10s} {'|grad|':>10s} {'common%':>9s}")
    print(f"  {'-'*46}")
    acc_g = {p:{'c':0,'t':0,'n':0} for p in range(5)}
    for s,gmap in grads.items():
        for p,(gc,gt) in gmap.items():
            acc_g[p]['c']+=gc;acc_g[p]['t']+=gt;acc_g[p]['n']+=1
    for p in range(5):
        ac=acc_g[p]['c']/acc_g[p]['n'];at=acc_g[p]['t']/acc_g[p]['n']
        print(f"  {PN[p]:<16s} {ac:>10.4f} {at:>10.4f} {ac/at*100:>8.1f}%")
