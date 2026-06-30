#!/usr/bin/env python3
"""Nested bottleneck: single-layer attention Transformer, matching Codex."""
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

def make_data():
    t = {'the':0,'sun':1,'moon':2,'a':3,'cake':4,'banana':5,'fruit':6,'<pad>':7}
    V = len(t); id2n = {v:k for k,v in t.items()}
    p = [[t['the'],t['sun'],t['<pad>']],[t['the'],t['moon'],t['<pad>']],
         [t['a'],t['moon'],t['cake']],[t['a'],t['banana'],t['cake']],[t['a'],t['fruit'],t['cake']]]
    m = [[1,0],[1,0],[1,1],[1,1],[1,1]]  # mask position 2 for <pad>
    return t,p,m,id2n

class LM(nn.Module):
    def __init__(self,V,d=8):
        super().__init__(); self.E=nn.Embedding(V,d);nn.init.normal_(self.E.weight,std=0.02)
        self.Wq=nn.Linear(d,d,bias=False);self.Wk=nn.Linear(d,d,bias=False)
        self.Wv=nn.Linear(d,d,bias=False);self.Wo=nn.Linear(d,d,bias=False)
    def forward(self,ids):
        B,T=ids.shape;x=self.E(ids);q=self.Wq(x);k=self.Wk(x);v=self.Wv(x)
        s=(q@k.transpose(-2,-1))/x.shape[-1]**0.5
        s=s+torch.triu(torch.ones(T,T,device=ids.device)*float('-inf'),1)
        o=F.softmax(s,dim=-1)@v;o=self.Wo(o);return (x+o)@self.E.weight.T

def train(patterns,masks,id2name,freqs,d=8,steps=2000,lr=0.05,reweight=False):
    V=len(id2name)
    ids=torch.tensor(patterns);ms=torch.tensor(masks,dtype=torch.float32)
    model=LM(V,d);opt=torch.optim.AdamW(model.parameters(),lr=lr)
    f=torch.tensor(freqs,dtype=torch.float32);fw=f/f.sum()
    if reweight:
        fw=torch.tensor([0.2,0.2,0.2,0.2,0.2])  # exact uniform reweight

    hist=[]
    for s in range(steps+1):
        logits=model(ids)
        tgt=ids[:,1:];lflat=logits[:,:2,:].reshape(-1,V);tflat=tgt.reshape(-1);mflat=ms.reshape(-1)
        pl=F.cross_entropy(lflat,tflat,reduction='none');pl=pl.reshape(5,2)
        pp=(pl*ms).sum(1)/ms.sum(1).clamp(min=1)
        loss=(pp*fw).sum()
        if s<steps: opt.zero_grad();loss.backward();opt.step()
        if s%100==0 or s==steps:
            model.eval()
            with torch.no_grad():
                ll=model(ids);ls=[]
                for pi in range(5):
                    lt=ll[pi:pi+1,:2,:].reshape(-1,V);tt=ids[pi:pi+1,1:].reshape(-1)
                    mt=ms[pi:pi+1].reshape(-1)
                    ls.append((F.cross_entropy(lt,tt,reduction='none')*mt).sum().item()/mt.sum().item())
                hist.append((s,*ls))
            model.train()

    model.eval();H=model.E.weight.detach();Hc=H-H.mean(0)
    _,S,V_svd=torch.svd(Hc);_,Bqk_S,_=torch.svd((model.Wq.weight.T@model.Wk.weight).detach().float())
    _,Bvo_S,_=torch.svd((model.Wo.weight@model.Wv.weight).detach().float())
    def effr(S): s2=(S**2).sum();return s2**2/max((S**4).sum(),1e-30)
    # Contextual HS
    with torch.no_grad():
        x=model.E(ids);q=model.Wq(x);k=model.Wk(x);v=model.Wv(x)
        a=F.softmax((q@k.transpose(-2,-1))/d**0.5+torch.triu(torch.ones(3,3)*float('-inf'),1),dim=-1)
        h=(x+model.Wo(a@v)).reshape(-1,d)
    _,Sh,_=torch.svd((h-h.mean(0)).float())
    H=model.E.weight.detach()
    return model,hist,S,Bqk_S,Bvo_S,Sh,effr,H

tokens,patterns,masks,id2name=make_data()
V,d=len(tokens),8

print(f"{'Dist':<16s} {'E effR':>7s} {'σ₁²%':>7s} {'Bqk σ₁²%':>9s} {'Bvo σ₁²%':>9s} {'HS effR':>7s} {'step200':>8s} {'step400':>8s} {'step800':>8s}")
print("-"*82)

for label,freqs,rew in [("Zipf",[6,6,1,1,1],False),("Uniform",[3,3,3,3,3],False),("Zipf+rew",[6,6,1,1,1],True)]:
    model,hist,S,Bqk_S,Bvo_S,Sh,er,H = train(patterns,masks,id2name,freqs,d=d,steps=800,lr=0.05,reweight=rew)
    s2s=(S**2).sum();bqk2=(Bqk_S**2).sum();bvo2=(Bvo_S**2).sum();sh2=(Sh**2).sum()
    er_param=er(S);er_bqk=er(Bqk_S);er_bvo=er(Bvo_S);er_hs=er(Sh)
    # Get a→moon→cake loss at key steps
    by_step={h[0]:h for h in hist}
    l200=by_step.get(200,[0]*6)[3] if 200 in by_step else 0
    l400=by_step.get(400,[0]*6)[3] if 400 in by_step else 0
    l800=by_step.get(800,[0]*6)[3] if 800 in by_step else 0
    print(f"{label:<16s} {er_param:>7.1f} {S[0]**2/s2s*100:>6.1f}% {Bqk_S[0]**2/bqk2*100:>8.1f}% {Bvo_S[0]**2/bvo2*100:>8.1f}% {er_hs:>7.1f} {l200:>8.3f} {l400:>8.3f} {l800:>8.3f}")
