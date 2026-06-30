#!/usr/bin/env python3
"""
Muon optimizer experiment: full-batch hypothesis.

Hypothesis: Muon and loss reweighting share the same principle — both correct
for frequency imbalance by equalizing per-direction update magnitudes.
  - Loss reweighting: proven to work perfectly when batch = full dataset
  - Muon: should also work perfectly when batch = full dataset
  - Both fail with mini-batch because rare features don't appear in every batch

Setup: 200 patterns, train = test (same 200 patterns, same token IDs).
Full batch = batch_size = 200. Model overfits but we're testing the optimization
principle, not generalization.

Usage:
  python3 workbuddy_scripts/muon_experiment.py --opt adam --loss_rew 0 --lr 3e-4
  python3 workbuddy_scripts/muon_experiment.py --opt muon --lr 0.02
  python3 workbuddy_scripts/muon_experiment.py --opt adam --loss_rew 0.5
"""

import argparse, json, math, os, sys, time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ==============================================================================
# Muon Optimizer
# ==============================================================================

def newton_schulz(G, steps=5):
    """Orthogonalize G such that all singular values → 1. G = U Σ V^T → U V^T."""
    assert G.ndim == 2
    X = G.float() / (G.norm() + 1e-7)
    if X.shape[0] > X.shape[1]:
        X = X.T
        transposed = True
    else:
        transposed = False
    for _ in range(steps):
        XTc = X.T @ X
        X = 0.5 * (3.0 * X - X @ XTc)
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.0,
                 ns_steps=5, nesterov=True):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        ns_steps=ns_steps, nesterov=nesterov)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, mu, wd, ns, nesterov = (group['lr'], group['momentum'],
                                          group['weight_decay'], group['ns_steps'],
                                          group['nesterov'])
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                state = self.state[p]
                if p.ndim >= 2:
                    if 'b' not in state: state['b'] = torch.zeros_like(p)
                    buf = state['b']
                    buf.mul_(mu).add_(grad)
                    update = grad.add(buf, alpha=mu) if nesterov else buf
                    update = newton_schulz(update, steps=ns)
                    p.mul_(1 - lr * wd)
                    p.sub_(update, alpha=lr)
                else:
                    if 'b' not in state: state['b'] = torch.zeros_like(p)
                    buf = state['b']
                    buf.mul_(mu).add_(grad)
                    p.mul_(1 - lr * wd)
                    p.sub_(buf, alpha=lr)


# ==============================================================================
# Full-batch fixed data (train = test, same 200 patterns, same token IDs)
# ==============================================================================

def make_full_data(n_patterns=200, uniform=False):
    """Generate patterns.

    uniform=False: 10 K tokens (30% of all positions, each ~3%), 490 R tokens
    uniform=True:  all 500 tokens uniformly distributed (no K/R distinction)
    """
    rng = np.random.RandomState(42)
    vocab, L = 500, 10

    if uniform:
        # All 500 tokens uniformly sampled — no frequency imbalance
        all_ids = rng.choice(vocab, size=n_patterns * L, replace=True).tolist()
        patterns = []
        idx = 0
        for pi in range(n_patterns):
            patterns.append(all_ids[idx:idx+L]); idx += L
        K_ids = set()  # no K/R distinction
        return patterns, K_ids, vocab, L, n_patterns

    # Original: 10 K + 490 R with frequency imbalance
    n_K = 10
    nKpp = 3
    nRpp = L - nKpp
    K_positions = [sorted(rng.choice(L, size=nKpp, replace=False).tolist())
                   for _ in range(n_patterns)]
    total_K = n_patterns * nKpp
    K_pool = sum(([k] * (total_K // n_K) for k in range(n_K)), [])
    rng.shuffle(K_pool)
    R_pool = rng.choice(vocab - n_K, size=n_patterns * nRpp, replace=True).tolist()
    R_pool = [r + n_K for r in R_pool]
    patterns = []
    ki = ri = 0
    for pi in range(n_patterns):
        seq = [0] * L
        for pos in K_positions[pi]:
            seq[pos] = K_pool[ki]; ki += 1
        for pos in range(L):
            if pos not in K_positions[pi]:
                seq[pos] = R_pool[ri]; ri += 1
        patterns.append(seq)
    return patterns, set(range(n_K)), vocab, L, n_patterns


class FixedDataset(torch.utils.data.Dataset):
    """Returns all patterns in order. n_samples = n_patterns for full batch."""
    def __init__(self, patterns):
        self.patterns = patterns
        self.L = len(patterns[0])
    def __len__(self): return len(self.patterns)
    def __getitem__(self, idx):
        s = self.patterns[idx]
        return torch.tensor(s, dtype=torch.long), torch.tensor(s, dtype=torch.long)


# ==============================================================================
# Model
# ==============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__(); self.weight = nn.Parameter(torch.ones(dim)); self.eps = eps
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight

def precompute_freqs_cis(dim, max_seq_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:dim//2].float() / dim))
    t = torch.arange(max_seq_len)
    return torch.polar(torch.ones_like(torch.outer(t, freqs)), torch.outer(t, freqs))

def apply_rotary_emb(x, freqs_cis):
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    f = freqs_cis[:x.shape[2]].unsqueeze(0).unsqueeze(0)
    return torch.view_as_real(xc * f).flatten(-2).type_as(x)

class CausalAttention(nn.Module):
    def __init__(self, d, nh, nkv, ms):
        super().__init__()
        self.nh, self.nkv = nh, nkv; self.hd = d // nh; self.sc = math.sqrt(self.hd)
        self.Wq = nn.Linear(d, nh*self.hd, bias=False)
        self.Wk = nn.Linear(d, nkv*self.hd, bias=False)
        self.Wv = nn.Linear(d, nkv*self.hd, bias=False)
        self.Wo = nn.Linear(nh*self.hd, d, bias=False)
        self.register_buffer('freqs', precompute_freqs_cis(self.hd, ms))
        self.register_buffer('mask', torch.triu(torch.full((ms, ms), float('-inf')), 1))
    def forward(self, h):
        B, T, D = h.shape
        q = self.Wq(h).reshape(B, T, self.nh, self.hd)
        k = self.Wk(h).reshape(B, T, self.nkv, self.hd)
        v = self.Wv(h).reshape(B, T, self.nkv, self.hd)
        q = apply_rotary_emb(q.permute(0,2,1,3), self.freqs)
        k = apply_rotary_emb(k.permute(0,2,1,3), self.freqs)
        v = v.permute(0,2,1,3)
        if self.nh > self.nkv:
            k = k.repeat_interleave(self.nh // self.nkv, dim=1)
            v = v.repeat_interleave(self.nh // self.nkv, dim=1)
        a = torch.matmul(q, k.transpose(-2,-1)) / self.sc + self.mask[:T,:T]
        o = torch.matmul(F.softmax(a, dim=-1), v)
        return self.Wo(o.permute(0,2,1,3).reshape(B,T,-1))

class MLP(nn.Module):
    def __init__(self, d, i):
        super().__init__()
        self.W1 = nn.Linear(d, i, bias=False)
        self.W2 = nn.Linear(i, d, bias=False)
        self.W3 = nn.Linear(d, i, bias=False)
    def forward(self, x):
        return self.W2(F.silu(self.W1(x)) * self.W3(x))

class Block(nn.Module):
    def __init__(self, d, nh, nkv, i, ms):
        super().__init__()
        self.an=RMSNorm(d); self.a=CausalAttention(d,nh,nkv,ms)
        self.mn=RMSNorm(d); self.m=MLP(d,i)
    def forward(self,h): h=h+self.a(self.an(h)); return h+self.m(self.mn(h))

class LM(nn.Module):
    def __init__(self, v, d=32, nl=1, nh=2, nkv=1, im=96, ms=12):
        super().__init__()
        self.e=nn.Embedding(v,d); nn.init.normal_(self.e.weight, std=1/math.sqrt(d))
        self.n=RMSNorm(d)
        self.l=nn.ModuleList([Block(d,nh,nkv,im,ms) for _ in range(nl)])
    def forward(self, ids):
        h = self.e(ids)
        for l in self.l: h = l(h)
        return self.n(h) @ self.e.weight.T
    def pc(self): return sum(p.numel() for p in self.parameters())


# ==============================================================================
# Training
# ==============================================================================

def train(args):
    device = torch.device('mps' if torch.backends.mps.is_available() else
                          'cuda' if torch.cuda.is_available() else 'cpu')

    # Data: train = test = same patterns, same token IDs
    patterns, K_ids, V, L, N = make_full_data(n_patterns=200, uniform=args.uniform)
    ds = FixedDataset(patterns)
    bs = args.batch_size if args.batch_size <= N else N
    loader = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True)
    is_full = (bs == N)

    # Token frequency for loss reweight
    tok_counts = defaultdict(int)
    for s in patterns:
        for t in s: tok_counts[t] += 1
    tok_weight = torch.ones(V, device=device)
    for t, c in tok_counts.items():
        tok_weight[t] = max(0.1, (1.0 / (c + 1)) ** args.loss_rew)

    print(f"Device: {device}  Opt: {args.opt}  loss_rew={args.loss_rew}")
    print(f"Data: {N} patterns, bs={bs} ({'FULL' if is_full else 'mini'})")
    print(f"  train=eval (same {N} patterns, same token IDs)")

    # Model
    model = LM(V, d=32, nh=2, nkv=1, im=96, ms=12).to(device)
    print(f"Model: {model.pc():,} params")

    # Optimizer
    if args.opt == 'muon':
        muon_p, other_p = [], []
        for n, p in model.named_parameters():
            (muon_p if p.ndim >= 2 else other_p).append(p)
        opt = Muon([
            {'params': muon_p, 'lr': args.lr, 'momentum': 0.95, 'weight_decay': args.wd},
            {'params': other_p, 'lr': args.lr, 'momentum': 0.95, 'weight_decay': 0.0},
        ])
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                                 weight_decay=args.wd)

    # Output
    tag = f"{args.opt}_rew{args.loss_rew}_bs{bs}{'_uniform' if args.uniform else ''}"
    out = os.path.join(args.output_dir, tag)
    os.makedirs(out, exist_ok=True)
    mpath = os.path.join(out, 'metrics.jsonl')

    model.train()
    step, t0 = 0, time.time()

    for epoch in range(200):
        for ids, targets in loader:
            if step >= args.steps: break
            ids, targets = ids.to(device), targets.to(device)

            logits = model(ids)
            loss_per_token = F.cross_entropy(
                logits.reshape(-1, V), targets.reshape(-1), reduction='none')
            # Loss reweight: multiply by token frequency weight
            w = tok_weight[targets.reshape(-1)]
            loss = (loss_per_token * w).mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1

            if step % args.log_every == 0:
                model.eval()
                K_l = R_l = K_n = R_n = 0.0
                with torch.no_grad():
                    for eids, etgt in loader:
                        eids, etgt = eids.to(device), etgt.to(device)
                        elogits = model(eids)
                        eloss = F.cross_entropy(
                            elogits.reshape(-1, V), etgt.reshape(-1), reduction='none')
                        eloss = eloss.reshape(bs, L)
                        for bi in range(eids.shape[0]):
                            for t in range(L):
                                tok = etgt[bi, t].item()
                                if not K_ids or tok in K_ids:
                                    K_l += eloss[bi, t].item(); K_n += 1
                                else:
                                    R_l += eloss[bi, t].item(); R_n += 1
                model.train()
                kk = K_l / max(K_n, 1); rr = R_l / max(R_n, 1)
                tok_s = (bs * L * args.log_every) / max(time.time() - t0, 1)
                print(f"step {step:>5d}  K={kk:.4f}  R={rr:.4f}  R/K={rr/kk:.1f}")
                with open(mpath, 'a') as f:
                    f.write(json.dumps({'step': step, 'K_loss': round(kk, 6),
                                        'R_loss': round(rr, 6)}) + '\n')
                t0 = time.time()
        if step >= args.steps: break

    # SVD — parameters + representations
    model.eval()
    print(f"{'='*55}")
    # Parameter spectra
    for name, W in [('Wq', model.l[0].a.Wq.weight.data),
                     ('Wk', model.l[0].a.Wk.weight.data),
                     ('Wo', model.l[0].a.Wo.weight.data)]:
        U, S_svd, _ = torch.linalg.svd(W.float(), full_matrices=False)
        S_svd = S_svd.cpu().numpy()
        er = np.sum(S_svd**2)**2 / max(np.sum(S_svd**4), 1e-30)
        print(f"  param {name}: σ₁={S_svd[0]:.3f}  effrank={er:.1f}")

    # Representation spectra: collect hidden states from all patterns
    all_h = []
    with torch.no_grad():
        for ids, _ in loader:
            ids = ids.to(device)
            h = model.e(ids)
            for l in model.l: h = l(h)
            h = model.n(h)
            all_h.append(h.reshape(-1, h.shape[-1]).cpu())
    H = torch.cat(all_h, dim=0)  # (N, d)
    # Separate K and R
    all_tokens = []
    for s in patterns: all_tokens.extend(s)
    all_tokens = torch.tensor(all_tokens)
    if K_ids:
        H_K = H[torch.isin(all_tokens, torch.tensor(list(K_ids)))]
        H_R = H[~torch.isin(all_tokens, torch.tensor(list(K_ids)))]
        rep_list = [('repr_K', H_K), ('repr_R', H_R), ('repr_all', H)]
    else:
        rep_list = [('repr_all', H)]

    for label, H_mat in rep_list:
        U, S_svd, _ = torch.linalg.svd(H_mat.float() - H_mat.float().mean(0),
                                        full_matrices=False)
        S_svd = S_svd.cpu().numpy()
        total = np.sum(S_svd**2)
        er = np.sum(S_svd**2)**2 / max(np.sum(S_svd**4), 1e-30)
        top3 = [round(float(S_svd[i]**2 / total), 3) for i in range(min(3, len(S_svd)))]
        print(f"  {label}: σ₁²/total={top3[0]:.3f}  effrank={er:.1f}  top3={top3}")
    print(f"Saved → {out}/")
    json.dump({}, open(os.path.join(out, 'done'), 'w'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--opt', default='adam', choices=['adam', 'muon'])
    p.add_argument('--loss_rew', type=float, default=0.0,
                   help='0=no reweight, 0.5=sqrt, 1.0=linear')
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--batch_size', type=int, default=200)
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--log_every', type=int, default=200)
    p.add_argument('--wd', type=float, default=0.1)
    p.add_argument('--uniform', action='store_true',
                   help='Uniform token distribution (no K/R imbalance)')
    p.add_argument('--output_dir', type=str,
                   default='/Users/bytedance/kv_cache/fdong_embedding_dim/outputs/muon_v2')
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    # Default Muon lr
    if args.opt == 'muon' and args.lr == 3e-4:
        args.lr = 0.02
    train(args)

