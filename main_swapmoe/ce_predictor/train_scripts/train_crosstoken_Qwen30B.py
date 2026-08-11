# 2026.1.23
from transformers import get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import os
import numpy as np

from core.ce_predictor import CEPredictor
from utils import StreamingTokenizedData
from torch.optim import Adam
from torch.utils.data import DataLoader
from core.qwen3_moe_wrapper import Qwen3MoEForPredictor

# =========================
# Deep Decoder
# 结构升级：堆叠多层 (2层) Transformer Block，增加推理深度
# =========================

PREDICTOR_DEVICE = 'cuda:3'
NUM_EXPERTS = 128
ACTIVATED_EXPERTS = 8
HIDDEN_SIZE = 2048
LAYER_NUM = 48
TOPK = 8
NUM_LAYERS = 2  # ★★★ 设定层数 ★★★
WINDOW_SIZE = 256
DATA_DOMAIN_DIR = os.environ.get("DATA_DOMAIN_DIR") or None

def resolve_eval_out_dir(ckpt_dir: str) -> str:
    out_dir = os.environ.get("EVAL_OUT_DIR") or ckpt_dir
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def build_eval_npy_name(topk: int) -> str:
    tag = os.environ.get("EVAL_TAG", "crosstoken30B")
    ts = os.environ.get("EVAL_TIMESTAMP") or "unknown"
    return f"{tag}_top{int(topk)}_{ts}.npy"

def cal_topk_acc(pred_topk: torch.Tensor, router_topk: torch.Tensor, activated_experts: int = None):
    pred_topk = pred_topk.long()
    router_topk = router_topk.long()
    if activated_experts is None: activated_experts = router_topk.size(1)
    hits = (pred_topk.unsqueeze(-1) == router_topk.unsqueeze(1))
    inter_cnt = hits.any(dim=-1).sum(dim=-1).float()
    acc_per_sample = inter_cnt / activated_experts
    return acc_per_sample.mean().item()

class PreNormTransformerBlock(nn.Module):
    """
    采用 Pre-Norm 结构：x = x + SubLayer(Norm(x))
    这种结构在层数加深时更容易训练，梯度流更稳定。
    """
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 1. Attention 部分
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        
        # 2. FFN 部分
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Pre-Norm 逻辑: 先 Norm，再 Attention，最后残差连接
        
        # Sub-layer 1: Attention
        residual = x
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=mask, need_weights=False)
        x = residual + self.dropout1(attn_out)
        
        # Sub-layer 2: FFN
        residual = x
        x_norm = self.ln2(x)
        ffn_out = self.ffn(x_norm)
        x = residual + ffn_out
        
        return x

class DeepNextTokenPredictor(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 堆叠层
        self.layers = nn.ModuleList([
            PreNormTransformerBlock(hidden_size, num_heads, dropout) 
            for _ in range(num_layers)
        ])
        
        # Pre-Norm 结构通常建议在最后再加一层 Norm
        self.final_ln = nn.LayerNorm(hidden_size)
        
        self._cached_mask = None
        self._cached_mask_S = None
        
        # ★★★ 关键：初始化权重 ★★★
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        初始化通常能显著改善深层网络的收敛情况
        """
        if isinstance(module, nn.Linear):
            # 使用正态分布初始化，std 稍小一点
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def _causal_mask(self, S: int, device):
        if self._cached_mask is None or self._cached_mask_S != S or self._cached_mask.device != device:
            # 确保 mask 是 float('-inf') 而不是 bool，这样更稳定
            # 不过 nn.MultiheadAttention 支持 bool mask，这里保持原样即可
            m = torch.triu(torch.ones(S, S, dtype=torch.bool, device=device), diagonal=1)
            self._cached_mask = m
            self._cached_mask_S = S
        return self._cached_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = self._causal_mask(x.size(1), x.device)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        # 最后过一层 Norm
        x = self.final_ln(x)
        return x

def _stack_layer_list(x):
    if isinstance(x, (list, tuple)): return torch.stack(list(x), dim=0)
    return x

def train_one_epoch(epoch, train_data, test_data, llm, pred_model, next_attn, loss_ce, optimizer, ckpt_dir):
    pred_model.train()
    next_attn.train()

    for batch, (text, _, _) in enumerate(train_data, 1):
        optimizer.zero_grad()
        text = text.to(PREDICTOR_DEVICE)
        with torch.no_grad():
            output = llm(text, output_layer_input=True, output_attn_output=False, output_expert_label=True)

        layer_inputs_4d = _stack_layer_list(output["layer_input"]).float()
        router_4d = _stack_layer_list(output["expert_label"]).long()
        L, B, S, H = layer_inputs_4d.shape
        if router_4d.dim() == 3:
            K = router_4d.size(-1)
            router_4d = router_4d.view(L, B, S, K)
        elif router_4d.dim() == 4:
            K = router_4d.size(-1)
        else:
            raise RuntimeError(f"Unexpected router_4d shape: {tuple(router_4d.shape)}")
        
        if S < 2: continue

        x = layer_inputs_4d.reshape(L * B, S, H).to(PREDICTOR_DEVICE)
        y = next_attn(x).reshape(L, B, S, H) # [L, B, S, H]

        pred_next_4d = y[:, :, :-1, :]  
        label_4d = router_4d[:, :, 1:, :].to(PREDICTOR_DEVICE)

        N = B * (S - 1)
        pred_inputs = pred_next_4d.contiguous().view(L, N, H)
        router_index = label_4d.contiguous().view(L, N, K)

        pred_logits = pred_model(pred_inputs)

        loss = pred_model.__class__.cal_loss(
            loss_ce, router_index, pred_logits,
            expert_num=NUM_EXPERTS, activated_expert_num=ACTIVATED_EXPERTS, is_router_index=True
        )

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Batch {batch}, Loss {loss.item():.4f}")

        if batch % 200 == 0:
            ckpt_path = f"{ckpt_dir}/{epoch}.{batch}.pth"
            torch.save({"pred_model": pred_model.state_dict(), "next_attn": next_attn.state_dict()}, ckpt_path)
            test(test_data, llm, pred_model, next_attn, epoch, batch, ckpt_dir, topk=TOPK)
    ckpt_path = f"{ckpt_dir}/{epoch}.last.pth"
    torch.save({"pred_model": pred_model.state_dict(), "next_attn": next_attn.state_dict()}, ckpt_path)

@torch.no_grad()
def test(test_data, llm, pred_model, next_attn, epoch, batch, ckpt_dir, topk=4):
    pred_model.eval()
    next_attn.eval()
    layer_wise_test_acc = [[[] for _ in range(LAYER_NUM)] for _ in range(LAYER_NUM)]

    for test_batch, (text, _, _) in enumerate(test_data, 1):
        text = text.to(PREDICTOR_DEVICE)
        output = llm(text, output_layer_input=True, output_attn_output=False, output_expert_label=True)
        layer_inputs_4d = _stack_layer_list(output["layer_input"]).float()
        router_4d = _stack_layer_list(output["expert_label"]).long()
        L, B, S, H = layer_inputs_4d.shape
        if router_4d.dim() == 3:
            K = router_4d.size(-1)
            router_4d = router_4d.view(L, B, S, K)
        elif router_4d.dim() == 4:
            K = router_4d.size(-1)
        else:
            raise RuntimeError(f"Unexpected router_4d shape: {tuple(router_4d.shape)}")
        if S < 2: continue

        x = layer_inputs_4d.reshape(L * B, S, H).to(PREDICTOR_DEVICE)
        y = next_attn(x).reshape(L, B, S, H)
        
        pred_next_4d = y[:, :, :-1, :]
        router_index = router_4d[:, :, 1:, :].to(PREDICTOR_DEVICE).contiguous().view(L, B*(S-1), -1)
        pred_logits = pred_model(pred_next_4d.contiguous().view(L, B*(S-1), H))

        for start_layer in range(LAYER_NUM):
            for target_layer in range(LAYER_NUM):
                logits_st = pred_logits[start_layer, target_layer]
                pred_index = pred_model.__class__.cal_pred_index(logits_st, topk, expert_num=NUM_EXPERTS)
                gt_index = router_index[target_layer]
                acc = cal_topk_acc(pred_index, gt_index, activated_experts=ACTIVATED_EXPERTS)
                layer_wise_test_acc[start_layer][target_layer].append(acc)

        if test_batch >= 10: break

    for i in range(LAYER_NUM):
        for j in range(LAYER_NUM):
            if not layer_wise_test_acc[i][j]: continue
            acc_ij = sum(layer_wise_test_acc[i][j]) / len(layer_wise_test_acc[i][j])
            print(f"Epoch {epoch}, Batch {batch}, Layer-{i}-to-{j}-top{topk}-Accuracy {acc_ij:.4f} (Deep_{NUM_LAYERS}L)")

    acc_matrix = np.zeros((LAYER_NUM, LAYER_NUM), dtype=np.float32)
    for i in range(LAYER_NUM):
        for j in range(LAYER_NUM):
            vals = layer_wise_test_acc[i][j]
            acc_matrix[i, j] = (sum(vals) / len(vals)) if vals else np.nan
    eval_out_dir = resolve_eval_out_dir(ckpt_dir)
    np.save(os.path.join(eval_out_dir, build_eval_npy_name(topk)), acc_matrix)

def _pick_heads(h, p): 
    return p if h%p==0 else [x for x in [32,16,8,4,2,1] if h%x==0][0]

if __name__ == "__main__":
    moe_model = Qwen3MoEForPredictor("/mnt/workspace/Qwen3-30B-A3B", device=PREDICTOR_DEVICE, device_map="auto")
    moe_model.base.eval()
    cfg = moe_model.base.config
    HIDDEN_SIZE = cfg.hidden_size

    if DATA_DOMAIN_DIR is None:
        train_ds = StreamingTokenizedData(WINDOW_SIZE, is_test=False, repeat=False)
        test_ds = StreamingTokenizedData(WINDOW_SIZE, is_test=True, repeat=False)
    else:
        train_ds = StreamingTokenizedData(WINDOW_SIZE, is_test=False, folder_prefix=DATA_DOMAIN_DIR, repeat=False)
        test_ds = StreamingTokenizedData(WINDOW_SIZE, is_test=True, folder_prefix=DATA_DOMAIN_DIR, repeat=False)

    train_data = DataLoader(train_ds, batch_size=2, shuffle=False, num_workers=4)
    test_data = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=4)

    lr = 1e-4
    pred_model = CEPredictor(layers=LAYER_NUM, input_dim=HIDDEN_SIZE, num_experts=NUM_EXPERTS).to(PREDICTOR_DEVICE)
    
    num_heads = _pick_heads(HIDDEN_SIZE, int(getattr(cfg, "num_attention_heads", 16)))
    next_attn = DeepNextTokenPredictor(HIDDEN_SIZE, num_heads, num_layers=NUM_LAYERS).to(PREDICTOR_DEVICE)

    loss_ce = CEPredictor.get_loss_func()
    optimizer = Adam(list(pred_model.parameters()) + list(next_attn.parameters()), lr=lr)

    ckpt_dir = f'../checkpoints/CEPredictor_Qwen3MoE_A3B_lr{lr}_Deep_{NUM_LAYERS}L_top{TOPK}'
    os.makedirs(ckpt_dir, exist_ok=True)

    train_one_epoch(0, train_data, test_data, moe_model, pred_model, next_attn, loss_ce, optimizer, ckpt_dir)
