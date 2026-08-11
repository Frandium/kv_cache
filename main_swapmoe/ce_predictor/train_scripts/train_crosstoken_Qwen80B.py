# 2026.3.29
# Offline raw layer_input extraction from Qwen3-Next-80B-A3B-Instruct
# + offline deep cross-token predictor training
#
# This script:
#   Phase 1: extract raw layer_input + expert_label to .pt files
#   Phase 2: unload 80B model, then train:
#            raw layer_input -> next_attn -> CEPredictor
#
# Saved cache format:
#   {
#       "layer_input": [L, S, H],   # bf16
#       "labels":      [L, S, K],   # int16
#   }

import os
import gc
import glob
import shutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from core.ce_predictor import CEPredictor
from utils import StreamingTokenizedData
from core.qwen3_moe_wrapper import Qwen3NextMoEForPredictor

# ===== 8-bit Adam =====
try:
    import bitsandbytes as bnb
except ImportError as e:
    raise ImportError(
        "bitsandbytes is required for Adam8bit.\n"
        "Please install it first.\n"
        f"Original error: {e}"
    )

# =========================================================
# 你需要改的地方
# =========================================================
PREDICTOR_DEVICE = "cuda:0"
GATHER_DEVICE = "cuda:1"
DTYPE = torch.bfloat16

MODEL_PATH = "/mnt/workspace/Qwen3-Next-80B-A3B-Instruct"
DEVICE_MAP = "auto"
NUM_EXPERTS = 512

DATA_DOMAIN_DIR = os.environ.get("DATA_DOMAIN_DIR") or None
WINDOW_SIZE = 256

# 原始 layer_input 离线缓存目录
RAW_CACHE_DIR = "../offline_cache/qwen3next80b_deep2L"
os.makedirs(RAW_CACHE_DIR, exist_ok=True)

# 训练输出目录
CKPT_DIR = "../checkpoints/DeepPredictor_Qwen3Next80B_deep2L"
os.makedirs(CKPT_DIR, exist_ok=True)

# 是否重新提取
FORCE_REEXTRACT = False

# 先做小规模验证
MAX_EXTRACT_BATCHES_TRAIN = 4000
MAX_EXTRACT_BATCHES_TEST = 10

# extractor dataloader
BATCH_SIZE_EXTRACT = 2

# predictor 训练超参
LR = 1e-4
BATCH_SIZE_TRAIN = 1
BATCH_SIZE_EVAL = 1
NUM_WORKERS = 4
PIN_MEMORY = True
EPOCHS = 1

# deep predictor 结构
NUM_LAYERS = 2
DROPOUT = 0.1

# 评测
TOPK_EVAL = 12      # None -> 用 router topk
MAX_TEST_BATCHES = 10
DO_EVAL_DURING_TRAIN = False
SAVE_EVERY = 50

PRINT_MEM = False
# =========================================================

def resolve_eval_out_dir(ckpt_dir: str) -> str:
    out_dir = os.environ.get("EVAL_OUT_DIR") or ckpt_dir
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def build_eval_npy_name(topk: int) -> str:
    tag = os.environ.get("EVAL_TAG", "crosstoken80B")
    ts = os.environ.get("EVAL_TIMESTAMP") or "unknown"
    return f"{tag}_top{int(topk)}_{ts}.npy"


def mem(tag: str):
    if not PRINT_MEM:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(PREDICTOR_DEVICE))
    alloc = torch.cuda.memory_allocated(PREDICTOR_DEVICE) / 1024**3
    reserved = torch.cuda.memory_reserved(PREDICTOR_DEVICE) / 1024**3
    print(f"[mem] {tag}: alloc={alloc:.2f} GB | reserved={reserved:.2f} GB")


def load_torch_file(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def save_ckpt(path, pred_model, next_attn):
    state = {
        "pred_model": pred_model.state_dict(),
        "next_attn": next_attn.state_dict(),
    }
    torch.save(state, path)


def cal_topk_acc(pred_topk: torch.Tensor, router_topk: torch.Tensor, activated_experts: int = None):
    pred_topk = pred_topk.long()
    router_topk = router_topk.long()
    if activated_experts is None:
        activated_experts = router_topk.size(1)
    hits = (pred_topk.unsqueeze(-1) == router_topk.unsqueeze(1))
    inter_cnt = hits.any(dim=-1).sum(dim=-1).float()
    acc_per_sample = inter_cnt / activated_experts
    return acc_per_sample.mean().item()


def _stack_layer_list(x):
    if isinstance(x, (list, tuple)):
        return torch.stack(list(x), dim=0)
    return x


class OfflineRawLayerInputDataset(Dataset):
    """
    Each cached file stores:
      {
        "layer_input": [L,S,H]
        "labels":      [L,S,K]
      }
    """
    def __init__(self, cache_dir, split="train"):
        super().__init__()
        self.files = sorted(glob.glob(os.path.join(cache_dir, f"{split}_*.pt")))
        assert len(self.files) > 0, f"no cached files for split={split} in {cache_dir}"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        item = load_torch_file(self.files[idx], map_location="cpu")
        layer_input = item["layer_input"]   # [L,S,H]
        labels = item["labels"]             # [L,S,K]
        return layer_input, labels


class PreNormTransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size

        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=mask, need_weights=False)
        x = residual + self.dropout1(attn_out)

        residual = x
        x_norm = self.ln2(x)
        ffn_out = self.ffn(x_norm)
        x = residual + ffn_out
        return x


class DeepNextTokenPredictor(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            PreNormTransformerBlock(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.final_ln = nn.LayerNorm(hidden_size)

        self._cached_mask = None
        self._cached_mask_S = None

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def _causal_mask(self, S: int, device):
        if self._cached_mask is None or self._cached_mask_S != S or self._cached_mask.device != device:
            m = torch.triu(torch.ones(S, S, dtype=torch.bool, device=device), diagonal=1)
            self._cached_mask = m
            self._cached_mask_S = S
        return self._cached_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [LB, S, H]
        mask = self._causal_mask(x.size(1), x.device)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.final_ln(x)
        return x


def _pick_heads(h, preferred):
    if h % preferred == 0:
        return preferred
    for cand in [32, 16, 8, 4, 2, 1]:
        if h % cand == 0:
            return cand
    return 1


# =========================
# Phase 1: Offline extraction
# =========================
@torch.no_grad()
def extract_split(llm, loader, split, cache_dir, max_batches=None):
    os.makedirs(cache_dir, exist_ok=True)

    count = 0
    for batch_idx, (text, _, _) in enumerate(loader, 1):
        if max_batches is not None and count >= max_batches:
            break

        text = text.to(PREDICTOR_DEVICE, non_blocking=True)

        with torch.inference_mode():
            output = llm(
                text,
                output_layer_input=True,
                output_attn_output=False,   # raw deep predictor 不需要 attn_output
                output_expert_label=True,
                output_value_states=False,
            )

        layer_inputs = _stack_layer_list(output["layer_input"]).float()   # [L,B,S,H]
        router_4d = _stack_layer_list(output["expert_label"]).long()      # [L,B,S,K]

        if router_4d.dim() == 3:
            L = router_4d.size(0)
            B = layer_inputs.size(1)
            S = layer_inputs.size(2)
            K = router_4d.size(-1)
            router_4d = router_4d.view(L, B, S, K)

        L, B, S, H = layer_inputs.shape

        for b in range(B):
            layer_input_3d = layer_inputs[:, b].contiguous().to("cpu", dtype=torch.bfloat16)  # [L,S,H]
            labels_3d = router_4d[:, b].contiguous().to("cpu", dtype=torch.int16)              # [L,S,K]

            save_path = os.path.join(cache_dir, f"{split}_{count:06d}.pt")
            torch.save(
                {
                    "layer_input": layer_input_3d,
                    "labels": labels_3d,
                },
                save_path,
            )

            count += 1
            if count % 10 == 0:
                print(f"[EXTRACT] {split}: saved {count} samples")

    print(f"[EXTRACT] done split={split}, total saved={count}")


def run_offline_extraction(window_size: int):
    if FORCE_REEXTRACT and os.path.exists(RAW_CACHE_DIR):
        print(f"[EXTRACT] FORCE_REEXTRACT=True, removing {RAW_CACHE_DIR}")
        shutil.rmtree(RAW_CACHE_DIR)
        os.makedirs(RAW_CACHE_DIR, exist_ok=True)

    train_glob = glob.glob(os.path.join(RAW_CACHE_DIR, "train_*.pt"))
    test_glob = glob.glob(os.path.join(RAW_CACHE_DIR, "test_*.pt"))
    if (not FORCE_REEXTRACT) and len(train_glob) > 0 and len(test_glob) > 0:
        print("[EXTRACT] cache exists, skip extraction")
        return

    moe_model = Qwen3NextMoEForPredictor(
        MODEL_PATH,
        device=GATHER_DEVICE,
        dtype=DTYPE,
        device_map=DEVICE_MAP,
        trust_remote_code=True,
        strict_attn=True,
        verbose=True,
    )
    moe_model.base.eval()

    if DATA_DOMAIN_DIR is None:
        train_ds = StreamingTokenizedData(window_size, is_test=False, repeat=False)
        test_ds  = StreamingTokenizedData(window_size, is_test=True, repeat=False)
    else:
        train_ds = StreamingTokenizedData(window_size, is_test=False, folder_prefix=DATA_DOMAIN_DIR, repeat=False)
        test_ds  = StreamingTokenizedData(window_size, is_test=True,  folder_prefix=DATA_DOMAIN_DIR, repeat=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE_EXTRACT,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE_EXTRACT,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    extract_split(
        moe_model,
        train_loader,
        split="train",
        cache_dir=RAW_CACHE_DIR,
        max_batches=MAX_EXTRACT_BATCHES_TRAIN,
    )
    extract_split(
        moe_model,
        test_loader,
        split="test",
        cache_dir=RAW_CACHE_DIR,
        max_batches=MAX_EXTRACT_BATCHES_TEST,
    )

    print("[EXTRACT] deleting 80B model from memory...")
    del moe_model
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            try:
                print(f"[cuda:{i}] alloc={torch.cuda.memory_allocated(i)/1024**3:.2f} GB | reserved={torch.cuda.memory_reserved(i)/1024**3:.2f} GB")
            except Exception:
                pass


# =========================
# Phase 2: Offline training
# =========================
def train_one_epoch_offline(epoch, train_loader, test_loader, pred_model, next_attn,
                            loss_ce, optimizer, ckpt_dir, num_experts, activated_experts,
                            layer_num, topk_eval):
    pred_model.train()
    next_attn.train()

    for batch, (layer_input, labels) in enumerate(train_loader, 1):
        optimizer.zero_grad(set_to_none=True)

        # layer_input: [B,L,S,H], labels: [B,L,S,K]
        layer_input = layer_input.to(PREDICTOR_DEVICE, non_blocking=True).float()
        labels = labels.to(PREDICTOR_DEVICE, non_blocking=True).long()

        B, L, S, H = layer_input.shape
        _, _, _, K = labels.shape

        # [B,L,S,H] -> [L,B,S,H]
        layer_input_4d = layer_input.permute(1, 0, 2, 3).contiguous()
        router_4d = labels.permute(1, 0, 2, 3).contiguous()

        if S < 2:
            print(f"[WARN] skip batch {batch}: S < 2")
            continue

        # 原始 deep predictor 逻辑
        x = layer_input_4d.reshape(L * B, S, H).to(PREDICTOR_DEVICE)
        y = next_attn(x).reshape(L, B, S, H)

        pred_next_4d = y[:, :, :-1, :]
        label_4d = router_4d[:, :, 1:, :]

        N = B * (S - 1)
        pred_inputs = pred_next_4d.contiguous().view(L, N, H)
        router_index = label_4d.contiguous().view(L, N, K)

        mem(f"batch {batch} before pred forward")
        pred_logits = pred_model(pred_inputs)
        mem(f"batch {batch} after pred forward")

        loss = pred_model.__class__.cal_loss(
            loss_ce,
            router_index,
            pred_logits,
            expert_num=num_experts,
            activated_expert_num=activated_experts,
            is_router_index=True,
        )
        mem(f"batch {batch} after loss")

        loss.backward()
        mem(f"batch {batch} after backward")

        optimizer.step()
        mem(f"batch {batch} after optimizer.step")

        print(f"[TRAIN] Epoch {epoch}, Batch {batch}, Loss {loss.item():.4f}")

        if batch % SAVE_EVERY == 0:
            ckpt_path = os.path.join(ckpt_dir, f"{epoch}.{batch}.pth")
            save_ckpt(ckpt_path, pred_model, next_attn)
            print("[SAVE]", ckpt_path)

            if DO_EVAL_DURING_TRAIN:
                test_offline(
                    test_loader=test_loader,
                    pred_model=pred_model,
                    next_attn=next_attn,
                    epoch=epoch,
                    batch=batch,
                    ckpt_dir=ckpt_dir,
                    num_experts=num_experts,
                    activated_experts=activated_experts,
                    layer_num=layer_num,
                    topk=topk_eval,
                )

    ckpt_path = os.path.join(ckpt_dir, f"{epoch}.last.pth")
    save_ckpt(ckpt_path, pred_model, next_attn)
    print("[SAVE]", ckpt_path)


@torch.no_grad()
def test_offline(test_loader, pred_model, next_attn, epoch, batch, ckpt_dir,
                 num_experts, activated_experts, layer_num, topk=8):
    pred_model.eval()
    next_attn.eval()
    layer_wise_test_acc = [[[] for _ in range(layer_num)] for _ in range(layer_num)]

    for test_batch, (layer_input, labels) in enumerate(test_loader, 1):
        layer_input = layer_input.to(PREDICTOR_DEVICE, non_blocking=True).float()
        labels = labels.to(PREDICTOR_DEVICE, non_blocking=True).long()

        B, L, S, H = layer_input.shape
        _, _, _, K = labels.shape

        layer_input_4d = layer_input.permute(1, 0, 2, 3).contiguous()
        router_4d = labels.permute(1, 0, 2, 3).contiguous()

        if S < 2:
            continue

        x = layer_input_4d.reshape(L * B, S, H).to(PREDICTOR_DEVICE)
        y = next_attn(x).reshape(L, B, S, H)

        pred_next_4d = y[:, :, :-1, :]
        router_index = router_4d[:, :, 1:, :].contiguous().view(L, B * (S - 1), K)
        pred_logits = pred_model(pred_next_4d.contiguous().view(L, B * (S - 1), H))

        for start_layer in range(layer_num):
            for target_layer in range(layer_num):
                logits_st = pred_logits[start_layer, target_layer]
                pred_index = pred_model.__class__.cal_pred_index(
                    logits_st, topk, expert_num=num_experts
                )
                gt_index = router_index[target_layer]
                acc = cal_topk_acc(pred_index, gt_index, activated_experts=activated_experts)
                layer_wise_test_acc[start_layer][target_layer].append(acc)

        if test_batch >= MAX_TEST_BATCHES:
            break

    acc_matrix = np.zeros((layer_num, layer_num), dtype=np.float32)
    for i in range(layer_num):
        for j in range(layer_num):
            vals = layer_wise_test_acc[i][j]
            acc_matrix[i, j] = (sum(vals) / len(vals)) if vals else np.nan

    eval_out_dir = resolve_eval_out_dir(ckpt_dir)
    out_path = os.path.join(eval_out_dir, build_eval_npy_name(topk))
    np.save(out_path, acc_matrix)
    print("[TEST] saved =>", out_path)


def run_predictor_training():
    # infer shapes from one cached file
    sample_path = sorted(glob.glob(os.path.join(RAW_CACHE_DIR, "train_*.pt")))[0]
    sample = load_torch_file(sample_path, map_location="cpu")
    layer_input = sample["layer_input"]   # [L,S,H]
    labels = sample["labels"]             # [L,S,K]

    layer_num = int(layer_input.shape[0])
    hidden_size = int(layer_input.shape[-1])
    activated_experts = int(labels.shape[-1])

    topk_eval = activated_experts if TOPK_EVAL is None else TOPK_EVAL

    print("[OFFLINE DATA] layer_num =", layer_num,
          "| hidden_size =", hidden_size,
          "| activated_experts =", activated_experts)
    print("[TRAIN] offline raw-layer-input cross-token deep predictor")
    print("[OPT] Adam8bit | LR =", LR)

    train_ds = OfflineRawLayerInputDataset(RAW_CACHE_DIR, split="train")
    test_ds = OfflineRawLayerInputDataset(RAW_CACHE_DIR, split="test")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=True,
        num_workers=NUM_WORKERS,   # 先保守
        pin_memory=PIN_MEMORY,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE_EVAL,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    pred_model = CEPredictor(
        layers=layer_num,
        input_dim=hidden_size,
        num_experts=NUM_EXPERTS,
    ).to(PREDICTOR_DEVICE)

    num_heads = _pick_heads(hidden_size, 16)
    next_attn = DeepNextTokenPredictor(
        hidden_size,
        num_heads,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(PREDICTOR_DEVICE)

    loss_ce = CEPredictor.get_loss_func()
    optimizer = bnb.optim.Adam8bit(
        list(pred_model.parameters()) + list(next_attn.parameters()),
        lr=LR,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    for epoch in range(EPOCHS):
        train_one_epoch_offline(
            epoch=epoch,
            train_loader=train_loader,
            test_loader=test_loader,
            pred_model=pred_model,
            next_attn=next_attn,
            loss_ce=loss_ce,
            optimizer=optimizer,
            ckpt_dir=CKPT_DIR,
            num_experts=NUM_EXPERTS,
            activated_experts=activated_experts,
            layer_num=layer_num,
            topk_eval=topk_eval,
        )

    print("[DONE] training finished.")
    print("建议单独再跑一个 eval 脚本，避免训练后直接 full-matrix eval 叠加显存。")


def main():
    # Phase 1: offline extraction
    run_offline_extraction(WINDOW_SIZE)

    # Phase 2: offline training
    run_predictor_training()


if __name__ == "__main__":
    main()
