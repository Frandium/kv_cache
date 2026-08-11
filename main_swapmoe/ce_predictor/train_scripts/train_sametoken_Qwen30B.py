# 2026.1.23 feature cat 版本
from transformers import get_linear_schedule_with_warmup
import torch
import os
import numpy as np

from core.ce_predictor import CEPredictor
from utils import StreamingTokenizedData
from torch.optim import Adam
from torch.utils.data import DataLoader
from core.qwen3_moe_wrapper import Qwen3MoEForPredictor

# ================= Configuration =================
PREDICTOR_DEVICE = 'cuda:0'
NUM_EXPERTS = 128
ACTIVATED_EXPERTS = 8
HIDDEN_SIZE = 2048
LAYER_NUM = 48
WINDOW_SIZE = 256
DATA_DOMAIN_DIR = os.environ.get("DATA_DOMAIN_DIR") or None

# ★★★ 实验配置区 ★★★
class FeatureConfig:
    HISTORY_N = 4         # 历史窗口大小 (例如前8个token的均值)
    
    # 开关：设置为 True 启用对应特征
    USE_BASE_INPUT  = True # 基础：当前 token 的 layer input (必选)
    USE_PREV_ORACLE = False # 实验1：前一 token 的 oracle (attn output)
    USE_AVG_ORACLE  = True # 实验2：前 N token 的 oracle 均值
    USE_AVG_VALUE   = False # 实验3：前 N token 的 value  均值

# 自动计算输入维度
def get_input_dim():
    dim = 0
    if FeatureConfig.USE_BASE_INPUT:  dim += HIDDEN_SIZE
    if FeatureConfig.USE_PREV_ORACLE: dim += HIDDEN_SIZE
    if FeatureConfig.USE_AVG_ORACLE:  dim += HIDDEN_SIZE
    if FeatureConfig.USE_AVG_VALUE:   dim += HIDDEN_SIZE
    return dim

INPUT_DIM = get_input_dim()
print(f"Computed Predictor Input Dim: {INPUT_DIM} (Base: {HIDDEN_SIZE})")
# =================================================

def resolve_eval_out_dir(ckpt_dir: str) -> str:
    out_dir = os.environ.get("EVAL_OUT_DIR") or ckpt_dir
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def build_eval_npy_name(topk: int) -> str:
    tag = os.environ.get("EVAL_TAG", "sametoken30B")
    ts = os.environ.get("EVAL_TIMESTAMP") or "unknown"
    return f"{tag}_top{int(topk)}_{ts}.npy"

def cal_topk_acc(pred_topk: torch.Tensor, router_topk: torch.Tensor, activated_experts: int = None):
    pred_topk = pred_topk.long()
    router_topk = router_topk.long()
    if activated_experts is None: activated_experts = router_topk.size(1)
    hits = (pred_topk.unsqueeze(-1) == router_topk.unsqueeze(1))
    inter_cnt = hits.any(dim=-1).sum(dim=-1).float()
    return inter_cnt.mean().item() / activated_experts

def _stack_layer_list(x):
    if isinstance(x, (list, tuple)): return torch.stack(list(x), dim=0)
    return x

def compute_rich_features(layer_inputs, attn_outputs, value_states=None):
    """
    计算并拼接历史特征
    layer_inputs: [L, B, S, H] (视为 Value)
    attn_outputs: [L, B, S, H] (视为 Oracle)
    Returns:
        features: [L, B, S_valid, Dim]
        valid_slice: slice(N, None)
    """
    L, B, S, H = layer_inputs.shape
    N = FeatureConfig.HISTORY_N
    
    if S <= N: return None, None

    feats_list = []

    # 1. 基础特征: 当前 token (t)
    if FeatureConfig.USE_BASE_INPUT:
        # t range: [N, S]
        curr_H = layer_inputs[:, :, N:, :]
        feats_list.append(curr_H)

    # 准备 Cumsum 用于计算均值
    if FeatureConfig.USE_AVG_ORACLE or FeatureConfig.USE_AVG_VALUE:
        zeros = torch.zeros(L, B, 1, H, device=layer_inputs.device)
        
    # 2. 实验1: 前一 token Oracle (t-1)
    if FeatureConfig.USE_PREV_ORACLE:
        # t-1 range: [N-1, S-1]
        prev_O = attn_outputs[:, :, N-1:-1, :]
        feats_list.append(prev_O)

    # 3. 实验2: 前 N token Oracle 均值
    if FeatureConfig.USE_AVG_ORACLE:
        # O_cumsum: [L, B, S+1, H] (padded)
        O_cumsum = torch.cat([zeros, attn_outputs.cumsum(dim=2)], dim=2)
        # Sum[t-1] - Sum[t-N-1]
        # t-1 range: [N-1, S-1] -> idx range [N, S] in padded
        # t-N-1 range: [-1, S-N-1] -> idx range [0, S-N] in padded
        sum_top = O_cumsum[:, :, N : S, :]
        sum_bot = O_cumsum[:, :, 0 : S-N, :]
        avg_O = (sum_top - sum_bot) / N
        feats_list.append(avg_O)

    # 4. 实验3: 前 N token Value 均值
    if FeatureConfig.USE_AVG_VALUE:
        if value_states is None:
            raise ValueError("USE_AVG_VALUE=True but value_states is None. Enable output_value_states in wrapper call.")
        V_cumsum = torch.cat([zeros, value_states.cumsum(dim=2)], dim=2)
        sum_top = V_cumsum[:, :, N : S, :]
        sum_bot = V_cumsum[:, :, 0 : S-N, :]
        avg_V = (sum_top - sum_bot) / N
        feats_list.append(avg_V)

    # 拼接
    features = torch.cat(feats_list, dim=-1) # [L, B, S-N, Total_Dim]
    return features, slice(N, None)


def train(epoch, train_data, test_data, llm, pred_model, loss_func, optimizer, ckpt_dir):
    for batch, (text, _, _) in enumerate(train_data, 1):
        pred_model.train()
        optimizer.zero_grad()
        text = text.to(PREDICTOR_DEVICE)

        with torch.no_grad():
            output = llm(
                text,
                output_layer_input=True,
                output_attn_output=True,  # ★ 必须开启以获取 Oracle
                output_expert_label=True,
                output_value_states=FeatureConfig.USE_AVG_VALUE
            )

        layer_inputs = _stack_layer_list(output['layer_input']).float()  # [L, B, S, H]
        attn_outputs = _stack_layer_list(output['attn_output']).float()  # [L, B, S, H]
        router_4d = _stack_layer_list(output['expert_label'])            # [L, B, S, K]
        if router_4d.dim() == 3:
            L = router_4d.size(0)
            B = layer_inputs.size(1)
            S = layer_inputs.size(2)
            K = router_4d.size(-1)
            router_4d = router_4d.view(L, B, S, K)


        value_states = None
        if FeatureConfig.USE_AVG_VALUE:
            value_states = _stack_layer_list(output["value_states"]).float()
        # ★★★ 计算丰富特征 ★★★
        features_4d, valid_slice = compute_rich_features(layer_inputs, attn_outputs, value_states)
        
        if features_4d is None:
            print(f"[WARN] Batch {batch} too short for history window {FeatureConfig.HISTORY_N}")
            continue

        # 对齐 Label (切掉前 N 个)
        label_4d = router_4d[:, :, valid_slice, :]

        L, B, S_valid, Dim = features_4d.shape
        _, _, _, K_label = label_4d.shape
        N_samples = B * S_valid

        # Flatten
        pred_inputs = features_4d.contiguous().view(L, N_samples, Dim).to(PREDICTOR_DEVICE)
        router_index = label_4d.contiguous().view(L, N_samples, K_label).to(PREDICTOR_DEVICE)

        # Forward
        pred_logits = pred_model(pred_inputs) # [L, L, N, 2E]

        # Loss
        loss = pred_model.__class__.cal_loss(
            loss_func, router_index, pred_logits,
            expert_num=NUM_EXPERTS, activated_expert_num=ACTIVATED_EXPERTS, is_router_index=True
        )

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Batch {batch}, Loss {loss.item():.4f}")

        if batch % 200 == 0:
            ckpt_path = f"{ckpt_dir}/{epoch}.{batch}.pth"
            torch.save(pred_model.state_dict(), ckpt_path)
            test(test_data, llm, pred_model, epoch, batch, ckpt_dir)

    ckpt_path = f"{ckpt_dir}/{epoch}.last.pth"
    torch.save(pred_model.state_dict(), ckpt_path)


def test(test_data, llm, pred_model, epoch, batch, ckpt_dir, topk=8):
    pred_model.eval()
    layer_wise_test_acc = [[[] for _ in range(LAYER_NUM)] for _ in range(LAYER_NUM)]

    for test_batch, (text, _, _) in enumerate(test_data, 1):
        text = text.to(PREDICTOR_DEVICE)
        with torch.no_grad():
            output = llm(
                text, output_layer_input=True, output_attn_output=True, output_expert_label=True, output_value_states=FeatureConfig.USE_AVG_VALUE
            )
            
            layer_inputs = _stack_layer_list(output['layer_input']).float()
            attn_outputs = _stack_layer_list(output['attn_output']).float()
            router_4d = _stack_layer_list(output['expert_label'])
            if router_4d.dim() == 3:
                L = router_4d.size(0)
                B = layer_inputs.size(1)
                S = layer_inputs.size(2)
                K = router_4d.size(-1)
                router_4d = router_4d.view(L, B, S, K)

            value_states = None
            if FeatureConfig.USE_AVG_VALUE:
                value_states = _stack_layer_list(output["value_states"]).float()
            # ★★★ 计算丰富特征 ★★★
            features_4d, valid_slice = compute_rich_features(layer_inputs, attn_outputs, value_states)
            if features_4d is None: continue
            
            label_4d = router_4d[:, :, valid_slice, :]
            
            L, B, S_valid, Dim = features_4d.shape
            N_samples = B * S_valid
            
            pred_inputs = features_4d.contiguous().view(L, N_samples, Dim).to(PREDICTOR_DEVICE)
            router_index = label_4d.contiguous().view(L, N_samples, -1).to(PREDICTOR_DEVICE)

            pred_logits = pred_model(pred_inputs)

            for start_layer in range(LAYER_NUM):
                for target_layer in range(LAYER_NUM):
                    logits_st = pred_logits[start_layer, target_layer]
                    pred_idx = pred_model.__class__.cal_pred_index(logits_st, topk, NUM_EXPERTS)
                    gt_idx = router_index[target_layer]
                    acc = cal_topk_acc(pred_idx, gt_idx, ACTIVATED_EXPERTS)
                    layer_wise_test_acc[start_layer][target_layer].append(acc)

        if test_batch >= 10: break

    # 打印简报
    for i in range(LAYER_NUM):
        for j in range(LAYER_NUM):
            if not layer_wise_test_acc[i][j]: continue
            acc_ij = sum(layer_wise_test_acc[i][j]) / len(layer_wise_test_acc[i][j])
            print(f"Layer-{i}-to-{j}-top{topk}-Accuracy {acc_ij:.4f} (RichFeat)")

    acc_matrix = np.zeros((LAYER_NUM, LAYER_NUM), dtype=np.float32)
    for i in range(LAYER_NUM):
        for j in range(LAYER_NUM):
            vals = layer_wise_test_acc[i][j]
            acc_matrix[i, j] = sum(vals)/len(vals) if vals else np.nan
            
    eval_out_dir = resolve_eval_out_dir(ckpt_dir)
    np.save(os.path.join(eval_out_dir, build_eval_npy_name(topk)), acc_matrix)


if __name__ == "__main__":
    moe_model = Qwen3MoEForPredictor("/mnt/workspace/Qwen3-30B-A3B", device=PREDICTOR_DEVICE, device_map="auto")
    moe_model.base.eval()
    cfg = moe_model.base.config
    # 覆盖全局超参，避免写死不一致
    HIDDEN_SIZE = cfg.hidden_size
    LAYER_NUM = cfg.num_hidden_layers
    NUM_EXPERTS = cfg.num_experts
    ACTIVATED_EXPERTS = cfg.num_experts_per_tok

    # 重新计算 INPUT_DIM（因为 HIDDEN_SIZE 可能变）
    INPUT_DIM = get_input_dim()
    
    # Data Setup
    if DATA_DOMAIN_DIR is None:
        train_ds = StreamingTokenizedData(WINDOW_SIZE, is_test=False, repeat=False)
        test_ds = StreamingTokenizedData(WINDOW_SIZE, is_test=True, repeat=False)
    else:
        train_ds = StreamingTokenizedData(WINDOW_SIZE, is_test=False, folder_prefix=DATA_DOMAIN_DIR, repeat=False)
        test_ds = StreamingTokenizedData(WINDOW_SIZE, is_test=True, folder_prefix=DATA_DOMAIN_DIR, repeat=False)

    train_data = DataLoader(train_ds, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)
    test_data = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)

    lr = 1e-3
    ckpt_dir = f'../checkpoints/RichFeat_Qwen3_lr{lr}_Win{FeatureConfig.HISTORY_N}_Baseline'
    os.makedirs(ckpt_dir, exist_ok=True)

    # Predictor
    pred_model = CEPredictor(
        layers=LAYER_NUM,
        input_dim=INPUT_DIM,  # 自动使用计算出的维度 (例如 4 * 2048)
        num_experts=NUM_EXPERTS
    ).to(PREDICTOR_DEVICE)

    loss_func = CEPredictor.get_loss_func()
    optimizer = Adam(pred_model.parameters(), lr=lr)

    train(0, train_data, test_data, moe_model, pred_model, loss_func, optimizer, ckpt_dir)
