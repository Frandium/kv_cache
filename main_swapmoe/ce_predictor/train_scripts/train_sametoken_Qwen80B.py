# 2026.3.29
# Offline feature extraction from Qwen3-Next-80B-A3B-Instruct
# + full-matrix 80B predictor training with Adam8bit
import os
import gc
import math
import glob
import shutil
import numpy as np
import torch
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

# 你可以先把 Tokenized_data 绑到目标数据集上（比如 WMT）
DATA_DOMAIN_DIR = os.environ.get("DATA_DOMAIN_DIR") or None
WINDOW_SIZE = 256

# 离线特征缓存目录
FEATURE_CACHE_DIR = "../offline_cache/qwen3next80b_richfeat"
os.makedirs(FEATURE_CACHE_DIR, exist_ok=True)

# 训练输出目录
CKPT_DIR = "../checkpoints/RichFeat_Qwen3Next80B_feature"
os.makedirs(CKPT_DIR, exist_ok=True)

# 是否重新提取
FORCE_REEXTRACT = False

# 先做小规模验证，别一上来全量
MAX_EXTRACT_BATCHES_TRAIN = 4000
MAX_EXTRACT_BATCHES_TEST = 10

# predictor 训练超参
LR = 1e-3
BATCH_SIZE_TRAIN = 1        # 离线训练也建议先 1
BATCH_SIZE_EXTRACT = 2
NUM_WORKERS = 4
PIN_MEMORY = True
EPOCHS = 1
TOPK_EVAL = 12            # None -> 用 router topk
PRINT_MEM = False
# =========================================================

def resolve_eval_out_dir(ckpt_dir: str) -> str:
    out_dir = os.environ.get("EVAL_OUT_DIR") or ckpt_dir
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def build_eval_npy_name(topk: int) -> str:
    tag = os.environ.get("EVAL_TAG", "sametoken80B")
    ts = os.environ.get("EVAL_TIMESTAMP") or "unknown"
    return f"{tag}_top{int(topk)}_{ts}.npy"


# ================= Feature Config =================
class FeatureConfig:
    HISTORY_N = 4
    USE_BASE_INPUT  = True
    USE_PREV_ORACLE = False
    USE_AVG_ORACLE  = True
    USE_AVG_VALUE   = False


# ================= Utils =================
def mem(tag: str):
    if not PRINT_MEM:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(PREDICTOR_DEVICE))
    alloc = torch.cuda.memory_allocated(PREDICTOR_DEVICE) / 1024**3
    reserved = torch.cuda.memory_reserved(PREDICTOR_DEVICE) / 1024**3
    print(f"[mem] {tag}: alloc={alloc:.2f} GB | reserved={reserved:.2f} GB")


def _stack_layer_list(x):
    if isinstance(x, (list, tuple)):
        return torch.stack(list(x), dim=0)
    return x


def cal_topk_acc(pred_topk: torch.Tensor, router_topk: torch.Tensor, activated_experts: int = None):
    pred_topk = pred_topk.long()
    router_topk = router_topk.long()
    if activated_experts is None:
        activated_experts = router_topk.size(1)
    hits = (pred_topk.unsqueeze(-1) == router_topk.unsqueeze(1))
    inter_cnt = hits.any(dim=-1).sum(dim=-1).float()
    return inter_cnt.mean().item() / activated_experts


def compute_rich_features(layer_inputs, attn_outputs, value_states=None):
    """
    layer_inputs: [L,B,S,H]
    attn_outputs: [L,B,S,H]
    return:
      features: [L,B,S-N,Dim]
      valid_slice: slice(N,None)
    """
    L, B, S, H = layer_inputs.shape
    N = FeatureConfig.HISTORY_N
    if S <= N:
        return None, None

    feats_list = []

    if FeatureConfig.USE_BASE_INPUT:
        feats_list.append(layer_inputs[:, :, N:, :])

    if FeatureConfig.USE_AVG_ORACLE or FeatureConfig.USE_AVG_VALUE:
        zeros = torch.zeros(L, B, 1, H, device=layer_inputs.device)

    if FeatureConfig.USE_PREV_ORACLE:
        feats_list.append(attn_outputs[:, :, N-1:-1, :])

    if FeatureConfig.USE_AVG_ORACLE:
        O_cumsum = torch.cat([zeros, attn_outputs.cumsum(dim=2)], dim=2)
        sum_top = O_cumsum[:, :, N:S, :]
        sum_bot = O_cumsum[:, :, 0:S-N, :]
        avg_O = (sum_top - sum_bot) / N
        feats_list.append(avg_O)

    if FeatureConfig.USE_AVG_VALUE:
        if value_states is None:
            raise ValueError("USE_AVG_VALUE=True but value_states is None.")
        V_cumsum = torch.cat([zeros, value_states.cumsum(dim=2)], dim=2)
        sum_top = V_cumsum[:, :, N:S, :]
        sum_bot = V_cumsum[:, :, 0:S-N, :]
        avg_V = (sum_top - sum_bot) / N
        feats_list.append(avg_V)

    features = torch.cat(feats_list, dim=-1)
    return features, slice(N, None)


def get_input_dim(hidden_size: int):
    dim = 0
    if FeatureConfig.USE_BASE_INPUT:
        dim += hidden_size
    if FeatureConfig.USE_PREV_ORACLE:
        dim += hidden_size
    if FeatureConfig.USE_AVG_ORACLE:
        dim += hidden_size
    if FeatureConfig.USE_AVG_VALUE:
        dim += hidden_size
    return dim


# ================= Offline feature dataset =================
class OfflineFeatureDataset(Dataset):
    """
    每个缓存文件保存：
      {
        "features": [L, S_valid, D]   (bf16 on CPU)
        "labels":   [L, S_valid, K]   (int16/int32 on CPU)
      }
    """
    def __init__(self, cache_dir, split="train"):
        super().__init__()
        self.cache_dir = cache_dir
        self.files = sorted(glob.glob(os.path.join(cache_dir, f"{split}_*.pt")))
        assert len(self.files) > 0, f"no cached feature files for split={split} in {cache_dir}"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        item = torch.load(self.files[idx], map_location="cpu")
        features = item["features"]   # [L,S,D]
        labels = item["labels"]       # [L,S,K]
        return features, labels


# ================= Extraction =================
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
                output_attn_output=True,
                output_expert_label=True,
                output_value_states=FeatureConfig.USE_AVG_VALUE,
            )

        layer_inputs = _stack_layer_list(output["layer_input"]).float()   # [L,B,S,H]
        attn_outputs = _stack_layer_list(output["attn_output"]).float()   # [L,B,S,H]
        router_4d = _stack_layer_list(output["expert_label"]).long()      # [L,B,S,K]

        value_states = None
        if FeatureConfig.USE_AVG_VALUE:
            value_states = _stack_layer_list(output["value_states"]).float()

        feats_4d, valid_slice = compute_rich_features(layer_inputs, attn_outputs, value_states)
        if feats_4d is None:
            print(f"[WARN] skip batch {batch_idx}: too short")
            continue

        label_4d = router_4d[:, :, valid_slice, :]   # [L,B,S_valid,K]
        # L, B, S_valid, D = feats_4d.shape

        # 这里默认 BATCH_SIZE_EXTRACT=1
        # 如果你以后想支持 B>1，可以拆成多个 sample 文件
        # assert B == 1, f"Current extractor expects batch_size=1, got B={B}"

        # features_3d = feats_4d[:, 0].contiguous().to("cpu", dtype=torch.bfloat16)   # [L,S,D]
        # labels_3d = label_4d[:, 0].contiguous().to("cpu", dtype=torch.int16)         # [L,S,K]

        # save_path = os.path.join(cache_dir, f"{split}_{count:06d}.pt")
        # torch.save(
        #     {
        #         "features": features_3d,
        #         "labels": labels_3d,
        #     },
        #     save_path,
        # )

        # count += 1
        # if count % 10 == 0:
        #     print(f"[EXTRACT] {split}: saved {count} samples")

        L, B, S_valid, D = feats_4d.shape

        # 支持 batch 内多个样本，逐个落盘
        for b in range(B):
            features_3d = feats_4d[:, b].contiguous().to("cpu", dtype=torch.bfloat16)   # [L,S,D]
            labels_3d = label_4d[:, b].contiguous().to("cpu", dtype=torch.int16)         # [L,S,K]

            save_path = os.path.join(cache_dir, f"{split}_{count:06d}.pt")
            torch.save(
                {
                    "features": features_3d,
                    "labels": labels_3d,
                },
                save_path,
            )

            count += 1

            if count % 10 == 0:
                print(f"[EXTRACT] {split}: saved {count} samples")

    print(f"[EXTRACT] done split={split}, total saved={count}")


def run_offline_extraction(window_size: int):
    if FORCE_REEXTRACT and os.path.exists(FEATURE_CACHE_DIR):
        print(f"[EXTRACT] FORCE_REEXTRACT=True, removing {FEATURE_CACHE_DIR}")
        shutil.rmtree(FEATURE_CACHE_DIR)
        os.makedirs(FEATURE_CACHE_DIR, exist_ok=True)

    train_glob = glob.glob(os.path.join(FEATURE_CACHE_DIR, "train_*.pt"))
    test_glob = glob.glob(os.path.join(FEATURE_CACHE_DIR, "test_*.pt"))
    if (not FORCE_REEXTRACT) and len(train_glob) > 0 and len(test_glob) > 0:
        print("[EXTRACT] cache exists, skip extraction")
        return

    # 1) load 80B
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

    # 2) source data
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

    # 3) extract
    extract_split(
        moe_model,
        train_loader,
        split="train",
        cache_dir=FEATURE_CACHE_DIR,
        max_batches=MAX_EXTRACT_BATCHES_TRAIN,
    )
    extract_split(
        moe_model,
        test_loader,
        split="test",
        cache_dir=FEATURE_CACHE_DIR,
        max_batches=MAX_EXTRACT_BATCHES_TEST,
    )

    # 4) free 80B
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


# ================= Predictor train/test from offline features =================
def train_one_epoch_offline(epoch, train_loader, test_loader, pred_model, loss_func, optimizer, ckpt_dir,
                            num_experts, activated_experts, layer_num):
    pred_model.train()

    for batch, (features, labels) in enumerate(train_loader, 1):
        optimizer.zero_grad(set_to_none=True)

        # features: [B,L,S,D], labels: [B,L,S,K]
        # 当前建议 batch_size=1
        # assert features.size(0) == 1, f"Use BATCH_SIZE_TRAIN=1 first, got {features.size(0)}"
        # features = features[0].to(PREDICTOR_DEVICE, non_blocking=True).float()   # [L,S,D]
        # labels = labels[0].to(PREDICTOR_DEVICE, non_blocking=True).long()         # [L,S,K]

        # L, S_valid, D = features.shape
        # N_samples = S_valid

        # pred_inputs = features.contiguous().view(L, N_samples, D)
        # router_index = labels.contiguous().view(L, N_samples, -1)

        features = features.to(PREDICTOR_DEVICE, non_blocking=True).float()   # [B,L,S,D]
        labels = labels.to(PREDICTOR_DEVICE, non_blocking=True).long()        # [B,L,S,K]

        B, L, S_valid, D = features.shape
        _, _, _, K = labels.shape

        # [B,L,S,D] -> [L,B,S,D] -> [L,B*S,D]
        pred_inputs = features.permute(1, 0, 2, 3).contiguous().view(L, B * S_valid, D)

        # [B,L,S,K] -> [L,B,S,K] -> [L,B*S,K]
        router_index = labels.permute(1, 0, 2, 3).contiguous().view(L, B * S_valid, K)

        mem(f"batch {batch} before pred forward")
        pred_logits = pred_model(pred_inputs)
        mem(f"batch {batch} after pred forward")

        loss = pred_model.__class__.cal_loss(
            loss_func,
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

        if batch % 50 == 0:
            ckpt_path = os.path.join(ckpt_dir, f"{epoch}.{batch}.pth")
            torch.save(pred_model.state_dict(), ckpt_path)
            print("[SAVE]", ckpt_path)

        #     test_full_matrix_offline(
        #         test_loader,
        #         pred_model,
        #         epoch=epoch,
        #         batch=batch,
        #         ckpt_dir=ckpt_dir,
        #         num_experts=num_experts,
        #         activated_experts=activated_experts,
        #         layer_num=layer_num,
        #         topk=(activated_experts if TOPK_EVAL is None else TOPK_EVAL),
        #     )

    ckpt_path = os.path.join(ckpt_dir, f"{epoch}.last.pth")
    torch.save(pred_model.state_dict(), ckpt_path)
    print("[SAVE]", ckpt_path)


@torch.no_grad()
def test_full_matrix_offline(test_loader, pred_model, epoch, batch, ckpt_dir,
                             num_experts, activated_experts, layer_num, topk=10):
    pred_model.eval()
    layer_wise = [[[] for _ in range(layer_num)] for _ in range(layer_num)]

    for test_batch, (features, labels) in enumerate(test_loader, 1):
        # assert features.size(0) == 1, f"Use BATCH_SIZE_TRAIN=1 first, got {features.size(0)}"
        # features = features[0].to(PREDICTOR_DEVICE, non_blocking=True).float()  # [L,S,D]
        # labels = labels[0].to(PREDICTOR_DEVICE, non_blocking=True).long()        # [L,S,K]

        # L, S_valid, D = features.shape
        # pred_inputs = features.contiguous().view(L, S_valid, D)
        # router_index = labels.contiguous().view(L, S_valid, -1)
        features = features.to(PREDICTOR_DEVICE, non_blocking=True).float()   # [B,L,S,D]
        labels = labels.to(PREDICTOR_DEVICE, non_blocking=True).long()        # [B,L,S,K]

        B, L, S_valid, D = features.shape
        _, _, _, K = labels.shape

        pred_inputs = features.permute(1, 0, 2, 3).contiguous().view(L, B * S_valid, D)
        router_index = labels.permute(1, 0, 2, 3).contiguous().view(L, B * S_valid, K)

        pred_logits = pred_model(pred_inputs)

        for i in range(layer_num):
            for j in range(layer_num):
                logits_ij = pred_logits[i, j]
                pred_idx = pred_model.__class__.cal_pred_index(logits_ij, topk, expert_num=num_experts)
                gt_idx = router_index[j]
                acc = cal_topk_acc(pred_idx, gt_idx, activated_experts)
                layer_wise[i][j].append(acc)

        if test_batch >= 10:
            break

    acc_matrix = np.full((layer_num, layer_num), np.nan, dtype=np.float32)
    for i in range(layer_num):
        for j in range(layer_num):
            vals = layer_wise[i][j]
            if vals:
                acc_matrix[i, j] = float(sum(vals) / len(vals))

    eval_out_dir = resolve_eval_out_dir(ckpt_dir)
    out_path = os.path.join(eval_out_dir, build_eval_npy_name(topk))
    np.save(out_path, acc_matrix)
    print(f"[TEST] saved => {out_path}")


def run_predictor_training():
    # infer shapes from one offline sample
    sample_path = sorted(glob.glob(os.path.join(FEATURE_CACHE_DIR, "train_*.pt")))[0]
    sample = torch.load(sample_path, map_location="cpu")
    features = sample["features"]  # [L,S,D]
    labels = sample["labels"]      # [L,S,K]

    layer_num = int(features.shape[0])
    input_dim = int(features.shape[-1])
    activated_experts = int(labels.shape[-1])

    # 80B config prior knowledge / or hard infer from labels/ckpt config not available here
    # 这里我们仍然需要知道 num_experts。直接按 80B 模型设定。
    num_experts = 512

    print("[OFFLINE DATA] layer_num =", layer_num, "| input_dim =", input_dim, "| activated_experts =", activated_experts)
    print("[TRAIN] full-matrix predictor training from offline features")
    print("[OPT] Adam8bit | LR =", LR)

    train_ds = OfflineFeatureDataset(FEATURE_CACHE_DIR, split="train")
    test_ds = OfflineFeatureDataset(FEATURE_CACHE_DIR, split="test")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=True,
        num_workers=0,   # 先保守，避免 CPU 读盘问题
        pin_memory=PIN_MEMORY,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=PIN_MEMORY,
    )

    pred_model = CEPredictor(
        layers=layer_num,
        input_dim=input_dim,
        num_experts=num_experts,
    ).to(PREDICTOR_DEVICE)

    loss_func = CEPredictor.get_loss_func()
    optimizer = bnb.optim.Adam8bit(
        pred_model.parameters(),
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
            loss_func=loss_func,
            optimizer=optimizer,
            ckpt_dir=CKPT_DIR,
            num_experts=num_experts,
            activated_experts=activated_experts,
            layer_num=layer_num,
        )

    # test_full_matrix_offline(
    #     test_loader=test_loader,
    #     pred_model=pred_model,
    #     epoch=EPOCHS - 1,
    #     batch=999999,
    #     ckpt_dir=CKPT_DIR,
    #     num_experts=num_experts,
    #     activated_experts=activated_experts,
    #     layer_num=layer_num,
    #     topk=(activated_experts if TOPK_EVAL is None else TOPK_EVAL),
    # )


# ================= Main =================
if __name__ == "__main__":
    # Phase 1: offline extraction
    run_offline_extraction(WINDOW_SIZE)

    # Phase 2: predictor-only full-matrix training
    run_predictor_training()
