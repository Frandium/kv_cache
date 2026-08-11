import glob
import os
import torch
from torch.utils.data import DataLoader

from core.ce_predictor import CEPredictor
import train_scripts.train_sametoken_Qwen80B as train_mod


def load_torch_file(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def pick_ckpt(default_dir):
    env_ckpt = os.environ.get("EVAL_CKPT")
    if env_ckpt:
        return env_ckpt
    candidates = sorted(glob.glob(os.path.join(default_dir, "*.last.pth")))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(f"No *.last.pth found in {default_dir}. Set EVAL_CKPT to override.")


def main():
    sample_path = sorted(glob.glob(os.path.join(train_mod.FEATURE_CACHE_DIR, "test_*.pt")))[0]
    sample = load_torch_file(sample_path, map_location="cpu")
    features = sample["features"]
    labels = sample["labels"]
    layer_num = int(features.shape[0])
    input_dim = int(features.shape[-1])
    activated_experts = int(labels.shape[-1])
    num_experts = 512

    topk = activated_experts if train_mod.TOPK_EVAL is None else int(train_mod.TOPK_EVAL)
    env_topk = os.environ.get("EVAL_TOPK")
    if env_topk:
        topk = int(env_topk)

    ckpt_path = pick_ckpt(train_mod.CKPT_DIR)
    out_dir = os.environ.get("EVAL_OUT_DIR") or os.path.dirname(ckpt_path)
    os.makedirs(out_dir, exist_ok=True)

    test_ds = train_mod.OfflineFeatureDataset(train_mod.FEATURE_CACHE_DIR, split="test")
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=train_mod.NUM_WORKERS,
        pin_memory=train_mod.PIN_MEMORY,
    )

    pred_model = CEPredictor(layers=layer_num, input_dim=input_dim, num_experts=num_experts).to(train_mod.PREDICTOR_DEVICE)
    state = load_torch_file(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "pred_model" in state:
        state = state["pred_model"]
    pred_model.load_state_dict(state)

    print(f"[EVAL] ckpt={ckpt_path}")
    print(f"[EVAL] topk={topk}")
    print(f"[EVAL] out_dir={out_dir}")
    train_mod.test_full_matrix_offline(
        test_loader=test_loader,
        pred_model=pred_model,
        epoch=0,
        batch=999999,
        ckpt_dir=out_dir,
        num_experts=num_experts,
        activated_experts=activated_experts,
        layer_num=layer_num,
        topk=topk,
    )


if __name__ == "__main__":
    main()
