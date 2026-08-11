import glob
import os
import torch
from torch.utils.data import DataLoader

from core.ce_predictor import CEPredictor
import train_scripts.train_crosstoken_Qwen80B as train_mod


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
    sample_path = sorted(glob.glob(os.path.join(train_mod.RAW_CACHE_DIR, "test_*.pt")))[0]
    sample = load_torch_file(sample_path, map_location="cpu")
    layer_input = sample["layer_input"]
    labels = sample["labels"]
    layer_num = int(layer_input.shape[0])
    hidden_size = int(layer_input.shape[-1])
    activated_experts = int(labels.shape[-1])

    topk = activated_experts if train_mod.TOPK_EVAL is None else int(train_mod.TOPK_EVAL)
    env_topk = os.environ.get("EVAL_TOPK")
    if env_topk:
        topk = int(env_topk)

    ckpt_path = pick_ckpt(train_mod.CKPT_DIR)
    out_dir = os.environ.get("EVAL_OUT_DIR") or os.path.dirname(ckpt_path)
    os.makedirs(out_dir, exist_ok=True)

    test_ds = train_mod.OfflineRawLayerInputDataset(train_mod.RAW_CACHE_DIR, split="test")
    test_loader = DataLoader(
        test_ds,
        batch_size=train_mod.BATCH_SIZE_EVAL,
        shuffle=False,
        num_workers=train_mod.NUM_WORKERS,
        pin_memory=train_mod.PIN_MEMORY,
    )

    pred_model = CEPredictor(
        layers=layer_num, input_dim=hidden_size, num_experts=train_mod.NUM_EXPERTS
    ).to(train_mod.PREDICTOR_DEVICE)
    num_heads = train_mod._pick_heads(hidden_size, 16)
    next_attn = train_mod.DeepNextTokenPredictor(
        hidden_size, num_heads, num_layers=train_mod.NUM_LAYERS, dropout=train_mod.DROPOUT
    ).to(train_mod.PREDICTOR_DEVICE)

    state = load_torch_file(ckpt_path, map_location="cpu")
    pred_model.load_state_dict(state["pred_model"])
    next_attn.load_state_dict(state["next_attn"])

    print(f"[EVAL] ckpt={ckpt_path}")
    print(f"[EVAL] topk={topk}")
    print(f"[EVAL] out_dir={out_dir}")
    train_mod.test_offline(
        test_loader=test_loader,
        pred_model=pred_model,
        next_attn=next_attn,
        epoch=0,
        batch=999999,
        ckpt_dir=out_dir,
        num_experts=train_mod.NUM_EXPERTS,
        activated_experts=activated_experts,
        layer_num=layer_num,
        topk=topk,
    )


if __name__ == "__main__":
    main()
