import glob
import os
import torch
from torch.utils.data import DataLoader

from core.ce_predictor import CEPredictor
from core.qwen3_moe_wrapper import Qwen3MoEForPredictor
from utils import StreamingTokenizedData
import train_scripts.train_sametoken_Qwen30B as train_mod


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
    moe_model = Qwen3MoEForPredictor(
        "/mnt/workspace/Qwen3-30B-A3B",
        device=train_mod.PREDICTOR_DEVICE,
        device_map="auto",
    )
    moe_model.base.eval()
    cfg = moe_model.base.config

    train_mod.HIDDEN_SIZE = cfg.hidden_size
    train_mod.LAYER_NUM = cfg.num_hidden_layers
    train_mod.NUM_EXPERTS = cfg.num_experts
    train_mod.ACTIVATED_EXPERTS = cfg.num_experts_per_tok
    input_dim = train_mod.get_input_dim()

    if train_mod.DATA_DOMAIN_DIR is None:
        test_ds = StreamingTokenizedData(train_mod.WINDOW_SIZE, is_test=True, repeat=False)
    else:
        test_ds = StreamingTokenizedData(
            train_mod.WINDOW_SIZE, is_test=True, folder_prefix=train_mod.DATA_DOMAIN_DIR, repeat=False
        )
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)

    pred_model = CEPredictor(
        layers=train_mod.LAYER_NUM, input_dim=input_dim, num_experts=train_mod.NUM_EXPERTS
    ).to(train_mod.PREDICTOR_DEVICE)
    pred_model.eval()

    ckpt_dir = f"../checkpoints/RichFeat_Qwen3_lr0.001_Win{train_mod.FeatureConfig.HISTORY_N}_Baseline"
    ckpt_path = pick_ckpt(ckpt_dir)
    pred_model.load_state_dict(load_torch_file(ckpt_path, map_location="cpu"))

    topk = int(os.environ.get("EVAL_TOPK", "8"))
    out_dir = os.environ.get("EVAL_OUT_DIR") or os.path.dirname(ckpt_path)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[EVAL] ckpt={ckpt_path}")
    print(f"[EVAL] topk={topk}")
    print(f"[EVAL] out_dir={out_dir}")
    train_mod.test(
        test_loader,
        moe_model,
        pred_model,
        epoch=0,
        batch=0,
        ckpt_dir=out_dir,
        topk=topk,
    )


if __name__ == "__main__":
    main()
