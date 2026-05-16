import torch

from train_qwen_common import parse_common_args, thread_main


def resolve_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    args = parse_common_args()

    print("Training Configuration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    device = resolve_device()
    print(f"local_rank: 0, world_size: 1, device: {device}")
    thread_main(0, 1, device, args)


if __name__ == "__main__":
    main()
