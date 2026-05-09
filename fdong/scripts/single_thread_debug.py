import torch

from train_qwen_common import parse_common_args, thread_main


def main():
    args = parse_common_args()

    print("Training Configuration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    print("local_rank: 0, world_size: 1")
    thread_main(0, 1, torch.device("cuda:0"), args)


if __name__ == "__main__":
    main()
