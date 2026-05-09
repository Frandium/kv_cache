import os

import torch
import torch.distributed as dist

from train_qwen_common import parse_common_args, thread_main


def main():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = int(os.environ["WORLD_SIZE"])

    args = parse_common_args()

    if local_rank == 0:
        print("Training Configuration:")
        for arg, value in vars(args).items():
            print(f"  {arg}: {value}")

    print(f"local_rank: {local_rank}, world_size: {world_size}")
    thread_main(local_rank, world_size, device, args)


if __name__ == "__main__":
    main()
