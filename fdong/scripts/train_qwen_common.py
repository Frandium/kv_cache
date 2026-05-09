import argparse
import os
import time

import torch
import torch.nn as nn
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoConfig, AutoTokenizer, get_cosine_schedule_with_warmup

from models import MyQwen3ForCausalLM
from utils import TokenizedJSONLData


def parse_int_list(value):
    if value is None or value == "":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def add_common_training_args(parser: argparse.ArgumentParser):
    parser.add_argument("--local_batch_size", type=int, default=16)
    parser.add_argument("--global_batch_size", type=int, default=256)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--seq_len", type=int, default=1024)

    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--optimizer", type=str, choices=["AdamW", "sgd"], default="AdamW")
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--total_training_steps", type=int, default=1000000)

    parser.add_argument("--data_shuffle", action="store_true", default=True)
    parser.add_argument("--no_data_shuffle", action="store_false", dest="data_shuffle")

    parser.add_argument("--use_bf16", action="store_true", default=True)
    parser.add_argument("--no_use_bf16", action="store_false", dest="use_bf16")

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--config_dir", type=str, default="../../Qwen3-0.6B")
    parser.add_argument("--data_dir", type=str, default="../../dclm/global-shard_01_of_10")
    parser.add_argument("--ckpt_dir", type=str, default="")

    parser.add_argument("--dataset_type", type=str, choices=["jsonl", "pruned", "synthetic_indexed"], default="jsonl")
    parser.add_argument("--per", type=float, default=1.0)

    parser.add_argument("--attention_stride_pattern", type=parse_int_list, default=None)
    parser.add_argument("--residual_source_pattern", type=parse_int_list, default=None)


def parse_common_args():
    parser = argparse.ArgumentParser(description="Training configuration")
    add_common_training_args(parser)
    return parser.parse_args()


@torch.no_grad()
def prepare_model(local_rank, world_size, device, args):
    config = AutoConfig.from_pretrained(args.config_dir, trust_remote_code=True)
    config.attention_stride_pattern = [
        1,1,1,1,1,1,1,1,1,
        4,4,4,4,4,4,4,4,4,
        1,1,1,1,1,1,1,1,1,1
    ]
    config.residual_source_pattern = [
        -1 for _ in range(config.num_hidden_layers)    
    ]
    model = MyQwen3ForCausalLM(config).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[device])

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"rank {local_rank} model ok, params: {trainable_params / 1e9:.2f}B/{total_params / 1e9:.2f}B")
    return model


def prepare_data(local_rank, world_size, args):
    tokenizer = AutoTokenizer.from_pretrained(args.config_dir, trust_remote_code=True)
    dataset =  TokenizedJSONLData(args.data_dir, args.seq_len, tokenizer)
    print(f"Construct dataset, total {len(dataset)} samples.")
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=args.data_shuffle)
    dataloader = DataLoader(dataset, batch_size=args.local_batch_size, num_workers=args.num_workers, sampler=sampler)
    return dataloader


def prepare_loss_optimizer(model, args):
    token_loss_fn = nn.CrossEntropyLoss(ignore_index=151643)
    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.01)
    else:
        print(f"Unsupported optimizer: {args.optimizer}, using Adam by default.")
        optimizer = torch.optim.Adam(params, lr=args.lr)

    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.total_training_steps)
    scaler = torch.amp.GradScaler("cuda")
    return token_loss_fn, optimizer, lr_scheduler, scaler


def forward_step(local_rank, device, source, target, model, token_loss_fn, args):
    source, target = source.to(device), target.to(device)
    output = model(source, output_hidden_states=False)
    target = target.reshape(-1)
    loss = token_loss_fn(output.logits.view(-1, output.logits.size(-1)), target)
    return loss


def update_step(optimizer, scheduler):
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()


def thread_main(local_rank, world_size, device, args):
    print(f"running on device {local_rank}")
    if local_rank == 0 and args.ckpt_dir and not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    dataloader = prepare_data(local_rank, world_size, args)
    model = prepare_model(local_rank, world_size, device, args)
    token_loss_fn, optimizer, lr_scheduler, scaler = prepare_loss_optimizer(model, args)

    gradient_accumulation_steps = args.global_batch_size // args.local_batch_size // world_size
    if gradient_accumulation_steps < 1:
        raise ValueError("global_batch_size must be >= local_batch_size * world_size")

    real_model = model.module if world_size > 1 else model

    for local_batch_idx, (source, target, real_lens) in enumerate(dataloader, 1):
        global_batch_idx = local_batch_idx // gradient_accumulation_steps
        start_time = time.time()

        with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda", enabled=args.use_bf16):
            loss = forward_step(local_rank, device, source, target, model, token_loss_fn, args)

        if world_size == 1:
            (loss / gradient_accumulation_steps).backward()
        else:
            scaler.scale(loss / gradient_accumulation_steps).backward()

        if local_batch_idx % gradient_accumulation_steps == 0:
            if world_size == 1:
                update_step(optimizer, lr_scheduler)
            else:
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()

            if local_rank == 0 and args.ckpt_dir and global_batch_idx % args.save_interval == 0:
                torch.save(real_model.state_dict(), f"{args.ckpt_dir}/{global_batch_idx}.pth")

        batch_time = time.time() - start_time
        if local_rank == 0:
            print(f"batch: {global_batch_idx}-{local_batch_idx}, loss: {loss:.3f}, batch_time: {batch_time:.3f}", flush=True)
