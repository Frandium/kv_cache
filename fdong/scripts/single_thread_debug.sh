export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

LOCAL_BATCH_SIZE=4
GLOBAL_BATCH_SIZE=16
SAVE_INTERVAL=1000
SEQ_LEN=1024
DATA_SHUFFLE=true  # 注意：bash 中 true/false 是字符串
USE_BF16=true

OPTIMIZER="AdamW"
LR=1e-4

NUM_WORKERS=4
CONFIG_DIR="../../../Qwen3-0.6B"
DATA_DIR="../../../dclm/global-shard_01_of_10"

ATTENTION_STRIDE_PATTERN="1,1,1,1,4,4,4,4,8,8,8,8,16,16,16,16,8,8,8,8,4,4,4,4,1,1,1,1"
RESIDUAL_SOURCE_PATTERN="-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1"


# ========== 动态构建 CKPT_DIR ==========
CKPT_DIR="../checkpoints/single-thread-debug"

# ========== 构建 Python 命令 ==========
ARGS=""
ARGS+=" --local_batch_size $LOCAL_BATCH_SIZE"
ARGS+=" --global_batch_size $GLOBAL_BATCH_SIZE"
ARGS+=" --save_interval $SAVE_INTERVAL"
ARGS+=" --seq_len $SEQ_LEN"
ARGS+=" --num_workers $NUM_WORKERS"
ARGS+=" --config_dir $CONFIG_DIR"
ARGS+=" --data_dir $DATA_DIR"
ARGS+=" --ckpt_dir $CKPT_DIR"  # ← 关键：传入构建好的路径
ARGS+=" --attention_stride_pattern=${ATTENTION_STRIDE_PATTERN}"
ARGS+=" --residual_source_pattern=${RESIDUAL_SOURCE_PATTERN}"


# 处理布尔参数
[ "$DATA_SHUFFLE" = "true" ] && ARGS+=" --data_shuffle"
[ "$USE_BF16" = "true" ] && ARGS+=" --use_bf16"

ARGS+=" --optimizer $OPTIMIZER"
ARGS+=" --lr $LR"

python single_thread_debug.py ${ARGS}
