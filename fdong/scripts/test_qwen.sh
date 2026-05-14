export CUDA_VISIBLE_DEVICES=2

LOCAL_BATCH_SIZE=4
TEST_BATCH_SIZE=32
SEQ_LEN=1024
PREFILL_LEN=100
DECODE_STEPS=128
DATA_SHUFFLE=false
USE_BF16=true

NUM_WORKERS=4
CONFIG_DIR="../../../Qwen3-0.6B"
DATA_DIR="../../../dclm/global-shard_01_of_10"

RUN_NAME="unet-4-8-16-8-4"
CKPT_DIR="../checkpoints/${RUN_NAME}"
CKPT_STEP=18000
CKPT_FILE=""
OUTPUT_JSON="../experiments/${RUN_NAME}/teacher_forced_kv_decode_${CKPT_STEP}.json"

# Leave these empty to load patterns from ${CKPT_DIR}/runtime_config.json.
# For old checkpoints without runtime_config.json, fill them manually here.
ATTENTION_STRIDE_PATTERN=""
RESIDUAL_SOURCE_PATTERN=""

ARGS=""
ARGS+=" --local_batch_size $LOCAL_BATCH_SIZE"
ARGS+=" --test_batch_size $TEST_BATCH_SIZE"
ARGS+=" --seq_len $SEQ_LEN"
ARGS+=" --prefill_len $PREFILL_LEN"
ARGS+=" --decode_steps $DECODE_STEPS"
ARGS+=" --num_workers $NUM_WORKERS"
ARGS+=" --config_dir $CONFIG_DIR"
ARGS+=" --data_dir $DATA_DIR"
ARGS+=" --ckpt_dir $CKPT_DIR"
ARGS+=" --ckpt_step $CKPT_STEP"
ARGS+=" --output_json $OUTPUT_JSON"
ARGS+=" --attn_implementation eager"

[ -n "$CKPT_FILE" ] && ARGS+=" --ckpt_file $CKPT_FILE"
[ -n "$ATTENTION_STRIDE_PATTERN" ] && ARGS+=" --attention_stride_pattern=${ATTENTION_STRIDE_PATTERN}"
[ -n "$RESIDUAL_SOURCE_PATTERN" ] && ARGS+=" --residual_source_pattern=${RESIDUAL_SOURCE_PATTERN}"
[ "$DATA_SHUFFLE" = "true" ] && ARGS+=" --data_shuffle"
[ "$USE_BF16" = "true" ] && ARGS+=" --use_bf16"

python test_qwen.py ${ARGS}
