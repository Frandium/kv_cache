task_name="arc_easy,hellaswag,lambada_openai" # piqa, siqa

RUN_NAME="baseline"
CKPT_STEP=35000
CKPT_PATH="/mnt/workspace/df-unet-transformer/fdong/checkpoints/${RUN_NAME}/${CKPT_STEP}.pth"
# ATTENTION_STRIDE_PATTERN="1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1"
# RESIDUAL_SOURCE_PATTERN="-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1"
OUTPUT_PATH="/mnt/workspace/df-unet-transformer/fdong/experiments/${RUN_NAME}/task_results_${CKPT_STEP}.json"

ARGS="pretrained=/mnt/workspace/Qwen3-0.6B,"
ARGS+="checkpoint_path=${CKPT_PATH},"
# ARGS+="attention_stride_pattern='${ATTENTION_STRIDE_PATTERN}',"
# ARGS+="residual_source_pattern='${RESIDUAL_SOURCE_PATTERN}',"
ARGS+="dtype=bfloat16"

DEVICE="cuda:6"

lm_eval --model hf \
    --model_args $ARGS \
    --tasks $task_name \
    --device $DEVICE \
    --batch_size 32 \
    --output_path $OUTPUT_PATH
    # --log_samples \
