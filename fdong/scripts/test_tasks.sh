task_name="arc_easy,hellaswag,lambada_openai" # piqa, siqa

RUN_NAME="unet-4-8-4"
CKPT_STEP=35000
CKPT_DIR="/mnt/workspace/df-unet-transformer/fdong/checkpoints/${RUN_NAME}"
OUTPUT_PATH="/mnt/workspace/df-unet-transformer/fdong/experiments/${RUN_NAME}/task_results_${CKPT_STEP}.json"

ARGS="pretrained=/mnt/workspace/Qwen3-0.6B,"
ARGS+="checkpoint_path=${CKPT_DIR},"
ARGS+="checkpoint_step=${CKPT_STEP},"
ARGS+="dtype=bfloat16"

DEVICE="cuda:6"

lm_eval --model hf \
    --model_args $ARGS \
    --tasks $task_name \
    --device $DEVICE \
    --batch_size 32 \
    --output_path $OUTPUT_PATH
    # --log_samples \
