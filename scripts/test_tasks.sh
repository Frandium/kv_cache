# arc_c, arc_e, boolq, hellaswag, lambada_openai, piqa, race, siqa
task_name="mmlu_abstract_algebra,mmlu_anatomy,mmlu_astronomy,mmlu_college_biology,mmlu_college_chemistry,mmlu_high_school_world_history,mmlu_international_law,mmlu_philosophy,mmlu_world_religions,mmlu_econometrics,mmlu_high_school_geography,mmlu_sociology,mmlu_us_foreign_policy,mmlu_business_ethics,mmlu_clinical_knowledge"

# task_name="arc_challenge,arc_easy,boolq,hellaswag,lambada_openai,piqa,race,siqa,mmlu,winogrande,openbookqa"

lm_eval --model hf \
    --model_args pretrained=/mnt/workspace/fyx/fyx-markov/data_modeling/Qwen1.5-MoE-A2.7B,dtype=bfloat16 \
    --tasks $task_name \
    --device cuda:3 \
    --batch_size auto \
    # --output_path "${model_name}.json" \
    # --log_samples \

# checkpoint_path="/mnt/workspace/active_learning/checkpoints/baseline-512/${BATCH}.pth"
# /home/jxzhou/PLM_PER/qwen/Qwen3-0.6B \
# /home/fdong/qwen/Qwen3-MoE-2.8B-0.8B,dtype=bfloat16,checkpoint_path=/home/fdong/lowmem_qwen/checkpoints/pretrain-switch.0.13000.pth \

