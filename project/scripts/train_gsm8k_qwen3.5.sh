export LD_LIBRARY_PATH=/data/miniconda3/envs/env-3.12.11/lib:$LD_LIBRARY_PATH

# Disable wandb to avoid import errors
export WANDB_DISABLED=true

# Suppress vLLM compatibility warnings
export PYTHONWARNINGS="ignore:TRL currently supports vLLM versions:ignore:You have version 0.17.1 installed"

SYSTEM_PROMPT="""You are a helpful math assistant. Solve the problem step by step and put your final answer within \\boxed{}."""

MODEL_PATH=/apdcephfs_qy3/share_301069248/huggingface/Qwen3.5-2B
DATASET_PATH=/apdcephfs_qy3/share_301069248/huggingface/gsm8k
OUTPUT_DIR=/apdcephfs_qy3/share_301069248/users/yougenyuan/software/github/ms-swift/output/Qwen3.5-2B/v14

CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
/data/miniconda3/envs/env-3.12.11/bin/swift rlhf \
    --rlhf_type grpo \
    --model "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --external_plugins examples/train/grpo/plugin/gsm8k/gsm8k_plugin.py \
    --reward_funcs gsm8k_accuracy gsm8k_format \
    --reward_weights 1.0 0.2 \
    --columns '{"answer": "solution"}' \
    --enable_thinking false \
    --use_vllm false \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --dataset "$DATASET_PATH" \
    --split_dataset_ratio 0.01 \
    --load_from_cache_file true \
    --max_length 8192 \
    --max_completion_length 4096 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-6 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --save_strategy steps \
    --save_steps 200 \
    --save_total_limit 100 \
    --save_only_model true \
    --eval_strategy steps \
    --eval_steps 200 \
    --per_device_eval_batch_size 4 \
    --logging_steps 5 \
    --dataloader_num_workers 4 \
    --num_generations 8 \
    --temperature 1.0 \
    --system "$SYSTEM_PROMPT" \
    --deepspeed zero2 \
    --log_completions true \
    --report_to tensorboard \
    --max_grad_norm 1.0 \
    --epsilon 0.2 \
    --epsilon_high 0.28 \
    --scale_rewards none \
    --seed 42
