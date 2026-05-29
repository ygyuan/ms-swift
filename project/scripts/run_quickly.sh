# 13GB
# Use a per-run output dir so multiple experiments don't overwrite each other.
RUN_TS="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="output/run_quickly_${RUN_TS}"

NPROC_PER_NODE=1 \
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model /apdcephfs_qy3/share_301069248/huggingface/Qwen3-4B/ \
    --tuner_type lora \
    --dataset '/apdcephfs_qy3/share_301069248/huggingface/alpaca-gpt4-data-zh#500' \
              '/apdcephfs_qy3/share_301069248/huggingface/alpaca-gpt4-data-en#500' \
              '/apdcephfs_qy3/share_301069248/huggingface/self-cognition#500' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir "${OUTPUT_DIR}" \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_author swift \
    --model_name swift-robot
