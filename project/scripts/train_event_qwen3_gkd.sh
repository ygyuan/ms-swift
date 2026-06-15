export LD_LIBRARY_PATH=/data/miniconda3/envs/env-3.12.11/lib:$LD_LIBRARY_PATH

# Disable wandb to avoid import errors
export WANDB_DISABLED=true

# Suppress vLLM compatibility warnings
export PYTHONWARNINGS="ignore:TRL currently supports vLLM versions:ignore:You have version 0.17.1 installed"

# Help reduce GPU memory fragmentation (teacher 14B on 2 GPUs is tight)
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

# Continue training from previous 1-epoch checkpoint to reach total 10 epochs.
# We do NOT use --resume_from_checkpoint because the previous run already finished
# (should_training_stop=True, lr fully decayed to ~0). Instead we treat
# checkpoint-321 as a fresh starting model and run 9 more epochs with a new
# cosine schedule. Learning rate is reduced to avoid destroying learned weights.
MODEL_PATH=/apdcephfs_qy3/share_301069248/huggingface/OperableTextEventFt_v9
TEACHER_MODEL_PATH=/apdcephfs_qy3/share_301069248/huggingface/OperableTextEventFt_4b/checkpoint-4932
DATASET_PATH=/apdcephfs_qy3/share_301069248/data/video/event_rag/merge/train_ayden_v1.jsonl
OUTPUT_DIR=/apdcephfs_qy3/share_301069248/users/yougenyuan/software/github/ms-swift/output/Qwen3-1.7B/event/gkd_v3

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
/data/miniconda3/envs/env-3.12.11/bin/swift rlhf \
    --rlhf_type gkd \
    --model "$MODEL_PATH" \
    --model_type qwen3 \
    --template qwen3 \
    --teacher_model "$TEACHER_MODEL_PATH" \
    --teacher_model_type qwen3 \
    --output_dir "$OUTPUT_DIR" \
    --enable_thinking false \
    --response_prefix '' \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --dataset "$DATASET_PATH" \
    --split_dataset_ratio 0.01 \
    --load_from_cache_file true \
    --max_length 6144 \
    --max_completion_length 2048 \
    --truncation_strategy left \
    --lmbda 0.5 \
    --seq_kd false \
    --beta 0.5 \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.3 \
    --vllm_tensor_parallel_size 1 \
    --vllm_max_model_len 10240 \
    --sleep_level 1 \
    --teacher_deepspeed zero3_offload \
    --deepspeed zero3 \
    --num_train_epochs 6 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-6 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --max_grad_norm 1.0 \
    --save_strategy steps \
    --save_steps 200 \
    --save_total_limit 100 \
    --save_only_model true \
    --eval_strategy steps \
    --eval_steps 600 \
    --per_device_eval_batch_size 2 \
    --logging_steps 5 \
    --dataloader_num_workers 32 \
    --dataset_num_proc 32 \
    --temperature 1.0 \
    --report_to tensorboard \
    --seed 42
