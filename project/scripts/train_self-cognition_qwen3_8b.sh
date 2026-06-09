# 2 * 60GB (NPROC_PER_NODE=2, CUDA_VISIBLE_DEVICES=0,1)
# 你可以通过设置`--dataset AI-ModelScope/alpaca-gpt4-data-en`跑通实验
# 注意：如果你指定了`--packing true`, 你必须额外设置`--attn_impl flash_attn`
    # --load_from_cache_file true \

NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
swift sft \
    --model /apdcephfs_qy3/share_301069248/huggingface/Qwen3-4B \
    --tuner_type full \
    --dataset '/apdcephfs_qy3/share_301069248/huggingface/self-cognition' \
              '/apdcephfs_qy3/share_301069248/huggingface/self-cognition' \
              '/apdcephfs_qy3/share_301069248/huggingface/self-cognition' \
              '/apdcephfs_qy3/share_301069248/huggingface/self-cognition' \
              '/apdcephfs_qy3/share_301069248/huggingface/self-cognition' \
              '/apdcephfs_qy3/share_301069248/huggingface/self-cognition' \
              '/apdcephfs_qy3/share_301069248/huggingface/self-cognition' \
              '/apdcephfs_qy3/share_301069248/huggingface/self-cognition' \
              '/apdcephfs_qy3/share_301069248/huggingface/self-cognition' \
    --torch_dtype bfloat16 \
    --num_train_epochs 100 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 4 \
    --packing true \
    --eval_steps 10 \
    --save_steps 10 \
    --save_total_limit 4 \
    --logging_steps 2 \
    --max_length 8192 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 2 \
    --dataset_num_proc 2 \
    --save_only_model true \
    --output_dir output \
    --deepspeed zero3 \
    --use_liger_kernel true \
    --model_author swift \
    --model_name swift-robot \
    --attn_impl flash_attn
