# 使用交互式命令行进行推理（单卡，full 微调输出用 --model）
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model /apdcephfs_qy3/share_301069248/users/yougenyuan/software/github/ms-swift/output/v7-20260311-101542/checkpoint-100 \
    --stream true \
    --temperature 0 \
    --max_new_tokens 2048

# 使用 vLLM 进行推理加速（full 微调无需 --merge_lora）
# CUDA_VISIBLE_DEVICES=0 \
# swift infer \
#     --model /apdcephfs_qy3/share_301069248/users/yougenyuan/software/github/ms-swift/output/v7-20260311-101542/checkpoint-100 \
#     --stream true \
#     --infer_backend vllm \
#     --vllm_max_model_len 8192 \
#     --temperature 0 \
#     --max_new_tokens 2048
