export LD_LIBRARY_PATH=/data/miniconda3/envs/env-3.12.11/lib:$LD_LIBRARY_PATH

IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model /apdcephfs_qy3/share_301069248/huggingface/Qwen3.5-4B \
    --enable_thinking false \
    --stream true \
    --temperature 0 \
    --max_new_tokens 2048
