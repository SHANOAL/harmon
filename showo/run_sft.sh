#!/bin/bash
# Show-o pure SFT training launcher (no HASVR)
#
# Usage:
#   bash run_sft.sh [GPU_IDS]
#   bash run_sft.sh 0,1,2,3

set -e

GPU_IDS=${1:-"0,1,2,3"}
NPROC=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)

echo "Using GPUs: $GPU_IDS  (${NPROC} processes)"

CUDA_VISIBLE_DEVICES=$GPU_IDS accelerate launch \
    --config_file accelerate_configs/${NPROC}_gpu$([ $NPROC -gt 1 ] && echo "s_deepspeed_zero2" || echo "").yaml \
    --main_process_port=8889 \
    training/train_sft.py \
    config=configs/showo_sft.yaml
