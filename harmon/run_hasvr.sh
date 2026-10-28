#!/bin/bash
# Hierarchical ASVR training on Harmon — two parallel experiments
#
#   Exp A (GPU 0-3): MMU only  — configs/examples/asvr_hierarchical_next.py
#   Exp B (GPU 4-7): MMU + T2I — configs/examples/hasvr_mmu_t2i.py
#
# Usage:
#   bash run_hasvr.sh          # start both in separate tmux windows
#   bash run_hasvr.sh mmu      # start Exp A only
#   bash run_hasvr.sh t2i      # start Exp B only

set -e
cd /home/szgx/Desktop/ctl/Harmon

CONDA_ENV="harmon"
NPROC=4

run_exp() {
    local session="$1"
    local gpus="$2"
    local config="$3"
    local work_dir="$4"

    mkdir -p "$work_dir"

    tmux new-window -t "${session}:" -n "$(basename $config .py)" 2>/dev/null || true
    tmux send-keys -t "${session}:" "
conda activate ${CONDA_ENV}
cd /home/szgx/Desktop/ctl/Harmon
export CUDA_VISIBLE_DEVICES=${gpus}
echo '[$(date)] Starting: ${config}'
torchrun --nproc-per-node=${NPROC} --master-port=\$(( 29500 + RANDOM % 100 )) \\
    scripts/train.py ${config} \\
    --work-dir ${work_dir} \\
    --launcher pytorch
echo '[$(date)] Done: ${config}'
" C-m
}

MODE="${1:-both}"

# Make sure tmux sessions exist
for sess in hasvr_mmu hasvr_t2i; do
    tmux has-session -t "$sess" 2>/dev/null || tmux new-session -d -s "$sess"
done

if [[ "$MODE" == "mmu" || "$MODE" == "both" ]]; then
    echo "Starting Exp A (MMU only) on GPU 0-3 in tmux: hasvr_mmu"
    run_exp "hasvr_mmu" "0,1,2,3" \
        "configs/examples/asvr_hierarchical_next.py" \
        "/data1/harmon_work_dirs/asvr_hierarchical_next"
fi

if [[ "$MODE" == "t2i" || "$MODE" == "both" ]]; then
    echo "Starting Exp B (MMU+T2I) on GPU 4-7 in tmux: hasvr_t2i"
    run_exp "hasvr_t2i" "4,5,6,7" \
        "configs/examples/hasvr_mmu_t2i.py" \
        "/data1/harmon_work_dirs/hasvr_mmu_t2i"
fi

echo ""
echo "Monitor:"
echo "  tmux attach -t hasvr_mmu   # Exp A"
echo "  tmux attach -t hasvr_t2i   # Exp B"
