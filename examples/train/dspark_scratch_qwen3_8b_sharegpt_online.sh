#!/bin/bash
# Online DSpark Training Script: FROM SCRATCH (no DFlash backbone init), Ascend NPU.
#
# Same data + identical hyperparameters as dspark_from_dflash_qwen3_8b_sharegpt_online.sh,
# the ONLY difference is the absence of --init-from-dflash / --draft-config: the DSpark
# decoder body + Markov/confidence heads all start from random init (no DFlash transfer).
# This gives an apples-to-apples comparison vs the DFlash-seeded run.
#
# Reuses the already-prepared ShareGPT data at $DATA_PATH (no re-tokenize). Writes new
# checkpoints/logs/hidden-states under /tmp so the DFlash-seeded run's outputs are kept.
#
# Usage:  setsid nohup bash examples/train/dspark_scratch_qwen3_8b_sharegpt_online.sh >/dev/null 2>&1 &
# Monitor: tail -f /tmp/dspark_scratch_logs/latest.log
# Stop:    kill -TERM -$(cat /tmp/dspark_scratch_logs/dspark_train.pid)

set -euo pipefail

# ============ Environment / Interpreter ============
PYTHON="${PYTHON:-/usr/local/python3.12.13/bin/python3}"
TORCHRUN="${TORCHRUN:-/usr/local/python3.12.13/bin/torchrun}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# ============ Logging (background-friendly, no symlinks/proc-sub) ============
LOG_DIR="/tmp/dspark_scratch_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/latest.log"
PID_FILE="$LOG_DIR/dspark_train.pid"
: > "$LOG_FILE"
exec > "$LOG_FILE" 2>&1
echo "$$" > "$PID_FILE"

# ============ Configuration ============
MODEL="/home/model/Qwen/Qwen3-8B"

# Reuse the ShareGPT data already prepared by the DFlash-seeded run (arrow + token_freq).
DATA_PATH="/tmp/dspark_from_dflash_output"
# New (separate) output / hidden-states so the DFlash-seeded run's artifacts are kept.
SAVE_PATH="/tmp/dspark_scratch_output/checkpoints"
HIDDEN_STATES_PATH="/tmp/dspark_scratch_hidden_states"
VLLM_PORT=8000
EPOCHS=5
LR=3e-4
SEQ_LENGTH=4096

# --- Identical DSpark hyperparameters to the DFlash-seeded run ---
SPECULATOR_TYPE="dspark"
BLOCK_SIZE=7
MAX_ANCHORS=512                       # full-vocab -> 512 (same as DFlash-seeded run)
TARGET_LAYER_IDS="2 10 18 26 34"      # same aux layers
MASK_TOKEN_ID=151669
NUM_LAYERS=5                          # was via --draft-config before; now explicit
DRAFT_ARCH="qwen3"                    # synthesized decoder == DFlash source decoder
# (no --draft-vocab-size: full verifier vocab, matching the DFlash-seeded run)
MARKOV_RANK=256
MARKOV_HEAD_TYPE="vanilla"
LOSS_FN='{"ce": 0.1, "tv": 0.9}'
CONFIDENCE_HEAD_ALPHA=1.0

# NPU assignments: 1 for vLLM, 3 for training (DDP+FSDP).
VLLM_NPUS="0"
TRAIN_NPUS="1,2,3"
NUM_TRAIN_NPUS=3

echo "=== DSpark (FROM SCRATCH) NPU training started at $(date) ==="
echo "Verifier        : $MODEL"
echo "Data (reused)   : $DATA_PATH"
echo "Save            : $SAVE_PATH"
echo "Hidden states   : $HIDDEN_STATES_PATH"
echo "vLLM NPUs       : $VLLM_NPUS | Training NPUs: $TRAIN_NPUS ($NUM_TRAIN_NPUS-way DDP)"
echo "NOTE: NO --init-from-dflash (random init); all other params == DFlash-seeded run."

# Step 1: reuse prepared data (skip if the arrow file is already there)
if [ -f "$DATA_PATH/data-00000-of-00001.arrow" ]; then
    echo "=== Step 1: prepared data found at $DATA_PATH, skipping prepare_data ==="
else
    echo "=== Step 1: Preparing data ==="
    "$PYTHON" scripts/prepare_data.py --model "$MODEL" --data sharegpt \
        --output "$DATA_PATH" --max-samples 5000 --seq-length "$SEQ_LENGTH"
fi

# Step 2: Launch vLLM (same flags as the DFlash-seeded run)
echo "=== Step 2: Launching vLLM server ==="
ASCEND_RT_VISIBLE_DEVICES="$VLLM_NPUS" "$PYTHON" scripts/launch_vllm.py "$MODEL" \
    --hidden-states-path "$HIDDEN_STATES_PATH" \
    --target-layer-ids $TARGET_LAYER_IDS \
    -- --port "$VLLM_PORT" --gpu-memory-utilization 0.85 --max-model-len 8192 &
VLLM_PID=$!
echo "vLLM server PID: $VLLM_PID"

cleanup() {
    echo "Stopping vLLM server (PID $VLLM_PID)..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for vLLM server to be ready..."
WAIT_TIMEOUT=600
ELAPSED=0
while true; do
    if curl -sf "http://localhost:${VLLM_PORT}/v1/models" > /dev/null 2>&1; then
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vLLM server (PID $VLLM_PID) exited before ready. Aborting."; exit 1
    fi
    ELAPSED=$((ELAPSED + 2))
    if [ "$ELAPSED" -ge "$WAIT_TIMEOUT" ]; then
        echo "ERROR: vLLM not ready within ${WAIT_TIMEOUT}s. Aborting."; exit 1
    fi
    sleep 2
done
echo "vLLM server ready (after ${ELAPSED}s)."

# Step 3: Train DSpark FROM SCRATCH (no --init-from-dflash / --draft-config).
# --num-layers 5 --draft-arch qwen3 synthesizes a decoder identical to the DFlash
# source (5 layers / 4096 / 12288 / 32 heads / head_dim 128).
echo "=== Step 3: Training (DSpark, FROM SCRATCH) ==="
ASCEND_RT_VISIBLE_DEVICES="$TRAIN_NPUS" "$TORCHRUN" \
    --standalone --nproc_per_node "$NUM_TRAIN_NPUS" \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --data-path "$DATA_PATH" \
    --hidden-states-path "$HIDDEN_STATES_PATH" \
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
    --save-path "$SAVE_PATH" \
    --log-dir "$LOG_DIR/tensorboard" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --total-seq-len "$SEQ_LENGTH" \
    --speculator-type "$SPECULATOR_TYPE" \
    --num-layers "$NUM_LAYERS" \
    --draft-arch "$DRAFT_ARCH" \
    --block-size "$BLOCK_SIZE" \
    --max-anchors "$MAX_ANCHORS" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --mask-token-id "$MASK_TOKEN_ID" \
    --markov-rank "$MARKOV_RANK" \
    --markov-head-type "$MARKOV_HEAD_TYPE" \
    --enable-confidence-head \
    --confidence-head-with-markov \
    --loss-fn "$LOSS_FN" \
    --confidence-head-alpha "$CONFIDENCE_HEAD_ALPHA" \
    --draft-attn-impl sdpa \
    --on-missing generate \
    --on-generate delete

echo "Done. Standard DSpark (from-scratch) checkpoint saved to $SAVE_PATH"
echo "=== DSpark (FROM SCRATCH) NPU training finished at $(date) ==="
