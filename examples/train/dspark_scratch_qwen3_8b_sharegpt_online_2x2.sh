#!/bin/bash
# DSpark online training from scratch (no DFlash backbone init), 4-device
# 2+2 layout (2 vLLM servers DP=2 + 2 training DDP).
#
# Same data and hyperparameters as dspark_from_dflash_qwen3_8b_sharegpt_online_2x2.sh;
# the only difference is random init (no --init-from-dflash).
#
# Usage:  setsid nohup bash examples/train/dspark_scratch_qwen3_8b_sharegpt_online_2x2.sh >/dev/null 2>&1 &
# Monitor: tail -f /tmp/dspark_scratch_2x2_logs/latest.log
# Stop:    kill -TERM -$(cat /tmp/dspark_scratch_2x2_logs/dspark_train.pid)

set -euo pipefail

# ============ Environment / Interpreter ============
PYTHON="${PYTHON:-/usr/local/python3.12.13/bin/python3}"
TORCHRUN="${TORCHRUN:-/usr/local/python3.12.13/bin/torchrun}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# ============ Logging (background-friendly, no symlinks/proc-sub) ============
LOG_DIR="/tmp/dspark_scratch_2x2_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/latest.log"
PID_FILE="$LOG_DIR/dspark_train.pid"
: > "$LOG_FILE"
exec > "$LOG_FILE" 2>&1
echo "$$" > "$PID_FILE"

# ============ Configuration ============
MODEL="/home/model/Qwen/Qwen3-8B"

# Reuse the ShareGPT data already prepared by the from-dflash run.
DATA_PATH="/tmp/dspark_from_dflash_output"
SAVE_PATH="/tmp/dspark_scratch_2x2_output/checkpoints"
HIDDEN_STATES_PATH="/tmp/dspark_hidden_states_2x2_scratch"
VLLM_PORT_0=8000
VLLM_PORT_1=8001
VLLM_ENDPOINTS="http://localhost:${VLLM_PORT_0}/v1,http://localhost:${VLLM_PORT_1}/v1"
EPOCHS=5
LR=3e-4
SEQ_LENGTH=4096

# --- Identical DSpark hyperparameters to the 2+2 from-dflash run ---
SPECULATOR_TYPE="dspark"
BLOCK_SIZE=7
MAX_ANCHORS=512
TARGET_LAYER_IDS="2 10 18 26 34"
MASK_TOKEN_ID=151669
NUM_LAYERS=5
DRAFT_ARCH="qwen3"
MARKOV_RANK=256
MARKOV_HEAD_TYPE="vanilla"
LOSS_FN='{"ce": 0.1, "tv": 0.9}'
CONFIDENCE_HEAD_ALPHA=1.0

# 2 devices for vLLM (DP=2), 2 for training (DDP).
VLLM_NPUS_0="0"
VLLM_NPUS_1="1"
TRAIN_NPUS="2,3"
NUM_TRAIN_NPUS=2

echo "=== DSpark (FROM SCRATCH, 2+2) NPU training started at $(date) ==="
echo "Verifier        : $MODEL"
echo "Data (reused)   : $DATA_PATH"
echo "Save            : $SAVE_PATH"
echo "Hidden states   : $HIDDEN_STATES_PATH (shared by both vLLM servers)"
echo "vLLM server #0  : NPU $VLLM_NPUS_0 port $VLLM_PORT_0"
echo "vLLM server #1  : NPU $VLLM_NPUS_1 port $VLLM_PORT_1"
echo "Training NPUs   : $TRAIN_NPUS ($NUM_TRAIN_NPUS-way DDP)"
echo "Endpoints (CSV): $VLLM_ENDPOINTS"
echo "NOTE: NO --init-from-dflash (random init); all other params == 2+2 from-dflash run."

# Step 1: reuse prepared data (skip if arrow file already exists)
if [ -f "$DATA_PATH/data-00000-of-00001.arrow" ]; then
    echo "=== Step 1: prepared data found at $DATA_PATH, skipping prepare_data ==="
else
    echo "=== Step 1: Preparing data ==="
    "$PYTHON" scripts/prepare_data.py \
        --model "$MODEL" --data sharegpt \
        --output "$DATA_PATH" --max-samples 5000 --seq-length "$SEQ_LENGTH"
fi

# Step 2a: Launch vLLM server #0 first so #1 can reuse its torch.compile cache.
echo "=== Step 2a: Launching vLLM server #0 (NPU $VLLM_NPUS_0, port $VLLM_PORT_0) ==="
ASCEND_RT_VISIBLE_DEVICES="$VLLM_NPUS_0" "$PYTHON" scripts/launch_vllm.py "$MODEL" \
    --hidden-states-path "$HIDDEN_STATES_PATH" \
    --target-layer-ids $TARGET_LAYER_IDS \
    -- --port "$VLLM_PORT_0" \
       --gpu-memory-utilization 0.85 \
       --max-model-len 8192 &
VLLM_PID_0=$!
echo "vLLM server #0 PID: $VLLM_PID_0"

cleanup() {
    echo "Stopping vLLM servers (PIDs $VLLM_PID_0, $VLLM_PID_1)..."
    kill "$VLLM_PID_0" "$VLLM_PID_1" 2>/dev/null || true
    wait "$VLLM_PID_0" "$VLLM_PID_1" 2>/dev/null || true
}
trap cleanup EXIT

wait_for_vllm() {
    local port="$1"
    local pid="$2"
    local label="$3"
    local elapsed=0
    while true; do
        if curl -sf "http://localhost:${port}/v1/models" > /dev/null 2>&1; then
            echo "vLLM server ${label} ready (port ${port}, after ${elapsed}s)"
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "ERROR: vLLM server ${label} (PID ${pid}) exited before ready. Aborting."
            exit 1
        fi
        elapsed=$((elapsed + 2))
        if [ "$elapsed" -ge 600 ]; then
            echo "ERROR: vLLM server ${label} not ready within 600s. Aborting."
            exit 1
        fi
        sleep 2
    done
}
wait_for_vllm "$VLLM_PORT_0" "$VLLM_PID_0" "#0"

# Step 2b: Launch vLLM server #1 (after #0 is ready, reuses its compile cache).
echo "=== Step 2b: Launching vLLM server #1 (NPU $VLLM_NPUS_1, port $VLLM_PORT_1) ==="
ASCEND_RT_VISIBLE_DEVICES="$VLLM_NPUS_1" "$PYTHON" scripts/launch_vllm.py "$MODEL" \
    --hidden-states-path "$HIDDEN_STATES_PATH" \
    --target-layer-ids $TARGET_LAYER_IDS \
    -- --port "$VLLM_PORT_1" \
       --gpu-memory-utilization 0.85 \
       --max-model-len 8192 &
VLLM_PID_1=$!
echo "vLLM server #1 PID: $VLLM_PID_1"
wait_for_vllm "$VLLM_PORT_1" "$VLLM_PID_1" "#1"

# Step 3: Train DSpark from scratch (no --init-from-dflash).
echo "=== Step 3: Training (DSpark, FROM SCRATCH, 2+2) ==="
ASCEND_RT_VISIBLE_DEVICES="$TRAIN_NPUS" "$TORCHRUN" \
    --standalone --nproc_per_node "$NUM_TRAIN_NPUS" \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --data-path "$DATA_PATH" \
    --hidden-states-path "$HIDDEN_STATES_PATH" \
    --vllm-endpoint "$VLLM_ENDPOINTS" \
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
    --on-generate delete \
    --save-best

echo "Done. Standard DSpark (from-scratch) checkpoint saved to $SAVE_PATH"
echo "=== DSpark (FROM SCRATCH, 2+2) NPU training finished at $(date) ==="
