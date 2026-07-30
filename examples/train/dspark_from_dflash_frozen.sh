#!/bin/bash
# DSpark training: init backbone from the DeepSeek DFlash checkpoint (block7),
# FREEZE the DFlash backbone, train ONLY the Markov + confidence heads.
# 10 epochs ShareGPT, sample_from_anchor=true. Ascend NPU (1 vLLM + 3 DDP).
#
# DFLASH_CKPT is the DeepSeek dflash (deepseek-ai/dflash_qwen3_8b_block7) converted
# to speculators DFlash format = /home/model/dflash_qwen3_8b_block7_speculators
# (bit-identical weights to the deepseek-ai checkpoint; --init-from-dflash requires
# speculators format). Reuses the ShareGPT data already prepared by the from-dflash run.
set -uo pipefail

PYTHON="${PYTHON:-/usr/local/python3.12.13/bin/python3}"
TORCHRUN="${TORCHRUN:-/usr/local/python3.12.13/bin/torchrun}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="/tmp/dspark_frozen_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/latest.log"
PID_FILE="$LOG_DIR/dspark_train.pid"
: > "$LOG_FILE"
exec > "$LOG_FILE" 2>&1
echo "$$" > "$PID_FILE"

MODEL="/home/model/Qwen/Qwen3-8B"
DFLASH_CKPT="/home/model/dflash_qwen3_8b_block7_speculators"   # = DeepSeek dflash, speculators format
DATA_PATH="/tmp/dspark_from_dflash_output"                    # reuse prepared ShareGPT data
OUTPUT_DIR="/tmp/dspark_frozen_output"
HIDDEN_STATES_PATH="/tmp/dspark_frozen_hidden_states"
VLLM_PORT=8000
EPOCHS=10
LR=3e-4
SEQ_LENGTH=4096

SPECULATOR_TYPE="dspark"
BLOCK_SIZE=7
MAX_ANCHORS=512
TARGET_LAYER_IDS="2 10 18 26 34"
MASK_TOKEN_ID=151669
MARKOV_RANK=256
MARKOV_HEAD_TYPE="vanilla"
LOSS_FN='{"ce": 0.1, "tv": 0.9}'
CONFIDENCE_HEAD_ALPHA=1.0

VLLM_NPUS="0"
TRAIN_NPUS="1,2,3"
NUM_TRAIN_NPUS=3

echo "=== DSpark (from DFlash, FROZEN backbone) NPU training started at $(date) ==="
echo "Verifier        : $MODEL"
echo "DFlash src      : $DFLASH_CKPT  (DeepSeek dflash, speculators format)"
echo "Data (reused)   : $DATA_PATH"
echo "Output          : $OUTPUT_DIR"
echo "Hidden states   : $HIDDEN_STATES_PATH"
echo "Backbone        : FROZEN (--freeze-backbone); only Markov+confidence heads train"

# Step 1: reuse prepared data
if [ -f "$DATA_PATH/data-00000-of-00001.arrow" ]; then
    echo "=== Step 1: prepared data found at $DATA_PATH, skipping prepare_data ==="
else
    echo "=== Step 1: Preparing data ==="
    "$PYTHON" scripts/prepare_data.py --model "$MODEL" --data sharegpt \
        --output "$DATA_PATH" --max-samples 5000 --seq-length "$SEQ_LENGTH"
fi

# Step 2: Launch vLLM server
echo "=== Step 2: Launching vLLM server ==="
ASCEND_RT_VISIBLE_DEVICES="$VLLM_NPUS" "$PYTHON" scripts/launch_vllm.py "$MODEL" \
    --hidden-states-path "$HIDDEN_STATES_PATH" \
    --target-layer-ids $TARGET_LAYER_IDS \
    -- --port "$VLLM_PORT" --gpu-memory-utilization 0.85 --max-model-len 8192 &
VLLM_PID=$!
echo "vLLM server PID: $VLLM_PID"
cleanup() { echo "Stopping vLLM server (PID $VLLM_PID)..."; kill "$VLLM_PID" 2>/dev/null || true; wait "$VLLM_PID" 2>/dev/null || true; }
trap cleanup EXIT

echo "Waiting for vLLM server to be ready..."
WAIT_TIMEOUT=600; ELAPSED=0
while true; do
    if curl -sf "http://localhost:${VLLM_PORT}/v1/models" > /dev/null 2>&1; then break; fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then echo "ERROR: vLLM exited before ready. Aborting."; exit 1; fi
    ELAPSED=$((ELAPSED + 2)); [ "$ELAPSED" -ge "$WAIT_TIMEOUT" ] && { echo "ERROR: vLLM not ready in ${WAIT_TIMEOUT}s. Aborting."; exit 1; }; sleep 2
done
echo "vLLM server ready (after ${ELAPSED}s)."

# Step 3: Train DSpark, init backbone from DFlash, FREEZE backbone
echo "=== Step 3: Training (DSpark, init from DFlash, FROZEN backbone) ==="
ASCEND_RT_VISIBLE_DEVICES="$TRAIN_NPUS" "$TORCHRUN" \
    --standalone --nproc_per_node "$NUM_TRAIN_NPUS" \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --data-path "$DATA_PATH" \
    --hidden-states-path "$HIDDEN_STATES_PATH" \
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --log-dir "$LOG_DIR/tensorboard" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --total-seq-len "$SEQ_LENGTH" \
    --speculator-type "$SPECULATOR_TYPE" \
    --init-from-dflash "$DFLASH_CKPT" \
    --draft-config "$DFLASH_CKPT" \
    --freeze-backbone \
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

echo "Done. DSpark (frozen backbone) checkpoint saved to $OUTPUT_DIR/checkpoints/"
echo "=== DSpark (from DFlash, FROZEN backbone) training finished at $(date) ==="
