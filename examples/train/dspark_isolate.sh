#!/bin/bash
# DSpark isolation training: reproduce the good run's config (z-lab SWA DFlash init,
# block7 SWA draft-config) with a chosen sample_from_anchor flag. 5 epochs ShareGPT.
# Usage: bash dspark_isolate.sh <label> <DFLASH_INIT> <DRAFT_CONFIG> <SFA_FLAG>
#   SFA_FLAG = "--no-sample-from-anchor" (sfa=false) or "" (sfa=true, default)
set -uo pipefail
LABEL="${1:?label required}"
DFLASH_INIT="${2:?dflash init path required}"
DRAFT_CONFIG="${3:?draft-config path required}"
SFA_FLAG="${4:-}"

PYTHON="${PYTHON:-/usr/local/python3.12.13/bin/python3}"
TORCHRUN="${TORCHRUN:-/usr/local/python3.12.13/bin/torchrun}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="/tmp/dspark_isolate_${LABEL}_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/latest.log"
: > "$LOG_FILE"
exec > "$LOG_FILE" 2>&1

MODEL="/home/model/Qwen/Qwen3-8B"
DATA_PATH="/tmp/dspark_from_dflash_output"   # reuse prepared ShareGPT
OUTPUT_DIR="/tmp/dspark_isolate_${LABEL}_output"
HIDDEN_STATES_PATH="/tmp/dspark_isolate_${LABEL}_hs"
VLLM_PORT=8000
EPOCHS=5
LR=3e-4
SEQ_LENGTH=4096

echo "=== DSpark isolate [${LABEL}] started at $(date) ==="
echo "DFLASH_INIT   : $DFLASH_INIT"
echo "DRAFT_CONFIG  : $DRAFT_CONFIG"
echo "SFA_FLAG      : '${SFA_FLAG}' (empty=sfa=true)"
echo "Data (reused) : $DATA_PATH | Output: $OUTPUT_DIR"

if [ -f "$DATA_PATH/data-00000-of-00001.arrow" ]; then
    echo "=== Step 1: prepared data found, skipping prepare_data ==="
else
    "$PYTHON" scripts/prepare_data.py --model "$MODEL" --data sharegpt --output "$DATA_PATH" --max-samples 5000 --seq-length "$SEQ_LENGTH"
fi

echo "=== Step 2: Launching vLLM server ==="
ASCEND_RT_VISIBLE_DEVICES="0" "$PYTHON" scripts/launch_vllm.py "$MODEL" \
    --hidden-states-path "$HIDDEN_STATES_PATH" \
    --target-layer-ids 2 10 18 26 34 \
    -- --port "$VLLM_PORT" --gpu-memory-utilization 0.85 --max-model-len 8192 &
VLLM_PID=$!
cleanup() { echo "Stopping vLLM ($VLLM_PID)"; kill "$VLLM_PID" 2>/dev/null || true; wait "$VLLM_PID" 2>/dev/null || true; }
trap cleanup EXIT

elapsed=0
while true; do
    if curl -sf "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then break; fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then echo "ERROR: vLLM exited early"; exit 1; fi
    elapsed=$((elapsed + 3)); [ "$elapsed" -ge 600 ] && { echo "ERROR: vLLM timeout"; exit 1; }; sleep 3
done
echo "vLLM ready (after ${elapsed}s)."

echo "=== Step 3: Training (DSpark isolate [${LABEL}]) ==="
ASCEND_RT_VISIBLE_DEVICES="1,2,3" "$TORCHRUN" --standalone --nproc_per_node 3 \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --data-path "$DATA_PATH" \
    --hidden-states-path "$HIDDEN_STATES_PATH" \
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --log-dir "$LOG_DIR/tensorboard" \
    --epochs "$EPOCHS" --lr "$LR" --total-seq-len "$SEQ_LENGTH" \
    $SFA_FLAG \
    --speculator-type dspark \
    --init-from-dflash "$DFLASH_INIT" \
    --draft-config "$DRAFT_CONFIG" \
    --block-size 7 --max-anchors 512 \
    --target-layer-ids 2 10 18 26 34 --mask-token-id 151669 \
    --markov-rank 256 --markov-head-type vanilla \
    --enable-confidence-head --confidence-head-with-markov \
    --loss-fn '{"ce": 0.1, "tv": 0.9}' --confidence-head-alpha 1.0 \
    --draft-attn-impl sdpa --on-missing generate --on-generate delete

echo "Done [${LABEL}]. checkpoint saved to $OUTPUT_DIR/checkpoints/"
echo "=== DSpark isolate [${LABEL}] finished at $(date) ==="
