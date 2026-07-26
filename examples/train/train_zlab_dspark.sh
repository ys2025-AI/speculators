#!/bin/bash
# Option A': maximize DFlash weight reuse. Uses the block7 native DFlash source
# (/tmp/dflash_qwen3_8b_swa, SWA config) — same as the working dflash_swa run.
# --freeze-backbone freezes decoder+fc+norms (lm_head frozen by default) so the
# DFlash weights are NOT modified. --draft-adapter-rank adds a tiny residual
# adapter on the lm_head path that learns to realign the frozen decoder's draft
# logits to sfa=true (gen-start friendly). Only adapter + markov/confidence
# heads train; the DFlash decoder is fully preserved.
# 5 epochs, lr=3e-4, default linear scheduler. Checkpoints under /home/model/dspark_ckpt.
# Usage: BLOCK_SIZE=7 bash train_zlab_dspark.sh

set -euo pipefail

PYTHON="${PYTHON:-/usr/local/python3.12.13/bin/python3}"
TORCHRUN="${TORCHRUN:-/usr/local/python3.12.13/bin/torchrun}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

BS="${BLOCK_SIZE:-7}"
# OPTION A': maximize dflash reuse. Same dflash source + SWA config as the
# working dflash_swa (sfa=true, gen-start 0.80) — block7 native weights via
# /tmp/dflash_qwen3_8b_swa. --freeze-backbone freezes decoder+fc+norms (lm_head
# frozen by default) so the DFlash weights are NOT modified. --draft-adapter-rank
# adds a tiny residual adapter on the lm_head path that learns to realign the
# frozen decoder's draft logits to sfa=true (gen-start friendly). Only the
# adapter + markov/confidence heads train.
DFLASH_WEIGHTS="/tmp/dflash_qwen3_8b_swa"
DFLASH_CONFIG="/tmp/dflash_qwen3_8b_swa"
LOG_DIR="/tmp/dspark_zlab_block${BS}_adapter_logs"
SAVE_PATH="/home/model/dspark_ckpt/zlab_block${BS}_adapter/checkpoints"
HIDDEN_STATES_PATH="/tmp/dspark_hidden_states_zlab_block${BS}_adapter"

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/latest.log"
: > "$LOG_FILE"
exec > "$LOG_FILE" 2>&1

MODEL="/home/model/Qwen/Qwen3-8B"
DATA_PATH="/tmp/dspark_from_dflash_output"
VLLM_PORT_0=8000
VLLM_PORT_1=8001
VLLM_ENDPOINTS="http://localhost:${VLLM_PORT_0}/v1,http://localhost:${VLLM_PORT_1}/v1"
EPOCHS=5
LR=3e-4
SEQ_LENGTH=4096

SPECULATOR_TYPE="dspark"
MAX_ANCHORS=${MAX_ANCHORS:-512}
TARGET_LAYER_IDS="2 10 18 26 34"
MASK_TOKEN_ID=151669
NUM_LAYERS=5
MARKOV_RANK=256
MARKOV_HEAD_TYPE="vanilla"
LOSS_FN='{"ce": 0.1, "tv": 0.9}'
CONFIDENCE_HEAD_ALPHA=1.0

VLLM_NPUS_0="0"
VLLM_NPUS_1="1"
TRAIN_NPUS="2,3"
NUM_TRAIN_NPUS=2

echo "=== DSpark (z-lab DFlash, block${BS}) started at $(date) ==="
echo "DFlash weights: $DFLASH_WEIGHTS"
echo "DFlash config:  $DFLASH_CONFIG (SWA, block${BS})"
echo "Save: $SAVE_PATH"

if [ -f "$DATA_PATH/data-00000-of-00001.arrow" ]; then
    echo "=== Step 1: data found ==="
else
    echo "=== Step 1: Preparing data ==="
    "$PYTHON" scripts/prepare_data.py --model "$MODEL" --data sharegpt \
        --output "$DATA_PATH" --max-samples 5000 --seq-length "$SEQ_LENGTH"
fi

echo "=== Step 2a: vLLM #0 ==="
ASCEND_RT_VISIBLE_DEVICES="$VLLM_NPUS_0" "$PYTHON" scripts/launch_vllm.py "$MODEL" \
    --hidden-states-path "$HIDDEN_STATES_PATH" \
    --target-layer-ids $TARGET_LAYER_IDS \
    -- --port "$VLLM_PORT_0" --gpu-memory-utilization 0.85 --max-model-len 8192 &
VLLM_PID_0=$!

cleanup() { kill "$VLLM_PID_0" "$VLLM_PID_1" 2>/dev/null || true; wait "$VLLM_PID_0" "$VLLM_PID_1" 2>/dev/null || true; }
trap cleanup EXIT

wait_for_vllm() {
    local port="$1" pid="$2" label="$3" elapsed=0
    while true; do
        if curl -sf "http://localhost:${port}/v1/models" > /dev/null 2>&1; then
            echo "vLLM ${label} ready (${port}, ${elapsed}s)"; return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then echo "ERROR: vLLM ${label} died"; exit 1; fi
        elapsed=$((elapsed + 2))
        if [ "$elapsed" -ge 600 ]; then echo "ERROR: timeout"; exit 1; fi
        sleep 2
    done
}
wait_for_vllm "$VLLM_PORT_0" "$VLLM_PID_0" "#0"

echo "=== Step 2b: vLLM #1 ==="
ASCEND_RT_VISIBLE_DEVICES="$VLLM_NPUS_1" "$PYTHON" scripts/launch_vllm.py "$MODEL" \
    --hidden-states-path "$HIDDEN_STATES_PATH" \
    --target-layer-ids $TARGET_LAYER_IDS \
    -- --port "$VLLM_PORT_1" --gpu-memory-utilization 0.85 --max-model-len 8192 &
VLLM_PID_1=$!
wait_for_vllm "$VLLM_PORT_1" "$VLLM_PID_1" "#1"

echo "=== Step 3: Training (block${BS}) ==="
ASCEND_RT_VISIBLE_DEVICES="$TRAIN_NPUS" "$TORCHRUN" \
    --standalone --nproc_per_node "$NUM_TRAIN_NPUS" \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --data-path "$DATA_PATH" \
    --hidden-states-path "$HIDDEN_STATES_PATH" \
    --vllm-endpoint "$VLLM_ENDPOINTS" \
    --save-path "$SAVE_PATH" \
    --log-dir "$LOG_DIR/tensorboard" \
    --epochs "$EPOCHS" --lr "$LR" --total-seq-len "$SEQ_LENGTH" \
    --freeze-backbone --draft-adapter-rank 128 \
    --speculator-type "$SPECULATOR_TYPE" \
    --init-from-dflash "$DFLASH_WEIGHTS" \
    --draft-config "$DFLASH_CONFIG" \
    --block-size "$BS" \
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

echo "Done. block${BS} checkpoint saved to $SAVE_PATH"
echo "=== DSpark (z-lab DFlash, block${BS}) finished at $(date) ==="
