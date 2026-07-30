#!/bin/bash
# On-policy DSpark Training: unfrozen backbone + co-trained Markov head +
# on-policy scheduled sampling, initialized from a DFlash checkpoint.
#
# This script validates the hypothesis that co-training the backbone with the
# Markov head (unfreezing) + on-policy scheduled sampling (sampling predecessors
# from the draft's own logits) addresses both root causes identified in the eval
# report:
#   1. Backbone-Markov co-training (DeepSeek's dspark achieves +62-115% with
#      markov; our frozen approach got -2.6 to -3.9%)
#   2. Exposure bias elimination (on-policy training matches inference-time
#      Markov chain behavior)
#
# Key differences from dspark_from_dflash_qwen3_8b_sharegpt_online.sh:
#   * NO --freeze-backbone: backbone is co-trained with the Markov head
#   * --on-policy-sampling: scheduled sampling for the Markov chain
#   * --on-policy-warmup-ratio 0.5: start 50% teacher forcing, decay to 0
#   * --sample-from-anchor: sfa=true (DSpark default, full block_size tokens)
#
# Usage:
#   setsid nohup bash examples/train/dspark_onpolicy_from_dflash_qwen3_8b_sharegpt.sh \
#       >/dev/null 2>&1 &
# Monitor:
#   tail -f /tmp/dspark_onpolicy_logs/latest.log
# Stop:
#   kill -TERM -$(cat /tmp/dspark_onpolicy_logs/dspark_train.pid)

set -euo pipefail

# ============ Environment / Interpreter ============
PYTHON="${PYTHON:-/usr/local/python3.12.13/bin/python3}"
TORCHRUN="${TORCHRUN:-/usr/local/python3.12.13/bin/torchrun}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
# ===============================================

# ============ Logging ============
LOG_DIR="/tmp/dspark_onpolicy_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/latest.log"
PID_FILE="$LOG_DIR/dspark_train.pid"
: > "$LOG_FILE"
exec > "$LOG_FILE" 2>&1
echo "$$" > "$PID_FILE"
# ===============================================

# ============ Configuration ============
MODEL="/home/model/Qwen/Qwen3-8B"

# z-lab DFlash checkpoint (block_size=16, full_attention, 4.3 GB weights).
# The decoder weights are block-size-agnostic, so they transfer to block_size=7
# without shape mismatch.
DFLASH_CKPT="/home/model/dspark_ckpt/dflash_zlab_speculators"

# SWA block7 config (block_size=7, sliding_attention, window=2048).
# Config-only directory (no weights) — provides the decoder shape that
# overrides the z-lab ckpt's original block_size=16/full_attention.
# Attention type (full vs sliding) is a config-level difference, not weight-level,
# so the z-lab weights load cleanly under this config.
DRAFT_CONFIG="/home/model/dspark_ckpt/dflash_zlab_swa_config_block7"

DATASET="sharegpt"
OUTPUT_DIR="/tmp/dspark_onpolicy_output"
HIDDEN_STATES_PATH="/tmp/dspark_onpolicy_hs"
VLLM_PORT=8000
MAX_SAMPLES=5000
SEQ_LENGTH=4096
EPOCHS=10
LR=3e-4

# DSpark parameters — MUST match the DFlash source ckpt's config.json for
# aux_hidden_state_layer_ids / mask_token_id (both z-lab and SWA config agree).
SPECULATOR_TYPE="dspark"
BLOCK_SIZE=7
MAX_ANCHORS=512
TARGET_LAYER_IDS="2 10 18 26 34"
MASK_TOKEN_ID=151669

# Markov + confidence head settings.
MARKOV_RANK=256
MARKOV_HEAD_TYPE="vanilla"
LOSS_FN='{"ce": 0.1, "tv": 0.9}'
CONFIDENCE_HEAD_ALPHA=1.0

# On-policy scheduled sampling.
ON_POLICY_WARMUP_RATIO=0.5

# NPU assignments: 1 vLLM + 3 training (DDP).
VLLM_NPUS="0"
TRAIN_NPUS="1,2,3"
NUM_TRAIN_NPUS=3
# =======================================

echo "=== On-policy DSpark (from DFlash) training started at $(date) ==="
echo "Log file        : $LOG_FILE"
echo "Verifier        : $MODEL"
echo "DFlash src (wt) : $DFLASH_CKPT"
echo "Draft config    : $DRAFT_CONFIG"
echo "Output          : $OUTPUT_DIR"
echo "Hidden states   : $HIDDEN_STATES_PATH"
echo "vLLM NPUs       : $VLLM_NPUS | Training NPUs: $TRAIN_NPUS ($NUM_TRAIN_NPUS-way DDP)"
echo "Key config      : backbone=UNFROZEN, sfa=true, on_policy_sampling=true, warmup=$ON_POLICY_WARMUP_RATIO"
echo "Python          : $PYTHON"

# Step 1: Prepare data (skipped if already exists)
echo "=== Step 1: Preparing data ==="
if [ -f "$OUTPUT_DIR/data-00000-of-00001.arrow" ]; then
    echo "Data already prepared at $OUTPUT_DIR, skipping."
else
    "$PYTHON" scripts/prepare_data.py \
        --model "$MODEL" \
        --data "$DATASET" \
        --output "$OUTPUT_DIR" \
        --max-samples "$MAX_SAMPLES" \
        --seq-length "$SEQ_LENGTH"
fi

# Step 2: Launch vLLM server
echo "=== Step 2: Launching vLLM server ==="
ASCEND_RT_VISIBLE_DEVICES="$VLLM_NPUS" "$PYTHON" scripts/launch_vllm.py "$MODEL" \
    --hidden-states-path "$HIDDEN_STATES_PATH" \
    --target-layer-ids $TARGET_LAYER_IDS \
    -- --port "$VLLM_PORT" \
       --gpu-memory-utilization 0.85 \
       --max-model-len 8192 \
       --enforce-eager &
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
        echo "ERROR: vLLM server process (PID $VLLM_PID) exited before becoming ready. Aborting."
        exit 1
    fi
    ELAPSED=$((ELAPSED + 2))
    if [ "$ELAPSED" -ge "$WAIT_TIMEOUT" ]; then
        echo "ERROR: vLLM server did not become ready within ${WAIT_TIMEOUT}s. Aborting."
        exit 1
    fi
    sleep 2
done
echo "vLLM server ready (after ${ELAPSED}s)."

# Step 3: Train DSpark with on-policy scheduled sampling.
#   --init-from-dflash  : load z-lab DFlash backbone weights (layers/fc/norms)
#   --draft-config       : use SWA block7 config (block_size=7, sliding_attention)
#   NO --freeze-backbone: backbone is co-trained with markov head (key change!)
#   --on-policy-sampling : enable scheduled sampling for markov chain
#   --on-policy-warmup-ratio 0.5: start 50% TF, decay to 0 (pure on-policy)
#   --sample-from-anchor : sfa=true (DSpark default, full block_size tokens)
#   Markov + confidence heads start from random init and train normally.
echo "=== Step 3: Training (DSpark, on-policy, backbone unfrozen, init from z-lab DFlash) ==="
ASCEND_RT_VISIBLE_DEVICES="$TRAIN_NPUS" "$TORCHRUN" \
    --standalone --nproc_per_node "$NUM_TRAIN_NPUS" \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --data-path "$OUTPUT_DIR" \
    --hidden-states-path "$HIDDEN_STATES_PATH" \
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --log-dir "$LOG_DIR/tensorboard" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --total-seq-len "$SEQ_LENGTH" \
    --speculator-type "$SPECULATOR_TYPE" \
    --init-from-dflash "$DFLASH_CKPT" \
    --draft-config "$DRAFT_CONFIG" \
    --sample-from-anchor \
    --on-policy-sampling \
    --on-policy-warmup-ratio "$ON_POLICY_WARMUP_RATIO" \
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

echo "Done. DSpark checkpoint saved to $OUTPUT_DIR/checkpoints/"
echo "Serve it with:  vllm serve $OUTPUT_DIR/checkpoints/<epoch>"
echo "=== On-policy DSpark training finished at $(date) ==="
