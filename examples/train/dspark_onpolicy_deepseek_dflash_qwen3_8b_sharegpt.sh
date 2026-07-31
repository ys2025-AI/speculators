#!/bin/bash
# On-policy DSpark Training: DeepSeek DFlash backbone (sfa=true native) +
# unfrozen co-training + on-policy scheduled sampling.
#
# This is the optimal path identified in onpolicy_dspark_experiment_report.md:
#   DeepSeek dflash backbone (sfa=true native, accept_len=5.86 baseline)
#     + unfrozen backbone co-training (backbone learns to delegate bigram to Markov)
#     + on-policy scheduled sampling (eliminates train-inference exposure bias)
#     + zero-init W2 (B=0 start, no Markov corruption)
#
# Previous experiment used z-lab backbone (sfa=false trained) → accept_len=1.38
# baseline. This run uses DeepSeek backbone (sfa=true native) → accept_len=5.86
# baseline. Expected: on-policy training pushes from 5.86 toward 6.35.
#
# Usage:
#   setsid nohup bash examples/train/dspark_onpolicy_deepseek_dflash_qwen3_8b_sharegpt.sh \
#       >/dev/null 2>&1 &
# Monitor:
#   tail -f /tmp/dspark_onpolicy_ds_logs/latest.log

set -euo pipefail

# ============ Environment / Interpreter ============
PYTHON="${PYTHON:-/usr/local/python3.12.13/bin/python3}"
TORCHRUN="${TORCHRUN:-/usr/local/python3.12.13/bin/torchrun}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# ============ Logging ============
LOG_DIR="/tmp/dspark_onpolicy_ds_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/latest.log"
PID_FILE="$LOG_DIR/dspark_train.pid"
: > "$LOG_FILE"
exec > "$LOG_FILE" 2>&1
echo "$$" > "$PID_FILE"
# ===============================================

# ============ Configuration ============
MODEL="/home/model/Qwen/Qwen3-8B"

# DeepSeek DFlash checkpoint (block_size=7, full_attention, sfa=true native,
# 4.3GB). This backbone was trained with sfa=true convention, matching DSpark's
# default. accept_len=5.86 standalone (vs z-lab sfa=true=1.38).
DFLASH_CKPT="/home/model/dflash_qwen3_8b_block7_speculators"

# SWA config (block_size=7, sliding_attention, window=2048).
# Config-only directory — provides the decoder shape that overrides the
# DeepSeek ckpt's original full_attention to SWA. Attention type is a
# config-level difference (not weight-level), so the DeepSeek weights load
# cleanly under this config. SWA reduces KV cache for inference (2.2x throughput).
DRAFT_CONFIG="/tmp/dflash_deepseek_swa_block7"

DATASET="sharegpt"
OUTPUT_DIR="/tmp/dspark_onpolicy_ds_output"
HIDDEN_STATES_PATH="/tmp/dspark_onpolicy_ds_hs"
VLLM_PORT=8000
MAX_SAMPLES=5000
SEQ_LENGTH=4096
EPOCHS=10
LR=3e-4

# DSpark parameters — match the DeepSeek DFlash ckpt's config.json.
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

echo "=== On-policy DSpark (DeepSeek DFlash backbone) started at $(date) ==="
echo "Verifier        : $MODEL"
echo "DFlash backbone  : $DFLASH_CKPT (DeepSeek, sfa=true native)"
echo "Output          : $OUTPUT_DIR"
echo "vLLM NPUs       : $VLLM_NPUS | Training NPUs: $TRAIN_NPUS ($NUM_TRAIN_NPUS-way DDP)"
echo "Key config      : backbone=UNFROZEN, sfa=true, on_policy=true, warmup=$ON_POLICY_WARMUP_RATIO"

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

# Step 2: Launch vLLM server (--enforce-eager avoids cudagraph compilation bug)
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
        echo "ERROR: vLLM server exited before becoming ready. Aborting."
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
#   --init-from-dflash  : load DeepSeek DFlash backbone (sfa=true native, 5.86 baseline)
#   --draft-config       : SWA block7 config (sliding_attention, window=2048)
#   NO --freeze-backbone : backbone co-trained with markov head
#   --on-policy-sampling  : scheduled sampling, 50% TF → 0
#   --sample-from-anchor : sfa=true (matches DeepSeek backbone convention)
echo "=== Step 3: Training (DSpark, on-policy, DeepSeek backbone, unfrozen) ==="
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
echo "=== On-policy DSpark (DeepSeek backbone) finished at $(date) ==="
