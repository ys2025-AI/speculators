#!/bin/bash
# DSpark: DeepSeek DFlash backbone (FROZEN) + zero-init W2 + on-policy training.
#
# Preserves the DeepSeek backbone's accept_len=5.86 baseline (frozen) while
# the Markov head learns useful bigram biases via on-policy scheduled sampling.
# Expected: from 5.86 baseline, on-policy Markov pushes toward 6.35.
#
# Usage:
#   setsid nohup bash examples/train/dspark_frozen_onpolicy_deepseek_qwen3_8b_sharegpt.sh \
#       >/dev/null 2>&1 &
# Monitor:
#   tail -f /tmp/dspark_frozen_ds_logs/latest.log

set -euo pipefail

PYTHON="${PYTHON:-/usr/local/python3.12.13/bin/python3}"
TORCHRUN="${TORCHRUN:-/usr/local/python3.12.13/bin/torchrun}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"; cd "$REPO_ROOT"

LOG_DIR="/tmp/dspark_frozen_ds_logs"; mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/latest.log"; : > "$LOG_FILE"; exec > "$LOG_FILE" 2>&1
echo "$$" > "$LOG_DIR/dspark_train.pid"

MODEL="/home/model/Qwen/Qwen3-8B"
DFLASH_CKPT="/home/model/dflash_qwen3_8b_block7_speculators"
DRAFT_CONFIG="/tmp/dflash_deepseek_swa_block7"
DATASET="sharegpt"
OUTPUT_DIR="/tmp/dspark_frozen_ds_output"
HIDDEN_STATES_PATH="/tmp/dspark_frozen_ds_hs"
VLLM_PORT=8000; MAX_SAMPLES=5000; SEQ_LENGTH=4096; EPOCHS=5; LR=3e-4

BLOCK_SIZE=7; MAX_ANCHORS=512; TARGET_LAYER_IDS="2 10 18 26 34"; MASK_TOKEN_ID=151669
MARKOV_RANK=256; MARKOV_HEAD_TYPE="vanilla"
LOSS_FN='{"ce": 0.1, "tv": 0.9}'; CONFIDENCE_HEAD_ALPHA=1.0
ON_POLICY_WARMUP_RATIO=0.5
VLLM_NPUS="0"; TRAIN_NPUS="1,2,3"; NUM_TRAIN_NPUS=3

echo "=== DSpark frozen+on-policy (DeepSeek backbone) started at $(date) ==="
echo "Backbone: $DFLASH_CKPT (FROZEN, sfa=true native, 5.86 baseline)"
echo "Key: frozen_backbone + zero_init_W2 + on_policy(warmup=$ON_POLICY_WARMUP_RATIO)"

if [ -f "$OUTPUT_DIR/data-00000-of-00001.arrow" ]; then echo "data exists"; else
    "$PYTHON" scripts/prepare_data.py --model "$MODEL" --data "$DATASET" \
        --output "$OUTPUT_DIR" --max-samples "$MAX_SAMPLES" --seq-length "$SEQ_LENGTH"; fi

echo "=== vLLM ==="
ASCEND_RT_VISIBLE_DEVICES="$VLLM_NPUS" "$PYTHON" scripts/launch_vllm.py "$MODEL" \
    --hidden-states-path "$HIDDEN_STATES_PATH" --target-layer-ids $TARGET_LAYER_IDS \
    -- --port "$VLLM_PORT" --gpu-memory-utilization 0.85 --max-model-len 8192 --enforce-eager &
VLLM_PID=$!
cleanup() { kill "$VLLM_PID" 2>/dev/null || true; wait "$VLLM_PID" 2>/dev/null || true; }
trap cleanup EXIT
elapsed=0
while true; do
    if curl -sf "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then break; fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then echo "ERROR: vLLM exited"; exit 1; fi
    elapsed=$((elapsed + 2)); [ "$elapsed" -ge 600 ] && { echo "ERROR: vLLM timeout"; exit 1; }; sleep 2
done
echo "vLLM ready (after ${elapsed}s)."

echo "=== Training ==="
ASCEND_RT_VISIBLE_DEVICES="$TRAIN_NPUS" "$TORCHRUN" --standalone --nproc_per_node "$NUM_TRAIN_NPUS" \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" --data-path "$OUTPUT_DIR" \
    --hidden-states-path "$HIDDEN_STATES_PATH" \
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
    --save-path "$OUTPUT_DIR/checkpoints" --log-dir "$LOG_DIR/tensorboard" \
    --epochs "$EPOCHS" --lr "$LR" --total-seq-len "$SEQ_LENGTH" \
    --speculator-type dspark \
    --init-from-dflash "$DFLASH_CKPT" --draft-config "$DRAFT_CONFIG" \
    --freeze-backbone \
    --sample-from-anchor --on-policy-sampling --on-policy-warmup-ratio "$ON_POLICY_WARMUP_RATIO" \
    --block-size "$BLOCK_SIZE" --max-anchors "$MAX_ANCHORS" \
    --target-layer-ids $TARGET_LAYER_IDS --mask-token-id "$MASK_TOKEN_ID" \
    --markov-rank "$MARKOV_RANK" --markov-head-type "$MARKOV_HEAD_TYPE" \
    --enable-confidence-head --confidence-head-with-markov \
    --loss-fn "$LOSS_FN" --confidence-head-alpha "$CONFIDENCE_HEAD_ALPHA" \
    --draft-attn-impl sdpa --on-missing generate --on-generate delete
echo "=== finished at $(date) ==="
