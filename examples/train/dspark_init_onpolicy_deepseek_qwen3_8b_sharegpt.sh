#!/bin/bash
# DSpark: Fine-tune DeepSeek official dspark checkpoint (co-trained backbone+markov, 6.35 baseline)
# with on-policy scheduled sampling. Unfrozen — backbone+markov adapt together.
set -euo pipefail
PYTHON="${PYTHON:-/usr/local/python3.12.13/bin/python3}"
TORCHRUN="${TORCHRUN:-/usr/local/python3.12.13/bin/torchrun}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"; cd "$REPO_ROOT"

LOG_DIR="/home/dataset/dspark_init_op_logs"; mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/latest.log"; : > "$LOG_FILE"; exec > "$LOG_FILE" 2>&1
echo "$$" > "$LOG_DIR/dspark_train.pid"

MODEL="/home/model/Qwen/Qwen3-8B"
# DeepSeek official dspark checkpoint (co-trained backbone+markov, 6.35 baseline)
# Converted to speculators format with SWA config
DSPARK_CKPT="/tmp/dspark_deepseek_speculators"

DATASET="sharegpt"
OUTPUT_DIR="/home/dataset/dspark_init_op_output"
HIDDEN_STATES_PATH="/home/dataset/dspark_init_op_hs"
VLLM_PORT=8000; MAX_SAMPLES=5000; SEQ_LENGTH=4096; EPOCHS=5; LR=1e-4

BLOCK_SIZE=7; MAX_ANCHORS=512; TARGET_LAYER_IDS="2 10 18 26 34"; MASK_TOKEN_ID=151669
MARKOV_RANK=256; LOSS_FN='{"ce": 0.1, "tv": 0.9}'; CONFIDENCE_HEAD_ALPHA=1.0
ON_POLICY_WARMUP_RATIO=0.5
VLLM_NPUS="0"; TRAIN_NPUS="1,2,3"; NUM_TRAIN_NPUS=3

echo "=== DSpark init+on-policy fine-tune started at $(date) ==="
echo "Init: $DSPARK_CKPT (DeepSeek co-trained dspark, 6.35 baseline)"
echo "Key: from_pretrained + unfrozen + on_policy(0.5) + SWA + lr=$LR"

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
# --from-pretrained loads the full dspark checkpoint (co-trained backbone+markov)
# NO --freeze-backbone: allow fine-tuning (gentle, lr=1e-4)
# NO --init-from-dflash: no zero-init W2 (use pre-trained W2 from dspark)
# --on-policy-sampling: scheduled sampling to eliminate exposure bias
# --sample-from-anchor: sfa=true (matches dspark convention)
ASCEND_RT_VISIBLE_DEVICES="$TRAIN_NPUS" "$TORCHRUN" --standalone --nproc_per_node "$NUM_TRAIN_NPUS" \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" --data-path "$OUTPUT_DIR" \
    --hidden-states-path "$HIDDEN_STATES_PATH" \
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
    --save-path "$OUTPUT_DIR/checkpoints" --log-dir "$LOG_DIR/tensorboard" \
    --epochs "$EPOCHS" --lr "$LR" --total-seq-len "$SEQ_LENGTH" \
    --speculator-type dspark \
    --from-pretrained "$DSPARK_CKPT" \
    --sample-from-anchor --on-policy-sampling --on-policy-warmup-ratio "$ON_POLICY_WARMUP_RATIO" \
    --block-size "$BLOCK_SIZE" --max-anchors "$MAX_ANCHORS" \
    --target-layer-ids $TARGET_LAYER_IDS --mask-token-id "$MASK_TOKEN_ID" \
    --markov-rank "$MARKOV_RANK" --markov-head-type vanilla \
    --enable-confidence-head --confidence-head-with-markov \
    --loss-fn "$LOSS_FN" --confidence-head-alpha "$CONFIDENCE_HEAD_ALPHA" \
    --draft-attn-impl sdpa --on-missing generate --on-generate delete
echo "=== finished at $(date) ==="
