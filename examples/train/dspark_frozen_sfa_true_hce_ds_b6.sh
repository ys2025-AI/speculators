#!/bin/bash
# DSpark: DeepSeek dflash init + sfa=true + FROZEN backbone + high-CE (ce:0.9/tv:0.1),
# block_size=6 (=> num_spec=6 for sfa=true), markov on, 10ep.
# Note: sfa=true -> num_spec = block_size (config.py); so num_spec=6 needs block_size=6.
# The DeepSeek dflash init (block7-trained) backbone is block-agnostic (decoder weights), loaded
# into a block6 DSpark; block_size=6 only changes the diffusion block size.
set -uo pipefail
PYTHON="${PYTHON:-/usr/local/python3.12.13/bin/python3}"
TORCHRUN="${TORCHRUN:-/usr/local/python3.12.13/bin/torchrun}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"; cd "$REPO_ROOT"

LABEL="frozen_sfa_true_hce_ds_b6"
LOG_DIR="/tmp/dspark_${LABEL}_logs"; mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/latest.log"; : > "$LOG_FILE"; exec > "$LOG_FILE" 2>&1

MODEL="/home/model/Qwen/Qwen3-8B"
DFLASH_INIT="/home/model/dflash_qwen3_8b_block7_speculators"   # DeepSeek dflash (speculators fmt)
DRAFT_CONFIG="/home/model/dflash_qwen3_8b_block7_speculators"   # full-attn block7 decoder config
DATA_PATH="/tmp/dspark_from_dflash_output"
OUTPUT_DIR="/tmp/dspark_${LABEL}_output"
HIDDEN_STATES_PATH="/tmp/dspark_${LABEL}_hs"
VLLM_PORT=8000
EPOCHS=10
LR=3e-4
SEQ_LENGTH=4096
LOSS_FN='{"ce": 0.9, "tv": 0.1}'
BLOCK_SIZE=6   # sfa=true -> num_spec = block_size = 6

echo "=== DSpark [${LABEL}] started at $(date) ==="
echo "DFLASH_INIT : $DFLASH_INIT (DeepSeek dflash)"
echo "block_size=$BLOCK_SIZE (sfa=true -> num_spec=$BLOCK_SIZE) | freeze_backbone=YES | sfa=true | loss=$LOSS_FN | ep=$EPOCHS | markov=on"

if [ -f "$DATA_PATH/data-00000-of-00001.arrow" ]; then echo "=== Step 1: data found, skip ==="; else
    "$PYTHON" scripts/prepare_data.py --model "$MODEL" --data sharegpt --output "$DATA_PATH" --max-samples 5000 --seq-length "$SEQ_LENGTH"; fi

echo "=== Step 2: Launching vLLM ==="
ASCEND_RT_VISIBLE_DEVICES="0" "$PYTHON" scripts/launch_vllm.py "$MODEL" \
    --hidden-states-path "$HIDDEN_STATES_PATH" --target-layer-ids 2 10 18 26 34 \
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

echo "=== Step 3: Training (DeepSeek init, frozen, sfa=true, high CE, block6) ==="
ASCEND_RT_VISIBLE_DEVICES="1,2,3" "$TORCHRUN" --standalone --nproc_per_node 3 \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" --data-path "$DATA_PATH" \
    --hidden-states-path "$HIDDEN_STATES_PATH" \
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
    --save-path "$OUTPUT_DIR/checkpoints" --log-dir "$LOG_DIR/tensorboard" \
    --epochs "$EPOCHS" --lr "$LR" --total-seq-len "$SEQ_LENGTH" \
    --speculator-type dspark \
    --init-from-dflash "$DFLASH_INIT" --draft-config "$DRAFT_CONFIG" \
    --freeze-backbone \
    --block-size "$BLOCK_SIZE" --max-anchors 512 \
    --target-layer-ids 2 10 18 26 34 --mask-token-id 151669 \
    --markov-rank 256 --markov-head-type vanilla \
    --enable-confidence-head --confidence-head-with-markov \
    --loss-fn "$LOSS_FN" --confidence-head-alpha 1.0 \
    --draft-attn-impl sdpa --on-missing generate --on-generate delete

echo "Done [${LABEL}]. saved to $OUTPUT_DIR/checkpoints/"
echo "=== DSpark [${LABEL}] finished at $(date) ==="
