#!/bin/bash
# DSpark: frozen DFlash backbone + sfa=false + high-CE loss (ce:0.9/tv:0.1).
# Preserve the DFlash backbone (frozen), train only Markov+confidence heads under the
# stable sfa=false regime, with a high CE weight (next-token accuracy). 10 epochs ShareGPT.
set -uo pipefail
PYTHON="${PYTHON:-/usr/local/python3.12.13/bin/python3}"
TORCHRUN="${TORCHRUN:-/usr/local/python3.12.13/bin/torchrun}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"; cd "$REPO_ROOT"

LABEL="frozen_sfa_false_hce"
LOG_DIR="/tmp/dspark_${LABEL}_logs"; mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/latest.log"; : > "$LOG_FILE"; exec > "$LOG_FILE" 2>&1

MODEL="/home/model/Qwen/Qwen3-8B"
DFLASH_INIT="/home/model/dspark_ckpt/dflash_zlab_speculators"          # z-lab SWA DFlash (best sfa=false init)
DRAFT_CONFIG="/home/model/dspark_ckpt/dflash_zlab_swa_config_block7"    # block7 SWA draft-config
DATA_PATH="/tmp/dspark_from_dflash_output"                              # reuse ShareGPT
OUTPUT_DIR="/tmp/dspark_${LABEL}_output"
HIDDEN_STATES_PATH="/tmp/dspark_${LABEL}_hs"
VLLM_PORT=8000
EPOCHS=10
LR=3e-4
SEQ_LENGTH=4096
LOSS_FN='{"ce": 0.9, "tv": 0.1}'   # HIGH CE (was 0.1/0.9)

echo "=== DSpark [${LABEL}] started at $(date) ==="
echo "DFLASH_INIT : $DFLASH_INIT  | DRAFT_CONFIG: $DRAFT_CONFIG"
echo "freeze_backbone=YES | sfa=false (--no-sample-from-anchor) | loss=$LOSS_FN | epochs=$EPOCHS"

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

echo "=== Step 3: Training (frozen backbone, sfa=false, high CE) ==="
ASCEND_RT_VISIBLE_DEVICES="1,2,3" "$TORCHRUN" --standalone --nproc_per_node 3 \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" --data-path "$DATA_PATH" \
    --hidden-states-path "$HIDDEN_STATES_PATH" \
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
    --save-path "$OUTPUT_DIR/checkpoints" --log-dir "$LOG_DIR/tensorboard" \
    --epochs "$EPOCHS" --lr "$LR" --total-seq-len "$SEQ_LENGTH" \
    --no-sample-from-anchor \
    --speculator-type dspark \
    --init-from-dflash "$DFLASH_INIT" --draft-config "$DRAFT_CONFIG" \
    --freeze-backbone \
    --block-size 7 --max-anchors 512 \
    --target-layer-ids 2 10 18 26 34 --mask-token-id 151669 \
    --markov-rank 256 --markov-head-type vanilla \
    --enable-confidence-head --confidence-head-with-markov \
    --loss-fn "$LOSS_FN" --confidence-head-alpha 1.0 \
    --draft-attn-impl sdpa --on-missing generate --on-generate delete

echo "Done [${LABEL}]. saved to $OUTPUT_DIR/checkpoints/"
echo "=== DSpark [${LABEL}] finished at $(date) ==="
