#!/bin/bash
# Online DSpark Training Script: initialize the DSpark backbone from a trained
# DFlash checkpoint (Ascend NPU, 4x NPU).
#
# DSpark extends DFlash with a Markov head (intra-block token dependency) and a
# confidence head (per-position acceptance prediction). The DSpark decoder body
# (layers / fc / hidden_norm / norm) is structurally identical to DFlash, so a
# trained DFlash checkpoint can seed the DSpark backbone via --init-from-dflash;
# only the two new heads start from random init and are trained normally.
#
# The trained checkpoint is a standard DSpark speculator (config.json with
# speculators_model_type="dspark"), directly loadable with
#   DSparkDraftModel.from_pretrained(<save_path>)  or  vllm serve <save_path>.
#
# This version is for Ascend NPU:
#   * Uses ASCEND_RT_VISIBLE_DEVICES (instead of CUDA_VISIBLE_DEVICES).
#   * Uses --draft-attn-impl sdpa (flex attention is unsupported on NPU).
#   * 4 NPUs total: NPU 0 runs the vLLM server, NPUs 1-3 run training (DDP).
#   * Writable paths (logs / output / hidden states) live on /tmp (overlay fs)
#     because /home/model is an object-store mount that does not support
#     symlinks, process substitution, or tmp-file+rename used by save_pretrained.
#     The verifier and DFlash source weights are read from /home/model.
#
# Usage (run in background, monitorable at any time):
#   setsid nohup bash examples/train/dspark_from_dflash_qwen3_8b_sharegpt_online.sh \
#       >/dev/null 2>&1 &
#
# Monitor:
#   tail -f /tmp/dspark_from_dflash_logs/latest.log
#
# Stop:
#   kill -TERM -$(cat /tmp/dspark_from_dflash_logs/dspark_train.pid)

### Example E2E run: DSpark Qwen3-8B on ShareGPT, backbone init from a DFlash ckpt ###

set -euo pipefail

# ============ Environment / Interpreter ============
# Python interpreter with torch + torch_npu + vllm + vllm_ascend installed.
PYTHON="${PYTHON:-/usr/local/python3.12.13/bin/python3}"
TORCHRUN="${TORCHRUN:-/usr/local/python3.12.13/bin/torchrun}"

# Run from the repo root so relative script paths resolve from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
# ===============================================

# ============ Logging (background-friendly, no symlinks/proc-sub) ============
LOG_DIR="/tmp/dspark_from_dflash_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/latest.log"
PID_FILE="$LOG_DIR/dspark_train.pid"
# Direct redirect (no tee / no process substitution) for object-store safety.
: > "$LOG_FILE"  # truncate
exec > "$LOG_FILE" 2>&1
echo "$$" > "$PID_FILE"
# ===============================================

# ============ Configuration ============
# The verifier (base) model the DFlash checkpoint was trained against.
MODEL="/home/model/Qwen/Qwen3-8B"

# A trained, speculators-format DFlash checkpoint used to seed the DSpark
# backbone. This is the z-lab `dflash_qwen3_8b_block7` checkpoint converted to
# speculators format (see README / `speculators convert --algorithm dflash`).
DFLASH_CKPT="/home/model/dflash_qwen3_8b_block7_speculators"

DATASET="sharegpt"                       # sharegpt, ultrachat, or custom path
OUTPUT_DIR="/tmp/dspark_from_dflash_output"
HIDDEN_STATES_PATH="/tmp/dspark_hidden_states"  # shared by vLLM (write) & dataloader (read)
VLLM_PORT=8000
MAX_SAMPLES=5000
SEQ_LENGTH=4096
EPOCHS=10
LR=3e-4

# DSpark parameters -- MUST match the DFlash source ckpt's config.json for
# block-size / target-layer-ids / mask-token-id, so the borrowed backbone
# tensors line up by name and shape. The DFlash ckpt here uses block_size=7,
# aux_hidden_state_layer_ids=[2,10,18,26,34], mask_token_id=151669, 5 decoder
# layers, and the full verifier vocab (151936) -- so --draft-vocab-size is
# omitted (no vocab pruning), matching the DFlash source.
SPECULATOR_TYPE="dspark"
BLOCK_SIZE=7
# max_anchors caps the per-batch draft positions (num_anchors * block_size),
# which drives the size of the targets/logits tensors ([1, N*7, vocab_size]).
# With the full 151936 vocab, max_anchors=3072 needs ~6.5 GiB just for targets
# and OOMs the 60 GiB NPUs. The DFlash source ckpt was trained with
# num_anchors=512, so use 512 here -- both memory-safe and config-consistent.
MAX_ANCHORS=512
TARGET_LAYER_IDS="2 10 18 26 34"         # must match vLLM's aux hidden-state layers
MASK_TOKEN_ID=151669                     # copy from the DFlash ckpt's config.json

# Markov + confidence head settings (DSpark-only; trained from scratch).
MARKOV_RANK=256
MARKOV_HEAD_TYPE="vanilla"               # vanilla | gated | rnn
LOSS_FN='{"ce": 0.1, "tv": 0.9}'
CONFIDENCE_HEAD_ALPHA=1.0

# NPU assignments (online training needs separate NPUs for vLLM and training).
# 4 NPUs total: 1 for the vLLM server, 3 for training (data parallel).
VLLM_NPUS="0"
TRAIN_NPUS="1,2,3"
NUM_TRAIN_NPUS=3
# =======================================

echo "=== DSpark (from DFlash) NPU training started at $(date) ==="
echo "Log file        : $LOG_FILE"
echo "Verifier        : $MODEL"
echo "DFlash src      : $DFLASH_CKPT"
echo "Output          : $OUTPUT_DIR"
echo "Hidden states   : $HIDDEN_STATES_PATH"
echo "vLLM NPUs       : $VLLM_NPUS | Training NPUs: $TRAIN_NPUS ($NUM_TRAIN_NPUS-way DDP)"
echo "Python          : $PYTHON"

# Step 1: Prepare data (skipped automatically if $OUTPUT_DIR already exists)
echo "=== Step 1: Preparing data ==="
"$PYTHON" scripts/prepare_data.py \
    --model "$MODEL" \
    --data "$DATASET" \
    --output "$OUTPUT_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --seq-length "$SEQ_LENGTH"

# Step 2: Launch vLLM server in the background (serves verifier hidden states).
# --hidden-states-path MUST match the training dataloader's path so generated
# hidden states are found.
echo "=== Step 2: Launching vLLM server ==="
# --gpu-memory-utilization 0.85: the default 0.92 needs 56.08 GiB but only ~55.6
# GiB is free on the 64 GiB NPU, so the engine core fails to init. 0.85 fits.
# --max-model-len 8192: caps the KV cache (verifier max_position_embeddings is
# 40960); 8192 comfortably covers the training total-seq-len of 4096.
ASCEND_RT_VISIBLE_DEVICES="$VLLM_NPUS" "$PYTHON" scripts/launch_vllm.py "$MODEL" \
    --hidden-states-path "$HIDDEN_STATES_PATH" \
    --target-layer-ids $TARGET_LAYER_IDS \
    -- --port "$VLLM_PORT" \
       --gpu-memory-utilization 0.85 \
       --max-model-len 8192 &
VLLM_PID=$!
echo "vLLM server PID: $VLLM_PID"

# Ensure vLLM is cleaned up on exit
cleanup() {
    echo "Stopping vLLM server (PID $VLLM_PID)..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for vLLM server to be ready..."
# Poll /v1/models (not /health): on Ascend, /health can return 200 before the
# engine core has finished initializing, which let training race ahead of a
# still-loading (or failing) server. /v1/models returns 200 only once the
# engine is loaded.
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

# Step 3: Train DSpark, seeding the backbone from the DFlash checkpoint.
#   --init-from-dflash  : load the DFlash backbone (layers/fc/norms) into DSpark.
#   --draft-config       : reuse the DFlash ckpt's decoder config so the DSpark
#                          decoder shape matches the borrowed weights exactly.
# The Markov + confidence heads start from random init and train normally.
# --draft-vocab-size is omitted: the DFlash source uses the full verifier vocab.
echo "=== Step 3: Training (DSpark, init from DFlash) ==="
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
    --draft-config "$DFLASH_CKPT" \
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

echo "Done. Standard DSpark checkpoint saved to $OUTPUT_DIR/checkpoints/"
echo "Serve it with:  vllm serve $OUTPUT_DIR/checkpoints/<epoch>"
echo "=== DSpark (from DFlash) NPU training finished at $(date) ==="
