#!/bin/bash
# Online DSpark Training Script (Ascend NPU, 4x NPU)
#
# Runs the full online DSpark training pipeline: data preparation, vLLM server
# launch, and training (with hidden states generated on-the-fly from the live
# server). DSpark extends DFlash with a Markov head (intra-block token
# dependency) and a confidence head (per-position acceptance prediction); the
# pipeline is the DFlash one plus a few DSpark-specific flags.
#
# This version is adapted for Ascend NPU:
#   * Uses ASCEND_RT_VISIBLE_DEVICES (instead of CUDA_VISIBLE_DEVICES).
#   * Uses --draft-attn-impl sdpa (flex attention is unsupported on NPU).
#   * 4 NPUs total: NPU 0 runs the vLLM server, NPUs 1-3 run training (DDP).
#
# Verifier notes for Qwen3.5-0.8B:
#   * Qwen3.5-0.8B is a multimodal (Qwen3_5ForConditionalGeneration) model with
#     a hybrid linear/full-attention text decoder (24 layers). vLLM 0.22.1 +
#     vllm-ascend register Qwen3_5ForConditionalGeneration, and transformers
#     >=5.2.0 provides the qwen3_5 model type (installed: 5.5.4).
#   * ShareGPT is text-only. speculators' data layer detects Qwen3.5-0.8B's
#     multimodal AutoProcessor and routes text-only samples through the
#     Completions API with pre-truncated prompt_token_ids (no image/video), so
#     no multimodal data is required.
#   * The DSpark draft decoder is synthesized with the qwen3 architecture
#     (default for dflash/dspark), independent of the verifier's qwen3_5_text
#     decoder -- it consumes the verifier's hidden states as input. This
#     mirrors the Qwen3-0.6B reference run.
#   * --enforce-eager is REQUIRED on Ascend NPU for Qwen3.5: the text decoder
#     is a qwen3_next-style hybrid linear/full-attention model whose AOT-
#     compiled graph launches _layer_norm_fwd_1pass_kernel_npu with
#     coreDim=65536, exceeding the NPU hardware limit of 65535 (EE1003
#     Invalid_Argument) during profile_run. Qwen3-0.6B (standard Qwen3ForCausalLM,
#     full attention) does not trigger this and ran compiled. Eager mode uses a
#     different kernel launch grid and avoids the overflow; the perf impact on
#     hidden-state extraction (max_tokens=1) is negligible.
#   * --max-model-len 5120 is REQUIRED on Ascend NPU for Qwen3.5: the NPU
#     layer-norm kernel's coreDim scales with the profile_run token budget,
#     which defaults to max_position_embeddings (262144 for Qwen3.5-0.8B).
#     262144 / 4 = 65536 > 65535 (hardware coreDim limit) -> EE1003 crash in
#     profile_run (same operator, same coreDim, in BOTH eager and compiled
#     mode -- it is not a compile-path artifact). Qwen3-0.6B (max_position_embeddings
#     40960 -> 40960/4=10240) stayed under the limit. Capping max_model_len to
#     the actual training sequence length (+1 generation token + headroom) brings
#     coreDim to 5120/4=1280, well under 65535. 5120 >= 4096 (seq_length) + 1
#     (max_tokens) so no prompt is rejected.
#
# Usage (run in background, monitorable at any time):
#   setsid nohup bash examples/train/dspark_qwen3_5_0_8b_sharegpt_online.sh >/dev/null 2>&1 &
#
# Monitor:
#   tail -f /home/dataset/dspark_logs/latest.log     # live training log
#   watch -n 2 npu-smi info                           # NPU utilization
#   cat /home/dataset/dspark_logs/dspark_train.pid    # main PID
#
# Stop:
#   kill -TERM -$(cat /home/dataset/dspark_logs/dspark_train.pid)   # whole group
#
# For a detailed walkthrough, see
# https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/train_dflash_online/

### Example E2E run for DSpark Qwen3.5-0.8B on 5k samples from ShareGPT (4x Ascend NPU) ###

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

# ============ Logging (background-friendly) ============
LOG_DIR="/home/dataset/dspark_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/dspark_train_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="${LOG_DIR}/dspark_train.pid"
# Mirror all stdout/stderr to a timestamped log file.
exec > >(tee -a "$LOG_FILE") 2>&1
# Stable symlink so "tail -f latest.log" always follows the current run.
ln -sf "$LOG_FILE" "$LOG_DIR/latest.log"
echo "$$" > "$PID_FILE"
# ===============================================

# ============ Configuration ============
MODEL="/home/model/Qwen3.5-0.8B"
DATASET="sharegpt"                # sharegpt, ultrachat, or path to custom data
OUTPUT_DIR="/home/dataset/dspark_qwen3_5_0_8b_sharegpt_output"
VLLM_PORT=8000
MAX_SAMPLES=5000
SEQ_LENGTH=4096
EPOCHS=5
LR=3e-4

# DSpark-specific parameters
SPECULATOR_TYPE="dspark"
BLOCK_SIZE=8
MAX_ANCHORS=3072
NUM_LAYERS=3
DRAFT_VOCAB_SIZE=32000
# Recomputed for Qwen3.5-0.8B's 24-layer text decoder (Qwen3-0.6B had 28 layers,
# so the reference run used "2 14 25"). Uses the same default formula
# [2, num_hidden_layers // 2, num_hidden_layers - 3] = [2, 12, 21]. Layer 25
# does NOT exist in a 24-layer model and would crash vLLM hidden-state extraction.
# These are the 3 draft target layers (one per --num-layers); launch_vllm.py
# additionally appends layer 24 (num_hidden_layers) as the verifier's final
# hidden state. Qwen3.5-0.8B alternates linear/full attention every 4 layers
# (full-attention layers: 3, 7, 11, 15, 19, 23); if you prefer full-attention
# targets, consider e.g. "3 11 19" -- but "2 12 21" matches the reference run's
# layer-spread strategy and is the codebase default.
TARGET_LAYER_IDS="2 12 21"  # Must match vLLM's eagle_aux_hidden_state_layer_ids

# Markov + confidence head settings
MARKOV_RANK=256
MARKOV_HEAD_TYPE="vanilla"   # vanilla | gated | rnn
LOSS_FN='{"ce": 0.1, "tv": 0.9}'
CONFIDENCE_HEAD_ALPHA=1.0

# NPU assignments (online training needs separate NPUs for vLLM and training).
# 4 NPUs total: 1 for the vLLM server, 3 for training (data parallel).
VLLM_NPUS="0"
TRAIN_NPUS="1,2,3"
NUM_TRAIN_NPUS=3
# =======================================

echo "=== DSpark NPU training started at $(date) ==="
echo "Log file : $LOG_FILE"
echo "PID file : $PID_FILE (main PID: $$)"
echo "Model    : $MODEL"
echo "Output   : $OUTPUT_DIR"
echo "vLLM NPUs: $VLLM_NPUS | Training NPUs: $TRAIN_NPUS ($NUM_TRAIN_NPUS-way DDP)"
echo "Python   : $PYTHON"

# Step 1: Prepare data (skipped automatically if $OUTPUT_DIR already exists)
echo "=== Step 1: Preparing data ==="
"$PYTHON" scripts/prepare_data.py \
    --model "$MODEL" \
    --data "$DATASET" \
    --output "$OUTPUT_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --seq-length "$SEQ_LENGTH"

# Step 2: Launch vLLM server in the background
echo "=== Step 2: Launching vLLM server ==="
ASCEND_RT_VISIBLE_DEVICES="$VLLM_NPUS" "$PYTHON" scripts/launch_vllm.py "$MODEL" \
    --target-layer-ids $TARGET_LAYER_IDS \
    -- --port "$VLLM_PORT" --enforce-eager --max-model-len 5120 &
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
WAIT_TIMEOUT=600
ELAPSED=0
while true; do
    if curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        break
    fi
    # Abort early if the vLLM process itself died.
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

# Step 3: Train DSpark against the live vLLM server
echo "=== Step 3: Training ==="
ASCEND_RT_VISIBLE_DEVICES="$TRAIN_NPUS" "$TORCHRUN" \
    --standalone --nproc_per_node "$NUM_TRAIN_NPUS" \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --data-path "$OUTPUT_DIR" \
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --log-dir "$LOG_DIR/tensorboard" \
    --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --total-seq-len "$SEQ_LENGTH" \
    --speculator-type "$SPECULATOR_TYPE" \
    --block-size "$BLOCK_SIZE" \
    --max-anchors "$MAX_ANCHORS" \
    --num-layers "$NUM_LAYERS" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --markov-rank "$MARKOV_RANK" \
    --markov-head-type "$MARKOV_HEAD_TYPE" \
    --enable-confidence-head \
    --confidence-head-with-markov \
    --loss-fn "$LOSS_FN" \
    --confidence-head-alpha "$CONFIDENCE_HEAD_ALPHA" \
    --draft-attn-impl sdpa \
    --on-missing generate \
    --on-generate delete

echo "Done. Checkpoints saved to $OUTPUT_DIR/checkpoints/"
echo "=== DSpark NPU training finished at $(date) ==="
