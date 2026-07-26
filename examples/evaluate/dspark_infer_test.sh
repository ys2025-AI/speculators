#!/bin/bash
# Inference test: serve the best DSpark checkpoint with vLLM (Ascend NPU) and
# measure speculative-decoding acceptance via GuideLLM.
#
# Usage: bash examples/evaluate/dspark_infer_test.sh [CKPT_PATH]

set -euo pipefail

PYTHON="${PYTHON:-/usr/local/python3.12.13/bin/python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

CKPT="${1:-/home/model/dspark_ckpt/zlab_block7_swa_sfa_false/checkpoints/4}"
NPU="${NPU:-0}"
PORT=8000
SERVER_URL="http://localhost:${PORT}"

LOG_DIR="/tmp/dspark_infer_test_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/latest.log"
: > "$LOG_FILE"
exec > "$LOG_FILE" 2>&1

echo "=== DSpark inference test started at $(date) ==="
echo "Checkpoint: $CKPT"
echo "NPU: $NPU  Port: $PORT"

echo "=== Step 1: Launching vLLM server (speculator serve) ==="
ASCEND_RT_VISIBLE_DEVICES="$NPU" vllm serve "$CKPT" \
    --port "$PORT" \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 &
VLLM_PID=$!

cleanup() {
    echo "Stopping vLLM server (PID $VLLM_PID)..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for vLLM server to be ready..."
elapsed=0
while true; do
    if curl -sf "${SERVER_URL}/v1/models" > /dev/null 2>&1; then
        echo "vLLM server ready (after ${elapsed}s)."
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vLLM server exited before ready. Aborting."
        exit 1
    fi
    elapsed=$((elapsed + 3))
    if [ "$elapsed" -ge 600 ]; then
        echo "ERROR: vLLM server not ready within 600s. Aborting."
        exit 1
    fi
    sleep 3
done

echo "=== Step 2: Quick smoke test (single completion) ==="
curl -s "${SERVER_URL}/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{"model": "'"$CKPT"'", "prompt": "The capital of France is", "max_tokens": 32, "temperature": 0}' \
    -o "$LOG_DIR/smoke_response.json" 2>&1 || true
echo "Smoke response saved to $LOG_DIR/smoke_response.json"
cat "$LOG_DIR/smoke_response.json" 2>/dev/null | head -c 800
echo ""

echo "=== Step 3: Acceptance evaluation (GuideLLM throughput) ==="
"$PYTHON" scripts/evaluate/evaluate.py \
    --target "${SERVER_URL}/v1" \
    --dataset "RedHatAI/speculator_benchmarks" \
    --output-dir "$LOG_DIR/eval_results" \
    throughput \
    --subsets "HumanEval" \
    --max-requests 50

echo ""
echo "=== Done. Results in $LOG_DIR/eval_results ==="
echo "=== DSpark inference test finished at $(date) ==="
