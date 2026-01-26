#!/usr/bin/env bash
set -euo pipefail

BIN="./build-release/test_alpha_blend"
INPUT="data/png_test"
OUT_ROOT="results_release"

CPUSET="${CPUSET:-0-15}"
ITERS="${ITERS:-10}"
SAVE="${SAVE:-0}"
OPACITY="${OPACITY:-0.7}"
MODE="${MODE:-over}"

THREADS_CSV="${THREADS_CSV:-1,2,4,8,16,24,32}"

mkdir -p "$OUT_ROOT"

run_case() {
  local impl="$1"      # scalar / simd
  local exec="$2"      # seq / par
  local threads="$3"   # "1" or "1,2,4,..."

  local out_dir="${OUT_ROOT}/blend_results_${impl}_${exec}"
  mkdir -p "$out_dir"

  echo "== impl=$impl exec=$exec threads=$threads cpuset=$CPUSET =="
  taskset -c "$CPUSET" \
    "$BIN" "$INPUT" "$out_dir" "$OPACITY" "$MODE" "$impl" "$ITERS" "$SAVE" \
    --exec-mode "$exec" --num-threads "$threads"
}

# Sequential baselines
run_case scalar seq 1
run_case simd   seq 1

# Parallel sweeps (handled internally by your program)
run_case scalar par "$THREADS_CSV"
run_case simd   par "$THREADS_CSV"
