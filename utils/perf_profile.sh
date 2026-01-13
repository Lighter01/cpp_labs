#!/usr/bin/env bash
# One-stop perf stat + perf record + flamegraph render (basic and/or DWARF+inline)
#
# Usage:
#   ./perf_profile.sh <testname> [--basic|--dwarf-inline|--both] --cmd <path/to/bin> [--out <dir>] [--] [prog args...]
#
# Examples:
#   ./perf_profile.sh test1 --cmd ./bench
#   ./perf_profile.sh test2 --both --cmd ./matrix_mult --out results -- --size 256 --iters 50
#
# Outputs (in <out>/<testname>/):
#   basic:
#     perf_basic.stat  perf_basic.data  flame_basic.svg
#   dwarf+inline:
#     perf_dwarf.stat  perf_dwarf.data  flame_dwarf_inline.svg
#
# Env knobs (optional):
#   CORE=0               # CPU core to pin
#   FREQ=199             # sampling Hz (record)
#   REPS=7               # perf stat repetitions
#   PERF_DELAY=          # ms delay before sampling (record)
#   FLAMEGRAPH_DIR=$HOME/FlameGraph

set -euo pipefail

usage() {
  cat >&2 <<EOF
Usage:
  $0 <testname> [--basic|--dwarf-inline|--both] --cmd <path/to/bin> [--out <dir>] [--] [prog args...]

Notes:
  - Program arguments go after "--".
  - If no variant flag is given, defaults to --dwarf-inline.

Env (optional):
  CORE=0
  FREQ=249
  REPS=7
  PERF_DELAY=
  FLAMEGRAPH_DIR=\$HOME/FlameGraph
EOF
  exit 1
}

[[ $# -ge 1 ]] || usage
TEST="$1"; shift

# Defaults
DO_BASIC=false
DO_DWARF=false
OUT_ROOT="results"
CMD=""
PROG_ARGS=()

# Parse args (simple, not overcomplicated)
while (( "$#" )); do
  case "$1" in
    --basic)        DO_BASIC=true; shift ;;
    --dwarf-inline) DO_DWARF=true; shift ;;
    --both)         DO_BASIC=true; DO_DWARF=true; shift ;;
    --cmd)          [[ $# -ge 2 ]] || usage; CMD="$2"; shift 2 ;;
    --out)          [[ $# -ge 2 ]] || usage; OUT_ROOT="$2"; shift 2 ;;
    --)             shift; PROG_ARGS=("$@"); break ;;
    -h|--help)      usage ;;
    *)              echo "Unknown flag: $1" >&2; usage ;;
  esac
done

# default if none specified: run DWARF+inline
if ! $DO_BASIC && ! $DO_DWARF; then
  DO_DWARF=true
fi

[[ -n "$CMD" ]] || { echo "Error: --cmd <path/to/bin> is required" >&2; exit 1; }
[[ -x "$CMD" ]] || { echo "Error: CMD '$CMD' not executable" >&2; exit 1; }

RESULTS_DIR="${OUT_ROOT%/}/${TEST}"
CORE="${CORE:-0}"
FREQ="${FREQ:-199}"
REPS="${REPS:-7}"
PERF_DELAY="${PERF_DELAY:-}"
FLAMEGRAPH_DIR="${FLAMEGRAPH_DIR:-$HOME/FlameGraph}"

mkdir -p "$RESULTS_DIR"

command -v perf >/dev/null 2>&1 || { echo "perf not found"; exit 1; }
[[ -x "$(command -v awk)" ]] || { echo "awk not found"; exit 1; }
[[ -f "$FLAMEGRAPH_DIR/stackcollapse-perf.pl" && -f "$FLAMEGRAPH_DIR/flamegraph.pl" ]] \
  || { echo "FlameGraph scripts not found in $FLAMEGRAPH_DIR"; exit 1; }

# Run command array (safe for args with spaces)
RUN=( "$CMD" "${PROG_ARGS[@]}" )

# Perf presets (reasonable + comparison-friendly)
# Keep these pretty portable; WSL may not support every extra event.
STAT_EVENTS="task-clock,context-switches,cpu-migrations,page-faults,cycles,instructions,branches,branch-misses,cache-references,cache-misses"

run_stat() {
  local out_stat="$1"
  echo "== perf stat -> $out_stat =="
  echo "Running: taskset -c $CORE perf stat -r $REPS -e $STAT_EVENTS -- ${RUN[*]}"
  taskset -c "$CORE" perf stat --all-user --no-big-num \
    -o "$out_stat" -r "$REPS" -e "$STAT_EVENTS" \
    -- "${RUN[@]}"
}

run_record_and_flame() {
  local variant="$1"      # basic | dwarf
  local data="$2"         # perf data path
  local svg="$3"          # flamegraph svg

  # Prefer frame-pointer unwinding for "basic" (lower overhead, great if built with -fno-omit-frame-pointer)
  # DWARF for "dwarf" (works even without frame pointers; higher overhead)
  local callgraph="fp"
  if [[ "$variant" == "dwarf" ]]; then
    callgraph="dwarf"
  fi

  local rec=( perf record --all-user --clockid mono -F "$FREQ" -g --call-graph "$callgraph" -o "$data" )
  [[ -n "$PERF_DELAY" ]] && rec+=( --delay "$PERF_DELAY" )

  echo "== perf record ($variant, call-graph=$callgraph) -> $data =="
  echo "Running: taskset -c $CORE ${rec[*]} -- ${RUN[*]}"
  taskset -c "$CORE" "${rec[@]}" -- "${RUN[@]}"

  echo "== flamegraph ($variant) -> $svg =="
  if [[ "$variant" == "dwarf" ]]; then
    perf script -i "$data" --inline \
      | "$FLAMEGRAPH_DIR/stackcollapse-perf.pl" \
      | "$FLAMEGRAPH_DIR/flamegraph.pl" > "$svg"
  else
    perf script -i "$data" \
      | "$FLAMEGRAPH_DIR/stackcollapse-perf.pl" \
      | "$FLAMEGRAPH_DIR/flamegraph.pl" > "$svg"
  fi
}

# ---- BASIC ----
# if $DO_BASIC; then
#   run_stat "$RESULTS_DIR/perf_basic.stat"
#   run_record_and_flame "basic" \
#     "$RESULTS_DIR/perf_basic.data" \
#     "$RESULTS_DIR/flame_basic.svg"
# fi

# ---- DWARF + inline ----
if $DO_DWARF; then
  run_stat "$RESULTS_DIR/perf_dwarf.stat"
  run_record_and_flame "dwarf" \
    "$RESULTS_DIR/perf_dwarf.data" \
    "$RESULTS_DIR/flame_dwarf_inline.svg"
fi

echo "Done. Results in: $RESULTS_DIR"
