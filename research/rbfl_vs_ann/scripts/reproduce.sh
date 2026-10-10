#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BFL_REPO="${BFL_REPO:-$(cd "$PROJECT/../.." && pwd)}"
RESULT_DIR="${RESULT_DIR:-$PROJECT/results}"
BUILD_DIR="${BUILD_DIR:-$PROJECT/build}"
CXX_BIN="${CXX:-g++}"
COMMIT=53c2f46
SEEDS="${1:-5}"
HISTORICAL_SEEDS="${2:-20}"

if [[ ! "$SEEDS" =~ ^[1-9][0-9]*$ || ! "$HISTORICAL_SEEDS" =~ ^[1-9][0-9]*$ ]]; then
    printf 'Uso: bash scripts/reproduce.sh [semillas >= 1] [semillas_históricas >= 20]\n' >&2
    exit 2
fi
if (( HISTORICAL_SEEDS < 20 )); then
    printf 'El ejecutable histórico requiere al menos 20 semillas.\n' >&2
    exit 2
fi
git -C "$BFL_REPO" cat-file -e "${COMMIT}^{commit}"

mkdir -p "$BUILD_DIR" "$RESULT_DIR"
SNAPSHOT="$BUILD_DIR/bfl-$COMMIT"
mkdir -p "$SNAPSHOT"
git -C "$BFL_REPO" archive "$COMMIT" | tar -x -C "$SNAPSHOT"
FLAGS=(-std=c++17 -O3 -march=native -Wall -Wextra -Wpedantic)

"$CXX_BIN" "${FLAGS[@]}" -I"$PROJECT/src" -I"$SNAPSHOT/include" \
    "$PROJECT/src/benchmark.cpp" "$PROJECT/src/rbfl.cpp" \
    "$PROJECT/src/greedy.cpp" "$PROJECT/src/ann_baseline.cpp" \
    "$SNAPSHOT/src/routed_binary.cpp" -o "$BUILD_DIR/rbfl_benchmark"
"$CXX_BIN" "${FLAGS[@]}" -I"$PROJECT/src" \
    "$PROJECT/src/test_rbfl.cpp" "$PROJECT/src/rbfl.cpp" \
    "$PROJECT/src/greedy.cpp" -o "$BUILD_DIR/rbfl_tests"
"$BUILD_DIR/rbfl_tests"
"$CXX_BIN" "${FLAGS[@]}" -I"$SNAPSHOT/include" \
    "$SNAPSHOT/src/bfl.cpp" "$SNAPSHOT/src/greedy.cpp" \
    "$SNAPSHOT/src/routed_binary.cpp" \
    "$SNAPSHOT/benchmarks/benchmark_bfl_vs_ann.cpp" \
    -o "$BUILD_DIR/historical_benchmark"

# Stage all output until both benchmark programs and the summarizer succeed.
STAGED="$BUILD_DIR/staged-results"
mkdir -p "$STAGED"
"$BUILD_DIR/historical_benchmark" "$HISTORICAL_SEEDS" > "$STAGED/historical_reproduction.csv"
"$BUILD_DIR/rbfl_benchmark" "$SEEDS" "$STAGED/training_trace.csv" > "$STAGED/benchmark.csv"
python3 "$PROJECT/scripts/summarize.py" "$STAGED"
{
    date -u '+UTC: %Y-%m-%d %H:%M:%S'
    "$CXX_BIN" --version | sed -n '1p'
    printf 'Flags: %s\n' "${FLAGS[*]}"
    if command -v lscpu >/dev/null 2>&1; then
        lscpu | sed -n 's/^Model name:[[:space:]]*//p'
    fi
    printf 'BFL commit: %s\nRBFL seeds: 1..%s\nHistorical seeds: 1..%s\n' \
        "$COMMIT" "$SEEDS" "$HISTORICAL_SEEDS"
    printf 'Hardware instructions: unavailable unless perf counters are accessible\n'
} > "$STAGED/environment.txt"

cp "$STAGED"/*.csv "$STAGED/environment.txt" "$RESULT_DIR/"
printf 'Resultados escritos en %s\n' "$RESULT_DIR"
