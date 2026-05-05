#!/usr/bin/env bash
# Run `index` crate HNSW SIFT1M Criterion benchmarks (`hnsw_sift1m`) over several env configurations.
#
# Usage:
#   export SIFT1M_BASE_PATH=/path/to/dir/with/sift_base.fvecs
#   ./scripts/run_hnsw_sift1m_benchmarks.sh
#
# Or pass the dataset directory as the first argument (sets SIFT1M_BASE_PATH for child processes):
#   ./scripts/run_hnsw_sift1m_benchmarks.sh /path/to/sift1m
#
# Optional environment (see also crates/index/benches/hnsw/sift1m.rs and common::try_load_sift_ctx):
#   HNSW_BENCH_PROFILE   — quick (default) | full  — controls how large the Cartesian sweep is
#   SIFT1M_HNSW_BENCH_SAMPLE_SIZE — repetitions per bench section; if unset, defaults come from profile:
#                                   quick uses HNSW_BENCH_QUICK_SAMPLE_SIZE (default 10), full uses
#                                   HNSW_BENCH_FULL_SAMPLE_SIZE (default 30). Values 1–9 enable Rust smoke mode
#                                   (Criterion disabled); 10+ use Criterion (requires sample_size >= 10).
#   EXTRA_CARGO_BENCH    — extra args for cargo bench (e.g. '-- --noplot'); word-split on spaces
#   FAIL_FAST            — if set to 1, stop on first cargo bench failure
#
# Swept variables (defaults below; edit arrays or use HNSW_BENCH_PROFILE=full):
#   SIFT1M_HNSW_EF_CONSTRUCTION
#   SIFT1M_LIMIT           — if set before invoking this script, used as the **only** corpus cap for every run
#                             (profile `LIMIT_VALUES` sweep is skipped). If unset, **quick** sweeps 10_000 and
#                             **full** sweeps several caps including 1_000_000. Plain `cargo bench` without
#                             `SIFT1M_LIMIT` uses the Rust bench default (1_000_000; see sift1m.rs).
#   SIFT1M_HNSW_EF         — search ef (loaded via SiftCtx)
#   SIFT1M_HNSW_BENCH_QUERIES — queries per timed search batch

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -n "${1:-}" ]]; then
  export SIFT1M_BASE_PATH="$1"
fi

if [[ -z "${SIFT1M_BASE_PATH:-}" ]]; then
  echo "error: set SIFT1M_BASE_PATH or pass the dataset directory as the first argument" >&2
  exit 1
fi

FAIL_FAST="${FAIL_FAST:-0}"
PROFILE="${HNSW_BENCH_PROFILE:-quick}"

# Criterion iterations (passed through to the bench via SIFT1M_HNSW_BENCH_SAMPLE_SIZE).
if [[ -z "${SIFT1M_HNSW_BENCH_SAMPLE_SIZE:-}" ]]; then
  case "$PROFILE" in
    full)
      export SIFT1M_HNSW_BENCH_SAMPLE_SIZE="${HNSW_BENCH_FULL_SAMPLE_SIZE:-30}"
      ;;
    *)
      export SIFT1M_HNSW_BENCH_SAMPLE_SIZE="${HNSW_BENCH_QUICK_SAMPLE_SIZE:-10}"
      ;;
  esac
fi
echo "profile=${PROFILE} SIFT1M_HNSW_BENCH_SAMPLE_SIZE=${SIFT1M_HNSW_BENCH_SAMPLE_SIZE}"

# Caller-set cap wins over the profile's LIMIT_VALUES sweep (avoids e.g. `SIFT1M_LIMIT=1000000 ./script.sh`
# being overwritten by quick's 10_000).
PRESET_SIFT1M_LIMIT="${SIFT1M_LIMIT:-}"

if [[ "$PROFILE" == "full" ]]; then
  EF_CONSTRUCTION_VALUES=(100 200 400)
  SEARCH_EF_VALUES=(64 100 200)
  BENCH_QUERIES_VALUES=(50 100)
else
  EF_CONSTRUCTION_VALUES=(200 400)
  SEARCH_EF_VALUES=(100)
  BENCH_QUERIES_VALUES=(100)
fi

if [[ -n "$PRESET_SIFT1M_LIMIT" ]]; then
  LIMIT_VALUES=("$PRESET_SIFT1M_LIMIT")
  echo "SIFT1M_LIMIT preset=${PRESET_SIFT1M_LIMIT} (single corpus size; profile sweep limits ignored)"
else
  if [[ "$PROFILE" == "full" ]]; then
    LIMIT_VALUES=(5000 10000 50000 1000000)
  else
    LIMIT_VALUES=(10000)
  fi
fi

run_one() {
  local tag="$1"
  shift
  echo ""
  echo "================================================================================"
  echo "run: $tag"
  echo "  env: $*"
  echo "================================================================================"
  # shellcheck disable=SC2086
  env "$@" cargo bench -p index --bench hnsw_sift1m ${EXTRA_CARGO_BENCH:-}
}

failed=0
for ef_c in "${EF_CONSTRUCTION_VALUES[@]}"; do
  for lim in "${LIMIT_VALUES[@]}"; do
    for ef_search in "${SEARCH_EF_VALUES[@]}"; do
      for n_bench_q in "${BENCH_QUERIES_VALUES[@]}"; do
        tag="efc${ef_c}_limit${lim}_ef${ef_search}_bq${n_bench_q}"
        if ! run_one "$tag" \
          SIFT1M_HNSW_EF_CONSTRUCTION="$ef_c" \
          SIFT1M_LIMIT="$lim" \
          SIFT1M_HNSW_EF="$ef_search" \
          SIFT1M_HNSW_BENCH_QUERIES="$n_bench_q"; then
          failed=1
          if [[ "$FAIL_FAST" == "1" ]]; then
            echo "error: benchmark run failed (-- FAIL_FAST=1)" >&2
            exit 1
          fi
        fi
      done
    done
  done
done

if [[ "$failed" -ne 0 ]]; then
  echo "warning: one or more cargo bench runs failed" >&2
  exit 1
fi

echo ""
echo "All benchmark configurations finished successfully."
