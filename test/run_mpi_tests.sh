#!/bin/bash
#
# MPI Test Runner for Tarang.jl
#
# Usage:
#   ./test/run_mpi_tests.sh           # Run with default 4 processes
#   ./test/run_mpi_tests.sh 2         # Run with 2 processes
#   ./test/run_mpi_tests.sh 4 --gpu   # Run with 4 processes including GPU tests
#

set -euo pipefail

NPROCS=4
RUN_GPU=false
for arg in "$@"; do
    case "$arg" in
        --gpu)
            RUN_GPU=true
            ;;
        *[!0-9]*|'')
            echo "Usage: $0 [nprocs] [--gpu]" >&2
            exit 2
            ;;
        *)
            NPROCS=$arg
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
JULIA_BIN=${JULIA:-julia}

if ! command -v "$JULIA_BIN" >/dev/null 2>&1; then
    echo "Error: Julia executable not found: $JULIA_BIN" >&2
    exit 1
fi

# `run_mpi_ci.jl` consumes test/file_lists.jl, the same registry CI and the
# inventory test use.  Keep this shell file as a convenience entry point only.
TARANG_MPI_FILESET=mpi "$JULIA_BIN" --project="$PROJECT_DIR" \
    "$SCRIPT_DIR/run_mpi_ci.jl" "$NPROCS"

if [[ "$RUN_GPU" == true ]]; then
    TARANG_MPI_FILESET=distributed_gpu "$JULIA_BIN" --project="$PROJECT_DIR" \
        "$SCRIPT_DIR/run_mpi_ci.jl" "$NPROCS"
fi
