#!/bin/bash
#
# MPI Test Runner for Tarang.jl
#
# Usage:
#   ./test/run_mpi_tests.sh           # Run with default 4 processes
#   ./test/run_mpi_tests.sh 2         # Run with 2 processes
#   ./test/run_mpi_tests.sh 4 --gpu   # Run with 4 processes including GPU tests
#

set -e

# Default number of processes
NPROCS=${1:-4}

# Check for GPU flag
GPU_FLAG=""
if [[ "$2" == "--gpu" ]] || [[ "$1" == "--gpu" ]]; then
    GPU_FLAG="--gpu"
    if [[ "$1" == "--gpu" ]]; then
        NPROCS=4
    fi
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}  Tarang.jl MPI Test Suite${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""
echo -e "Project directory: ${PROJECT_DIR}"
echo -e "Number of MPI processes: ${YELLOW}${NPROCS}${NC}"
echo ""

# Check for mpiexec
if ! command -v mpiexec &> /dev/null; then
    echo -e "${RED}Error: mpiexec not found. Please install MPI.${NC}"
    echo "  On macOS: brew install open-mpi"
    echo "  On Ubuntu: sudo apt-get install openmpi-bin libopenmpi-dev"
    exit 1
fi

# Check Julia
if ! command -v julia &> /dev/null; then
    echo -e "${RED}Error: julia not found.${NC}"
    exit 1
fi

echo -e "${BLUE}MPI implementation:${NC}"
mpiexec --version | head -1
echo ""

# Function to run a test file
run_test() {
    local test_file=$1
    local test_name=$(basename "$test_file" .jl)

    echo -e "${YELLOW}Running: ${test_name}${NC}"
    echo -e "Command: mpiexec -n ${NPROCS} julia --project=${PROJECT_DIR} ${test_file}"
    echo ""

    if mpiexec -n ${NPROCS} julia --project="${PROJECT_DIR}" "${test_file}"; then
        echo -e "${GREEN}✓ ${test_name} passed${NC}"
        echo ""
        return 0
    else
        echo -e "${RED}✗ ${test_name} failed${NC}"
        echo ""
        return 1
    fi
}

# Track results
PASSED=0
FAILED=0
FAILED_TESTS=""

# List of MPI test files
MPI_TESTS=(
    "${SCRIPT_DIR}/test_mpi_distributor.jl"
    "${SCRIPT_DIR}/test_distributed_gpu_transpose.jl"
)

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}  Running MPI Tests${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

for test_file in "${MPI_TESTS[@]}"; do
    if [[ -f "$test_file" ]]; then
        if run_test "$test_file"; then
            ((PASSED++))
        else
            ((FAILED++))
            FAILED_TESTS="${FAILED_TESTS}\n  - $(basename "$test_file")"
        fi
    else
        echo -e "${YELLOW}Warning: Test file not found: ${test_file}${NC}"
    fi
done

# Summary
echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}  Test Summary${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""
echo -e "Total tests: $((PASSED + FAILED))"
echo -e "${GREEN}Passed: ${PASSED}${NC}"
echo -e "${RED}Failed: ${FAILED}${NC}"

if [[ $FAILED -gt 0 ]]; then
    echo -e "\n${RED}Failed tests:${FAILED_TESTS}${NC}"
    exit 1
else
    echo -e "\n${GREEN}All MPI tests passed!${NC}"
    exit 0
fi
