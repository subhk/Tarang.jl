#!/usr/bin/env bash

# Behavioral regression tests for gpu_ci_report.sh's commit-attestation
# preflight. The fake Julia executable makes a successful run deterministic and
# records whether the GPU suite was reached; unsafe checkouts must be rejected
# before that point.

set -euo pipefail

SOURCE_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/gpu_ci_report.sh"
TMP_ROOT="$(mktemp -d -t tarang-gpu-ci-report-test.XXXXXX)"
trap 'rm -rf "$TMP_ROOT"' EXIT

REPO="$TMP_ROOT/repo"
FAKE_BIN="$TMP_ROOT/bin"
JULIA_CALLS="$TMP_ROOT/julia-calls"
GH_CALLS="$TMP_ROOT/gh-calls"
FAKE_CUDA_RC=0
FAKE_SUITE_RC=0
FAKE_MUTATE_TRACKED=0
FAKE_MUTATE_HEAD=0
GH_API_RC=0
mkdir -p "$REPO/scripts" "$FAKE_BIN"
cp "$SOURCE_SCRIPT" "$REPO/scripts/gpu_ci_report.sh"

printf '%s\n' '#!/usr/bin/env bash' \
    'printf "called\n" >> "${JULIA_CALLS:?}"' \
    'call_number="$(wc -l < "${JULIA_CALLS:?}" | tr -d " ")"' \
    'if [ "$call_number" -eq 1 ]; then' \
    '  printf "%s\n" "fake CUDA device"' \
    '  exit "${FAKE_CUDA_RC:-0}"' \
    'fi' \
    'if [ "${FAKE_MUTATE_TRACKED:-0}" -eq 1 ]; then' \
    '  printf "%s\n" "changed during suite" >> "${FAKE_REPO:?}/tracked.txt"' \
    'fi' \
    'if [ "${FAKE_MUTATE_HEAD:-0}" -eq 1 ]; then' \
    '  git -C "${FAKE_REPO:?}" commit -q --allow-empty -m "changed during suite"' \
    'fi' \
    'printf "%s\n" "GPU summary: 1 passed, 0 failed"' \
    'exit "${FAKE_SUITE_RC:-0}"' > "$FAKE_BIN/julia"
chmod +x "$FAKE_BIN/julia" "$REPO/scripts/gpu_ci_report.sh"

printf '%s\n' '#!/usr/bin/env bash' \
    'case "${1:-}" in' \
    '  auth) exit 0 ;;' \
    '  api)' \
    '    printf "%s\n" "$*" >> "${GH_CALLS:?}"' \
    '    exit "${GH_API_RC:-0}"' \
    '    ;;' \
    '  *) exit 0 ;;' \
    'esac' > "$FAKE_BIN/gh"
chmod +x "$FAKE_BIN/gh"

git -C "$REPO" init -q
git -C "$REPO" config user.name "GPU CI test"
git -C "$REPO" config user.email "gpu-ci-test@example.invalid"
printf '%s\n' "tracked" > "$REPO/tracked.txt"
git -C "$REPO" add scripts/gpu_ci_report.sh tracked.txt
git -C "$REPO" commit -q -m "initial"
git -C "$REPO" commit -q --allow-empty -m "second"

run_report() {
    local output_file="$1"
    shift
    set +e
    (
        cd "$REPO"
        PATH="$FAKE_BIN:$PATH" TMPDIR="$TMP_ROOT" \
            JULIA_CALLS="$JULIA_CALLS" GH_CALLS="$GH_CALLS" \
            FAKE_REPO="$REPO" FAKE_CUDA_RC="$FAKE_CUDA_RC" \
            FAKE_SUITE_RC="$FAKE_SUITE_RC" \
            FAKE_MUTATE_TRACKED="$FAKE_MUTATE_TRACKED" \
            FAKE_MUTATE_HEAD="$FAKE_MUTATE_HEAD" GH_API_RC="$GH_API_RC" \
            ./scripts/gpu_ci_report.sh --repo example/project "$@"
    ) >"$output_file" 2>&1
    local rc=$?
    set -e
    return "$rc"
}

assert_rejected_before_julia() {
    local label="$1" expected="$2"
    shift 2
    local output_file="$TMP_ROOT/${label}.log"
    rm -f "$JULIA_CALLS"
    if run_report "$output_file" "$@"; then
        printf 'FAIL: %s checkout was accepted\n' "$label" >&2
        sed -n '1,160p' "$output_file" >&2
        exit 1
    fi
    if ! grep -F "$expected" "$output_file" >/dev/null; then
        printf 'FAIL: %s rejection did not explain the invariant\n' "$label" >&2
        sed -n '1,160p' "$output_file" >&2
        exit 1
    fi
    if [ -e "$JULIA_CALLS" ]; then
        printf 'FAIL: %s checkout reached Julia before rejection\n' "$label" >&2
        exit 1
    fi
}

assert_rejected_before_julia "mismatched-sha" "must match checkout HEAD" --no-status --sha HEAD^

printf '%s\n' "dirty" >> "$REPO/tracked.txt"
assert_rejected_before_julia "dirty-tracked" "working tree must be clean" --no-status
git -C "$REPO" restore tracked.txt

printf '%s\n' "untracked" > "$REPO/untracked.txt"
assert_rejected_before_julia "dirty-untracked" "working tree must be clean" --no-status
rm -f "$REPO/untracked.txt"

rm -f "$JULIA_CALLS"
MATCHING_LOG="$TMP_ROOT/matching.log"
if ! run_report "$MATCHING_LOG" --no-status --sha HEAD; then
    printf 'FAIL: clean matching checkout was rejected\n' >&2
    sed -n '1,160p' "$MATCHING_LOG" >&2
    exit 1
fi
[ "$(wc -l < "$JULIA_CALLS" | tr -d ' ')" -eq 2 ] || {
    printf 'FAIL: clean matching checkout did not run CUDA gate and suite\n' >&2
    exit 1
}

assert_changed_source_rejected() {
    local label="$1"
    local output_file="$TMP_ROOT/${label}.log"
    rm -f "$JULIA_CALLS" "$GH_CALLS"
    if run_report "$output_file" --sha HEAD; then
        printf 'FAIL: %s source mutation was reported as a valid result\n' "$label" >&2
        sed -n '1,160p' "$output_file" >&2
        exit 1
    fi
    grep -F "source changed during GPU suite" "$output_file" >/dev/null || {
        printf 'FAIL: %s source mutation was not diagnosed\n' "$label" >&2
        sed -n '1,160p' "$output_file" >&2
        exit 1
    }
    if grep -F "state=success" "$GH_CALLS" >/dev/null; then
        printf 'FAIL: %s source mutation posted a success status\n' "$label" >&2
        exit 1
    fi
    grep -F "state=error" "$GH_CALLS" >/dev/null || {
        printf 'FAIL: %s source mutation did not close the pending status as error\n' "$label" >&2
        exit 1
    }
}

FAKE_MUTATE_TRACKED=1
assert_changed_source_rejected "dirty-during-suite"
FAKE_MUTATE_TRACKED=0
git -C "$REPO" restore tracked.txt

FAKE_MUTATE_HEAD=1
assert_changed_source_rejected "head-changed-during-suite"
FAKE_MUTATE_HEAD=0

FAKE_CUDA_RC=3
rm -f "$JULIA_CALLS" "$GH_CALLS"
CUDA_ERROR_LOG="$TMP_ROOT/cuda-error.log"
if run_report "$CUDA_ERROR_LOG" --sha HEAD; then
    printf 'FAIL: CUDA-gate error returned success\n' >&2
    exit 1
fi
[ "$(wc -l < "$JULIA_CALLS" | tr -d ' ')" -eq 1 ] || {
    printf 'FAIL: CUDA-gate error still reached the GPU suite\n' >&2
    exit 1
}
grep -F "state=error" "$GH_CALLS" >/dev/null || {
    printf 'FAIL: CUDA-gate error did not post an error status\n' >&2
    exit 1
}
FAKE_CUDA_RC=0

FAKE_SUITE_RC=7
rm -f "$JULIA_CALLS" "$GH_CALLS"
SUITE_FAILURE_LOG="$TMP_ROOT/suite-failure.log"
if run_report "$SUITE_FAILURE_LOG" --sha HEAD; then
    printf 'FAIL: failing GPU suite returned success\n' >&2
    exit 1
fi
grep -F "state=failure" "$GH_CALLS" >/dev/null || {
    printf 'FAIL: failing GPU suite did not post a failure status\n' >&2
    exit 1
}
if grep -F "state=success" "$GH_CALLS" >/dev/null; then
    printf 'FAIL: failing GPU suite posted a success status\n' >&2
    exit 1
fi
FAKE_SUITE_RC=0

GH_API_RC=42
STATUS_LOG="$TMP_ROOT/status-failure.log"
if run_report "$STATUS_LOG" --sha HEAD; then
    printf 'FAIL: GitHub status-post failure did not make the report fail\n' >&2
    sed -n '1,160p' "$STATUS_LOG" >&2
    exit 1
fi
grep -F "failed to post" "$STATUS_LOG" >/dev/null || {
    printf 'FAIL: GitHub status-post failure was not diagnosed\n' >&2
    sed -n '1,160p' "$STATUS_LOG" >&2
    exit 1
}
GH_API_RC=0

printf '%s\n' "gpu_ci_report.sh attestation tests passed"
