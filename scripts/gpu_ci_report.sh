#!/usr/bin/env bash
#
# Run Tarang's single-GPU test suite on a machine that has an NVIDIA GPU, then
# report the result back to GitHub as a commit status.
#
# GitHub-hosted runners have no GPU, so nothing in GitHub Actions ever executes
# test/run_gpu_ci.jl. This script is the manual substitute: run it on the GPU
# server and the commit gets a green (or red) check on github.com, visible in the
# commit list and on any PR containing that commit.
#
#   ./scripts/gpu_ci_report.sh                    # test HEAD, post status
#   ./scripts/gpu_ci_report.sh --sha 6a4da42      # require HEAD to be this commit
#   ./scripts/gpu_ci_report.sh --no-status        # run only, post nothing
#   ./scripts/gpu_ci_report.sh --gist             # also upload the log (see below)
#
# Requires: julia, git, and gh authenticated with a token carrying `repo:status`
# (the plain `repo` scope includes it). --gist additionally needs `gist`.
#
# WHY THE CUDA GATE EXISTS
# Every file in GPU_TEST_FILES self-guards with CUDA.functional() and exits 0
# when no GPU is present. That is correct for CI, but it means running this suite
# on a CUDA-less machine produces a *vacuous pass* — every test skipped, exit
# code 0, and a green status posted for a GPU suite that never touched a GPU.
# This script therefore refuses to run at all unless CUDA is genuinely functional.
# A missing GPU is reported as an `error` status, never as `success`.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

SHA=""
CONTEXT="gpu/cuda"
REPO=""
POST_STATUS=1
POST_PENDING=1
MAKE_GIST=0

usage() {
    sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'USAGE'

Options:
  --sha SHA        Commit to test and stamp; must match HEAD (default: HEAD)
  --context NAME   Status context shown on GitHub (default: gpu/cuda)
  --repo OWNER/REP Target repository (default: derived from the origin remote)
  --gist           Upload the run log as a SECRET gist and link it from the
                   status. This sends the log to GitHub's servers; off by default.
  --no-status      Run the suite but post nothing
  --skip-pending   Do not post the in-progress "pending" status
  -h, --help       This message
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --sha)         SHA="${2:?--sha needs a value}"; shift 2 ;;
        --context)     CONTEXT="${2:?--context needs a value}"; shift 2 ;;
        --repo)        REPO="${2:?--repo needs a value}"; shift 2 ;;
        --gist)        MAKE_GIST=1; shift ;;
        --no-status)   POST_STATUS=0; shift ;;
        --skip-pending) POST_PENDING=0; shift ;;
        -h|--help)     usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 64 ;;
    esac
done

die() { echo "error: $*" >&2; exit 1; }

attest_checkout() {
    local current_head worktree_status
    current_head="$(git rev-parse 'HEAD^{commit}')" || {
        echo "error: cannot resolve checkout HEAD" >&2
        return 1
    }
    if [ "$SHA" != "$current_head" ]; then
        echo "error: requested commit $SHA must match checkout HEAD $current_head" >&2
        return 1
    fi

    worktree_status="$(git status --porcelain --untracked-files=all)" || {
        echo "error: cannot inspect the working tree" >&2
        return 1
    }
    if [ -n "$worktree_status" ]; then
        echo "error: working tree must be clean, including untracked files" >&2
        printf '%s\n' "$worktree_status" >&2
        return 1
    fi
}

# ---------------------------------------------------------------- preflight ---

command -v git >/dev/null   || die "git not found"
command -v julia >/dev/null || die "julia not found on PATH"
if [ "$POST_STATUS" -eq 1 ] || [ "$MAKE_GIST" -eq 1 ]; then
    command -v gh >/dev/null || die "gh not found (install the GitHub CLI, or pass --no-status)"
    gh auth status >/dev/null 2>&1 || die "gh is not authenticated — run: gh auth login"
fi

HEAD_SHA="$(git rev-parse 'HEAD^{commit}')" || die "cannot resolve HEAD"
if [ -n "$SHA" ]; then
    SHA="$(git rev-parse "${SHA}^{commit}")" || die "not a valid commit: $SHA"
else
    SHA="$HEAD_SHA"
fi

attest_checkout || exit 1

if [ -z "$REPO" ]; then
    REPO="$(git config --get remote.origin.url \
            | sed -E 's#^git@github\.com:#-#; s#^https://github\.com/#-#; s#\.git$##; s#^-##')"
    [ -n "$REPO" ] || die "cannot derive the repository from remote.origin.url — pass --repo OWNER/REPO"
fi

echo "repo    : $REPO"
echo "commit  : $SHA"
echo "context : $CONTEXT"

# --------------------------------------------------------------- CUDA gate ---

echo
echo "checking that CUDA is actually functional..."
GPU_INFO="$(julia --project=. -e '
    try
        @eval using CUDA
    catch
        println("CUDA.jl is not available in the stacked Julia environments.")
        println("Install it without modifying this checkout:")
        println("  julia --project=@v#.# -e \"using Pkg; Pkg.add(\\\"CUDA\\\")\"")
        exit(2)
    end
    if !CUDA.functional()
        println("CUDA.jl loaded but CUDA.functional() is false (no driver / no device)")
        exit(3)
    end
    name = try
        CUDA.name(CUDA.device())
    catch
        "unknown device"
    end
    println(name, " (", CUDA.ndevices(), " device(s))")
' 2>&1)"
GPU_RC=$?

post_status() {  # state, description, [target_url]
    [ "$POST_STATUS" -eq 1 ] || return 0
    local state="$1" desc="$2" url="${3:-}"
    desc="$(printf '%.140s' "$desc")"   # GitHub truncates descriptions past 140 chars
    local args=(-X POST "repos/$REPO/statuses/$SHA"
                -f "state=$state" -f "context=$CONTEXT" -f "description=$desc")
    [ -n "$url" ] && args+=(-f "target_url=$url")
    if gh api "${args[@]}" >/dev/null 2>&1; then
        echo "posted $state status to $REPO@${SHA:0:7} ($CONTEXT)"
    else
        echo "warning: failed to post the $state status (token needs repo:status)" >&2
        return 1
    fi
}

if [ "$GPU_RC" -ne 0 ]; then
    echo "$GPU_INFO" >&2
    echo >&2
    echo "refusing to run: the GPU tests skip themselves without a working GPU," >&2
    echo "so running them here would report a pass that proves nothing." >&2
    post_status "error" "no functional CUDA device on $(hostname -s) — suite not run" || true
    exit 1
fi
echo "  $GPU_INFO"

# ------------------------------------------------------------------- run it ---

LOG="$(mktemp -t tarang-gpu-ci.XXXXXX)"
KEEP_LOG=0
STATUS_POST_FAILED=0
trap '[ "$KEEP_LOG" -eq 1 ] || rm -f "$LOG"' EXIT

if [ "$POST_PENDING" -eq 1 ]; then
    post_status "pending" "running run_gpu_ci.jl on $(hostname -s)" || STATUS_POST_FAILED=1
fi

echo
echo "running test/run_gpu_ci.jl ..."
echo
JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-2}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" \
    julia --project=. test/run_gpu_ci.jl 2>&1 | tee "$LOG"
RC="${PIPESTATUS[0]}"

# run_gpu_ci.jl prints e.g. "  GPU summary: 6 passed, 0 failed"
SUMMARY="$(grep -o 'GPU summary: .*' "$LOG" | tail -1)"
[ -n "$SUMMARY" ] || SUMMARY="run_gpu_ci.jl exited $RC with no summary line"

TARGET_URL=""
if [ "$MAKE_GIST" -eq 1 ]; then
    echo
    echo "uploading log as a secret gist..."
    TARGET_URL="$(gh gist create "$LOG" \
        --desc "Tarang GPU CI — $REPO@${SHA:0:7} on $(hostname -s)" 2>/dev/null | tail -1)"
    [ -n "$TARGET_URL" ] && echo "  $TARGET_URL"
fi

# The suite (and optional log upload) may take hours. Re-check the attested
# source immediately before publishing the final result so a checkout,
# generator, or editor change during the run can never be reported as the
# result of the original commit.
if ! attest_checkout; then
    KEEP_LOG=1
    echo "error: source changed during GPU suite; refusing to report its result" >&2
    post_status "error" "source changed during GPU suite on $(hostname -s) — result discarded" || \
        STATUS_POST_FAILED=1
    echo "full log kept at: $LOG"
    exit 1
fi

echo
if [ "$RC" -eq 0 ]; then
    post_status "success" "$SUMMARY on $(hostname -s)" "$TARGET_URL" || STATUS_POST_FAILED=1
else
    KEEP_LOG=1
    post_status "failure" "$SUMMARY on $(hostname -s)" "$TARGET_URL" || STATUS_POST_FAILED=1
    echo "full log kept at: $LOG"
fi

[ "$RC" -ne 0 ] && exit "$RC"
exit "$STATUS_POST_FAILED"
