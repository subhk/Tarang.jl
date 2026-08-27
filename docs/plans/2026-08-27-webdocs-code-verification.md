# Webdocs Code Verification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Verify every code fence published from `docs/src`, repair stale runnable examples, and keep a regression check that catches future syntax and API drift.

**Architecture:** Add a lightweight default test that discovers Markdown files and parses every Julia, Bash, and TOML fence (plus basic Dockerfile structure), with opt-in runtime modes that execute self-contained CPU examples in isolated Julia processes and standalone MPI examples under two ranks. Treat CUDA, package-installation, optional-output, expected-error, and intentionally contextual blocks as environment-specific or illustrative; require a reason for every excluded self-contained example, and continue to validate its Julia syntax.

**Tech Stack:** Julia 1.10+, Tarang.jl, Documenter.jl, Markdown code fences, Julia `Test`.

### Task 1: Establish the documentation-code baseline

**Files:**
- Verify: `docs/src/**/*.md`
- Verify: `docs/make.jl`

**Step 1:** Count and parse every `julia` fence under `docs/src`.

**Step 2:** Build the full site with `julia --project=docs docs/make.jl` and confirm the existing build result before edits.

**Step 3:** Inventory self-contained examples (blocks importing Tarang) and classify CPU, MPI, CUDA, package-management, expected-error, and schematic examples.

### Task 2: Add a failing webdocs regression test

**Files:**
- Create: `test/test_webdocs_code.jl`
- Modify: `test/file_lists.jl`

**Step 1:** Add tests that discover all Markdown sources, parse each Julia, Bash, and TOML fence, and sanity-check Dockerfile fences.

**Step 2:** Add an opt-in runtime sweep that executes eligible CPU examples in isolated Julia processes and reports the source file and fence line on failure.

**Step 3:** Run the runtime sweep against the unmodified docs and verify it fails on current stale examples for the expected runtime reasons.

### Task 3: Repair verified failures

**Files:**
- Modify: only the `docs/src/**/*.md` files reported by `test/test_webdocs_code.jl`

**Step 1:** For each failure, reproduce it independently and compare the example with the current public API, shipped examples, and focused tests.

**Step 2:** Update the smallest stale snippet or explicitly classify a hardware-dependent, expected-error, or schematic block without weakening syntax validation.

**Step 3:** Re-run the focused example after every correction and preserve unrelated staged or unstaged edits.

### Task 4: Verify the completed audit

**Files:**
- Verify: `test/test_webdocs_code.jl`
- Verify: `docs/src/**/*.md`
- Verify: `docs/make.jl`

**Step 1:** Run the default webdocs test and the opt-in CPU runtime sweep.

**Step 2:** Run standalone MPI examples under two ranks and perform syntax/source-API checks for CUDA examples when CUDA hardware is unavailable.

**Step 3:** Build the full Documenter site with doctests enabled.

**Step 4:** Run `git diff --check`, inspect the complete diff, and confirm no unrelated worktree changes were overwritten.
