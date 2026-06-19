# Webdocs Codebase Structure Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the contributor-facing architecture page accurately describe Tarang.jl's current package layers and stable source entry points.

**Architecture:** Replace the volatile near-exhaustive source tree with a high-level map derived from `src/Tarang.jl`, `src/load_order.jl`, and the domain loader files. Retain the existing runtime explanation while adding concise guidance for bootstrap, public API, core, tools, extras, CUDA extension, tests, and docs.

**Tech Stack:** Documenter.jl, Markdown, Julia package source manifests.

### Task 1: Update and verify the architecture map

**Files:**
- Modify: `docs/src/pages/architecture.md`
- Create: `docs/plans/2026-06-19-webdocs-codebase-structure.md`

**Step 1: Verify the old page lacks required stable entry points**

Run an assertion for `src/load_order.jl`, `src/public_api.jl`, `src/api/`, `src/tools/load_runtime.jl`, `src/extras/load_extras.jl`, and `ext/TarangCUDAExt.jl`.

Expected: FAIL because the current structure map omits these boundaries.

**Step 2: Replace the package tree and contributor map**

Document the package bootstrap sequence, layer responsibilities, stable loader files, and the current focused implementation directories. Avoid listing every implementation file.

**Step 3: Verify documented paths**

Extract repository paths from the updated package-structure section and assert that each exists.

Expected: PASS with no missing paths.

**Step 4: Build the web documentation**

Run: `julia --project=docs docs/make.jl`

Expected: Documenter completes without an error.

**Step 5: Review the diff**

Run: `git diff --check` and inspect the architecture-page diff for accuracy and scope.
