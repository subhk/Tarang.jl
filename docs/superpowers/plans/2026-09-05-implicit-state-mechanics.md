# Implicit-State Mechanics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the three implicit-state classes (layout, collectiveness, ownership) that produce this package's recurring silent-wrong and deadlock bugs, without changing any numerical result.

**Architecture:** Every change is either a mechanical fold into an existing accessor, a refusal where a hidden collective used to run, a transfer of ownership to the Distributor, or a pure file move. Each task ends with the serial suite green; the branch ends with MPI at 1, 2 and 4 ranks green.

**Tech Stack:** Julia 1.10–1.12, MPI.jl (MPICH_jll locally), PencilArrays/PencilFFTs, KernelAbstractions CPU backend for GPU-kernel tests.

**Spec:** `docs/superpowers/specs/2026-09-05-implicit-state-mechanics-design.md`

## Global Constraints

- Julia binary on this machine: `~/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia --project=.` (the `julia` launcher is broken).
- Full serial suite is `Pkg.test()`; MPI suite is `test/run_mpi_ci.jl <nprocs>` (launch through Julia so MPI.jl sets `DYLD_FALLBACK_LIBRARY_PATH`; a bare `mpiexec` under `nohup`/`sh` loses it).
- Never `git checkout`/`reset` tracked files while a test run is in progress: each MPI file spawns a fresh `julia` that recompiles from the current tree.
- Every new test file must be registered in `test/file_lists.jl` and `git add`ed (Task 1 makes the inventory test enforce the second half).
- Ratchet constants (`LAYOUT_RATCHET`, `SILENT_RATCHET`, JET count) may only go down.
- Commit messages end with the `Co-Authored-By` / `Claude-Session` trailers used on this branch.

---

### Task 1: Inventory test requires registered test files to be git-tracked

**Files:**
- Modify: `test/test_test_inventory.jl`

**Interfaces:**
- Consumes: `TEST_FILES`, `OPTIONAL_TEST_FILES`, `GPU_TEST_FILES`, `MPI_TEST_FILES`, `DISTRIBUTED_GPU_TEST_FILES` (globals from `test/file_lists.jl`, already loaded by `runtests.jl`).
- Produces: nothing new.

- [ ] **Step 1: Add the failing assertion**

Append inside the existing `@testset "Test file inventory"` block, after the `run_mpi_tests.sh` checks:

```julia
    # A registered file that exists on disk but is not tracked by git passes
    # locally and fails on every clean clone. It has happened three times.
    tracked = try
        Set(basename.(filter(!isempty, split(read(
            setenv(`git ls-files -- test/`; dir=joinpath(@__DIR__, "..")),
            String), '\n'))))
    catch err
        @info "git unavailable; skipping tracked-file check" exception = err
        nothing
    end
    if tracked !== nothing
        untracked = sort!(collect(setdiff(known_test_files, tracked)))
        isempty(untracked) || @error "registered test files not tracked by git" untracked
        @test isempty(untracked)
    end
```

- [ ] **Step 2: Verify it fails on an untracked registration**

Run:
```bash
touch test/test_zzz_untracked_probe.jl
sed -i '' 's|    "test_spectra.jl",|    "test_zzz_untracked_probe.jl",\n    "test_spectra.jl",|' test/file_lists.jl
~/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test; include("test/file_lists.jl"); include("test/test_test_inventory.jl")'
```
Expected: one failure, `untracked = ["test_zzz_untracked_probe.jl"]`.

- [ ] **Step 3: Remove the probe and verify it passes**

```bash
git checkout -- test/file_lists.jl && rm test/test_zzz_untracked_probe.jl
~/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test; include("test/file_lists.jl"); include("test/test_test_inventory.jl")'
```
Expected: `Test file inventory | 5 5`.

- [ ] **Step 4: Commit**

```bash
git add test/test_test_inventory.jl
git commit -m "test: inventory requires registered test files to be git-tracked"
```

---

### Task 2: An accessor never plans — `_field_transform_bundle` refuses instead of building

**Files:**
- Modify: `src/core/transforms/transform_types.jl:375-388`
- Modify: `src/core/module_contracts.jl` (new section after "BUFFER OWNERSHIP")
- Test: `test/test_field_typestability.jl` (append a testset)

**Interfaces:**
- Consumes: `ScalarField.transform_bundle::Any` (set by both inner constructors in `src/core/field/field_types.jl:138,163`).
- Produces: `_field_transform_bundle(field)` now throws `ArgumentError` when `field.transform_bundle` is not a `TransformPlanBundle`; it never calls `transform_plan_bundle`.

- [ ] **Step 1: Write the failing test**

Append to `test/test_field_typestability.jl`:

```julia
@testset "_field_transform_bundle never plans collectively" begin
    coords = CartesianCoordinates("x")
    dist = Distributor(coords; dtype=Float64)
    xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
    f = ScalarField(dist, "f", (xb,), Float64)
    @test Tarang._field_transform_bundle(f) isa Tarang.TransformPlanBundle

    # Detach the bundle and clear the Distributor's cache: the accessor must
    # refuse rather than rebuild (plan construction is collective under MPI).
    f.transform_bundle = nothing
    empty!(dist.transform_plan_cache)
    @test_throws ArgumentError Tarang._field_transform_bundle(f)
    @test isempty(dist.transform_plan_cache)
end
```

- [ ] **Step 2: Run to verify it fails**

Run: `~/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test; include("test/test_field_typestability.jl")'`
Expected: FAIL — `_field_transform_bundle` returns a bundle (it rebuilt), `transform_plan_cache` non-empty.

- [ ] **Step 3: Replace the fallback**

In `src/core/transforms/transform_types.jl` replace the tail of `_field_transform_bundle`:

```julia
    bundle = field.transform_bundle
    bundle isa TransformPlanBundle && return bundle
    # Constructors attach the bundle; rebuilding it here would run the
    # collective planner from an accessor that any single rank may call.
    throw(ArgumentError(
        "field $(repr(field.name)) carries no transform bundle; construct fields " *
        "through ScalarField/VectorField/TensorField so the plan is attached, " *
        "and do not detach `transform_bundle`."))
end
```

- [ ] **Step 4: Document the collective contract**

Append to `src/core/module_contracts.jl` after the BUFFER OWNERSHIP block:

```julia
# ---------------------------------------------------------------------------
# COLLECTIVE ENTRY POINTS — which calls every rank must make together
# ---------------------------------------------------------------------------
#
# Nothing in a signature says "collective". These are, and nothing else in the
# public API may become one without being added here:
#
#   Distributor(...)                    MPI topology + Comm_split
#   Domain(dist, bases) / plan_transforms!  PencilFFT plan construction
#   ScalarField/VectorField/TensorField constructors on a distributed dist
#   TransposableField(field), transpose_workspace!(dist, field)
#   InitialValueSolver / BoundaryValueSolver construction
#   forward_transform!/backward_transform!, evaluate_rhs, step!, solve!
#   NetCDFFileHandler(...), process!(handler), save_state/load_state!
#   close(dist)
#
# Three rules keep the count of collectives identical on every rank:
#   1. An accessor never plans. `_field_transform_bundle` refuses when a field
#      has no bundle; `local_shape(domain, :c)` derives geometry locally.
#   2. A per-rank decision gates the BODY of a collective, never the call
#      (`create_current_file!(handler; create=needs_create)`).
#   3. Per-rank-fallible I/O is settled with `_collectively`, so a failure on
#      one rank aborts all ranks before the next collective.
```

- [ ] **Step 5: Run the test and the serial suite**

Run: `~/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test; include("test/test_field_typestability.jl")'`
Expected: PASS.
Run: `~/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Pkg; Pkg.test()'` (background; ~50 min)
Expected: `Testing Tarang tests passed`. If any test constructs a field without a bundle (e.g. via `copy`), attach the bundle in that constructor rather than relaxing the refusal.

- [ ] **Step 6: Commit**

```bash
git add src/core/transforms/transform_types.jl src/core/module_contracts.jl test/test_field_typestability.jl
git commit -m "fix: _field_transform_bundle refuses instead of planning collectively; document collective entry points"
```

---

### Task 3: The Distributor is the single owner of communicators

**Files:**
- Modify: `src/core/gpu_distributed.jl:1474-1512` (`setup_transposable_workspace!`, `Base.close(::DistributedGPUTransform)`)
- Modify: `src/core/distributor/distributor_core.jl:336-360` (`Base.close(dist)`)
- Create: hook `_close_backend_plan_caches!(dist)` in `src/core/distributor/distributor_core.jl` (default no-op)
- Modify: `ext/cuda/transforms.jl` (implement the hook; keep `clear_distributed_dct_plan_cache!`)
- Test: `test/test_cuda_dct_cache_context.jl` (update AST guards), `test/test_transposable_field.jl` (check its `DistributedGPUTransform` usage still holds)

**Interfaces:**
- Consumes: `transpose_workspace!(dist::Distributor, field::ScalarField) -> TransposableField` (`src/core/transposable_field.jl:274`), `_distributed_dct_comm_token(comm) = objectid(comm)` (`ext/cuda/transforms.jl`).
- Produces: `Tarang._close_backend_plan_caches!(dist::Distributor)` with a default method returning `nothing`; the CUDA extension adds a method that finalizes and deletes every `DISTRIBUTED_DCT_PLAN_CACHE` entry whose key's comm token equals `_distributed_dct_comm_token(dist.comm)`.

- [ ] **Step 1: Update the AST guards (failing first)**

In `test/test_cuda_dct_cache_context.jl` replace the "TransposableField has no GC finalizer either" block with:

```julia
    # The Distributor owns every transpose workspace. DistributedGPUTransform
    # must borrow through transpose_workspace! rather than construct (and then
    # have to close) a TransposableField of its own.
    gpu_distributed_ast = Meta.parseall(read(GPU_DISTRIBUTED_SOURCE, String))
    setup_workspace = _find_definition(gpu_distributed_ast, :setup_transposable_workspace!)
    @test setup_workspace !== nothing
    @test _calls(setup_workspace, :transpose_workspace!)
    @test !_calls(setup_workspace, :TransposableField)

    # close(dist) releases backend plan caches through a hook the CUDA
    # extension implements for its distributed DCT plans.
    distributor_ast = Meta.parseall(read(DISTRIBUTOR_SOURCE, String))
    close_dist = _find_definition(distributor_ast, :close)
    @test close_dist !== nothing
    @test _calls(close_dist, :_close_backend_plan_caches!)
    ext_hook = _find_definition(transforms_ast, :_close_backend_plan_caches!)
    @test ext_hook !== nothing
    @test _calls(ext_hook, :finalize_distributed_dct_plan!)
    @test _calls(ext_hook, :_distributed_dct_comm_token)
```

and add near the other `const *_SOURCE` lines:

```julia
const DISTRIBUTOR_SOURCE = joinpath(@__DIR__, "..", "src", "core", "distributor", "distributor_core.jl")
```

- [ ] **Step 2: Run to verify it fails**

Run: `~/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test; include("test/test_cuda_dct_cache_context.jl")'`
Expected: FAIL on `_calls(setup_workspace, :transpose_workspace!)` and on the hook lookups.

- [ ] **Step 3: Make `DistributedGPUTransform` borrow**

Replace `setup_transposable_workspace!` and `Base.close(::DistributedGPUTransform)` in `src/core/gpu_distributed.jl`:

```julia
function setup_transposable_workspace!(transform::DistributedGPUTransform, field)
    # The Distributor owns one TransposableField per (global shape, eltype) and
    # releases its communicators in close(dist); borrowing it here means this
    # transform never creates, and therefore never has to close, a wrapper.
    workspace = transpose_workspace!(field.dist, field)
    transform.workspace = workspace
    return workspace
end

"""
    close(transform::DistributedGPUTransform)

Drop the transform's reference to the Distributor-owned workspace. The
communicators belong to the Distributor and are released by `close(dist)`.
"""
function Base.close(transform::DistributedGPUTransform)
    transform.workspace = nothing
    return nothing
end
```

- [ ] **Step 4: Add the hook and call it from `close(dist)`**

In `src/core/distributor/distributor_core.jl`, before `Base.close(dist::Distributor)`:

```julia
"""
    _close_backend_plan_caches!(dist::Distributor)

Release any backend-owned plan caches keyed on this Distributor's communicator.
The core package owns no such cache; the CUDA extension adds a method that
finalizes its distributed DCT plans. Called from `close(dist)` before the
communicators are freed, so every collective teardown pairs up across ranks.
"""
_close_backend_plan_caches!(dist::Distributor) = nothing
```

and inside `Base.close(dist::Distributor)`, immediately after `dist.closed && return nothing`:

```julia
    _close_backend_plan_caches!(dist)
```

- [ ] **Step 5: Implement the hook in the extension**

Append to `ext/cuda/transforms.jl` after `clear_distributed_dct_plan_cache!`:

```julia
"""Finalize and drop every cached distributed DCT plan built on `dist.comm`."""
function Tarang._close_backend_plan_caches!(dist::Tarang.Distributor)
    token = _distributed_dct_comm_token(dist.comm)
    lock(DISTRIBUTED_DCT_PLAN_LOCK) do
        for key in collect(keys(DISTRIBUTED_DCT_PLAN_CACHE))
            key[5] == token || continue
            finalize_distributed_dct_plan!(DISTRIBUTED_DCT_PLAN_CACHE[key])
            delete!(DISTRIBUTED_DCT_PLAN_CACHE, key)
        end
    end
    return nothing
end
```

(`key[5]` is the `comm_token` position in `_distributed_dct_plan_cache_key(global_shape, proc_grid, T, axis_kind, comm_token, device_id)`.)

- [ ] **Step 6: Check `test_transposable_field.jl`'s use of the transform**

Run: `grep -n "DistributedGPUTransform\|setup_transposable_workspace!\|\.workspace" test/test_transposable_field.jl`
If a test asserts that the transform's workspace is a distinct object from the Distributor's cache, change that assertion to `@test transform.workspace === Tarang.transpose_workspace!(dist, field)`.

- [ ] **Step 7: Run the affected tests and the serial suite**

Run:
```bash
~/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test; include("test/test_cuda_dct_cache_context.jl"); include("test/test_transposable_field.jl"); include("test/test_cuda_extension_loads.jl")'
```
Expected: all pass (the extension-load test parses the new ext method without CUDA hardware).
Then `Pkg.test()` in the background; expected `Testing Tarang tests passed`.

- [ ] **Step 8: Commit**

```bash
git add src/core/gpu_distributed.jl src/core/distributor/distributor_core.jl ext/cuda/transforms.jl test/test_cuda_dct_cache_context.jl test/test_transposable_field.jl
git commit -m "refactor: Distributor is the single owner of communicators and backend plan caches"
```

---

### Task 4: Fold adjacent `ensure_layout!` + `get_*_data` pairs into the accessors

**Files:**
- Modify: every `src/**/*.jl` with an adjacent pair (106 sites at the start: 65 `:g`, 41 `:c`), except `src/core/field/field_layout/field_layout_access.jl`.
- Modify: `test/test_layout_discipline_ratchet.jl:99` (`LAYOUT_RATCHET = 277` → the new count).

**Interfaces:**
- Consumes: `grid_data!(field) = (ensure_layout!(field, :g); get_grid_data(field))` and `coeff_data!` likewise (`src/core/field/field_layout/field_layout_access.jl:235-243`).
- Produces: nothing new; behaviour identical by construction.

- [ ] **Step 1: Record the starting count**

Run: `~/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test; include("test/test_layout_discipline_ratchet.jl")'`
Expected: passes and logs `total = 277`.

- [ ] **Step 2: Apply the fold script**

Save as `scripts/fold_layout_pairs.py` and run `python3 scripts/fold_layout_pairs.py`:

```python
#!/usr/bin/env python3
"""Fold `ensure_layout!(x, :L)` immediately followed by `get_<L>_data(x)` into
`<L>_data!(x)`. Only the adjacent, same-operand pattern is touched; anything
else is left for the layout ratchet to count."""
import os, re, sys
ROOT = "src"
ENS = re.compile(r'^(\s*)ensure_layout!\(\s*([A-Za-z_][\w\.\[\]]*)\s*,\s*:(g|c)\s*\)\s*$')
ACC = {"g": ("get_grid_data", "grid_data!"), "c": ("get_coeff_data", "coeff_data!")}
folded = 0
for d, _, files in os.walk(ROOT):
    for f in files:
        if not f.endswith(".jl") or f == "field_layout_access.jl":
            continue
        p = os.path.join(d, f)
        L = open(p).read().split("\n")
        out, i, changed = [], 0, False
        while i < len(L):
            m = ENS.match(L[i])
            if m:
                x, lay = m.group(2), m.group(3)
                getter, acc = ACC[lay]
                j = i + 1
                while j < len(L) and (L[j].strip() == "" or L[j].strip().startswith("#")):
                    j += 1
                pat = re.compile(r'\b' + getter + r'\(\s*' + re.escape(x) + r'\s*\)')
                if j < len(L) and pat.search(L[j]) and "ensure_layout!" not in L[j]:
                    out.extend(L[i + 1:j])              # keep blank/comment lines
                    out.append(pat.sub(acc + "(" + x + ")", L[j], count=1))
                    i = j + 1
                    folded += 1
                    changed = True
                    continue
            out.append(L[i]); i += 1
        if changed:
            open(p, "w").write("\n".join(out))
print("folded", folded)
```

Expected output: `folded 106` (±: the count must equal the number reported by the pre-scan in the spec's session, 106).

- [ ] **Step 3: Parse-check every changed file**

Run:
```bash
~/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia -e 'bad=0; for f in readlines(`git diff --name-only`); endswith(f,".jl") || continue; ex=Meta.parseall(read(f,String)); if any(a->a isa Expr && a.head==:error, ex.args); println("PARSE ERROR ", f); global bad+=1; end; end; println("errors=",bad)'
```
Expected: `errors=0`.

- [ ] **Step 4: Lower the ratchet and run it**

Edit `test/test_layout_discipline_ratchet.jl`: `LAYOUT_RATCHET = 171` (277 − 106; use the number the test logs if it differs).
Run: `~/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test; include("test/test_layout_discipline_ratchet.jl")'`
Expected: PASS with `total = 171`.

- [ ] **Step 5: Full serial suite, then MPI at 2 ranks**

Run `Pkg.test()` (background). Expected: `Testing Tarang tests passed`.
Run `~/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia --project=. test/run_mpi_ci.jl 2` (background). Expected: `MPI summary (mpi @ 2 ranks): 58 passed, 0 failed`.
If either fails, revert only the offending file's hunk with `git checkout -p -- <file>` and re-run; the ratchet count then goes up by the number of reverted sites.

- [ ] **Step 6: Commit**

```bash
git add -A src test/test_layout_discipline_ratchet.jl scripts/fold_layout_pairs.py
git commit -m "refactor: fold 106 adjacent ensure_layout!/get_*_data pairs into grid_data!/coeff_data!; ratchet 277 -> 171"
```

---

### Task 5: Split `gpu_distributed.jl` into one file per section

**Files:**
- Create: `src/core/gpu_distributed/config.jl` (lines 1–125: header + "Distributed GPU Configuration"), `fft.jl` (126–539 "Distributed GPU FFT"), `transpose.jl` (540–796 "MPI Transpose Operations"), `utils.jl` (797–898 "Utility Functions"), `nccl.jl` (899–1316 "NCCL Support for GPU Collectives"), `pinned.jl` (1317–1413 "Pinned Memory Staging"), `transform.jl` (1414–1565 "Enhanced Distributed GPU Transform")
- Modify: `src/core/gpu_distributed.jl` becomes the loader: the seven `include`s in that order followed by the original "Exports" section (lines 1566–end).
- Test: `test/test_root_module_structure.jl` (add the new files to whatever list it maintains, if it enumerates files); `test/test_gpu_test_files_reachable.jl` already parses `src/core/gpu_distributed.jl` — point `GPU_DISTRIBUTED_SOURCE` in `test/test_cuda_dct_cache_context.jl` at `src/core/gpu_distributed/transform.jl`.

**Interfaces:**
- Consumes: the section boundaries above (verify with `awk '/^# =+$/{getline t; print NR-1": "t}' src/core/gpu_distributed.jl` before cutting; the numbers may have shifted by Task 3).
- Produces: identical module contents; `include("gpu_distributed.jl")` in `src/core/load_solver_stack.jl:9` is unchanged.

- [ ] **Step 1: Cut the file**

```bash
cd src/core && mkdir -p gpu_distributed
python3 - <<'EOF'
import re
src=open("gpu_distributed.jl").read().split("\n")
# A section opens with a rule line immediately followed by a "# Title" line; the
# closing rule line is followed by a blank line and so is not matched here.
hdr=[i for i,l in enumerate(src) if re.match(r'^# =+$',l) and i+1<len(src) and src[i+1].startswith("# ")]
starts=hdr
names=["config","fft","transpose","utils","nccl","pinned","transform","exports"]
assert len(starts)==8, starts
bounds=starts+[len(src)]
for k,name in enumerate(names[:-1]):
    a = 0 if k==0 else bounds[k]
    open(f"gpu_distributed/{name}.jl","w").write("\n".join(src[a:bounds[k+1]])+"\n")
loader = ["# Distributed GPU support, split by section; see gpu_distributed/*.jl.",""]
loader += [f'include("gpu_distributed/{n}.jl")' for n in names[:-1]]
loader += [""] + src[bounds[7]:]
open("gpu_distributed.jl","w").write("\n".join(loader))
EOF
```

- [ ] **Step 2: Prove nothing but whitespace moved**

```bash
cat src/core/gpu_distributed/{config,fft,transpose,utils,nccl,pinned,transform}.jl | diff - <(git show HEAD:src/core/gpu_distributed.jl | sed -n '1,1565p') | head
```
Expected: empty diff (or only the trailing newline).

- [ ] **Step 3: Update the test source path and run the affected tests**

In `test/test_cuda_dct_cache_context.jl` set `GPU_DISTRIBUTED_SOURCE = joinpath(@__DIR__, "..", "src", "core", "gpu_distributed", "transform.jl")`.
Run: `~/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test; include("test/test_root_module_structure.jl"); include("test/test_cuda_dct_cache_context.jl"); include("test/test_gpu_test_files_reachable.jl")'`
Expected: all pass. If `test_root_module_structure.jl` enumerates source files, add the seven new paths where `gpu_distributed.jl` is listed.

- [ ] **Step 4: Serial suite**

Run `Pkg.test()` (background). Expected: `Testing Tarang tests passed`.

- [ ] **Step 5: Commit**

```bash
git add src/core/gpu_distributed.jl src/core/gpu_distributed test
git commit -m "refactor: split gpu_distributed.jl into one file per section (pure move)"
```

---

### Task 6: Final gate and merge

**Files:** none new.

- [ ] **Step 1: MPI at 1, 2 and 4 ranks (sequential, background)**

```bash
J=~/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia
nohup sh -c "$J --project=. test/run_mpi_ci.jl 1 > mpi1.log 2>&1; $J --project=. test/run_mpi_ci.jl 2 > mpi2.log 2>&1; $J --project=. test/run_mpi_ci.jl 4 > mpi4.log 2>&1" &
```
Expected: each log ends with `58 passed, 0 failed`.

- [ ] **Step 2: Push the branch, open a PR, wait for CI**

```bash
git push -u origin mechanics/implicit-state-2026-09-05
gh pr create --fill --title "Implicit-state mechanics: layout fold, collective refusals, single-owner communicators, gpu_distributed split"
gh pr checks --watch
```
Expected: all CI jobs green (including the new macOS aarch64 jobs).

- [ ] **Step 3: Merge**

```bash
gh pr merge --merge --delete-branch
```
