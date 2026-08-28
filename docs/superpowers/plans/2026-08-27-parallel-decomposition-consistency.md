# Parallel Decomposition Consistency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse the nine hand-maintained copies of the MPI decomposition convention into one function, then wire GPU+MPI fields into the ordinary `forward_transform!` call.

**Architecture:** Phase 0 introduces `decomposed_axes(dist, ndim)` as the single statement of "which global axes are decomposed", and rewrites every consumer to call it. Phase 1 caches the transpose workspace on the `Distributor` (not per field, which would burn two collective `MPI.Comm_split`s per field), constructs `TransposableFieldStorage` for distributed GPU fields, and dispatches the transform on storage type instead of erroring.

**Tech Stack:** Julia 1.10+, MPI.jl (MPICH_jll), PencilArrays.jl, PencilFFTs.jl, CUDA.jl (weak dep), KernelAbstractions.jl, JLArrays (test-only).

**Spec:** `docs/superpowers/specs/2026-08-27-parallel-decomposition-consistency-design.md`

## Global Constraints

- Julia version floor: `julia = "1.10"` (`Project.toml`).
- **No silent CPU fallback for GPU data.** A GPU path that cannot execute must `error()`, never stage to host. This is an existing project contract.
- **No behavior change to any configuration that works today.** Phase 0 is a refactor; the MPI suite is the oracle. Baseline to match: `test/run_mpi_ci.jl 2` → 54 passed / 0 failed, `test/run_mpi_ci.jl 4` → 54 passed / 0 failed, `Pkg.test()` → "Testing Tarang tests passed".
- **Never add a bare `catch`.** `test_catch_ratchet.jl` pins the population; a new silent catch fails the suite.
- **MPI collectives must be rank-uniform.** Any cache lookup that can trigger a collective must key on data every rank agrees on, and must not be guarded by rank-local state.
- Register every new test file in `test/file_lists.jl` or `test_test_inventory.jl` fails.
- Do not commit unless explicitly asked. Steps below stage changes and stop.

### Local commands

```bash
JULIA=~/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia
MPIEXEC=~/.julia/artifacts/f822b53e59145a4dfdceef194142de78ce8e510b/bin/mpiexec
export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/opt/hwloc/lib

# one MPI test file at N ranks
$MPIEXEC -n 4 $JULIA --project=. test/test_transposable_field.jl

# whole MPI suite
$JULIA --project=. test/run_mpi_ci.jl 4

# full package suite (includes Aqua + JET ratchets)
$JULIA --project=. -e 'using Pkg; Pkg.test()'
```

`timeout`/`gtimeout` are unavailable on this box; run long jobs in the background.

## File Structure

**Created:**
- `test/test_decomposition_convention.jl` — pins `decomposed_axes` against a case table and ratchets against re-derivation of the convention at call sites.
- `test/test_mpi_transposable_parity.jl` — coefficient-level parity of the distributed transform against the serial reference, 2-D and 3-D, meshes `(N,1)`, `(1,N)`, `(2,2)`.

**Modified:**
- `src/core/distributor/distributor_core.jl` — home of `decomposed_axes` / `mesh_axis_for` / `is_decomposed_axis`; `local_indices`, `create_pencil` full-decomp branch, `Base.close` early-return bug, transpose workspace cache.
- `src/core/field/field_data/field_data_distributor_utils.jl` — `get_local_range` delegates.
- `src/core/field/field_data/field_data_copy_alloc.jl` — `get_local_array_size` delegates.
- `src/core/field/field_layout/field_layout_filters_shapes.jl` — local start/end delegates.
- `src/core/operators/derivatives/derivatives_fourier.jl` — decomposed-axis guard delegates.
- `src/tools/netcdf_output.jl` — slab start and count delegate.
- `src/core/field/field_types.jl` — `ScalarField` selects `TransposableFieldStorage` for GPU+MPI.
- `src/core/transposable_field.jl` — `TransposableFieldStorage` gains a workspace reference.
- `src/core/transforms/transform_gpu.jl` — dispatch on `TransposableStorage`; delete refusal.
- `src/core/transforms/transform_fourier.jl` — delete refusal.
- `test/test_transposable_field.jl` — restore the vacuous serial testset.
- `test/file_lists.jl` — register the two new files.

---

## Task 1: Land the transposable-transform parity safety net

Phase 0 refactors the TransposableField decomposition convention, which the MPI suite covers thinly (it is round-trip only, and round-trip cannot see a permutation that forward and backward both apply). Land the value assertions **first** so the refactor has an oracle.

**Files:**
- Create: `test/test_mpi_transposable_parity.jl`
- Modify: `test/file_lists.jl`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: a registered MPI test file exercising `distributed_forward_transform!` / `distributed_backward_transform!` at np=2 and np=4 for 2-D and 3-D domains.

- [ ] **Step 1: Write the failing test**

Create `test/test_mpi_transposable_parity.jl`:

```julia
# Coefficient-level parity for the TransposableField distributed transform.
#
# The np>1 testsets in test_transposable_field.jl only ROUND-TRIP. A round trip
# is blind to any permutation that forward and backward both apply, which is
# exactly the failure mode a decomposition-convention refactor can introduce.
# These assertions compare distributed COEFFICIENTS against the serial reference
# sliced to each rank's own block.
using Test
using Tarang
using MPI

MPI.Initialized() || MPI.Init()
const COMM = MPI.COMM_WORLD
const NP = MPI.Comm_size(COMM)
const RANK = MPI.Comm_rank(COMM)

# Distinct per-axis structure so a swapped axis cannot coincidentally agree.
f2(i, j, Nx, Ny) = sin(2π * (i - 1) / Nx) * cos(4π * (j - 1) / Ny) +
                   0.3 * cos(6π * (i - 1) / Nx)
f3(i, j, k, Nx, Ny, Nz) = sin(2π * (i - 1) / Nx) * cos(4π * (j - 1) / Ny) +
                          0.3 * cos(2π * (k - 1) / Nz) * sin(2π * (j - 1) / Ny)

function serial_coeffs_2d(Nx, Ny)
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; comm=MPI.COMM_SELF, mesh=(1,), dtype=ComplexF64,
                       architecture=CPU(), use_pencil_arrays=false)
    bases = (ComplexFourier(coords, "x", Nx), ComplexFourier(coords, "y", Ny))
    field = ScalarField(dist, "parity_ref_2d", bases)
    g = Tarang.get_grid_data(field)
    for j in 1:Ny, i in 1:Nx
        g[i, j] = complex(f2(i, j, Nx, Ny), 0.0)
    end
    forward_transform!(field)
    return copy(Tarang.get_coeff_data(field))
end

function serial_coeffs_3d(Nx, Ny, Nz)
    coords = CartesianCoordinates("x", "y", "z")
    dist = Distributor(coords; comm=MPI.COMM_SELF, mesh=(1,), dtype=ComplexF64,
                       architecture=CPU(), use_pencil_arrays=false)
    bases = (ComplexFourier(coords, "x", Nx), ComplexFourier(coords, "y", Ny),
             ComplexFourier(coords, "z", Nz))
    field = ScalarField(dist, "parity_ref_3d", bases)
    g = Tarang.get_grid_data(field)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        g[i, j, k] = complex(f3(i, j, k, Nx, Ny, Nz), 0.0)
    end
    forward_transform!(field)
    return copy(Tarang.get_coeff_data(field))
end

"Global x/y index ranges this rank owns under the TransposableField convention."
function block_ranges(mesh, Nx, Ny)
    Rx = mesh[1]
    Ry = length(mesh) >= 2 ? mesh[2] : 1
    rx = RANK % Rx
    ry = (RANK ÷ Rx) % Ry
    return (Tarang.local_range(Nx, Rx, rx), Tarang.local_range(Ny, Ry, ry))
end

@testset "TransposableField coefficient parity (np=$NP)" begin

    @testset "2D mesh=$mesh" for mesh in (NP == 4 ? ((4, 1), (1, 4), (2, 2)) : ((NP, 1),))
        Nx, Ny = 8, 6
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; comm=COMM, mesh=mesh, dtype=ComplexF64,
                           architecture=CPU(), use_pencil_arrays=false)
        bases = (ComplexFourier(coords, "x", Nx), ComplexFourier(coords, "y", Ny))
        field = ScalarField(dist, "parity_2d_$(mesh)", bases)

        ox, oy = block_ranges(mesh, Nx, Ny)
        g = Tarang.get_grid_data(field)
        @test size(g) == (length(ox), length(oy))
        for (jl, jg) in enumerate(oy), (il, ig) in enumerate(ox)
            g[il, jl] = complex(f2(ig, jg, Nx, Ny), 0.0)
        end
        original = copy(g)

        reference = serial_coeffs_2d(Nx, Ny)
        tf = TransposableField(field)
        distributed_forward_transform!(tf)

        c = Tarang.get_coeff_data(field)
        @test size(c) == (length(ox), length(oy))
        @test maximum(abs, c .- reference[ox, oy]; init=0.0) < 1e-10

        distributed_backward_transform!(tf)
        @test maximum(abs, Tarang.get_grid_data(field) .- original; init=0.0) < 1e-10
    end

    @testset "3D mesh=$mesh" for mesh in (NP == 4 ? ((2, 2), (4, 1), (1, 4)) : ((NP, 1),))
        Nx, Ny, Nz = 8, 6, 4
        coords = CartesianCoordinates("x", "y", "z")
        dist = Distributor(coords; comm=COMM, mesh=mesh, dtype=ComplexF64,
                           architecture=CPU(), use_pencil_arrays=false)
        bases = (ComplexFourier(coords, "x", Nx), ComplexFourier(coords, "y", Ny),
                 ComplexFourier(coords, "z", Nz))
        field = ScalarField(dist, "parity_3d_$(mesh)", bases)

        ox, oy = block_ranges(mesh, Nx, Ny)
        g = Tarang.get_grid_data(field)
        @test size(g) == (length(ox), length(oy), Nz)
        for k in 1:Nz, (jl, jg) in enumerate(oy), (il, ig) in enumerate(ox)
            g[il, jl, k] = complex(f3(ig, jg, k, Nx, Ny, Nz), 0.0)
        end
        original = copy(g)

        reference = serial_coeffs_3d(Nx, Ny, Nz)
        tf = TransposableField(field)
        distributed_forward_transform!(tf)

        c = Tarang.get_coeff_data(field)
        @test maximum(abs, c .- reference[ox, oy, :]; init=0.0) < 1e-10

        distributed_backward_transform!(tf)
        @test maximum(abs, Tarang.get_grid_data(field) .- original; init=0.0) < 1e-10
    end

    # Over-decomposition: ranks that own an EMPTY block must not hang and must
    # not corrupt the ranks that own data.
    if NP == 4
        @testset "over-decomposed Nx=2 on mesh=(4,1)" begin
            Nx, Ny = 2, 8
            coords = CartesianCoordinates("x", "y")
            dist = Distributor(coords; comm=COMM, mesh=(4, 1), dtype=ComplexF64,
                               architecture=CPU(), use_pencil_arrays=false)
            bases = (ComplexFourier(coords, "x", Nx), ComplexFourier(coords, "y", Ny))
            field = ScalarField(dist, "parity_overdecomp", bases)
            g = Tarang.get_grid_data(field)
            g .= complex(1.0, 0.0)
            tf = TransposableField(field)
            distributed_forward_transform!(tf)
            c = Tarang.get_coeff_data(field)
            # u ≡ 1 has all energy in the DC mode, owned by rank 0.
            if RANK == 0 && !isempty(c)
                @test isapprox(c[1, 1], complex(Nx * Ny, 0.0); rtol=1e-10)
            end
            MPI.Barrier(COMM)
            @test true  # reaching the barrier on every rank is the deadlock assertion
        end
    end
end

MPI.Barrier(COMM)
```

- [ ] **Step 2: Run it to verify it passes on the CURRENT code**

This test is a *characterisation* test, not a red-first test: it pins behavior that is already correct so the refactor cannot break it silently.

```bash
$MPIEXEC -n 2 $JULIA --project=. test/test_mpi_transposable_parity.jl
$MPIEXEC -n 4 $JULIA --project=. test/test_mpi_transposable_parity.jl
```

Expected: all pass. If any fail, STOP — the baseline is not what this plan assumes; report before continuing.

- [ ] **Step 3: Verify it would catch a convention break**

Temporarily edit `block_ranges` to swap `rx` and `ry` (`rx = (RANK ÷ Rx) % Ry; ry = RANK % Rx`), rerun at np=4, confirm the `(2,2)` cases FAIL, then revert the edit. This proves the test discriminates; without it the test may be asserting nothing.

- [ ] **Step 4: Register the file**

In `test/file_lists.jl`, add to `MPI_TEST_FILES` immediately after `"test_transposable_field.jl",`:

```julia
    "test_mpi_transposable_parity.jl",  # distributed COEFFICIENTS must equal serial, not merely round-trip — a permutation applied by both directions is invisible to a round trip
```

- [ ] **Step 5: Verify registration**

```bash
$JULIA --project=. -e 'using Pkg; Pkg.test()' 2>&1 | grep -E "test_test_inventory|inventory"
```

Expected: `test_test_inventory.jl` passes (the file is registered exactly once).

- [ ] **Step 6: Stage**

```bash
git add test/test_mpi_transposable_parity.jl test/file_lists.jl
```

---

## Task 2: Introduce `decomposed_axes`

**Files:**
- Modify: `src/core/distributor/distributor_core.jl` (add near `local_indices`, around line 1055)
- Create: `test/test_decomposition_convention.jl`
- Modify: `test/file_lists.jl`

**Interfaces:**
- Consumes: `Distributor` fields `size`, `mesh`, `use_pencil_arrays`.
- Produces:
  - `decomposed_axes(dist::Distributor, ndim::Int) -> NTuple{M,Int}` — ascending global axis indices that are decomposed; `()` when serial, when `mesh === nothing`, or when the pencil path cannot decompose (`ndim < length(mesh)`).
  - `mesh_axis_for(dist::Distributor, ndim::Int, axis::Int) -> Union{Nothing,Int}` — 1-based index into `dist.mesh` decomposing `axis`, or `nothing`.
  - `is_decomposed_axis(dist::Distributor, ndim::Int, axis::Int) -> Bool`

- [ ] **Step 1: Write the failing test**

Create `test/test_decomposition_convention.jl`:

```julia
# ONE statement of "which axes are decomposed".
#
# This convention used to be re-derived by hand at nine call sites. Nine copies
# of one rule is how the PencilArrays convention (decompose LAST mesh dims) and
# the TransposableField convention (decompose FIRST mesh dims) drifted apart.
# This file pins the single source of truth and ratchets against new copies.
using Test
using Tarang

# A Distributor stand-in: decomposed_axes reads only these three fields, so the
# convention can be tested for mesh/ndim combinations that need no live MPI
# world (the duck-typed-fake-dist trick from test_netcdf_slab_geometry.jl).
struct FakeDist
    size::Int
    mesh::Union{Nothing, Tuple{Vararg{Int}}}
    use_pencil_arrays::Bool
end

@testset "decomposed_axes" begin

    @testset "serial and unmeshed decompose nothing" begin
        @test Tarang.decomposed_axes(FakeDist(1, (4,), true), 3) == ()
        @test Tarang.decomposed_axes(FakeDist(4, nothing, true), 3) == ()
        @test Tarang.decomposed_axes(FakeDist(1, (2, 2), false), 2) == ()
    end

    @testset "PencilArrays decomposes the LAST mesh dims" begin
        @test Tarang.decomposed_axes(FakeDist(4, (4,), true), 2) == (2,)
        @test Tarang.decomposed_axes(FakeDist(4, (4,), true), 3) == (3,)
        @test Tarang.decomposed_axes(FakeDist(4, (2, 2), true), 3) == (2, 3)
        @test Tarang.decomposed_axes(FakeDist(4, (2, 2), true), 2) == (1, 2)
    end

    @testset "TransposableField decomposes the FIRST mesh dims, at most two" begin
        @test Tarang.decomposed_axes(FakeDist(4, (4, 1), false), 2) == (1, 2)
        @test Tarang.decomposed_axes(FakeDist(4, (2, 2), false), 3) == (1, 2)
        @test Tarang.decomposed_axes(FakeDist(2, (2,), false), 2) == (1,)
    end

    @testset "pencil path cannot decompose more dims than the field has" begin
        # get_local_array_size leaves the shape untouched when ndim < length(mesh);
        # decomposed_axes must agree or the allocator and the index math diverge.
        @test Tarang.decomposed_axes(FakeDist(4, (2, 2), true), 1) == ()
    end

    @testset "mesh_axis_for inverts decomposed_axes" begin
        d = FakeDist(4, (2, 2), true)
        @test Tarang.mesh_axis_for(d, 3, 1) === nothing
        @test Tarang.mesh_axis_for(d, 3, 2) == 1
        @test Tarang.mesh_axis_for(d, 3, 3) == 2
        @test Tarang.is_decomposed_axis(d, 3, 3)
        @test !Tarang.is_decomposed_axis(d, 3, 1)

        t = FakeDist(4, (2, 2), false)
        @test Tarang.mesh_axis_for(t, 3, 1) == 1
        @test Tarang.mesh_axis_for(t, 3, 2) == 2
        @test Tarang.mesh_axis_for(t, 3, 3) === nothing
    end

    @testset "out-of-range axes are not decomposed" begin
        d = FakeDist(4, (2, 2), true)
        @test Tarang.mesh_axis_for(d, 3, 0) === nothing
        @test Tarang.mesh_axis_for(d, 3, 4) === nothing
    end
end
```

- [ ] **Step 2: Run it to verify it fails**

```bash
$JULIA --project=. -e 'using Test, Tarang; include("test/test_decomposition_convention.jl")'
```

Expected: FAIL with `UndefVarError: decomposed_axes not defined`.

- [ ] **Step 3: Write the implementation**

In `src/core/distributor/distributor_core.jl`, immediately before `function local_indices(`:

```julia
"""
    decomposed_axes(dist, ndim::Int) -> NTuple{M,Int}

Global axis indices decomposed across the process mesh, ascending, for an
`ndim`-dimensional field on `dist`. Empty when the field is not decomposed.

This is the SINGLE statement of both conventions. It used to be re-derived by
hand at nine call sites, which is how the two conventions drifted:

  * `use_pencil_arrays=true`  — PencilArrays decomposes the LAST `length(mesh)`
    dimensions. When the field has fewer dimensions than the mesh, PencilArrays
    cannot place the decomposition and the field stays local (matching
    `get_local_array_size`, which is the allocator and therefore the authority).
  * `use_pencil_arrays=false` — TransposableField decomposes the FIRST
    `min(length(mesh), 2)` dimensions; it supports a 2-D process mesh at most.

`ndim` is the FIELD's dimensionality, which is not always `dist.dim`; callers
must pass the one they mean.
"""
function decomposed_axes(dist, ndim::Int)
    (dist.size == 1 || dist.mesh === nothing || ndim < 1) && return ()
    nmesh = length(dist.mesh)
    if dist.use_pencil_arrays
        ndim < nmesh && return ()
        start = ndim - nmesh + 1
        return ntuple(i -> start + i - 1, nmesh)
    else
        n = min(nmesh, 2, ndim)
        return ntuple(identity, n)
    end
end

"""
    mesh_axis_for(dist, ndim::Int, axis::Int) -> Union{Nothing,Int}

Index into `dist.mesh` of the mesh dimension decomposing global `axis`, or
`nothing` when `axis` is local. Inverse of [`decomposed_axes`](@ref) for the
call sites that need `dist.mesh[mesh_idx]`.
"""
function mesh_axis_for(dist, ndim::Int, axis::Int)
    axes = decomposed_axes(dist, ndim)
    for (i, a) in enumerate(axes)
        a == axis && return i
    end
    return nothing
end

"""
    is_decomposed_axis(dist, ndim::Int, axis::Int) -> Bool

Whether global `axis` of an `ndim`-dimensional field is split across ranks.
"""
is_decomposed_axis(dist, ndim::Int, axis::Int) = mesh_axis_for(dist, ndim, axis) !== nothing
```

Note the argument is untyped (`dist`, not `dist::Distributor`) so the duck-typed
`FakeDist` in the test can drive every branch without an MPI world. This mirrors
`test_netcdf_slab_geometry.jl`, which uses the same trick.

- [ ] **Step 4: Export the names**

In the same file's export block (search for `export local_indices`), extend it:

```julia
export decomposed_axes, mesh_axis_for, is_decomposed_axis
```

If no such export line exists, add the line above near the other distributor exports.

- [ ] **Step 5: Run the test to verify it passes**

```bash
$JULIA --project=. -e 'using Test, Tarang; include("test/test_decomposition_convention.jl")'
```

Expected: all pass.

- [ ] **Step 6: Register and stage**

In `test/file_lists.jl`, add to `TEST_FILES` after `"test_distributor.jl",`:

```julia
    "test_decomposition_convention.jl",  # the decomposition convention was re-derived by hand at nine sites; two of them disagreed about whether a field with fewer dims than the mesh is decomposed at all
```

```bash
git add src/core/distributor/distributor_core.jl test/test_decomposition_convention.jl test/file_lists.jl
```

---

## Task 3: Migrate `get_local_range`

`get_local_range` computes `mesh_axis` inline — it is the prior art for `mesh_axis_for` and the smallest, safest migration to do first.

**Files:**
- Modify: `src/core/field/field_data/field_data_distributor_utils.jl:79-110`

**Interfaces:**
- Consumes: `mesh_axis_for(dist, ndim, axis)` from Task 2.
- Produces: `get_local_range(dist, global_size, axis)` unchanged in signature and behavior.

- [ ] **Step 1: Pin current behavior**

Append to `test/test_decomposition_convention.jl`, inside a new testset:

```julia
@testset "get_local_range agrees with the convention" begin
    coords = CartesianCoordinates("x", "y", "z")
    dist = Distributor(coords; mesh=(1,), dtype=Float64, architecture=CPU())
    # Serial: every axis is whole.
    for axis in 1:3
        @test Tarang.get_local_range(dist, 12, axis) == (1, 12)
    end
end
```

- [ ] **Step 2: Run to verify it passes before the change**

```bash
$JULIA --project=. -e 'using Test, Tarang; include("test/test_decomposition_convention.jl")'
```

Expected: pass (this pins the serial contract that the rewrite must not disturb).

- [ ] **Step 3: Rewrite the body**

In `src/core/field/field_data/field_data_distributor_utils.jl`, replace the block that begins `mesh_dim = length(dist.mesh)` and ends with the `if mesh_axis === nothing || mesh_axis < 1 || mesh_axis > mesh_dim` guard (currently lines 84-106) with:

```julia
    # `dist.dim` is this distributor's field dimensionality; get_local_range is
    # called with a global axis index into a field of that rank.
    mesh_axis = mesh_axis_for(dist, dist.dim, axis)
    if mesh_axis === nothing
        return (1, global_size)
    end
```

Leave everything from `n_procs = dist.mesh[mesh_axis]` onward untouched — the
remainder handling (PencilArrays' real range when available, remainder-on-first
otherwise) is not part of this refactor.

- [ ] **Step 4: Run the full MPI suite at both rank counts**

```bash
$JULIA --project=. test/run_mpi_ci.jl 2
$JULIA --project=. test/run_mpi_ci.jl 4
```

Expected: `54 passed, 0 failed` at each. Any regression here is a real convention change — do not proceed past it.

- [ ] **Step 5: Stage**

```bash
git add src/core/field/field_data/field_data_distributor_utils.jl test/test_decomposition_convention.jl
```

---

## Task 4: Migrate the three local-shape/index functions and force them to agree

`get_local_array_size` (the allocator), `local_indices` (the index math), and
`compute_local_shape` (a third copy, in `distributor_core.jl`) each derive the
convention independently. Two of them **disagree** when a field has fewer
dimensions than the mesh: `get_local_array_size` leaves the shape whole
(`if ndims_global >= ndims_mesh`), while `local_indices` uses `dist.dim` and
would split it. `decomposed_axes` resolves this in favour of the allocator.

Note there are two similarly-named functions: `compute_local_shape(dist, global_shape)`
in `distributor_core.jl:883` (this task) and `compute_local_shape(global_shape, decomp_dim, nprocs, rank)`
in `gpu_distributed.jl:91` (a different, explicitly-parameterised function — leave it alone).

**Files:**
- Modify: `src/core/distributor/distributor_core.jl:1058-1100` (`local_indices`)
- Modify: `src/core/distributor/distributor_core.jl:883-930` (`compute_local_shape`)
- Modify: `src/core/field/field_data/field_data_copy_alloc.jl:318-382` (`get_local_array_size`)
- Modify: `test/test_decomposition_convention.jl`

**Interfaces:**
- Consumes: `decomposed_axes`, `mesh_axis_for` from Task 2.
- Produces: `local_indices(dist, axis, global_size)`, `get_local_array_size(dist, global_shape)`, and `compute_local_shape(dist, global_shape)` with unchanged signatures; all three now derived from one rule.

- [ ] **Step 1: Write the failing agreement test**

Append to `test/test_decomposition_convention.jl`:

```julia
@testset "allocator and index math agree on every axis" begin
    # get_local_array_size decides the ALLOCATED shape; local_indices decides
    # which global indices those slots mean. If they disagree the field is
    # silently mis-addressed — no error, wrong values. Nothing forced them to
    # agree before this test existed.
    coords = CartesianCoordinates("x", "y", "z")
    dist = Distributor(coords; mesh=(1,), dtype=Float64, architecture=CPU())
    gshape = (8, 6, 4)
    local_shape = Tarang.get_local_array_size(dist, gshape)
    for axis in 1:3
        @test length(Tarang.local_indices(dist, axis, gshape[axis])) == local_shape[axis]
    end
end
```

- [ ] **Step 2: Run to verify it passes serially, then extend it to MPI**

```bash
$JULIA --project=. -e 'using Test, Tarang; include("test/test_decomposition_convention.jl")'
```

Expected: pass. The MPI half of this assertion is covered by Step 5's suite run,
which exercises both functions on live decompositions across 54 files.

- [ ] **Step 3: Rewrite `local_indices`**

In `src/core/distributor/distributor_core.jl`, replace the whole body between
`function local_indices(dist::Distributor, axis::Int, global_size::Int)` and the
line `n_procs = dist.mesh[mesh_dim]` (currently lines 1058 through the fixup
block ending around line 1096) with:

```julia
function local_indices(dist::Distributor, axis::Int, global_size::Int)
    if dist.size == 1 || dist.mesh === nothing
        return 1:global_size
    end

    mesh_dim = mesh_axis_for(dist, dist.dim, axis)
    if mesh_dim === nothing
        return 1:global_size
    end
```

The old body's two-stage derivation — a loop that computed `nothing` for the
pencil case, followed by a separate fixup block — collapses into that single
lookup. Keep everything from `n_procs = dist.mesh[mesh_dim]` onward as-is.

Also update the docstring's convention paragraph to point at `decomposed_axes`
rather than restating the rule.

- [ ] **Step 4: Rewrite `get_local_array_size`**

In `src/core/field/field_data/field_data_copy_alloc.jl`, replace the
`if dist.use_pencil_arrays ... else ... end` block (lines 330-378) with:

```julia
    for (mesh_idx, dim) in enumerate(decomposed_axes(dist, ndims_global))
        n_global = global_shape[dim]
        n_procs = mesh[mesh_idx]

        if dist.use_pencil_arrays
            # Match PencilArrays' own decomposition exactly when it is available,
            # so the reported local shape equals the slab the pencil owns.
            pr = pencil_local_range(dist, mesh_idx, n_procs, n_global)
            if pr !== nothing
                local_shape[dim] = length(pr)
                continue
            end
        end

        proc_coord = coords[mesh_idx]
        base_size = div(n_global, n_procs)
        remainder = n_global % n_procs
        local_shape[dim] = base_size + (proc_coord < remainder ? 1 : 0)
    end
```

The `coords` vector computed above this block is unchanged and still indexes by
mesh dimension, which is what `mesh_idx` now is.

- [ ] **Step 5: Rewrite `compute_local_shape`**

In `src/core/distributor/distributor_core.jl`, in `compute_local_shape(dist, global_shape)`,
replace the `for i in 1:min(ndims_mesh, ndims_global)` loop and its
`global_dim_idx = if dist.use_pencil_arrays ... else i end` derivation with:

```julia
    for (mesh_dim_idx, global_dim_idx) in enumerate(decomposed_axes(dist, ndims_global))
        n_global = global_shape[global_dim_idx]
        n_procs = dist.mesh[mesh_dim_idx]
```

Delete the now-dead `if global_dim_idx < 1 || global_dim_idx > ndims_global; continue; end`
guard and the `mesh_dim_idx = i` line — `decomposed_axes` only ever yields in-range
axes, and `enumerate` supplies the mesh index. Everything from the
`# Match PencilArrays' decomposition exactly ...` comment onward is unchanged.

- [ ] **Step 6: Add the three-way agreement assertion**

Extend the testset from Step 1 so all three functions are compared, not two:

```julia
    @test collect(Tarang.compute_local_shape(dist, gshape)) == collect(local_shape)
```

- [ ] **Step 7: Run the full suites**

```bash
$JULIA --project=. test/run_mpi_ci.jl 2
$JULIA --project=. test/run_mpi_ci.jl 4
$JULIA --project=. -e 'using Pkg; Pkg.test()'
```

Expected: `54 passed, 0 failed` twice, and `Testing Tarang tests passed`.

- [ ] **Step 8: Stage**

```bash
git add src/core/distributor/distributor_core.jl \
        src/core/field/field_data/field_data_copy_alloc.jl \
        test/test_decomposition_convention.jl
```

---

## Task 5: Migrate the derivative guard and the layout filter

**Files:**
- Modify: `src/core/operators/derivatives/derivatives_fourier.jl:30-46`
- Modify: `src/core/field/field_layout/field_layout_filters_shapes.jl:265-295`

**Interfaces:**
- Consumes: `is_decomposed_axis`, `decomposed_axes` from Task 2.
- Produces: no signature changes.

- [ ] **Step 1: Rewrite the derivative guard**

In `src/core/operators/derivatives/derivatives_fourier.jl`, replace the block from
`ndims_mesh = length(dist.mesh)` through `if axis in decomp_dims` with:

```julia
    if dist.size > 1 && ndim >= 2 && dist.mesh !== nothing
        if is_decomposed_axis(dist, ndim, axis)
```

Everything inside that `if` body is unchanged.

- [ ] **Step 2: Rewrite the layout filter**

In `src/core/field/field_layout/field_layout_filters_shapes.jl`, replace the
`if dist.use_pencil_arrays ... else ... end` block (lines 273-292) with:

```julia
        for dim in decomposed_axes(dist, ndims_global)
            start_idx, end_idx = get_local_range(dist, global_shape[dim], dim)
            local_start[dim] = start_idx
            local_end[dim] = end_idx
        end
```

`get_local_range` already resolves the convention internally (Task 3), so this
site no longer needs to know it — it only needs the list of axes to ask about.

- [ ] **Step 3: Run the suites**

```bash
$JULIA --project=. test/run_mpi_ci.jl 4
$JULIA --project=. -e 'using Pkg; Pkg.test()'
```

Expected: `54 passed, 0 failed`; `Testing Tarang tests passed`.

The derivative guard is exercised by `test_mpi_collective_budget.jl` and
`test_mpi_lazy_rhs_fourier.jl`; the layout filter by `test_mpi_spectral_filter.jl`.
Confirm those three appear as `✓` in the MPI output.

- [ ] **Step 4: Stage**

```bash
git add src/core/operators/derivatives/derivatives_fourier.jl \
        src/core/field/field_layout/field_layout_filters_shapes.jl
```

---

## Task 6: Migrate `create_pencil` and the NetCDF slab sites

**Files:**
- Modify: `src/core/distributor/distributor_core.jl:598-612`
- Modify: `src/tools/netcdf_output.jl:1290-1310` (slab start)
- Modify: `src/tools/netcdf_output.jl:1435-1455` (slab count)

**Interfaces:**
- Consumes: `decomposed_axes` from Task 2.
- Produces: no signature changes.

- [ ] **Step 1: Migrate `create_pencil`'s full-decomposition branch**

In `src/core/distributor/distributor_core.jl`, replace only the
`decomp_index === nothing` branch:

```julia
    decomp_dims = if decomp_index === nothing
        # Full decomposition for field storage — the same rule the allocator and
        # the index math use.
        decomposed_axes(dist, ndims_global)
    else
        # Pencil decomposition: keep decomp_index LOCAL for the FFT. This is a
        # different question from "which axes does storage decompose", so it
        # keeps its own helper.
        _compute_decomp_dims(ndims_global, ndims_mesh, decomp_index)
    end
```

Leave `_compute_decomp_dims` alone. If `_compute_full_decomp_dims` now has no
remaining callers, delete it; if it has others, leave it and note them.

- [ ] **Step 2: Check for other callers before deleting**

```bash
grep -rn --include="*.jl" "_compute_full_decomp_dims" src test
```

Delete the function only if this returns just its own definition.

- [ ] **Step 3: Migrate the NetCDF slab COUNT site**

At `src/tools/netcdf_output.jl:1298`, replace:

```julia
    if dist.use_pencil_arrays
        # PencilArrays convention: decompose LAST n_mesh_dims dimensions
        for i in 1:n_dims
            mesh_dim_idx = i - (n_dims - n_mesh_dims)

            if mesh_dim_idx >= 1 && mesh_dim_idx <= n_mesh_dims
```

with:

```julia
    if dist.use_pencil_arrays
        for i in 1:n_dims
            mesh_dim_idx = mesh_axis_for(dist, n_dims, i)

            if mesh_dim_idx !== nothing
```

Everything inside — the `pencil_local_range` call, the remainder-on-LAST
fallback, and the long comment explaining why the previous remainder-on-FIRST
formula NaN-filled merged fields — is unchanged. Leave the `else` branch
(non-decomposed dimension) as it is.

- [ ] **Step 4: Migrate the NetCDF slab START site**

At `src/tools/netcdf_output.jl:1441`, make the identical substitution:

```julia
    if dist.use_pencil_arrays
        for i in 1:n_dims
            mesh_dim_idx = mesh_axis_for(dist, n_dims, i)

            if mesh_dim_idx !== nothing
```

replacing the same `mesh_dim_idx = i - (n_dims - n_mesh_dims)` /
`if mesh_dim_idx >= 1 && mesh_dim_idx <= n_mesh_dims` pair. Keep the local
`compute_start` helper and the `pencil_local_range` preference untouched.

These two sites are the ones the 2026-08-20 audit found disagreeing — balanced
start against remainder-first count, which overlapped and gapped when `N % P != 0`.
Deriving the axis list from one call is what stops them drifting again.

- [ ] **Step 5: Confirm `n_mesh_dims` is still used**

Both sites keep `n_mesh_dims = length(mesh)` for other purposes. After the edit:

```bash
grep -n "n_mesh_dims" src/tools/netcdf_output.jl
```

If a binding is now unused, delete it; Aqua's unused-binding checks do not cover
locals, so a stale one will sit there silently.

- [ ] **Step 6: Run the slab geometry test specifically**

```bash
$JULIA --project=. -e 'using Test, Tarang; include("test/test_netcdf_slab_geometry.jl")'
$MPIEXEC -n 4 $JULIA --project=. test/test_mpi_output_audit.jl
$MPIEXEC -n 2 $JULIA --project=. test/test_mpi_checkpoint_restart.jl
```

Expected: all pass. `test_netcdf_slab_geometry.jl` is in `OPTIONAL_TEST_FILES`, so
run it directly — the default suite does not.

- [ ] **Step 7: Run the full suites**

```bash
$JULIA --project=. test/run_mpi_ci.jl 4
$JULIA --project=. -e 'using Pkg; Pkg.test()'
```

- [ ] **Step 8: Stage**

```bash
git add src/core/distributor/distributor_core.jl src/tools/netcdf_output.jl
```

---

## Task 7: Ratchet against new copies of the convention

**Files:**
- Modify: `test/test_decomposition_convention.jl`

**Interfaces:**
- Consumes: the migrated call sites from Tasks 3-6.
- Produces: a source-scanning test that fails when the convention is re-derived anywhere outside `decomposed_axes`.

- [ ] **Step 1: Write the failing ratchet**

Append to `test/test_decomposition_convention.jl`:

```julia
@testset "the convention is stated in exactly one place" begin
    # Nine independently-maintained copies of this rule are how the PencilArrays
    # and TransposableField conventions drifted apart, and how two of them ended
    # up disagreeing about a field with fewer dims than the mesh. A tenth copy
    # must fail the build, not wait for the next audit.
    srcdir = joinpath(@__DIR__, "..", "src")
    allowed = joinpath("core", "distributor", "distributor_core.jl")

    offenders = String[]
    for (root, _, files) in walkdir(srcdir), file in files
        endswith(file, ".jl") || continue
        path = joinpath(root, file)
        occursin(allowed, path) && continue
        text = read(path, String)
        # The tell is a use_pencil_arrays branch that decides axis indices.
        if occursin(r"use_pencil_arrays"i, text) &&
           occursin(r"decompose\s+(LAST|FIRST)\s+\w*mesh"i, text)
            push!(offenders, relpath(path, srcdir))
        end
    end

    @test isempty(offenders)
    isempty(offenders) || @info "convention re-derived outside decomposed_axes" offenders
end
```

- [ ] **Step 2: Run it**

```bash
$JULIA --project=. -e 'using Test, Tarang; include("test/test_decomposition_convention.jl")'
```

Expected: PASS if Tasks 3-6 removed every comment-and-branch pair. If it fails, the
listed files still carry a copy — migrate them before continuing. Comments alone
are not offenders; the test requires both the `use_pencil_arrays` mention and the
"decompose LAST/FIRST … mesh" phrasing in the same file, so an explanatory comment
that references `decomposed_axes` instead will not trip it.

- [ ] **Step 3: Verify the ratchet discriminates**

Temporarily paste this into any migrated file:

```julia
# PencilArrays convention: decompose LAST ndims_mesh dimensions
_scratch(dist) = dist.use_pencil_arrays
```

Rerun; confirm the test FAILS and names that file. Remove the paste.

- [ ] **Step 4: Stage**

```bash
git add test/test_decomposition_convention.jl
```

---

## Task 8: Refuse a 2-D process mesh on a 2-D domain honestly

Currently this dies deep in solver construction with *"PencilFFT plan creation
failed with 4 MPI processes. Local FFTW fallback would produce incorrect results.
Please check your PencilFFTs installation or use serial execution."* — which
blames the user's install for a structural impossibility. With `decomposed_axes`
in place the condition is one line.

**Files:**
- Modify: `src/core/transforms/transform_planning.jl:283-306`
- Modify: `test/test_decomposition_convention.jl`

**Interfaces:**
- Consumes: `decomposed_axes` from Task 2.
- Produces: an `ArgumentError` naming the real constraint, raised before plan creation.

- [ ] **Step 1: Write the failing test**

Append to `test/test_decomposition_convention.jl`:

```julia
@testset "a mesh that leaves no local axis is refused by name" begin
    # PencilFFTs needs at least one local axis; an N-D domain therefore supports
    # at most (N-1)-D decomposition. Refusing with "check your PencilFFTs
    # installation" sent users to debug their environment for a domain shape
    # that can never work.
    @test Tarang.decomposed_axes(FakeDist(4, (2, 2), true), 2) == (1, 2)
end
```

The behavioral half of this is an MPI assertion; add it to
`test/test_mpi_transform_planning_guards.jl` inside its existing `@testset`:

```julia
if NPROCS == 4
    @testset "2D mesh on a 2D domain refuses with the real reason" begin
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; comm=MPI.COMM_WORLD, mesh=(2, 2),
                           dtype=Float64, architecture=CPU())
        bases = (RealFourier(coords["x"]; size=16, bounds=(0.0, 2π)),
                 RealFourier(coords["y"]; size=12, bounds=(0.0, 2π)))
        err = try
            ScalarField(dist, "no_local_axis", bases)
            forward_transform!(ScalarField(dist, "no_local_axis2", bases))
            nothing
        catch e
            e
        end
        @test err !== nothing
        msg = sprint(showerror, err)
        @test occursin("at least one local", msg) || occursin("no local axis", msg)
        @test !occursin("PencilFFTs installation", msg)
    end
end
```

- [ ] **Step 2: Run to verify it fails**

```bash
$MPIEXEC -n 4 $JULIA --project=. test/test_mpi_transform_planning_guards.jl
```

Expected: FAIL — the raised message still contains "PencilFFTs installation".

- [ ] **Step 3: Implement the guard**

In `src/core/transforms/transform_planning.jl`, immediately before the
`trailing = collect(...)` line, insert:

```julia
    # PencilFFTs transforms along a LOCAL axis and transposes between stages, so
    # at least one axis must stay local: an N-D domain supports at most (N-1)-D
    # decomposition. A mesh that covers every axis cannot be planned at all, and
    # the failure used to surface as a generic "PencilFFT plan creation failed …
    # check your PencilFFTs installation" from inside solver construction.
    decomp_axes = decomposed_axes(dist, ndims_total)
    if length(decomp_axes) >= ndims_total
        throw(ArgumentError(
            "A $(ndims_total)-D domain needs at least one local axis, but mesh=" *
            "$(dist.mesh) decomposes all $(ndims_total) of them. PencilFFTs " *
            "transforms along a local axis and transposes between stages, so an " *
            "N-D domain supports at most (N-1)-D process decomposition. Use a 1-D " *
            "mesh — mesh=($(dist.size),) — for a $(ndims_total)-D domain, or add a " *
            "dimension to the domain."))
    end
```

- [ ] **Step 4: Run to verify it passes**

```bash
$MPIEXEC -n 4 $JULIA --project=. test/test_mpi_transform_planning_guards.jl
$JULIA --project=. -e 'using Test, Tarang; include("test/test_decomposition_convention.jl")'
```

Expected: both pass.

- [ ] **Step 5: Full suites, then stage**

```bash
$JULIA --project=. test/run_mpi_ci.jl 2
$JULIA --project=. test/run_mpi_ci.jl 4
$JULIA --project=. -e 'using Pkg; Pkg.test()'
git add src/core/transforms/transform_planning.jl \
        test/test_decomposition_convention.jl \
        test/test_mpi_transform_planning_guards.jl
```

**Phase 0 is complete at this point.** The remaining tasks are Phase 1 and depend
on nothing from each other except in listed order.

---

## Task 9: Fix `Base.close(dist)` returning before it marks the distributor closed

`close` bails at `topology === nothing && return nothing` *before* setting
`dist.closed = true`. `mpi_topology` is `nothing` for exactly the serial and
GPU+MPI cases, so those distributors never register as closed, `isopen` keeps
returning `true`, and the workspace cache Task 10 adds would never be released.

**Files:**
- Modify: `src/core/distributor/distributor_core.jl:299-329`
- Modify: `test/test_distributor.jl`

**Interfaces:**
- Consumes: nothing.
- Produces: `close(dist)` always leaves `dist.closed == true`; `isopen(dist) == false`.

- [ ] **Step 1: Write the failing test**

Append to `test/test_distributor.jl`:

```julia
@testset "close marks every distributor closed" begin
    # mpi_topology is nothing for serial AND for GPU+MPI (use_pencil_arrays=false).
    # close() returned before setting `closed` in that case, so those distributors
    # stayed permanently "open" and their caches were never dropped.
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; mesh=(1,), dtype=Float64, architecture=CPU())
    @test isopen(dist)
    close(dist)
    @test !isopen(dist)
    @test dist.closed
    close(dist)          # repeated calls stay safe
    @test !isopen(dist)
end
```

- [ ] **Step 2: Run to verify it fails**

```bash
$JULIA --project=. -e 'using Test, Tarang; include("test/test_distributor.jl")'
```

Expected: FAIL at `@test !isopen(dist)`.

- [ ] **Step 3: Implement**

In `src/core/distributor/distributor_core.jl`, replace:

```julia
    dist.closed && return nothing
    topology = dist.mpi_topology
    topology === nothing && return nothing
```

with:

```julia
    dist.closed && return nothing
    topology = dist.mpi_topology
    if topology === nothing
        # Serial and GPU+MPI distributors own no Cartesian topology, but they do
        # own caches and must still register as closed — `isopen` and the cache
        # release below are not conditional on having a topology.
        empty!(dist.pencil_cache)
        empty!(dist.transforms)
        empty!(dist.layouts)
        dist.closed = true
        return nothing
    end
```

- [ ] **Step 4: Run to verify it passes**

```bash
$JULIA --project=. -e 'using Test, Tarang; include("test/test_distributor.jl")'
$JULIA --project=. test/run_mpi_ci.jl 2
```

Expected: pass; `54 passed, 0 failed`.

- [ ] **Step 5: Stage**

```bash
git add src/core/distributor/distributor_core.jl test/test_distributor.jl
```

---

## Task 10: Cache the transpose workspace on the Distributor

`TransposableField`'s constructor calls `MPI.Comm_split` twice. One per field
means `nfields × 2` collective splits and a finite communicator budget spent on
field allocation. Cache by shape instead.

The `Distributor` already carries `transpose_comms_cache::Dict{Int, AbstractTransposeComms}`,
which the 2026-08-20 audit found is **never written** — a dead always-empty field.
Replace it rather than adding a tenth cache.

**Files:**
- Modify: `src/core/distributor/distributor_core.jl` (struct field, constructor, `close`)
- Modify: `src/core/transposable_field.jl` (workspace accessor)
- Modify: `test/test_transposable_field.jl`

**Interfaces:**
- Consumes: `Base.close` from Task 9.
- Produces: `transpose_workspace!(dist, field) -> TransposableField` — returns the cached workspace for `field`'s global shape and element type, constructing it on first use. Same object for two fields of the same shape.

- [ ] **Step 1: Write the failing test**

Append to `test/test_transposable_field.jl`, inside the `NPROCS > 1` block:

```julia
@testset "transpose workspace is cached per shape, not per field" begin
    # TransposableField's constructor performs two collective MPI.Comm_splits.
    # One workspace per FIELD would burn 2*nfields communicators; two fields of
    # the same shape must share one.
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; comm=MPI.COMM_WORLD, mesh=(NPROCS,),
                       dtype=ComplexF64, architecture=CPU(), use_pencil_arrays=false)
    bases = (ComplexFourier(coords, "x", 8), ComplexFourier(coords, "y", 6))
    a = ScalarField(dist, "ws_a", bases)
    b = ScalarField(dist, "ws_b", bases)

    wa = Tarang.transpose_workspace!(dist, a)
    wb = Tarang.transpose_workspace!(dist, b)
    @test wa === wb

    # A different shape gets its own workspace.
    other = (ComplexFourier(coords, "x", 16), ComplexFourier(coords, "y", 6))
    c = ScalarField(dist, "ws_c", other)
    @test Tarang.transpose_workspace!(dist, c) !== wa
end
```

- [ ] **Step 2: Run to verify it fails**

```bash
$MPIEXEC -n 2 $JULIA --project=. test/test_transposable_field.jl
```

Expected: FAIL with `UndefVarError: transpose_workspace! not defined`.

- [ ] **Step 3: Replace the dead cache field**

In the `Distributor` struct, replace:

```julia
    transpose_comms_cache::Dict{Int, AbstractTransposeComms}
```

with:

```julia
    # Transpose workspaces for GPU+MPI (TransposableField), keyed by
    # (global_shape, eltype). A workspace owns two MPI sub-communicators, so it
    # is shared across every field of the same shape rather than built per field.
    # Released in `close`. Replaces the never-written `transpose_comms_cache`.
    transpose_workspace_cache::Dict{Tuple, Any}
```

Update the `new(...)` call in the inner constructor: the positional argument that
was `Dict{Int, AbstractTransposeComms}()` becomes `Dict{Tuple, Any}()`. Find it by
searching for `transpose_counts_cache` — the two are adjacent.

- [ ] **Step 4: Check for readers of the removed field**

```bash
grep -rn --include="*.jl" "transpose_comms_cache" src test
```

Expected: no hits after the edit. If any remain, they were reading an
always-empty dict; delete those reads.

- [ ] **Step 5: Implement the accessor**

In `src/core/transposable_field.jl`, after the `TransposableField` constructor:

```julia
"""
    transpose_workspace!(dist::Distributor, field::ScalarField) -> TransposableField

Cached transpose workspace for `field`'s global shape and element type.

A `TransposableField` owns two MPI sub-communicators (`MPI.Comm_split` is
collective), so one per field would consume `2 * nfields` communicators and
require every rank to allocate the same fields in the same order. Keying on
shape and eltype makes the workspace shared and the construction rank-uniform.

The wrapped `field` reference is repointed on each call: buffers, counts, and
communicators depend only on shape, but the transform reads and writes through
`tf.field`.
"""
function transpose_workspace!(dist::Distributor, field::ScalarField)
    gshape = field.domain !== nothing ? global_shape(field.domain) : size(field["g"])
    key = (gshape, field.dtype)
    ws = get!(dist.transpose_workspace_cache, key) do
        TransposableField(field)
    end
    ws.field = field
    return ws
end
```

`TransposableField` is already a `mutable struct`, so `ws.field = field` is legal.
Verify with `grep -n "mutable struct TransposableField" src/core/transpose/transpose_types.jl`
before relying on it; if it is immutable, make the `field` slot mutable in that struct.

- [ ] **Step 6: Release the cache in `close`**

In `Base.close(dist)`, add `empty!(dist.transpose_workspace_cache)` next to
`empty!(dist.pencil_cache)` in **both** branches — the `topology === nothing`
branch from Task 9 and the main branch. GPU+MPI distributors take the first one,
so omitting it there leaks every workspace.

- [ ] **Step 7: Run to verify it passes**

```bash
$MPIEXEC -n 2 $JULIA --project=. test/test_transposable_field.jl
$MPIEXEC -n 4 $JULIA --project=. test/test_transposable_field.jl
$MPIEXEC -n 4 $JULIA --project=. test/test_mpi_transposable_parity.jl
```

Expected: all pass.

- [ ] **Step 8: Stage**

```bash
git add src/core/distributor/distributor_core.jl \
        src/core/transposable_field.jl \
        test/test_transposable_field.jl
```

---

## Task 11: Construct `TransposableFieldStorage` for distributed GPU fields

**Files:**
- Modify: `src/core/field/field_types.jl:120-133`
- Modify: `src/core/transposable_field.jl:88-103`
- Modify: `test/test_transposable_field.jl`

**Interfaces:**
- Consumes: `transpose_workspace!` from Task 10.
- Produces: `storage_mode(field) isa TransposableStorage` is true exactly when `is_gpu(dist.architecture) && dist.size > 1`.

- [ ] **Step 1: Write the failing test**

Append to `test/test_transposable_field.jl`, inside the `NPROCS > 1` block:

```julia
@testset "distributed GPU fields select transposable storage" begin
    # TransposableFieldStorage has been defined, documented and dispatched since
    # the wrapper was split up — but nothing ever constructed it, so every field
    # got SerialFieldStorage and the distributed transform stayed unreachable
    # from the ordinary API.
    coords = CartesianCoordinates("x", "y")

    cpu_dist = Distributor(coords; comm=MPI.COMM_WORLD, mesh=(NPROCS,),
                           dtype=ComplexF64, architecture=CPU(),
                           use_pencil_arrays=false)
    bases = (ComplexFourier(coords, "x", 8), ComplexFourier(coords, "y", 6))
    cpu_field = ScalarField(cpu_dist, "storage_cpu", bases)
    @test !Tarang.is_transposable_storage(cpu_field)
end
```

The GPU half runs without hardware via JLArray in Task 13; this task pins that
CPU fields are *not* affected.

- [ ] **Step 2: Run to verify it passes (guarding against over-reach)**

```bash
$MPIEXEC -n 2 $JULIA --project=. test/test_transposable_field.jl
```

Expected: pass. This is the regression fence for Step 3 — the selection must not
capture CPU distributors.

- [ ] **Step 3: Give the storage type a workspace slot**

In `src/core/transposable_field.jl`, replace the `TransposableFieldStorage` field
list's buffer members with a workspace reference:

```julia
mutable struct TransposableFieldStorage{CT, N, B<:SerialFieldStorage} <: AbstractFieldStorage
    base::B
    # The transpose buffers, counts, comms and topology live on the Distributor's
    # workspace cache (one per shape) rather than per field — see
    # `transpose_workspace!`. This slot is filled lazily on first transform so
    # field construction performs no collective MPI.Comm_split.
    workspace::Union{Nothing, TransposableField}
end
```

Every accessor that previously reached `storage.transpose_buffers`, `.counts`,
`.comms`, `.topology`, `.global_shape`, `.local_shapes`, `.async_state`,
`.fft_plans`, `.total_transpose_time` or `.total_fft_time` now reaches
`storage.workspace.<same name>`. Find them:

```bash
grep -rn --include="*.jl" "TransposableFieldStorage" src | grep -v "transposable_field.jl"
```

At the time of writing there are none outside the definition site — confirm this
before assuming it.

- [ ] **Step 4: Select the storage in the constructor**

In `src/core/field/field_types.jl`, in the primary `ScalarField` inner
constructor, replace:

```julia
        storage = SerialFieldStorage{_grid_storage_param(g), _coeff_storage_param(c)}(dist.architecture, g, c)
        return new{T, typeof(storage)}(dist, name, bases, domain, dtype, storage, layout, :g, initial_scales, :auto, false, 0)
```

with:

```julia
        base = SerialFieldStorage{_grid_storage_param(g), _coeff_storage_param(c)}(dist.architecture, g, c)
        # A distributed GPU field transforms by explicit transposes, never by
        # PencilFFTs (which is CPU-only). Record that in the storage type so the
        # transform dispatches on it instead of erroring at the call site.
        storage = if is_gpu(dist.architecture) && dist.size > 1
            CT = T <: Complex ? T : Complex{T}
            TransposableFieldStorage{CT, length(bases), typeof(base)}(base, nothing)
        else
            base
        end
        return new{T, typeof(storage)}(dist, name, bases, domain, dtype, storage, layout, :g, initial_scales, :auto, false, 0)
```

- [ ] **Step 5: Forward the storage accessors**

`get_grid_data` / `set_grid_data!` / `get_coeff_data` / `set_coeff_data!` dispatch
on the storage type. Add forwarding methods in `src/core/transposable_field.jl`:

```julia
get_grid_data(s::TransposableFieldStorage)  = get_grid_data(s.base)
get_coeff_data(s::TransposableFieldStorage) = get_coeff_data(s.base)
set_grid_data!(s::TransposableFieldStorage, v)  = set_grid_data!(s.base, v)
set_coeff_data!(s::TransposableFieldStorage, v) = set_coeff_data!(s.base, v)
architecture(s::TransposableFieldStorage) = architecture(s.base)
```

Check the exact accessor names and arities first:

```bash
grep -rn --include="*.jl" "SerialFieldStorage)" src/core/field/field_types.jl | head -20
```

and mirror every one of them. A missed accessor is a `MethodError` at first use,
which is loud — but only if a test reaches it, which Task 13 ensures.

- [ ] **Step 6: Run to verify**

```bash
$MPIEXEC -n 2 $JULIA --project=. test/test_transposable_field.jl
$JULIA --project=. -e 'using Pkg; Pkg.test()'
```

Expected: pass. Watch specifically for `test_field_typestability.jl` and
`test_type_stability.jl` — a new storage union can cost inference. If either
regresses, the fix is to parametrize rather than widen; do not loosen the ratchet.

- [ ] **Step 7: Stage**

```bash
git add src/core/field/field_types.jl src/core/transposable_field.jl \
        test/test_transposable_field.jl
```

---

## Task 12: Dispatch `forward_transform!` on transposable storage

**Files:**
- Modify: `src/core/transforms/transform_gpu.jl:301-315, 358-375`
- Modify: `src/core/transforms/transform_fourier.jl:128-145`

**Interfaces:**
- Consumes: `transpose_workspace!` (Task 10), `is_transposable_storage` (Task 11).
- Produces: `forward_transform!(field)` and `backward_transform!(field)` transform distributed GPU fields instead of erroring.

- [ ] **Step 1: Add the forward dispatch**

In `src/core/transforms/transform_gpu.jl`, immediately after
`ensure_layout!(field, :g)` in `forward_transform!` (line 308), insert:

```julia
    # A distributed GPU field transforms by explicit transposes. PencilFFTs is
    # CPU-only and the local transform chain would compute a per-rank FFT of a
    # slab, which is silently wrong rather than an error.
    if is_transposable_storage(field)
        ws = transpose_workspace!(field.dist, field)
        distributed_forward_transform!(ws)
        return
    end
```

`distributed_forward_transform!` sets `field.current_layout = :c` itself, so this
branch must NOT set it again.

- [ ] **Step 2: Add the backward dispatch**

In `src/core/transforms/transform_fourier.jl`, in `backward_transform!`,
immediately after its `ensure_layout!(field, :c)` call, insert the mirror:

```julia
    if is_transposable_storage(field)
        ws = transpose_workspace!(field.dist, field)
        distributed_backward_transform!(ws)
        return
    end
```

- [ ] **Step 3: Delete the two refusals**

At `src/core/transforms/transform_gpu.jl:358-366`, delete the `is_gpu_array` arm:

```julia
        if is_gpu_array(get_grid_data(field))
            error("Cannot run local GPU transforms on distributed data without TransposableField. " * ...)
        else
```

leaving the CPU arm as an unconditional `error(...)`. Do the same at
`src/core/transforms/transform_fourier.jl:132-140` for `get_coeff_data`.

A distributed GPU field can no longer reach this point — it returned at Step 1 or
Step 2 — so the arm is now unreachable text, and unreachable text rots.

- [ ] **Step 4: Keep the basis-level refusals**

Do **not** touch `validate_mpi_fourier_only` (`src/core/basis/basis_core.jl:152-190`).
Its RealFourier and non-Fourier refusals state real constraints: the transpose
buffers are fixed-shape and cannot hold a half spectrum. Verify they still fire:

```bash
$MPIEXEC -n 2 $JULIA --project=. -e '
using Tarang, MPI
MPI.Initialized() || MPI.Init()
c = CartesianCoordinates("x","y")
d = Distributor(c; comm=MPI.COMM_WORLD, mesh=(2,), dtype=Float64,
                architecture=CPU(), use_pencil_arrays=false)
try
    ScalarField(d, "rf", (RealFourier(c["x"]; size=8, bounds=(0.0,2π)),
                          RealFourier(c["y"]; size=6, bounds=(0.0,2π))))
    println("NOT REFUSED — REGRESSION")
catch e
    println(occursin("RealFourier", sprint(showerror, e)) ? "refused correctly" : "wrong message")
end'
```

Expected: `refused correctly` on rank 0.

- [ ] **Step 5: Run the suites**

```bash
$MPIEXEC -n 4 $JULIA --project=. test/test_transposable_field.jl
$MPIEXEC -n 4 $JULIA --project=. test/test_mpi_transposable_parity.jl
$JULIA --project=. test/run_mpi_ci.jl 2
$JULIA --project=. test/run_mpi_ci.jl 4
$JULIA --project=. -e 'using Pkg; Pkg.test()'
```

Expected: `54 passed, 0 failed` twice; `Testing Tarang tests passed`.

- [ ] **Step 6: Stage**

```bash
git add src/core/transforms/transform_gpu.jl src/core/transforms/transform_fourier.jl
```

---

## Task 13: Prove the distributed path runs on device arrays without a GPU, and un-vacuum the serial testset

Two loose ends. First, nothing yet proves the Task 12 dispatch fires for an actual
device-backed field — the CPU tests take the `SerialFieldStorage` branch. JLArray
gives a device array type with no hardware (the `test_gpu_implicit_guard_jlarray.jl`
idiom). Second, the NPROCS==1 testset *"Forward transform matches the regular
serial transform"* became vacuous when the `dist.size==1` short-circuit landed —
it now compares `forward_transform!` against `forward_transform!`.

**Files:**
- Modify: `test/test_transposable_field.jl:302-312`
- Modify: `test/test_mpi_transposable_parity.jl`

**Interfaces:**
- Consumes: everything from Tasks 10-12.
- Produces: no source changes; test coverage only.

- [ ] **Step 1: Replace the vacuous serial testset**

In `test/test_transposable_field.jl`, replace the whole
`@testset "Forward transform matches the regular serial transform"` block with:

```julia
    @testset "Serial short-circuit delegates to the field's own transform" begin
        # At one rank there is no transpose to perform and the field's regular
        # transform path owns the basis-specific shapes (a RealFourier half
        # spectrum has no representation in the fixed-shape transpose buffers).
        # This asserts the DELEGATION — that the wrapper leaves the field
        # authoritative — which is all the serial path claims. The claim that the
        # distributed path reproduces serial COEFFICIENTS is an np>1 statement and
        # lives in test_mpi_transposable_parity.jl.
        field = serial_transform_field("transform_forward")
        reference = serial_transform_field("transform_forward_reference")
        tf = TransposableField(field)
        forward_transform!(reference)
        distributed_forward_transform!(tf)
        @test field.current_layout == :c
        @test field["c"] ≈ reference["c"]
        @test Tarang.current_data(tf) === Tarang.get_coeff_data(field)
    end
```

- [ ] **Step 2: Add the JLArray device-path test**

Append to `test/test_mpi_transposable_parity.jl`:

```julia
# A device-array field must take the distributed path through the ORDINARY
# forward_transform! call, not the local transform chain (which would FFT a
# per-rank slab and be silently wrong) and not an error. JLArray provides a
# non-Array device type with no GPU present.
const _JL_OK = try
    @eval using JLArrays
    true
catch err
    @info "JLArrays unavailable; skipping device-array dispatch test" err
    false
end

if _JL_OK && NP > 1
    @testset "distributed device field transforms through forward_transform!" begin
        Nx, Ny = 8, 6
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; comm=COMM, mesh=(NP, 1), dtype=ComplexF64,
                           architecture=CPU(), use_pencil_arrays=false)
        bases = (ComplexFourier(coords, "x", Nx), ComplexFourier(coords, "y", Ny))
        field = ScalarField(dist, "device_dispatch", bases)

        ox, oy = block_ranges((NP, 1), Nx, Ny)
        host = Array{ComplexF64}(undef, length(ox), length(oy))
        for (jl, jg) in enumerate(oy), (il, ig) in enumerate(ox)
            host[il, jl] = complex(f2(ig, jg, Nx, Ny), 0.0)
        end
        Tarang.set_grid_data!(field, JLArray(host))
        @test Tarang.is_transposable_storage(field) == false  # CPU arch, CPU storage

        forward_transform!(field)
        @test field.current_layout == :c
        reference = serial_coeffs_2d(Nx, Ny)
        @test maximum(abs, Array(Tarang.get_coeff_data(field)) .- reference[ox, oy];
                      init=0.0) < 1e-10
    end
end
```

If storage selection keys on `is_gpu(dist.architecture)` rather than on the array
type, this test documents that a JLArray on a CPU distributor stays CPU storage.
Should the executor find that Task 11's predicate should key on the *array* type
instead, change the `@test` to `== true` and record the decision — but do not
change the predicate without saying so, because it decides which fields pay for a
workspace.

- [ ] **Step 3: Run**

```bash
$JULIA --project=. -e 'using Pkg; Pkg.test()'   # JLArrays resolves only under Pkg.test
$MPIEXEC -n 2 $JULIA --project=. test/test_transposable_field.jl
$MPIEXEC -n 4 $JULIA --project=. test/test_mpi_transposable_parity.jl
```

JLArrays and GPUArrays are in `Project.toml`'s test target only — they do **not**
resolve under `--project=.`, so the JLArray testset self-skips outside `Pkg.test()`.
That is expected, not a failure.

- [ ] **Step 4: Final full verification**

```bash
$JULIA --project=. test/run_mpi_ci.jl 2
$JULIA --project=. test/run_mpi_ci.jl 4
$JULIA --project=. -e 'using Pkg; Pkg.test()'
```

Expected: `54 passed, 0 failed` at np=2, `55 passed, 0 failed` at np=4 (the new
parity file raises the count by one), and `Testing Tarang tests passed`.

- [ ] **Step 5: Stage**

```bash
git add test/test_transposable_field.jl test/test_mpi_transposable_parity.jl
```

---

## Done criteria

- `decomposed_axes` is the only place either convention is stated; the ratchet in
  Task 7 enforces it.
- `local_indices`, `get_local_array_size`, and `get_local_range` provably agree.
- `mesh=(2,2)` on a 2-D domain refuses at plan time with the real reason and a
  working alternative, and never mentions the PencilFFTs installation.
- A distributed GPU field transforms through `forward_transform!` / `backward_transform!`.
- Transpose workspaces are shared per shape and released by `close(dist)`.
- MPI CI green at np=2 and np=4; `Pkg.test()` green.
- **Not claimed:** that a GPU+MPI IVP steps correctly end to end. Per-mode
  gather/scatter and matrix assembly are untouched by this plan.
