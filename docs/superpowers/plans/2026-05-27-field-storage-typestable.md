# Type-Stable Field Storage Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `ScalarField` grid/coeff data access type-stable at the source by parametrizing field storage on concrete array types, eliminating the `Union{Nothing, AbstractArray}` abstraction and the hand-written function barriers that work around it.

**Architecture:** Three sequential phases, each leaving the full test suite green. Phase 1 removes the `nothing` half of the storage union (eager allocation + typed empty sentinel for 0-D fields). Phase 2 makes a field's array *types* stable for its lifetime by removing the 4 in-place array-type-swap sites (architecture moves), the precondition that makes parametrization legal. Phase 3 parametrizes `SerialFieldStorage{G,C}`, propagates the parameters to `TransposableFieldStorage` and the 12 annotation sites, adds an inference-regression test, and deletes the now-dead barriers. The ~725 *read* sites (`get_grid_data`/`get_coeff_data`/`get_local_data`) need no edits — they become concrete automatically once the accessors return a concrete type.

**Tech Stack:** Julia 1.12, FFTW, PencilFFTs (MPI), CUDA (GPU), MPI.jl, `Test`, `InteractiveUtils`.

**Environment / commands used throughout this plan:**
- `JULIA` = `~/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia` (the `julia`/`juliaup` launcher is broken in this environment — always use this absolute path).
- Run one test file: `$JULIA --project=. -e 'include("test/test_NAME.jl")'`
- Package load smoke test: `$JULIA --project=. -e 'using Tarang; println("LOAD OK")'`
- 2-rank MPI test: prefix `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/opt/hwloc/lib` then the MPICH mpiexec then `-n 2 $JULIA --project=. test/test_NAME.jl` (Homebrew's OpenMPI launcher is incompatible). Get the MPICH mpiexec path via `$JULIA --project=. -e 'using MPI; println(MPI.mpiexec())'`.
- **No commits without explicit user instruction** (project rule). The `git commit` steps below are written for completeness; the executing agent MUST pause and ask before running any of them.

**Key facts verified before writing this plan:**
- `SerialFieldStorage` (src/core/field/field_types.jl:52) has `grid::Union{Nothing,AbstractArray}`, `coeff::Union{Nothing,AbstractArray}`; constructed as `new(arch, nothing, nothing)`.
- 0-D "tau" fields have empty `bases`; `allocate_data!` early-returns for them, so they keep `nothing` forever. Every consumer already guards them with `isempty(field.bases)`.
- Accessors `get_grid_data`/`get_coeff_data` (src/core/field/field_data/field_data_copy_alloc.jl:203,214) are one-line `getfield(getfield(field,:storage),:grid/:coeff)`.
- Exactly 4 array-type-changing `set_*_data!` sites: `synchronize_field_architecture!` (copy_alloc.jl:83,88) and pencil setup (nonlinear_pencil_utils.jl:285-286,322-323). All other ~65 set-sites preserve element/array type.
- `SpectralLinearOperator{T,N,A<:AbstractArray{T,N}}` (src/core/timesteppers/spectral_operators.jl:37) is an in-codebase proof the parametrization pattern works.
- `TransposableFieldStorage{CT,N}` (src/core/transposable_field.jl:90) embeds `base::SerialFieldStorage` — must absorb the new parameters.
- 12 `ScalarField{...}` annotation sites total (grep `ScalarField{` in src/).

---

## File Structure

- `src/core/field/field_types.jl` — `SerialFieldStorage` struct + `ScalarField` inner constructors. **Core of the refactor.**
- `src/core/field/field_data/field_data_copy_alloc.jl` — `allocate_data!`, accessors, `copy`, `deepcopy_internal`, `synchronize_field_architecture!`. **Construction-order inversion + sentinel.**
- `src/core/transposable_field.jl` — `TransposableFieldStorage` carries the new parameters via its `base` field.
- `src/core/nonlinear/nonlinear_pencil_utils.jl` — 2 of the 4 type-swap sites (pencil setup).
- `src/core/timesteppers/state.jl`, `src/core/timesteppers/state_utils.jl`, `src/core/solvers/lazy_rhs.jl` — `ScalarField{<:Any, SerialFieldStorage}` dispatch annotations.
- `test/test_field_typestability.jl` — **new** regression test (inference + tau-field). The TDD anchor.
- Barrier-deletion touches (Phase 3 Task 11): `src/core/solvers/lazy_rhs.jl` (+ enumerated siblings).

---

## PHASE 1 — Eliminate `nothing` from field storage

Goal: `SerialFieldStorage.grid`/`.coeff` are always concrete arrays. 0-D tau fields hold a length-0 typed array. Narrows the union to `AbstractArray`. Suite stays green.

### Task 1: Add the regression test file

**Files:**
- Create: `test/test_field_typestability.jl`

- [ ] **Step 1: Write the test file** — only the Phase-1 set must pass now; later sets are skipped until their phase.

```julia
using Test
using Tarang
using InteractiveUtils

@testset "Field storage type stability" begin
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; dtype=Float64)
    xb = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
    yb = ComplexFourier(coords["y"]; size=16, bounds=(0.0, 2π))

    @testset "Phase 1: no nothing in storage" begin
        u = ScalarField(dist, "u", (xb, yb), Float64)
        ensure_layout!(u, :g)
        @test Tarang.get_grid_data(u) !== nothing
        @test Tarang.get_coeff_data(u) !== nothing

        tau = ScalarField(dist, "tau", (), Float64)
        @test Tarang.get_grid_data(tau) !== nothing
        @test length(Tarang.get_grid_data(tau)) == 0
        @test Tarang.get_coeff_data(tau) !== nothing
        @test length(Tarang.get_coeff_data(tau)) == 0
    end

    @testset "Phase 3: get_grid_data is type-stable" begin
        u = ScalarField(dist, "u", (xb, yb), Float64)
        ensure_layout!(u, :g)
        @test_skip (@inferred Tarang.get_grid_data(u); true)
        @test_skip (@inferred Tarang.get_coeff_data(u); true)
    end
end
```

- [ ] **Step 2: Run; the tau asserts must FAIL now**

Run: `$JULIA --project=. -e 'include("test/test_field_typestability.jl")'`
Expected: FAIL — `get_grid_data(tau)` returns `nothing` today, so `length(nothing)` errors. (The two `u` asserts already pass.)

- [ ] **Step 3: Commit** (ask user first)

```
git add test/test_field_typestability.jl && git commit -m "test: add field storage type-stability regression test"
```

### Task 2: Typed empty sentinel + eager allocation for 0-D fields

**Files:**
- Modify: `src/core/field/field_data/field_data_copy_alloc.jl`
- Modify: `src/core/field/field_types.jl`

- [ ] **Step 1: Add sentinel helpers** near the other allocation helpers in `field_data_copy_alloc.jl`:

```julia
# Typed length-0 placeholder so storage is never `nothing`. Grid uses the field
# element type; coeff uses the complex coefficient type. The 1-D length-0 arrays
# are never indexed (every 0-D-field consumer guards with isempty(field.bases)).
_empty_grid(::Type{T}) where {T} = Array{T,1}(undef, 0)
_empty_coeff(::Type{T}) where {T} = Array{coefficient_eltype(T),1}(undef, 0)
```

- [ ] **Step 2: Install the sentinel for 0-D fields** in the primary `ScalarField` inner constructor (field_types.jl). Replace:

```julia
        if domain !== nothing
            allocate_data!(field)
        end
        return field
```

with:

```julia
        if domain !== nothing
            allocate_data!(field)
        else
            set_grid_data!(field, _empty_grid(T))
            set_coeff_data!(field, _empty_coeff(T))
        end
        return field
```

Apply the same `else` sentinel install in the second inner constructor (field_types.jl:122) for its `domain === nothing` case.

- [ ] **Step 3: Run the regression test; Phase-1 set must PASS**

Run: `$JULIA --project=. -e 'include("test/test_field_typestability.jl")'`
Expected: PASS for "Phase 1: no nothing in storage".

- [ ] **Step 4: Field + arithmetic regression**

Run: `$JULIA --project=. -e 'using Tarang; for f in ["test/test_arithmetic.jl","test/test_boundary_conditions.jl","test/test_convenience_api.jl"]; include(f); end'`
Expected: all PASS.

- [ ] **Step 5: Commit** (ask user first)

```
git add src/core/field/field_types.jl src/core/field/field_data/field_data_copy_alloc.jl && git commit -m "refactor: install typed length-0 sentinels for 0-D field storage"
```

### Task 3: Narrow the storage field declarations to `AbstractArray`

**Files:**
- Modify: `src/core/field/field_types.jl:54-59`

- [ ] **Step 1: Audit the now-affected guards** (no edit). Code that skipped tau fields via `data === nothing` must instead use `isempty(field.bases)`; a sentinel is `!== nothing`.

Run: `grep -rn "get_grid_data(.*) === nothing\|get_coeff_data(.*) === nothing" src/`
Expected: ~30 sites. **Leave them** — they stay correct (a length-0 sentinel is `!== nothing`, so those branches simply no longer fire for tau fields, which is the intended behavior since tau fields are guarded earlier by `isempty(bases)`). Only narrow the type in Step 2.

- [ ] **Step 2: Narrow the field types and the zero-arg constructor.** Change:

```julia
    grid::Union{Nothing, AbstractArray}
    coeff::Union{Nothing, AbstractArray}

    function SerialFieldStorage(arch::AbstractArchitecture)
        new(arch, nothing, nothing)
    end
```

to:

```julia
    grid::AbstractArray
    coeff::AbstractArray

    function SerialFieldStorage(arch::AbstractArchitecture)
        # Transient empty sentinels; the ScalarField constructor overwrites these
        # via set_grid_data!/set_coeff_data! before the field is observable.
        new(arch, Array{Float64,1}(undef, 0), Array{ComplexF64,1}(undef, 0))
    end
```

- [ ] **Step 3: Smoke-load + regression test**

Run: `$JULIA --project=. -e 'using Tarang; println("LOAD OK")'`
Run: `$JULIA --project=. -e 'include("test/test_field_typestability.jl")'`
Expected: LOAD OK; Phase-1 set PASS.

- [ ] **Step 4: Broad serial regression**

Run: `$JULIA --project=. -e 'using Tarang; for f in ["test/test_solvers.jl","test/test_lazy_rhs_fourier.jl","test/test_nonlinear.jl","test/test_chebyshev.jl","test/test_diagonal_imex.jl"]; include(f); end'`
Expected: all PASS.

- [ ] **Step 5: Commit** (ask user first)

```
git add src/core/field/field_types.jl && git commit -m "refactor: narrow SerialFieldStorage data fields to AbstractArray (no nothing)"
```

---

## PHASE 2 — Make a field's array types stable for its lifetime

Goal: remove the 4 sites that replace a field's array with a *different array type* (CPU<->GPU). After this a field's grid/coeff concrete types never change post-construction. A field is born on its final architecture.

### Task 4: Architecture-fix fields; make `synchronize_field_architecture!` an assertion

**Files:**
- Modify: `src/core/field/field_data/field_data_copy_alloc.jl:79-93`

- [ ] **Step 1: Append a Phase-2 test** inside the outer testset in `test/test_field_typestability.jl`:

```julia
    @testset "Phase 2: field array type fixed at construction" begin
        u = ScalarField(dist, "u", (xb, yb), Float64)
        ensure_layout!(u, :g)
        gtype = typeof(Tarang.get_grid_data(u))
        Tarang.synchronize_field_architecture!(u; arch=dist.architecture)
        @test typeof(Tarang.get_grid_data(u)) === gtype
    end
```

- [ ] **Step 2: Run it**

Run: `$JULIA --project=. -e 'include("test/test_field_typestability.jl")'`
Expected: PASS on CPU (same-arch is a no-op today). This locks the invariant before Step 3.

- [ ] **Step 3: Restrict the function to same-architecture.** Replace the whole body of `synchronize_field_architecture!`:

```julia
function synchronize_field_architecture!(field::ScalarField; arch::AbstractArchitecture=field.dist.architecture,
                                          move_grid::Bool=true, move_coefficients::Bool=true)
    field.buffers.architecture == arch ||
        throw(ArgumentError("synchronize_field_architecture!: in-place architecture moves are no longer supported " *
                            "(field on $(field.buffers.architecture), requested $arch). Construct the field on the target architecture."))
    return field
end
```

- [ ] **Step 4: Audit callers**

Run: `grep -rn "synchronize_field_architecture!" src/`
Expected: a list. Confirm each passes `arch == field.dist.architecture`. (Verified at plan time: primary callers pass the field's own dist architecture.) If any genuinely moves architecture, change it to construct the field on the target architecture up front.

- [ ] **Step 5: Architecture + solver regression**

Run: `$JULIA --project=. -e 'using Tarang; for f in ["test/test_architectures.jl","test/test_field_typestability.jl","test/test_solvers.jl"]; include(f); end'`
Expected: all PASS. (Ignore `test/test_cpu_architecture.jl` — pre-existing unrelated `KernelAbstractions not defined` failure.)

- [ ] **Step 6: Commit** (ask user first)

```
git add src/core/field/field_data/field_data_copy_alloc.jl test/test_field_typestability.jl && git commit -m "refactor: make ScalarField architecture-fixed; forbid in-place arch swaps"
```

### Task 5: Pencil setup builds arrays on the field's architecture

**Files:**
- Modify: `src/core/nonlinear/nonlinear_pencil_utils.jl:283-323`

- [ ] **Step 1: Inspect the 4 lines**

Run: `sed -n '280,325p' src/core/nonlinear/nonlinear_pencil_utils.jl`
Expected: the `set_grid_data!(field, create_array(arch, ...))` / `set_coeff_data!` calls.

- [ ] **Step 2: Add an architecture-match assertion** immediately before each `set_*_data!` pair (both functions):

```julia
            arch == field.dist.architecture ||
                throw(ArgumentError("pencil setup: array architecture $arch must match field architecture $(field.dist.architecture)"))
```

This is behavior-preserving for valid inputs and makes the invariant explicit. (Same-arch, same-eltype shape changes remain allowed — `Array{T,N}` is one type regardless of size, compatible with Phase 3's `{G,C}`.)

- [ ] **Step 3: Smoke-load + MPI dealiasing test**

Run: `$JULIA --project=. -e 'using Tarang; println("LOAD OK")'`
Run (2 ranks): `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/opt/hwloc/lib $MPIEXEC -n 2 $JULIA --project=. test/test_mpi_dealiasing_product.jl`
Expected: LOAD OK; "MPI dealiasing product tests completed" all Pass.

- [ ] **Step 4: Commit** (ask user first)

```
git add src/core/nonlinear/nonlinear_pencil_utils.jl && git commit -m "refactor: assert architecture invariant in pencil-compatible data setup"
```

---

## PHASE 3 — Parametrize storage on concrete array types

Goal: `SerialFieldStorage{G,C}` carries the concrete grid/coeff array types; accessors return concrete types; inference test passes; dead barriers deleted.

### Task 6: Value-returning allocator (invert construction order)

**Files:**
- Modify: `src/core/field/field_data/field_data_copy_alloc.jl`

- [ ] **Step 1: Add `_build_field_arrays`** (the body of `allocate_data!` refactored to return `(grid, coeff)` without a field):

```julia
function _build_field_arrays(dist::Distributor, domain::Domain, dtype::Type{T}) where {T}
    gshape = global_shape(domain)
    cshape = get_coefficient_shape_for_context(domain, dist)
    arch = dist.architecture
    coeff_dtype = coefficient_eltype(T)
    if dist.use_pencil_arrays
        pencil_plan = _find_pencil_plan(dist)
        if pencil_plan !== nothing
            g = PencilFFTs.allocate_input(pencil_plan); fill!(g, zero(T))
            c = PencilFFTs.allocate_output(pencil_plan); fill!(c, zero(coeff_dtype))
            return (g, c)
        elseif dist.pencil_fft_input !== nothing && dist.pencil_fft_output !== nothing
            g = PencilArrays.PencilArray{T}(undef, dist.pencil_fft_input); fill!(g, zero(T))
            c = PencilArrays.PencilArray{coeff_dtype}(undef, dist.pencil_fft_output); fill!(c, zero(coeff_dtype))
            return (g, c)
        else
            g = create_pencil(dist, gshape, nothing, dtype=T); fill!(g, zero(T))
            c = create_pencil(dist, cshape, nothing, dtype=coeff_dtype); fill!(c, zero(coeff_dtype))
            return (g, c)
        end
    else
        local_gsize = get_local_array_size(dist, gshape)
        local_csize = get_local_array_size(dist, cshape)
        return (zeros(arch, T, local_gsize...), zeros(arch, coeff_dtype, local_csize...))
    end
end
```

- [ ] **Step 2: Reimplement `allocate_data!` on top of it** (keep the public signature):

```julia
function allocate_data!(field::ScalarField)
    field.domain === nothing && return
    g, c = _build_field_arrays(field.dist, field.domain, field.dtype)
    set_grid_data!(field, g)
    set_coeff_data!(field, c)
    field.buffers.architecture = field.dist.architecture
end
```

- [ ] **Step 3: Regression**

Run: `$JULIA --project=. -e 'using Tarang; for f in ["test/test_field_typestability.jl","test/test_solvers.jl","test/test_arithmetic.jl"]; include(f); end'`
Expected: all PASS (pure refactor).

- [ ] **Step 4: Commit** (ask user first)

```
git add src/core/field/field_data/field_data_copy_alloc.jl && git commit -m "refactor: extract _build_field_arrays value-returning allocator"
```

### Task 7: Parametrize `SerialFieldStorage{G,C}`

**Files:**
- Modify: `src/core/field/field_types.jl`

- [ ] **Step 1: Rewrite the struct + constructor:**

```julia
mutable struct SerialFieldStorage{G<:AbstractArray, C<:AbstractArray} <: AbstractFieldStorage
    architecture::AbstractArchitecture
    grid::G
    coeff::C
end

SerialFieldStorage(arch::AbstractArchitecture, grid::G, coeff::C) where {G<:AbstractArray, C<:AbstractArray} =
    SerialFieldStorage{G,C}(arch, grid, coeff)
```

Remove the old zero-arg `SerialFieldStorage(arch)` constructor.

- [ ] **Step 2: Rewrite the primary `ScalarField` inner constructor** to build arrays first:

```julia
    function ScalarField(dist::Distributor, name::String="field", bases::Tuple{Vararg{Basis}}=(),
                         dtype::Type{T}=dist.dtype) where T
        domain = length(bases) > 0 ? Domain(dist, bases) : nothing
        layout = length(bases) > 0 ? get_layout(dist, bases, dtype) : nothing
        initial_scales = length(bases) > 0 ? ntuple(_ -> 1.0, dist.dim) : nothing
        g, c = domain !== nothing ? _build_field_arrays(dist, domain, T) : (_empty_grid(T), _empty_coeff(T))
        storage = SerialFieldStorage(dist.architecture, g, c)
        return new{T, typeof(storage)}(dist, name, bases, domain, dtype, storage, layout, :g, initial_scales, :auto, false, 0)
    end
```

- [ ] **Step 3: Second inner constructor** (field_types.jl:122) already receives a built `storage::S`; confirm it no longer calls the removed zero-arg constructor. If it builds storage internally, mirror Step 2.

- [ ] **Step 4: Update component element types.** For `VectorField`, replace the `ScalarField{T, SerialFieldStorage}[]` build:

```julia
        components = ScalarField[]
        for (i, coord_name) in enumerate(coordsys.names)
            push!(components, ScalarField(dist, "$(name)_$coord_name", bases, dtype))
        end
        comps = identity.(components)   # narrows abstract ScalarField[] to Vector{ScalarField{T,S}}
```

and pass `comps` to `new`. For `TensorField`, build the matrix then narrow: `comps = identity.(components_matrix)`. (All components share bases/dtype/dist, hence one concrete `S`.)

- [ ] **Step 5: Update `storage_mode`** (field_types.jl:237):

```julia
storage_mode(::ScalarField{T, <:SerialFieldStorage}) where T = SerialStorage()
```

- [ ] **Step 6: Smoke-load**

Run: `$JULIA --project=. -e 'using Tarang; println("LOAD OK")'`
Expected: LOAD OK. (If `copy`/`deepcopy_internal` break on the removed zero-arg constructor, route them through the normal `ScalarField(...)` constructor + `copyto!` the live array — they already `copy(get_*_data(field))`, producing same-typed arrays.)

- [ ] **Step 7: Commit** (ask user first)

```
git add src/core/field/field_types.jl && git commit -m "refactor: parametrize SerialFieldStorage{G,C} on concrete array types"
```

### Task 8: Propagate parameters through `TransposableFieldStorage`

**Files:**
- Modify: `src/core/transposable_field.jl`

- [ ] **Step 1: Constrain the embedded `base`.** Change:

```julia
mutable struct TransposableFieldStorage{CT, N} <: AbstractFieldStorage
    base::SerialFieldStorage
```

to:

```julia
mutable struct TransposableFieldStorage{CT, N, B<:SerialFieldStorage} <: AbstractFieldStorage
    base::B
```

Add `B` to every `TransposableFieldStorage{...}` construction in the file (find with `grep -n "TransposableFieldStorage{" src/core/transposable_field.jl`), inferring `B = typeof(base)`.

- [ ] **Step 2: Loosen annotations.** Any `ScalarField{T, TransposableFieldStorage}` becomes `ScalarField{T, <:TransposableFieldStorage}`. (`storage_mode(::ScalarField{T, <:TransposableFieldStorage})` already uses `<:` — no change.)

- [ ] **Step 3: Smoke-load + GPU-on-CPU solver test**

Run: `$JULIA --project=. -e 'using Tarang; println("LOAD OK")'`
Run: `$JULIA --project=. -e 'include("test/test_gpu_solver_cpu.jl")'`
Expected: LOAD OK; PASS.

- [ ] **Step 4: Commit** (ask user first)

```
git add src/core/transposable_field.jl && git commit -m "refactor: thread storage type param through TransposableFieldStorage"
```

### Task 9: Loosen remaining storage-type dispatch annotations

**Files:**
- Modify: `src/core/solvers/lazy_rhs.jl:676`, `src/core/timesteppers/state_utils.jl:611,630`, `src/core/timesteppers/state.jl:42`

- [ ] **Step 1: Enumerate**

Run: `grep -rn "SerialFieldStorage" src/ | grep "ScalarField{"`
Expected: the dispatch sites above.

- [ ] **Step 2: Loosen each** — replace `SerialFieldStorage` with `<:SerialFieldStorage` inside the `ScalarField{...}` constraint. Example (lazy_rhs.jl:676):

```julia
_lazy_plan_field_type(::Type{F}) where {F<:ScalarField{<:Any, <:SerialFieldStorage}} =
    isconcretetype(F) ? F : ScalarField
```

Apply identically at state_utils.jl:611, state_utils.jl:630, state.jl:42.

- [ ] **Step 3: Timestepper regression**

Run: `$JULIA --project=. -e 'using Tarang; for f in ["test/test_solvers.jl","test/test_diagonal_imex.jl","test/test_lazy_rhs_fourier.jl"]; include(f); end'`
Expected: all PASS.

- [ ] **Step 4: Commit** (ask user first)

```
git add src/core/solvers/lazy_rhs.jl src/core/timesteppers/state_utils.jl src/core/timesteppers/state.jl && git commit -m "refactor: loosen ScalarField storage-type dispatch constraints"
```

### Task 10: Turn on the inference regression test

**Files:**
- Modify: `test/test_field_typestability.jl`

- [ ] **Step 1: Un-skip the Phase 3 set:**

```julia
    @testset "Phase 3: get_grid_data is type-stable" begin
        u = ScalarField(dist, "u", (xb, yb), Float64)
        ensure_layout!(u, :g)
        @test (@inferred Tarang.get_grid_data(u); true)
        @test (@inferred Tarang.get_coeff_data(u); true)
    end
```

- [ ] **Step 2: Run it**

Run: `$JULIA --project=. -e 'include("test/test_field_typestability.jl")'`
Expected: PASS — `get_grid_data` now infers a concrete array type. If it still fails, inspect with `$JULIA --project=. -e 'using Tarang, InteractiveUtils; <build u>; @code_warntype Tarang.get_grid_data(u)'` and fix the offending accessor (likely a `getproperty` widening).

- [ ] **Step 3: Commit** (ask user first)

```
git add test/test_field_typestability.jl && git commit -m "test: assert type-stable field data access"
```

### Task 11: Delete the now-dead field-data function barriers

**Files (enumerate, then edit):**
- `src/core/solvers/lazy_rhs.jl` (+ siblings found by grep)

- [ ] **Step 1: Enumerate barrier candidates** (only those whose sole purpose was recovering a concrete type from a field-data accessor):

Run: `grep -rn "Function barrier\|arrives .Any.-typed\|Any.-typed from .get" src/`
Expected: comment-tagged barriers, e.g. `_lazy_scale_along_axis!`, `_matmul_axis_into!` (both lazy_rhs.jl).

- [ ] **Step 2: Inline and delete each, worked example** — `_lazy_scale_along_axis!`. Replace the call:

```julia
    mult_shape = ntuple(i -> i == axis ? length(deriv_mult) : 1, ndims(data))
    _lazy_scale_along_axis!(data, reshape(deriv_mult, mult_shape...))
    return coeff_storage
```

with the inlined body (now type-stable because `data` is concrete):

```julia
    mult_shape = ntuple(i -> i == axis ? length(deriv_mult) : 1, ndims(data))
    data .*= reshape(deriv_mult, mult_shape...)
    return coeff_storage
```

Delete the `_lazy_scale_along_axis!` definition. Apply the same inline-and-delete to `_matmul_axis_into!` (fold into `_apply_1d_matrix!`). **DO NOT delete** `_rk_ldiv!`, `_ddi_sbdf1_update!`, `_ddi_sbdf2_update!`, `_copy_convert_into!` — those recover types from `Dict{,Any}` caches or the abstract `ws.arch`, which this refactor does not touch (they belong to the separate "typed caches" workstream).

- [ ] **Step 3: Per-deletion test + allocation guard**

Run: `$JULIA --project=. -e 'include("test/test_lazy_rhs_fourier.jl")'`
Expected: PASS.

Allocation guard (inlined broadcast must not regress past the prior 96 bytes):
`$JULIA --project=. -e 'using Tarang; coords=CartesianCoordinates("x","y"); dist=Distributor(coords;dtype=Float64); xb=RealFourier(coords["x"];size=64,bounds=(0.0,2π)); yb=ComplexFourier(coords["y"];size=64,bounds=(0.0,2π)); u=ScalarField(dist,"u",(xb,yb),Float64); ensure_layout!(u,:c); cs=Tarang.get_coeff_data(u); Tarang._apply_lazy_fourier_diff!(cs,u,xb,1,2); println("alloc: ", @allocated Tarang._apply_lazy_fourier_diff!(cs,u,xb,1,2))'`
Expected: alloc ≤ 96 bytes.

- [ ] **Step 4: Commit** (ask user first)

```
git add src/core/solvers/lazy_rhs.jl && git commit -m "refactor: inline field-data barriers made redundant by type-stable storage"
```

### Task 12: Full regression — serial + MPI

**Files:** none (verification)

- [ ] **Step 1: Full serial suite** (excluding known-broken `test_aqua.jl`, `test_cpu_architecture.jl`, `test_compatibility.jl`):

Run: `$JULIA --project=. -e 'using Tarang; for f in filter(f->endswith(f,".jl") && !any(b->occursin(b,f), ["aqua","mpi","cpu_architecture","compatibility"]), readdir("test",join=true)); try; include(f); catch e; println("FAIL ",basename(f),": ",sprint(showerror,e)); end; end'`
Expected: no `FAIL` lines.

- [ ] **Step 2: MPI tests at 2 ranks**

Run: get `MPIEXEC=$($JULIA --project=. -e 'using MPI; print(MPI.mpiexec())')`, then for `test_mpi_lazy_rhs_fourier.jl` and `test_mpi_dealiasing_product.jl`:
`DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/opt/hwloc/lib $MPIEXEC -n 2 $JULIA --project=. test/<file>`
Expected: each prints its "tests completed" banner, all Pass.

- [ ] **Step 3: Final inference confirmation**

Run: `$JULIA --project=. -e 'include("test/test_field_typestability.jl")'`
Expected: all PASS including Phase 3 inference asserts.

- [ ] **Step 4: Commit** (ask user first)

```
git commit --allow-empty -m "test: full serial+MPI regression green after type-stable storage refactor"
```

---

## Self-Review

**Spec coverage** (vs the field-storage investigation findings):
- "Eliminate `nothing`" -> Phase 1 (Tasks 1-3). ✓
- "Construction-order inversion (`copy`/`deepcopy`/`allocate_data!` mutate-after-construct)" -> Task 6 (value-returning allocator) + Task 7 (array-first construction). `copy`/`deepcopy_internal` already do `copy(get_*_data(field))` producing same-typed arrays; Task 7 Step 6 verifies, with the explicit fallback to route them through `_build_field_arrays` if they break. ✓
- "PencilArray/CuArray type explosion" -> mitigated by the `G<:AbstractArray` bound (Julia infers the concrete type per field; no exhaustive enumeration) plus architecture-fixed fields (Phase 2). Risk noted in header. ✓
- "Sentinel semantics drift" (the `=== nothing` guards) -> Task 3 Step 1 explicitly leaves guards in place; only the declared type narrows. ✓
- "Delete ~100 barriers" -> Task 11 deletes only field-data-fed barriers and explicitly preserves the `Dict{,Any}`/`ws.arch`-fed ones (separate workstream). Honest scope. ✓

**Placeholder scan:** No "TBD"/"handle edge cases"/"similar to Task N". Task 11 enumerates by grep + gives a complete worked example + an explicit keep-list rather than pre-listing every deletion — they are mechanically identical and discovered by grep, and pre-enumerating code that depends on Task 7's realized types would be guessing. This one deviation is called out.

**Type consistency:** `_build_field_arrays(dist, domain, T)` defined Task 6, used Tasks 6-7. `_empty_grid`/`_empty_coeff` defined Task 2, used Tasks 2 & 7. `SerialFieldStorage(arch, grid, coeff)` defined Task 7, used in Task 7's constructor. `storage_mode(::ScalarField{T,<:SerialFieldStorage})` consistent with the `<:` loosening in Task 9. Consistent.

**Known-flaky tests to ignore** (pre-existing, unrelated): `test_aqua.jl` (missing Aqua dep), `test_cpu_architecture.jl` (missing KernelAbstractions dep), `test_compatibility.jl` (example-file string asserts).
