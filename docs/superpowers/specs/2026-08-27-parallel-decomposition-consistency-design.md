# Parallel Decomposition Consistency — Design (Phases 0 and 1)

**Date:** 2026-08-27
**Status:** proposed
**Scope of this spec:** Phase 0 (single source of truth for decomposed axes) and
Phase 1 (GPU+MPI wired into `forward_transform!`). Phase 2 (axis permutation) is
sketched at the end and gets its own spec.

## Problem

Three user-visible inconsistencies, one root cause.

1. **Axis order is load-bearing under MPI.** A 2-D domain declared as
   `(RealFourier x, ChebyshevT z)` — the ordering every channel-flow example
   writes — is refused at solver-build time. `(ChebyshevT z, RealFourier x)` works.
   Measured at np=4: the first throws *"the decomposed (trailing) axis/axes [2] are
   non-Fourier"*, the second round-trips at `4.4e-16`.

2. **`mesh=(2,2)` means different things on different backends.** On the
   PencilArrays path it dies with *"PencilFFT plan creation failed … Please check
   your PencilFFTs installation"* — which blames the install for a structural
   impossibility. On the TransposableField path the same mesh works and is
   coefficient-exact against serial (`1.9e-15`).

3. **GPU+MPI is not reachable through the ordinary API.** `forward_transform!` on a
   distributed GPU field errors and tells the caller to hand-drive a
   `TransposableField`.

The root cause is shared: **the decomposition convention is re-derived inline at
every consumer instead of being recorded once.** The two conventions —
PencilArrays decomposes the *last* `ndims_mesh` dims, TransposableField decomposes
the *first* — appear as an independently-maintained `if/else` in nine places:

| file | line | what it derives |
|---|---|---|
| `core/operators/derivatives/derivatives_fourier.jl` | 34 | `decomp_dims`, then `axis in decomp_dims` |
| `core/field/field_data/field_data_copy_alloc.jl` | 330 | `get_local_array_size`, both branches |
| `core/field/field_layout/field_layout_filters_shapes.jl` | 272 | local start/end per axis |
| `core/field/field_data/field_data_distributor_utils.jl` | 79 | `get_local_range` — computes `mesh_axis` inline |
| `core/distributor/distributor_core.jl` | 603 | `decomp_dims` for `create_pencil` |
| `core/distributor/distributor_core.jl` | 898 | `global_dim_idx` per mesh dim |
| `core/distributor/distributor_core.jl` | 1069 | `local_indices` |
| `tools/netcdf_output.jl` | 1299 | slab start |
| `tools/netcdf_output.jl` | 1442 | slab count |

Nine copies of one rule is how the two conventions drifted apart, and it is how
they will drift again. Every phase below depends on collapsing them first.

## Measured constraints

These are facts established by probe, not assumptions. They bound the design.

**PencilFFTs requires decomposed dims to be trailing.** Not a Tarang convention —
the library's own check:

```
Pencil(topo, (16,12), (2,))  -> Pencil OK, PencilFFTPlan OK
Pencil(topo, (16,12), (1,))  -> Pencil OK
                                PencilFFTPlan: ArgumentError:
                                decomposed dimensions of input data must be (2,) (got (1,))
```

So "Chebyshev must come first" is downstream of PencilFFTs. It cannot be relaxed
by choosing different `decomp_dims`.

**PencilFFTs rejects permuted input pencils.**

```
Pencil(topo,(16,12),(1,); permute=Permutation(2,1)) -> logical (16,12), memory-local (12,8)
PencilFFTPlan over it -> ArgumentError: dimensions of input array must be unpermuted
```

PencilArrays supports a logical/memory split, but PencilFFTs will not consume it.
Any permutation must therefore sit at Tarang's accessor boundary, above an
unpermuted storage-order pencil. That route is verified to work:

```
Pencil(topo,(12,16),(2,))          # storage order (z,x)
PermutedDimsArray(a, (2,1))        # presented as (x,z)
write through view -> lands in pencil; PencilFFTPlan over the pencil OK
```

**A 2-D domain cannot use a 2-D process mesh on PencilFFTs.** An N-D domain gets at
most (N−1)-D decomposition, because at least one axis must stay local. This is not
a bug to fix; it is a constraint to report honestly.

---

## Phase 0 — one source of truth for decomposed axes

**Goal:** every consumer asks one function which axes are decomposed. No behavior
change to any configuration that works today.

### Interface

```julia
"""
    decomposed_axes(dist::Distributor, ndim::Int) -> NTuple{M,Int}

Global axis indices decomposed across the process mesh, ascending, for an
`ndim`-dimensional field on `dist`. Empty when `dist.size == 1` or
`dist.mesh === nothing`.

Conventions (the single statement of both):
  * `use_pencil_arrays=true`  — PencilArrays decomposes the LAST `length(mesh)` dims.
  * `use_pencil_arrays=false` — TransposableField decomposes the FIRST `min(length(mesh), 2)` dims.
"""
function decomposed_axes end

"""
    mesh_axis_for(dist, ndim, axis) -> Union{Nothing,Int}

Which mesh dimension decomposes global `axis`, or `nothing` if `axis` is local.
Inverse lookup for the sites that need `dist.mesh[mesh_idx]`.
"""
function mesh_axis_for end

is_decomposed_axis(dist, ndim, axis) = mesh_axis_for(dist, ndim, axis) !== nothing
```

Placement: `src/core/distributor/distributor_core.jl`, beside `local_indices`,
which becomes its first consumer. `mesh_axis_for` is not new logic — it is
`get_local_range`'s inline `mesh_axis` computation
(`field_data_distributor_utils.jl:86-100`) lifted out and named, so that site
becomes a caller rather than a tenth copy.

### Work

Rewrite each of the nine sites to call these. Two sites need care rather than
mechanical substitution:

- `distributor_core.jl:603` derives `decomp_dims` for `create_pencil` with a
  `decomp_index` variant that keeps one dim local for FFT. That is a *different*
  question from "which axes are decomposed for storage" and keeps its own helper;
  only its `decomp_index === nothing` branch delegates to `decomposed_axes`.
- `field_data_copy_alloc.jl:330` is the definition that everything else must match
  (it decides the allocated array shape). It delegates, and its remainder handling
  — PencilArrays' real range via `pencil_local_range` when available, the
  remainder-on-first fallback otherwise — stays where it is.

### Verification

- **Behavior-preservation is the whole test.** Full `Pkg.test()`, MPI CI at np=2
  and np=4 (baseline today: 54/54 both) must stay green with no diff in any
  numeric output.
- **New ratchet** `test_decomposition_convention.jl`: asserts the nine call sites
  contain no local re-derivation of the convention (grep-style, matching the
  existing `test_catch_ratchet.jl` / `test_hasfield_ratchet.jl` idiom), and pins
  `decomposed_axes` against a table of `(use_pencil_arrays, mesh, ndim)` cases
  including the 1-D mesh, the 2-D mesh on a 3-D domain, and the unit-factor mesh.
- Direct unit test that `get_local_array_size`, `local_indices`, and
  `decomposed_axes` agree for every case in that table — they are three views of
  one fact and have no test forcing them to agree today.

### Falls out for free

`mesh=(2,2)` on a 2-D domain can now be detected at `Distributor` construction:
`decomposed_axes` returns both axes, leaving none local, which PencilFFTs cannot
serve. Replace the misleading *"check your PencilFFTs installation"* with a
statement of the real constraint and the working alternative. This costs a few
lines once the predicate exists, and it fixes inconsistency 2 without touching
any working configuration.

---

## Phase 1 — GPU+MPI through `forward_transform!`

**Goal:** a distributed GPU field transforms through the ordinary call. No
hand-driven `TransposableField`.

### What already exists

- `TransposableFieldStorage{CT,N,B}` is defined (`transposable_field.jl:88`) and
  documented as *"Absorbs the functionality previously in TransposableField
  wrapper"*. **Nothing constructs it.**
- `storage_mode(::ScalarField{T,<:TransposableFieldStorage}) = TransposableStorage()`
  is dispatched (`transposable_field.jl:103`); `is_transposable_storage` exists.
- `ScalarField` has an inner constructor taking an explicit storage
  (`field_types.jl:133`).
- `_build_field_arrays` already allocates the correct ZLocal-shaped arrays on the
  non-pencil path (verified: `mesh=(2,2)`, 16×12 at np=4 → local `(8,6)`).
- The transform engine itself is verified coefficient-exact against serial: 2-D
  meshes `(2,)`, `(4,)`, `(2,2)`, `(1,4)` and 3-D `(2,2)`, `(4,1)`, `(1,4)`, all
  `≤4.8e-15`, plus over-decomposed cases with empty rank slabs.

So the numerics are done. What is missing is construction and dispatch.

### Design decision: workspace on the Distributor, not per field

`TransposableField`'s constructor calls `MPI.Comm_split` twice and registers a
finalizer to free them. Building one per field means `nfields × 2` collective
comm splits and a finite communicator budget consumed by field allocation.

Instead, cache the transpose workspace on the `Distributor`, keyed by
`(global_shape, eltype, topology)`, beside the existing `pencil_cache`. Fields
stay cheap; the comms are split once per distinct shape and freed with the
distributor. This deviates from what the `TransposableFieldStorage` stub implies
(per-field buffers) — the stub's buffer fields move to the cached workspace and
the storage type holds a reference.

### Work

1. `transpose_workspace!(dist, shape, T)` on `Distributor`, cached, finalized in
   `Base.close(dist)` alongside the pencil cache.
2. `_build_field_arrays` unchanged; `ScalarField` selects `TransposableFieldStorage`
   when `is_gpu(dist.architecture) && dist.size > 1`, wrapping the arrays it
   already builds.
3. `forward_transform!` / `backward_transform!` dispatch on
   `storage_mode(field) isa TransposableStorage` → the verified
   `distributed_forward_transform!` / `distributed_backward_transform!`.
4. Delete the two refusals (`transform_gpu.jl:366`, `transform_fourier.jl:135`).
   The basis-level refusals in `validate_mpi_fourier_only` stay — they are correct
   and state a real constraint (RealFourier's half spectrum has no representation
   in the fixed-shape transpose buffers).
5. `Base.close(dist)` currently returns before setting `dist.closed` when
   `topology === nothing`, which is exactly the GPU+MPI case. Fix while here, or
   the new workspace cache never gets freed.

### Verification

- The np>1 half of `test_transposable_field.jl` gains a **coefficient-parity**
  testset, not just round-trip: distributed coefficients compared against the
  serial reference sliced to the local block, for meshes `(4,1)`, `(1,4)`, `(2,2)`
  in 2-D and 3-D. Round-trip alone cannot see a permutation that forward and
  backward both apply — the probes for this already exist and pass.
- A test that `forward_transform!` on a distributed GPU-array field takes the
  distributed path and matches serial, via JLArray so it runs without hardware
  (the `test_gpu_implicit_guard_jlarray.jl` idiom).
- The NPROCS==1 *"Forward transform matches the regular serial transform"* testset
  went vacuous when the `dist.size==1` short-circuit landed — it now compares
  `forward_transform!` to itself. Restore it to compare the transpose path against
  the serial path at np>1, where it means something.
- Buildkite single-GPU job unchanged; the multi-GPU NCCL job stays disabled.

### Out of scope for Phase 1

Solver integration. This phase makes the *transform* reachable; whether a full
GPU+MPI IVP steps correctly is a separate question with its own unknowns
(per-mode gather/scatter, matrix assembly). Do not claim end-to-end GPU+MPI on
the strength of this phase.

---

## Phase 2 — axis permutation (sketch, separate spec)

Declared basis order becomes the truth at every user-facing surface; storage order
is whatever MPI requires (non-Fourier axes first, Fourier trailing).

Mechanism, fixed by the measured constraints above: an unpermuted storage-order
pencil handed to PencilFFTs, with `get_grid_data` / `get_coeff_data` returning a
`PermutedDimsArray` view in declared order — the identity array when no
permutation is needed, so the common case pays nothing.

Known costs to work through in that spec: every `isa PencilArray`, `parent()`, and
`_get_underlying_pencil_array` site must tolerate the wrapper; hot loops routed
through `field["g"]` pay lazy-view index arithmetic and need an explicit
storage-order accessor; NetCDF dim order, slab start/count, and checkpoint
metadata must all report declared order.

Depends on Phase 0.

## Risks

- **Phase 0 is a refactor across nine sites with no behavior change — which means
  the test suite passing is the only signal that it is right.** The existing MPI
  suite covers the PencilArrays convention well (54 files, np=2 and np=4) but the
  TransposableField convention is thinner. Land the Phase 1 parity tests' probe
  coverage before or with Phase 0, not after.
- **Distributor-cached workspaces are collective objects.** Creating one on a
  subset of ranks deadlocks. The cache key must be derived from data every rank
  agrees on, and lookup must never be conditional on rank-local state.
- The transpose engine is verified on CPU arrays standing in for device arrays.
  Phase 1 does not change that; real multi-GPU remains unexercised by CI.
