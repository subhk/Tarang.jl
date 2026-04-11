# Tarang.jl API Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clean up Tarang.jl's public API by trimming exports, removing redundant domain constructors, and removing dead module globals.

**Architecture:** Remove ~25 internal symbols from the export list (still accessible via `Tarang.symbol`), delete ~20 unused `create_*` functions from `quick_domains.jl`, remove 3 dead module constants, and update 2 test files that reference removed functions.

**Tech Stack:** Julia, Tarang.jl

---

### Task 1: Remove dead module-level constants

**Files:**
- Modify: `src/Tarang.jl:8-11`

- [ ] **Step 1: Remove `const dtype`, `const shape`, `const evaluator`**

Keep `const __version__` (used by NetCDF output and tests). Remove the other three placeholder constants:
```julia
# REMOVE these three lines:
const dtype = Float64
const shape = ()
const evaluator = nothing
```

- [ ] **Step 2: Run tests to verify nothing breaks**

Run: `julia --project -e 'using Tarang'`
Expected: No errors

- [ ] **Step 3: Commit**

---

### Task 2: Trim export list to user-facing API only

**Files:**
- Modify: `src/Tarang.jl:126-321` (export block)

Remove these internal symbols from the export list (25 symbols). They remain accessible via `Tarang.symbol_name`:

**Transpose internals:**
- `pack_for_transpose!`, `unpack_from_transpose!`
- `compute_local_shapes`, `compute_local_shapes_2d`
- `divide_evenly`, `local_range`
- `create_transpose_comms`
- `get_transpose_stats`, `reset_transpose_stats!`

**Unused vectorized ops:**
- `vectorized_add!`, `vectorized_sub!`, `vectorized_mul!`, `vectorized_scale!`
- `vectorized_axpy!`, `vectorized_linear_combination!`
- `fast_axpy!`

**Unused field internals:**
- `has_spectral_bases`, `apply_dealiasing_to_product!`
- `stack_tensor_components`, `unstack_tensor_components!`

**Unused RNG internals:**
- `chunked_rng`, `rng_element`, `rng_elements`
- `IndexArray`, `ChunkedRandomArray`

- [ ] **Step 1: Remove the 25 symbols from the export block**
- [ ] **Step 2: Run `julia --project -e 'using Tarang'` to verify module loads**
- [ ] **Step 3: Commit**

---

### Task 3: Remove redundant domain/field creation functions from quick_domains.jl

**Files:**
- Modify: `src/extras/quick_domains.jl`

Keep ONLY these functions (the zero-boilerplate API):
- `PeriodicDomain` (lines 583-606)
- `ChebyshevDomain` (lines 622-642)
- `ChannelDomain` (lines 657-669)
- `ChannelDomain3D` (lines 676-691)
- `ScalarField(domain::Domain, ...)` convenience constructor (lines 699-700)
- `VectorField(domain::Domain, ...)` convenience constructor (lines 707-708)

Remove ALL of these (none are used outside their definition + 2 test files):
- `_require_coords`
- `create_fourier_domain`, `create_chebyshev_domain`, `create_legendre_domain`
- `create_2d_periodic_domain`, `create_channel_domain`, `create_rectangular_domain`
- `create_3d_periodic_domain`, `create_box_domain`
- `rayleigh_benard_domain`, `taylor_couette_domain`
- `taylor_green_vortex_domain`, `channel_flow_3d_domain`
- `mixing_layer_3d_domain`, `turbulent_convection_3d_domain`
- `create_fields`, `create_scalar_fields`, `create_vector_fields`
- `create_navier_stokes_3d_fields`, `create_thermal_convection_3d_fields`, `create_mhd_3d_fields`
- `domain_info`, `analyze_3d_performance`, `estimate_memory_usage`
- `benchmark_domain_operations`, `benchmark_cpu_performance`
- `create_domain` (symbol-dispatch), `create_optimized_domain`, `create_gpu_optimized_domain`

- [ ] **Step 1: Rewrite quick_domains.jl keeping only the zero-boilerplate API**
- [ ] **Step 2: Run `julia --project -e 'using Tarang'` to verify module loads**
- [ ] **Step 3: Commit**

---

### Task 4: Update test files that reference removed functions

**Files:**
- Modify: `test/test_quick_domains.jl`
- Modify: `test/test_plot_tools.jl`

Both files use `Tarang.create_fourier_domain`, `Tarang.create_2d_periodic_domain`, `Tarang.create_channel_domain`, `Tarang.create_3d_periodic_domain`. Replace with the zero-boilerplate API (`PeriodicDomain`, `ChannelDomain`).

- [ ] **Step 1: Rewrite test_quick_domains.jl using PeriodicDomain/ChannelDomain**
- [ ] **Step 2: Rewrite test_plot_tools.jl domain creation using PeriodicDomain**
- [ ] **Step 3: Run test suite: `julia --project -e 'using Pkg; Pkg.test()'`**
- [ ] **Step 4: Commit**
