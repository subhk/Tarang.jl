# API Ergonomics Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add ergonomic convenience layers to Tarang.jl that reduce boilerplate and improve discoverability without changing the existing core API.

**Architecture:** All improvements are additive — new methods, new functions, new `show` methods layered on top of existing internals. No changes to core data structures or existing method signatures. New code goes in `src/extras/` (user-facing conveniences) and `src/tools/pretty_printing.jl` (display).

**Tech Stack:** Julia, Tarang.jl internals (Domain, Field, Problem, Solver types)

---

### Task 1: Domain `show` Method

The `Domain` type has no custom `show` method — it dumps raw struct fields. Add a clean box-formatted display.

**Files:**
- Modify: `src/tools/pretty_printing.jl` (append after Distributor section, before Exports section ~line 368)
- Test: `test/test_pretty_printing.jl` (create)

- [ ] **Step 1: Write the failing test**

Create `test/test_pretty_printing.jl`:

```julia
using Test
using Tarang

@testset "Domain pretty printing" begin
    domain = PeriodicDomain(64, 64)

    # Compact form
    s = sprint(show, domain)
    @test contains(s, "Domain")
    @test contains(s, "2D")

    # Rich form
    s = sprint(show, MIME("text/plain"), domain)
    @test contains(s, "Domain")
    @test contains(s, "RealFourier")
    @test contains(s, "64")
    @test contains(s, "Architecture")
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/subhajitkar/Documents/GitHub/Tarang.jl && julia --project -e 'using Pkg; Pkg.test(test_args=["test_pretty_printing"])'`

Or directly: `cd /Users/subhajitkar/Documents/GitHub/Tarang.jl && julia --project test/test_pretty_printing.jl`

Expected: FAIL — Domain uses default struct printing, so `contains(s, "Architecture")` will fail.

- [ ] **Step 3: Implement Domain show methods**

Append to `src/tools/pretty_printing.jl` before the `# Exports` section (~line 368):

```julia
# ============================================================================
# Domain pretty printing
# ============================================================================

function Base.show(io::IO, domain::Domain)
    ndims = domain.dim
    sizes = [b.meta.size for b in domain.bases]
    print(io, "Domain($(ndims)D, $(join(sizes, "×")))")
end

function Base.show(io::IO, ::MIME"text/plain", domain::Domain)
    println(io, _box_line(BOX_TL, BOX_H, BOX_TR))
    println(io, _box_text_centered("Tarang.jl Domain"))
    println(io, _box_line(BOX_LT, BOX_H, BOX_RT))

    println(io, _box_text("Dimensions:    $(domain.dim)"))

    # Grid shape
    sizes = [b.meta.size for b in domain.bases]
    total = prod(sizes)
    println(io, _box_text("Grid size:     $(join(sizes, " × "))"))
    println(io, _box_text(@sprintf("Total points:  %d (%.2fM)", total, total/1e6)))

    # Architecture
    arch = domain.dist.architecture
    arch_name = isa(arch, CPU) ? "CPU" : "GPU"
    println(io, _box_text("Architecture:  $arch_name"))
    println(io, _box_text("Data type:     $(domain.dist.dtype)"))

    # Bases
    println(io, _box_line(BOX_LT, BOX_H, BOX_RT))
    println(io, _box_text("Bases:"))
    for (i, b) in enumerate(domain.bases)
        bname = _basis_type_name(b)
        coord = b.meta.coord.name
        N = b.meta.size
        bounds = b.meta.bounds
        dealias = b.meta.dealias
        println(io, _box_text(@sprintf("  [%s] %s(N=%d, [%.4g, %.4g], dealias=%.1f)", coord, bname, N, bounds[1], bounds[2], dealias)))
    end

    # Volume
    vol = volume(domain)
    if isfinite(vol)
        println(io, _box_line(BOX_LT, BOX_H, BOX_RT))
        println(io, _box_text(@sprintf("Volume:        %.4g", vol)))
    end

    print(io, _box_line(BOX_BL, BOX_H, BOX_BR))
end
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/subhajitkar/Documents/GitHub/Tarang.jl && julia --project test/test_pretty_printing.jl`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tools/pretty_printing.jl test/test_pretty_printing.jl
git commit -m "feat: add Domain show method with box-formatted display"
```

---

### Task 2: Auto-Transform `get_grid_data` / `get_coeff_data`

Currently users must call `ensure_layout!(field, :g)` before `get_grid_data(field)`. Add optional auto-transform via a keyword argument `ensure=true` (default `false` for backward compatibility), plus a convenience function `grid_data(field)` that always auto-transforms.

**Files:**
- Modify: `src/extras/quick_domains.jl` (append new convenience functions)
- Modify: `src/Tarang.jl` (add exports)
- Test: `test/test_convenience_api.jl` (create)

- [ ] **Step 1: Write the failing test**

Create `test/test_convenience_api.jl`:

```julia
using Test
using Tarang

@testset "grid_data auto-transform" begin
    domain = PeriodicDomain(16)
    T = ScalarField(domain, "T")

    # Set data in grid space
    ensure_layout!(T, :g)
    get_grid_data(T) .= 1.0

    # Transform to coefficient space
    ensure_layout!(T, :c)
    @test T.current_layout == :c

    # grid_data should auto-transform back
    data = grid_data(T)
    @test T.current_layout == :g
    @test data === get_grid_data(T)
end

@testset "coeff_data auto-transform" begin
    domain = PeriodicDomain(16)
    T = ScalarField(domain, "T")

    ensure_layout!(T, :g)
    get_grid_data(T) .= 1.0

    # coeff_data should auto-transform to :c
    data = coeff_data(T)
    @test T.current_layout == :c
    @test data === get_coeff_data(T)
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/subhajitkar/Documents/GitHub/Tarang.jl && julia --project test/test_convenience_api.jl`

Expected: FAIL — `grid_data` is not defined.

- [ ] **Step 3: Implement convenience functions**

Append to `src/extras/quick_domains.jl`:

```julia
# ============================================================================
# Auto-transforming data access
# ============================================================================

"""
    grid_data(field)

Return grid-space data, automatically transforming if needed.
Equivalent to `ensure_layout!(field, :g); get_grid_data(field)`.
"""
function grid_data(field::ScalarField)
    ensure_layout!(field, :g)
    return get_grid_data(field)
end

function grid_data(field::VectorField)
    for c in field.components
        ensure_layout!(c, :g)
    end
    return [get_grid_data(c) for c in field.components]
end

"""
    coeff_data(field)

Return coefficient-space data, automatically transforming if needed.
Equivalent to `ensure_layout!(field, :c); get_coeff_data(field)`.
"""
function coeff_data(field::ScalarField)
    ensure_layout!(field, :c)
    return get_coeff_data(field)
end

function coeff_data(field::VectorField)
    for c in field.components
        ensure_layout!(c, :c)
    end
    return [get_coeff_data(c) for c in field.components]
end
```

- [ ] **Step 4: Add exports to `src/Tarang.jl`**

In the export block, add `grid_data, coeff_data` to the "High-level convenience API" section (near line 286):

```julia
    # High-level convenience API (zero-boilerplate domain + field creation)
    PeriodicDomain, ChebyshevDomain, ChannelDomain, ChannelDomain3D,
    grid_data, coeff_data,
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /Users/subhajitkar/Documents/GitHub/Tarang.jl && julia --project test/test_convenience_api.jl`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/extras/quick_domains.jl src/Tarang.jl test/test_convenience_api.jl
git commit -m "feat: add grid_data/coeff_data with auto-transform"
```

---

### Task 3: Ergonomic Initial Condition Interface (`set!`)

Add `set!(field, f)` where `f` is a function of coordinates. This handles `ensure_layout!`, grid coordinate extraction, meshgrid broadcasting, and assignment in one call.

**Files:**
- Modify: `src/extras/quick_domains.jl` (append)
- Modify: `src/Tarang.jl` (add export)
- Modify: `test/test_convenience_api.jl` (append)

- [ ] **Step 1: Write the failing test**

Append to `test/test_convenience_api.jl`:

```julia
@testset "set! with function" begin
    # 1D
    domain1d = PeriodicDomain(32)
    T = ScalarField(domain1d, "T")
    set!(T, (x,) -> sin(x))
    @test T.current_layout == :g

    x = range(0, 2π; length=32)  # approximate check
    # Data should be populated (not zeros)
    @test maximum(abs.(get_grid_data(T))) > 0.5

    # 2D
    domain2d = PeriodicDomain(16, 16)
    T2 = ScalarField(domain2d, "T2")
    set!(T2, (x, y) -> sin(x) * cos(y))
    @test T2.current_layout == :g
    @test maximum(abs.(get_grid_data(T2))) > 0.5
end

@testset "set! with constant" begin
    domain = PeriodicDomain(16)
    T = ScalarField(domain, "T")
    set!(T, 42.0)
    @test all(get_grid_data(T) .== 42.0)
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/subhajitkar/Documents/GitHub/Tarang.jl && julia --project test/test_convenience_api.jl`

Expected: FAIL — `set!` is not defined.

- [ ] **Step 3: Implement `set!`**

Append to `src/extras/quick_domains.jl`:

```julia
# ============================================================================
# Ergonomic initial condition setting
# ============================================================================

"""
    set!(field::ScalarField, f::Function)

Set field data from a function of grid coordinates.
The function should accept as many arguments as the domain has dimensions.

# Examples
```julia
set!(T, (x,) -> sin(x))                    # 1D
set!(T, (x, y) -> sin(x) * cos(y))         # 2D
set!(T, (x, y, z) -> exp(-x^2 - y^2 - z^2)) # 3D
```
"""
function set!(field::ScalarField, f::Function)
    ensure_layout!(field, :g)
    domain = field.domain
    if domain === nothing
        domain = Domain(field.dist, field.bases)
    end
    grids = create_meshgrid(domain; on_device=false)
    data = f.(grids...)
    # Handle GPU transfer if needed
    if is_gpu(field.dist.architecture)
        get_grid_data(field) .= on_architecture(field.dist.architecture, data)
    else
        get_grid_data(field) .= data
    end
    return field
end

"""
    set!(field::ScalarField, value::Number)

Set all grid points to a constant value.
"""
function set!(field::ScalarField, value::Number)
    ensure_layout!(field, :g)
    get_grid_data(field) .= value
    return field
end

"""
    set!(field::VectorField, fs::Tuple{Vararg{Function}})

Set each component of a vector field from a tuple of functions.

# Example
```julia
set!(u, ((x,y) -> sin(y), (x,y) -> -sin(x)))  # 2D velocity
```
"""
function set!(field::VectorField, fs::Tuple{Vararg{Function}})
    if length(fs) != length(field.components)
        throw(ArgumentError("Expected $(length(field.components)) functions, got $(length(fs))"))
    end
    for (comp, f) in zip(field.components, fs)
        set!(comp, f)
    end
    return field
end
```

- [ ] **Step 4: Add export**

In `src/Tarang.jl`, add `set!` to the convenience API exports:

```julia
    # High-level convenience API (zero-boilerplate domain + field creation)
    PeriodicDomain, ChebyshevDomain, ChannelDomain, ChannelDomain3D,
    grid_data, coeff_data, set!,
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /Users/subhajitkar/Documents/GitHub/Tarang.jl && julia --project test/test_convenience_api.jl`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/extras/quick_domains.jl src/Tarang.jl test/test_convenience_api.jl
git commit -m "feat: add set! for ergonomic initial conditions"
```

---

### Task 4: Declarative Callback Helpers

Add `on_interval`, `on_sim_time` helpers that make the callback API more readable than raw `Pair` tuples.

**Files:**
- Modify: `src/extras/quick_domains.jl` (append)
- Modify: `src/Tarang.jl` (add exports)
- Modify: `test/test_convenience_api.jl` (append)

- [ ] **Step 1: Write the failing test**

Append to `test/test_convenience_api.jl`:

```julia
@testset "Callback helpers" begin
    # on_interval should produce a Pair
    cb = on_interval(10) do solver
        nothing
    end
    @test cb isa Pair
    @test cb.first == 10
    @test cb.second isa Function

    # on_sim_time should produce a Pair with Float
    cb2 = on_sim_time(0.5) do solver
        nothing
    end
    @test cb2 isa Pair
    @test cb2.first isa AbstractFloat
    @test cb2.first == 0.5
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/subhajitkar/Documents/GitHub/Tarang.jl && julia --project test/test_convenience_api.jl`

Expected: FAIL — `on_interval` not defined.

- [ ] **Step 3: Implement callback helpers**

Append to `src/extras/quick_domains.jl`:

```julia
# ============================================================================
# Declarative callback helpers
# ============================================================================

"""
    on_interval(f, n::Integer)

Create a callback that fires every `n` iterations.

# Example
```julia
run!(solver; stop_time=10.0, callbacks=[
    on_interval(100) do solver
        @info "Step \$(solver.iteration), t=\$(solver.sim_time)"
    end
])
```
"""
on_interval(f::Function, n::Integer) = Pair(n, f)

"""
    on_sim_time(f, dt::Real)

Create a callback that fires every `dt` simulation time units.

# Example
```julia
run!(solver; stop_time=10.0, callbacks=[
    on_sim_time(0.5) do solver
        @info "t = \$(solver.sim_time)"
    end
])
```
"""
on_sim_time(f::Function, dt::Real) = Pair(Float64(dt), f)
```

- [ ] **Step 4: Add exports**

In `src/Tarang.jl`, add to convenience API exports:

```julia
    # High-level convenience API (zero-boilerplate domain + field creation)
    PeriodicDomain, ChebyshevDomain, ChannelDomain, ChannelDomain3D,
    grid_data, coeff_data, set!,
    on_interval, on_sim_time,
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /Users/subhajitkar/Documents/GitHub/Tarang.jl && julia --project test/test_convenience_api.jl`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/extras/quick_domains.jl src/Tarang.jl test/test_convenience_api.jl
git commit -m "feat: add on_interval/on_sim_time callback helpers"
```

---

### Task 5: MPI Convenience Macro `@root_only`

Add `@root_only` macro that only executes on MPI rank 0, eliminating the repeated `if rank == 0 ... end` pattern.

**Files:**
- Modify: `src/extras/quick_domains.jl` (append)
- Modify: `src/Tarang.jl` (add export)
- Modify: `test/test_convenience_api.jl` (append)

- [ ] **Step 1: Write the failing test**

Append to `test/test_convenience_api.jl`:

```julia
@testset "@root_only macro" begin
    # Should execute on rank 0 (single-process test)
    executed = Ref(false)
    @root_only executed[] = true
    @test executed[]

    # Should work with blocks
    result = Ref(0)
    @root_only begin
        result[] = 42
    end
    @test result[] == 42
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/subhajitkar/Documents/GitHub/Tarang.jl && julia --project test/test_convenience_api.jl`

Expected: FAIL — `@root_only` not defined.

- [ ] **Step 3: Implement `@root_only`**

Append to `src/extras/quick_domains.jl`:

```julia
# ============================================================================
# MPI convenience macros
# ============================================================================

"""
    @root_only expr

Execute `expr` only on MPI rank 0. Useful for printing, file I/O, etc.

# Example
```julia
@root_only @info "Simulation starting..."
@root_only begin
    println("Results: \$energy")
    save_data(output)
end
```
"""
macro root_only(expr)
    quote
        if !MPI.Initialized() || MPI.Comm_rank(MPI.COMM_WORLD) == 0
            $(esc(expr))
        end
    end
end
```

- [ ] **Step 4: Add export**

In `src/Tarang.jl`, add to convenience API exports:

```julia
    # High-level convenience API (zero-boilerplate domain + field creation)
    PeriodicDomain, ChebyshevDomain, ChannelDomain, ChannelDomain3D,
    grid_data, coeff_data, set!,
    on_interval, on_sim_time,
    @root_only,
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /Users/subhajitkar/Documents/GitHub/Tarang.jl && julia --project test/test_convenience_api.jl`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/extras/quick_domains.jl src/Tarang.jl test/test_convenience_api.jl
git commit -m "feat: add @root_only macro for MPI convenience"
```

---

### Task 6: `TensorField` and `CFL` Show Methods

Add missing `show` methods for `TensorField` and `CFL` types.

**Files:**
- Modify: `src/tools/pretty_printing.jl`
- Modify: `test/test_pretty_printing.jl` (append)

- [ ] **Step 1: Write the failing test**

Append to `test/test_pretty_printing.jl`:

```julia
@testset "TensorField pretty printing" begin
    domain = PeriodicDomain(16, 16)
    T = TensorField(domain, "stress")

    s = sprint(show, T)
    @test contains(s, "TensorField")
    @test contains(s, "stress")

    s = sprint(show, MIME("text/plain"), T)
    @test contains(s, "TensorField")
    @test contains(s, "Components")
end

@testset "CFL pretty printing" begin
    domain = PeriodicDomain(16)
    T = ScalarField(domain, "T")
    problem = IVP([T])
    add_equation!(problem, "dt(T) = 0")
    solver = InitialValueSolver(problem, RK111(); device="cpu")

    cfl = CFL(solver; initial_dt=0.01, safety=0.5, max_dt=0.1)

    s = sprint(show, cfl)
    @test contains(s, "CFL")

    s = sprint(show, MIME("text/plain"), cfl)
    @test contains(s, "CFL")
    @test contains(s, "Safety")
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/subhajitkar/Documents/GitHub/Tarang.jl && julia --project test/test_pretty_printing.jl`

Expected: FAIL — TensorField and CFL use default struct printing.

- [ ] **Step 3: Implement show methods**

Append to `src/tools/pretty_printing.jl` (before the Exports section):

```julia
# ============================================================================
# TensorField pretty printing
# ============================================================================

function Base.show(io::IO, field::TensorField)
    name = field.name
    ncomp = size(field.components)
    if isempty(field.bases)
        print(io, "TensorField($name, $(ncomp[1])×$(ncomp[2]), 0D)")
    else
        sizes = [b.meta.size for b in field.bases]
        print(io, "TensorField($name, $(ncomp[1])×$(ncomp[2]), $(join(sizes, "×")))")
    end
end

function Base.show(io::IO, ::MIME"text/plain", field::TensorField)
    println(io, _box_line(BOX_TL, BOX_H, BOX_TR))
    println(io, _box_text_centered("TensorField: $(field.name)"))
    println(io, _box_line(BOX_LT, BOX_H, BOX_RT))

    nrows, ncols = size(field.components)
    println(io, _box_text("Components:    $(nrows) × $(ncols)"))

    if !isempty(field.bases)
        sizes = [b.meta.size for b in field.bases]
        total = prod(sizes) * nrows * ncols
        println(io, _box_text("Grid size:     $(join(sizes, " × ")) × $(nrows)×$(ncols)"))
        println(io, _box_text(@sprintf("Total points:  %d (%.2fM)", total, total/1e6)))
        println(io, _box_text("Data type:     $(field.dtype)"))
    end
    print(io, _box_line(BOX_BL, BOX_H, BOX_BR))
end

# ============================================================================
# CFL pretty printing
# ============================================================================

function Base.show(io::IO, cfl::CFL)
    nvel = length(cfl.velocities)
    print(io, "CFL(dt=$(round(cfl.current_dt; sigdigits=3)), safety=$(cfl.safety), $nvel velocities)")
end

function Base.show(io::IO, ::MIME"text/plain", cfl::CFL)
    println(io, _box_line(BOX_TL, BOX_H, BOX_TR))
    println(io, _box_text_centered("CFL Controller"))
    println(io, _box_line(BOX_LT, BOX_H, BOX_RT))
    println(io, _box_text(@sprintf("Current dt:    %.4e", cfl.current_dt)))
    println(io, _box_text(@sprintf("Safety:        %.2f", cfl.safety)))
    println(io, _box_text(@sprintf("Max dt:        %.4e", cfl.max_dt)))
    println(io, _box_text("Cadence:       $(cfl.cadence)"))
    println(io, _box_text(@sprintf("Max change:    %.2f", cfl.max_change)))
    println(io, _box_text(@sprintf("Min change:    %.2f", cfl.min_change)))
    println(io, _box_line(BOX_LT, BOX_H, BOX_RT))
    nvel = length(cfl.velocities)
    println(io, _box_text("Velocities:    $nvel registered"))
    print(io, _box_line(BOX_BL, BOX_H, BOX_BR))
end
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/subhajitkar/Documents/GitHub/Tarang.jl && julia --project test/test_pretty_printing.jl`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tools/pretty_printing.jl test/test_pretty_printing.jl
git commit -m "feat: add TensorField and CFL show methods"
```

---

### Task 7: Structured Boundary Condition Helpers

Add `no_slip!`, `fixed_temperature!`, `free_slip!` convenience functions that wrap the existing `add_bc!` with structured BC objects. These reduce the most common BC patterns to one-liners.

**Files:**
- Modify: `src/extras/quick_domains.jl` (append)
- Modify: `src/Tarang.jl` (add exports)
- Modify: `test/test_convenience_api.jl` (append)

- [ ] **Step 1: Write the failing test**

Append to `test/test_convenience_api.jl`:

```julia
@testset "Structured BC helpers" begin
    domain = ChannelDomain(16, 8; Lx=2π, Lz=1.0)
    u = VectorField(domain, "u")
    T = ScalarField(domain, "T")
    tau_u1 = VectorField(domain.dist, domain.dist.coordsys, "tau_u1",
                         (domain.bases[1],), domain.dist.dtype)
    tau_T1 = ScalarField(domain.dist, "tau_T1", (domain.bases[1],), domain.dist.dtype)
    tau_T2 = ScalarField(domain.dist, "tau_T2", (domain.bases[1],), domain.dist.dtype)

    problem = IVP([u, T, tau_u1, tau_T1, tau_T2])

    # These should not error
    no_slip!(problem, "u", "z", 0.0)
    no_slip!(problem, "u", "z", 1.0)
    fixed_value!(problem, "T", "z", 0.0, 1.0)
    fixed_value!(problem, "T", "z", 1.0, 0.0)

    @test length(problem.boundary_conditions) >= 4
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/subhajitkar/Documents/GitHub/Tarang.jl && julia --project test/test_convenience_api.jl`

Expected: FAIL — `no_slip!` not defined.

- [ ] **Step 3: Implement BC helpers**

Append to `src/extras/quick_domains.jl`:

```julia
# ============================================================================
# Structured boundary condition helpers
# ============================================================================

"""
    no_slip!(problem, field_name, coord, position)

Add no-slip (zero Dirichlet) boundary condition for a velocity field.

# Example
```julia
no_slip!(problem, "u", "z", 0.0)  # u = 0 at z = 0
no_slip!(problem, "u", "z", 1.0)  # u = 0 at z = 1
```
"""
function no_slip!(problem::Problem, field_name::String, coord::String, position::Real)
    bc = dirichlet_bc(field_name, coord, Float64(position), 0.0)
    add_bc!(problem, bc)
    return bc
end

"""
    fixed_value!(problem, field_name, coord, position, value)

Add a Dirichlet boundary condition with a fixed value.

# Example
```julia
fixed_value!(problem, "T", "z", 0.0, 1.0)   # T = 1 at z = 0
fixed_value!(problem, "T", "z", 1.0, 0.0)   # T = 0 at z = 1
```
"""
function fixed_value!(problem::Problem, field_name::String, coord::String,
                      position::Real, value)
    bc = dirichlet_bc(field_name, coord, Float64(position), value)
    add_bc!(problem, bc)
    return bc
end

"""
    free_slip!(problem, field_name, coord, position)

Add free-slip (zero Neumann) boundary condition.

# Example
```julia
free_slip!(problem, "u", "z", 0.0)  # ∂u/∂z = 0 at z = 0
```
"""
function free_slip!(problem::Problem, field_name::String, coord::String, position::Real)
    bc = neumann_bc(field_name, coord, Float64(position), 0.0)
    add_bc!(problem, bc)
    return bc
end

"""
    insulating!(problem, field_name, coord, position)

Add insulating (zero Neumann) boundary condition for a scalar field.

# Example
```julia
insulating!(problem, "T", "z", 1.0)  # ∂T/∂z = 0 at z = 1
```
"""
function insulating!(problem::Problem, field_name::String, coord::String, position::Real)
    bc = neumann_bc(field_name, coord, Float64(position), 0.0)
    add_bc!(problem, bc)
    return bc
end
```

- [ ] **Step 4: Add exports**

In `src/Tarang.jl`, add to convenience API exports:

```julia
    # High-level convenience API (zero-boilerplate domain + field creation)
    PeriodicDomain, ChebyshevDomain, ChannelDomain, ChannelDomain3D,
    grid_data, coeff_data, set!,
    on_interval, on_sim_time,
    @root_only,
    no_slip!, fixed_value!, free_slip!, insulating!,
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /Users/subhajitkar/Documents/GitHub/Tarang.jl && julia --project test/test_convenience_api.jl`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/extras/quick_domains.jl src/Tarang.jl test/test_convenience_api.jl
git commit -m "feat: add no_slip!, fixed_value!, free_slip!, insulating! BC helpers"
```

---

### Task 8: `@parameters` Macro for Bulk Substitutions

Replace repeated `add_substitution!` calls with a single macro.

**Files:**
- Modify: `src/extras/quick_domains.jl` (append)
- Modify: `src/Tarang.jl` (add export)
- Modify: `test/test_convenience_api.jl` (append)

- [ ] **Step 1: Write the failing test**

Append to `test/test_convenience_api.jl`:

```julia
@testset "add_parameters!" begin
    domain = PeriodicDomain(16)
    T = ScalarField(domain, "T")
    problem = IVP([T])

    add_parameters!(problem, nu=1e-3, kappa=1e-4, Ra=1e6)

    @test problem.namespace["nu"] == 1e-3
    @test problem.namespace["kappa"] == 1e-4
    @test problem.namespace["Ra"] == 1e6
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/subhajitkar/Documents/GitHub/Tarang.jl && julia --project test/test_convenience_api.jl`

Expected: FAIL — `add_parameters!` not defined.

- [ ] **Step 3: Implement `add_parameters!`**

Append to `src/extras/quick_domains.jl`:

```julia
# ============================================================================
# Bulk parameter substitution
# ============================================================================

"""
    add_parameters!(problem; kwargs...)

Add multiple parameter substitutions at once.

# Example
```julia
# Instead of:
add_substitution!(problem, "nu", 1e-3)
add_substitution!(problem, "kappa", 1e-4)
add_substitution!(problem, "Ra", 1e6)

# Write:
add_parameters!(problem, nu=1e-3, kappa=1e-4, Ra=1e6)
```
"""
function add_parameters!(problem::Problem; kwargs...)
    for (name, value) in kwargs
        add_substitution!(problem, String(name), value)
    end
    return problem
end
```

- [ ] **Step 4: Add export**

In `src/Tarang.jl`, add `add_parameters!` to convenience API exports:

```julia
    # High-level convenience API (zero-boilerplate domain + field creation)
    PeriodicDomain, ChebyshevDomain, ChannelDomain, ChannelDomain3D,
    grid_data, coeff_data, set!,
    on_interval, on_sim_time,
    @root_only,
    no_slip!, fixed_value!, free_slip!, insulating!,
    add_parameters!,
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /Users/subhajitkar/Documents/GitHub/Tarang.jl && julia --project test/test_convenience_api.jl`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/extras/quick_domains.jl src/Tarang.jl test/test_convenience_api.jl
git commit -m "feat: add add_parameters! for bulk substitutions"
```

---

### Task 9: Integration Test — Ergonomic 2D Turbulence Example

Write a new example that uses all the ergonomic improvements to verify they compose well. This validates the full API surface end-to-end.

**Files:**
- Create: `examples/ivp/ergonomic_2d_turbulence.jl`

- [ ] **Step 1: Write the ergonomic example**

Create `examples/ivp/ergonomic_2d_turbulence.jl`:

```julia
"""
2D Decaying Turbulence — Ergonomic API Demo

Same physics as quickstart_2d_turbulence.jl but using the
convenience API for comparison.
"""
using Tarang

# Domain (one line)
domain = PeriodicDomain(128, 128)

# Fields
ψ = ScalarField(domain, "ψ")
q = ScalarField(domain, "q")
u = VectorField(domain, "u")

# Problem with bulk parameters
problem = IVP([q, ψ, u])
add_parameters!(problem, nu=1e-4)

add_equation!(problem, "Δ(ψ) - q = 0")
add_equation!(problem, "u - skew(grad(ψ)) = 0")
add_equation!(problem, "∂t(q) - nu*Δ(q) = -u⋅∇(q)")

# Initial condition (one line per field)
set!(q, (x, y) -> sin(4x) * cos(4y) + 0.5 * cos(3x) * sin(5y))

# Solver
solver = InitialValueSolver(problem, RK222(); dt=1e-3)

# Display setup
@root_only println(solver)
@root_only println(domain)

# Run with readable callbacks
run!(solver; stop_time=1.0, callbacks=[
    on_interval(100) do s
        data = grid_data(q)
        enstrophy = sum(data.^2) / length(data)
        @root_only @info "Step $(s.iteration): enstrophy = $(round(enstrophy; sigdigits=4))"
    end
])

@root_only @info "Done! Final time: $(solver.sim_time)"
```

- [ ] **Step 2: Run the example to verify it works**

Run: `cd /Users/subhajitkar/Documents/GitHub/Tarang.jl && julia --project examples/ivp/ergonomic_2d_turbulence.jl`

Expected: Runs to completion with enstrophy output.

- [ ] **Step 3: Commit**

```bash
git add examples/ivp/ergonomic_2d_turbulence.jl
git commit -m "feat: add ergonomic 2D turbulence example showcasing new API"
```

---

## Summary

| Task | What | Files Modified |
|------|------|----------------|
| 1 | Domain `show` method | `pretty_printing.jl`, new test file |
| 2 | `grid_data`/`coeff_data` auto-transform | `quick_domains.jl`, `Tarang.jl`, new test file |
| 3 | `set!` for initial conditions | `quick_domains.jl`, `Tarang.jl`, test file |
| 4 | `on_interval`/`on_sim_time` callback helpers | `quick_domains.jl`, `Tarang.jl`, test file |
| 5 | `@root_only` MPI macro | `quick_domains.jl`, `Tarang.jl`, test file |
| 6 | `TensorField` and `CFL` show methods | `pretty_printing.jl`, test file |
| 7 | Structured BC helpers | `quick_domains.jl`, `Tarang.jl`, test file |
| 8 | `add_parameters!` bulk substitution | `quick_domains.jl`, `Tarang.jl`, test file |
| 9 | Ergonomic example (integration test) | new example file |
