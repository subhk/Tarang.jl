"""
Quick domain creation utilities

Zero-boilerplate constructors for common domain configurations.
For full control, use `CartesianCoordinates`, `Distributor`, basis constructors,
and `Domain` directly.
"""

# ============================================================================
# Zero-Boilerplate API — one-line domain + field creation
# ============================================================================

"""
    PeriodicDomain(N...; L=ntuple(_->2π, length(N)), dtype=Float64, arch=CPU(), dealias=3/2)

Create a periodic Fourier domain with a single call. Returns a `Domain` with
all coordinates, distributor, and bases fully configured.

# Examples
```julia
# 2D periodic domain, 128×128, default [0,2π)²
domain = PeriodicDomain(128, 128)
u = VectorField(domain, "u")
p = ScalarField(domain, "p")

# 3D periodic domain, 64³, custom box size
domain = PeriodicDomain(64, 64, 64; L=(1.0, 1.0, 1.0))

# GPU
domain = PeriodicDomain(256, 256; arch=GPU())
```
"""
function PeriodicDomain(N::Integer...; L=nothing, dtype::Type=Float64,
                        arch::AbstractArchitecture=CPU(), dealias::Real=3/2)
    ndim = length(N)
    if ndim < 1 || ndim > 3
        throw(ArgumentError("PeriodicDomain supports 1D, 2D, or 3D (got $(ndim)D)"))
    end

    L_vals = L === nothing ? ntuple(_ -> 2π, ndim) : Tuple(L)
    if length(L_vals) != ndim
        throw(ArgumentError("Length of L=$(length(L_vals)) must match number of dimensions=$ndim"))
    end

    coord_names = ("x", "y", "z")[1:ndim]
    coords = CartesianCoordinates(coord_names...)
    dist = Distributor(coords; dtype=dtype, architecture=arch)

    bases = ntuple(ndim) do i
        RealFourier(coords[coord_names[i]]; size=Int(N[i]),
                    bounds=(0.0, Float64(L_vals[i])),
                    dealias=Float64(dealias), dtype=dtype)
    end

    return Domain(dist, bases)
end

"""
    ChebyshevDomain(N...; bounds=ntuple(_->(-1.0, 1.0), length(N)), dtype=Float64, arch=CPU())

Create a Chebyshev domain with a single call.

# Examples
```julia
# 1D Chebyshev on [-1, 1]
domain = ChebyshevDomain(64)

# 2D Chebyshev box
domain = ChebyshevDomain(64, 64; bounds=((0.0, 1.0), (0.0, 2.0)))
```
"""
function ChebyshevDomain(N::Integer...; bounds=nothing, dtype::Type=Float64,
                         arch::AbstractArchitecture=CPU(), dealias::Real=1.0)
    ndim = length(N)
    if ndim < 1 || ndim > 3
        throw(ArgumentError("ChebyshevDomain supports 1D, 2D, or 3D (got $(ndim)D)"))
    end

    bounds_vals = bounds === nothing ? ntuple(_ -> (-1.0, 1.0), ndim) : Tuple(bounds)

    coord_names = ("x", "y", "z")[1:ndim]
    coords = CartesianCoordinates(coord_names...)
    dist = Distributor(coords; dtype=dtype, architecture=arch)

    bases = ntuple(ndim) do i
        ChebyshevT(coords[coord_names[i]]; size=Int(N[i]),
                   bounds=Float64.(bounds_vals[i]),
                   dealias=Float64(dealias), dtype=dtype)
    end

    return Domain(dist, bases)
end

"""
    ChannelDomain(Nx, Nz; Lx=2π, Lz=2.0, dtype=Float64, arch=CPU(), dealias=3/2)

Create a 2D channel domain: periodic in x (Fourier), bounded in z (Chebyshev).
Common setup for Rayleigh-Bénard, Poiseuille flow, etc.

# Example
```julia
domain = ChannelDomain(256, 64; Lx=4.0, Lz=1.0)
u = VectorField(domain, "u")
T = ScalarField(domain, "T")
```
"""
function ChannelDomain(Nx::Integer, Nz::Integer; Lx::Real=2π, Lz::Real=2.0,
                       dtype::Type=Float64, arch::AbstractArchitecture=CPU(),
                       dealias::Real=3/2)
    coords = CartesianCoordinates("x", "z")
    dist = Distributor(coords; dtype=dtype, architecture=arch)

    x_basis = RealFourier(coords["x"]; size=Int(Nx), bounds=(0.0, Float64(Lx)),
                          dealias=Float64(dealias), dtype=dtype)
    z_basis = ChebyshevT(coords["z"]; size=Int(Nz), bounds=(0.0, Float64(Lz)),
                         dealias=Float64(dealias), dtype=dtype)

    return Domain(dist, (x_basis, z_basis))
end

"""
    ChannelDomain3D(Nx, Ny, Nz; Lx=4π, Ly=2π, Lz=2.0, ...)

Create a 3D channel domain: periodic in x,y (Fourier), bounded in z (Chebyshev).
"""
function ChannelDomain3D(Nx::Integer, Ny::Integer, Nz::Integer;
                         Lx::Real=4π, Ly::Real=2π, Lz::Real=2.0,
                         dtype::Type=Float64, arch::AbstractArchitecture=CPU(),
                         dealias::Real=3/2)
    coords = CartesianCoordinates("x", "y", "z")
    dist = Distributor(coords; dtype=dtype, architecture=arch)

    x_basis = RealFourier(coords["x"]; size=Int(Nx), bounds=(0.0, Float64(Lx)),
                          dealias=Float64(dealias), dtype=dtype)
    y_basis = RealFourier(coords["y"]; size=Int(Ny), bounds=(0.0, Float64(Ly)),
                          dealias=Float64(dealias), dtype=dtype)
    z_basis = ChebyshevT(coords["z"]; size=Int(Nz), bounds=(-1.0, 1.0),
                         dealias=Float64(dealias), dtype=dtype)

    return Domain(dist, (x_basis, y_basis, z_basis))
end

# Convenience field constructors that take Domain instead of (dist, bases) pair
"""
    ScalarField(domain::Domain, name::String; dtype=domain.dist.dtype)

Create a ScalarField directly from a Domain.
"""
ScalarField(domain::Domain, name::String; dtype::Type=domain.dist.dtype) =
    ScalarField(domain.dist, name, domain.bases, dtype)

"""
    VectorField(domain::Domain, name::String; dtype=domain.dist.dtype)

Create a VectorField directly from a Domain.
"""
VectorField(domain::Domain, name::String; dtype::Type=domain.dist.dtype) =
    VectorField(domain.dist, domain.dist.coordsys, name, domain.bases, dtype)

"""
    TensorField(domain::Domain, name::String; dtype=domain.dist.dtype)

Create a TensorField directly from a Domain.
"""
TensorField(domain::Domain, name::String; dtype::Type=domain.dist.dtype) =
    TensorField(domain.dist, domain.dist.coordsys, name, domain.bases, dtype)

# ============================================================================
# Auto-transforming data access
# ============================================================================

"""
    grid_data(field::ScalarField)

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
    coeff_data(field::ScalarField)

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
    meshgrid = create_meshgrid(domain; on_device=false)
    # Extract arrays in basis order (meshgrid is a Dict{String, Array})
    grids = Tuple(meshgrid[b.meta.element_label] for b in domain.bases)
    data = f.(grids...)
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
