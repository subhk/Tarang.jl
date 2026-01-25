"""
Operator classes for spectral operations

Supports both CPU and GPU arrays. GPU operations use CUDA.jl when available.
"""

using LinearAlgebra
using LinearAlgebra: BLAS
using SparseArrays
using LoopVectorization  # For SIMD loops
using FFTW

# Operator registration tables (for parsing registries)
const OPERATOR_ALIASES = Dict{String, Any}()
const OPERATOR_PARSEABLES = Dict{String, Any}()
const OPERATOR_PREFIXES = Dict{String, Any}()

function register_operator_alias!(op, names::AbstractString...)
    for name in names
        OPERATOR_ALIASES[name] = op
    end
    return op
end

function register_operator_parseable!(op, names::AbstractString...)
    for name in names
        OPERATOR_PARSEABLES[name] = op
    end
    return op
end

function register_operator_prefix!(op, names::AbstractString...)
    for name in names
        OPERATOR_PREFIXES[name] = op
    end
    return op
end

# CPU-only (CPU support removed)

abstract type Operator <: Operand end

# Basic differential operators
struct Gradient <: Operator
    operand::Operand
    coordsys::CoordinateSystem
end
const _Gradient_constructor = Gradient

struct Divergence <: Operator
    operand::Operand
end
const _Divergence_constructor = Divergence

struct Curl <: Operator
    operand::Operand
    coordsys::CoordinateSystem
end
const _Curl_constructor = Curl

struct Laplacian <: Operator
    operand::Operand
end
const _Laplacian_constructor = Laplacian

# Fractional Laplacian: (-Δ)^α where α can be any real number
# In spectral space: (-Δ)^α f̂(k) = |k|^(2α) f̂(k)
# Common values: α=1/2 for SQG dissipation, α=-1/2 for SQG streamfunction inversion
struct FractionalLaplacian <: Operator
    operand::Operand
    α::Float64  # Fractional exponent

    function FractionalLaplacian(operand::Operand, α::Real)
        new(operand, Float64(α))
    end
end
const _FractionalLaplacian_constructor = FractionalLaplacian

struct Trace <: Operator
    operand::Operand
end
const _Trace_constructor = Trace

struct Skew <: Operator
    operand::Operand
end
const _Skew_constructor = Skew

struct TransposeComponents <: Operator
    operand::Operand
end
const _TransposeComponents_constructor = TransposeComponents

# Time derivative
struct TimeDerivative <: Operator
    operand::Operand
    order::Int
    
    function TimeDerivative(operand::Operand, order::Int=1)
        new(operand, order)
    end
end

# Interpolation and integration
struct Interpolate <: Operator
    operand::Operand
    coord::Coordinate
    position::Real
end
const _Interpolate_constructor = Interpolate

struct Integrate <: Operator
    operand::Operand
    coord::Union{Coordinate, Tuple{Vararg{Coordinate}}}
end
const _Integrate_constructor = Integrate

struct Average <: Operator
    operand::Operand
    coord::Coordinate
end
const _Average_constructor = Average

# Conversion operators
struct Convert <: Operator
    operand::Operand
    basis::Basis
end
const _Convert_constructor = Convert

struct Grid <: Operator
    operand::Operand
end
const _Grid_constructor = Grid

struct Coeff <: Operator
    operand::Operand
end
const _Coeff_constructor = Coeff

# Lifting operator for tau method boundary conditions
# Following Dedalus Lift operator (operators.py:4264-4309, basis.py:790-814)
# Creates polynomial P with coefficient 1 at mode n, returns P * operand
struct Lift <: Operator
    operand::Operand  # Field to lift (typically tau variable)
    basis::Basis       # Output basis (where to place lifted field)
    n::Int            # Mode index (negative recommended: -1=last, -2=second-last)
end
const _Lift_constructor = Lift

# Component extraction
struct Component <: Operator
    operand::Operand
    index::Int
end
const _Component_constructor = Component

struct RadialComponent <: Operator
    operand::Operand
end
const _RadialComponent_constructor = RadialComponent

struct AngularComponent <: Operator
    operand::Operand
end
const _AngularComponent_constructor = AngularComponent

struct AzimuthalComponent <: Operator
    operand::Operand
end
const _AzimuthalComponent_constructor = AzimuthalComponent

# Spectral differentiation
struct Differentiate <: Operator
    operand::Operand
    coord::Coordinate
    order::Int
end
const _Differentiate_constructor = Differentiate

# Outer product (tensor product of vectors)
struct Outer <: Operator
    left::Operand
    right::Operand
end
const _Outer_constructor = Outer

# AdvectiveCFL operator - computes grid-crossing frequency field
struct AdvectiveCFL <: Operator
    operand::Operand  # Velocity vector field
    coords::CoordinateSystem
end
const _AdvectiveCFL_constructor = AdvectiveCFL

# Constructor functions (following API)

"""Gradient operator."""
function grad(operand::Operand, coordsys::CoordinateSystem=operand.dist.coordsys)
    return multiclass_new(Gradient, operand, coordsys)
end

"""Divergence operator."""
function divergence(operand::Operand)
    return multiclass_new(Divergence, operand)
end

# Alias for convenience (but users should prefer `divergence` to avoid confusion with Base.div)
const div_op = divergence

"""Curl operator."""
function curl(operand::Operand, coordsys::CoordinateSystem=operand.dist.coordsys)
    return multiclass_new(Curl, operand, coordsys)
end

"""Laplacian operator."""
function lap(operand::Operand)
    return multiclass_new(Laplacian, operand)
end

# Unicode aliases for differential operators
# ∇ (nabla) for gradient
const ∇ = grad

# Δ (Delta) for Laplacian
const Δ = lap

# ∇² as alternate notation for Laplacian
const ∇² = lap

# Note: ∂t = dt alias is defined after the dt() function below

# Fractional Laplacian constructor
"""
    fraclap(operand, α)

Fractional Laplacian operator: (-Δ)^α

In spectral space, this multiplies by |k|^(2α) where k is the wavenumber magnitude.

# Common use cases:
- `fraclap(θ, 0.5)` or `(-Δ)^(1/2)`: SQG dissipation
- `fraclap(θ, -0.5)` or `(-Δ)^(-1/2)`: SQG streamfunction inversion (ψ = (-Δ)^(-1/2) θ)
- `fraclap(f, 1.0)`: Standard Laplacian (equivalent to `lap(f)`)

# Example - SQG equation:
```julia
# ∂θ/∂t + u⋅∇θ = κ(-Δ)^(1/2)θ
add_equation!(problem, "∂t(θ) = -u⋅∇(θ) + κ*fraclap(θ, 0.5)")

# Streamfunction from buoyancy: ψ = (-Δ)^(-1/2)θ
ψ = fraclap(θ, -0.5)
```
"""
function fraclap(operand::Operand, α::Real)
    return FractionalLaplacian(operand, α)
end

# Unicode alias for fractional Laplacian with specific exponents
# Note: For general α, use fraclap(f, α) or Δᵅ(f, α)
"""
    Δᵅ(operand, α)

Unicode alias for fractional Laplacian. Equivalent to `fraclap(operand, α)`.
Type: \\Delta Tab \\_\\alpha Tab
"""
const Δᵅ = fraclap

# Square root Laplacian: (-Δ)^(1/2), common in SQG
"""
    sqrtlap(operand)

Square root Laplacian: (-Δ)^(1/2)

Equivalent to `fraclap(operand, 0.5)`.
Common in Surface Quasi-Geostrophic (SQG) equations for dissipation.
"""
sqrtlap(operand::Operand) = fraclap(operand, 0.5)

# Inverse square root Laplacian: (-Δ)^(-1/2), for SQG streamfunction
"""
    invsqrtlap(operand)

Inverse square root Laplacian: (-Δ)^(-1/2)

Equivalent to `fraclap(operand, -0.5)`.
Used in SQG to compute streamfunction from buoyancy: ψ = (-Δ)^(-1/2)θ
"""
invsqrtlap(operand::Operand) = fraclap(operand, -0.5)

# Note: Unicode aliases like √Δ, Δ½, Δ⁻½ are not valid Julia identifiers
# (√ is a prefix operator, superscripts cause parsing issues)
# Use sqrtlap, invsqrtlap, or Δᵅ(f, α) instead

# ============================================================================
# Hyperviscosity / Higher-Order Laplacian Operators
# ============================================================================

"""
    hyperlap(operand, n::Integer)

Higher-order Laplacian (hyperviscosity) operator: (-Δ)^n

In Fourier space: (-Δ)^n f̂(k) = |k|^(2n) f̂(k)

Common use cases:
- `hyperlap(u, 2)` or `Δ²(u)`: Biharmonic operator for 4th-order hyperviscosity
- `hyperlap(u, 4)` or `Δ⁴(u)`: 8th-order hyperviscosity
- `hyperlap(u, 8)` or `Δ⁸(u)`: 16th-order hyperviscosity

For turbulence simulations with hyperviscosity dissipation:
```julia
# 4th-order hyperviscosity (biharmonic)
add_equation!(problem, "∂t(u) = -u⋅∇(u) - ν₄*Δ²(u)")

# 8th-order hyperviscosity
add_equation!(problem, "∂t(u) = -u⋅∇(u) - ν₈*Δ⁴(u)")
```

Note: For Fourier-based spectral methods, hyperviscosity is efficient because
it's just multiplication by |k|^(2n) in spectral space. For non-periodic
bases (Chebyshev), higher-order operators require more tau corrections.
"""
function hyperlap(operand::Operand, n::Integer)
    n >= 1 || throw(ArgumentError("hyperlap order n must be >= 1, got $n"))
    return FractionalLaplacian(operand, Float64(n))
end

# Unicode shortcuts for common hyperviscosity orders
# In spectral space: Δⁿ computes |k|^(2n) (same as (-Δ)^n for integer n≥1)

"""
    Δ²(operand)

Biharmonic operator: (-Δ)² = |k|⁴ in Fourier space.

Commonly used for 4th-order hyperviscosity in turbulence simulations:
```julia
add_equation!(problem, "∂t(ω) = -u⋅∇(ω) - ν₄*Δ²(ω)")
```

Type: \\Delta Tab \\^2 Tab
"""
Δ²(operand::Operand) = hyperlap(operand, 2)

"""
    Δ⁴(operand)

4th power Laplacian: (-Δ)⁴ = |k|⁸ in Fourier space.

Used for 8th-order hyperviscosity:
```julia
add_equation!(problem, "∂t(ω) = -u⋅∇(ω) - ν₈*Δ⁴(ω)")
```

Type: \\Delta Tab \\^4 Tab
"""
Δ⁴(operand::Operand) = hyperlap(operand, 4)

"""
    Δ⁶(operand)

6th power Laplacian: (-Δ)⁶ = |k|¹² in Fourier space.

Used for 12th-order hyperviscosity.

Type: \\Delta Tab \\^6 Tab
"""
Δ⁶(operand::Operand) = hyperlap(operand, 6)

"""
    Δ⁸(operand)

8th power Laplacian: (-Δ)⁸ = |k|¹⁶ in Fourier space.

Used for 16th-order hyperviscosity for very high Reynolds number simulations.

Type: \\Delta Tab \\^8 Tab
"""
Δ⁸(operand::Operand) = hyperlap(operand, 8)

# Register hyperviscosity operators for equation parsing
register_operator_parseable!(hyperlap, "hyperlap")

"""Trace operator."""
function trace(operand::Operand)
    return multiclass_new(Trace, operand)
end

"""Skew-symmetric part of tensor."""
function skew(operand::Operand)
    return multiclass_new(Skew, operand)
end

"""Transpose tensor components."""
function transpose_components(operand::Operand)
    return multiclass_new(TransposeComponents, operand)
end

"""Interpolate operand along a coordinate at a given position."""
function interpolate(operand::Operand, coord::Coordinate, position::Real)
    return multiclass_new(Interpolate, operand, coord, position)
end

"""Integrate operand over the specified coordinate(s)."""
function integrate(operand::Operand, coord::Union{Coordinate, Tuple{Vararg{Coordinate}}})
    return multiclass_new(Integrate, operand, coord)
end

"""Average operand along a coordinate."""
function average(operand::Operand, coord::Coordinate)
    return multiclass_new(Average, operand, coord)
end

"""Convert operand to a different basis."""
function convert_basis(operand::Operand, basis::Basis)
    return multiclass_new(Convert, operand, basis)
end

"""Time derivative."""
function dt(operand::Operand, order::Int=1)
    return TimeDerivative(operand, order)
end

# ∂t for time derivative (Unicode alias)
const ∂t = dt

# Differentiation functions
"""Differentiate with respect to coordinate."""
function d(operand::Operand, coord::Coordinate, order::Int=1)
    return multiclass_new(Differentiate, operand, coord, order)
end

"""
    lift(operand, basis, n)

Apply lifting operator for tau method boundary conditions.
Following Dedalus Lift operator (operators.py:4264-4309, basis.py:790-814).

Creates a polynomial P on the output basis with coefficient 1 at mode n,
then returns P * operand. This "lifts" the operand (typically a tau variable)
into spectral space at the specified mode.

Arguments:
- operand: The field to lift (typically a tau variable)
- basis: The output basis (where to place the lifted field)
- n: Mode index. Convention:
  - n < 0: Negative indexing from end (recommended for tau method)
    - n = -1: last mode (N)
    - n = -2: second-to-last mode (N-1)
  - n >= 0: Direct 0-indexed mode (less common)

Note: Dedalus only allows n < 0. We allow positive n for flexibility,
but negative indices are recommended for tau method boundary conditions.

Example:
```julia
# Typical tau method usage: lift tau to highest modes
tau = ScalarField(dist, "tau", ())  # Tau variable (no spectral basis)
lift_term = lift(tau, chebyshev_basis, -1)  # Lift to last Chebyshev mode
lift_term2 = lift(tau, chebyshev_basis, -2)  # Lift to second-to-last mode
```
"""
function lift(operand::Operand, basis::Basis, n::Int)
    return multiclass_new(Lift, operand, basis, n)
end

"""Extract component from vector/tensor field."""
function component(operand::Operand, index::Int)
    return multiclass_new(Component, operand, index)
end

"""Extract radial component from vector field."""
function radial(operand::Operand)
    return multiclass_new(RadialComponent, operand)
end

"""Extract angular component from vector field."""
function angular(operand::Operand)
    return multiclass_new(AngularComponent, operand)
end

"""Extract azimuthal component from vector field."""
function azimuthal(operand::Operand)
    return multiclass_new(AzimuthalComponent, operand)
end

"""Convert operand to grid space."""
function grid(operand::Operand)
    return multiclass_new(Grid, operand)
end

"""Convert operand to coefficient space."""
function coeff(operand::Operand)
    return multiclass_new(Coeff, operand)
end

"""
    outer(left, right)

Outer product (tensor product) of two vector fields.

Returns a rank-2 tensor field T where T_ij = u_i * v_j.

# Arguments
- `left`: First vector field (u)
- `right`: Second vector field (v)

# Returns
- TensorField with components T_ij = u_i * v_j
"""
function outer(left::Operand, right::Operand)
    return multiclass_new(Outer, left, right)
end

"""
    advective_cfl(velocity, coords)

Compute advective CFL grid-crossing frequency field.

Returns a scalar field representing the local grid-crossing frequency:
f = |u|/Δx + |v|/Δy + |w|/Δz

This is useful for adaptive timestepping where Δt < 1/max(f).

# Arguments
- `velocity`: Vector field representing fluid velocity
- `coords`: Coordinate system for the domain

# Returns
- ScalarField with CFL frequency at each grid point
"""
function advective_cfl(velocity::Operand, coords::CoordinateSystem)
    return multiclass_new(AdvectiveCFL, velocity, coords)
end

# Alias for convenience
const cfl = advective_cfl

# Register core operators for parsing namespace consistency
register_operator_alias!(grad, "grad", "gradient")
register_operator_parseable!(grad, "grad", "gradient")

register_operator_alias!(divergence, "div", "divergence")
register_operator_parseable!(divergence, "div", "divergence")

register_operator_alias!(curl, "curl")
register_operator_parseable!(curl, "curl")

register_operator_alias!(lap, "lap", "laplacian")
register_operator_parseable!(lap, "lap", "laplacian")

register_operator_alias!(trace, "trace")
register_operator_parseable!(trace, "trace")

register_operator_alias!(skew, "skew")
register_operator_parseable!(skew, "skew")

register_operator_alias!(transpose_components, "transpose_components", "transpose", "trans")
register_operator_parseable!(transpose_components, "transpose_components", "transpose", "trans")

register_operator_alias!(interpolate, "interpolate")
register_operator_parseable!(interpolate, "interpolate")

register_operator_alias!(integrate, "integrate", "integ")
register_operator_parseable!(integrate, "integrate", "integ")

register_operator_alias!(average, "average", "avg", "ave")
register_operator_parseable!(average, "average", "avg", "ave")

register_operator_alias!(convert_basis, "convert")
register_operator_parseable!(convert_basis, "convert")

register_operator_alias!(dt, "dt")
register_operator_parseable!(dt, "dt")

register_operator_alias!(d, "d", "differentiate")
register_operator_parseable!(d, "d", "differentiate")

register_operator_alias!(lift, "lift")
register_operator_parseable!(lift, "lift")

register_operator_alias!(component, "component", "comp")
register_operator_parseable!(component, "component", "comp")

register_operator_alias!(radial, "radial")
register_operator_parseable!(radial, "radial")

register_operator_alias!(angular, "angular")
register_operator_parseable!(angular, "angular")

register_operator_alias!(azimuthal, "azimuthal")
register_operator_parseable!(azimuthal, "azimuthal")

register_operator_alias!(grid, "grid")
register_operator_parseable!(grid, "grid")

register_operator_alias!(coeff, "coeff")
register_operator_parseable!(coeff, "coeff")

register_operator_alias!(outer, "outer", "tensor_product")
register_operator_parseable!(outer, "outer", "tensor_product")

register_operator_alias!(advective_cfl, "advective_cfl", "cfl")
register_operator_parseable!(advective_cfl, "advective_cfl", "cfl")

# Note: ∇ and ∇² are defined as const aliases to grad and lap above (lines ~213-216)
# Do not redefine here to avoid method overwrite warnings

# Include nonlinear terms for integration
# This will be included after nonlinear.jl is loaded

# Operator evaluation functions

"""Evaluate gradient operator."""
function evaluate_gradient(grad_op::Gradient, layout::Symbol=:g)
    operand = grad_op.operand
    coordsys = grad_op.coordsys
    
    if isa(operand, ScalarField)
        # Create vector field for result
        result = VectorField(operand.dist, coordsys, "grad_$(operand.name)", operand.bases, operand.dtype)
        
        # Compute partial derivatives for each component
        for (i, coord_name) in enumerate(coordsys.names)
            coord = coordsys[coord_name]
            # Apply differentiation operator
            result.components[i] = evaluate_differentiate(Differentiate(operand, coord, 1), layout)
        end
        
        return result
    else
        throw(ArgumentError("Gradient not implemented for operand type $(typeof(operand))"))
    end
end

"""Evaluate divergence operator."""
function evaluate_divergence(div_op::Divergence, layout::Symbol=:g)
    operand = div_op.operand
    
    if isa(operand, VectorField)
        # Sum partial derivatives of components
        coordsys = operand.coordsys
        result = ScalarField(operand.dist, "div_$(operand.name)", operand.bases, operand.dtype)
        
        # Initialize result to zero
        ensure_layout!(result, layout)
        fill!(result[string(layout)], 0)
        
        for (i, coord_name) in enumerate(coordsys.names)
            coord = coordsys[coord_name]
            # Add ∂u_i/∂x_i
            component_deriv = evaluate_differentiate(Differentiate(operand.components[i], coord, 1), layout)
            result = result + component_deriv
        end
        
        return result
    else
        throw(ArgumentError("Divergence not implemented for operand type $(typeof(operand))"))
    end
end

"""Evaluate differentiation operator."""
function evaluate_differentiate(diff_op::Differentiate, layout::Symbol=:g)
    operand = diff_op.operand
    coord = diff_op.coord
    order = diff_op.order
    
    if !isa(operand, ScalarField)
        throw(ArgumentError("Differentiation currently only supports scalar fields"))
    end
    
    # Find which basis corresponds to this coordinate
    basis_index = nothing
    for (i, basis) in enumerate(operand.bases)
        if basis.meta.element_label == coord.name
            basis_index = i
            break
        end
    end
    
    if basis_index === nothing
        throw(ArgumentError("Coordinate $(coord.name) not found in field bases"))
    end
    
    basis = operand.bases[basis_index]
    result = ScalarField(operand.dist, "d$(order)_$(operand.name)_d$(coord.name)$(order)", operand.bases, operand.dtype)
    
    # Apply differentiation based on basis type
    if isa(basis, RealFourier) || isa(basis, ComplexFourier)
        evaluate_fourier_derivative!(result, operand, basis_index, order, layout)
    elseif isa(basis, ChebyshevT)
        evaluate_chebyshev_derivative!(result, operand, basis_index, order, layout)
    elseif isa(basis, Legendre)
        evaluate_legendre_derivative!(result, operand, basis_index, order, layout)
    else
        throw(ArgumentError("Differentiation not implemented for basis type $(typeof(basis))"))
    end
    
    return result
end

"""
    evaluate_fourier_derivative!(result, operand, axis, order, layout)

Evaluate Fourier derivative using FFT operations along a single axis.
Supports both CPU (FFTW) and GPU (CUFFT via extension) arrays.

This function computes spectral derivatives by:
1. Applying FFT along the specified axis only (not full N-D FFT)
2. Multiplying by (ik)^order for each wavenumber k along that axis
3. Applying inverse FFT along the same axis to get derivative in grid space

This correctly handles multi-dimensional derivatives where we want d/dx
to only apply FFT along the x-axis, not all axes.
"""
function evaluate_fourier_derivative!(result::ScalarField, operand::ScalarField, axis::Int, order::Int, layout::Symbol)
    # Ensure operand is in grid space
    if operand.current_layout != :g
        @warn "evaluate_fourier_derivative!: operand not in grid space, results may be unexpected"
    end

    # Get the basis for the specified axis
    basis = operand.bases[axis]
    N = basis.meta.size
    L = basis.meta.bounds[2] - basis.meta.bounds[1]

    # Use grid data for computation
    data_g = copy(get_grid_data(operand))  # Make a copy to avoid modifying original (works for CPU and GPU)
    dims = ndims(data_g)
    data_shape = size(data_g)

    # Check if we're on GPU
    use_gpu = is_gpu_array(data_g)

    # Compute wavenumbers for the axis
    # For periodic FFT: k = [0, 1, ..., N/2-1, -N/2, ..., -1] * (2π/L)
    k_axis_cpu = fftfreq(data_shape[axis], data_shape[axis]/L) .* 2π

    # Build derivative multiplier array (ik)^order
    deriv_mult_cpu = (im .* k_axis_cpu) .^ order

    if use_gpu
        # GPU path: use broadcasting for all operations
        evaluate_fourier_derivative_gpu!(result, data_g, deriv_mult_cpu, axis, dims, data_shape, layout)
    else
        # CPU path: optimized with explicit loops
        evaluate_fourier_derivative_cpu!(result, data_g, deriv_mult_cpu, axis, dims, data_shape, layout)
    end
end

"""
    evaluate_fourier_derivative_gpu!(result, data_g, deriv_mult, axis, dims, data_shape, layout)

GPU-specific implementation using broadcasting operations.
"""
function evaluate_fourier_derivative_gpu!(result::ScalarField, data_g::AbstractArray, deriv_mult_cpu::AbstractVector, axis::Int, dims::Int, data_shape::Tuple, layout::Symbol)
    # Move derivative multiplier to GPU
    deriv_mult = copy_to_device(deriv_mult_cpu, data_g)

    # GPU FFT and element-wise operations via broadcasting
    # Note: For GPU, fft/ifft dispatch to CUFFT when CUDA.jl is loaded
    if dims == 1
        f_hat = fft(data_g)
        f_hat .*= deriv_mult
        deriv_g = real.(ifft(f_hat))
        get_grid_data(result) .= deriv_g
        result.current_layout = :g

    elseif dims == 2
        f_hat = fft(data_g, axis)

        # Reshape deriv_mult for broadcasting along the correct axis
        if axis == 1
            # Shape: (N, 1) to broadcast along first dimension
            mult_shaped = reshape(deriv_mult, :, 1)
        else  # axis == 2
            # Shape: (1, N) to broadcast along second dimension
            mult_shaped = reshape(deriv_mult, 1, :)
        end

        f_hat .*= mult_shaped
        deriv_g = real.(ifft(f_hat, axis))
        get_grid_data(result) .= deriv_g
        result.current_layout = :g

    elseif dims == 3
        f_hat = fft(data_g, axis)

        # Reshape deriv_mult for broadcasting along the correct axis
        if axis == 1
            mult_shaped = reshape(deriv_mult, :, 1, 1)
        elseif axis == 2
            mult_shaped = reshape(deriv_mult, 1, :, 1)
        else  # axis == 3
            mult_shaped = reshape(deriv_mult, 1, 1, :)
        end

        f_hat .*= mult_shaped
        deriv_g = real.(ifft(f_hat, axis))
        get_grid_data(result) .= deriv_g
        result.current_layout = :g
    else
        throw(ArgumentError("Fourier derivative only implemented for 1D, 2D, and 3D"))
    end

    # If coefficient space is requested, do full FFT on result
    if layout == :c
        get_coeff_data(result) .= fft(get_grid_data(result))
        result.current_layout = :c
    end
end

"""
    evaluate_fourier_derivative_cpu!(result, data_g, deriv_mult, axis, dims, data_shape, layout)

CPU-specific implementation using optimized loops.
"""
function evaluate_fourier_derivative_cpu!(result::ScalarField, data_g::AbstractArray, deriv_mult::AbstractVector, axis::Int, dims::Int, data_shape::Tuple, layout::Symbol)
    if dims == 1
        # 1D case - FFT along only dimension
        f_hat = fft(data_g)

        # Apply derivative: multiply by (ik)^order
        @inbounds for i in 1:length(f_hat)
            f_hat[i] *= deriv_mult[i]
        end

        # Inverse transform
        deriv_g = real.(ifft(f_hat))
        get_grid_data(result) .= deriv_g
        result.current_layout = :g

    elseif dims == 2
        # 2D case: apply FFT only along the specified axis
        f_hat = fft(data_g, axis)

        if axis == 1
            # Derivative along first axis - wavenumbers vary with first index
            @inbounds for i in 1:data_shape[1]
                factor = deriv_mult[i]
                for j in 1:data_shape[2]
                    f_hat[i, j] *= factor
                end
            end
        else  # axis == 2
            # Derivative along second axis - wavenumbers vary with second index
            @inbounds for j in 1:data_shape[2]
                factor = deriv_mult[j]
                for i in 1:data_shape[1]
                    f_hat[i, j] *= factor
                end
            end
        end

        # Inverse transform along same axis
        deriv_g = real.(ifft(f_hat, axis))
        get_grid_data(result) .= deriv_g
        result.current_layout = :g

    elseif dims == 3
        # 3D case: apply FFT only along the specified axis
        f_hat = fft(data_g, axis)

        if axis == 1
            @inbounds for i in 1:data_shape[1]
                factor = deriv_mult[i]
                for j in 1:data_shape[2], k in 1:data_shape[3]
                    f_hat[i, j, k] *= factor
                end
            end
        elseif axis == 2
            @inbounds for j in 1:data_shape[2]
                factor = deriv_mult[j]
                for i in 1:data_shape[1], k in 1:data_shape[3]
                    f_hat[i, j, k] *= factor
                end
            end
        else  # axis == 3
            @inbounds for k in 1:data_shape[3]
                factor = deriv_mult[k]
                for i in 1:data_shape[1], j in 1:data_shape[2]
                    f_hat[i, j, k] *= factor
                end
            end
        end

        # Inverse transform along same axis
        deriv_g = real.(ifft(f_hat, axis))
        get_grid_data(result) .= deriv_g
        result.current_layout = :g
    else
        throw(ArgumentError("Fourier derivative only implemented for 1D, 2D, and 3D"))
    end

    # If coefficient space is requested, do full FFT on result
    if layout == :c
        get_coeff_data(result) .= fft(get_grid_data(result))
        result.current_layout = :c
    end
end


"""
    evaluate_real_fourier_derivative_groups!(result, operand, axis, order, N, L)

Real Fourier derivative following 2x2 group matrix approach for multi-dimensional arrays.

Applies the derivative along the specified axis only.
For RealFourier, coefficients along axis are stored as [cos_0, cos_1, sin_1, cos_2, sin_2, ..., cos_nyq]

For f = a·cos(kx) + b·sin(kx):
  f' = -ka·sin(kx) + kb·cos(kx) = kb·cos(kx) - ka·sin(kx)

So the coefficient transformation is:
  new_cos_coeff = k * sin_coeff
  new_sin_coeff = -k * cos_coeff

This corresponds to the matrix on coefficients [a; b]:
  [a']   [ 0   k] [a]   [ k*b]
  [b'] = [-k   0] [b] = [-k*a]
"""
function evaluate_real_fourier_derivative_groups!(result::ScalarField, operand::ScalarField, axis::Int, order::Int, N::Int, L::Float64)
    # Initialize result to zero
    fill!(get_coeff_data(result), 0.0)

    # Process each wavenumber group
    k_max = N ÷ 2
    is_even = (N % 2 == 0)

    # Get array dimensions
    dims = ndims(get_coeff_data(operand))
    data_shape = size(get_coeff_data(operand))

    if dims == 1
        # 1D case: original implementation
        for k in 1:k_max-(is_even ? 1 : 0)
            k_phys = 2π * k / L
            cos_idx = 2*k
            sin_idx = 2*k + 1

            if cos_idx <= N && sin_idx <= N
                cos_coeff = get_coeff_data(operand)[cos_idx]
                sin_coeff = get_coeff_data(operand)[sin_idx]
                # d/dx(a*cos + b*sin) = kb*cos - ka*sin
                get_coeff_data(result)[cos_idx] =  k_phys * sin_coeff   # new_cos = k * b
                get_coeff_data(result)[sin_idx] = -k_phys * cos_coeff   # new_sin = -k * a
            end
        end

        # Handle Nyquist
        if is_even && N <= length(get_coeff_data(result))
            if order % 2 == 1
                get_coeff_data(result)[N] = 0.0
            else
                k_nyquist = 2π * k_max / L
                get_coeff_data(result)[N] = ((-1)^(order÷2)) * (k_nyquist^order) * get_coeff_data(operand)[N]
            end
        end
    elseif dims == 2
        # 2D case: apply derivative along specified axis
        if axis == 1
            # Derivative along first axis (x): loop over second axis
            for j in 1:data_shape[2]
                # DC component (k=0): derivative is always 0
                get_coeff_data(result)[1, j] = 0.0

                for k in 1:k_max-(is_even ? 1 : 0)
                    k_phys = 2π * k / L
                    cos_idx = 2*k
                    sin_idx = 2*k + 1

                    if cos_idx <= data_shape[1] && sin_idx <= data_shape[1]
                        cos_coeff = get_coeff_data(operand)[cos_idx, j]
                        sin_coeff = get_coeff_data(operand)[sin_idx, j]
                        # d/dx(a*cos + b*sin) = kb*cos - ka*sin
                        get_coeff_data(result)[cos_idx, j] =  k_phys * sin_coeff   # new_cos = k * b
                        get_coeff_data(result)[sin_idx, j] = -k_phys * cos_coeff   # new_sin = -k * a
                    end
                end

                # Handle Nyquist
                if is_even && N <= data_shape[1]
                    if order % 2 == 1
                        get_coeff_data(result)[N, j] = 0.0
                    else
                        k_nyquist = 2π * k_max / L
                        get_coeff_data(result)[N, j] = ((-1)^(order÷2)) * (k_nyquist^order) * get_coeff_data(operand)[N, j]
                    end
                end
            end
        else  # axis == 2
            # Derivative along second axis (y): loop over first axis
            for i in 1:data_shape[1]
                # DC component (k=0): derivative is always 0
                get_coeff_data(result)[i, 1] = 0.0

                for k in 1:k_max-(is_even ? 1 : 0)
                    k_phys = 2π * k / L
                    cos_idx = 2*k
                    sin_idx = 2*k + 1

                    if cos_idx <= data_shape[2] && sin_idx <= data_shape[2]
                        cos_coeff = get_coeff_data(operand)[i, cos_idx]
                        sin_coeff = get_coeff_data(operand)[i, sin_idx]
                        # d/dx(a*cos + b*sin) = kb*cos - ka*sin
                        get_coeff_data(result)[i, cos_idx] =  k_phys * sin_coeff   # new_cos = k * b
                        get_coeff_data(result)[i, sin_idx] = -k_phys * cos_coeff   # new_sin = -k * a
                    end
                end

                # Handle Nyquist
                if is_even && N <= data_shape[2]
                    if order % 2 == 1
                        get_coeff_data(result)[i, N] = 0.0
                    else
                        k_nyquist = 2π * k_max / L
                        get_coeff_data(result)[i, N] = ((-1)^(order÷2)) * (k_nyquist^order) * get_coeff_data(operand)[i, N]
                    end
                end
            end
        end
    elseif dims == 3
        # 3D case: apply derivative along specified axis
        if axis == 1
            for j in 1:data_shape[2], k3 in 1:data_shape[3]
                get_coeff_data(result)[1, j, k3] = 0.0
                for k in 1:k_max-(is_even ? 1 : 0)
                    k_phys = 2π * k / L
                    cos_idx = 2*k
                    sin_idx = 2*k + 1
                    if cos_idx <= data_shape[1] && sin_idx <= data_shape[1]
                        cos_coeff = get_coeff_data(operand)[cos_idx, j, k3]
                        sin_coeff = get_coeff_data(operand)[sin_idx, j, k3]
                        # d/dx(a*cos + b*sin) = kb*cos - ka*sin
                        get_coeff_data(result)[cos_idx, j, k3] =  k_phys * sin_coeff   # new_cos = k * b
                        get_coeff_data(result)[sin_idx, j, k3] = -k_phys * cos_coeff   # new_sin = -k * a
                    end
                end
                if is_even && N <= data_shape[1]
                    if order % 2 == 1
                        get_coeff_data(result)[N, j, k3] = 0.0
                    else
                        k_nyquist = 2π * k_max / L
                        get_coeff_data(result)[N, j, k3] = ((-1)^(order÷2)) * (k_nyquist^order) * get_coeff_data(operand)[N, j, k3]
                    end
                end
            end
        elseif axis == 2
            for i in 1:data_shape[1], k3 in 1:data_shape[3]
                get_coeff_data(result)[i, 1, k3] = 0.0
                for k in 1:k_max-(is_even ? 1 : 0)
                    k_phys = 2π * k / L
                    cos_idx = 2*k
                    sin_idx = 2*k + 1
                    if cos_idx <= data_shape[2] && sin_idx <= data_shape[2]
                        cos_coeff = get_coeff_data(operand)[i, cos_idx, k3]
                        sin_coeff = get_coeff_data(operand)[i, sin_idx, k3]
                        # d/dx(a*cos + b*sin) = kb*cos - ka*sin
                        get_coeff_data(result)[i, cos_idx, k3] =  k_phys * sin_coeff   # new_cos = k * b
                        get_coeff_data(result)[i, sin_idx, k3] = -k_phys * cos_coeff   # new_sin = -k * a
                    end
                end
                if is_even && N <= data_shape[2]
                    if order % 2 == 1
                        get_coeff_data(result)[i, N, k3] = 0.0
                    else
                        k_nyquist = 2π * k_max / L
                        get_coeff_data(result)[i, N, k3] = ((-1)^(order÷2)) * (k_nyquist^order) * get_coeff_data(operand)[i, N, k3]
                    end
                end
            end
        else  # axis == 3
            for i in 1:data_shape[1], j in 1:data_shape[2]
                get_coeff_data(result)[i, j, 1] = 0.0
                for k in 1:k_max-(is_even ? 1 : 0)
                    k_phys = 2π * k / L
                    cos_idx = 2*k
                    sin_idx = 2*k + 1
                    if cos_idx <= data_shape[3] && sin_idx <= data_shape[3]
                        cos_coeff = get_coeff_data(operand)[i, j, cos_idx]
                        sin_coeff = get_coeff_data(operand)[i, j, sin_idx]
                        # d/dx(a*cos + b*sin) = kb*cos - ka*sin
                        get_coeff_data(result)[i, j, cos_idx] =  k_phys * sin_coeff   # new_cos = k * b
                        get_coeff_data(result)[i, j, sin_idx] = -k_phys * cos_coeff   # new_sin = -k * a
                    end
                end
                if is_even && N <= data_shape[3]
                    if order % 2 == 1
                        get_coeff_data(result)[i, j, N] = 0.0
                    else
                        k_nyquist = 2π * k_max / L
                        get_coeff_data(result)[i, j, N] = ((-1)^(order÷2)) * (k_nyquist^order) * get_coeff_data(operand)[i, j, N]
                    end
                end
            end
        end
    else
        throw(ArgumentError("Real Fourier derivative not implemented for $(dims)D arrays"))
    end
end

"""Vectorized Real Fourier derivative implementation."""
function evaluate_real_fourier_derivative_vectorized!(result::ScalarField, operand::ScalarField, N::Int, L::Float64, k_max::Int, is_even::Bool)
    # Create wavenumber arrays
    k_range = 1:(k_max-(is_even ? 1 : 0))
    if !isempty(k_range)
        # Vectorized operations
        for k in k_range
            k_phys = 2π * k / L
            cos_idx = 2*k
            sin_idx = 2*k + 1

            if cos_idx <= length(get_coeff_data(operand)) && sin_idx <= length(get_coeff_data(operand))
                cos_coeff = get_coeff_data(operand)[cos_idx]
                sin_coeff = get_coeff_data(operand)[sin_idx]

                # d/dx(a*cos + b*sin) = kb*cos - ka*sin
                # Apply coefficient matrix: [0 k; -k 0]
                get_coeff_data(result)[cos_idx] =  k_phys * sin_coeff   # new_cos = k * b
                get_coeff_data(result)[sin_idx] = -k_phys * cos_coeff   # new_sin = -k * a
            end
        end
    end
end

"""Fallback implementation for higher-order derivatives."""
function evaluate_real_fourier_derivative_fallback!(result::ScalarField, operand::ScalarField, N::Int, L::Float64, k_max::Int, is_even::Bool, order::Int)
    for k in 1:k_max-(is_even ? 1 : 0)
        k_phys = 2π * k / L
        cos_idx = 2*k
        sin_idx = 2*k + 1

        if cos_idx <= length(get_coeff_data(operand)) && sin_idx <= length(get_coeff_data(operand))
            cos_coeff = get_coeff_data(operand)[cos_idx]
            sin_coeff = get_coeff_data(operand)[sin_idx]

            if order == 1
                # d/dx(a*cos + b*sin) = kb*cos - ka*sin
                get_coeff_data(result)[cos_idx] =  k_phys * sin_coeff   # new_cos = k * b
                get_coeff_data(result)[sin_idx] = -k_phys * cos_coeff   # new_sin = -k * a
            elseif order == 2
                # d²/dx²(a*cos + b*sin) = -k²a*cos - k²b*sin
                get_coeff_data(result)[cos_idx] = -k_phys^2 * cos_coeff
                get_coeff_data(result)[sin_idx] = -k_phys^2 * sin_coeff
            else
                # General case: use complex representation
                # f = a*cos(kx) + b*sin(kx) = Re[c * e^{ikx}] where c = a - ib
                # d^n/dx^n f = Re[(ik)^n * c * e^{ikx}]
                # New complex coefficient: c' = (ik)^n * (a - ib)
                complex_coeff = complex(cos_coeff, -sin_coeff)  # c = a - ib
                factor = (im * k_phys)^order
                result_complex = factor * complex_coeff
                # Extract new coefficients: c' = a' - ib' => a' = real(c'), b' = -imag(c')
                get_coeff_data(result)[cos_idx] = real(result_complex)
                get_coeff_data(result)[sin_idx] = -imag(result_complex)
            end
        end
    end
end

"""
    evaluate_complex_fourier_derivative!(result, operand, axis, order, N, L)

Complex Fourier derivative: multiplication by (ik)^order along specified axis.

For multi-dimensional arrays, applies the derivative only along the specified axis.
Complex FFT wavenumbers: [0, 1, ..., N/2-1, -N/2, -(N/2-1), ..., -1]
"""
function evaluate_complex_fourier_derivative!(result::ScalarField, operand::ScalarField, axis::Int, order::Int, N::Int, L::Float64)
    # Build wavenumber array for this axis
    k_vals = 2π/L * [0:(N÷2-1); -(N÷2):(-1)]

    # Get array dimensions
    dims = ndims(get_coeff_data(operand))
    data_shape = size(get_coeff_data(operand))

    if dims == 1
        # 1D case: original implementation
        for i in 1:length(get_coeff_data(operand))
            k_val = k_vals[i]
            factor = (im * k_val)^order
            get_coeff_data(result)[i] = get_coeff_data(operand)[i] * factor
        end
    elseif dims == 2
        # 2D case: apply derivative along specified axis
        if axis == 1
            for i in 1:data_shape[1]
                k_val = k_vals[i]
                factor = (im * k_val)^order
                for j in 1:data_shape[2]
                    get_coeff_data(result)[i, j] = get_coeff_data(operand)[i, j] * factor
                end
            end
        else  # axis == 2
            for j in 1:data_shape[2]
                k_val = k_vals[j]
                factor = (im * k_val)^order
                for i in 1:data_shape[1]
                    get_coeff_data(result)[i, j] = get_coeff_data(operand)[i, j] * factor
                end
            end
        end
    elseif dims == 3
        # 3D case: apply derivative along specified axis
        if axis == 1
            for i in 1:data_shape[1]
                k_val = k_vals[i]
                factor = (im * k_val)^order
                for j in 1:data_shape[2], k3 in 1:data_shape[3]
                    get_coeff_data(result)[i, j, k3] = get_coeff_data(operand)[i, j, k3] * factor
                end
            end
        elseif axis == 2
            for j in 1:data_shape[2]
                k_val = k_vals[j]
                factor = (im * k_val)^order
                for i in 1:data_shape[1], k3 in 1:data_shape[3]
                    get_coeff_data(result)[i, j, k3] = get_coeff_data(operand)[i, j, k3] * factor
                end
            end
        else  # axis == 3
            for k3 in 1:data_shape[3]
                k_val = k_vals[k3]
                factor = (im * k_val)^order
                for i in 1:data_shape[1], j in 1:data_shape[2]
                    get_coeff_data(result)[i, j, k3] = get_coeff_data(operand)[i, j, k3] * factor
                end
            end
        end
    else
        throw(ArgumentError("Complex Fourier derivative not implemented for $(dims)D arrays"))
    end
end

"""
    evaluate_chebyshev_derivative!(result, operand, axis, order, layout)

Evaluate Chebyshev derivative using direct DCT operations.
Supports both CPU and GPU arrays (GPU arrays are processed on CPU for DCT).

This function computes Chebyshev spectral derivatives by:
1. Applying DCT-I to grid data to get Chebyshev coefficients
2. Applying the Chebyshev derivative recurrence on coefficients
3. Applying DCT-I (inverse) to get derivative in grid space

For Chebyshev polynomials on [-1, 1]:
d/dx T_n(x) = n * U_{n-1}(x)
where U_n are Chebyshev polynomials of the second kind.

Using the recurrence relation for derivatives in terms of T_n:
c'_{n-1} = 2*n*c_n + c'_{n+1}  (backward recurrence)
"""
function evaluate_chebyshev_derivative!(result::ScalarField, operand::ScalarField, axis::Int, order::Int, layout::Symbol)
    # Get the basis for the specified axis
    basis = operand.bases[axis]
    N = basis.meta.size
    a, b = basis.meta.bounds

    # Domain transformation scale factor (for mapping [a,b] to [-1,1])
    scale = 2.0 / (b - a)

    # Use grid data for computation
    data_g = get_grid_data(operand)
    dims = ndims(data_g)
    data_shape = size(data_g)

    # Check if we're on GPU - DCT requires CPU computation
    use_gpu = is_gpu_array(data_g)
    if use_gpu
        # Copy to CPU for DCT operations (CUFFT doesn't support DCT)
        data_g_cpu = Array(data_g)
    else
        data_g_cpu = data_g
    end

    if dims == 1
        # 1D case: use DCT directly
        deriv_g_cpu = chebyshev_derivative_1d(data_g_cpu, scale)
        if use_gpu
            get_grid_data(result) .= copy_to_device(deriv_g_cpu, get_grid_data(result))
        else
            get_grid_data(result) .= deriv_g_cpu
        end
        result.current_layout = :g

    elseif dims == 2
        # 2D case: apply derivative along specified axis only
        deriv_g_cpu = zeros(eltype(data_g_cpu), data_shape)

        if axis == 1
            # Derivative along first axis: process each column
            for j in 1:data_shape[2]
                col = data_g_cpu[:, j]
                deriv_g_cpu[:, j] .= chebyshev_derivative_1d(col, scale)
            end
        else  # axis == 2
            # Derivative along second axis: process each row
            # Get scale factor for axis 2
            basis2 = operand.bases[2]
            a2, b2 = basis2.meta.bounds
            scale2 = 2.0 / (b2 - a2)
            for i in 1:data_shape[1]
                row = data_g_cpu[i, :]
                deriv_g_cpu[i, :] .= chebyshev_derivative_1d(row, scale2)
            end
        end

        if use_gpu
            get_grid_data(result) .= copy_to_device(deriv_g_cpu, get_grid_data(result))
        else
            get_grid_data(result) .= deriv_g_cpu
        end
        result.current_layout = :g

    elseif dims == 3
        # 3D case
        deriv_g_cpu = zeros(eltype(data_g_cpu), data_shape)

        if axis == 1
            for j in 1:data_shape[2], k in 1:data_shape[3]
                col = data_g_cpu[:, j, k]
                deriv_g_cpu[:, j, k] .= chebyshev_derivative_1d(col, scale)
            end
        elseif axis == 2
            basis2 = operand.bases[2]
            a2, b2 = basis2.meta.bounds
            scale2 = 2.0 / (b2 - a2)
            for i in 1:data_shape[1], k in 1:data_shape[3]
                slice = data_g_cpu[i, :, k]
                deriv_g_cpu[i, :, k] .= chebyshev_derivative_1d(slice, scale2)
            end
        else  # axis == 3
            basis3 = operand.bases[3]
            a3, b3 = basis3.meta.bounds
            scale3 = 2.0 / (b3 - a3)
            for i in 1:data_shape[1], j in 1:data_shape[2]
                slice = data_g_cpu[i, j, :]
                deriv_g_cpu[i, j, :] .= chebyshev_derivative_1d(slice, scale3)
            end
        end

        if use_gpu
            get_grid_data(result) .= copy_to_device(deriv_g_cpu, get_grid_data(result))
        else
            get_grid_data(result) .= deriv_g_cpu
        end
        result.current_layout = :g
    else
        throw(ArgumentError("Chebyshev derivative only implemented for 1D, 2D, and 3D"))
    end

    # If coefficient space is requested, transform result
    if layout == :c
        # Apply forward DCT to transform grid values to Chebyshev coefficients
        # This must be done for each Chebyshev axis in the domain
        # Note: DCT requires CPU computation, so copy to CPU if on GPU

        # Get data on CPU for DCT operations
        if use_gpu
            result_data_cpu = Array(get_grid_data(result))
        else
            result_data_cpu = get_grid_data(result)
        end

        if dims == 1
            N_result = size(result_data_cpu, 1)
            coeffs = FFTW.r2r(result_data_cpu, FFTW.REDFT00)
            coeffs ./= (N_result - 1)
            coeffs[1] /= 2
            coeffs[end] /= 2
            if use_gpu
                get_coeff_data(result) .= copy_to_device(coeffs, get_coeff_data(result))
            else
                get_coeff_data(result) .= coeffs
            end
        elseif dims == 2
            # Apply DCT-I along each Chebyshev axis
            coeffs = copy(result_data_cpu)
            data_shape_coeff = size(coeffs)

            # Transform along axis 1 if it's a Chebyshev basis
            if operand.bases[1] isa Union{ChebyshevT, ChebyshevU, ChebyshevV}
                N1 = data_shape_coeff[1]
                for j in 1:data_shape_coeff[2]
                    col = coeffs[:, j]
                    col_dct = FFTW.r2r(col, FFTW.REDFT00)
                    col_dct ./= (N1 - 1)
                    col_dct[1] /= 2
                    col_dct[end] /= 2
                    coeffs[:, j] .= col_dct
                end
            end

            # Transform along axis 2 if it's a Chebyshev basis
            if operand.bases[2] isa Union{ChebyshevT, ChebyshevU, ChebyshevV}
                N2 = data_shape_coeff[2]
                for i in 1:data_shape_coeff[1]
                    row = coeffs[i, :]
                    row_dct = FFTW.r2r(row, FFTW.REDFT00)
                    row_dct ./= (N2 - 1)
                    row_dct[1] /= 2
                    row_dct[end] /= 2
                    coeffs[i, :] .= row_dct
                end
            end

            if use_gpu
                get_coeff_data(result) .= copy_to_device(coeffs, get_coeff_data(result))
            else
                get_coeff_data(result) .= coeffs
            end
        elseif dims == 3
            # Apply DCT-I along each Chebyshev axis
            coeffs = copy(result_data_cpu)
            data_shape_coeff = size(coeffs)

            # Transform along axis 1 if it's a Chebyshev basis
            if operand.bases[1] isa Union{ChebyshevT, ChebyshevU, ChebyshevV}
                N1 = data_shape_coeff[1]
                for j in 1:data_shape_coeff[2], k in 1:data_shape_coeff[3]
                    col = coeffs[:, j, k]
                    col_dct = FFTW.r2r(col, FFTW.REDFT00)
                    col_dct ./= (N1 - 1)
                    col_dct[1] /= 2
                    col_dct[end] /= 2
                    coeffs[:, j, k] .= col_dct
                end
            end

            # Transform along axis 2 if it's a Chebyshev basis
            if operand.bases[2] isa Union{ChebyshevT, ChebyshevU, ChebyshevV}
                N2 = data_shape_coeff[2]
                for i in 1:data_shape_coeff[1], k in 1:data_shape_coeff[3]
                    slice = coeffs[i, :, k]
                    slice_dct = FFTW.r2r(slice, FFTW.REDFT00)
                    slice_dct ./= (N2 - 1)
                    slice_dct[1] /= 2
                    slice_dct[end] /= 2
                    coeffs[i, :, k] .= slice_dct
                end
            end

            # Transform along axis 3 if it's a Chebyshev basis
            if operand.bases[3] isa Union{ChebyshevT, ChebyshevU, ChebyshevV}
                N3 = data_shape_coeff[3]
                for i in 1:data_shape_coeff[1], j in 1:data_shape_coeff[2]
                    slice = coeffs[i, j, :]
                    slice_dct = FFTW.r2r(slice, FFTW.REDFT00)
                    slice_dct ./= (N3 - 1)
                    slice_dct[1] /= 2
                    slice_dct[end] /= 2
                    coeffs[i, j, :] .= slice_dct
                end
            end

            if use_gpu
                get_coeff_data(result) .= copy_to_device(coeffs, get_coeff_data(result))
            else
                get_coeff_data(result) .= coeffs
            end
        end
        result.current_layout = :c
    end

    # Apply higher order derivatives by recursion if needed
    if order > 1
        temp = copy(get_grid_data(result))
        for _ in 2:order
            temp_field = ScalarField(operand.dist, "temp", operand.bases, operand.dtype)
            get_grid_data(temp_field) .= temp
            temp_field.current_layout = :g
            evaluate_chebyshev_derivative!(result, temp_field, axis, 1, layout)
            temp = copy(get_grid_data(result))
        end
    end
end

"""
    chebyshev_derivative_1d(f, scale)

Compute the Chebyshev spectral derivative of a 1D array using DCT-I.

Arguments:
- f: Function values at Chebyshev-Gauss-Lobatto points
- scale: Domain scaling factor (2/(b-a) for domain [a,b])

Returns the derivative at the same grid points.

Note: The native grid for Chebyshev uses x_k = -cos(πk/(N-1)) which gives points
in ascending order (from -1 to +1). However, the standard DCT-I assumes points
cos(πk/(N-1)) in descending order (+1 to -1). To handle this, we reverse the
data before the DCT-I transform and reverse the result back.
"""
function chebyshev_derivative_1d(f::AbstractVector, scale::Float64)
    N = length(f)

    # Tarang uses ascending grid: x_k = -cos(πk/(N-1)) for k = 0, 1, ..., N-1
    # This goes from x_0 = -1 to x_{N-1} = +1
    #
    # Standard DCT-I convention uses descending grid: x_k = cos(πk/(N-1))
    # This goes from x_0 = +1 to x_{N-1} = -1
    #
    # To use DCT-I correctly with our ascending grid, we need to:
    # 1. Reverse f to get function values at standard descending grid points
    # 2. Compute derivative using standard algorithm
    # 3. Negate and reverse result to get derivative at ascending grid points
    #
    # The negation comes from: if g(y) = f(-y), then g'(y) = -f'(-y)

    f_std = reverse(f)  # Function values at standard (descending) grid points

    # Forward DCT-I to get Chebyshev coefficients
    coeffs_raw = FFTW.r2r(f_std, FFTW.REDFT00)

    # Normalize: DCT-I on N points needs (N-1) normalization
    # with boundary coefficients halved
    coeffs = copy(coeffs_raw)
    coeffs ./= (N - 1)
    coeffs[1] /= 2
    coeffs[end] /= 2

    # Apply Chebyshev derivative recurrence: c'_{k-1} = 2k * c_k + c'_{k+1}
    # In Julia's 1-indexing: c'[k] = 2*k * c[k+1] + c'[k+2]
    deriv_coeffs = zeros(eltype(coeffs), N)
    deriv_coeffs[N] = 0.0

    for k in (N-1):-1:1
        if k + 2 <= N
            deriv_coeffs[k] = 2 * k * coeffs[k + 1] + deriv_coeffs[k + 2]
        else
            deriv_coeffs[k] = 2 * k * coeffs[k + 1]
        end
    end

    # First coefficient has factor of 1/2 due to Chebyshev series normalization
    deriv_coeffs[1] /= 2

    # Apply domain scaling (on standard [-1, 1] domain, scale = 2/(b-a))
    deriv_coeffs .*= scale

    # Un-normalize for inverse DCT-I (multiply boundary coefficients by 2)
    deriv_coeffs[1] *= 2
    deriv_coeffs[end] *= 2

    # Inverse DCT-I and normalize to get derivative at standard (descending) grid
    deriv_std = FFTW.r2r(deriv_coeffs, FFTW.REDFT00) ./ 2

    # Convert derivative back to our ascending grid:
    # Just reverse to reorder points from ascending to match our grid ordering.
    # The derivative values are the same regardless of point ordering convention,
    # we just need to reorder them to match Tarang's ascending point convention.
    return reverse(deriv_std)
end

"""
    chebyshev_derivative_recurrence!(deriv_coeffs, coeffs, scale)

Apply the Chebyshev derivative recurrence relation:
c'_{n-1} = 2*n*c_n + c'_{n+1}  (backward recurrence)

For N coefficients (c_0 through c_{N-1} in 0-indexed notation):
- c'_{N-1} = 0 (highest derivative coefficient)
- c'_{N-2} = 2*(N-1)*c_{N-1}
"""
function chebyshev_derivative_recurrence!(deriv_coeffs::AbstractVector, coeffs::AbstractVector, scale::Float64)
    N = length(coeffs)

    # Backward recurrence for Chebyshev derivative
    # c'_{n-1} = 2*n*c_n + c'_{n+1}
    deriv_coeffs[N] = 0.0
    if N >= 2
        deriv_coeffs[N-1] = 2 * (N - 1) * coeffs[N]
    end

    for n in (N-2):-1:1
        # Recurrence: c'_{n-1} = 2*n*c_n + c'_{n+1} (in 0-indexed math)
        # In Julia 1-indexed: deriv_coeffs[n] = 2*n*coeffs[n+1] + deriv_coeffs[n+2]
        deriv_coeffs[n] = 2 * n * coeffs[n + 1] + deriv_coeffs[n + 2]
    end

    # First coefficient has factor of 1/2 due to normalization
    if N >= 1
        deriv_coeffs[1] *= 0.5
    end

    # Apply domain scaling
    deriv_coeffs .*= scale
end

"""
    evaluate_chebyshev_single_derivative!(result, operand, N, scale)

Single Chebyshev derivative using correct backward recurrence.

The standard Chebyshev derivative formula is:
c'_k = sum_{j=k+1, j-k odd} 2*j*c_j  for k >= 0

This implements the backward recurrence efficiently.
"""
function evaluate_chebyshev_single_derivative!(result::ScalarField, operand::ScalarField, N::Int, scale::Float64)
    # Initialize result to zero
    fill!(get_coeff_data(result), 0.0)
    
    if true  # CPU-only && N > 100 && length(get_coeff_data(operand)) > 100
        # CPU path without LoopVectorization to keep precompilation simple
        for k in 1:min(N, length(get_coeff_data(result)))
            deriv_sum = 0.0
            for j in (k+1):min(N, length(get_coeff_data(operand)))
                if (j - k) % 2 == 1  # j-k is odd
                    deriv_sum += 2.0 * (j - 1) * get_coeff_data(operand)[j]
                end
            end
            get_coeff_data(result)[k] = deriv_sum * scale
        end
    else
        # Vectorized or standard implementation
        for k in 1:min(N, length(get_coeff_data(result)))
            deriv_sum = 0.0
            
            # Apply the standard Chebyshev derivative recurrence:
            # c'_k = sum_{j=k+1, j-k odd} 2*j*c_j
            
            for j in (k+1):min(N, length(get_coeff_data(operand)))
                if (j - k) % 2 == 1  # j-k is odd
                    # Coefficient j corresponds to T_{j-1} polynomial (1-based indexing)
                    deriv_sum += 2.0 * (j - 1) * get_coeff_data(operand)[j]
                end
            end
            
            get_coeff_data(result)[k] = deriv_sum * scale
        end
    end
end

"""
    build_chebyshev_differentiation_matrix(N)

Build the Chebyshev differentiation matrix using the correct backward recurrence.

Uses the standard formula: c'_k = sum_{j=k+1, j-k odd} 2*j*c_j
"""
function build_chebyshev_differentiation_matrix(N::Int)
    D = zeros(Float64, N, N)
    
    # Apply the standard Chebyshev derivative recurrence:
    # result_coeff[k] = sum_j D[k,j] * input_coeff[j]
    # where D[k,j] = 2*j if j > k and (j-k) is odd, 0 otherwise
    
    for k in 1:N      # Output coefficient index  
        for j in 1:N  # Input coefficient index
            if j > k && (j - k) % 2 == 1  # j > k and j-k is odd
                # Coefficient j corresponds to T_{j-1} polynomial (1-based indexing)
                D[k, j] = 2.0 * (j - 1)
            end
        end
    end
    
    return sparse(D)
end

"""Evaluate Legendre derivative using compatible Jacobi implementation."""
function evaluate_legendre_derivative!(result::ScalarField, operand::ScalarField, axis::Int, order::Int, layout::Symbol)
    ensure_layout!(operand, :c)  # Work in coefficient space
    ensure_layout!(result, :c)

    basis = operand.bases[axis]
    N = basis.meta.size
    a, b = basis.meta.bounds

    # Domain transformation scale factor
    scale = 2.0 / (b - a)

    # Check if we're on GPU - recurrence requires CPU computation
    use_gpu = is_gpu_array(get_coeff_data(operand))

    # Apply multiple derivatives if order > 1
    if order == 1
        # Single derivative
        evaluate_legendre_single_derivative!(result, operand, N, scale, use_gpu)
    else
        # Multiple derivatives: apply single derivative 'order' times
        temp_field = ScalarField(operand.dist, "temp_deriv", operand.bases, operand.dtype)
        current_operand = operand

        for i in 1:order
            if i == order
                # Last iteration: store in result
                evaluate_legendre_single_derivative!(result, current_operand, N, scale, use_gpu)
            else
                # Intermediate iterations: use temp field
                evaluate_legendre_single_derivative!(temp_field, current_operand, N, scale, use_gpu)
                current_operand = temp_field
            end
        end
    end


    if layout == :g
        backward_transform!(result)
    end
end

"""
    evaluate_legendre_single_derivative!(result, operand, N, scale, use_gpu)

Single Legendre derivative using Jacobi approach.
Supports both CPU and GPU arrays (GPU arrays are processed on CPU).

Legendre polynomials are Jacobi polynomials with a=0, b=0.
From Jacobi D(+1): bands = [(N + a + b + 1) * 2^(-1)]
For Legendre: bands = [(N + 1) * 0.5]

The standard Legendre derivative recurrence relation is:
P'_n = (2n-1)*P_{n-1} + (2n-5)*P_{n-3} + (2n-9)*P_{n-5} + ...

This can be written in backward form for coefficient transformation.
"""
function evaluate_legendre_single_derivative!(result::ScalarField, operand::ScalarField, N::Int, scale::Float64, use_gpu::Bool=false)
    # For GPU arrays, copy to CPU for recurrence computation
    if use_gpu
        operand_data_cpu = Array(get_coeff_data(operand))
        result_data_cpu = zeros(eltype(operand_data_cpu), size(get_coeff_data(result)))
    else
        operand_data_cpu = get_coeff_data(operand)
        result_data_cpu = get_coeff_data(result)
        fill!(result_data_cpu, 0.0)
    end

    # Legendre spectral derivative formula:
    # If f(x) = Σ c_n P_n(x), then f'(x) = Σ c'_k P_k(x) where
    # c'_k = (2k+1) × Σ_{n>k, n-k odd} c_n  (in 0-indexed math notation)
    #
    # In Julia 1-indexed (where c[j] represents c_{j-1}):
    # c'[k] = (2(k-1)+1) × Σ_{j: j>k, j-k odd} c[j]
    #       = (2k-1) × Σ_{j: j>k, j-k odd} c[j]
    #
    # Note: The factor (2k-1) depends on the OUTPUT index k, not the input index j.
    # This is different from Chebyshev where the factor depends on the input index.

    @inbounds for k in 1:min(N, length(result_data_cpu))
        coeff_sum = 0.0
        for j in (k+1):min(N, length(operand_data_cpu))
            if (j - k) % 2 == 1  # j-k is odd
                coeff_sum += operand_data_cpu[j]
            end
        end
        # Factor (2k-1) = (2*(k-1)+1) is applied once to the sum
        result_data_cpu[k] = (2.0 * k - 1.0) * coeff_sum * scale
    end

    # Copy back to GPU if needed
    if use_gpu
        get_coeff_data(result) .= copy_to_device(result_data_cpu, get_coeff_data(result))
    end
end

# Helper functions for matrix application along array axes

"""
    apply_matrix_along_axis(matrix, array, axis; out=nothing)

Apply matrix along any axis of an array.
Following array:77-82 and apply_dense:104-126 implementation.
"""
function apply_matrix_along_axis(matrix::AbstractMatrix, array::AbstractArray, axis::Int; out=nothing)
    if issparse(matrix)
        return apply_sparse_along_axis(matrix, array, axis; out=out)
    else
        return apply_dense_along_axis(matrix, array, axis; out=out)
    end
end

"""
    apply_dense_along_axis(matrix, array, axis; out=nothing)

Apply dense matrix along any axis of an array.
Following apply_dense implementation in array:104-126.
"""
function apply_dense_along_axis(matrix::AbstractMatrix, array::AbstractArray, axis::Int; out=nothing)
    ndim = ndims(array)
    use_gpu = is_gpu_array(array)
    arch = architecture(array)
    array_cpu = use_gpu ? Array(array) : array

    # Resolve wraparound axis (convert to 1-based indexing)
    axis = mod1(axis, ndim)

    # Prepare output buffer on CPU
    out_is_gpu = out !== nothing && is_gpu_array(out)

    if out === nothing
        out_shape = collect(size(array_cpu))
        out_shape[axis] = size(matrix, 1)
        out_cpu = zeros(eltype(array_cpu), out_shape...)
    elseif out === array
        throw(ArgumentError("Cannot apply in place"))
    else
        out_cpu = out_is_gpu ? Array(out) : out
    end

    # Move target axis to position 1 (Julia's first dimension)
    if axis != 1
        perm = collect(1:ndim)
        perm[1] = axis
        perm[axis] = 1
        array_cpu = permutedims(array_cpu, perm)
    end

    # Get array shape after permutation
    array_shape = size(array_cpu)

    # Flatten later axes for matrix multiplication
    if ndim > 2
        array_cpu = reshape(array_cpu, (array_shape[1], prod(array_shape[2:end])))
    end

    # Apply matrix multiplication (CPU-compatible)
    temp = matrix * array_cpu

    # Unflatten later axes
    if ndim > 2
        new_shape = (size(temp, 1), array_shape[2:end]...)
        temp = reshape(temp, new_shape)
    end

    # Move axis back from position 1
    if axis != 1
        perm = collect(1:ndim)
        perm[1] = axis
        perm[axis] = 1
        temp = permutedims(temp, perm)
    end

    # Copy to output buffer on CPU
    copyto!(out_cpu, temp)

    if use_gpu || out_is_gpu
        if out === nothing
            return on_architecture(arch, out_cpu)
        else
            if out_is_gpu
                out .= copy_to_device(out_cpu, out)
            else
                copyto!(out, out_cpu)
            end
            return out
        end
    else
        return out_cpu
    end
end

"""
    apply_sparse_along_axis(matrix, array, axis; out=nothing, check_shapes=false)

Apply sparse matrix along any axis of an array.
Supports both CPU and GPU arrays (GPU arrays are copied to CPU for sparse operations).
Following apply_sparse implementation in array:171-203.
Note: Uses SparseMatrixCSC (Julia's sparse format) instead of CSR.
"""
function apply_sparse_along_axis(matrix::SparseMatrixCSC, array::AbstractArray, axis::Int; out=nothing, check_shapes=false)
    ndim = ndims(array)

    # Check if array is on GPU - sparse matrices require CPU
    use_gpu = is_gpu_array(array)
    if use_gpu
        array_cpu = Array(array)
    else
        array_cpu = array
    end

    # Resolve wraparound axis
    axis = mod1(axis, ndim)

    # Check output allocation (always on CPU for sparse operations)
    if out === nothing
        out_shape = collect(size(array_cpu))
        out_shape[axis] = size(matrix, 1)
        out_cpu = zeros(eltype(array_cpu), out_shape...)
    elseif out === array
        throw(ArgumentError("Cannot apply in place"))
    else
        out_cpu = use_gpu ? Array(out) : out
    end
    
    # Check shapes if requested
    if check_shapes
        if !(1 <= axis <= ndim)
            throw(BoundsError("Axis out of bounds"))
        end
        if size(matrix, 2) != size(array_cpu, axis) || size(matrix, 1) != size(out_cpu, axis)
            throw(DimensionMismatch("Matrix shape mismatch"))
        end
    end

    # Move target axis to position 1
    if axis != 1
        perm = collect(1:ndim)
        perm[1] = axis
        perm[axis] = 1
        array_cpu = permutedims(array_cpu, perm)
    end

    # Get array shape after permutation
    array_shape = size(array_cpu)

    # Flatten later axes
    if ndim > 2
        array_cpu = reshape(array_cpu, (array_shape[1], prod(array_shape[2:end])))
    end

    # Apply sparse matrix multiplication (CPU operation)
    temp = matrix * array_cpu

    # Unflatten later axes
    if ndim > 2
        new_shape = (size(temp, 1), array_shape[2:end]...)
        temp = reshape(temp, new_shape)
    end

    # Move axis back from position 1
    if axis != 1
        perm = collect(1:ndim)
        perm[1] = axis
        perm[axis] = 1
        temp = permutedims(temp, perm)
    end

    # Copy to output and handle GPU transfer if needed
    copyto!(out_cpu, temp)

    if use_gpu
        # Copy result back to GPU
        if out === nothing
            return copy_to_device(out_cpu, array)
        else
            out .= copy_to_device(out_cpu, out)
            return out
        end
    else
        return out_cpu
    end
end

# Operator arithmetic
Base.:+(op1::Operator, op2::Operator) = AddOperator(op1, op2)
Base.:-(op1::Operator, op2::Operator) = SubtractOperator(op1, op2)  
Base.:*(op1::Operator, op2::Union{Real, Operator}) = MultiplyOperator(op1, op2)

# ---------------------------------------------------------------------------
# Dispatch integration for core operators
# ---------------------------------------------------------------------------

function dispatch_preprocess(::Type{Gradient}, args::Tuple, kwargs::NamedTuple)
    if length(args) == 1
        operand = args[1]
        coordsys = hasfield(typeof(operand), :dist) ? operand.dist.coordsys : operand.coordsys
        return ((operand, coordsys), kwargs)
    elseif length(args) == 2
        return (args, kwargs)
    else
        throw(ArgumentError("Gradient expects 1 or 2 arguments"))
    end
end

function dispatch_check(::Type{Gradient}, args::Tuple, kwargs::NamedTuple)
    operand = args[1]
    if !isa(operand, Operand)
        throw(ArgumentError("Gradient requires an Operand"))
    end
    return true
end

function invoke_constructor(::Type{Gradient}, args::Tuple, kwargs::NamedTuple)
    operand, coordsys = args
    return _Gradient_constructor(operand, coordsys)
end

function dispatch_check(::Type{Divergence}, args::Tuple, kwargs::NamedTuple)
    operand = args[1]
    if !isa(operand, VectorField)
        throw(ArgumentError("Divergence requires a VectorField"))
    end
    return true
end

function invoke_constructor(::Type{Divergence}, args::Tuple, kwargs::NamedTuple)
    return _Divergence_constructor(args[1])
end

function dispatch_preprocess(::Type{Curl}, args::Tuple, kwargs::NamedTuple)
    if length(args) == 1
        operand = args[1]
        coordsys = hasfield(typeof(operand), :dist) ? operand.dist.coordsys : operand.coordsys
        return ((operand, coordsys), kwargs)
    elseif length(args) == 2
        return (args, kwargs)
    else
        throw(ArgumentError("Curl expects 1 or 2 arguments"))
    end
end

function dispatch_check(::Type{Curl}, args::Tuple, kwargs::NamedTuple)
    operand = args[1]
    if !isa(operand, VectorField)
        throw(ArgumentError("Curl requires a VectorField"))
    end
    return true
end

function invoke_constructor(::Type{Curl}, args::Tuple, kwargs::NamedTuple)
    operand, coordsys = args
    return _Curl_constructor(operand, coordsys)
end

function dispatch_check(::Type{Laplacian}, args::Tuple, kwargs::NamedTuple)
    operand = args[1]
    if !isa(operand, Operand)
        throw(ArgumentError("Laplacian requires an Operand"))
    end
    return true
end

function invoke_constructor(::Type{Laplacian}, args::Tuple, kwargs::NamedTuple)
    return _Laplacian_constructor(args[1])
end

function dispatch_check(::Type{Trace}, args::Tuple, kwargs::NamedTuple)
    operand = args[1]
    if !isa(operand, TensorField)
        throw(ArgumentError("Trace requires a TensorField"))
    end
    return true
end

function invoke_constructor(::Type{Trace}, args::Tuple, kwargs::NamedTuple)
    return _Trace_constructor(args[1])
end

function dispatch_check(::Type{Interpolate}, args::Tuple, kwargs::NamedTuple)
    operand, coord, position = args
    if !isa(operand, Operand)
        throw(ArgumentError("Interpolate requires an Operand"))
    end
    if !isa(coord, Coordinate)
        throw(ArgumentError("Interpolate requires a Coordinate"))
    end
    if !isa(position, Real)
        throw(ArgumentError("Interpolate position must be real"))
    end
    return true
end

function invoke_constructor(::Type{Interpolate}, args::Tuple, kwargs::NamedTuple)
    operand, coord, position = args
    return _Interpolate_constructor(operand, coord, position)
end

function dispatch_check(::Type{Integrate}, args::Tuple, kwargs::NamedTuple)
    operand, coord = args
    if !isa(operand, Operand)
        throw(ArgumentError("Integrate requires an Operand"))
    end
    return true
end

function invoke_constructor(::Type{Integrate}, args::Tuple, kwargs::NamedTuple)
    return _Integrate_constructor(args[1], args[2])
end

function dispatch_check(::Type{Average}, args::Tuple, kwargs::NamedTuple)
    operand, coord = args
    if !isa(operand, Operand)
        throw(ArgumentError("Average requires an Operand"))
    end
    if !isa(coord, Coordinate)
        throw(ArgumentError("Average requires a Coordinate"))
    end
    return true
end

function invoke_constructor(::Type{Average}, args::Tuple, kwargs::NamedTuple)
    return _Average_constructor(args[1], args[2])
end

function dispatch_preprocess(::Type{Differentiate}, args::Tuple, kwargs::NamedTuple)
    if length(args) == 2
        operand, coord = args
        return ((operand, coord, 1), kwargs)
    elseif length(args) == 3
        return (args, kwargs)
    else
        throw(ArgumentError("Differentiate expects 2 or 3 arguments"))
    end
end

function dispatch_check(::Type{Differentiate}, args::Tuple, kwargs::NamedTuple)
    operand, coord, order = args
    if !isa(operand, Operand)
        throw(ArgumentError("Differentiate requires an Operand"))
    end
    if !isa(coord, Coordinate)
        throw(ArgumentError("Differentiate requires a Coordinate"))
    end
    if !(order isa Integer) || order < 0
        throw(ArgumentError("Differentiate order must be a non-negative integer"))
    end
    return true
end

function invoke_constructor(::Type{Differentiate}, args::Tuple, kwargs::NamedTuple)
    operand, coord, order = args
    return _Differentiate_constructor(operand, coord, order)
end

function dispatch_check(::Type{Convert}, args::Tuple, kwargs::NamedTuple)
    operand, basis = args
    if !isa(operand, Operand)
        throw(ArgumentError("Convert requires an Operand"))
    end
    if !isa(basis, Basis)
        throw(ArgumentError("Convert requires a Basis"))
    end
    return true
end

function invoke_constructor(::Type{Convert}, args::Tuple, kwargs::NamedTuple)
    return _Convert_constructor(args[1], args[2])
end

function dispatch_check(::Type{Grid}, args::Tuple, kwargs::NamedTuple)
    if !isa(args[1], Operand)
        throw(ArgumentError("Grid conversion requires an Operand"))
    end
    return true
end

function dispatch_check(::Type{Coeff}, args::Tuple, kwargs::NamedTuple)
    if !isa(args[1], Operand)
        throw(ArgumentError("Coeff conversion requires an Operand"))
    end
    return true
end

function invoke_constructor(::Type{Grid}, args::Tuple, kwargs::NamedTuple)
    return _Grid_constructor(args[1])
end

function invoke_constructor(::Type{Coeff}, args::Tuple, kwargs::NamedTuple)
    return _Coeff_constructor(args[1])
end

function dispatch_check(::Type{Skew}, args::Tuple, kwargs::NamedTuple)
    operand = args[1]
    # Accept any Operand - type checking happens at evaluation time
    # Supports: TensorField (skew-symmetric part), VectorField (2D rotation),
    # and operators that produce these types (e.g., Gradient)
    if !isa(operand, Operand)
        throw(ArgumentError("Skew requires an Operand"))
    end
    return true
end

function invoke_constructor(::Type{Skew}, args::Tuple, kwargs::NamedTuple)
    return _Skew_constructor(args[1])
end

function dispatch_check(::Type{TransposeComponents}, args::Tuple, kwargs::NamedTuple)
    operand = args[1]
    if !isa(operand, TensorField)
        throw(ArgumentError("Transpose requires a TensorField"))
    end
    return true
end

function invoke_constructor(::Type{TransposeComponents}, args::Tuple, kwargs::NamedTuple)
    return _TransposeComponents_constructor(args[1])
end

function dispatch_check(::Type{Lift}, args::Tuple, kwargs::NamedTuple)
    operand, basis, n = args
    if !isa(operand, Operand)
        throw(ArgumentError("Lift requires an Operand"))
    end
    if !isa(basis, Basis)
        throw(ArgumentError("Lift requires a Basis"))
    end
    if !(n isa Integer)
        throw(ArgumentError("Lift mode index must be an integer"))
    end
    return true
end

function invoke_constructor(::Type{Lift}, args::Tuple, kwargs::NamedTuple)
    operand, basis, n = args
    return _Lift_constructor(operand, basis, n)
end

function dispatch_check(::Type{Component}, args::Tuple, kwargs::NamedTuple)
    operand, index = args
    if !isa(operand, Operand)
        throw(ArgumentError("Component extraction requires an Operand"))
    end
    if !(index isa Integer) || index < 1
        throw(ArgumentError("Component index must be a positive integer"))
    end
    return true
end

function invoke_constructor(::Type{Component}, args::Tuple, kwargs::NamedTuple)
    operand, index = args
    return _Component_constructor(operand, index)
end

function dispatch_check(::Type{RadialComponent}, args::Tuple, kwargs::NamedTuple)
    operand = args[1]
    if !isa(operand, VectorField)
        throw(ArgumentError("RadialComponent requires a VectorField"))
    end
    return true
end

function invoke_constructor(::Type{RadialComponent}, args::Tuple, kwargs::NamedTuple)
    return _RadialComponent_constructor(args[1])
end

function dispatch_check(::Type{AngularComponent}, args::Tuple, kwargs::NamedTuple)
    operand = args[1]
    if !isa(operand, VectorField)
        throw(ArgumentError("AngularComponent requires a VectorField"))
    end
    return true
end

function invoke_constructor(::Type{AngularComponent}, args::Tuple, kwargs::NamedTuple)
    return _AngularComponent_constructor(args[1])
end

function dispatch_check(::Type{AzimuthalComponent}, args::Tuple, kwargs::NamedTuple)
    operand = args[1]
    if !isa(operand, VectorField)
        throw(ArgumentError("AzimuthalComponent requires a VectorField"))
    end
    return true
end

function invoke_constructor(::Type{AzimuthalComponent}, args::Tuple, kwargs::NamedTuple)
    return _AzimuthalComponent_constructor(args[1])
end

struct AddOperator <: Operator
    left::Any
    right::Any
end

struct SubtractOperator <: Operator
    left::Any
    right::Any
end

struct MultiplyOperator <: Operator
    left::Any
    right::Any
end

# ============================================================================
# Expression matrices for matrix assembly (following operators)
# ============================================================================

"""
    expression_matrices(op::Operator, sp, vars; kwargs...)

Build expression matrices for operator applied to each variable.
Following operators expression_matrices method.

Returns Dict mapping variables to sparse matrices.
"""
function expression_matrices(op::Operator, sp, vars; kwargs...)
    # Default: return empty dict (override for specific operators)
    return Dict{Any, SparseMatrixCSC}()
end

"""
    expression_matrices(op::TimeDerivative, sp, vars; kwargs...)

Time derivative matrices: returns M matrix contribution.
Following operators TimeDerivative.expression_matrices.
"""
function expression_matrices(op::TimeDerivative, sp, vars; kwargs...)
    operand = op.operand
    result = Dict{Any, SparseMatrixCSC}()

    for var in vars
        if var === operand || (hasfield(typeof(var), :name) && hasfield(typeof(operand), :name) && var.name == operand.name)
            # Identity matrix for time derivative term
            n = field_dofs(var)
            result[var] = sparse(I, n, n) * Float64(op.order)
        end
    end

    return result
end

"""
    expression_matrices(op::Differentiate, sp, vars; kwargs...)

Spatial differentiation matrices.
Following operators Differentiate.expression_matrices.
"""
function expression_matrices(op::Differentiate, sp, vars; kwargs...)
    operand = op.operand
    coord = op.coord
    order = op.order
    result = Dict{Any, SparseMatrixCSC}()

    for var in vars
        if var === operand || (hasfield(typeof(var), :name) && hasfield(typeof(operand), :name) && var.name == operand.name)
            # Build differentiation matrix for this variable
            D = build_operator_differentiation_matrix(var, coord, order; kwargs...)
            if D !== nothing
                result[var] = D
            end
        end
    end

    return result
end

"""
    expression_matrices(op::Laplacian, sp, vars; kwargs...)

Laplacian matrices: sum of second derivatives.
Following operators Laplacian.expression_matrices.
"""
function expression_matrices(op::Laplacian, sp, vars; kwargs...)
    operand = op.operand
    result = Dict{Any, SparseMatrixCSC}()

    for var in vars
        if var === operand || (hasfield(typeof(var), :name) && hasfield(typeof(operand), :name) && var.name == operand.name)
            # Build Laplacian matrix = sum of D_i^2 for each coordinate
            lap_mat = nothing

            if hasfield(typeof(var), :bases)
                for basis in var.bases
                    if basis !== nothing
                        coord = get_coord_for_basis(basis)
                        D2 = build_operator_differentiation_matrix(var, coord, 2; kwargs...)
                        if D2 !== nothing
                            if lap_mat === nothing
                                lap_mat = D2
                            else
                                lap_mat = lap_mat + D2
                            end
                        end
                    end
                end
            end

            if lap_mat !== nothing
                result[var] = lap_mat
            end
        end
    end

    return result
end

"""
    expression_matrices(op::Gradient, sp, vars; kwargs...)

Gradient matrices for scalar -> vector.
Following operators Gradient.expression_matrices.
"""
function expression_matrices(op::Gradient, sp, vars; kwargs...)
    operand = op.operand
    coordsys = op.coordsys
    result = Dict{Any, SparseMatrixCSC}()

    for var in vars
        if var === operand || (hasfield(typeof(var), :name) && hasfield(typeof(operand), :name) && var.name == operand.name)
            # For Cartesian: gradient is vector of partial derivatives
            # Build block diagonal matrix with D_i for each component
            n = field_dofs(var)
            ndim = coordsys.dim

            blocks = SparseMatrixCSC[]
            for coord in coordsys.coords
                D = build_operator_differentiation_matrix(var, coord, 1; kwargs...)
                if D !== nothing
                    push!(blocks, D)
                else
                    push!(blocks, spzeros(Float64, n, n))
                end
            end

            # Stack blocks vertically for vector output
            if !isempty(blocks)
                result[var] = vcat(blocks...)
            end
        end
    end

    return result
end

"""
    expression_matrices(op::Divergence, sp, vars; kwargs...)

Divergence matrices for vector -> scalar.
Following operators Divergence.expression_matrices.
"""
function expression_matrices(op::Divergence, sp, vars; kwargs...)
    operand = op.operand
    result = Dict{Any, SparseMatrixCSC}()

    if !isa(operand, VectorField)
        return result
    end

    coordsys = operand.coordsys

    for var in vars
        if var === operand || (hasfield(typeof(var), :name) && var.name == operand.name)
            # For Cartesian: divergence is sum of partial derivatives of components
            # Build row of blocks [D_x, D_y, D_z]
            n_comp = length(operand.components)
            n_per_comp = n_comp > 0 ? field_dofs(operand.components[1]) : 0

            blocks = SparseMatrixCSC[]
            for (i, coord) in enumerate(coordsys.coords)
                comp = operand.components[i]
                D = build_operator_differentiation_matrix(comp, coord, 1; kwargs...)
                if D !== nothing
                    push!(blocks, D)
                else
                    push!(blocks, spzeros(Float64, n_per_comp, n_per_comp))
                end
            end

            # Concatenate blocks horizontally for scalar output
            if !isempty(blocks)
                result[var] = hcat(blocks...)
            end
        end
    end

    return result
end

"""
    expression_matrices(op::Lift, sp, vars; kwargs...)

Lift matrices for boundary conditions (tau method).
Following operators Lift.expression_matrices.
"""
function expression_matrices(op::Lift, sp, vars; kwargs...)
    operand = op.operand
    basis = op.basis
    n = op.n
    result = Dict{Any, SparseMatrixCSC}()

    for var in vars
        if var === operand || (hasfield(typeof(var), :name) && hasfield(typeof(operand), :name) && var.name == operand.name)
            # Lift matrix places tau values at specific spectral modes
            # Following tau method
            lift_mat = build_lift_matrix(var, basis, n; kwargs...)
            if lift_mat !== nothing
                result[var] = lift_mat
            end
        end
    end

    return result
end

"""
    expression_matrices(op::Convert, sp, vars; kwargs...)

Basis conversion matrices.
Following operators Convert.expression_matrices.
"""
function expression_matrices(op::Convert, sp, vars; kwargs...)
    operand = op.operand
    out_basis = op.basis
    result = Dict{Any, SparseMatrixCSC}()

    for var in vars
        if var === operand || (hasfield(typeof(var), :name) && hasfield(typeof(operand), :name) && var.name == operand.name)
            # Get input basis and build conversion matrix
            if hasfield(typeof(var), :bases) && !isempty(var.bases)
                in_basis = var.bases[1]
                if in_basis !== nothing && isa(in_basis, JacobiBasis) && isa(out_basis, JacobiBasis)
                    conv_mat = conversion_matrix(in_basis, out_basis)
                    result[var] = conv_mat
                end
            end
        end
    end

    return result
end

# ============================================================================
# Helper functions for building operator matrices
# ============================================================================

"""
    build_operator_differentiation_matrix(var, coord, order; kwargs...)

Build differentiation matrix for variable with respect to coordinate.
"""
function build_operator_differentiation_matrix(var, coord::Coordinate, order::Int; kwargs...)
    if !hasfield(typeof(var), :bases)
        return nothing
    end

    # Find the basis corresponding to this coordinate
    basis_idx = nothing
    target_basis = nothing

    for (i, basis) in enumerate(var.bases)
        if basis !== nothing && basis.meta.element_label == coord.name
            basis_idx = i
            target_basis = basis
            break
        end
    end

    if target_basis === nothing
        return nothing
    end

    n_total = field_dofs(var)
    n_basis = target_basis.meta.size

    # Build 1D differentiation matrix based on basis type
    D1d = nothing

    if isa(target_basis, JacobiBasis)
        D1d = differentiation_matrix(target_basis, order)
    elseif isa(target_basis, FourierBasis)
        D1d = fourier_differentiation_matrix(target_basis, order)
    end

    if D1d === nothing
        return nothing
    end

    # For multi-dimensional fields, apply Kronecker product
    if length(var.bases) == 1
        return D1d
    else
        # Build identity matrices for other dimensions
        matrices = AbstractMatrix[]
        for (i, basis) in enumerate(var.bases)
            if basis === nothing
                continue
            end
            if i == basis_idx
                push!(matrices, D1d)
            else
                push!(matrices, sparse(I, basis.meta.size, basis.meta.size))
            end
        end

        # Kronecker product in reverse order (column-major)
        result = matrices[end]
        for i in (length(matrices)-1):-1:1
            result = kron(result, matrices[i])
        end

        return result
    end
end

"""
    fourier_differentiation_matrix(basis::FourierBasis, order::Int)

Build Fourier differentiation matrix.
"""
function fourier_differentiation_matrix(basis::RealFourier, order::Int)
    N = basis.meta.size
    L = basis.meta.bounds[2] - basis.meta.bounds[1]
    k0 = 2π / L

    # RealFourier differentiation follows Dedalus convention:
    # Modes: [cos(0x), cos(1x), -sin(1x), cos(2x), -sin(2x), ...]
    # Note: Using -sin (msin) convention
    #
    # For differentiation with (ik)^order factor:
    # d/dx cos(kx) = -k sin(kx) = k * (-sin(kx))  -> k * msin
    # d/dx (-sin(kx)) = -k cos(kx)                -> -k * cos
    #
    # The 2x2 block for each wavenumber k is:
    # | 0  -k |   (maps: cos <- -k*msin, msin <- k*cos)
    # | k   0 |
    #
    # For order n, we apply this matrix n times, or equivalently compute (ik)^n
    # and extract the real/imaginary parts for the rotation.

    I_list = Int[]
    J_list = Int[]
    V_list = Float64[]

    # DC mode (k=0) -> 0 for any derivative order
    push!(I_list, 1)
    push!(J_list, 1)
    push!(V_list, 0.0)

    k_max = (N - 1) ÷ 2

    for k in 1:k_max
        cos_idx = 2*k      # 1-indexed: mode 2 is cos(1x), mode 4 is cos(2x), etc.
        sin_idx = 2*k + 1  # 1-indexed: mode 3 is -sin(1x), mode 5 is -sin(2x), etc.
        k_phys = k0 * k

        if cos_idx <= N && sin_idx <= N
            # Compute (ik)^order = k^order * i^order
            # i^0 = 1, i^1 = i, i^2 = -1, i^3 = -i, i^4 = 1, ...
            # For real representation: (ik)^n = k^n * (cos(nπ/2) + i*sin(nπ/2))
            #
            # The 2x2 derivative block D^n for the (cos, msin) pair:
            # D^1 = | 0  -k |    D^2 = |-k²  0  |    D^3 = | 0   k³ |   D^4 = |k⁴  0 |
            #       | k   0 |          | 0  -k² |          |-k³  0  |         | 0  k⁴|
            #
            # Pattern: D^n has form k^n * |cos(nπ/2)  -sin(nπ/2)|
            #                             |sin(nπ/2)   cos(nπ/2)|

            kn = k_phys^order
            phase = order * π / 2
            c = cos(phase)
            s = sin(phase)

            # Matrix entries for the 2x2 block:
            # cos_out = c * k^n * cos_in - s * k^n * msin_in
            # msin_out = s * k^n * cos_in + c * k^n * msin_in

            # (cos_idx, cos_idx): c * k^n
            if abs(c * kn) > 1e-15
                push!(I_list, cos_idx); push!(J_list, cos_idx); push!(V_list, c * kn)
            end
            # (cos_idx, sin_idx): -s * k^n
            if abs(s * kn) > 1e-15
                push!(I_list, cos_idx); push!(J_list, sin_idx); push!(V_list, -s * kn)
            end
            # (sin_idx, cos_idx): s * k^n
            if abs(s * kn) > 1e-15
                push!(I_list, sin_idx); push!(J_list, cos_idx); push!(V_list, s * kn)
            end
            # (sin_idx, sin_idx): c * k^n
            if abs(c * kn) > 1e-15
                push!(I_list, sin_idx); push!(J_list, sin_idx); push!(V_list, c * kn)
            end
        end
    end

    if isempty(I_list)
        return spzeros(Float64, N, N)
    end

    return sparse(I_list, J_list, V_list, N, N)
end

function fourier_differentiation_matrix(basis::ComplexFourier, order::Int)
    N = basis.meta.size
    L = basis.meta.bounds[2] - basis.meta.bounds[1]
    k0 = 2π / L

    # ComplexFourier: diagonal matrix with (ik)^order
    k_native = [0:(N÷2-1); -(N÷2):-1]
    k_phys = k0 .* k_native

    diag_vals = (im .* k_phys).^order

    return spdiagm(0 => diag_vals)
end

"""
    build_lift_matrix(var, basis, n; kwargs...)

Build lifting matrix for tau method boundary conditions.
Following the standard basis LiftJacobi implementation (lines 790-814).

the standard convention:
- n >= 0: sets mode n directly (0-indexed convention, 1-indexed in Julia)
- n < 0: wraps around (n = -1 means last mode, n = -2 means second-to-last, etc.)

Example: For N=10 modes
- Lift(tau, basis, 0) → sets mode 1 (Julia 1-indexed)
- Lift(tau, basis, -1) → sets mode N (last mode)
- Lift(tau, basis, -2) → sets mode N-1 (second-to-last mode)

Following Dedalus LiftJacobi pattern (basis.py:790-814):
The matrix places the tau variable's coefficient at mode n in the solution.
For LBVP solvers, this creates the "tau polynomial" that adds boundary
condition enforcement terms to the highest modes.
"""
function build_lift_matrix(var, basis, n::Int; kwargs...)
    N = basis.meta.size

    # Resolve mode index: negative wrap-around (Dedalus convention)
    # n < 0: index from end (e.g., -1 → N-1 in 0-indexed → N in 1-indexed)
    lift_mode = n
    if lift_mode < 0
        lift_mode = N + lift_mode
    end
    lift_mode += 1  # Convert 0-indexed to 1-indexed Julia convention

    if lift_mode < 1 || lift_mode > N
        n_total = hasfield(typeof(var), :bases) ? max(1, field_dofs(var)) : 1
        return spzeros(Float64, N * (n_total ÷ max(1, n_total)), n_total)
    end

    # Build the 1D lift column vector: e_{lift_mode} of size (N, 1)
    e_lift = sparse([lift_mode], [1], [1.0], N, 1)

    # If var has no bases or only the lift basis, return 1D lift vector
    if !hasfield(typeof(var), :bases) || isempty(var.bases) || all(b -> b === nothing, var.bases)
        return e_lift
    end

    # Find which var basis (if any) matches the lift basis coordinate
    lift_coord = basis.meta.element_label
    basis_idx = nothing
    for (i, b) in enumerate(var.bases)
        if b !== nothing && b.meta.element_label == lift_coord
            basis_idx = i
            break
        end
    end

    # Multi-dimensional case: build Kronecker product
    # Following the same convention as build_operator_differentiation_matrix:
    # result = kron(matrices[end], kron(matrices[end-1], ...kron(matrices[2], matrices[1])))
    #
    # For each dimension of the EQUATION space:
    # - If it's the lift dimension and tau doesn't have it: use e_lift (N×1)
    # - If it's the lift dimension and tau has it: use e_lift (N × N_var_basis)
    # - If tau has this basis: use identity I(basis_size)
    #
    # The resulting matrix maps from tau DOFs to equation DOFs.

    if basis_idx !== nothing
        # Tau variable already has the lift basis — rare case
        # Matrix is (N, var_basis_size) with 1 at row lift_mode for each col
        # This zeroes everything except the lift_mode row
        var_basis_size = var.bases[basis_idx].meta.size
        lift_1d = sparse([lift_mode], [lift_mode], [1.0], N, var_basis_size)

        if length(var.bases) == 1
            return lift_1d
        end

        # Kronecker product for multi-dimensional
        matrices = AbstractMatrix[]
        for (i, b) in enumerate(var.bases)
            if b === nothing
                continue
            end
            if i == basis_idx
                push!(matrices, lift_1d)
            else
                push!(matrices, sparse(LinearAlgebra.I, b.meta.size, b.meta.size))
            end
        end

        result = matrices[end]
        for i in (length(matrices)-1):-1:1
            result = kron(result, matrices[i])
        end
        return result
    else
        # Tau variable does NOT have the lift basis — standard tau method case
        # The lift basis is an extra dimension added to the tau variable
        # Matrix maps from tau DOFs to (tau_tangential × lift_basis) DOFs
        #
        # We need to determine WHERE the lift basis goes in the equation's
        # dimension ordering. We use the basis coordinate to infer position.

        # Collect the tau variable's existing basis sizes
        tangential_bases = [(i, b) for (i, b) in enumerate(var.bases) if b !== nothing]

        if isempty(tangential_bases)
            # Scalar tau (no bases) — just the lift vector
            return e_lift
        end

        # Determine insertion position for lift basis among existing bases
        # Convention: bases are ordered by coordinate, lift basis goes at its
        # natural position. For simplicity, we append it at the end (last dim).
        # This matches Dedalus where the bounded direction is typically last.
        matrices = AbstractMatrix[]
        for (i, b) in tangential_bases
            push!(matrices, sparse(LinearAlgebra.I, b.meta.size, b.meta.size))
        end
        push!(matrices, e_lift)

        # Kronecker product: result = kron(matrices[end], ...kron(matrices[2], matrices[1]))
        result = matrices[end]
        for i in (length(matrices)-1):-1:1
            result = kron(result, matrices[i])
        end
        return result
    end
end

"""
    get_coord_for_basis(basis::Basis)

Get coordinate associated with a basis.
"""
function get_coord_for_basis(basis::Basis)
    if hasfield(typeof(basis), :meta) && hasfield(typeof(basis.meta), :coordsys)
        coordsys = basis.meta.coordsys
        coord_name = basis.meta.element_label
        if hasfield(typeof(coordsys), :coords)
            for coord in coordsys.coords
                if coord.name == coord_name
                    return coord
                end
            end
        end
    end
    return nothing
end

"""
    field_dofs(field)

Get total degrees of freedom for a field.
"""
function field_dofs(field)
    if hasfield(typeof(field), :buffers) && get_coeff_data(field) !== nothing
        return length(get_coeff_data(field))
    elseif hasfield(typeof(field), :buffers) && get_grid_data(field) !== nothing
        return length(get_grid_data(field))
    elseif hasfield(typeof(field), :bases)
        total = 1
        for basis in field.bases
            if basis !== nothing
                total *= basis.meta.size
            end
        end
        return total
    elseif hasfield(typeof(field), :components)
        # VectorField or TensorField
        return sum(field_dofs(comp) for comp in field.components)
    end
    return 0
end

# ============================================================================
# Utility Operator Evaluation Functions
# Following operators implementation
# ============================================================================

"""
    evaluate_interpolate(interp_op::Interpolate, layout::Symbol=:g)

Evaluate interpolation operator at a specific position along a coordinate.
Following operators Interpolate implementation.

For Fourier bases: uses spectral interpolation (sum of modes)
For Jacobi bases: uses barycentric interpolation or Clenshaw algorithm
"""
function evaluate_interpolate(interp_op::Interpolate, layout::Symbol=:g)
    operand = interp_op.operand
    coord = interp_op.coord
    position = interp_op.position

    if !isa(operand, ScalarField)
        throw(ArgumentError("Interpolate currently only supports scalar fields"))
    end

    # Find which basis corresponds to this coordinate
    basis_index = nothing
    for (i, basis) in enumerate(operand.bases)
        if basis.meta.element_label == coord.name
            basis_index = i
            break
        end
    end

    if basis_index === nothing
        throw(ArgumentError("Coordinate $(coord.name) not found in field bases"))
    end

    basis = operand.bases[basis_index]

    # Work in coefficient space for spectral interpolation
    ensure_layout!(operand, :c)

    if isa(basis, RealFourier) || isa(basis, ComplexFourier)
        return interpolate_fourier(operand, basis, basis_index, position, layout)
    elseif isa(basis, JacobiBasis)
        return interpolate_jacobi(operand, basis, basis_index, position, layout)
    else
        throw(ArgumentError("Interpolation not implemented for basis type $(typeof(basis))"))
    end
end

"""
    interpolate_fourier(field, basis, axis, position, layout)

Interpolate Fourier field at a specific position using spectral reconstruction.
f(x) = Σ c_k exp(i k x) for ComplexFourier
f(x) = a_0 + Σ (a_k cos(kx) + b_k sin(kx)) for RealFourier
"""
function interpolate_fourier(field::ScalarField, basis::FourierBasis, axis::Int, position::Real, layout::Symbol)
    N = basis.meta.size
    L = basis.meta.bounds[2] - basis.meta.bounds[1]
    x0 = basis.meta.bounds[1]

    # Normalize position to [0, L)
    x = mod(position - x0, L)

    # Get coefficient data (copy to CPU if on GPU - interpolation uses scalar indexing)
    if is_gpu_array(get_coeff_data(field))
        coeffs = Array(get_coeff_data(field))
    else
        coeffs = get_coeff_data(field)
    end

    if isa(basis, RealFourier)
        # RealFourier: [a_0, a_1, b_1, a_2, b_2, ..., a_nyq]
        result = coeffs[1]  # DC component

        k_max = N ÷ 2
        is_even = (N % 2 == 0)

        for k in 1:(k_max - (is_even ? 1 : 0))
            k_phys = 2π * k / L
            cos_idx = 2*k
            sin_idx = 2*k + 1

            if cos_idx <= length(coeffs) && sin_idx <= length(coeffs)
                result += coeffs[cos_idx] * cos(k_phys * x)
                result += coeffs[sin_idx] * sin(k_phys * x)
            end
        end

        # Nyquist component for even N
        if is_even && N <= length(coeffs)
            k_nyq = 2π * k_max / L
            result += coeffs[N] * cos(k_nyq * x)
        end

    else  # ComplexFourier
        # ComplexFourier: standard FFT ordering [0, 1, ..., N/2-1, -N/2, ..., -1]
        result = complex(0.0, 0.0)

        for i in 1:N
            if i <= N÷2
                k = i - 1
            else
                k = i - N - 1
            end
            k_phys = 2π * k / L
            result += coeffs[i] * exp(im * k_phys * x)
        end

        result = real(result)
    end

    return result
end

"""
    interpolate_jacobi(field, basis, axis, position, layout)

Interpolate Jacobi-type field (Chebyshev, Legendre) using Clenshaw algorithm.
"""
function interpolate_jacobi(field::ScalarField, basis::JacobiBasis, axis::Int, position::Real, layout::Symbol)
    N = basis.meta.size
    a, b = basis.meta.bounds

    # Map position to native [-1, 1] interval
    x_native = 2.0 * (position - a) / (b - a) - 1.0

    # Clamp to valid range
    x_native = clamp(x_native, -1.0, 1.0)

    # Get coefficient data (copy to CPU if on GPU - Clenshaw uses scalar indexing)
    if is_gpu_array(get_coeff_data(field))
        coeffs = Array(get_coeff_data(field))
    else
        coeffs = get_coeff_data(field)
    end

    if isa(basis, ChebyshevT)
        # Clenshaw algorithm for Chebyshev T_n
        return clenshaw_chebyshev_t(coeffs, x_native)
    elseif isa(basis, ChebyshevU)
        # Clenshaw algorithm for Chebyshev U_n
        return clenshaw_chebyshev_u(coeffs, x_native)
    elseif isa(basis, Legendre)
        # Clenshaw algorithm for Legendre P_n
        return clenshaw_legendre(coeffs, x_native)
    else
        # General Jacobi: use direct polynomial evaluation
        return clenshaw_jacobi(coeffs, x_native, basis.a, basis.b)
    end
end

"""
    clenshaw_chebyshev_t(coeffs, x)

Clenshaw algorithm for evaluating Chebyshev T polynomial sum.
T_n(x) satisfies: T_{n+1}(x) = 2x T_n(x) - T_{n-1}(x)
"""
function clenshaw_chebyshev_t(coeffs::AbstractVector, x::Real)
    n = length(coeffs)
    if n == 0
        return 0.0
    elseif n == 1
        return coeffs[1]
    end

    # Backward recurrence: b_k = c_k + 2x b_{k+1} - b_{k+2}
    b_k2 = 0.0  # b_{n+1}
    b_k1 = 0.0  # b_{n}

    for k in n:-1:2
        b_k = coeffs[k] + 2x * b_k1 - b_k2
        b_k2 = b_k1
        b_k1 = b_k
    end

    # Final step: f(x) = c_0 + x*b_1 - b_2
    return coeffs[1] + x * b_k1 - b_k2
end

"""
    clenshaw_chebyshev_u(coeffs, x)

Clenshaw algorithm for evaluating Chebyshev U polynomial sum.
U_n(x) satisfies: U_{n+1}(x) = 2x U_n(x) - U_{n-1}(x)
"""
function clenshaw_chebyshev_u(coeffs::AbstractVector, x::Real)
    n = length(coeffs)
    if n == 0
        return 0.0
    elseif n == 1
        return coeffs[1]
    end

    # Same recurrence as T_n
    b_k2 = 0.0
    b_k1 = 0.0

    for k in n:-1:2
        b_k = coeffs[k] + 2x * b_k1 - b_k2
        b_k2 = b_k1
        b_k1 = b_k
    end

    # For U_n: f(x) = c_0 * U_0(x) + b_1 * U_1(x) - b_2 * U_0(x)
    # where U_0(x) = 1, U_1(x) = 2x
    return coeffs[1] + 2x * b_k1 - b_k2
end

"""
    clenshaw_legendre(coeffs, x)

Clenshaw algorithm for evaluating Legendre polynomial sum.
P_n(x) satisfies: (n+1) P_{n+1}(x) = (2n+1) x P_n(x) - n P_{n-1}(x)
"""
function clenshaw_legendre(coeffs::AbstractVector, x::Real)
    n = length(coeffs)
    if n == 0
        return 0.0
    elseif n == 1
        return coeffs[1]
    end

    # Backward recurrence for Legendre
    b_k2 = 0.0
    b_k1 = 0.0

    for k in n:-1:2
        # Recurrence: b_k = c_k + ((2k-1)/(k)) x b_{k+1} - (k/(k+1)) b_{k+2}
        # Note: k here is 0-indexed polynomial degree, so adjust
        deg = k - 1  # 0-indexed degree
        alpha = (2*deg + 1) / (deg + 1) * x
        beta = deg / (deg + 1)
        b_k = coeffs[k] + alpha * b_k1 - beta * b_k2
        b_k2 = b_k1
        b_k1 = b_k
    end

    # Final step
    return coeffs[1] + x * b_k1 - 0.5 * b_k2
end

"""
    clenshaw_jacobi(coeffs, x, a, b)

Clenshaw algorithm for evaluating general Jacobi polynomial sum.
"""
function clenshaw_jacobi(coeffs::AbstractVector, x::Real, a::Float64, b::Float64)
    n = length(coeffs)
    if n == 0
        return 0.0
    elseif n == 1
        return coeffs[1]
    end

    # Use direct evaluation for general Jacobi (less efficient but correct)
    result = 0.0
    for k in 1:n
        result += coeffs[k] * jacobi_polynomial(k-1, a, b, x)
    end
    return result
end

"""
    jacobi_polynomial(n, a, b, x)

Evaluate Jacobi polynomial P_n^{(a,b)}(x) using three-term recurrence.
"""
function jacobi_polynomial(n::Int, a::Float64, b::Float64, x::Real)
    if n == 0
        return 1.0
    elseif n == 1
        return 0.5 * (a - b + (a + b + 2) * x)
    end

    p_km2 = 1.0
    p_km1 = 0.5 * (a - b + (a + b + 2) * x)

    for k in 2:n
        # Three-term recurrence for Jacobi polynomials
        k_f = Float64(k)
        a1 = 2 * k_f * (k_f + a + b) * (2*k_f + a + b - 2)
        a2 = (2*k_f + a + b - 1) * (a^2 - b^2)
        a3 = (2*k_f + a + b - 2) * (2*k_f + a + b - 1) * (2*k_f + a + b)
        a4 = 2 * (k_f + a - 1) * (k_f + b - 1) * (2*k_f + a + b)

        p_k = ((a2 + a3 * x) * p_km1 - a4 * p_km2) / a1
        p_km2 = p_km1
        p_km1 = p_k
    end

    return p_km1
end

"""
    evaluate_integrate(int_op::Integrate, layout::Symbol=:g)

Evaluate integration operator over specified coordinate(s).
Following operators Integrate implementation.

Uses appropriate quadrature weights for each basis type:
- Fourier: trapezoidal rule (uniform weights)
- Chebyshev: Clenshaw-Curtis quadrature
- Legendre: Gauss-Legendre quadrature
"""
function evaluate_integrate(int_op::Integrate, layout::Symbol=:g)
    operand = int_op.operand
    coord = int_op.coord

    if !isa(operand, ScalarField)
        throw(ArgumentError("Integrate currently only supports scalar fields"))
    end

    # Handle single coordinate or tuple of coordinates
    coords = isa(coord, Coordinate) ? (coord,) : coord

    # Start with the operand
    result_field = copy(operand)

    # Integrate over each coordinate in sequence
    for c in coords
        result_field = integrate_along_coord(result_field, c)
    end

    # If all dimensions integrated, return scalar
    if length(coords) == length(operand.bases)
        ensure_layout!(result_field, :g)
        return sum(get_grid_data(result_field))
    end

    if layout == :g
        ensure_layout!(result_field, :g)
    else
        ensure_layout!(result_field, :c)
    end

    return result_field
end

"""
    integrate_along_coord(field, coord)

Integrate field along a single coordinate using appropriate quadrature.
"""
function integrate_along_coord(field::ScalarField, coord::Coordinate)
    # Find which basis corresponds to this coordinate
    basis_index = nothing
    for (i, basis) in enumerate(field.bases)
        if basis.meta.element_label == coord.name
            basis_index = i
            break
        end
    end

    if basis_index === nothing
        throw(ArgumentError("Coordinate $(coord.name) not found in field bases"))
    end

    basis = field.bases[basis_index]

    # Work in grid space for integration
    ensure_layout!(field, :g)

    # Get quadrature weights for this basis
    weights = get_integration_weights(basis)

    # Apply weighted sum along the axis
    data = get_grid_data(field)

    # Sum along the specified axis with weights
    result_data = integrate_weighted_sum(data, weights, basis_index)

    # Create result field with reduced dimensionality
    # For now, return scalar for 1D or the summed array
    if ndims(data) == 1
        return sum(data .* weights)
    else
        # Create new field without the integrated dimension
        new_bases = [b for (i, b) in enumerate(field.bases) if i != basis_index]
        if isempty(new_bases)
            return sum(data .* reshape(weights, size_for_axis(length(weights), basis_index, ndims(data))))
        end

        result = ScalarField(field.dist, "int_$(field.name)", tuple(new_bases...), field.dtype)
        set_grid_data!(result, result_data)
        result.current_layout = :g
        return result
    end
end

"""
    get_integration_weights(basis)

Get quadrature weights for integration over a basis.
"""
function get_integration_weights(basis::Basis)
    N = basis.meta.size
    a, b = basis.meta.bounds
    L = b - a

    if isa(basis, RealFourier) || isa(basis, ComplexFourier)
        # Uniform weights for periodic Fourier
        return fill(L / N, N)

    elseif isa(basis, ChebyshevT)
        # Clenshaw-Curtis quadrature weights
        return clenshaw_curtis_weights(N, L)

    elseif isa(basis, Legendre)
        # Gauss-Legendre quadrature weights
        _, weights = gauss_legendre_quadrature(N)
        return weights .* (L / 2)  # Scale from [-1,1] to [a,b]

    else
        # Default: uniform weights
        return fill(L / N, N)
    end
end

"""
    clenshaw_curtis_weights(N, L)

Compute Clenshaw-Curtis quadrature weights for Chebyshev integration.
"""
function clenshaw_curtis_weights(N::Int, L::Float64)
    weights = zeros(N)

    if N == 1
        return [L]
    end

    # Clenshaw-Curtis weights on [-1, 1]
    for j in 1:N
        theta_j = π * (j - 1) / (N - 1)
        w = 0.0
        for k in 0:(N÷2)
            if k == 0 || k == N÷2
                factor = 1.0
            else
                factor = 2.0
            end
            if 2*k != N - 1
                w += factor * cos(2*k * theta_j) / (1 - 4*k^2)
            end
        end
        weights[j] = 2 * w / (N - 1)
    end

    # Handle endpoints
    weights[1] *= 0.5
    weights[N] *= 0.5

    # Scale to interval [a, b]
    return weights .* (L / 2)
end

"""
    gauss_legendre_quadrature(N)

Compute Gauss-Legendre quadrature points and weights on [-1, 1].
"""
function gauss_legendre_quadrature(N::Int)
    points = zeros(N)
    weights = zeros(N)

    # Initial guesses for roots using Chebyshev nodes
    for i in 1:N
        points[i] = -cos(π * (i - 0.25) / (N + 0.5))
    end

    # Newton-Raphson iteration for roots
    for i in 1:N
        x = points[i]
        for _ in 1:100  # Max iterations
            # Evaluate P_N and P_{N-1} using recurrence
            p_km2 = 1.0
            p_km1 = x

            for k in 2:N
                p_k = ((2*k - 1) * x * p_km1 - (k - 1) * p_km2) / k
                p_km2 = p_km1
                p_km1 = p_k
            end

            # Derivative: P'_N = N(x P_N - P_{N-1}) / (x^2 - 1)
            dp = N * (x * p_km1 - p_km2) / (x^2 - 1)

            # Newton update
            dx = -p_km1 / dp
            x += dx

            if abs(dx) < 1e-14
                break
            end
        end

        points[i] = x

        # Compute weight
        p_km2 = 1.0
        p_km1 = x
        for k in 2:N
            p_k = ((2*k - 1) * x * p_km1 - (k - 1) * p_km2) / k
            p_km2 = p_km1
            p_km1 = p_k
        end
        dp = N * (x * p_km1 - p_km2) / (x^2 - 1)
        weights[i] = 2 / ((1 - x^2) * dp^2)
    end

    return points, weights
end

"""
    integrate_weighted_sum(data, weights, axis)

Apply weighted sum along specified axis.
"""
function integrate_weighted_sum(data::AbstractArray, weights::AbstractVector, axis::Int)
    nd = ndims(data)

    # Move weights to same device as data if needed
    if is_gpu_array(data)
        weights_device = copy_to_device(weights, data)
    else
        weights_device = weights
    end

    if nd == 1
        return sum(data .* weights_device)
    end

    # Reshape weights to broadcast along the correct axis
    shape = ones(Int, nd)
    shape[axis] = length(weights_device)
    w_shaped = reshape(weights_device, shape...)

    # Weighted sum along axis
    return dropdims(sum(data .* w_shaped, dims=axis), dims=axis)
end

"""
    size_for_axis(n, axis, ndims)

Create a size tuple with n in the specified axis position and 1 elsewhere.
"""
function size_for_axis(n::Int, axis::Int, nd::Int)
    shape = ones(Int, nd)
    shape[axis] = n
    return tuple(shape...)
end

"""
    evaluate_average(avg_op::Average, layout::Symbol=:g)

Evaluate averaging operator along a coordinate.
Average = Integrate / (interval length)
"""
function evaluate_average(avg_op::Average, layout::Symbol=:g)
    operand = avg_op.operand
    coord = avg_op.coord

    if !isa(operand, ScalarField)
        throw(ArgumentError("Average currently only supports scalar fields"))
    end

    # Find the basis for this coordinate
    basis_index = nothing
    for (i, basis) in enumerate(operand.bases)
        if basis.meta.element_label == coord.name
            basis_index = i
            break
        end
    end

    if basis_index === nothing
        throw(ArgumentError("Coordinate $(coord.name) not found in field bases"))
    end

    basis = operand.bases[basis_index]
    L = basis.meta.bounds[2] - basis.meta.bounds[1]

    # Integrate and divide by interval length
    int_result = integrate_along_coord(operand, coord)

    if isa(int_result, Real)
        return int_result / L
    else
        # Scale the field data
        if get_grid_data(int_result) !== nothing
            get_grid_data(int_result) ./= L
        end
        if get_coeff_data(int_result) !== nothing
            get_coeff_data(int_result) ./= L
        end
        return int_result
    end
end

"""
    evaluate_lift(lift_op::Lift, layout::Symbol=:g)

Evaluate lifting operator for tau method boundary conditions.
Following the Dedalus LiftJacobi implementation (basis.py:790-814).

The Lift operator creates a polynomial field P on the output basis with coefficient
at mode n set to 1, then returns P * operand. This "lifts" the operand (typically
a tau variable) into spectral space at the specified mode.

Convention (following Dedalus):
- n < 0: wraps around (n = -1 means last mode, n = -2 means second-to-last, etc.)
- n >= 0: sets mode n directly (0-indexed convention, 1-indexed in Julia)

The Dedalus implementation:
```python
def build_polynomial(dist, basis, n):
    if n < 0:
        n += basis.size
    P = dist.Field(bases=basis)
    axis = dist.get_basis_axis(basis)
    P['c'][axslice(axis, n, n+1)] = 1
    return P
# Then returns P * operand
```

Arguments:
- lift_op: Lift operator containing operand, output basis, and mode index n
- layout: Output layout (:g for grid, :c for coefficient)

Returns:
- ScalarField representing P * operand where P has coefficient 1 at mode n
"""
function evaluate_lift(lift_op::Lift, layout::Symbol=:g)
    operand = lift_op.operand
    output_basis = lift_op.basis  # The output basis (like Dedalus output_basis)
    n = lift_op.n

    if !isa(operand, ScalarField)
        throw(ArgumentError("Lift currently only supports scalar fields"))
    end

    # Get basis size
    N = output_basis.meta.size

    # Handle negative index wrap-around (Dedalus convention)
    # In Dedalus: if n < 0: n += basis.size
    lift_mode = n
    if lift_mode < 0
        lift_mode = N + lift_mode  # e.g., -1 → N-1 (0-indexed), then +1 for Julia
    end
    lift_mode += 1  # Convert from 0-indexed to 1-indexed Julia convention

    # Validate mode index
    if lift_mode < 1 || lift_mode > N
        throw(ArgumentError("Lift mode index $n (resolved to $lift_mode) out of bounds for basis size $N"))
    end

    # Find or create the output bases tuple
    output_bases = _get_lift_output_bases(operand, output_basis)

    # Step 1: Build polynomial P with coefficient 1 at mode n
    # Following Dedalus: P = dist.Field(bases=basis); P['c'][axslice(axis, n, n+1)] = 1
    P = ScalarField(operand.dist, "lift_poly", output_bases, operand.dtype)
    ensure_layout!(P, :c)

    # Find which axis corresponds to the output basis
    basis_axis = _find_basis_axis(output_bases, output_basis)

    # Build P coefficients on CPU, then transfer to GPU if needed
    p_data = get_coeff_data(P)
    arch = operand.dist.architecture
    if is_gpu_array(p_data)
        # Build on CPU first, then copy to GPU
        cpu_p = zeros(eltype(p_data), size(p_data))
        if ndims(cpu_p) == 1
            cpu_p[lift_mode] = one(eltype(cpu_p))
        else
            selectdim(cpu_p, basis_axis, lift_mode) .= one(eltype(cpu_p))
        end
        copyto!(p_data, on_architecture(arch, cpu_p))
    else
        fill!(p_data, zero(eltype(p_data)))
        if ndims(p_data) == 1
            p_data[lift_mode] = one(eltype(p_data))
        else
            selectdim(p_data, basis_axis, lift_mode) .= one(eltype(p_data))
        end
    end

    # Step 2: Compute result = P * operand
    ensure_layout!(operand, :c)

    # Create result field
    result = ScalarField(operand.dist, "lift_$(operand.name)", output_bases, operand.dtype)
    ensure_layout!(result, :c)

    # Multiply P * operand: place operand's data at mode lift_mode
    _multiply_lift_polynomial!(get_coeff_data(result), get_coeff_data(P),
                               get_coeff_data(operand), basis_axis, lift_mode, arch)

    if layout == :g
        backward_transform!(result)
    end

    return result
end

"""
    _get_lift_output_bases(operand, output_basis)

Get output bases for lift operation, substituting input basis with output basis.
Following Dedalus: domain.substitute_basis(input_basis, output_basis)
"""
function _get_lift_output_bases(operand::ScalarField, output_basis::Basis)
    # If operand has no bases on the output coordinate, use output_basis
    output_coord = output_basis.meta.element_label

    new_bases = Vector{Any}(undef, length(operand.bases))
    found = false

    for (i, b) in enumerate(operand.bases)
        if b === nothing
            new_bases[i] = nothing
        elseif b.meta.element_label == output_coord
            new_bases[i] = output_basis
            found = true
        else
            new_bases[i] = b
        end
    end

    # If no matching basis found, this is a lift from no-basis to output_basis
    # In this case, we need to expand the field
    if !found
        # Add output_basis to the bases
        push!(new_bases, output_basis)
    end

    return tuple(new_bases...)
end

"""
    _find_basis_axis(bases, target_basis)

Find which axis (1-indexed) corresponds to the target basis.
"""
function _find_basis_axis(bases::Tuple, target_basis::Basis)
    for (i, b) in enumerate(bases)
        if b === target_basis ||
           (b !== nothing && b.meta.element_label == target_basis.meta.element_label)
            return i
        end
    end
    return 1  # Default to first axis
end

"""
    _set_lift_coefficient!(data, axis, mode, value)

Set coefficient at specified mode along axis to given value.
Equivalent to P['c'][axslice(axis, n, n+1)] = value
"""
function _set_lift_coefficient!(data::AbstractArray, axis::Int, mode::Int, value::Real)
    # Use selectdim to get view and set
    view = selectdim(data, axis, mode)
    fill!(view, value)
end

"""
    _multiply_lift_polynomial!(result, P_data, operand_data, basis_axis, lift_mode, arch)

Multiply lift polynomial P by operand.
P has a single non-zero coefficient at lift_mode.
Result = P * operand places operand's values at mode lift_mode.

GPU-compatible: avoids scalar indexing by building on CPU and copying,
or using broadcasting operations that work on GPU arrays.
"""
function _multiply_lift_polynomial!(result::AbstractArray, P_data::AbstractArray,
                                    operand_data::AbstractArray, basis_axis::Int,
                                    lift_mode::Int, arch=nothing)
    if is_gpu_array(result)
        # GPU path: build result on CPU, then copy to GPU
        cpu_result = zeros(eltype(result), size(result))
        cpu_operand = is_gpu_array(operand_data) ? Array(operand_data) : operand_data

        if ndims(cpu_result) == 1
            if length(cpu_operand) >= 1
                cpu_result[lift_mode] = cpu_operand[1]
            end
        else
            result_slice = selectdim(cpu_result, basis_axis, lift_mode)
            if ndims(cpu_operand) == ndims(cpu_result)
                operand_slice = selectdim(cpu_operand, basis_axis, 1)
                result_slice .= operand_slice
            elseif ndims(cpu_operand) < ndims(cpu_result)
                result_slice .= cpu_operand
            else
                result_slice .= selectdim(cpu_operand, basis_axis, 1)
            end
        end

        # Transfer to GPU
        if arch !== nothing
            copyto!(result, on_architecture(arch, cpu_result))
        else
            copyto!(result, cpu_result)
        end
    else
        # CPU path: direct operations
        fill!(result, zero(eltype(result)))

        if ndims(result) == 1
            if length(operand_data) >= 1
                result[lift_mode] = operand_data[1]
            end
        else
            result_slice = selectdim(result, basis_axis, lift_mode)
            if ndims(operand_data) == ndims(result)
                operand_slice = selectdim(operand_data, basis_axis, 1)
                result_slice .= operand_slice
            elseif ndims(operand_data) < ndims(result)
                result_slice .= operand_data
            else
                result_slice .= selectdim(operand_data, basis_axis, 1)
            end
        end
    end
end

"""
    apply_lift_nd!(result, operand, axis, lift_mode)

Apply lift operation along specified axis for multi-dimensional arrays.
(Legacy helper - kept for compatibility)
"""
function apply_lift_nd!(result::AbstractArray, operand::AbstractArray, axis::Int, lift_mode::Int)
    nd = ndims(result)

    # Use selectdim to access the lift mode slice
    selectdim(result, axis, lift_mode) .= selectdim(operand, axis, 1)
end

"""
    evaluate_convert(conv_op::Convert, layout::Symbol=:g)

Evaluate basis conversion operator.
Following operators Convert implementation.

Converts field from one basis representation to another using
spectral conversion matrices.
"""
function evaluate_convert(conv_op::Convert, layout::Symbol=:g)
    operand = conv_op.operand
    out_basis = conv_op.basis

    if !isa(operand, ScalarField)
        throw(ArgumentError("Convert currently only supports scalar fields"))
    end

    # Find the input basis to convert
    in_basis_index = nothing
    in_basis = nothing

    for (i, b) in enumerate(operand.bases)
        if b !== nothing && isa(b, JacobiBasis) && isa(out_basis, JacobiBasis)
            # Check if bases are on same coordinate
            if b.meta.element_label == out_basis.meta.element_label
                in_basis_index = i
                in_basis = b
                break
            end
        end
    end

    if in_basis === nothing
        # No conversion needed or not applicable
        return copy(operand)
    end

    # Work in coefficient space
    ensure_layout!(operand, :c)

    # Build or retrieve conversion matrix
    conv_mat = conversion_matrix(in_basis, out_basis)

    # Create result field
    new_bases = collect(operand.bases)
    new_bases[in_basis_index] = out_basis
    result = ScalarField(operand.dist, "conv_$(operand.name)", tuple(new_bases...), operand.dtype)
    ensure_layout!(result, :c)

    # Apply conversion matrix
    if ndims(get_coeff_data(operand)) == 1
        get_coeff_data(result) .= conv_mat * get_coeff_data(operand)
    else
        get_coeff_data(result) .= apply_matrix_along_axis(conv_mat, get_coeff_data(operand), in_basis_index)
    end

    if layout == :g
        backward_transform!(result)
    end

    return result
end

# ============================================================================
# General Function Operators
# ============================================================================

# Note: Power operator is defined in arithmetic.jl as Power <: Future
# Use field^p syntax for exponentiation (e.g., u^2, T^0.5)

"""
    GeneralFunction <: Operator

Apply general function to field in grid space.
"""
struct GeneralFunction <: Operator
    operand::Operand
    func::Function
    name::String
end
const _GeneralFunction_constructor = GeneralFunction

"""
    apply_function(operand::Operand, f::Function, name::String="func")

Apply arbitrary function to operand in grid space.
"""
function apply_function(operand::Operand, f::Function, name::String="func")
    return multiclass_new(GeneralFunction, operand, f, name)
end

register_operator_alias!(apply_function, "apply_function", "apply_func")
register_operator_parseable!(apply_function, "apply_function", "apply_func")

"""
    evaluate_general_function(gf_op::GeneralFunction, layout::Symbol=:g)

Evaluate general function operator in grid space.
"""
function evaluate_general_function(gf_op::GeneralFunction, layout::Symbol=:g)
    operand = gf_op.operand
    f = gf_op.func
    name = gf_op.name

    if !isa(operand, ScalarField)
        throw(ArgumentError("GeneralFunction currently only supports scalar fields"))
    end

    # Work in grid space
    ensure_layout!(operand, :g)

    # Create result field
    result = ScalarField(operand.dist, "$(name)_$(operand.name)", operand.bases, operand.dtype)
    ensure_layout!(result, :g)

    # Apply function element-wise
    get_grid_data(result) .= f.(get_grid_data(operand))

    if layout == :c
        forward_transform!(result)
    end

    return result
end

function dispatch_check(::Type{GeneralFunction}, args::Tuple, kwargs::NamedTuple)
    operand, f, name = args
    if !isa(operand, Operand)
        throw(ArgumentError("GeneralFunction requires an Operand"))
    end
    if !isa(f, Function)
        throw(ArgumentError("GeneralFunction requires a Function"))
    end
    return true
end

function invoke_constructor(::Type{GeneralFunction}, args::Tuple, kwargs::NamedTuple)
    operand, f, name = args
    return _GeneralFunction_constructor(operand, f, name)
end

"""
    UnaryGridFunction <: Operator

Apply unary numpy-style function (sin, cos, exp, etc.) to field.
"""
struct UnaryGridFunction <: Operator
    operand::Operand
    func::Function
    name::String
end
const _UnaryGridFunction_constructor = UnaryGridFunction

# Common unary functions following
function sin_field(operand::Operand)
    return UnaryGridFunction(operand, sin, "sin")
end

function cos_field(operand::Operand)
    return UnaryGridFunction(operand, cos, "cos")
end

function tan_field(operand::Operand)
    return UnaryGridFunction(operand, tan, "tan")
end

function exp_field(operand::Operand)
    return UnaryGridFunction(operand, exp, "exp")
end

function log_field(operand::Operand)
    return UnaryGridFunction(operand, log, "log")
end

function sqrt_field(operand::Operand)
    return UnaryGridFunction(operand, sqrt, "sqrt")
end

function abs_field(operand::Operand)
    return UnaryGridFunction(operand, abs, "abs")
end

function tanh_field(operand::Operand)
    return UnaryGridFunction(operand, tanh, "tanh")
end

# Register unary functions
register_operator_parseable!(sin_field, "sin")
register_operator_parseable!(cos_field, "cos")
register_operator_parseable!(tan_field, "tan")
register_operator_parseable!(exp_field, "exp")
register_operator_parseable!(log_field, "log")
register_operator_parseable!(sqrt_field, "sqrt")
register_operator_parseable!(abs_field, "abs")
register_operator_parseable!(tanh_field, "tanh")

"""
    evaluate_unary_grid_function(ugf_op::UnaryGridFunction, layout::Symbol=:g)

Evaluate unary grid function operator.
"""
function evaluate_unary_grid_function(ugf_op::UnaryGridFunction, layout::Symbol=:g)
    return evaluate_general_function(
        GeneralFunction(ugf_op.operand, ugf_op.func, ugf_op.name),
        layout
    )
end

# ============================================================================
# Grid and Coeff conversion evaluation
# ============================================================================

"""
    evaluate_grid(grid_op::Grid)

Convert operand to grid space.
"""
function evaluate_grid(grid_op::Grid)
    operand = grid_op.operand

    if isa(operand, ScalarField)
        ensure_layout!(operand, :g)
        return operand
    else
        throw(ArgumentError("Grid conversion not implemented for $(typeof(operand))"))
    end
end

"""
    evaluate_coeff(coeff_op::Coeff)

Convert operand to coefficient space.
"""
function evaluate_coeff(coeff_op::Coeff)
    operand = coeff_op.operand

    if isa(operand, ScalarField)
        ensure_layout!(operand, :c)
        return operand
    else
        throw(ArgumentError("Coeff conversion not implemented for $(typeof(operand))"))
    end
end

# ============================================================================
# Component extraction evaluation
# ============================================================================

"""
    evaluate_component(comp_op::Component)

Extract specific component from vector/tensor field.
"""
function evaluate_component(comp_op::Component)
    operand = comp_op.operand
    index = comp_op.index

    if isa(operand, VectorField)
        if index < 1 || index > length(operand.components)
            throw(BoundsError("Component index $index out of bounds"))
        end
        return operand.components[index]

    elseif isa(operand, TensorField)
        # For tensors, index could be linear or we need (i,j)
        # Use linear indexing for now
        if index < 1 || index > length(operand.components)
            throw(BoundsError("Component index $index out of bounds"))
        end
        return operand.components[index]

    else
        throw(ArgumentError("Component extraction requires VectorField or TensorField"))
    end
end

"""
    evaluate_radial_component(rc_op::RadialComponent)

Extract radial component from vector field.
For Cartesian coordinates, this is the x-component.
"""
function evaluate_radial_component(rc_op::RadialComponent)
    operand = rc_op.operand

    if !isa(operand, VectorField)
        throw(ArgumentError("RadialComponent requires a VectorField"))
    end

    # In Cartesian, "radial" is typically the first component
    # For proper polar/spherical, would need coordinate system info
    return operand.components[1]
end

"""
    evaluate_angular_component(ac_op::AngularComponent)

Extract angular component from vector field.
For Cartesian 2D, this is the y-component.
"""
function evaluate_angular_component(ac_op::AngularComponent)
    operand = ac_op.operand

    if !isa(operand, VectorField)
        throw(ArgumentError("AngularComponent requires a VectorField"))
    end

    if length(operand.components) < 2
        throw(ArgumentError("VectorField must have at least 2 components"))
    end

    return operand.components[2]
end

"""
    evaluate_azimuthal_component(az_op::AzimuthalComponent)

Extract azimuthal component from vector field.
For Cartesian 3D, this is the z-component.
"""
function evaluate_azimuthal_component(az_op::AzimuthalComponent)
    operand = az_op.operand

    if !isa(operand, VectorField)
        throw(ArgumentError("AzimuthalComponent requires a VectorField"))
    end

    if length(operand.components) < 3
        throw(ArgumentError("VectorField must have at least 3 components"))
    end

    return operand.components[3]
end

# ============================================================================
# Trace and Skew evaluation for tensors
# ============================================================================

"""
    evaluate_trace(trace_op::Trace, layout::Symbol=:g)

Evaluate trace of a tensor field.
trace(T) = Σ_i T_ii
"""
function evaluate_trace(trace_op::Trace, layout::Symbol=:g)
    operand = trace_op.operand

    if !isa(operand, TensorField)
        throw(ArgumentError("Trace requires a TensorField"))
    end

    # Create result scalar field
    result = ScalarField(operand.dist, "trace_$(operand.name)", operand.bases, operand.dtype)

    # Ensure diagonal components are in correct layout
    dim = size(operand.components, 1)

    for i in 1:dim
        ensure_layout!(operand.components[i,i], layout)
    end

    ensure_layout!(result, layout)

    # Sum diagonal components
    if layout == :g
        fill!(get_grid_data(result), 0.0)
        for i in 1:dim
            get_grid_data(result) .+= get_grid_data(operand.components[i,i])
        end
    else
        fill!(get_coeff_data(result), 0.0)
        for i in 1:dim
            get_coeff_data(result) .+= get_coeff_data(operand.components[i,i])
        end
    end

    return result
end

"""
    evaluate_skew(skew_op::Skew, layout::Symbol=:g)

Evaluate skew operator. Behavior depends on operand type:
- TensorField: Returns skew-symmetric part, skew(T) = (T - T^T) / 2
- VectorField (2D): Returns 90° rotation, skew(u_x, u_y) = (-u_y, u_x)
  This is used for 2D QG: u = skew(grad(ψ)) gives divergence-free velocity.
"""
function evaluate_skew(skew_op::Skew, layout::Symbol=:g)
    operand = skew_op.operand

    # If operand is an operator, evaluate it first
    if isa(operand, Operator)
        operand = evaluate(operand, layout)
    end

    # Dispatch based on evaluated operand type
    if isa(operand, VectorField)
        # 2D vector rotation: skew(u_x, u_y) = (-u_y, u_x)
        # Delegate to _evaluate_skew_vector which is defined in cartesian_operators.jl
        return _evaluate_skew_vector(operand, layout)
    elseif isa(operand, TensorField)
        # Tensor skew-symmetric part: skew(T) = (T - T^T) / 2
        return _evaluate_tensor_skew(operand, layout)
    else
        throw(ArgumentError("Skew requires a TensorField or VectorField, got $(typeof(operand))"))
    end
end

# Forward declaration - actual implementation in cartesian_operators.jl
function _evaluate_skew_vector end

"""
    _evaluate_tensor_skew(operand::TensorField, layout::Symbol)

Internal: Evaluate skew-symmetric part of a tensor field.
"""
function _evaluate_tensor_skew(operand::TensorField, layout::Symbol)
    coordsys = operand.coordsys
    result = TensorField(operand.dist, coordsys, "skew_$(operand.name)", operand.bases, operand.dtype)

    dim = size(operand.components, 1)

    for i in 1:dim
        for j in 1:dim
            ensure_layout!(operand.components[i,j], layout)
            ensure_layout!(operand.components[j,i], layout)
            ensure_layout!(result.components[i,j], layout)

            if layout == :g
                get_grid_data(result.components[i,j]) .= 0.5 .* (get_grid_data(operand.components[i,j]) .- get_grid_data(operand.components[j,i]))
            else
                get_coeff_data(result.components[i,j]) .= 0.5 .* (get_coeff_data(operand.components[i,j]) .- get_coeff_data(operand.components[j,i]))
            end
        end
    end

    return result
end

"""
    evaluate_transpose_components(trans_op::TransposeComponents, layout::Symbol=:g)

Evaluate transpose of tensor field components.
"""
function evaluate_transpose_components(trans_op::TransposeComponents, layout::Symbol=:g)
    operand = trans_op.operand

    if !isa(operand, TensorField)
        throw(ArgumentError("TransposeComponents requires a TensorField"))
    end

    coordsys = operand.coordsys
    result = TensorField(operand.dist, coordsys, "trans_$(operand.name)", operand.bases, operand.dtype)

    dim = size(operand.components, 1)

    for i in 1:dim
        for j in 1:dim
            ensure_layout!(operand.components[j,i], layout)
            ensure_layout!(result.components[i,j], layout)

            if layout == :g
                copyto!(get_grid_data(result.components[i,j]), get_grid_data(operand.components[j,i]))
            else
                copyto!(get_coeff_data(result.components[i,j]), get_coeff_data(operand.components[j,i]))
            end
        end
    end

    return result
end

# ============================================================================
# Curl evaluation for 2D and 3D
# ============================================================================

"""
    evaluate_curl(curl_op::Curl, layout::Symbol=:g)

Evaluate curl of a vector field.
2D: curl(v) = ∂v_y/∂x - ∂v_x/∂y (scalar)
3D: curl(v) = (∂v_z/∂y - ∂v_y/∂z, ∂v_x/∂z - ∂v_z/∂x, ∂v_y/∂x - ∂v_x/∂y)
"""
function evaluate_curl(curl_op::Curl, layout::Symbol=:g)
    operand = curl_op.operand
    coordsys = curl_op.coordsys

    if !isa(operand, VectorField)
        throw(ArgumentError("Curl requires a VectorField"))
    end

    dim = length(operand.components)

    if dim == 2
        # 2D curl: returns scalar
        return evaluate_curl_2d(operand, coordsys, layout)
    elseif dim == 3
        # 3D curl: returns vector
        return evaluate_curl_3d(operand, coordsys, layout)
    else
        throw(ArgumentError("Curl only implemented for 2D and 3D"))
    end
end

function evaluate_curl_2d(operand::VectorField, coordsys::CoordinateSystem, layout::Symbol)
    # curl(v) = ∂v_y/∂x - ∂v_x/∂y
    vx = operand.components[1]
    vy = operand.components[2]

    coord_x = coordsys.coords[1]
    coord_y = coordsys.coords[2]

    # ∂v_y/∂x
    dvy_dx = evaluate_differentiate(Differentiate(vy, coord_x, 1), layout)

    # ∂v_x/∂y
    dvx_dy = evaluate_differentiate(Differentiate(vx, coord_y, 1), layout)

    # Result = dvy_dx - dvx_dy
    result = ScalarField(operand.dist, "curl_$(operand.name)", operand.bases, operand.dtype)
    ensure_layout!(result, layout)

    if layout == :g
        get_grid_data(result) .= get_grid_data(dvy_dx) .- get_grid_data(dvx_dy)
    else
        get_coeff_data(result) .= get_coeff_data(dvy_dx) .- get_coeff_data(dvx_dy)
    end

    return result
end

function evaluate_curl_3d(operand::VectorField, coordsys::CoordinateSystem, layout::Symbol)
    vx = operand.components[1]
    vy = operand.components[2]
    vz = operand.components[3]

    coord_x = coordsys.coords[1]
    coord_y = coordsys.coords[2]
    coord_z = coordsys.coords[3]

    # Component 1: ∂v_z/∂y - ∂v_y/∂z
    dvz_dy = evaluate_differentiate(Differentiate(vz, coord_y, 1), layout)
    dvy_dz = evaluate_differentiate(Differentiate(vy, coord_z, 1), layout)

    # Component 2: ∂v_x/∂z - ∂v_z/∂x
    dvx_dz = evaluate_differentiate(Differentiate(vx, coord_z, 1), layout)
    dvz_dx = evaluate_differentiate(Differentiate(vz, coord_x, 1), layout)

    # Component 3: ∂v_y/∂x - ∂v_x/∂y
    dvy_dx = evaluate_differentiate(Differentiate(vy, coord_x, 1), layout)
    dvx_dy = evaluate_differentiate(Differentiate(vx, coord_y, 1), layout)

    result = VectorField(operand.dist, coordsys, "curl_$(operand.name)", operand.bases, operand.dtype)

    for comp in result.components
        ensure_layout!(comp, layout)
    end

    # Set component data
    if layout == :g
        get_grid_data(result.components[1]) .= get_grid_data(dvz_dy) .- get_grid_data(dvy_dz)
        get_grid_data(result.components[2]) .= get_grid_data(dvx_dz) .- get_grid_data(dvz_dx)
        get_grid_data(result.components[3]) .= get_grid_data(dvy_dx) .- get_grid_data(dvx_dy)
    else
        get_coeff_data(result.components[1]) .= get_coeff_data(dvz_dy) .- get_coeff_data(dvy_dz)
        get_coeff_data(result.components[2]) .= get_coeff_data(dvx_dz) .- get_coeff_data(dvz_dx)
        get_coeff_data(result.components[3]) .= get_coeff_data(dvy_dx) .- get_coeff_data(dvx_dy)
    end

    return result
end

# ============================================================================
# Laplacian evaluation
# ============================================================================

"""
    evaluate_laplacian(lap_op::Laplacian, layout::Symbol=:g)

Evaluate Laplacian operator.
∇²f = Σ_i ∂²f/∂x_i²
"""
function evaluate_laplacian(lap_op::Laplacian, layout::Symbol=:g)
    operand = lap_op.operand

    if isa(operand, ScalarField)
        return evaluate_scalar_laplacian(operand, layout)
    elseif isa(operand, VectorField)
        return evaluate_vector_laplacian(operand, layout)
    else
        throw(ArgumentError("Laplacian not implemented for $(typeof(operand))"))
    end
end

function evaluate_scalar_laplacian(operand::ScalarField, layout::Symbol)
    result = ScalarField(operand.dist, "lap_$(operand.name)", operand.bases, operand.dtype)
    ensure_layout!(result, layout)

    if layout == :g
        fill!(get_grid_data(result), 0.0)
    else
        fill!(get_coeff_data(result), 0.0)
    end

    for (i, basis) in enumerate(operand.bases)
        # Find coordinate for this basis via CoordinateSystem indexing
        coord = basis.meta.coordsys[basis.meta.element_label]

        # Second derivative
        d2f = evaluate_differentiate(Differentiate(operand, coord, 2), layout)

        if layout == :g
            get_grid_data(result) .+= get_grid_data(d2f)
        else
            get_coeff_data(result) .+= get_coeff_data(d2f)
        end
    end

    return result
end

function evaluate_vector_laplacian(operand::VectorField, layout::Symbol)
    result = VectorField(operand.dist, operand.coordsys, "lap_$(operand.name)",
                        operand.bases, operand.dtype)

    for (i, comp) in enumerate(operand.components)
        lap_comp = evaluate_scalar_laplacian(comp, layout)

        ensure_layout!(result.components[i], layout)
        if layout == :g
            copyto!(get_grid_data(result.components[i]), get_grid_data(lap_comp))
        else
            copyto!(get_coeff_data(result.components[i]), get_coeff_data(lap_comp))
        end
    end

    return result
end

# ============================================================================
# Fractional Laplacian evaluation
# ============================================================================

"""
    evaluate_fractional_laplacian(frac_lap::FractionalLaplacian, layout::Symbol=:g)

Evaluate fractional Laplacian operator: (-Δ)^α

In spectral space, this multiplies each Fourier coefficient by |k|^(2α),
where k = √(k₁² + k₂² + ...) is the wavenumber magnitude.

For α > 0: High-order dissipation (smoothing)
For α < 0: Inverse operation (integration/smoothing)
For α = 1: Standard Laplacian
For α = 1/2: Square root Laplacian (SQG dissipation)
For α = -1/2: Inverse square root (SQG streamfunction from buoyancy)

Note: For α < 0, the k=0 mode is handled specially to avoid division by zero.
The k=0 mode is set to zero (removes mean).
"""
function evaluate_fractional_laplacian(frac_lap::FractionalLaplacian, layout::Symbol=:g)
    operand = frac_lap.operand
    α = frac_lap.α

    if isa(operand, ScalarField)
        return evaluate_scalar_fractional_laplacian(operand, α, layout)
    elseif isa(operand, VectorField)
        return evaluate_vector_fractional_laplacian(operand, α, layout)
    else
        throw(ArgumentError("Fractional Laplacian not implemented for $(typeof(operand))"))
    end
end

function evaluate_scalar_fractional_laplacian(operand::ScalarField, α::Float64, layout::Symbol)
    # Create result field
    α_str = α >= 0 ? "$(α)" : "m$(abs(α))"  # Handle negative exponents in name
    result = ScalarField(operand.dist, "fraclap$(α_str)_$(operand.name)", operand.bases, operand.dtype)

    # Ensure operand is in coefficient space for spectral operation
    ensure_layout!(operand, :c)
    ensure_layout!(result, :c)

    # Get wavenumber grids for each Fourier basis
    # This returns an array on the same device as get_coeff_data(operand)
    k_squared_total = compute_wavenumber_squared_grid(operand)

    # Compute |k|^(2α) factor
    # For α < 0, we need to handle k=0 specially to avoid division by zero
    if α >= 0
        k_factor = k_squared_total .^ α
    else
        # For inverse operations, set k=0 mode to zero
        # Use broadcasting to handle both CPU and GPU arrays
        k_factor = similar(k_squared_total)
        # Broadcasting approach: where k² > threshold, compute k^α, else 0
        threshold = 1e-14
        k_factor .= ifelse.(k_squared_total .> threshold, k_squared_total .^ α, zero(eltype(k_squared_total)))
    end

    # Apply the fractional Laplacian in spectral space
    # Note: We use |k|^(2α) not (-k²)^α to ensure real output for real input
    get_coeff_data(result) .= get_coeff_data(operand) .* k_factor

    # Transform to requested layout if needed
    if layout == :g
        ensure_layout!(result, :g)
    end

    return result
end

function evaluate_vector_fractional_laplacian(operand::VectorField, α::Float64, layout::Symbol)
    α_str = α >= 0 ? "$(α)" : "m$(abs(α))"
    result = VectorField(operand.dist, operand.coordsys, "fraclap$(α_str)_$(operand.name)",
                        operand.bases, operand.dtype)

    for (i, comp) in enumerate(operand.components)
        frac_lap_comp = evaluate_scalar_fractional_laplacian(comp, α, layout)

        ensure_layout!(result.components[i], layout)
        if layout == :g
            copyto!(get_grid_data(result.components[i]), get_grid_data(frac_lap_comp))
        else
            copyto!(get_coeff_data(result.components[i]), get_coeff_data(frac_lap_comp))
        end
    end

    return result
end

"""
    compute_wavenumber_squared_grid(field::ScalarField)

Compute |k|² = k₁² + k₂² + ... for each point in spectral space.
Supports both CPU and GPU arrays.

Returns an array with the same shape as get_coeff_data(field) containing
the squared wavenumber magnitude at each spectral coefficient location.
The returned array is on the same device (CPU/GPU) as get_coeff_data(field).
"""
function compute_wavenumber_squared_grid(field::ScalarField)
    bases = field.bases
    data_shape = size(get_coeff_data(field))

    # Initialize with zeros on the same device as get_coeff_data(field)
    # Use similar_zeros to preserve array type (CPU Array or GPU CuArray)
    k_squared = similar_zeros(get_coeff_data(field), Float64, data_shape...)

    # Add contribution from each basis
    for (axis, basis) in enumerate(bases)
        if isa(basis, RealFourier) || isa(basis, ComplexFourier)
            # Get wavenumbers for this Fourier basis (always returns CPU array)
            k_axis_cpu = wavenumbers(basis)

            # Add k² contribution to each point
            # Need to broadcast along the correct axis
            add_wavenumber_squared_contribution!(k_squared, k_axis_cpu, axis, length(bases))
        end
        # For Chebyshev/Legendre bases, no wavenumber contribution (they're not Fourier)
        # This means fractional Laplacian only applies to Fourier directions
    end

    return k_squared
end

"""
Add k² contribution from one axis to the total wavenumber grid.
Works with both CPU and GPU arrays.
"""
function add_wavenumber_squared_contribution!(k_squared::AbstractArray, k_axis_cpu::Vector{Float64}, axis::Int, ndims::Int)
    # Create shape for broadcasting: all 1s except for the current axis
    shape = ones(Int, ndims)
    shape[axis] = length(k_axis_cpu)

    # Move k_axis to the same device as k_squared if needed
    if is_gpu_array(k_squared)
        k_axis = copy_to_device(k_axis_cpu, k_squared)
    else
        k_axis = k_axis_cpu
    end

    # Reshape k_axis for broadcasting
    k_reshaped = reshape(k_axis, Tuple(shape))

    # Add k² contribution
    k_squared .+= k_reshaped.^2
end

# ============================================================================
# Fractional Laplacian - Matrix methods for implicit/LHS treatment
# ============================================================================

"""
    matrix_dependence(op::FractionalLaplacian, vars...)

Determine which variables this fractional Laplacian operator depends on.
Returns a boolean vector indicating dependence on each variable.
"""
function matrix_dependence(op::FractionalLaplacian, vars...)
    result = falses(length(vars))
    for (i, var) in enumerate(vars)
        if op.operand === var || (hasfield(typeof(op.operand), :name) &&
                                   hasfield(typeof(var), :name) &&
                                   op.operand.name == var.name)
            result[i] = true
        end
    end
    return result
end

"""
    matrix_coupling(op::FractionalLaplacian, vars...)

Determine variable coupling for fractional Laplacian.
The fractional Laplacian is diagonal in spectral space - no coupling between variables.
"""
function matrix_coupling(op::FractionalLaplacian, vars...)
    # Fractional Laplacian only couples a variable to itself
    return matrix_dependence(op, vars...)
end

"""
    subproblem_matrix(op::FractionalLaplacian, subproblem)

Build the sparse matrix representation of fractional Laplacian for implicit solvers.

In spectral space, (-Δ)^α is diagonal with entries |k|^(2α) on the diagonal.
This makes it very efficient for implicit treatment.
"""
function subproblem_matrix(op::FractionalLaplacian, subproblem)
    operand = op.operand
    α = op.α

    # Get the spectral size from the subproblem or operand
    if isa(operand, ScalarField)
        n = prod(size(get_coeff_data(operand)))
    else
        throw(ArgumentError("subproblem_matrix for FractionalLaplacian only implemented for ScalarField"))
    end

    # Compute wavenumber squared grid
    # Note: For implicit solvers, we need CPU arrays for sparse matrices
    k_squared = compute_wavenumber_squared_grid(operand)

    # Convert to CPU if on GPU (sparse matrices are CPU-only)
    if is_gpu_array(k_squared)
        k_squared = Array(k_squared)
    end

    k_squared_flat = vec(k_squared)

    # Compute |k|^(2α) diagonal entries
    if α >= 0
        diag_entries = k_squared_flat .^ α
    else
        # For negative α, handle k=0 specially using broadcasting
        threshold = 1e-14
        diag_entries = ifelse.(k_squared_flat .> threshold, k_squared_flat .^ α, 0.0)
    end

    # Return sparse diagonal matrix
    return spdiagm(0 => diag_entries)
end

"""
    check_conditions(op::FractionalLaplacian)

Check that operand is in proper layout for fractional Laplacian.
"""
function check_conditions(op::FractionalLaplacian)
    operand = op.operand

    if isa(operand, ScalarField)
        if hasfield(typeof(operand), :current_layout)
            layout = operand.current_layout
            if layout == :c
                return get_coeff_data(operand) !== nothing
            elseif layout == :g
                return get_grid_data(operand) !== nothing
            end
        end
    elseif isa(operand, VectorField)
        for comp in operand.components
            if !check_conditions(FractionalLaplacian(comp, op.α))
                return false
            end
        end
    end

    return true
end

"""
    enforce_conditions(op::FractionalLaplacian)

Ensure operand is in coefficient layout for spectral fractional Laplacian.
"""
function enforce_conditions(op::FractionalLaplacian)
    operand = op.operand

    if isa(operand, ScalarField)
        ensure_layout!(operand, :c)
    elseif isa(operand, VectorField)
        for comp in operand.components
            ensure_layout!(comp, :c)
        end
    end
end

"""
    is_linear(op::FractionalLaplacian)

Fractional Laplacian is a linear operator.
"""
is_linear(op::FractionalLaplacian) = true

"""
    operator_order(op::FractionalLaplacian)

Return the effective derivative order of the fractional Laplacian.
For (-Δ)^α, the effective order is 2α.
"""
operator_order(op::FractionalLaplacian) = 2 * op.α

# ============================================================================
# Outer Product (Tensor Product) evaluation
# ============================================================================

"""
    evaluate_outer(outer_op::Outer, layout::Symbol=:g)

Evaluate outer product (tensor product) of two vector fields.

For vectors u and v, returns tensor T where T_ij = u_i * v_j.

Following pattern for tensor construction from vector products.

Arguments:
- outer_op: Outer operator containing left and right operands
- layout: :g for grid space, :c for coefficient space

Returns:
- TensorField with components T_ij = u_i * v_j
"""
function evaluate_outer(outer_op::Outer, layout::Symbol=:g)
    left = outer_op.left
    right = outer_op.right

    # Validate operands are vector fields
    if !isa(left, VectorField) || !isa(right, VectorField)
        throw(ArgumentError("Outer product requires two VectorFields"))
    end

    # Get dimensions
    dim_left = length(left.components)
    dim_right = length(right.components)

    # Create result tensor field
    dist = left.dist
    coordsys = left.coordsys
    bases = left.bases
    dtype = promote_type(left.dtype, right.dtype)

    result = TensorField(dist, coordsys, "outer_$(left.name)_$(right.name)", bases, dtype)

    # Ensure proper dimensions
    if size(result.components) != (dim_left, dim_right)
        # Reinitialize with correct dimensions
        result.components = Matrix{ScalarField}(undef, dim_left, dim_right)
        for i in 1:dim_left
            for j in 1:dim_right
                result.components[i,j] = ScalarField(dist, "T_$(i)$(j)", bases, dtype)
            end
        end
    end

    # Compute T_ij = u_i * v_j for each component
    for i in 1:dim_left
        for j in 1:dim_right
            # Ensure operands are in correct layout
            ensure_layout!(left.components[i], layout)
            ensure_layout!(right.components[j], layout)
            ensure_layout!(result.components[i,j], layout)

            # Compute outer product component: T_ij = u_i * v_j
            if layout == :g
                get_grid_data(result.components[i,j]) .= get_grid_data(left.components[i]) .* get_grid_data(right.components[j])
            else
                get_coeff_data(result.components[i,j]) .= get_coeff_data(left.components[i]) .* get_coeff_data(right.components[j])
            end
        end
    end

    return result
end

# ============================================================================
# AdvectiveCFL evaluation
# ============================================================================

"""
    evaluate_advective_cfl(cfl_op::AdvectiveCFL, layout::Symbol=:g)

Evaluate advective CFL grid-crossing frequency field.

Computes the local grid-crossing frequency:
    f = |u|/Δx + |v|/Δy + |w|/Δz

where u, v, w are velocity components and Δx, Δy, Δz are local grid spacings.

This field can be used for adaptive timestepping: Δt < 1/max(f).

Following operators:4342-4411 AdvectiveCFL pattern.

Arguments:
- cfl_op: AdvectiveCFL operator containing velocity field and coordinate system
- layout: Must be :g (grid space) for CFL computation

Returns:
- ScalarField with CFL frequency at each grid point
"""
function evaluate_advective_cfl(cfl_op::AdvectiveCFL, layout::Symbol=:g)
    velocity = cfl_op.operand
    coords = cfl_op.coords

    # CFL must be computed in grid space
    if layout != :g
        @warn "AdvectiveCFL requires grid space; converting to :g"
        layout = :g
    end

    # Validate operand is a vector field
    if !isa(velocity, VectorField)
        throw(ArgumentError("AdvectiveCFL requires a VectorField (velocity)"))
    end

    dim = length(velocity.components)

    # Create result scalar field
    dist = velocity.dist
    bases = velocity.bases
    dtype = velocity.dtype

    result = ScalarField(dist, "cfl_freq", bases, dtype)
    ensure_layout!(result, :g)

    # Initialize result to zero
    get_grid_data(result) .= 0

    # Compute CFL frequency for each coordinate direction
    for i in 1:dim
        # Ensure velocity component is in grid space
        ensure_layout!(velocity.components[i], :g)
        vel_data = get_grid_data(velocity.components[i])

        # Get grid spacing for this coordinate
        # Grid spacing depends on the basis type
        basis = bases[i]
        grid_spacing = compute_grid_spacing(basis, dist, i)

        # Add contribution: |u_i| / Δx_i
        if isa(grid_spacing, AbstractArray)
            # Variable grid spacing (e.g., Chebyshev)
            get_grid_data(result) .+= abs.(vel_data) ./ grid_spacing
        else
            # Uniform grid spacing (e.g., Fourier)
            get_grid_data(result) .+= abs.(vel_data) ./ grid_spacing
        end
    end

    return result
end

"""
    compute_grid_spacing(basis::Basis, dist, axis::Int)

Compute local grid spacing for a basis.

For Fourier bases: uniform spacing Δx = L/N
For Chebyshev bases: variable spacing based on Chebyshev nodes

Following basis CartesianAdvectiveCFL.grid_spacing pattern.
"""
function compute_grid_spacing(basis::Basis, dist, axis::Int)
    if basis === nothing
        return 1.0  # Fallback for undefined basis
    end

    N = basis.meta.size

    if isa(basis, FourierBasis)
        # Uniform spacing for Fourier
        L = basis.meta.bounds[2] - basis.meta.bounds[1]
        return L / N
    elseif isa(basis, ChebyshevT) || isa(basis, ChebyshevU) || isa(basis, ChebyshevV)
        # Chebyshev grid spacing (clustered at boundaries)
        # For Chebyshev-Gauss-Lobatto points: x_j = cos(π*j/N)
        # Local spacing: Δx_j ≈ π * sqrt(1 - x_j²) / N
        L = basis.meta.bounds[2] - basis.meta.bounds[1]

        # Compute grid points
        j = 0:(N-1)
        x_j = cos.(π .* j ./ (N - 1))

        # Local spacing (avoiding singularity at boundaries)
        spacing = (π / N) .* sqrt.(max.(1 .- x_j.^2, 1e-10))

        # Scale to physical domain
        return spacing .* (L / 2)
    elseif isa(basis, Legendre)
        # Legendre-Gauss-Lobatto spacing (similar to Chebyshev)
        L = basis.meta.bounds[2] - basis.meta.bounds[1]
        return L / N  # Approximate with uniform for simplicity
    elseif isa(basis, Jacobi)
        # General Jacobi basis
        L = basis.meta.bounds[2] - basis.meta.bounds[1]
        return L / N  # Approximate with uniform
    else
        # Default uniform spacing
        return 1.0
    end
end

# ============================================================================
# Unified operator evaluation dispatcher
# ============================================================================

"""
    evaluate(op::Operator, layout::Symbol=:g)

Unified evaluation function that dispatches to specific operator evaluators.
"""
function evaluate(op::Operator, layout::Symbol=:g)
    if isa(op, Gradient)
        return evaluate_gradient(op, layout)
    elseif isa(op, Divergence)
        return evaluate_divergence(op, layout)
    elseif isa(op, Curl)
        return evaluate_curl(op, layout)
    elseif isa(op, Laplacian)
        return evaluate_laplacian(op, layout)
    elseif isa(op, FractionalLaplacian)
        return evaluate_fractional_laplacian(op, layout)
    elseif isa(op, Differentiate)
        return evaluate_differentiate(op, layout)
    elseif isa(op, Interpolate)
        return evaluate_interpolate(op, layout)
    elseif isa(op, Integrate)
        return evaluate_integrate(op, layout)
    elseif isa(op, Average)
        return evaluate_average(op, layout)
    elseif isa(op, Lift)
        return evaluate_lift(op, layout)
    elseif isa(op, Convert)
        return evaluate_convert(op, layout)
    elseif isa(op, GeneralFunction)
        return evaluate_general_function(op, layout)
    elseif isa(op, UnaryGridFunction)
        return evaluate_unary_grid_function(op, layout)
    elseif isa(op, Grid)
        return evaluate_grid(op)
    elseif isa(op, Coeff)
        return evaluate_coeff(op)
    elseif isa(op, Component)
        return evaluate_component(op)
    elseif isa(op, RadialComponent)
        return evaluate_radial_component(op)
    elseif isa(op, AngularComponent)
        return evaluate_angular_component(op)
    elseif isa(op, AzimuthalComponent)
        return evaluate_azimuthal_component(op)
    elseif isa(op, Trace)
        return evaluate_trace(op, layout)
    elseif isa(op, Skew)
        return evaluate_skew(op, layout)
    elseif isa(op, TransposeComponents)
        return evaluate_transpose_components(op, layout)
    elseif isa(op, Outer)
        return evaluate_outer(op, layout)
    elseif isa(op, AdvectiveCFL)
        return evaluate_advective_cfl(op, layout)
    elseif isa(op, TimeDerivative)
        # TimeDerivative is handled by solvers, not direct evaluation
        throw(ArgumentError("TimeDerivative cannot be directly evaluated; use solver"))
    else
        throw(ArgumentError("Evaluation not implemented for operator type $(typeof(op))"))
    end
end

# ============================================================================
# Exports
# ============================================================================

# Export abstract type
export Operator

# Export differential operator types
export Gradient, Divergence, Curl, Laplacian, FractionalLaplacian
export Trace, Skew, TransposeComponents, TimeDerivative

# Export interpolation/integration operator types
export Interpolate, Integrate, Average

# Export conversion operator types
export Convert, Grid, Coeff, Lift

# Export component extraction operator types
export Component, RadialComponent, AngularComponent, AzimuthalComponent

# Export differentiation and product operator types
export Differentiate, Outer, AdvectiveCFL

# Export arithmetic operator types
export AddOperator, SubtractOperator, MultiplyOperator

# Export function operator types
export GeneralFunction, UnaryGridFunction

# Export constructor functions
export grad, divergence, div_op, curl, lap, trace, skew, transpose_components
export interpolate, integrate, average, convert_basis, lift, d, dt

# Export fractional/hyperviscosity constructors
export fraclap, sqrtlap, invsqrtlap, hyperlap

# Export other operator constructors
export outer, advective_cfl, cfl

# Export Unicode aliases
export ∇, Δ, ∇², ∂t, Δᵅ, Δ², Δ⁴, Δ⁶, Δ⁸

# Export evaluation functions
export evaluate
export evaluate_gradient, evaluate_divergence, evaluate_curl, evaluate_laplacian
export evaluate_fractional_laplacian, evaluate_differentiate
export evaluate_interpolate, evaluate_integrate, evaluate_average
export evaluate_lift, evaluate_convert
export evaluate_grid, evaluate_coeff, evaluate_component
export evaluate_radial_component, evaluate_angular_component, evaluate_azimuthal_component
export evaluate_trace, evaluate_skew, evaluate_transpose_components
export evaluate_outer, evaluate_advective_cfl
export evaluate_general_function, evaluate_unary_grid_function

# Export matrix interface functions
export matrix_dependence, matrix_coupling, subproblem_matrix
export check_conditions, enforce_conditions, is_linear, operator_order

# Export operator registration functions
export register_operator_alias!, register_operator_parseable!, register_operator_prefix!
export OPERATOR_ALIASES, OPERATOR_PARSEABLES, OPERATOR_PREFIXES
