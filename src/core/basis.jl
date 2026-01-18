"""
Spectral basis classes

This module implements the spectral basis hierarchy:
- Basis: Abstract base type
- IntervalBasis: 1D bases on intervals
- Jacobi: General Jacobi polynomial basis (base for Chebyshev, Legendre, etc.)
- Ultraspherical/ChebyshevT/ChebyshevU: Jacobi with specific parameters
- RealFourier/ComplexFourier: Periodic Fourier bases

Key features implemented:
- Proper Jacobi parameter inheritance (a, b, a0, b0)
- product_matrix for NCC (Non-Constant Coefficient) support
- valid_elements for mode filtering
- derivative_basis returning correct output basis type
- Conversion matrices between bases

## GPU Compatibility

This module is designed for CPU-only setup operations. The basis module computes:
- Grid coordinates (returned as CPU arrays)
- Spectral matrices (returned as sparse CPU matrices)
- Wavenumbers (returned as CPU vectors)

For GPU execution, the domain.jl and field.jl modules handle data movement:
- Grid coordinates are computed on CPU, then moved to GPU via `on_architecture()`
- Field data is allocated on the appropriate architecture
- Transforms use GPU-accelerated FFT (CUFFT) when available

The `evaluate_basis` functions accept GPU coordinates and will automatically
convert them to CPU for evaluation, with a warning message. For GPU-native
operations, use transform-based methods instead of direct basis evaluation.
"""

using LinearAlgebra
using SparseArrays
using FFTW
using SpecialFunctions: gamma, lgamma

export Basis, IntervalBasis, JacobiBasis, FourierBasis,
       Jacobi, Ultraspherical, ChebyshevT, ChebyshevU, ChebyshevV, Chebyshev, Legendre,
       RealFourier, ComplexFourier, Fourier,
       AffineCOV, BasisMeta,
       wavenumbers, derivative_basis, conversion_matrix, differentiation_matrix,
       product_matrix, ncc_matrix, valid_elements,
       grid_shape, coeff_shape, element_label, coordsys, pencil_compatible_size,
       local_grid, local_grids, evaluate_basis

# ============================================================================
# Abstract types for basis hierarchy
# ============================================================================

abstract type Basis end
abstract type IntervalBasis <: Basis end
abstract type JacobiBasis <: IntervalBasis end  # Base for all Jacobi-type bases
abstract type FourierBasis <: IntervalBasis end  # Base for Fourier bases

# ============================================================================
# Affine Change of Variables
# ============================================================================

"""
    AffineCOV

Class for affine change-of-variables for remapping space bounds.
"""
struct AffineCOV
    native_bounds::Tuple{Float64, Float64}
    problem_bounds::Tuple{Float64, Float64}
    native_left::Float64
    native_right::Float64
    native_length::Float64
    native_center::Float64
    problem_left::Float64
    problem_right::Float64
    problem_length::Float64
    problem_center::Float64
    stretch::Float64

    function AffineCOV(native_bounds::Tuple{Float64, Float64}, problem_bounds::Tuple{Float64, Float64})
        native_left, native_right = native_bounds
        native_length = native_right - native_left
        native_center = (native_left + native_right) / 2
        problem_left, problem_right = problem_bounds
        problem_length = problem_right - problem_left
        problem_center = (problem_left + problem_right) / 2

        # Guard against zero native_length (degenerate interval)
        if abs(native_length) < 1e-14
            throw(ArgumentError("AffineCOV: native interval has zero length"))
        end
        stretch = problem_length / native_length

        new(native_bounds, problem_bounds, native_left, native_right, native_length,
            native_center, problem_left, problem_right, problem_length, problem_center, stretch)
    end
end

"""Convert native coordinates to problem coordinates."""
function problem_coord(cov::AffineCOV, native_coord)
    return @. cov.problem_center + cov.stretch * (native_coord - cov.native_center)
end

"""Convert problem coordinates to native coordinates."""
function native_coord(cov::AffineCOV, problem_coord)
    return @. cov.native_center + (problem_coord - cov.problem_center) / cov.stretch
end

# ============================================================================
# BasisMeta: Metadata for all basis types
# ============================================================================

mutable struct BasisMeta
    coordsys::CoordinateSystem
    element_label::String
    dim::Int
    size::Int
    bounds::Tuple{Float64, Float64}
    dealias::Union{Float64, Vector{Float64}, Tuple{Vararg{Float64}}}
    dtype::Type
    constant::Vector{Bool}
    subaxis_dependence::Vector{Bool}
    # Additional fields
    COV::Union{Nothing, AffineCOV}
    constant_mode_value::Float64

    function BasisMeta(coordsys, element_label, dim, size, bounds, dealias, dtype,
                       constant::Vector{Bool}, subaxis_dependence::Vector{Bool};
                       native_bounds::Union{Nothing, Tuple{Float64,Float64}}=nothing,
                       constant_mode_value::Float64=1.0)
        cov = if native_bounds !== nothing
            AffineCOV(native_bounds, bounds)
        else
            nothing
        end
        new(coordsys, element_label, dim, size, bounds, dealias, dtype,
            constant, subaxis_dependence, cov, constant_mode_value)
    end
end

function BasisMeta(coordsys, element_label, dim, size, bounds, dealias, dtype;
                   constant=nothing, subaxis_dependence=nothing, kwargs...)
    const_vec = if constant === nothing
        fill(false, dim)
    elseif constant isa Bool
        fill(constant, dim)
    else
        collect(Bool.(constant))
    end
    dep_vec = if subaxis_dependence === nothing
        fill(true, dim)
    elseif subaxis_dependence isa Bool
        fill(subaxis_dependence, dim)
    else
        collect(Bool.(subaxis_dependence))
    end
    if length(const_vec) != dim
        throw(ArgumentError("constant metadata length $(length(const_vec)) does not match basis dimension $dim"))
    end
    if length(dep_vec) != dim
        throw(ArgumentError("subaxis_dependence length $(length(dep_vec)) does not match basis dimension $dim"))
    end
    return BasisMeta(coordsys, element_label, dim, size, bounds, dealias, dtype,
                     const_vec, dep_vec; kwargs...)
end

# ============================================================================
# Jacobi Basis
# ============================================================================

"""
    Jacobi <: JacobiBasis

General Jacobi polynomial basis P_n^{(a,b)}(x).

Parameters:
- a, b: Jacobi parameters for the basis polynomials
- a0, b0: Jacobi parameters for the output basis (used in derivatives)

Key relationships:
- ChebyshevT: a = b = -1/2, alpha = 0 (Ultraspherical)
- ChebyshevU: a = b = 1/2, alpha = 1 (Ultraspherical)
- Legendre: a = b = 0
"""
struct Jacobi <: JacobiBasis
    meta::BasisMeta
    transforms::Dict{String, Any}
    a::Float64      # Jacobi parameter a
    b::Float64      # Jacobi parameter b
    a0::Float64     # Output basis parameter a (for derivatives)
    b0::Float64     # Output basis parameter b (for derivatives)
    # Cached matrices for NCC operations
    _product_matrix_cache::Dict{Tuple, AbstractMatrix}
    _conversion_matrix_cache::Dict{Tuple, AbstractMatrix}
    _differentiation_matrix_cache::Dict{Int, AbstractMatrix}
end

const JACOBI_NATIVE_BOUNDS = (-1.0, 1.0)

function _build_jacobi(coord::Coordinate;
                       a::Float64=0.0, b::Float64=0.0,
                       a0::Union{Nothing, Float64}=nothing,
                       b0::Union{Nothing, Float64}=nothing,
                       size::Int=32,
                       bounds::Tuple{Float64,Float64}=(-1.0,1.0),
                       dealias::Float64=1.0,
                       dtype=Float64)
    # Default a0, b0 to a, b if not specified
    a0 = a0 === nothing ? a : a0
    b0 = b0 === nothing ? b : b0

    meta = BasisMeta(coord.coordsys, coord.name, 1, size, bounds, dealias, dtype;
                     native_bounds=JACOBI_NATIVE_BOUNDS, constant_mode_value=1.0)
    transforms = Dict{String, Any}()
    product_cache = Dict{Tuple, AbstractMatrix}()
    conversion_cache = Dict{Tuple, AbstractMatrix}()
    diff_cache = Dict{Int, AbstractMatrix}()

    return Jacobi(meta, transforms, a, b, a0, b0, product_cache, conversion_cache, diff_cache)
end

const _Jacobi_constructor = _build_jacobi

function Jacobi(coord::Coordinate; kwargs...)
    return multiclass_new(Jacobi, coord; kwargs...)
end

# ============================================================================
# Ultraspherical Basis (Jacobi with a=b)
# ============================================================================

"""
    Ultraspherical <: JacobiBasis

Ultraspherical (Gegenbauer) polynomial basis C_n^{alpha}(x).
Special case of Jacobi with a = b = alpha - 1/2.

ChebyshevT = Ultraspherical(alpha=0).
"""
struct Ultraspherical <: JacobiBasis
    meta::BasisMeta
    transforms::Dict{String, Any}
    alpha::Float64  # Gegenbauer parameter
    a::Float64      # Jacobi parameter (= alpha - 1/2)
    b::Float64      # Jacobi parameter (= alpha - 1/2)
    a0::Float64
    b0::Float64
    _product_matrix_cache::Dict{Tuple, AbstractMatrix}
    _conversion_matrix_cache::Dict{Tuple, AbstractMatrix}
    _differentiation_matrix_cache::Dict{Int, AbstractMatrix}
end

function _build_ultraspherical(coord::Coordinate;
                               alpha::Float64=0.0,
                               size::Int=32,
                               bounds::Tuple{Float64,Float64}=(-1.0,1.0),
                               dealias::Float64=1.0,
                               dtype=Float64)
    # Ultraspherical C_n^alpha corresponds to Jacobi P_n^{(a,b)} with a = b = alpha - 1/2
    a = alpha - 0.5
    b = alpha - 0.5
    a0 = a
    b0 = b

    meta = BasisMeta(coord.coordsys, coord.name, 1, size, bounds, dealias, dtype;
                     native_bounds=JACOBI_NATIVE_BOUNDS, constant_mode_value=1.0)
    transforms = Dict{String, Any}()
    product_cache = Dict{Tuple, AbstractMatrix}()
    conversion_cache = Dict{Tuple, AbstractMatrix}()
    diff_cache = Dict{Int, AbstractMatrix}()

    return Ultraspherical(meta, transforms, alpha, a, b, a0, b0,
                          product_cache, conversion_cache, diff_cache)
end

const _Ultraspherical_constructor = _build_ultraspherical

function Ultraspherical(coord::Coordinate; kwargs...)
    return multiclass_new(Ultraspherical, coord; kwargs...)
end

# ============================================================================
# ChebyshevT = Ultraspherical(alpha=0) = Jacobi(a=-1/2, b=-1/2)
# ============================================================================

"""
    ChebyshevT <: JacobiBasis

Chebyshev polynomials of the first kind T_n(x).
Equivalent to Ultraspherical(alpha=0) or Jacobi(a=-1/2, b=-1/2).
"""
struct ChebyshevT <: JacobiBasis
    meta::BasisMeta
    transforms::Dict{String, Any}
    a::Float64
    b::Float64
    a0::Float64
    b0::Float64
    _product_matrix_cache::Dict{Tuple, AbstractMatrix}
    _conversion_matrix_cache::Dict{Tuple, AbstractMatrix}
    _differentiation_matrix_cache::Dict{Int, AbstractMatrix}
end

function _build_chebyshev_t(coord::Coordinate;
                            size::Int=32,
                            bounds::Tuple{Float64,Float64}=(-1.0,1.0),
                            dealias::Float64=1.0,
                            dtype=Float64)
    # ChebyshevT = Jacobi(a=-1/2, b=-1/2) = Ultraspherical(alpha=0)
    a = -0.5
    b = -0.5
    a0 = a
    b0 = b

    meta = BasisMeta(coord.coordsys, coord.name, 1, size, bounds, dealias, dtype;
                     native_bounds=JACOBI_NATIVE_BOUNDS, constant_mode_value=1.0)
    transforms = Dict{String, Any}()
    product_cache = Dict{Tuple, AbstractMatrix}()
    conversion_cache = Dict{Tuple, AbstractMatrix}()
    diff_cache = Dict{Int, AbstractMatrix}()

    return ChebyshevT(meta, transforms, a, b, a0, b0,
                      product_cache, conversion_cache, diff_cache)
end

const _ChebyshevT_constructor = _build_chebyshev_t

function ChebyshevT(coord::Coordinate; kwargs...)
    return multiclass_new(ChebyshevT, coord; kwargs...)
end

# ============================================================================
# ChebyshevU = Ultraspherical(alpha=1) = Jacobi(a=1/2, b=1/2)
# ============================================================================

"""
    ChebyshevU <: JacobiBasis

Chebyshev polynomials of the second kind U_n(x).
Equivalent to Ultraspherical(alpha=1) or Jacobi(a=1/2, b=1/2).
"""
struct ChebyshevU <: JacobiBasis
    meta::BasisMeta
    transforms::Dict{String, Any}
    a::Float64
    b::Float64
    a0::Float64
    b0::Float64
    _product_matrix_cache::Dict{Tuple, AbstractMatrix}
    _conversion_matrix_cache::Dict{Tuple, AbstractMatrix}
    _differentiation_matrix_cache::Dict{Int, AbstractMatrix}
end

function _build_chebyshev_u(coord::Coordinate;
                            size::Int=32,
                            bounds::Tuple{Float64,Float64}=(-1.0,1.0),
                            dealias::Float64=1.0,
                            dtype=Float64)
    # ChebyshevU = Jacobi(a=1/2, b=1/2) = Ultraspherical(alpha=1)
    a = 0.5
    b = 0.5
    a0 = a
    b0 = b

    meta = BasisMeta(coord.coordsys, coord.name, 1, size, bounds, dealias, dtype;
                     native_bounds=JACOBI_NATIVE_BOUNDS, constant_mode_value=1.0)
    transforms = Dict{String, Any}()
    product_cache = Dict{Tuple, AbstractMatrix}()
    conversion_cache = Dict{Tuple, AbstractMatrix}()
    diff_cache = Dict{Int, AbstractMatrix}()

    return ChebyshevU(meta, transforms, a, b, a0, b0,
                      product_cache, conversion_cache, diff_cache)
end

const _ChebyshevU_constructor = _build_chebyshev_u

function ChebyshevU(coord::Coordinate; kwargs...)
    return multiclass_new(ChebyshevU, coord; kwargs...)
end

# Alias for backwards compatibility
const Chebyshev = ChebyshevT

# ============================================================================
# ChebyshevV = Ultraspherical(alpha=2) = Jacobi(a=3/2, b=3/2)
# ============================================================================

"""
    ChebyshevV <: JacobiBasis

Chebyshev-like polynomials with alpha=2 (third kind variant).
Equivalent to Ultraspherical(alpha=2) or Jacobi(a=3/2, b=3/2).

This basis is useful for second-order derivatives and appears in
spectral methods for fourth-order PDEs.
"""
struct ChebyshevV <: JacobiBasis
    meta::BasisMeta
    transforms::Dict{String, Any}
    a::Float64
    b::Float64
    a0::Float64
    b0::Float64
    _product_matrix_cache::Dict{Tuple, AbstractMatrix}
    _conversion_matrix_cache::Dict{Tuple, AbstractMatrix}
    _differentiation_matrix_cache::Dict{Int, AbstractMatrix}
end

function _build_chebyshev_v(coord::Coordinate;
                            size::Int=32,
                            bounds::Tuple{Float64,Float64}=(-1.0,1.0),
                            dealias::Float64=1.0,
                            dtype=Float64)
    # ChebyshevV = Jacobi(a=3/2, b=3/2) = Ultraspherical(alpha=2)
    a = 1.5
    b = 1.5
    a0 = a
    b0 = b

    meta = BasisMeta(coord.coordsys, coord.name, 1, size, bounds, dealias, dtype;
                     native_bounds=JACOBI_NATIVE_BOUNDS, constant_mode_value=1.0)
    transforms = Dict{String, Any}()
    product_cache = Dict{Tuple, AbstractMatrix}()
    conversion_cache = Dict{Tuple, AbstractMatrix}()
    diff_cache = Dict{Int, AbstractMatrix}()

    return ChebyshevV(meta, transforms, a, b, a0, b0,
                      product_cache, conversion_cache, diff_cache)
end

const _ChebyshevV_constructor = _build_chebyshev_v

function ChebyshevV(coord::Coordinate; kwargs...)
    return multiclass_new(ChebyshevV, coord; kwargs...)
end

# ============================================================================
# Legendre = Jacobi(a=0, b=0)
# ============================================================================

"""
    Legendre <: JacobiBasis

Legendre polynomial basis P_n(x).
Equivalent to Jacobi(a=0, b=0).
"""
struct Legendre <: JacobiBasis
    meta::BasisMeta
    transforms::Dict{String, Any}
    a::Float64
    b::Float64
    a0::Float64
    b0::Float64
    _product_matrix_cache::Dict{Tuple, AbstractMatrix}
    _conversion_matrix_cache::Dict{Tuple, AbstractMatrix}
    _differentiation_matrix_cache::Dict{Int, AbstractMatrix}
end

function _build_legendre(coord::Coordinate;
                         size::Int=32,
                         bounds::Tuple{Float64,Float64}=(-1.0,1.0),
                         dealias::Float64=1.0,
                         dtype=Float64)
    # Legendre = Jacobi(a=0, b=0)
    a = 0.0
    b = 0.0
    a0 = a
    b0 = b

    meta = BasisMeta(coord.coordsys, coord.name, 1, size, bounds, dealias, dtype;
                     native_bounds=JACOBI_NATIVE_BOUNDS, constant_mode_value=1.0)
    transforms = Dict{String, Any}()
    product_cache = Dict{Tuple, AbstractMatrix}()
    conversion_cache = Dict{Tuple, AbstractMatrix}()
    diff_cache = Dict{Int, AbstractMatrix}()

    return Legendre(meta, transforms, a, b, a0, b0,
                    product_cache, conversion_cache, diff_cache)
end

const _Legendre_constructor = _build_legendre

function Legendre(coord::Coordinate; kwargs...)
    return multiclass_new(Legendre, coord; kwargs...)
end

# ============================================================================
# Fourier Bases
# ============================================================================

const FOURIER_NATIVE_BOUNDS = (0.0, 2π)

"""
    RealFourier <: FourierBasis

Real Fourier sine/cosine basis (msin = -sin convention).
Modes: [cos(0*x), cos(1*x), -sin(1*x), cos(2*x), -sin(2*x), ...]
"""
struct RealFourier <: FourierBasis
    meta::BasisMeta
    transforms::Dict{String, Any}
    # Cached wavenumbers
    _wavenumbers::Union{Nothing, Vector{Float64}}
    _product_matrix_cache::Dict{Tuple, AbstractMatrix}
end

function _build_real_fourier(coord::Coordinate;
                             size::Int=32,
                             bounds::Tuple{Float64,Float64}=(0.0,2π),
                             dealias::Float64=1.0,
                             dtype=Float64)
    meta = BasisMeta(coord.coordsys, coord.name, 1, size, bounds, dealias, dtype;
                     native_bounds=FOURIER_NATIVE_BOUNDS, constant_mode_value=1.0)
    transforms = Dict{String, Any}()
    product_cache = Dict{Tuple, AbstractMatrix}()
    return RealFourier(meta, transforms, nothing, product_cache)
end

const _RealFourier_constructor = _build_real_fourier

function RealFourier(coord::Coordinate; kwargs...)
    return multiclass_new(RealFourier, coord; kwargs...)
end

"""
    ComplexFourier <: FourierBasis

Complex Fourier exponential basis.
Modes: [exp(i*0*x), exp(i*1*x), exp(-i*1*x), exp(i*2*x), exp(-i*2*x), ...]
"""
struct ComplexFourier <: FourierBasis
    meta::BasisMeta
    transforms::Dict{String, Any}
    _wavenumbers::Union{Nothing, Vector{Float64}}
    _product_matrix_cache::Dict{Tuple, AbstractMatrix}
end

function _build_complex_fourier(coord::Coordinate;
                                size::Int=32,
                                bounds::Tuple{Float64,Float64}=(0.0,2π),
                                dealias::Float64=1.0,
                                dtype=ComplexF64)
    meta = BasisMeta(coord.coordsys, coord.name, 1, size, bounds, dealias, dtype;
                     native_bounds=FOURIER_NATIVE_BOUNDS, constant_mode_value=1.0)
    transforms = Dict{String, Any}()
    product_cache = Dict{Tuple, AbstractMatrix}()
    return ComplexFourier(meta, transforms, nothing, product_cache)
end

const _ComplexFourier_constructor = _build_complex_fourier

function ComplexFourier(coord::Coordinate; kwargs...)
    return multiclass_new(ComplexFourier, coord; kwargs...)
end

# Alias
const Fourier = RealFourier

# ============================================================================
# Wavenumber computation
# ============================================================================

"""
    wavenumbers(basis::RealFourier)

Get wavenumbers for RealFourier basis.
"""
function wavenumbers(basis::RealFourier)
    N = basis.meta.size
    L = basis.meta.bounds[2] - basis.meta.bounds[1]
    if abs(L) < 1e-14
        throw(ArgumentError("wavenumbers: domain length is zero"))
    end
    k0 = 2π / L
    # Build wavenumber sequence matching RealFourier storage:
    # [cos_0, cos_1, sin_1, cos_2, sin_2, ..., (optional) cos_nyquist]
    if iseven(N)
        kmax = N ÷ 2
        # cos_0, (cos_k, sin_k) for k=1..kmax-1, then cos_kmax (Nyquist)
        k_native = vcat([0], vec(repeat(1:(kmax - 1), inner=2)), [kmax])
    else
        kmax = (N - 1) ÷ 2
        # cos_0, (cos_k, sin_k) for k=1..kmax
        k_native = vcat([0], vec(repeat(1:kmax, inner=2)))
    end
    return k0 .* k_native
end

"""
    wavenumbers(basis::ComplexFourier)

Get wavenumbers for ComplexFourier basis.
"""
function wavenumbers(basis::ComplexFourier)
    N = basis.meta.size
    L = basis.meta.bounds[2] - basis.meta.bounds[1]
    if abs(L) < 1e-14
        throw(ArgumentError("wavenumbers: domain length is zero"))
    end
    k0 = 2π / L
    # FFT ordering:
    # Even N: [0, 1, 2, ..., N/2-1, -N/2, ..., -2, -1]
    # Odd N:  [0, 1, 2, ..., (N-1)/2, -(N-1)/2, ..., -2, -1]
    if iseven(N)
        k_native = [0:(N ÷ 2 - 1); -(N ÷ 2):-1]
    else
        kmax = (N - 1) ÷ 2
        k_native = [0:kmax; -kmax:-1]
    end
    return k0 .* k_native
end

# ============================================================================
# Product matrices for NCC support
# ============================================================================

"""
    ncc_matrix(ncc_basis, arg_basis, out_basis, coeffs; cutoff=1e-6)

Build full NCC matrix via direct summation of product matrices.

The NCC matrix represents multiplication by a spatially-varying coefficient field:
    (ncc * operand)_coeffs = NCC_matrix @ operand_coeffs

where ncc is expanded in spectral coefficients and each mode contributes
via its product_matrix.

# Arguments
- `ncc_basis`: Basis for the NCC field
- `arg_basis`: Basis for the argument/operand field
- `out_basis`: Basis for the output field
- `coeffs`: Spectral coefficients of the NCC field
- `cutoff`: Coefficient cutoff for sparsity (default 1e-6)

# Returns
Sparse matrix representing the NCC multiplication operation.
"""
function ncc_matrix(ncc_basis::Basis, arg_basis, out_basis, coeffs::AbstractVector; cutoff::Float64=1e-6)
    N = length(coeffs)
    total = nothing

    for i in 1:N
        coeff = coeffs[i]

        # Skip small coefficients
        if abs(coeff) <= cutoff
            continue
        end

        # Get product matrix for mode i-1 (0-based mode index)
        matrix = product_matrix(ncc_basis, arg_basis, out_basis, i - 1)

        # Scale by coefficient and accumulate
        if total === nothing
            total = coeff * matrix
        else
            total = total + coeff * matrix
        end
    end

    if total === nothing
        N_out = out_basis === nothing ? ncc_basis.meta.size : out_basis.meta.size
        N_arg = arg_basis === nothing ? 1 : arg_basis.meta.size
        return spzeros(Float64, N_out, N_arg)
    end

    # Eliminate small entries
    droptol!(total, cutoff)
    return total
end

"""
    product_matrix(basis::JacobiBasis, arg_basis, out_basis, ncc_mode::Int)

Build multiplication matrix for Non-Constant Coefficient (NCC) terms.

This computes the matrix M such that:
    (f * g)_coeffs = M @ g_coeffs
where f is the NCC field (expanded to ncc_mode) and g is the argument field.
"""
function product_matrix(basis::JacobiBasis, arg_basis, out_basis, ncc_mode::Int)
    cache_key = (arg_basis, out_basis, ncc_mode)

    # Check cache
    if haskey(basis._product_matrix_cache, cache_key)
        return basis._product_matrix_cache[cache_key]
    end

    N = basis.meta.size
    a, b = basis.a, basis.b

    # Build Jacobi product matrix using linearization coefficients
    # P_m * P_n = sum_k c_{m,n,k} P_k
    matrix = _jacobi_product_matrix(N, a, b, ncc_mode, arg_basis, out_basis)

    basis._product_matrix_cache[cache_key] = matrix
    return matrix
end

"""
    product_matrix(basis::RealFourier, arg_basis, out_basis, ncc_mode::Int)

Build multiplication matrix for RealFourier NCC.

For Fourier: cos(m*x) * cos(n*x) = 0.5*(cos((m-n)*x) + cos((m+n)*x))
"""
function product_matrix(basis::RealFourier, arg_basis, out_basis, ncc_mode::Int)
    cache_key = (arg_basis, out_basis, ncc_mode)

    if haskey(basis._product_matrix_cache, cache_key)
        return basis._product_matrix_cache[cache_key]
    end

    N_out = out_basis === nothing ? basis.meta.size : out_basis.meta.size
    N_arg = arg_basis === nothing ? 1 : arg_basis.meta.size

    L = basis.meta.bounds[2] - basis.meta.bounds[1]
    k0 = 2π / L

    # Get NCC wavenumber
    k_ncc = wavenumbers(basis)
    m = ncc_mode < length(k_ncc) ? Int(round(k_ncc[ncc_mode+1] / k0)) : 0
    # RealFourier storage: index 1 = cos0, index 2 = cos1, index 3 = sin1, etc.
    is_sin_mode = ncc_mode != 0 && isodd(ncc_mode + 1)

    # Build sparse product matrix
    if m == 0
        # Constant NCC: identity or truncation
        if !is_sin_mode  # cos mode
            matrix = sparse(I, N_out, N_arg)
        else  # sin mode (which is zero for k=0)
            matrix = spzeros(Float64, N_out, N_arg)
        end
    else
        matrix = _build_real_fourier_product_matrix(N_arg, N_out, m, is_sin_mode)
    end

    basis._product_matrix_cache[cache_key] = matrix
    return matrix
end

"""
Build RealFourier product matrix for specific wavenumber.

The RealFourier product matrix handles multiplication of trig functions:
- 2 cos(mx) cos(nx) = cos((m+n)x) + cos((m-n)x)
- 2 cos(mx) sin(nx) = sin((m+n)x) - sin((m-n)x)  (msin = -sin notation)
- 2 sin(mx) cos(nx) = sin((m+n)x) + sin((m-n)x)
- 2 sin(mx) sin(nx) = -cos((m+n)x) + cos((m-n)x)
"""
function _build_real_fourier_product_matrix(N_arg::Int, N_out::Int, m::Int, is_sin::Bool)
    # Use wavenumber intersection approach
    # Indexing: 1 -> cos0, even indices -> cos(k>=1), odd indices >1 -> -sin(k) (msin)

    I_list = Int[]
    J_list = Int[]
    V_list = Float64[]

    # Generate wavenumber arrays
    # RealFourier stores: [cos_0, cos_1, msin_1, cos_2, msin_2, ...]
    # where msin = -sin

    # Process the three coupling cases:
    # 1. k_out = k_ncc + k_arg (rows_p, cols_p)
    # 2. k_out = k_ncc - k_arg (rows_m, cols_m)
    # 3. k_out = k_arg - k_ncc for negative result (rows_mn, cols_mn)

    # In Julia 1-based:
    # cos_0 at 1, cos_1 at 2, msin_1 at 3, cos_2 at 4, msin_2 at 5...
    for j_arg in 1:N_arg
        if j_arg == 1
            n = 0
            is_sin_arg = false
        elseif iseven(j_arg)
            n = j_arg ÷ 2
            is_sin_arg = false
        else
            n = (j_arg - 1) ÷ 2
            is_sin_arg = true
        end

        if !is_sin_arg
            # cos input modes
            k_plus = m + n
            if k_plus >= 0
                out_cos_idx = k_plus == 0 ? 1 : 2 * k_plus
                out_sin_idx = 2 * k_plus + 1

                if is_sin
                    # sin(mx) * cos(nx) -> sin((m+n)x)
                    if out_sin_idx <= N_out
                        push!(I_list, out_sin_idx)
                        push!(J_list, j_arg)
                        push!(V_list, 0.5)
                    end
                else
                    # cos(mx) * cos(nx) -> cos((m+n)x)
                    if out_cos_idx <= N_out
                        push!(I_list, out_cos_idx)
                        push!(J_list, j_arg)
                        push!(V_list, 0.5)
                    end
                end
            end

            k_minus = m - n
            if k_minus >= 0
                out_cos_idx = k_minus == 0 ? 1 : 2 * k_minus
                out_sin_idx = 2 * k_minus + 1

                if is_sin
                    # sin(mx) * cos(nx) -> sin((m-n)x)
                    if out_sin_idx <= N_out && k_minus > 0
                        push!(I_list, out_sin_idx)
                        push!(J_list, j_arg)
                        push!(V_list, 0.5)
                    end
                else
                    # cos(mx) * cos(nx) -> cos((m-n)x)
                    if out_cos_idx <= N_out
                        push!(I_list, out_cos_idx)
                        push!(J_list, j_arg)
                        push!(V_list, 0.5)
                    end
                end
            end

            k_nm = n - m
            if k_nm > 0
                out_cos_idx = 2 * k_nm
                out_sin_idx = 2 * k_nm + 1

                if is_sin
                    # sin(mx) * cos(nx) -> -sin((n-m)x) = msin((n-m)x)
                    if out_sin_idx <= N_out
                        push!(I_list, out_sin_idx)
                        push!(J_list, j_arg)
                        push!(V_list, -0.5)
                    end
                else
                    # cos(mx) * cos(nx) -> cos((n-m)x)
                    if out_cos_idx <= N_out
                        push!(I_list, out_cos_idx)
                        push!(J_list, j_arg)
                        push!(V_list, 0.5)
                    end
                end
            end
        else
            if n == 0
                continue
            end

            # msin input modes
            k_plus = m + n
            out_cos_idx = k_plus == 0 ? 1 : 2 * k_plus
            out_sin_idx = 2 * k_plus + 1

            if is_sin
                # sin(mx) * sin(nx) = -0.5*cos((m+n)x) + 0.5*cos((m-n)x)
                # -> -cos((m+n)x) contribution
                if out_cos_idx <= N_out
                    push!(I_list, out_cos_idx)
                    push!(J_list, j_arg)
                    push!(V_list, -0.5)
                end
            else
                # cos(mx) * sin(nx) -> sin((m+n)x)
                if out_sin_idx <= N_out
                    push!(I_list, out_sin_idx)
                    push!(J_list, j_arg)
                    push!(V_list, 0.5)
                end
            end

            k_minus = m - n
            if k_minus >= 0
                out_cos_idx = k_minus == 0 ? 1 : 2 * k_minus
                out_sin_idx = 2 * k_minus + 1

                if is_sin
                    # sin(mx) * sin(nx) -> cos((m-n)x)
                    if out_cos_idx <= N_out
                        push!(I_list, out_cos_idx)
                        push!(J_list, j_arg)
                        push!(V_list, 0.5)
                    end
                else
                    # cos(mx) * sin(nx) -> -sin((m-n)x)
                    if out_sin_idx <= N_out && k_minus > 0
                        push!(I_list, out_sin_idx)
                        push!(J_list, j_arg)
                        push!(V_list, -0.5)
                    end
                end
            end

            k_nm = n - m
            if k_nm > 0
                out_cos_idx = 2 * k_nm
                out_sin_idx = 2 * k_nm + 1

                if is_sin
                    # sin(mx) * sin(nx) -> cos((n-m)x)
                    if out_cos_idx <= N_out
                        push!(I_list, out_cos_idx)
                        push!(J_list, j_arg)
                        push!(V_list, 0.5)
                    end
                else
                    # cos(mx) * sin(nx) -> sin((n-m)x)
                    if out_sin_idx <= N_out
                        push!(I_list, out_sin_idx)
                        push!(J_list, j_arg)
                        push!(V_list, 0.5)
                    end
                end
            end
        end
    end

    if isempty(I_list)
        return spzeros(Float64, N_out, N_arg)
    end

    return sparse(I_list, J_list, V_list, N_out, N_arg)
end

"""Build Jacobi product matrix using linearization coefficients."""
function _jacobi_product_matrix(N::Int, a::Float64, b::Float64,
                                 ncc_mode::Int, arg_basis, out_basis)
    N_out = out_basis === nothing ? N : out_basis.meta.size
    N_arg = arg_basis === nothing ? 1 : arg_basis.meta.size

    # For Jacobi polynomials: P_m^{(a,b)} * P_n^{(a,b)} = sum_k c_{m,n,k} P_k^{(a,b)}
    # The linearization coefficients c_{m,n,k} are computed using Clebsch-Gordan-like formulas

    m = ncc_mode

    I_list = Int[]
    J_list = Int[]
    V_list = Float64[]

    for n in 0:(N_arg-1)
        # Compute linearization coefficients for P_m * P_n
        coeffs = _jacobi_linearization_coefficients(m, n, a, b, N_out)

        for (k, c) in enumerate(coeffs)
            if abs(c) > 1e-14 && k <= N_out
                push!(I_list, k)
                push!(J_list, n + 1)
                push!(V_list, c)
            end
        end
    end

    if isempty(I_list)
        return spzeros(Float64, N_out, N_arg)
    end

    return sparse(I_list, J_list, V_list, N_out, N_arg)
end

"""
Compute Jacobi polynomial linearization coefficients.
P_m^{(a,b)}(x) * P_n^{(a,b)}(x) = sum_{k=|m-n|}^{m+n} c_k P_k^{(a,b)}(x)

Uses Clenshaw algorithm with Jacobi matrices.
"""
function _jacobi_linearization_coefficients(m::Int, n::Int, a::Float64, b::Float64, N_max::Int)
    coeffs = zeros(Float64, N_max)

    # Note: Linearization coefficients are only non-zero for |m-n| <= k <= m+n
    # This constraint is implicitly handled by the quadrature-based computation

    # Special case: m=0 or n=0 (multiplication by P_0 = 1)
    if m == 0
        if n < N_max
            coeffs[n + 1] = 1.0
        end
        return coeffs
    end
    if n == 0
        if m < N_max
            coeffs[m + 1] = 1.0
        end
        return coeffs
    end

    # General Jacobi case (includes all special cases: Chebyshev, Legendre, etc.):
    # Note: The Chebyshev T (a=b=-1/2) and U (a=b=1/2) formulas for product
    # linearization use the normalized Chebyshev polynomials T_n and U_n,
    # not the standard Jacobi polynomials P_n^{(a,b)}. Since this function
    # computes coefficients for standard Jacobi polynomials, we use numerical
    # quadrature for all cases to ensure correctness.
    # P_m^{(a,b)} * P_n^{(a,b)} = sum_k A_{m,n,k}^{(a,b)} P_k^{(a,b)}

    # Build Jacobi matrix J (tridiagonal) for the recurrence relation
    # x * P_n^{(a,b)} = A_n * P_{n-1} + B_n * P_n + C_n * P_{n+1}

    N_work = max(m + n + 2, N_max + 1)

    # Compute using matrix Clenshaw algorithm
    # This evaluates P_m(J) where J is the Jacobi matrix
    coeffs = _jacobi_linearization_clenshaw(m, n, a, b, N_max, N_work)

    return coeffs
end

"""
Compute Jacobi linearization using Gauss-Jacobi quadrature.
This provides accurate linearization coefficients for general Jacobi polynomials.

The linearization coefficients are computed via projection:
c_k = ∫_{-1}^{1} P_m^{(a,b)}(x) P_n^{(a,b)}(x) P_k^{(a,b)}(x) w(x) dx / h_k

where w(x) = (1-x)^a (1+x)^b is the weight function and h_k is the normalization.
"""
function _jacobi_linearization_clenshaw(m::Int, n::Int, a::Float64, b::Float64, N_max::Int, N_work::Int)
    coeffs = zeros(Float64, N_max)

    # Use Gauss-Jacobi quadrature with enough points for exactness
    # For product of P_m * P_n * P_k (degrees m, n, k), we need N_quad >= (m+n+k+1)/2
    N_quad = max(m + n + N_max + 2, 2 * N_work)

    # Get Gauss-Jacobi quadrature nodes and weights
    nodes, weights = _gauss_jacobi_quadrature(N_quad, a, b)

    # Evaluate P_m and P_n at quadrature nodes
    P_m = _jacobi_polynomial_values(m, a, b, nodes)
    P_n = _jacobi_polynomial_values(n, a, b, nodes)

    # Product values
    product = P_m .* P_n

    # Compute coefficients by projection
    for k in 0:(N_max-1)
        P_k = _jacobi_polynomial_values(k, a, b, nodes)

        # Normalization factor h_k for Jacobi polynomials
        h_k = _jacobi_norm_squared(k, a, b)

        # Inner product: ∫ P_m * P_n * P_k * w dx ≈ sum of weights * values
        inner_prod = sum(weights .* product .* P_k)

        coeffs[k + 1] = inner_prod / h_k
    end

    # Clean up small values
    cutoff = 1e-12
    for k in 1:N_max
        if abs(coeffs[k]) < cutoff
            coeffs[k] = 0.0
        end
    end

    return coeffs
end

"""
Compute Gauss-Jacobi quadrature nodes and weights.
Uses the Golub-Welsch algorithm via eigenvalue decomposition of the Jacobi matrix.
Special handling for Chebyshev T (a=b=-0.5) and Chebyshev U (a=b=0.5) cases.
"""
function _gauss_jacobi_quadrature(N::Int, a::Float64, b::Float64)
    if N < 1
        return Float64[], Float64[]
    end

    # Special case: Chebyshev T (a=b=-0.5)
    # The standard Golub-Welsch algorithm fails for a+b=-1 because beta[1]=0.
    # Use the known Chebyshev-Gauss quadrature formula instead.
    if abs(a + 0.5) < 1e-10 && abs(b + 0.5) < 1e-10
        nodes = zeros(Float64, N)
        weights = zeros(Float64, N)
        for k in 1:N
            nodes[k] = cos((2*k - 1) * π / (2*N))
            weights[k] = π / N
        end
        perm = sortperm(nodes)
        return nodes[perm], weights[perm]
    end

    # Build the symmetric tridiagonal Jacobi matrix
    alpha = zeros(Float64, N)   # Main diagonal
    beta = zeros(Float64, N-1)  # Sub/super diagonal

    for k in 0:(N-1)
        # Diagonal element
        denom = (2*k + a + b) * (2*k + a + b + 2)
        if abs(denom) > 1e-14
            alpha[k+1] = (b^2 - a^2) / denom
        else
            alpha[k+1] = 0.0
        end

        # Off-diagonal element
        if k < N - 1
            num = 4 * (k + 1) * (k + 1 + a) * (k + 1 + b) * (k + 1 + a + b)
            denom = (2*k + a + b + 1) * (2*k + a + b + 2)^2 * (2*k + a + b + 3)
            if abs(denom) > 1e-14 && num >= 0
                beta[k+1] = sqrt(num / denom)
            else
                beta[k+1] = 0.0
            end
        end
    end

    # Build symmetric tridiagonal matrix and compute eigendecomposition
    J = SymTridiagonal(alpha, beta)
    eigen_decomp = eigen(J)
    nodes = eigen_decomp.values
    V = eigen_decomp.vectors

    # Weights: w_i = μ_0 * v_{1,i}^2
    mu_0 = 2^(a + b + 1) * exp(lgamma(a + 1) + lgamma(b + 1) - lgamma(a + b + 2))
    weights = mu_0 * (V[1, :].^2)

    return nodes, weights
end

"""
Evaluate Jacobi polynomial P_n^{(a,b)}(x) at given points using stable recurrence.
"""
function _jacobi_polynomial_values(n::Int, a::Float64, b::Float64, x::AbstractVector{Float64})
    N_pts = length(x)

    if n == 0
        return ones(Float64, N_pts)
    end

    P_prev = ones(Float64, N_pts)
    P_curr = @. (a - b) / 2 + (a + b + 2) / 2 * x

    if n == 1
        return P_curr
    end

    for k in 1:(n-1)
        k_ab = 2*k + a + b
        A_k = (k_ab + 1) * (k_ab + 2) / (2 * (k + 1) * (k + a + b + 1))
        B_k = (a^2 - b^2) * (k_ab + 1) / (2 * (k + 1) * (k + a + b + 1) * k_ab)
        C_k = (k + a) * (k + b) * (k_ab + 2) / ((k + 1) * (k + a + b + 1) * k_ab)

        P_next = @. (A_k * x + B_k) * P_curr - C_k * P_prev
        P_prev = P_curr
        P_curr = P_next
    end

    return P_curr
end

"""
Compute the squared norm (h_n) of Jacobi polynomial P_n^{(a,b)}.
"""
function _jacobi_norm_squared(n::Int, a::Float64, b::Float64)
    if n == 0
        return 2^(a + b + 1) * exp(lgamma(a + 1) + lgamma(b + 1) - lgamma(a + b + 2))
    end

    log_h = (a + b + 1) * log(2) - log(2*n + a + b + 1)
    log_h += lgamma(n + a + 1) + lgamma(n + b + 1)
    log_h -= lgamma(n + 1) + lgamma(n + a + b + 1)

    return exp(log_h)
end

"""Compute Legendre linearization coefficient using Clebsch-Gordan formula."""
function _legendre_linearization_coeff(m::Int, n::Int, k::Int)
    # P_m * P_n = sum_k c_{m,n,k} P_k
    # c_{m,n,k} = (2k+1) * C(m,n,k)^2 where C is Clebsch-Gordan

    # Selection rules: |m-n| <= k <= m+n, and m+n+k is even
    if k < abs(m - n) || k > m + n || (m + n + k) % 2 != 0
        return 0.0
    end

    # Use 3j symbol formula (Wigner 3j symbols via Clebsch-Gordan coefficients)
    # Reference: Varshalovich et al., "Quantum Theory of Angular Momentum"
    s = (m + n + k) ÷ 2

    # Compute using factorials (for small m, n, k)
    if max(m, n, k) < 20
        num = factorial(big(s - m)) * factorial(big(s - n)) * factorial(big(s - k))
        den = factorial(big(s + 1))

        f1 = factorial(big(2*s - 2*m)) * factorial(big(2*s - 2*n)) * factorial(big(2*s - 2*k))
        f2 = factorial(big(2*s + 1))

        coeff = Float64((2*k + 1) * (num^2 / den^2) * (f2 / f1))
        return coeff
    else
        # Approximate for large indices
        return 1.0 / (m + n - abs(m - n) + 1)
    end
end

# ============================================================================
# Valid elements / mode filtering
# ============================================================================

"""
    valid_elements(basis::Basis, tensorsig, grid_space, elements)

Determine which elements are valid for the given tensor signature.
"""
function valid_elements(basis::RealFourier, tensorsig, grid_space, elements)
    vshape = (length(tensorsig) > 0 ? prod(cs.dim for cs in tensorsig) : 1,) .* size(elements[1])
    valid = trues(vshape)

    if !grid_space[1]
        # Drop msin part of k=0 for all Cartesian components
        groups = elements_to_groups(basis, grid_space, elements)
        for i in eachindex(valid)
            if groups[1][i] == 0 && elements[1][i] % 2 == 1
                valid[i] = false
            end
        end
    end

    return valid
end

function valid_elements(basis::JacobiBasis, tensorsig, grid_space, elements)
    # Jacobi bases have all elements valid
    vshape = (length(tensorsig) > 0 ? prod(cs.dim for cs in tensorsig) : 1,) .* size(elements[1])
    return trues(vshape)
end

"""Convert elements to groups."""
function elements_to_groups(basis::RealFourier, grid_space, elements)
    # RealFourier has group_shape = (2,), groups are element ÷ 2
    return (elements[1] .÷ 2,)
end

function elements_to_groups(basis::JacobiBasis, grid_space, elements)
    # Jacobi has group_shape = (1,), groups equal elements
    return elements
end

# ============================================================================
# Derivative basis (D maps T to U)
# ============================================================================

"""
    derivative_basis(basis::JacobiBasis, order::Int=1)

Return the basis for the derivative of fields in this basis.

Derivative basis chain:
- ∂/∂x(ChebyshevT) → ChebyshevU (Jacobi a,b: -1/2,-1/2 → 1/2,1/2)
- ∂/∂x(ChebyshevU) → ChebyshevV (Jacobi a,b: 1/2,1/2 → 3/2,3/2)
- ∂/∂x(ChebyshevV) → Jacobi(5/2, 5/2)
- General: ∂/∂x P_n^{(a,b)} is proportional to P_{n-1}^{(a+1,b+1)}
"""
function derivative_basis(basis::ChebyshevT, order::Int=1)
    # ∂/∂x(T_n) is proportional to U_{n-1}
    # After differentiation, output is in ChebyshevU
    # Create new ChebyshevU basis with same domain parameters
    if order < 0
        throw(ArgumentError("derivative_basis: order must be non-negative, got $order"))
    end
    if order == 0
        return basis
    end

    coord = Coordinate(basis.meta.element_label; cs=basis.meta.coordsys)
    output = ChebyshevU(coord;
                        size=basis.meta.size,
                        bounds=basis.meta.bounds,
                        dealias=basis.meta.dealias,
                        dtype=basis.meta.dtype)

    # Recursively apply for higher orders
    if order > 1
        return derivative_basis(output, order - 1)
    end
    return output
end

function derivative_basis(basis::ChebyshevU, order::Int=1)
    # ∂/∂x(U_n) is proportional to polynomials in ChebyshevV family
    # ChebyshevU = Jacobi(1/2, 1/2), derivative -> Jacobi(3/2, 3/2) = ChebyshevV
    if order < 0
        throw(ArgumentError("derivative_basis: order must be non-negative, got $order"))
    end
    if order == 0
        return basis
    end

    coord = Coordinate(basis.meta.element_label; cs=basis.meta.coordsys)
    output = ChebyshevV(coord;
                        size=basis.meta.size,
                        bounds=basis.meta.bounds,
                        dealias=basis.meta.dealias,
                        dtype=basis.meta.dtype)

    if order > 1
        return derivative_basis(output, order - 1)
    end
    return output
end

function derivative_basis(basis::ChebyshevV, order::Int=1)
    # ∂/∂x(V_n) → Jacobi(5/2, 5/2)
    # Continue the Jacobi parameter increment chain
    if order < 0
        throw(ArgumentError("derivative_basis: order must be non-negative, got $order"))
    end
    if order == 0
        return basis
    end

    coord = Coordinate(basis.meta.element_label; cs=basis.meta.coordsys)
    # Create Jacobi basis with incremented parameters
    output = Jacobi(coord;
                    a=basis.a + 1.0,
                    b=basis.b + 1.0,
                    size=basis.meta.size,
                    bounds=basis.meta.bounds,
                    dealias=basis.meta.dealias,
                    dtype=basis.meta.dtype)

    if order > 1
        return derivative_basis(output, order - 1)
    end
    return output
end

function derivative_basis(basis::Legendre, order::Int=1)
    # ∂/∂x(P_n) is proportional to Jacobi with a=1, b=1
    # Legendre = Jacobi(0,0), derivative -> Jacobi(1,1)
    if order < 0
        throw(ArgumentError("derivative_basis: order must be non-negative, got $order"))
    end
    if order == 0
        return basis
    end

    coord = Coordinate(basis.meta.element_label; cs=basis.meta.coordsys)
    output = Jacobi(coord;
                    a=1.0,
                    b=1.0,
                    size=basis.meta.size,
                    bounds=basis.meta.bounds,
                    dealias=basis.meta.dealias,
                    dtype=basis.meta.dtype)

    if order > 1
        return derivative_basis(output, order - 1)
    end
    return output
end

function derivative_basis(basis::Jacobi, order::Int=1)
    # ∂/∂x P_n^{(a,b)} is proportional to P_{n-1}^{(a+1,b+1)}
    if order < 0
        throw(ArgumentError("derivative_basis: order must be non-negative, got $order"))
    end
    if order == 0
        return basis
    end

    coord = Coordinate(basis.meta.element_label; cs=basis.meta.coordsys)
    output = Jacobi(coord;
                    a=basis.a + 1.0,
                    b=basis.b + 1.0,
                    size=basis.meta.size,
                    bounds=basis.meta.bounds,
                    dealias=basis.meta.dealias,
                    dtype=basis.meta.dtype)

    if order > 1
        return derivative_basis(output, order - 1)
    end
    return output
end

function derivative_basis(basis::FourierBasis, order::Int=1)
    # Fourier derivative stays in same basis
    # ∂/∂x(exp(ikx)) = ik·exp(ikx), still in Fourier space
    if order < 0
        throw(ArgumentError("derivative_basis: order must be non-negative, got $order"))
    end
    return basis
end

# ============================================================================
# Conversion matrices between bases
# ============================================================================

"""
    conversion_matrix(input_basis::JacobiBasis, output_basis::JacobiBasis)

Build conversion matrix from input basis to output basis.
"""
function conversion_matrix(input_basis::JacobiBasis, output_basis::JacobiBasis)
    cache_key = (input_basis.a, input_basis.b, output_basis.a, output_basis.b)

    if haskey(input_basis._conversion_matrix_cache, cache_key)
        return input_basis._conversion_matrix_cache[cache_key]
    end

    N = input_basis.meta.size
    a0, b0 = input_basis.a, input_basis.b
    a1, b1 = output_basis.a, output_basis.b

    matrix = _jacobi_conversion_matrix(N, a0, b0, a1, b1)

    input_basis._conversion_matrix_cache[cache_key] = matrix
    return matrix
end

"""Build Jacobi conversion matrix."""
function _jacobi_conversion_matrix(N::Int, a0::Float64, b0::Float64, a1::Float64, b1::Float64)
    # Convert from P_n^{(a0,b0)} to P_n^{(a1,b1)}
    # This uses recurrence relations from Jacobi theory

    if abs(a0 - a1) < 1e-10 && abs(b0 - b1) < 1e-10
        return sparse(I, N, N)
    end

    # For ChebyshevT -> ChebyshevU conversion (a=-1/2 -> a=1/2)
    if abs(a0 + 0.5) < 1e-10 && abs(b0 + 0.5) < 1e-10 &&
       abs(a1 - 0.5) < 1e-10 && abs(b1 - 0.5) < 1e-10
        return _chebyshev_t_to_u_matrix(N)
    end

    # General case: use recursion
    return _general_jacobi_conversion(N, a0, b0, a1, b1)
end

"""Build ChebyshevT to ChebyshevU conversion matrix."""
function _chebyshev_t_to_u_matrix(N::Int)
    # T_n = (U_n - U_{n-2}) / 2 for n >= 2
    # T_0 = U_0, T_1 = U_1 / 2

    I_list = Int[]
    J_list = Int[]
    V_list = Float64[]

    # T_0 -> U_0
    push!(I_list, 1); push!(J_list, 1); push!(V_list, 1.0)

    # T_1 -> U_1 / 2
    if N > 1
        push!(I_list, 2); push!(J_list, 2); push!(V_list, 0.5)
    end

    # T_n -> (U_n - U_{n-2}) / 2 for n >= 2
    for n in 2:(N-1)
        push!(I_list, n + 1); push!(J_list, n + 1); push!(V_list, 0.5)
        push!(I_list, n - 1); push!(J_list, n + 1); push!(V_list, -0.5)
    end

    return sparse(I_list, J_list, V_list, N, N)
end

"""
    _general_jacobi_conversion(N, a0, b0, a1, b1)

General Jacobi polynomial conversion matrix from P_n^{(a0,b0)} to P_n^{(a1,b1)}.

Uses the connection coefficient formula. For integer parameter shifts, the conversion
matrix is sparse (banded). For general shifts, we use quadrature-based projection.

The conversion is computed via:
    P_n^{(a0,b0)}(x) = Σ_m C_{nm} P_m^{(a1,b1)}(x)

where C_{nm} are the connection coefficients.

Reference:
- NIST DLMF 18.9 (Connection formulas)
- Dedalus Project jacobi.py
"""
function _general_jacobi_conversion(N::Int, a0::Float64, b0::Float64, a1::Float64, b1::Float64)
    # Check if parameters are the same (identity conversion)
    if abs(a0 - a1) < 1e-12 && abs(b0 - b1) < 1e-12
        return sparse(I, N, N)
    end

    # Check for integer shifts - these have sparse representations
    da = a1 - a0
    db = b1 - b0

    # Special case: shift by integer in both parameters
    if abs(da - round(da)) < 1e-12 && abs(db - round(db)) < 1e-12
        da_int = Int(round(da))
        db_int = Int(round(db))

        # Build conversion through successive single-step shifts
        if da_int >= 0 && db_int >= 0
            return _jacobi_conversion_positive_shift(N, a0, b0, da_int, db_int)
        elseif da_int <= 0 && db_int <= 0
            return _jacobi_conversion_negative_shift(N, a0, b0, -da_int, -db_int)
        end
    end

    # General case: use quadrature-based projection
    # Evaluate input basis at output basis quadrature points, then project
    return _jacobi_conversion_quadrature(N, a0, b0, a1, b1)
end

"""
Build conversion for positive integer shifts in (a,b) parameters.
Uses the recurrence: P_n^{(a,b)} can be written in terms of P_m^{(a+1,b)} or P_m^{(a,b+1)}.
"""
function _jacobi_conversion_positive_shift(N::Int, a0::Float64, b0::Float64, da::Int, db::Int)
    result = sparse(I, N, N)

    # Shift a first, then b
    a_curr, b_curr = a0, b0

    for _ in 1:da
        step = _jacobi_a_shift_up_matrix(N, a_curr, b_curr)
        result = step * result
        a_curr += 1.0
    end

    for _ in 1:db
        step = _jacobi_b_shift_up_matrix(N, a_curr, b_curr)
        result = step * result
        b_curr += 1.0
    end

    return result
end

"""
Build conversion for negative integer shifts in (a,b) parameters.
"""
function _jacobi_conversion_negative_shift(N::Int, a0::Float64, b0::Float64, da::Int, db::Int)
    result = sparse(I, N, N)

    a_curr, b_curr = a0, b0

    for _ in 1:da
        step = _jacobi_a_shift_down_matrix(N, a_curr, b_curr)
        result = step * result
        a_curr -= 1.0
    end

    for _ in 1:db
        step = _jacobi_b_shift_down_matrix(N, a_curr, b_curr)
        result = step * result
        b_curr -= 1.0
    end

    return result
end

"""
Single step conversion: P_n^{(a,b)} -> P_m^{(a+1,b)}

Uses the recurrence relation (DLMF 18.9.5):
P_n^{(a,b)}(x) = c1 * P_n^{(a+1,b)}(x) + c2 * P_{n-1}^{(a+1,b)}(x)

where:
c1 = (n + a + b + 1) / (2n + a + b + 1)
c2 = (n + b) / (2n + a + b + 1)
"""
function _jacobi_a_shift_up_matrix(N::Int, a::Float64, b::Float64)
    I_list = Int[]
    J_list = Int[]
    V_list = Float64[]

    for n in 0:(N-1)
        denom = 2*n + a + b + 1

        if abs(denom) > 1e-14
            c1 = (n + a + b + 1) / denom
            c2 = (n + b) / denom

            # P_n^{(a,b)} contributes to P_n^{(a+1,b)} with coefficient c1
            push!(I_list, n + 1); push!(J_list, n + 1); push!(V_list, c1)

            # P_n^{(a,b)} contributes to P_{n-1}^{(a+1,b)} with coefficient c2
            if n > 0
                push!(I_list, n); push!(J_list, n + 1); push!(V_list, c2)
            end
        else
            # Degenerate case: identity
            push!(I_list, n + 1); push!(J_list, n + 1); push!(V_list, 1.0)
        end
    end

    return sparse(I_list, J_list, V_list, N, N)
end

"""
Single step conversion: P_n^{(a,b)} -> P_m^{(a,b+1)}

Similar recurrence with swapped roles of a and b.
"""
function _jacobi_b_shift_up_matrix(N::Int, a::Float64, b::Float64)
    I_list = Int[]
    J_list = Int[]
    V_list = Float64[]

    for n in 0:(N-1)
        denom = 2*n + a + b + 1

        if abs(denom) > 1e-14
            c1 = (n + a + b + 1) / denom
            c2 = (n + a) / denom

            push!(I_list, n + 1); push!(J_list, n + 1); push!(V_list, c1)

            if n > 0
                push!(I_list, n); push!(J_list, n + 1); push!(V_list, c2)
            end
        else
            push!(I_list, n + 1); push!(J_list, n + 1); push!(V_list, 1.0)
        end
    end

    return sparse(I_list, J_list, V_list, N, N)
end

"""
Single step conversion: P_n^{(a,b)} -> P_m^{(a-1,b)}

Uses the inverse/adjoint relationship.
"""
function _jacobi_a_shift_down_matrix(N::Int, a::Float64, b::Float64)
    I_list = Int[]
    J_list = Int[]
    V_list = Float64[]

    for n in 0:(N-1)
        # Use recurrence: P_n^{(a-1,b)} = d1 * P_n^{(a,b)} + d2 * P_{n+1}^{(a,b)}
        # Derived from inverting the shift-up relation

        denom_n = 2*n + a + b
        denom_np1 = 2*(n+1) + a + b

        if abs(denom_n) > 1e-14
            d1 = (2*n + a + b) / (n + a + b)
            push!(I_list, n + 1); push!(J_list, n + 1); push!(V_list, d1)
        end

        if n + 1 < N && abs(denom_np1) > 1e-14
            d2 = -(n + 1) / (n + a + b + 1)
            push!(I_list, n + 1); push!(J_list, n + 2); push!(V_list, d2)
        end
    end

    if isempty(I_list)
        return sparse(I, N, N)
    end

    return sparse(I_list, J_list, V_list, N, N)
end

"""
Single step conversion: P_n^{(a,b)} -> P_m^{(a,b-1)}
"""
function _jacobi_b_shift_down_matrix(N::Int, a::Float64, b::Float64)
    I_list = Int[]
    J_list = Int[]
    V_list = Float64[]

    for n in 0:(N-1)
        denom_n = 2*n + a + b
        denom_np1 = 2*(n+1) + a + b

        if abs(denom_n) > 1e-14
            d1 = (2*n + a + b) / (n + a + b)
            push!(I_list, n + 1); push!(J_list, n + 1); push!(V_list, d1)
        end

        if n + 1 < N && abs(denom_np1) > 1e-14
            d2 = -(n + 1) / (n + a + b + 1)
            push!(I_list, n + 1); push!(J_list, n + 2); push!(V_list, d2)
        end
    end

    if isempty(I_list)
        return sparse(I, N, N)
    end

    return sparse(I_list, J_list, V_list, N, N)
end

"""
Quadrature-based conversion for non-integer parameter shifts.

Evaluates input polynomials at output basis quadrature points,
then computes weighted projection coefficients.
"""
function _jacobi_conversion_quadrature(N::Int, a0::Float64, b0::Float64, a1::Float64, b1::Float64)
    # Get Gauss-Jacobi quadrature for output basis (need 2N points for exactness)
    N_quad = 2 * N
    nodes, weights = gauss_jacobi_quadrature(N_quad, a1, b1)

    # Evaluate input basis polynomials at quadrature points
    P_input = zeros(Float64, N_quad, N)
    for j in 1:N
        P_input[:, j] = jacobi_polynomial.(nodes, j - 1, a0, b0)
    end

    # Evaluate output basis polynomials at quadrature points
    P_output = zeros(Float64, N_quad, N)
    for j in 1:N
        P_output[:, j] = jacobi_polynomial.(nodes, j - 1, a1, b1)
    end

    # Compute conversion matrix via weighted inner products
    # C_ij = <P_i^{out}, P_j^{in}>_w / <P_i^{out}, P_i^{out}>_w
    # where w is the Jacobi weight for output basis

    W = Diagonal(weights)

    # Normalization factors for output basis
    norms = zeros(Float64, N)
    for i in 1:N
        norms[i] = dot(P_output[:, i], W * P_output[:, i])
    end

    # Projection matrix
    C = zeros(Float64, N, N)
    for i in 1:N
        for j in 1:N
            if abs(norms[i]) > 1e-14
                C[i, j] = dot(P_output[:, i], W * P_input[:, j]) / norms[i]
            end
        end
    end

    # Sparsify small entries
    threshold = 1e-14 * maximum(abs.(C))
    C[abs.(C) .< threshold] .= 0.0

    return sparse(C)
end

"""
Evaluate Jacobi polynomial P_n^{(a,b)}(x) using three-term recurrence.
"""
function jacobi_polynomial(x::Float64, n::Int, a::Float64, b::Float64)
    if n == 0
        return 1.0
    elseif n == 1
        return 0.5 * (a - b + (a + b + 2) * x)
    end

    P_prev2 = 1.0
    P_prev1 = 0.5 * (a - b + (a + b + 2) * x)

    for k in 2:n
        # Three-term recurrence coefficients
        k_f = Float64(k)
        c1 = 2 * k_f * (k_f + a + b) * (2*k_f + a + b - 2)
        c2 = (2*k_f + a + b - 1) * (a^2 - b^2)
        c3 = (2*k_f + a + b - 2) * (2*k_f + a + b - 1) * (2*k_f + a + b)
        c4 = 2 * (k_f + a - 1) * (k_f + b - 1) * (2*k_f + a + b)

        if abs(c1) > 1e-14
            P_curr = ((c2 + c3 * x) * P_prev1 - c4 * P_prev2) / c1
        else
            P_curr = P_prev1
        end

        P_prev2 = P_prev1
        P_prev1 = P_curr
    end

    return P_prev1
end

"""
Compute Gauss-Jacobi quadrature nodes and weights.

Uses the Golub-Welsch algorithm based on the eigenvalues of the Jacobi matrix.
Special handling for Chebyshev T (a=b=-0.5) where the standard algorithm fails.
"""
function gauss_jacobi_quadrature(N::Int, a::Float64, b::Float64)
    if N < 1
        return Float64[], Float64[]
    end

    # Special case: Chebyshev T (a=b=-0.5)
    # The standard Golub-Welsch algorithm fails for a+b=-1 because denominators vanish.
    # Use the known Chebyshev-Gauss quadrature formula instead.
    if abs(a + 0.5) < 1e-10 && abs(b + 0.5) < 1e-10
        nodes = zeros(Float64, N)
        weights = zeros(Float64, N)
        for k in 1:N
            nodes[k] = cos((2*k - 1) * π / (2*N))
            weights[k] = π / N
        end
        perm = sortperm(nodes)
        return nodes[perm], weights[perm]
    end

    # Build tridiagonal Jacobi matrix
    # The eigenvalues give the nodes, eigenvectors give the weights

    # Diagonal elements
    d = zeros(Float64, N)
    for n in 0:(N-1)
        num = b^2 - a^2
        denom = (2*n + a + b) * (2*n + a + b + 2)
        if abs(denom) > 1e-14
            d[n + 1] = num / denom
        end
    end

    # Sub/super-diagonal elements
    e = zeros(Float64, N - 1)
    for n in 1:(N-1)
        num = 2 * sqrt(n * (n + a) * (n + b) * (n + a + b))
        denom = (2*n + a + b) * sqrt((2*n + a + b - 1) * (2*n + a + b + 1))
        if abs(denom) > 1e-14
            e[n] = num / denom
        end
    end

    # Build tridiagonal matrix and compute eigendecomposition
    J = SymTridiagonal(d, e)
    eigenvalues, eigenvectors = eigen(J)

    # Nodes are eigenvalues
    nodes = eigenvalues

    # Weights from first component of eigenvectors
    # w_i = μ_0 * v_i[1]^2 where μ_0 = ∫_{-1}^{1} (1-x)^a (1+x)^b dx
    mu0 = 2^(a + b + 1) * gamma(a + 1) * gamma(b + 1) / gamma(a + b + 2)
    weights = mu0 .* eigenvectors[1, :].^2

    return nodes, weights
end

# ============================================================================
# Differentiation matrices
# ============================================================================

"""
    differentiation_matrix(basis::JacobiBasis, order::Int=1)

Build spectral differentiation matrix.
"""
function differentiation_matrix(basis::JacobiBasis, order::Int=1)
    if haskey(basis._differentiation_matrix_cache, order)
        return basis._differentiation_matrix_cache[order]
    end

    N = basis.meta.size
    a, b = basis.a, basis.b

    # Get domain scaling factor
    L = basis.meta.bounds[2] - basis.meta.bounds[1]
    if abs(L) < 1e-14
        throw(ArgumentError("differentiation_matrix: domain length is zero"))
    end
    scale = 2.0 / L

    matrix = _jacobi_differentiation_matrix(N, a, b, order) * scale^order

    basis._differentiation_matrix_cache[order] = matrix
    return matrix
end

"""Build Jacobi differentiation matrix."""
function _jacobi_differentiation_matrix(N::Int, a::Float64, b::Float64, order::Int)
    # d/dx P_n^{(a,b)} = (n + a + b + 1) / 2 * P_{n-1}^{(a+1,b+1)}

    if order == 0
        return sparse(I, N, N)
    end

    # Build single derivative matrix
    D = spzeros(Float64, N, N)

    # Chebyshev T case (a = b = -1/2)
    if abs(a + 0.5) < 1e-10 && abs(b + 0.5) < 1e-10
        D = _chebyshev_t_differentiation_matrix(N)
    # Legendre case (a = b = 0)
    elseif abs(a) < 1e-10 && abs(b) < 1e-10
        D = _legendre_differentiation_matrix(N)
    else
        # General Jacobi
        D = _general_jacobi_differentiation_matrix(N, a, b)
    end

    # Apply multiple times for higher orders
    result = D
    for _ in 2:order
        result = D * result
    end

    return result
end

"""Build Chebyshev T differentiation matrix."""
function _chebyshev_t_differentiation_matrix(N::Int)
    # Standard Chebyshev differentiation recurrence:
    # c'_0 = sum_{j odd} j*c_j  (factor of 1/2 relative to other rows)
    # c'_k = sum_{j=k+1, j-k odd} 2*j*c_j for k >= 1

    I_list = Int[]
    J_list = Int[]
    V_list = Float64[]

    for k in 0:(N-2)
        for j in (k+1):(N-1)
            if (j - k) % 2 == 1
                push!(I_list, k + 1)
                push!(J_list, j + 1)
                # Factor of 1/2 for k=0 row due to Chebyshev normalization
                coeff = k == 0 ? Float64(j) : 2.0 * j
                push!(V_list, coeff)
            end
        end
    end

    if isempty(I_list)
        return spzeros(Float64, N, N)
    end

    return sparse(I_list, J_list, V_list, N, N)
end

"""Build Legendre differentiation matrix."""
function _legendre_differentiation_matrix(N::Int)
    # Legendre differentiation formula:
    # d(P_n)/dx = sum_{k<n, (n-k) odd} (2k+1) * P_k
    #
    # In matrix form: D[k+1, j+1] = (2k+1) when (j-k) is odd and j > k
    # This represents: (df/dx)_k = sum_{j>k, (j-k) odd} (2k+1) * f_j

    I_list = Int[]
    J_list = Int[]
    V_list = Float64[]

    for k in 0:(N-2)
        for j in (k+1):(N-1)
            if (j - k) % 2 == 1
                push!(I_list, k + 1)
                push!(J_list, j + 1)
                push!(V_list, 2.0 * k + 1.0)  # Note: 2*k+1, not 2*j+1
            end
        end
    end

    if isempty(I_list)
        return spzeros(Float64, N, N)
    end

    return sparse(I_list, J_list, V_list, N, N)
end

"""Build general Jacobi differentiation matrix."""
function _general_jacobi_differentiation_matrix(N::Int, a::Float64, b::Float64)
    # d/dx P_n^{(a,b)} = (n + a + b + 1) / 2 * P_{n-1}^{(a+1, b+1)}

    I_list = Int[]
    J_list = Int[]
    V_list = Float64[]

    for n in 1:(N-1)
        coeff = (n + a + b + 1) / 2
        push!(I_list, n)
        push!(J_list, n + 1)
        push!(V_list, coeff)
    end

    if isempty(I_list)
        return spzeros(Float64, N, N)
    end

    return sparse(I_list, J_list, V_list, N, N)
end

# ============================================================================
# Basis dispatcher helpers
# ============================================================================

_basis_builder(::Type{RealFourier}) = _RealFourier_constructor
_basis_builder(::Type{ComplexFourier}) = _ComplexFourier_constructor
_basis_builder(::Type{ChebyshevT}) = _ChebyshevT_constructor
_basis_builder(::Type{ChebyshevU}) = _ChebyshevU_constructor
_basis_builder(::Type{ChebyshevV}) = _ChebyshevV_constructor
_basis_builder(::Type{Legendre}) = _Legendre_constructor
_basis_builder(::Type{Ultraspherical}) = _Ultraspherical_constructor
_basis_builder(::Type{Jacobi}) = _Jacobi_constructor
_basis_builder(::Type{T}) where {T<:Basis} = error("No basis builder registered for type $(T)")

function dispatch_preprocess(::Type{T}, args::Tuple, kwargs::NamedTuple) where {T<:Basis}
    if length(args) != 1
        throw(ArgumentError("$(T) expects exactly one Coordinate argument"))
    end
    return (args, kwargs)
end

function dispatch_check(::Type{T}, args::Tuple, kwargs::NamedTuple) where {T<:Basis}
    coord = args[1]
    if !isa(coord, Coordinate)
        throw(ArgumentError("$(T) requires a Coordinate argument"))
    end
    return true
end

function invoke_constructor(::Type{T}, args::Tuple, kwargs::NamedTuple) where {T<:Basis}
    builder = _basis_builder(T)
    coord = args[1]
    return builder(coord; kwargs...)
end

# ============================================================================
# Basis interface methods
# ============================================================================

function grid_shape(basis::Basis)
    return (basis.meta.size,)
end

function coeff_shape(basis::Basis)
    return (basis.meta.size,)
end

function element_label(basis::Basis)
    return basis.meta.element_label
end

function coordsys(basis::Basis)
    return basis.meta.coordsys
end

function pencil_compatible_size(basis::Basis)
    return basis.meta.size
end

# ============================================================================
# Local grid methods
# ============================================================================

function local_grids(basis::FourierBasis, dist, scales; move_to_arch::Bool=true)
    return (local_grid(basis, dist, scales[1]; move_to_arch=move_to_arch),)
end

function local_grids(basis::JacobiBasis, dist, scales; move_to_arch::Bool=true)
    return (local_grid(basis, dist, scales[1]; move_to_arch=move_to_arch),)
end

function local_grid(basis::Basis, dist, scale; move_to_arch::Bool=true)
    """
    Local grid for a basis.

    GPU-aware: By default, the grid is moved to the distributor's architecture.
    This enables efficient broadcasting with field data on GPU.
    Set `move_to_arch=false` to always return CPU arrays (e.g., for file I/O).
    """
    axis = get_basis_axis(dist, basis)
    native_grid = _native_grid(basis, scale)
    global_size = length(native_grid)
    local_elements = local_indices(dist, axis + 1, global_size)
    local_grid_data = native_grid[local_elements]

    # Map to problem coordinates
    # For FourierBasis, _native_grid already returns grid in problem coordinates
    # so we skip the COV transformation (which would incorrectly rescale)
    local_grid_result = if isa(basis, FourierBasis)
        local_grid_data
    elseif basis.meta.COV !== nothing
        problem_coord(basis.meta.COV, local_grid_data)
    else
        _problem_coord_fallback(basis, local_grid_data)
    end

    # Move to distributor's architecture (GPU or CPU) for efficient broadcasting
    if move_to_arch
        return on_architecture(dist.architecture, local_grid_result)
    else
        return local_grid_result
    end
end

function _problem_coord_fallback(basis::Basis, native_grid)
    if isa(basis, FourierBasis)
        return native_grid
    else
        a, b = basis.meta.bounds
        return @. (b - a) / 2 * native_grid + (b + a) / 2
    end
end

function _native_grid(basis::RealFourier, scale)
    if scale <= 0
        throw(ArgumentError("_native_grid: scale must be positive, got $scale"))
    end
    N = ceil(Int, basis.meta.size * scale)
    if N < 1
        N = 1
    end
    L = basis.meta.bounds[2] - basis.meta.bounds[1]
    dx = L / N
    return [basis.meta.bounds[1] + i * dx for i in 0:N-1]
end

function _native_grid(basis::ComplexFourier, scale)
    if scale <= 0
        throw(ArgumentError("_native_grid: scale must be positive, got $scale"))
    end
    N = ceil(Int, basis.meta.size * scale)
    if N < 1
        N = 1
    end
    L = basis.meta.bounds[2] - basis.meta.bounds[1]
    dx = L / N
    return [basis.meta.bounds[1] + i * dx for i in 0:N-1]
end

function _native_grid(basis::ChebyshevT, scale)
    if scale <= 0
        throw(ArgumentError("_native_grid: scale must be positive, got $scale"))
    end
    N = ceil(Int, basis.meta.size * scale)
    if N < 1
        N = 1
    end
    # Handle N=1 edge case (single point at center)
    if N == 1
        return [0.0]
    end
    # Gauss-Lobatto points: x_k = -cos(π*k/(N-1))
    return [-cos(π * k / (N - 1)) for k in 0:N-1]
end

function _native_grid(basis::ChebyshevU, scale)
    if scale <= 0
        throw(ArgumentError("_native_grid: scale must be positive, got $scale"))
    end
    N = ceil(Int, basis.meta.size * scale)
    if N < 1
        N = 1
    end
    # Gauss points: x_k = -cos(π*(k+0.5)/N)
    return [-cos(π * (k + 0.5) / N) for k in 0:N-1]
end

function _native_grid(basis::ChebyshevV, scale)
    if scale <= 0
        throw(ArgumentError("_native_grid: scale must be positive, got $scale"))
    end
    N = ceil(Int, basis.meta.size * scale)
    if N < 1
        N = 1
    end
    # ChebyshevV = Jacobi(3/2, 3/2), use Gauss-Jacobi quadrature points
    nodes, _ = gauss_jacobi_quadrature(N, basis.a, basis.b)
    return nodes
end

function _native_grid(basis::Legendre, scale)
    if scale <= 0
        throw(ArgumentError("_native_grid: scale must be positive, got $scale"))
    end
    N = ceil(Int, basis.meta.size * scale)
    if N < 1
        N = 1
    end
    # Gauss-Legendre points (roots of P_N)
    nodes, _ = gauss_jacobi_quadrature(N, 0.0, 0.0)
    return nodes
end

function _native_grid(basis::Ultraspherical, scale)
    if scale <= 0
        throw(ArgumentError("_native_grid: scale must be positive, got $scale"))
    end
    N = ceil(Int, basis.meta.size * scale)
    if N < 1
        N = 1
    end
    nodes, _ = gauss_jacobi_quadrature(N, basis.a, basis.b)
    return nodes
end

function _native_grid(basis::Jacobi, scale)
    if scale <= 0
        throw(ArgumentError("_native_grid: scale must be positive, got $scale"))
    end
    N = ceil(Int, basis.meta.size * scale)
    if N < 1
        N = 1
    end
    nodes, _ = gauss_jacobi_quadrature(N, basis.a, basis.b)
    return nodes
end

# ============================================================================
# Basis evaluation functions
# ============================================================================

"""
    _ensure_cpu_coords(coords)

Ensure coordinates are on CPU for basis evaluation.
If coords is a GPU array, convert to CPU and issue a warning.
Returns (cpu_coords, was_gpu) tuple.
"""
function _ensure_cpu_coords(coords)
    if is_gpu_array(coords)
        @warn "evaluate_basis: GPU coordinates detected, converting to CPU for evaluation. " *
              "For GPU-native evaluation, consider using transform-based methods."
        return (on_architecture(CPU(), coords), true)
    end
    return (coords, false)
end

"""
    evaluate_basis(basis::RealFourier, coords, modes)

Evaluate RealFourier basis functions at given coordinates for specified modes.

Returns a matrix of size (n_points, n_modes) where result[i, j] is the
value of mode j at coordinate point i.

Note: This function operates on CPU. GPU coordinates will be automatically
converted to CPU for evaluation.
"""
function evaluate_basis(basis::RealFourier, coords, modes)
    # Ensure coordinates are on CPU
    cpu_coords, _ = _ensure_cpu_coords(coords)

    L = basis.meta.bounds[2] - basis.meta.bounds[1]
    if abs(L) < 1e-14
        throw(ArgumentError("evaluate_basis: domain length is zero"))
    end
    normalized_coords = @. 2π * (cpu_coords - basis.meta.bounds[1]) / L

    n_modes = length(modes)
    n_points = length(cpu_coords)
    result = zeros(basis.meta.dtype, n_points, n_modes)

    for (i, mode) in enumerate(modes)
        if mode == 0
            result[:, i] .= 1.0
        else
            k = (mode + 1) ÷ 2
            if isodd(mode)
                result[:, i] .= cos.(k * normalized_coords)
            else
                result[:, i] .= .-sin.(k * normalized_coords)
            end
        end
    end

    return result
end

"""
    evaluate_basis(basis::ComplexFourier, coords, modes)

Evaluate ComplexFourier basis functions at given coordinates for specified modes.

Note: This function operates on CPU. GPU coordinates will be automatically
converted to CPU for evaluation.
"""
function evaluate_basis(basis::ComplexFourier, coords, modes)
    # Ensure coordinates are on CPU
    cpu_coords, _ = _ensure_cpu_coords(coords)

    L = basis.meta.bounds[2] - basis.meta.bounds[1]
    if abs(L) < 1e-14
        throw(ArgumentError("evaluate_basis: domain length is zero"))
    end
    normalized_coords = @. 2π * (cpu_coords - basis.meta.bounds[1]) / L

    n_modes = length(modes)
    n_points = length(cpu_coords)
    result = zeros(basis.meta.dtype, n_points, n_modes)

    for (i, k) in enumerate(modes)
        result[:, i] .= exp.(im * k * normalized_coords)
    end

    return result
end

"""
    evaluate_basis(basis::ChebyshevT, coords, modes)

Evaluate ChebyshevT basis functions at given coordinates for specified modes.

Note: This function operates on CPU. GPU coordinates will be automatically
converted to CPU for evaluation.
"""
function evaluate_basis(basis::ChebyshevT, coords, modes)
    # Ensure coordinates are on CPU
    cpu_coords, _ = _ensure_cpu_coords(coords)

    a, b = basis.meta.bounds
    if abs(b - a) < 1e-14
        throw(ArgumentError("evaluate_basis: domain length is zero"))
    end
    mapped_coords = @. 2 * (cpu_coords - a) / (b - a) - 1

    n_modes = length(modes)
    n_points = length(cpu_coords)
    result = zeros(basis.meta.dtype, n_points, n_modes)

    for (i, n) in enumerate(modes)
        if n == 0
            result[:, i] .= 1.0
        elseif n == 1
            result[:, i] .= mapped_coords
        else
            result[:, i] .= cos.(n * acos.(clamp.(mapped_coords, -1, 1)))
        end
    end

    return result
end

"""
    evaluate_basis(basis::ChebyshevU, coords, modes)

Evaluate ChebyshevU basis functions at given coordinates for specified modes.

Note: This function operates on CPU. GPU coordinates will be automatically
converted to CPU for evaluation.
"""
function evaluate_basis(basis::ChebyshevU, coords, modes)
    # Ensure coordinates are on CPU
    cpu_coords, _ = _ensure_cpu_coords(coords)

    a, b = basis.meta.bounds
    if abs(b - a) < 1e-14
        throw(ArgumentError("evaluate_basis: domain length is zero"))
    end
    mapped_coords = @. 2 * (cpu_coords - a) / (b - a) - 1

    n_modes = length(modes)
    n_points = length(cpu_coords)
    result = zeros(basis.meta.dtype, n_points, n_modes)

    for (i, n) in enumerate(modes)
        # U_n(x) = sin((n+1)*arccos(x)) / sin(arccos(x))
        # Using stable form: U_n(cos(θ)) = sin((n+1)θ)/sin(θ)
        theta = acos.(clamp.(mapped_coords, -1, 1))
        sin_theta = sin.(theta)
        # Handle x = ±1 (theta = 0 or π) where sin(theta) = 0
        for j in eachindex(mapped_coords)
            if abs(sin_theta[j]) < 1e-14
                # L'Hôpital: U_n(±1) = (±1)^n * (n+1)
                result[j, i] = mapped_coords[j]^n * (n + 1)
            else
                result[j, i] = sin((n + 1) * theta[j]) / sin_theta[j]
            end
        end
    end

    return result
end

"""
    evaluate_basis(basis::JacobiBasis, coords, modes)

Evaluate general Jacobi basis functions at given coordinates for specified modes.

Note: This function operates on CPU. GPU coordinates will be automatically
converted to CPU for evaluation.
"""
function evaluate_basis(basis::JacobiBasis, coords, modes)
    # Ensure coordinates are on CPU
    cpu_coords, _ = _ensure_cpu_coords(coords)

    # General Jacobi polynomial evaluation using three-term recurrence
    a_param, b_param = basis.a, basis.b
    a, b = basis.meta.bounds
    if abs(b - a) < 1e-14
        throw(ArgumentError("evaluate_basis: domain length is zero"))
    end
    mapped_coords = @. 2 * (cpu_coords - a) / (b - a) - 1

    n_modes = length(modes)
    n_points = length(cpu_coords)
    result = zeros(basis.meta.dtype, n_points, n_modes)

    for (i, n) in enumerate(modes)
        result[:, i] .= jacobi_polynomial.(mapped_coords, n, a_param, b_param)
    end

    return result
end

function synchronize_basis!(basis::Basis)
    """No-op for CPU-only mode."""
end
