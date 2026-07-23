# Core basis types, metadata, constructors, and MPI compatibility helpers.

# ============================================================================
# Abstract types for basis hierarchy
# ============================================================================

abstract type Basis end
abstract type IntervalBasis <: Basis end
abstract type JacobiBasis <: IntervalBasis end  # Base for all Jacobi-type bases
abstract type FourierBasis <: IntervalBasis end  # Base for Fourier bases

# ============================================================================
# Fourier-only MPI validation helpers
# ============================================================================

"""
    is_fourier_basis(basis)

Check if a basis is a Fourier basis (RealFourier or ComplexFourier).
"""
is_fourier_basis(::FourierBasis) = true
is_fourier_basis(::Basis) = false
is_fourier_basis(::Nothing) = true  # No basis = OK (e.g., 1D/2D cases)

"""
    is_complex_fourier_basis(basis)

Check if a basis is ComplexFourier (not RealFourier).
MPI parallelization requires ComplexFourier because RealFourier's half-spectrum
(rfft) layout is incompatible with MPI transposes.
"""
is_complex_fourier_basis(basis::FourierBasis) = isa(basis, ComplexFourier)
is_complex_fourier_basis(::Basis) = false
is_complex_fourier_basis(::Nothing) = true  # No basis = OK

"""
    is_pure_fourier_domain(bases::Tuple)

Check if all bases in the domain are Fourier bases.
"""
function is_pure_fourier_domain(bases::Tuple)
    return all(is_fourier_basis(b) for b in bases)
end

is_pure_fourier_domain(bases::Vector) = all(is_fourier_basis(b) for b in bases)

"""
    is_pure_complex_fourier_domain(bases)

Check if all bases are ComplexFourier (required for MPI).
"""
is_pure_complex_fourier_domain(bases::Tuple) = all(is_complex_fourier_basis(b) for b in bases)
is_pure_complex_fourier_domain(bases::Vector) = all(is_complex_fourier_basis(b) for b in bases)

"""Basis-level eligibility for the custom three-dimensional GPU+MPI DCT-I path."""
function _distributed_gpu_dct_bases_supported(bases)
    length(bases) == 3 || return false
    all(b -> isa(b, RealFourier) || isa(b, ComplexFourier) || isa(b, ChebyshevT),
        bases) || return false
    any(b -> isa(b, ChebyshevT), bases) || return false
    any(is_fourier_basis, bases) || return false
    any(i -> isa(bases[i], RealFourier), 2:3) && return false
    if isa(bases[1], RealFourier) && any(is_fourier_basis, bases[2:3])
        return false
    end
    return true
end

"""
    validate_mpi_fourier_only(bases, nprocs::Int; use_pencil_arrays::Bool=true)

Validate that MPI parallelization is compatible with the provided bases.

device=CPU() with MPI (PencilArrays/PencilFFTs):
  - Pure Fourier domains (RealFourier + ComplexFourier): fully supported
  - Mixed Fourier-Chebyshev domains: supported via decomp_dims + solve-layout transpose

device=GPU() with MPI (TransposableField):
  - Pure ComplexFourier domains
  - Eligible 3D Fourier/Chebyshev layouts handled by the distributed DCT-I path
  - Other RealFourier or non-Fourier layouts are rejected explicitly
"""
function validate_mpi_fourier_only(bases, nprocs::Int; use_pencil_arrays::Bool=true)
    if nprocs <= 1
        return true
    end

    # CPU+PencilArrays: supports pure Fourier AND mixed Fourier-Chebyshev
    if use_pencil_arrays
        # Only reject bases that have no MPI path at all (e.g., pure Legendre)
        has_fourier = any(is_fourier_basis(b) for b in bases)
        has_legendre = any(isa(b, Legendre) for b in bases)
        if has_legendre
            error("MPI parallelization is not yet supported for Legendre bases. " *
                  "Use serial execution (nprocs=1).")
        end
        return true
    end

    # GPU+MPI custom distributed DCT-I: allow exactly the basis layouts that the
    # CUDA transform dispatcher can execute. Keep this basis-level predicate in
    # core so Domain validation and the extension cannot drift apart.
    _distributed_gpu_dct_bases_supported(bases) && return true

    # Other GPU+MPI TransposableField domains: pure ComplexFourier only.
    if !is_pure_fourier_domain(bases)
        non_fourier = [typeof(b).name.name for b in bases if !is_fourier_basis(b)]
        error("GPU+MPI (TransposableField, nprocs=$nprocs) does not support this basis layout. " *
              "Found non-Fourier bases: $(join(non_fourier, ", ")). " *
              "Use an eligible 3D distributed GPU DCT-I layout or device=CPU() (PencilArrays); " *
              "CPU staging fallback is disabled.")
    end

    if !is_pure_complex_fourier_domain(bases)
        real_fourier = [typeof(b).name.name for b in bases if is_fourier_basis(b) && !is_complex_fourier_basis(b)]
        error("GPU+MPI (TransposableField, nprocs=$nprocs) requires ComplexFourier bases. " *
              "Found RealFourier: $(join(real_fourier, ", ")). " *
              "RealFourier's half-spectrum is incompatible with custom MPI transposes. " *
              "Use ComplexFourier for GPU+MPI, or device=CPU() with MPI for RealFourier.")
    end

    return true
end

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
    transforms::Dict{Any, Any}
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
    transforms = Dict{Any, Any}()
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
    transforms::Dict{Any, Any}
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
    transforms = Dict{Any, Any}()
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
    transforms::Dict{Any, Any}
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
    transforms = Dict{Any, Any}()
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
    transforms::Dict{Any, Any}
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
    transforms = Dict{Any, Any}()
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
    transforms::Dict{Any, Any}
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
    transforms = Dict{Any, Any}()
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
    transforms::Dict{Any, Any}
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
    transforms = Dict{Any, Any}()
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
    transforms::Dict{Any, Any}
    # Cached wavenumbers
    _wavenumbers::Union{Nothing, Vector{Float64}}
    _product_matrix_cache::Dict{Tuple, AbstractMatrix}
end

function _build_real_fourier(coord::Coordinate;
                             size::Int=32,
                             bounds::Tuple{Float64,Float64}=(0.0,2π),
                             dealias::Float64=3/2,
                             dtype=Float64)
    meta = BasisMeta(coord.coordsys, coord.name, 1, size, bounds, dealias, dtype;
                     native_bounds=FOURIER_NATIVE_BOUNDS, constant_mode_value=1.0)
    transforms = Dict{Any, Any}()
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
    transforms::Dict{Any, Any}
    _wavenumbers::Union{Nothing, Vector{Float64}}
    _product_matrix_cache::Dict{Tuple, AbstractMatrix}
end

function _build_complex_fourier(coord::Coordinate;
                                size::Int=32,
                                bounds::Tuple{Float64,Float64}=(0.0,2π),
                                dealias::Float64=3/2,
                                dtype=ComplexF64)
    meta = BasisMeta(coord.coordsys, coord.name, 1, size, bounds, dealias, dtype;
                     native_bounds=FOURIER_NATIVE_BOUNDS, constant_mode_value=1.0)
    transforms = Dict{Any, Any}()
    product_cache = Dict{Tuple, AbstractMatrix}()
    return ComplexFourier(meta, transforms, nothing, product_cache)
end

const _ComplexFourier_constructor = _build_complex_fourier

function ComplexFourier(coord::Coordinate; kwargs...)
    return multiclass_new(ComplexFourier, coord; kwargs...)
end

# Alias
const Fourier = RealFourier

# Convenience constructors for test compatibility: Fourier(coords, name, size)
function RealFourier(coords::CoordinateSystem, name::String, size::Int; bounds::Tuple=(0.0, 2π), kwargs...)
    coord = coords[name]
    return RealFourier(coord; size=size, bounds=bounds, kwargs...)
end

function ComplexFourier(coords::CoordinateSystem, name::String, size::Int; bounds::Tuple=(0.0, 2π), kwargs...)
    coord = coords[name]
    return ComplexFourier(coord; size=size, bounds=bounds, kwargs...)
end

# ============================================================================
