"""
Spectral basis classes

Translated from dedalus/core/basis.py
"""

using LinearAlgebra
using SparseArrays
using FFTW

# Include GPU manager for device-agnostic operations

abstract type Basis end

struct AffineCOV
    a::Float64
    b::Float64
    c::Float64
    d::Float64
    
    function AffineCOV(a=1.0, b=0.0, c=0.0, d=1.0)
        new(a, b, c, d)
    end
end

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
    
    
    function BasisMeta(coordsys, element_label, dim, size, bounds, dealias, dtype, constant::Vector{Bool}, subaxis_dependence::Vector{Bool}, )
        new(coordsys, element_label, dim, size, bounds, dealias, dtype, constant, subaxis_dependence, device_config)
    end
end

function BasisMeta(coordsys, element_label, dim, size, bounds, dealias, dtype, device_config=nothing; constant=nothing, subaxis_dependence=nothing)
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
    return BasisMeta(coordsys, element_label, dim, size, bounds, dealias, dtype, const_vec, dep_vec, device_config)
end

struct RealFourier <: Basis
    meta::BasisMeta
    transforms::Dict{String, Any}
end

function _build_real_fourier(coord::Coordinate; size::Int=32, bounds::Tuple{Float64,Float64}=(0.0,2π), dealias::Float64=1.0, dtype=Float64)
    meta = BasisMeta(coord.coordsys, coord.name, 1, size, bounds, dealias, dtype, device_config)
    transforms = Dict{String, Any}()
    return RealFourier(meta, transforms)
end

const _RealFourier_constructor = _build_real_fourier

function RealFourier(coord::Coordinate; kwargs...)
    return multiclass_new(RealFourier, coord; kwargs...)
end

struct ComplexFourier <: Basis
    meta::BasisMeta
    transforms::Dict{String, Any}
end

function _build_complex_fourier(coord::Coordinate; size::Int=32, bounds::Tuple{Float64,Float64}=(0.0,2π), dealias::Float64=1.0, dtype=ComplexF64)
    meta = BasisMeta(coord.coordsys, coord.name, 1, size, bounds, dealias, dtype, device_config)
    transforms = Dict{String, Any}()
    return ComplexFourier(meta, transforms)
end

const _ComplexFourier_constructor = _build_complex_fourier

function ComplexFourier(coord::Coordinate; kwargs...)
    return multiclass_new(ComplexFourier, coord; kwargs...)
end

const Fourier = RealFourier

struct ChebyshevT <: Basis
    meta::BasisMeta
    transforms::Dict{String, Any}
    a::Float64
    b::Float64
end

function _build_chebyshev_t(coord::Coordinate; size::Int=32, bounds::Tuple{Float64,Float64}=(-1.0,1.0), dealias::Float64=1.0, dtype=Float64, a=0.0, b=0.0)
    meta = BasisMeta(coord.coordsys, coord.name, 1, size, bounds, dealias, dtype, device_config)
    transforms = Dict{String, Any}()
    return ChebyshevT(meta, transforms, a, b)
end

const _ChebyshevT_constructor = _build_chebyshev_t

function ChebyshevT(coord::Coordinate; kwargs...)
    return multiclass_new(ChebyshevT, coord; kwargs...)
end

struct ChebyshevU <: Basis
    meta::BasisMeta
    transforms::Dict{String, Any}
end

function _build_chebyshev_u(coord::Coordinate; size::Int=32, bounds::Tuple{Float64,Float64}=(-1.0,1.0), dealias::Float64=1.0, dtype=Float64)
    meta = BasisMeta(coord.coordsys, coord.name, 1, size, bounds, dealias, dtype, device_config)
    transforms = Dict{String, Any}()
    return ChebyshevU(meta, transforms)
end

const _ChebyshevU_constructor = _build_chebyshev_u

function ChebyshevU(coord::Coordinate; kwargs...)
    return multiclass_new(ChebyshevU, coord; kwargs...)
end

struct Legendre <: Basis
    meta::BasisMeta
    transforms::Dict{String, Any}
end

function _build_legendre(coord::Coordinate; size::Int=32, bounds::Tuple{Float64,Float64}=(-1.0,1.0), dealias::Float64=1.0, dtype=Float64)
    meta = BasisMeta(coord.coordsys, coord.name, 1, size, bounds, dealias, dtype, device_config)
    transforms = Dict{String, Any}()
    return Legendre(meta, transforms)
end

const _Legendre_constructor = _build_legendre

function Legendre(coord::Coordinate; kwargs...)
    return multiclass_new(Legendre, coord; kwargs...)
end

struct Ultraspherical <: Basis
    meta::BasisMeta
    transforms::Dict{String, Any}
    alpha::Float64
end

function _build_ultraspherical(coord::Coordinate; alpha::Float64=0.5, size::Int=32, bounds::Tuple{Float64,Float64}=(-1.0,1.0), dealias::Float64=1.0, dtype=Float64)
    meta = BasisMeta(coord.coordsys, coord.name, 1, size, bounds, dealias, dtype, device_config)
    transforms = Dict{String, Any}()
    return Ultraspherical(meta, transforms, alpha)
end

const _Ultraspherical_constructor = _build_ultraspherical

function Ultraspherical(coord::Coordinate; kwargs...)
    return multiclass_new(Ultraspherical, coord; kwargs...)
end

# Jacobi polynomials base class
struct Jacobi <: Basis
    meta::BasisMeta
    transforms::Dict{String, Any}
    a::Float64
    b::Float64
end

function _build_jacobi(coord::Coordinate; a::Float64=0.0, b::Float64=0.0, size::Int=32, bounds::Tuple{Float64,Float64}=(-1.0,1.0), dealias::Float64=1.0, dtype=Float64)
    meta = BasisMeta(coord.coordsys, coord.name, 1, size, bounds, dealias, dtype, device_config)
    transforms = Dict{String, Any}()
    return Jacobi(meta, transforms, a, b)
end

const _Jacobi_constructor = _build_jacobi

function Jacobi(coord::Coordinate; kwargs...)
    return multiclass_new(Jacobi, coord; kwargs...)
end

# Disk and annulus bases for polar coordinates
struct DiskBasis <: Basis
    meta::BasisMeta
    transforms::Dict{String, Any}
    radius::Float64
end

function _build_disk_basis(coord::Coordinate; radius::Float64=1.0, size::Int=32, dealias::Union{Float64, Tuple{Vararg{Float64}}}=(1.0,1.0), dtype=ComplexF64)
    dealias_tuple = dealias isa Tuple ? dealias : (dealias, dealias)
    bounds = (0.0, radius)
    meta = BasisMeta(coord.coordsys, coord.name, 2, size, bounds, dealias_tuple, dtype, device_config;
                     constant=[false, false], subaxis_dependence=[true, true])
    transforms = Dict{String, Any}()
    return DiskBasis(meta, transforms, radius)
end

const _DiskBasis_constructor = _build_disk_basis

function DiskBasis(coord::Coordinate; kwargs...)
    return multiclass_new(DiskBasis, coord; kwargs...)
end

struct AnnulusBasis <: Basis
    meta::BasisMeta
    transforms::Dict{String, Any}
    radii::Tuple{Float64, Float64}
end

function _build_annulus_basis(coord::Coordinate; radii::Tuple{Float64,Float64}=(0.5,1.0), size::Int=32, dealias::Union{Float64, Tuple{Vararg{Float64}}}=(1.0,1.0), dtype=ComplexF64)
    dealias_tuple = dealias isa Tuple ? dealias : (dealias, dealias)
    meta = BasisMeta(coord.coordsys, coord.name, 2, size, radii, dealias_tuple, dtype, device_config;
                     constant=[false, false], subaxis_dependence=[false, true])
    transforms = Dict{String, Any}()
    return AnnulusBasis(meta, transforms, radii)
end

const _AnnulusBasis_constructor = _build_annulus_basis

function AnnulusBasis(coord::Coordinate; kwargs...)
    return multiclass_new(AnnulusBasis, coord; kwargs...)
end

# Spherical bases
struct SphereBasis <: Basis
    meta::BasisMeta
    transforms::Dict{String, Any}
    radius::Float64
end

function _build_sphere_basis(coord::Coordinate; radius::Float64=1.0, size::Int=32, dealias::Union{Float64, Tuple{Vararg{Float64}}}=(1.0,1.0,1.0), dtype=ComplexF64)
    dealias_tuple = dealias isa Tuple ? dealias : (dealias, dealias, dealias)
    bounds = (0.0, radius)
    meta = BasisMeta(coord.coordsys, coord.name, 3, size, bounds, dealias_tuple, dtype, device_config;
                     constant=[false, false, false], subaxis_dependence=[false, true, true])
    transforms = Dict{String, Any}()
    return SphereBasis(meta, transforms, radius)
end

const _SphereBasis_constructor = _build_sphere_basis

function SphereBasis(coord::Coordinate; kwargs...)
    return multiclass_new(SphereBasis, coord; kwargs...)
end

struct BallBasis <: Basis
    meta::BasisMeta
    transforms::Dict{String, Any}
    radius::Float64
end

function _build_ball_basis(coord::Coordinate; radius::Float64=1.0, size::Int=32, dealias::Union{Float64, Tuple{Vararg{Float64}}}=(1.0,1.0,1.0), dtype=ComplexF64)
    dealias_tuple = dealias isa Tuple ? dealias : (dealias, dealias, dealias)
    bounds = (0.0, radius)
    meta = BasisMeta(coord.coordsys, coord.name, 3, size, bounds, dealias_tuple, dtype, device_config;
                     constant=[false, false, false], subaxis_dependence=[false, true, true])
    transforms = Dict{String, Any}()
    return BallBasis(meta, transforms, radius)
end

const _BallBasis_constructor = _build_ball_basis

function BallBasis(coord::Coordinate; kwargs...)
    return multiclass_new(BallBasis, coord; kwargs...)
end

struct ShellBasis <: Basis
    meta::BasisMeta
    transforms::Dict{String, Any}
    radii::Tuple{Float64, Float64}
end

function _build_shell_basis(coord::Coordinate; radii::Tuple{Float64,Float64}=(0.5,1.0), size::Int=32, dealias::Union{Float64, Tuple{Vararg{Float64}}}=(1.0,1.0,1.0), dtype=ComplexF64)
    dealias_tuple = dealias isa Tuple ? dealias : (dealias, dealias, dealias)
    meta = BasisMeta(coord.coordsys, coord.name, 3, size, radii, dealias_tuple, dtype, device_config;
                     constant=[false, false, false], subaxis_dependence=[false, true, true])
    transforms = Dict{String, Any}()
    return ShellBasis(meta, transforms, radii)
end

const _ShellBasis_constructor = _build_shell_basis

function ShellBasis(coord::Coordinate; kwargs...)
    return multiclass_new(ShellBasis, coord; kwargs...)
end

# ---------------------------------------------------------------------------
# Basis dispatcher helpers
# ---------------------------------------------------------------------------

_basis_builder(::Type{RealFourier}) = _RealFourier_constructor
_basis_builder(::Type{ComplexFourier}) = _ComplexFourier_constructor
_basis_builder(::Type{ChebyshevT}) = _ChebyshevT_constructor
_basis_builder(::Type{ChebyshevU}) = _ChebyshevU_constructor
_basis_builder(::Type{Legendre}) = _Legendre_constructor
_basis_builder(::Type{Ultraspherical}) = _Ultraspherical_constructor
_basis_builder(::Type{Jacobi}) = _Jacobi_constructor
_basis_builder(::Type{DiskBasis}) = _DiskBasis_constructor
_basis_builder(::Type{AnnulusBasis}) = _AnnulusBasis_constructor
_basis_builder(::Type{SphereBasis}) = _SphereBasis_constructor
_basis_builder(::Type{BallBasis}) = _BallBasis_constructor
_basis_builder(::Type{ShellBasis}) = _ShellBasis_constructor
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

# Basis interface methods
function derivative_basis(basis::Basis, order::Int=1)
    # Return derivative basis - placeholder implementation
    return basis
end

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

# For compatibility with PencilArrays
function pencil_compatible_size(basis::Basis)
    return basis.meta.size
end

# Local grid methods for each basis type
# Following Dedalus implementation in basis.py

function local_grids(basis::RealFourier, dist, scales)
    """Local grids for real Fourier basis."""
    return (local_grid(basis, dist, scales[1]),)
end

function local_grids(basis::ComplexFourier, dist, scales)
    """Local grids for complex Fourier basis."""
    return (local_grid(basis, dist, scales[1]),)
end

function local_grids(basis::ChebyshevT, dist, scales)
    """Local grids for Chebyshev T basis."""
    return (local_grid(basis, dist, scales[1]),)
end

function local_grids(basis::ChebyshevU, dist, scales)
    """Local grids for Chebyshev U basis."""
    return (local_grid(basis, dist, scales[1]),)
end

function local_grids(basis::Legendre, dist, scales)
    """Local grids for Legendre basis."""
    return (local_grid(basis, dist, scales[1]),)
end

function local_grids(basis::Ultraspherical, dist, scales)
    """Local grids for Ultraspherical basis."""
    return (local_grid(basis, dist, scales[1]),)
end

function local_grids(basis::Jacobi, dist, scales)
    """Local grids for Jacobi basis."""
    return (local_grid(basis, dist, scales[1]),)
end

function local_grids(basis::DiskBasis, dist, scales)
    """Local grids for Disk basis."""
    return (local_grid(basis, dist, scales[1]),)
end

function local_grid(basis::Basis, dist, scale)
    """
    Local grid for a basis.
    Following Dedalus implementation in basis.py:374
    """
    # Get the axis index for this basis
    axis = get_basis_axis(dist, basis)
    
    # Get local elements from the layout
    layout = get_layout(dist, (basis,))
    local_elements = local_indices(layout.dist, axis + 1)  # Julia is 1-indexed
    
    # Create the native grid with scaling
    native_grid = _native_grid(basis, scale)
    
    # Extract local elements
    local_grid_data = native_grid[local_elements]
    
    # Map to problem coordinates if needed
    problem_grid = problem_coord(basis, local_grid_data)
    
    return problem_grid
end

function _native_grid(basis::RealFourier, scale)
    """Native grid for real Fourier basis."""
    N = Int(round(basis.meta.size * scale))
    L = basis.meta.bounds[2] - basis.meta.bounds[1]
    dx = L / N
    grid_cpu = [basis.meta.bounds[1] + i * dx for i in 0:N-1]
    return grid_cpu
end

function _native_grid(basis::ComplexFourier, scale)
    """Native grid for complex Fourier basis."""
    N = Int(round(basis.meta.size * scale))
    L = basis.meta.bounds[2] - basis.meta.bounds[1]
    dx = L / N
    grid_cpu = [basis.meta.bounds[1] + i * dx for i in 0:N-1]
    return grid_cpu
end

function _native_grid(basis::ChebyshevT, scale)
    """Native grid for Chebyshev T basis."""
    N = Int(round(basis.meta.size * scale))
    # Gauss-Lobatto points
    grid_cpu = [-cos(π * k / (N - 1)) for k in 0:N-1]
    return grid_cpu
end

function _native_grid(basis::ChebyshevU, scale)
    """Native grid for Chebyshev U basis."""
    N = Int(round(basis.meta.size * scale))
    # Gauss points
    grid_cpu = [-cos(π * (k + 0.5) / N) for k in 0:N-1]
    return grid_cpu
end

function _native_grid(basis::Legendre, scale)
    """Native grid for Legendre basis."""
    N = Int(round(basis.meta.size * scale))
    # Use Gauss-Lobatto points for Legendre
    grid_cpu = [-cos(π * k / (N - 1)) for k in 0:N-1]
    return grid_cpu
end

function _native_grid(basis::Ultraspherical, scale)
    """Native grid for Ultraspherical basis."""
    N = Int(round(basis.meta.size * scale))
    # Use Gauss-Lobatto points
    grid_cpu = [-cos(π * k / (N - 1)) for k in 0:N-1]
    return grid_cpu
end

function _native_grid(basis::Jacobi, scale)
    """Native grid for Jacobi basis."""
    N = Int(round(basis.meta.size * scale))
    # Use Gauss-Lobatto points
    grid_cpu = [-cos(π * k / (N - 1)) for k in 0:N-1]
    return grid_cpu
end

function _native_grid(basis::DiskBasis, scale)
    """Native grid for Disk basis."""
    N = Int(round(basis.meta.size * scale))
    # Radial coordinates from 0 to radius
    dr = basis.radius / (N - 1)
    grid_cpu = [i * dr for i in 0:N-1]
    return grid_cpu
end

function problem_coord(basis::Basis, native_grid)
    """
    Map native coordinates to problem coordinates.
    For most bases, this involves scaling to the physical bounds.
    """
    if isa(basis, RealFourier) || isa(basis, ComplexFourier)
        # Fourier bases are already in problem coordinates
        return native_grid
    elseif isa(basis, DiskBasis)
        # Disk basis is already in problem coordinates
        return native_grid
    else
        # Spectral bases need to be mapped to physical bounds
        a, b = basis.meta.bounds
        return @. (b - a) / 2 * native_grid + (b + a) / 2
    end
end

# Note: get_basis_axis is implemented in distributor.jl

# GPU-compatible basis evaluation functions

function evaluate_basis_gpu(basis::RealFourier, coords, modes)
    """Evaluate real Fourier basis functions on GPU."""
    # Ensure coordinates are on the correct device
    coords_device = coords
    modes_device = modes
    
    # Normalize coordinates to [0, 2π]
    L = basis.meta.bounds[2] - basis.meta.bounds[1]
    normalized_coords = @. 2π * (coords_device - basis.meta.bounds[1]) / L
    
    # Evaluate Fourier modes: cos(k*x) and sin(k*x)
    n_modes = length(modes_device)
    n_points = length(coords_device)
    result = device_zeros(basis.meta.dtype, (n_points, n_modes), basis.meta)
    
    # GPU-optimized trigonometric evaluation
    for (i, k) in enumerate(modes_device)
        if k == 0
            result[:, i] .= 1.0  # Constant mode
        else
            result[:, i] .= cos.(k * normalized_coords)
        end
    end
    
    return result
end

function evaluate_basis_gpu(basis::ComplexFourier, coords, modes)
    """Evaluate complex Fourier basis functions on GPU."""
    coords_device = coords
    modes_device = modes
    
    L = basis.meta.bounds[2] - basis.meta.bounds[1]
    normalized_coords = @. 2π * (coords_device - basis.meta.bounds[1]) / L
    
    n_modes = length(modes_device)
    n_points = length(coords_device)
    result = device_zeros(basis.meta.dtype, (n_points, n_modes), basis.meta)
    
    # Complex exponentials: exp(i*k*x)
    for (i, k) in enumerate(modes_device)
        result[:, i] .= exp.(im * k * normalized_coords)
    end
    
    return result
end

function evaluate_basis_gpu(basis::ChebyshevT, coords, modes)
    """Evaluate Chebyshev T polynomials on GPU."""
    coords_device = coords
    modes_device = modes
    
    # Map to [-1, 1] interval
    a, b = basis.meta.bounds
    mapped_coords = @. 2 * (coords_device - a) / (b - a) - 1
    
    n_modes = length(modes_device)
    n_points = length(coords_device)
    result = device_zeros(basis.meta.dtype, (n_points, n_modes), basis.meta)
    
    # Evaluate Chebyshev polynomials using recurrence relation
    for (i, n) in enumerate(modes_device)
        if n == 0
            result[:, i] .= 1.0
        elseif n == 1
            result[:, i] .= mapped_coords
        else
            # Use cos(n*arccos(x)) for stability on GPU
            result[:, i] .= cos.(n * acos.(clamp.(mapped_coords, -1, 1)))
        end
    end
    
    return result
end

function derivative_basis_gpu(basis::RealFourier, coords, modes, order=1)
    """Evaluate derivatives of real Fourier basis on GPU."""
    coords_device = coords
    modes_device = modes
    
    L = basis.meta.bounds[2] - basis.meta.bounds[1]
    normalized_coords = @. 2π * (coords_device - basis.meta.bounds[1]) / L
    
    n_modes = length(modes_device)
    n_points = length(coords_device)
    result = device_zeros(basis.meta.dtype, (n_points, n_modes), basis.meta)
    
    # Derivative factor
    factor = (2π / L)^order
    
    for (i, k) in enumerate(modes_device)
        if k == 0
            result[:, i] .= 0.0  # Derivative of constant is zero
        else
            # Alternating cos/sin for derivatives
            if order % 4 == 1
                result[:, i] .= -factor * k^order * sin.(k * normalized_coords)
            elseif order % 4 == 2
                result[:, i] .= -factor * k^order * cos.(k * normalized_coords)
            elseif order % 4 == 3
                result[:, i] .= factor * k^order * sin.(k * normalized_coords)
            else
                result[:, i] .= factor * k^order * cos.(k * normalized_coords)
            end
        end
    end
    
    return result
end

function to_device!(basis::Basis, )
    """Move basis to specified device."""
    basis.meta = device_config
    return basis
end

    """Get device configuration of basis."""
    return basis.meta
end

function synchronize_basis!(basis::Basis)
    """Synchronize basis operations on GPU."""
end

# Transform methods for basis types
# Note: For this Julia implementation, transforms are handled at the distributor level
# using PencilFFTs for parallel multi-dimensional transforms rather than 
# basis-specific FFTW transforms. This provides better scalability and follows
# the PencilArrays/PencilFFTs integration pattern specified in the requirements.
#
# The actual transform logic is implemented in:
# - src/core/transforms.jl (PencilFFTs integration)  
# - src/core/distributor.jl (layout management)
# - src/core/field.jl (field-level transform interface)
#
# This differs from Dedalus which uses basis-specific transforms, but provides
# better performance for the Julia/MPI parallel computing environment.
