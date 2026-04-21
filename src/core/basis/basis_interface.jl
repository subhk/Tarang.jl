# Basis interface methods, local grids, and direct basis evaluation helpers.

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

"""
    Local grid for a basis.

    GPU-aware: By default, the grid is moved to the distributor's architecture.
    This enables efficient broadcasting with field data on GPU.
    Set `move_to_arch=false` to always return CPU arrays (e.g., for file I/O).
    """
function local_grid(basis::Basis, dist, scale; move_to_arch::Bool=true)
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

"""No-op for CPU-only mode."""
function synchronize_basis!(basis::Basis)
end
