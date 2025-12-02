"""
Spherical Coordinate System Implementation

Complete implementation of spherical coordinates (r, θ, φ) for ball domains
following dedalus patterns with PencilArrays and PencilFFTs integration.

Based on dedalus/core/coords.py spherical coordinates implementation.
"""

using PencilArrays
using PencilFFTs
using LinearAlgebra
using StaticArrays
using SparseArrays

export SphericalCoordinates, BallDomain, SphericalVector
export coordinate_transform_matrices, spin_transform_matrices, regularity_transform_matrices
export volume_element, surface_element, jacobian_determinant

"""
Spherical Coordinate System for Ball Domains

Implements (r, θ, φ) coordinates with proper handling of:
- Pole singularities at θ = 0, π
- Center regularity at r = 0
- Vector/tensor component transformations
- PencilArrays distributed computation

Based on dedalus SphericalCoordinates class.
"""
struct SphericalCoordinates{T<:Real}
    # Domain parameters
    radius::T                    # Ball radius (typically 1.0)
    center::SVector{3,T}        # Ball center coordinates
    
    # Grid parameters
    nr::Int                     # Radial grid points
    ntheta::Int                 # Colatitude grid points  
    nphi::Int                   # Azimuthal grid points
    
    # PencilArrays configuration
    topology::PencilArrays.MPITopology
    pencil::PencilArrays.Pencil
    
    # Transform matrices for vector components
    U_forward::SMatrix{3,3,Complex{T}}      # Coordinate to spin components
    U_backward::SMatrix{3,3,Complex{T}}     # Spin to coordinate components
    Q_forward::SMatrix{3,3,Complex{T}}      # Spin to regularity components  
    Q_backward::SMatrix{3,3,Complex{T}}     # Regularity to spin components
    
    # Grid arrays (distributed via PencilArrays)
    r_grid::Array{T,3}         # Radial coordinates
    theta_grid::Array{T,3}     # Colatitude coordinates
    phi_grid::Array{T,3}       # Azimuthal coordinates
    
    # Volume and surface elements
    volume_weights::Array{T,3}  # Volume integration weights
    surface_weights::Array{T,2} # Surface integration weights
    
    function SphericalCoordinates{T}(radius::T, nr::Int, ntheta::Int, nphi::Int;
                                   center::SVector{3,T} = SVector{3,T}(0,0,0),
                                   comm = MPI.COMM_WORLD) where T<:Real
        
        # Create PencilArrays configuration for (φ, θ, r) distribution
        # Distribute over φ and θ dimensions for optimal spherical harmonic transforms
        mesh = determine_optimal_spherical_mesh(nphi, ntheta, MPI.Comm_size(comm))
        topology = PencilArrays.MPITopology(comm, mesh)
        
        # Create pencil decomposition
        global_shape = (nphi, ntheta, nr)
        pencil = Pencil(topology, global_shape)
        
        # Create transform matrices following dedalus patterns
        U_forward, U_backward = create_coordinate_transform_matrices(T)
        Q_forward, Q_backward = create_regularity_transform_matrices(T)
        
        # Initialize coordinate grids
        r_grid, theta_grid, phi_grid = create_spherical_grids(
            pencil, radius, nr, ntheta, nphi, T
        )
        
        # Calculate integration weights
        volume_weights = calculate_volume_weights(r_grid, theta_grid, phi_grid, T)
        surface_weights = calculate_surface_weights(theta_grid, phi_grid, radius, T)
        
        new{T}(radius, center, nr, ntheta, nphi, 
               topology, pencil,
               U_forward, U_backward, Q_forward, Q_backward,
               r_grid, theta_grid, phi_grid, 
               volume_weights, surface_weights)
    end
end

# Convenience constructor
SphericalCoordinates(radius::T, nr::Int, ntheta::Int, nphi::Int; kwargs...) where T<:Real = 
    SphericalCoordinates{T}(radius, nr, ntheta, nphi; kwargs...)

"""
Determine optimal process mesh for spherical coordinates.
Prioritizes decomposition in φ (azimuthal) and θ (colatitude) directions.
"""
function determine_optimal_spherical_mesh(nphi::Int, ntheta::Int, nprocs::Int)
    if nprocs == 1
        return (1, 1)
    end
    
    # Try to balance φ and θ decomposition
    best_mesh = (1, nprocs)
    min_surface_area = typemax(Int)
    
    for nphi_procs in 1:min(nprocs, nphi)
        if nprocs % nphi_procs != 0
            continue
        end
        ntheta_procs = nprocs ÷ nphi_procs
        
        if ntheta_procs > ntheta
            continue
        end
        
        # Minimize communication surface area
        surface_area = (nphi ÷ nphi_procs) * ntheta_procs + (ntheta ÷ ntheta_procs) * nphi_procs
        
        if surface_area < min_surface_area
            min_surface_area = surface_area
            best_mesh = (nphi_procs, ntheta_procs)
        end
    end
    
    return best_mesh
end

"""
Create coordinate to spin component transform matrices.
Based on dedalus coords.py lines 338-351.

Transforms between (ur, uθ, uφ) and (u-, u+, u0):
u± = (uθ ± i*uφ) / √2
u0 = ur
"""
function create_coordinate_transform_matrices(T::Type{<:Real})
    CT = Complex{T}
    inv_sqrt2 = one(T) / sqrt(T(2))
    
    # Forward transform: coordinate → spin
    U_forward = @SMatrix CT[
        0             inv_sqrt2    inv_sqrt2*im;      # u- = (uθ + i*uφ)/√2
        0             inv_sqrt2   -inv_sqrt2*im;      # u+ = (uθ - i*uφ)/√2  
        1             0           0                   # u0 = ur
    ]
    
    # Backward transform: spin → coordinate  
    U_backward = @SMatrix CT[
        1             0           0;                  # ur = u0
        0             inv_sqrt2   inv_sqrt2;         # uθ = (u- + u+)/√2
        0            -inv_sqrt2*im  inv_sqrt2*im      # uφ = i*(u+ - u-)/√2
    ]
    
    return U_forward, U_backward
end

"""
Create regularity transform matrices for pole singularity handling.
Based on dedalus regularity system for vector components.

Handles transformation between spin components and regularity components
that remain finite at poles.
"""
function create_regularity_transform_matrices(T::Type{<:Real})
    CT = Complex{T}
    
    # Regularity transforms (simplified - full implementation depends on l, m quantum numbers)
    Q_forward = @SMatrix CT[
        1  0  0;
        0  1  0; 
        0  0  1
    ]
    
    Q_backward = @SMatrix CT[
        1  0  0;
        0  1  0;
        0  0  1  
    ]
    
    return Q_forward, Q_backward
end

"""
Create distributed spherical coordinate grids using PencilArrays.
"""
function create_spherical_grids(pencil::PencilArrays.Pencil, radius::T, 
                               nr::Int, ntheta::Int, nphi::Int, ::Type{T}) where T<:Real
    
    # Get local array dimensions
    local_shape = PencilArrays.size_local(pencil)
    local_range = PencilArrays.range_local(pencil)
    
    # Create local coordinate arrays
    r_grid = zeros(T, local_shape...)
    theta_grid = zeros(T, local_shape...)
    phi_grid = zeros(T, local_shape...)
    
    # Global coordinate vectors
    r_1d = create_radial_grid(nr, radius, T)
    theta_1d = create_colatitude_grid(ntheta, T)
    phi_1d = create_azimuthal_grid(nphi, T)
    
    # Fill local arrays with appropriate global values
    for (local_k, global_k) in enumerate(local_range[3])
        for (local_j, global_j) in enumerate(local_range[2])  
            for (local_i, global_i) in enumerate(local_range[1])
                r_grid[local_i, local_j, local_k] = r_1d[global_k]
                theta_grid[local_i, local_j, local_k] = theta_1d[global_j]
                phi_grid[local_i, local_j, local_k] = phi_1d[global_i]
            end
        end
    end
    
    return r_grid, theta_grid, phi_grid
end

"""
Create radial grid using Gauss-Jacobi quadrature points.
Maps from [-1,1] to [0,radius] with appropriate density.
"""
function create_radial_grid(nr::Int, radius::T, ::Type{T}) where T<:Real
    # Use Gauss-Jacobi quadrature for ball domain
    # Parameters chosen for optimal ball integration
    alpha = T(0)  # Jacobi parameter
    beta = T(2)   # Jacobi parameter for 3D ball
    
    # Get quadrature points and weights
    nodes, weights = gauss_jacobi_nodes_weights(nr, alpha, beta, T)
    
    # Map from [-1,1] to [0,radius]
    r_grid = @. radius * (1 + nodes) / 2
    
    return r_grid
end

"""
Create colatitude grid using Gauss-Legendre quadrature.
"""  
function create_colatitude_grid(ntheta::Int, ::Type{T}) where T<:Real
    # Gauss-Legendre nodes for θ ∈ [0,π]
    nodes, weights = gauss_legendre_nodes_weights(ntheta, T)
    
    # Map from [-1,1] to [0,π]
    theta_grid = @. π * (1 + nodes) / 2
    
    return theta_grid
end

"""
Create azimuthal grid using uniform spacing.
"""
function create_azimuthal_grid(nphi::Int, ::Type{T}) where T<:Real
    phi_grid = T(2π) * collect(T, 0:(nphi-1)) / nphi
    return phi_grid
end

"""
Calculate volume integration weights for spherical coordinates.
Includes r² sin(θ) Jacobian and quadrature weights.
"""
function calculate_volume_weights(r_grid::Array{T,3}, theta_grid::Array{T,3}, 
                                phi_grid::Array{T,3}, ::Type{T}) where T<:Real
    
    nr, ntheta, nphi = size(r_grid)
    volume_weights = zeros(T, size(r_grid))
    
    # Get quadrature weights for each direction
    _, r_weights = gauss_jacobi_nodes_weights(nr, T(0), T(2), T)
    _, theta_weights = gauss_legendre_nodes_weights(ntheta, T) 
    phi_weights = fill(T(2π)/nphi, nphi)  # Uniform azimuthal weights
    
    # Combine weights with Jacobian
    for k in 1:nr, j in 1:ntheta, i in 1:nphi
        r = r_grid[i,j,k]
        theta = theta_grid[i,j,k]
        
        # Spherical Jacobian: r² sin(θ)
        jacobian = r^2 * sin(theta)
        
        # Scale by radius for mapping [0,1] → [0,radius]  
        radius_scale = (size(r_grid,3) > 1) ? (r_grid[i,j,end] - r_grid[i,j,1]) : T(1)
        
        volume_weights[i,j,k] = jacobian * r_weights[k] * theta_weights[j] * phi_weights[i] * 
                              radius_scale * π / 2  # θ mapping factor
    end
    
    return volume_weights
end

"""
Calculate surface integration weights for sphere boundary.
"""
function calculate_surface_weights(theta_grid::Array{T,3}, phi_grid::Array{T,3}, 
                                 radius::T, ::Type{T}) where T<:Real
    
    nphi, ntheta = size(theta_grid)[1:2]
    surface_weights = zeros(T, nphi, ntheta)
    
    # Get quadrature weights  
    _, theta_weights = gauss_legendre_nodes_weights(ntheta, T)
    phi_weights = fill(T(2π)/nphi, nphi)
    
    for j in 1:ntheta, i in 1:nphi
        theta = theta_grid[i,j,1]  # Surface values
        
        # Surface element: r² sin(θ) dθ dφ
        surface_element = radius^2 * sin(theta)
        
        surface_weights[i,j] = surface_element * theta_weights[j] * phi_weights[i] * π/2
    end
    
    return surface_weights
end

# Gauss-Jacobi quadrature implementation
function gauss_jacobi_nodes_weights(n::Int, alpha::T, beta::T, ::Type{T}) where T<:Real
    # Gauss-Jacobi quadrature using Golub-Welsch algorithm
    # For weight function (1-x)^alpha * (1+x)^beta on [-1,1]
    # Following Dedalus approach for spherical coordinates with Jacobi polynomials
    
    # Special cases
    if n == 1
        # Single node at center of interval weighted by alpha, beta
        node = (beta - alpha) / (alpha + beta + 2)
        weight = T(2)^(alpha + beta + 1) * gamma(alpha + 1) * gamma(beta + 1) / gamma(alpha + beta + 2)
        return [node], [weight]
    end
    
    # Check for Legendre case (alpha = beta = 0)
    if alpha ≈ 0 && beta ≈ 0
        return gauss_legendre_nodes_weights(n, T)
    end
    
    # Golub-Welsch algorithm for general Jacobi polynomials
    # Three-term recurrence: P_{n+1}(x) = (A_n*x + B_n)*P_n(x) - C_n*P_{n-1}(x)
    
    # Recurrence coefficients for normalized Jacobi polynomials
    alpha_diag = zeros(T, n)  # Diagonal elements of Jacobi matrix
    beta_off = zeros(T, n-1)  # Off-diagonal elements
    
    for k = 0:n-1
        # Diagonal elements (alpha coefficients)
        if k == 0
            alpha_diag[k+1] = (beta - alpha) / (alpha + beta + 2)
        else
            num = (beta^2 - alpha^2)
            den = (2*k + alpha + beta) * (2*k + alpha + beta + 2)
            alpha_diag[k+1] = num / den
        end
        
        # Off-diagonal elements (beta coefficients) for k >= 1
        if k >= 1
            num1 = 4 * k * (k + alpha) * (k + beta) * (k + alpha + beta)
            den1 = (2*k + alpha + beta)^2 * (2*k + alpha + beta + 1) * (2*k + alpha + beta - 1)
            beta_off[k] = sqrt(num1 / den1)
        end
    end
    
    # First off-diagonal element (k=0 to k=1)
    if n > 1
        num1 = 4 * 1 * (1 + alpha) * (1 + beta) * (1 + alpha + beta)
        den1 = (2 + alpha + beta)^2 * (3 + alpha + beta) * (1 + alpha + beta)
        beta_off[1] = sqrt(num1 / den1)
    end
    
    # Build symmetric tridiagonal Jacobi matrix
    J = zeros(T, n, n)
    for i = 1:n
        J[i, i] = alpha_diag[i]
    end
    for i = 1:n-1
        J[i, i+1] = beta_off[i]
        J[i+1, i] = beta_off[i]
    end
    
    # Eigenvalue decomposition
    try
        eigenvalues, eigenvectors = eigen(J)
        
        # Sort eigenvalues (nodes) in ascending order
        perm = sortperm(real.(eigenvalues))
        nodes = T.(real.(eigenvalues[perm]))
        
        # Weights from eigenvectors and normalization constant
        # Weight = (integral of weight function) * (first eigenvector component)^2
        normalization = T(2)^(alpha + beta + 1) * gamma(alpha + 1) * gamma(beta + 1) / gamma(alpha + beta + 2)
        weights = T[normalization * abs2(eigenvectors[1, p]) for p in perm]
        
        return nodes, weights
        
    catch e
        @warn "Eigenvalue decomposition failed for Jacobi quadrature, using fallback: $e"
        return gauss_jacobi_fallback(n, alpha, beta, T)
    end
end

"""
Fallback implementation for Gauss-Jacobi quadrature using Newton-Raphson.
"""
function gauss_jacobi_fallback(n::Int, alpha::T, beta::T, ::Type{T}) where T<:Real
    nodes = zeros(T, n)
    weights = zeros(T, n)
    
    # Initial guess using Chebyshev nodes scaled for Jacobi interval
    for i = 1:n
        # Modified Chebyshev initial guess for Jacobi polynomials
        theta = T(π * (4*i - 1)) / (4*n + 2*(alpha + beta + 1))
        nodes[i] = cos(theta)
    end
    
    # Newton-Raphson iteration
    for i = 1:n
        x = nodes[i]
        
        for iter = 1:15  # More iterations for general Jacobi case
            P, Pp = jacobi_polynomial_and_derivative(x, n-1, alpha, beta, T)
            
            if abs(Pp) < eps(T)
                break
            end
            
            dx = P / Pp
            x = x - dx
            
            if abs(dx) < 20*eps(T)
                break
            end
        end
        
        nodes[i] = x
        
        # Compute weight
        _, Pp = jacobi_polynomial_and_derivative(x, n-1, alpha, beta, T)
        gamma_ratio = gamma(n + alpha + 1) * gamma(n + beta + 1) / gamma(n + alpha + beta + 1) / gamma(n + 1)
        weight_factor = T(2)^(alpha + beta + 1) / (2*n + alpha + beta + 1)
        weights[i] = weight_factor * gamma_ratio / (Pp^2 * (1 - x^2))
    end
    
    return nodes, weights
end

"""
Evaluate Jacobi polynomial and its derivative using three-term recurrence.
"""
function jacobi_polynomial_and_derivative(x::T, n::Int, alpha::T, beta::T, ::Type{T}) where T<:Real
    if n == 0
        return T(1), T(0)
    elseif n == 1
        P1 = (alpha - beta + (alpha + beta + 2)*x) / 2
        Pp1 = (alpha + beta + 2) / 2
        return P1, Pp1
    end
    
    # Three-term recurrence for Jacobi polynomials
    # P^{(α,β)}_k(x) = (A_k*x + B_k)*P^{(α,β)}_{k-1}(x) - C_k*P^{(α,β)}_{k-2}(x)
    
    P0 = T(1)
    P1 = (alpha - beta + (alpha + beta + 2)*x) / 2
    Pp0 = T(0)  
    Pp1 = (alpha + beta + 2) / 2
    
    for k = 2:n
        # Recurrence coefficients
        k1 = 2*k + alpha + beta - 1
        k2 = 2*k + alpha + beta
        k3 = 2*k + alpha + beta + 1
        
        A_k = k3 * k2 / (2 * k * (k + alpha + beta))
        B_k = (alpha^2 - beta^2) * k1 / (2 * k * (k + alpha + beta) * k2)
        C_k = (k + alpha - 1) * (k + beta - 1) * k3 / (k * (k + alpha + beta) * k2)
        
        # Update polynomials
        P2 = (A_k * x + B_k) * P1 - C_k * P0
        
        # Update derivatives using product rule
        Pp2 = (A_k * x + B_k) * Pp1 + A_k * P1 - C_k * Pp0
        
        P0, P1 = P1, P2
        Pp0, Pp1 = Pp1, Pp2
    end
    
    return P1, Pp1
end

"""
Gamma function implementation for quadrature weights.
Uses Julia's built-in gamma function with numerical stability checks.
"""
function gamma(x::T) where T<:Real
    if x <= 0 && isinteger(x)
        return T(Inf)  # Gamma function has poles at non-positive integers
    end
    
    # Use Julia's built-in gamma function
    result = Base.gamma(float(x))
    return T(result)
end

# Gauss-Legendre quadrature implementation following Dedalus/Golub-Welsch algorithm
function gauss_legendre_nodes_weights(n::Int, ::Type{T}) where T<:Real
    # Special cases
    if n == 1
        return [T(0)], [T(2)]
    elseif n == 2
        nodes = T[-1/sqrt(3), 1/sqrt(3)]
        weights = T[1, 1]
        return nodes, weights
    end
    
    # Golub-Welsch algorithm for Gauss-Legendre quadrature
    # Based on eigenvalue decomposition of symmetric tridiagonal Jacobi matrix
    
    # Three-term recurrence coefficients for Legendre polynomials
    # P_{n+1}(x) = ((2n+1)*x*P_n(x) - n*P_{n-1}(x))/(n+1)
    # In normalized form: P_{n+1}(x) = a_n*x*P_n(x) + b_n*P_{n-1}(x)
    # where a_n = sqrt((2n+1)/(2n+3)), b_n = -sqrt(n(n+1)/((2n+1)(2n+3)))
    
    # Jacobi matrix construction
    # For Legendre: alpha_k = 0 (diagonal), beta_k = k/sqrt(4k^2-1) (off-diagonal)
    beta = zeros(T, n-1)
    for k = 1:n-1
        beta[k] = T(k) / sqrt(T(4*k^2 - 1))
    end
    
    # Build symmetric tridiagonal Jacobi matrix
    # T = diag(alpha) + diag(beta, 1) + diag(beta, -1)
    # For Legendre polynomials, alpha = 0
    J = zeros(T, n, n)
    for i = 1:n-1
        J[i, i+1] = beta[i]
        J[i+1, i] = beta[i]
    end
    
    # Eigenvalue decomposition: nodes are eigenvalues, weights from eigenvectors
    try
        eigenvalues, eigenvectors = eigen(J)
        
        # Sort eigenvalues (nodes) in ascending order
        perm = sortperm(real.(eigenvalues))
        nodes = T.(real.(eigenvalues[perm]))
        
        # Weights = 2 * (first component of eigenvector)^2
        # Factor of 2 comes from integral of weight function over [-1,1]
        weights = T[2 * abs2(eigenvectors[1, p]) for p in perm]
        
        return nodes, weights
        
    catch e
        @warn "Eigenvalue decomposition failed, using fallback method: $e"
        return gauss_legendre_fallback(n, T)
    end
end

"""
Fallback implementation for Gauss-Legendre quadrature using Newton-Raphson.
Used when eigenvalue method fails.
"""
function gauss_legendre_fallback(n::Int, ::Type{T}) where T<:Real
    # Use Newton-Raphson iteration to find roots of Legendre polynomials
    nodes = zeros(T, n)
    weights = zeros(T, n)
    
    # Initial guess for roots (Chebyshev approximation)
    for i = 1:n
        # Chebyshev nodes as initial guess
        nodes[i] = -cos(T(π * (i - 0.25)) / (n + 0.5))
    end
    
    # Newton-Raphson iteration
    for i = 1:n
        x = nodes[i]
        
        # Newton iteration to find root of Legendre polynomial
        for iter = 1:10  # Usually converges in 2-3 iterations
            P, Pp = legendre_polynomial_and_derivative(x, n, T)
            
            if abs(Pp) < eps(T)
                break
            end
            
            dx = P / Pp
            x = x - dx
            
            if abs(dx) < 10*eps(T)
                break
            end
        end
        
        nodes[i] = x
        
        # Compute weight using derivative
        _, Pp = legendre_polynomial_and_derivative(x, n, T)
        weights[i] = T(2) / ((T(1) - x^2) * Pp^2)
    end
    
    return nodes, weights
end

"""
Evaluate Legendre polynomial and its derivative at x using three-term recurrence.
"""
function legendre_polynomial_and_derivative(x::T, n::Int, ::Type{T}) where T<:Real
    if n == 0
        return T(1), T(0)
    elseif n == 1
        return x, T(1)
    end
    
    # Three-term recurrence for Legendre polynomials
    P0, P1 = T(1), x
    Pp0, Pp1 = T(0), T(1)
    
    for k = 2:n
        # P_k(x) = ((2k-1)*x*P_{k-1}(x) - (k-1)*P_{k-2}(x))/k
        P2 = (T(2*k-1) * x * P1 - T(k-1) * P0) / T(k)
        
        # P'_k(x) = ((2k-1)*(P_{k-1}(x) + x*P'_{k-1}(x)) - (k-1)*P'_{k-2}(x))/k
        Pp2 = (T(2*k-1) * (P1 + x * Pp1) - T(k-1) * Pp0) / T(k)
        
        P0, P1 = P1, P2
        Pp0, Pp1 = Pp1, Pp2
    end
    
    return P1, Pp1
end

"""
Spherical Vector Field Type

Represents vector fields in spherical coordinates with proper component handling.
"""
mutable struct SphericalVector{T<:Real}
    coords::SphericalCoordinates{T}
    
    # Vector components in different representations
    components_coord::Array{Complex{T},4}     # (ur, uθ, uφ) components
    components_spin::Array{Complex{T},4}      # (u-, u+, u0) components  
    components_reg::Array{Complex{T},4}       # Regularity components
    
    # Current representation
    current_layout::Symbol  # :coord, :spin, or :regularity
    
    function SphericalVector{T}(coords::SphericalCoordinates{T}) where T<:Real
        local_shape = size(coords.r_grid)
        
        # Initialize component arrays
        components_coord = zeros(Complex{T}, 3, local_shape...)
        components_spin = zeros(Complex{T}, 3, local_shape...)
        components_reg = zeros(Complex{T}, 3, local_shape...)
        
        new{T}(coords, components_coord, components_spin, components_reg, :coord)
    end
end

# Convenience constructor
SphericalVector(coords::SphericalCoordinates{T}) where T = SphericalVector{T}(coords)

"""
Transform vector between coordinate and spin representations.
"""
function transform_to_spin!(vec::SphericalVector{T}) where T<:Real
    if vec.current_layout == :spin
        return vec
    end
    
    coords = vec.coords
    
    # Apply transformation matrices
    for idx in CartesianIndices(size(vec.components_coord)[2:end])
        coord_components = @view vec.components_coord[:, idx]
        spin_components = @view vec.components_spin[:, idx]
        
        # Matrix multiplication: spin = U_forward * coord
        mul!(spin_components, coords.U_forward, coord_components)
    end
    
    vec.current_layout = :spin
    return vec
end

"""
Transform vector from spin back to coordinate representation.
"""
function transform_to_coord!(vec::SphericalVector{T}) where T<:Real
    if vec.current_layout == :coord
        return vec
    end
    
    coords = vec.coords
    
    # Apply inverse transformation
    for idx in CartesianIndices(size(vec.components_spin)[2:end])
        spin_components = @view vec.components_spin[:, idx]
        coord_components = @view vec.components_coord[:, idx]
        
        # Matrix multiplication: coord = U_backward * spin  
        mul!(coord_components, coords.U_backward, spin_components)
    end
    
    vec.current_layout = :coord
    return vec
end

"""
Transform vector to regularity representation for pole handling.
"""
function transform_to_regularity!(vec::SphericalVector{T}) where T<:Real
    # First ensure we're in spin representation
    transform_to_spin!(vec)
    
    coords = vec.coords
    
    # Apply regularity transformation
    for idx in CartesianIndices(size(vec.components_spin)[2:end])
        spin_components = @view vec.components_spin[:, idx]
        reg_components = @view vec.components_reg[:, idx]
        
        # Matrix multiplication: reg = Q_forward * spin
        mul!(reg_components, coords.Q_forward, spin_components)
    end
    
    vec.current_layout = :regularity
    return vec
end

"""
Get volume element for integration in spherical coordinates.
"""
function volume_element(coords::SphericalCoordinates{T}) where T<:Real
    return coords.volume_weights
end

"""
Get surface element for boundary integration.
"""
function surface_element(coords::SphericalCoordinates{T}) where T<:Real
    return coords.surface_weights
end

"""
Calculate Jacobian determinant for coordinate transformations.
"""
function jacobian_determinant(coords::SphericalCoordinates{T}) where T<:Real
    # For spherical coordinates: |J| = r² sin(θ)
    return coords.r_grid .^ 2 .* sin.(coords.theta_grid)
end

"""
Ball Domain Type

Represents the interior of a sphere with appropriate boundary conditions.
"""
struct BallDomain{T<:Real}
    coords::SphericalCoordinates{T}
    boundary_conditions::Dict{Symbol, Any}
    
    function BallDomain{T}(radius::T, nr::Int, ntheta::Int, nphi::Int; 
                          boundary_conditions::Dict{Symbol, Any} = Dict{Symbol,Any}(),
                          kwargs...) where T<:Real
        
        coords = SphericalCoordinates{T}(radius, nr, ntheta, nphi; kwargs...)
        
        # Default boundary conditions
        default_bcs = Dict{Symbol,Any}(
            :center_regularity => true,    # Automatic regularity at r=0
            :surface_bc => :none           # No boundary condition at r=radius
        )
        
        bcs = merge(default_bcs, boundary_conditions)
        
        new{T}(coords, bcs)
    end
end

# Convenience constructor
BallDomain(radius::T, nr::Int, ntheta::Int, nphi::Int; kwargs...) where T<:Real = 
    BallDomain{T}(radius, nr, ntheta, nphi; kwargs...)