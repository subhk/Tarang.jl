"""
Layout Transformations for Spherical Fields

Handles transformations between different data layouts:
- GRID_LAYOUT: Physical space (φ, θ, r) coordinates
- SPECTRAL_LAYOUT: Spectral space (m, l, n) coefficients
- MIXED_PHI: FFT in φ, grid in θ,r  
- MIXED_THETA: FFT in φ, SHT in θ, grid in r

Based on dedalus/core/transforms.py layout transformation system.
"""

"""
Transform field between different layouts.
"""
function transform_layout!(field::SphericalScalarField{T}, from::SphericalLayout, 
                          to::SphericalLayout) where T<:Real
    if from == to
        return field
    end
    
    # Direct transformations
    if from == GRID_LAYOUT && to == SPECTRAL_LAYOUT
        grid_to_spectral!(field)
    elseif from == SPECTRAL_LAYOUT && to == GRID_LAYOUT
        spectral_to_grid!(field)
    else
        # Multi-step transformations through intermediate layouts
        if from == GRID_LAYOUT
            grid_to_mixed_phi!(field)
            from = MIXED_PHI
        end
        
        if from == MIXED_PHI && to == SPECTRAL_LAYOUT
            mixed_phi_to_spectral!(field)
        elseif from == MIXED_PHI && to == GRID_LAYOUT
            mixed_phi_to_grid!(field)
        end
    end
    
    field.layout_manager.current_layout[] = to
    return field
end

function transform_layout!(field::SphericalVectorField{T}, from::SphericalLayout, 
                          to::SphericalLayout) where T<:Real
    if from == to
        return field
    end
    
    # Transform each component
    for comp in 1:3
        component_data_grid = view(field.data_grid, comp, :, :, :)
        component_data_spectral = view(field.data_spectral, comp, :, :, :)
        
        # Create temporary scalar field for transformation
        temp_field = SphericalScalarField{T}("temp", field.domain)
        temp_field.data_grid .= component_data_grid
        temp_field.data_spectral .= component_data_spectral
        temp_field.layout_manager.current_layout[] = from
        
        # Transform
        transform_layout!(temp_field, from, to)
        
        # Copy back
        component_data_grid .= temp_field.data_grid
        component_data_spectral .= temp_field.data_spectral
    end
    
    field.layout_manager.current_layout[] = to
    return field
end

"""
Apply FFT transform in phi direction.
"""
function apply_phi_transform!(field::SphericalScalarField{T}, forward::Bool) where T<:Real
    coords = field.domain.coords
    
    if forward
        # Forward FFT: grid → Fourier modes in φ
        for idx_r in 1:coords.nr, idx_theta in 1:coords.ntheta
            phi_slice = view(field.data_grid, :, idx_theta, idx_r)
            fft!(phi_slice)
        end
    else
        # Inverse FFT: Fourier modes → grid in φ
        for idx_r in 1:coords.nr, idx_theta in 1:coords.ntheta
            phi_slice = view(field.data_grid, :, idx_theta, idx_r)
            ifft!(phi_slice)
        end
    end
end

"""
Grid to spectral transformation using spherical harmonic transform.
"""
function grid_to_spectral!(field::SphericalScalarField{T}) where T<:Real
    # Apply 3D transform: FFT in φ, SHT in θ, Zernike transform in r
    
    # Step 1: FFT in φ direction
    apply_phi_transform!(field, true)
    
    # Step 2: Spherical harmonic transform in θ direction
    apply_theta_transform!(field, true)
    
    # Step 3: Radial transform using Zernike polynomials
    apply_radial_transform!(field, true)
    
    # Copy to spectral data storage
    field.data_spectral .= field.data_grid
end

"""
Spectral to grid transformation.
"""
function spectral_to_grid!(field::SphericalScalarField{T}) where T<:Real
    # Copy from spectral storage
    field.data_grid .= field.data_spectral
    
    # Apply inverse transforms: Zernike → SHT^(-1) → IFFT
    
    # Step 1: Inverse radial transform
    apply_radial_transform!(field, false)
    
    # Step 2: Inverse spherical harmonic transform in θ
    apply_theta_transform!(field, false)
    
    # Step 3: Inverse FFT in φ
    apply_phi_transform!(field, false)
end

"""
Apply spherical harmonic transform in theta direction.
"""
function apply_theta_transform!(field::SphericalScalarField{T}, forward::Bool) where T<:Real
    coords = field.domain.coords
    l_max = field.layout_manager.ball_basis.l_max
    
    # Create transform matrices if not cached
    if !haskey(field.layout_manager.transform_buffers, "theta_forward")
        forward_matrix, backward_matrix = create_theta_transform_matrices(coords, l_max, T)
        field.layout_manager.transform_buffers["theta_forward"] = forward_matrix
        field.layout_manager.transform_buffers["theta_backward"] = backward_matrix
    end
    
    forward_matrix = field.layout_manager.transform_buffers["theta_forward"]
    backward_matrix = field.layout_manager.transform_buffers["theta_backward"]
    
    # Apply transform
    for idx_r in 1:coords.nr, idx_phi in 1:coords.nphi
        theta_slice = view(field.data_grid, idx_phi, :, idx_r)
        
        if forward
            theta_slice .= forward_matrix * theta_slice
        else
            theta_slice .= backward_matrix * theta_slice
        end
    end
end

"""
Apply radial transform using Zernike polynomials.
Based on dedalus/core/ball_wrapper.py and zernike.py implementation.
"""
function apply_radial_transform!(field::SphericalScalarField{T}, forward::Bool) where T<:Real
    coords = field.domain.coords
    ball_basis = field.layout_manager.ball_basis
    nr = coords.nr
    
    # Get or create transform matrices
    transform_key = forward ? "radial_forward" : "radial_backward"
    
    if !haskey(field.layout_manager.transform_buffers, transform_key)
        create_radial_transform_matrices!(field.layout_manager, ball_basis, T)
    end
    
    # For each (l,m) mode, apply radial transform
    for l in 0:ball_basis.l_max, m in -l:l
        m_idx = m + ball_basis.l_max + 1
        l_idx = l + 1
        
        radial_slice = view(field.data_grid, m_idx, l_idx, :)
        
        if forward
            # Grid → spectral coefficients using pushW matrix
            pushW_key = "pushW_l$(l)"
            if haskey(field.layout_manager.transform_buffers, pushW_key)
                pushW = field.layout_manager.transform_buffers[pushW_key]
                # Forward transform: c_n = ∑_i W_{ni} f(r_i)
                temp_coeffs = pushW * radial_slice
                radial_slice .= temp_coeffs
            end
        else
            # Spectral coefficients → grid using pullW matrix
            pullW_key = "pullW_l$(l)"
            if haskey(field.layout_manager.transform_buffers, pullW_key)
                pullW = field.layout_manager.transform_buffers[pullW_key]
                # Backward transform: f(r_i) = ∑_n W_{in} c_n
                temp_grid = pullW * radial_slice
                radial_slice .= temp_grid
            end
        end
    end
end

"""
Create theta direction transform matrices.
Based on Dedalus SWSHColatitudeTransform implementation.
"""
function create_theta_transform_matrices(coords::SphericalCoordinateSystem{T}, l_max::Int, ::Type{T}) where T<:Real
    ntheta = coords.ntheta
    
    # Get Gauss-Legendre quadrature points for theta direction
    cos_theta_grid, theta_weights = compute_theta_quadrature_points(ntheta, T)
    
    # Create forward and backward transform matrices for each m mode
    forward_matrices = Dict{Int, Matrix{T}}()
    backward_matrices = Dict{Int, Matrix{T}}()
    
    # For each azimuthal wavenumber m
    for m in -l_max:l_max
        forward_matrices[m], backward_matrices[m] = create_spherical_harmonic_matrices(
            l_max, m, cos_theta_grid, theta_weights, T)
    end
    
    # Return combined matrices (simplified approach - use m=0 as default)
    forward_matrix = forward_matrices[0]
    backward_matrix = backward_matrices[0]
    
    return forward_matrix, backward_matrix
end

"""
Compute theta direction quadrature points and weights.
Based on Dedalus sphere.quadrature implementation.
"""
function compute_theta_quadrature_points(ntheta::Int, ::Type{T}) where T<:Real
    # Use Gauss-Legendre quadrature for θ ∈ [0,π]
    # This maps to Gauss-Legendre on x ∈ [-1,1] with cos(θ) = x
    
    # For Legendre polynomials: α = β = 0
    cos_theta_nodes, weights = compute_gauss_jacobi_quadrature(ntheta, zero(T), zero(T), T)
    
    # Ensure proper ordering: cos(θ) from 1 to -1 (θ from 0 to π)
    cos_theta_grid = -reverse(cos_theta_nodes)  # Flip and negate
    theta_weights = reverse(weights)
    
    return cos_theta_grid, theta_weights
end

"""
Create spherical harmonic transform matrices for given m.
Based on Dedalus _forward_SWSH_matrices and _backward_SWSH_matrices.
"""
function create_spherical_harmonic_matrices(l_max::Int, m::Int, cos_theta_grid::Vector{T}, 
                                          theta_weights::Vector{T}, ::Type{T}) where T<:Real
    ntheta = length(cos_theta_grid)
    
    # Calculate range of valid l values for this m
    l_min = abs(m)
    n_modes = l_max - l_min + 1
    
    # Forward transform matrix: spectral coefficients ← grid values
    # Forward[l-l_min, θ_idx] = weight[θ_idx] * Y_l^m(θ_idx)
    forward_matrix = zeros(T, n_modes, ntheta)
    
    # Backward transform matrix: grid values ← spectral coefficients
    # Backward[θ_idx, l-l_min] = Y_l^m(θ_idx)
    backward_matrix = zeros(T, ntheta, n_modes)
    
    # Evaluate spherical harmonics for all valid l values
    for (i, l) in enumerate(l_min:l_max)
        for (j, cos_theta) in enumerate(cos_theta_grid)
            # Evaluate spherical harmonic Y_l^m(θ) using Associated Legendre polynomials
            # Y_l^m(θ,φ) = sqrt((2l+1)/(4π) * (l-|m|)!/(l+|m|)!) * P_l^|m|(cos θ) * e^(imφ)
            # For theta transform, we only need the θ part: P_l^|m|(cos θ)
            
            harmonic_value = evaluate_associated_legendre(l, abs(m), cos_theta, T)
            
            # Apply normalization factor (without azimuthal part)
            norm_factor = sqrt((2*l + 1) / (4*π) * factorial(l - abs(m)) / factorial(l + abs(m)))
            harmonic_value *= norm_factor
            
            # Handle negative m values with (-1)^m factor
            if m < 0 && abs(m) % 2 == 1
                harmonic_value *= -1
            end
            
            # Fill matrices
            forward_matrix[i, j] = theta_weights[j] * harmonic_value  # With integration weight
            backward_matrix[j, i] = harmonic_value                    # Without weight for evaluation
        end
    end
    
    return forward_matrix, backward_matrix
end

"""
Evaluate Associated Legendre polynomial P_l^m(x).
Based on three-term recurrence relation for numerical stability.
"""
function evaluate_associated_legendre(l::Int, m::Int, x::T, ::Type{T}) where T<:Real
    if m > l
        return zero(T)
    end
    
    if m == 0
        # Regular Legendre polynomial
        return evaluate_legendre_polynomial(l, x, T)
    end
    
    # For m > 0, use the recurrence relation
    # Starting values for P_m^m(x)
    if l == m
        # P_m^m(x) = (-1)^m * (2m-1)!! * (1-x²)^(m/2)
        fact_m = double_factorial(2*m - 1)
        return (-1)^m * fact_m * (1 - x^2)^(m/2)
    end
    
    # P_{m+1}^m(x) = x * (2m+1) * P_m^m(x)
    if l == m + 1
        pmm = (-1)^m * double_factorial(2*m - 1) * (1 - x^2)^(m/2)
        return x * (2*m + 1) * pmm
    end
    
    # Use upward recurrence for l > m+1
    # (l-m) * P_l^m(x) = x * (2l-1) * P_{l-1}^m(x) - (l+m-1) * P_{l-2}^m(x)
    pmm = (-1)^m * double_factorial(2*m - 1) * (1 - x^2)^(m/2)
    pmm1 = x * (2*m + 1) * pmm
    
    for k in (m+2):l
        pmm2 = (x * (2*k - 1) * pmm1 - (k + m - 1) * pmm) / (k - m)
        pmm, pmm1 = pmm1, pmm2
    end
    
    return pmm1
end

"""
Evaluate regular Legendre polynomial P_l(x).
"""
function evaluate_legendre_polynomial(l::Int, x::T, ::Type{T}) where T<:Real
    if l == 0
        return one(T)
    elseif l == 1
        return x
    end
    
    # Bonnet's recurrence relation: (l+1) * P_{l+1}(x) = (2l+1) * x * P_l(x) - l * P_{l-1}(x)
    p0, p1 = one(T), x
    
    for k in 2:l
        p2 = ((2*k - 1) * x * p1 - (k - 1) * p0) / k
        p0, p1 = p1, p2
    end
    
    return p1
end

"""
Compute double factorial (2n-1)!! = 1*3*5*...*(2n-1).
"""
function double_factorial(n::Int)
    if n <= 0
        return 1
    end
    result = 1
    for i in 1:2:n
        result *= i
    end
    return result
end

"""
Create radial transform matrices for all l modes.
Based on Dedalus ball_wrapper.py implementation.
"""
function create_radial_transform_matrices!(layout_manager, ball_basis::BallBasis{T}, ::Type{T}) where T<:Real
    nr = ball_basis.nr
    n_max = ball_basis.n_max
    
    # Get radial grid points (Gauss-Jacobi quadrature points)
    r_grid, weights = compute_gauss_jacobi_quadrature(nr, 0, 0, T)  # (α,β) = (0,0) for ball
    
    # Create transform matrices for each l mode
    for l in 0:ball_basis.l_max
        # Forward transform matrix (pushW): grid → spectral
        # pushW[n,i] = weight[i] * Z_n^(l)(r[i]) for integration
        pushW = zeros(T, n_max + 1, nr)
        
        # Backward transform matrix (pullW): spectral → grid  
        # pullW[i,n] = Z_n^(l)(r[i]) for evaluation
        pullW = zeros(T, nr, n_max + 1)
        
        for i in 1:nr
            r = r_grid[i]
            w = weights[i]
            
            for n in 0:n_max
                # Evaluate Zernike polynomial Z_n^(l)(r)
                z_val = evaluate_zernike_polynomial(n, l, r, T)
                
                # Forward transform with quadrature weight
                pushW[n + 1, i] = w * z_val
                
                # Backward transform (evaluation)
                pullW[i, n + 1] = z_val
            end
        end
        
        # Store matrices
        layout_manager.transform_buffers["pushW_l$(l)"] = pushW
        layout_manager.transform_buffers["pullW_l$(l)"] = pullW
    end
    
    # Set completion flags
    layout_manager.transform_buffers["radial_forward"] = true
    layout_manager.transform_buffers["radial_backward"] = true
end

"""
Compute Gauss-Jacobi quadrature points and weights.
Based on Dedalus jacobi.py implementation.
"""
function compute_gauss_jacobi_quadrature(n::Int, α::T, β::T, ::Type{T}) where T<:Real
    if n == 0
        return T[], T[]
    elseif n == 1
        r = (β - α) / (α + β + 2)
        w = 2^(α + β + 1) * beta(α + 1, β + 1)
        return [r], [w]
    end
    
    # Build tridiagonal matrix for eigenvalue problem
    # Following Golub-Welsch algorithm
    diagonal = zeros(T, n)
    off_diagonal = zeros(T, n-1)
    
    for i in 1:n
        k = i - 1  # 0-based indexing for formulas
        diagonal[i] = (β^2 - α^2) / ((2*k + α + β) * (2*k + α + β + 2))
    end
    
    for i in 1:(n-1)
        k = i - 1  # 0-based indexing
        numerator = 4 * (k + 1) * (k + α + 1) * (k + β + 1) * (k + α + β + 1)
        denominator = (2*k + α + β + 1) * (2*k + α + β + 2)^2 * (2*k + α + β + 3)
        off_diagonal[i] = sqrt(numerator / denominator)
    end
    
    # Solve eigenvalue problem
    eigenvalues, eigenvectors = eigen(SymTridiagonal(diagonal, off_diagonal))
    
    # Compute weights from first component of eigenvectors
    μ₀ = 2^(α + β + 1) * beta(α + 1, β + 1)  # ∫₋₁¹ (1-x)^α (1+x)^β dx
    weights = μ₀ * eigenvectors[1, :].^2
    
    return eigenvalues, weights
end

"""
Evaluate Zernike polynomial Z_n^(l)(r).
Based on Dedalus zernike.py implementation.
"""
function evaluate_zernike_polynomial(n::Int, l::Int, r::T, ::Type{T}) where T<:Real
    if n < l || (n - l) % 2 != 0
        return zero(T)
    end
    
    # Zernike polynomial Z_n^(l)(r) = r^l * P_{(n-l)/2}^{(l,0)}(2r² - 1)
    # where P_k^{(α,β)} is the Jacobi polynomial
    
    if r ≈ zero(T)
        return l == 0 && n == 0 ? one(T) : zero(T)
    end
    
    k = (n - l) ÷ 2
    ξ = 2 * r^2 - 1  # Map r ∈ [0,1] to ξ ∈ [-1,1]
    
    # Evaluate Jacobi polynomial P_k^{(l,0)}(ξ)
    jacobi_val = evaluate_jacobi_polynomial(k, l, 0, ξ, T)
    
    return r^l * jacobi_val
end

"""
Evaluate Jacobi polynomial P_n^{(α,β)}(x).
"""
function evaluate_jacobi_polynomial(n::Int, α::T, β::T, x::T, ::Type{T}) where T<:Real
    if n == 0
        return one(T)
    elseif n == 1
        return (α - β + (α + β + 2) * x) / 2
    end
    
    # Three-term recurrence relation
    p0 = one(T)
    p1 = (α - β + (α + β + 2) * x) / 2
    
    for k in 2:n
        a_k = 2 * k * (k + α + β) * (2 * k + α + β - 2)
        b_k = (2 * k + α + β - 1) * (α^2 - β^2 + (2 * k + α + β) * (2 * k + α + β - 2) * x)
        c_k = 2 * (k + α - 1) * (k + β - 1) * (2 * k + α + β)
        
        p2 = (b_k * p1 - c_k * p0) / a_k
        p0, p1 = p1, p2
    end
    
    return p1
end

# Intermediate layout transformations
function grid_to_mixed_phi!(field::SphericalScalarField{T}) where T<:Real
    apply_phi_transform!(field, true)
end

function mixed_phi_to_grid!(field::SphericalScalarField{T}) where T<:Real
    apply_phi_transform!(field, false)
end

function mixed_phi_to_spectral!(field::SphericalScalarField{T}) where T<:Real
    apply_theta_transform!(field, true)
    apply_radial_transform!(field, true)
end