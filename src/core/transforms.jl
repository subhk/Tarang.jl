"""
Spectral transform classes with PencilFFTs integration

CPU-only (GPU support removed).
"""

using PencilFFTs
using FFTW
using LinearAlgebra
using SparseArrays

abstract type Transform end

# PencilFFTs-based transforms for parallel 2D FFTs
struct PencilFFTTransform <: Transform
    plan::Union{Nothing, PencilFFTs.PencilFFTPlan}
    basis::Basis

    function PencilFFTTransform(basis::Basis)
        new(nothing, basis)
    end
end

struct FourierTransform <: Transform
    plan_forward::Union{Nothing, Any}
    plan_backward::Union{Nothing, Any}
    basis::Basis

    function FourierTransform(basis::Basis)
        new(nothing, nothing, basis)
    end
end

mutable struct ChebyshevTransform <: Transform
    matrices::Dict{String, AbstractMatrix}
    basis::ChebyshevT

    # FFTW DCT plans
    forward_plan::Union{Nothing, Any}
    backward_plan::Union{Nothing, Any}

    # Scaling factors for FastCosineTransform
    forward_rescale_zero::Float64
    forward_rescale_pos::Float64
    backward_rescale_zero::Float64
    backward_rescale_pos::Float64

    # Size information for padding/truncation
    grid_size::Int
    coeff_size::Int
    Kmax::Int
    axis::Int

    function ChebyshevTransform(basis::ChebyshevT)
        new(
            Dict{String, AbstractMatrix}(),
            basis,
            nothing, nothing,      # FFTW plans
            0.0, 0.0, 0.0, 0.0,    # Scaling factors
            0, 0, 0, 0             # Sizes and axis
        )
    end
end

mutable struct LegendreTransform <: Transform
    matrices::Dict{String, AbstractMatrix}
    basis::Legendre

    # Quadrature information
    grid_points::Union{Nothing, AbstractVector{Float64}}
    quad_weights::Union{Nothing, AbstractVector{Float64}}

    # Size information for dealiasing
    grid_size::Int
    coeff_size::Int
    axis::Int

    function LegendreTransform(basis::Legendre)
        new(
            Dict{String, AbstractMatrix}(),
            basis,
            nothing, nothing,  # Quadrature points and weights
            0, 0, 0            # Sizes and axis
        )
    end
end

# Transform planning and execution
function plan_transforms!(dist::Distributor, domain::Domain)
    """Plan all transforms for a domain using PencilFFTs for parallel multi-D FFT"""

    gshape = global_shape(domain)
    ndim = length(domain.bases)

    # Analyze basis types
    fourier_axes = Int[]
    chebyshev_axes = Int[]
    legendre_axes = Int[]

    for (i, basis) in enumerate(domain.bases)
        if isa(basis, RealFourier) || isa(basis, ComplexFourier)
            push!(fourier_axes, i)
        elseif isa(basis, ChebyshevT)
            push!(chebyshev_axes, i)
        elseif isa(basis, Legendre)
            push!(legendre_axes, i)
        end
    end

    # Use PencilFFTs for multi-dimensional problems with Fourier components
    if ndim >= 2 && length(fourier_axes) >= 1
        if ndim == 2
            setup_pencil_fft_transforms_2d!(dist, domain, global_shape, fourier_axes)
        elseif ndim == 3
            setup_pencil_fft_transforms_3d!(dist, domain, global_shape, fourier_axes)
        else
            @warn "PencilFFTs not configured for $(ndim)D, falling back to FFTW"
            setup_fftw_transforms_nd!(dist, domain, fourier_axes)
        end

    # Use parallel Chebyshev transforms for multi-D Chebyshev-only domains
    elseif ndim >= 2 && length(chebyshev_axes) >= 2 && dist.size > 1
        # Pure Chebyshev multi-dimensional domain with MPI
        setup_parallel_chebyshev_transforms!(dist, domain, chebyshev_axes)

        # Also setup any non-Chebyshev transforms
        for (i, basis) in enumerate(domain.bases)
            if isa(basis, Legendre)
                setup_legendre_transform!(dist, basis, i)
            end
        end

        if dist.rank == 0
            @info "Using parallel Chebyshev-Chebyshev transforms"
            @info "  Chebyshev axes: $chebyshev_axes"
            @info "  Process mesh: $(dist.mesh)"
        end

    else
        # 1D case, serial, or mixed non-Fourier - use regular transforms
        for (i, basis) in enumerate(domain.bases)
            if isa(basis, RealFourier) || isa(basis, ComplexFourier)
                setup_fftw_transform!(dist, basis, i)
            elseif isa(basis, ChebyshevT)
                setup_chebyshev_transform!(dist, basis, i)
            elseif isa(basis, Legendre)
                setup_legendre_transform!(dist, basis, i)
            end
        end
    end
end

function setup_pencil_fft_transforms_2d!(dist::Distributor, domain::Domain, 
                                    global_shape::Tuple, fourier_axes::Vector{Int})

    """Setup PencilFFTs transforms for parallel 2D FFT"""
    
    if dist.pencil_config === nothing
        setup_pencil_arrays(dist, global_shape)
    end
    
    # Create PencilFFT plan for the Fourier dimensions
    if length(fourier_axes) == 1
        # 1D FFT using PencilFFTs
        transform_type = PencilFFTs.Transforms.FFT()
        dims = (fourier_axes[1],)
    elseif length(fourier_axes) == 2
        # 2D FFT using PencilFFTs - this enables both vertical and horizontal parallelization
        transform_type = PencilFFTs.Transforms.FFT()
        dims = tuple(fourier_axes...)
    else
        throw(ArgumentError("Too many Fourier axes for PencilFFTs"))
    end
    
    # Create the PencilFFT plan
    pencil = create_pencil(dist, global_shape, 1)
    fft_plan = PencilFFTs.PencilFFTPlan(pencil, transform_type, dims)
    
    # Store the plan in the distributor
    push!(dist.transforms, fft_plan)
    
    @info "Set up PencilFFT transform for axes $fourier_axes with global shape $global_shape"
    @info "Parallel decomposition: vertical=$(dist.mesh[1]) × horizontal=$(dist.mesh[2]) processes"
end

function setup_fftw_transform!(dist::Distributor, basis::Union{RealFourier, ComplexFourier}, axis::Int)
    """Setup FFTW transforms for 1D case (CPU only)."""
    transform = FourierTransform(basis)
    setup_cpu_fft_transform!(transform, basis)
    push!(dist.transforms, transform)
end

function setup_cpu_fft_transform!(transform::FourierTransform, basis::Union{RealFourier, ComplexFourier})
    """Setup CPU FFTW transforms"""
    
    # Create dummy arrays for planning
    if isa(basis, RealFourier)
        dummy_in = zeros(Float64, basis.meta.size)
        dummy_out = zeros(ComplexF64, div(basis.meta.size, 2) + 1)
        
        transform.plan_forward = FFTW.plan_rfft(dummy_in)
        transform.plan_backward = FFTW.plan_irfft(dummy_out, basis.meta.size)
    else # ComplexFourier
        dummy = zeros(ComplexF64, basis.meta.size)
        
        transform.plan_forward = FFTW.plan_fft(dummy)
        transform.plan_backward = FFTW.plan_ifft(dummy)
    end
end


function setup_chebyshev_transform!(dist::Distributor, basis::ChebyshevT, axis::Int)
    """
    Setup Chebyshev transform following Tarang FastChebyshevTransform implementation.

    Based on Tarang transforms:
    - Uses DCT-II for forward transform (grid to coefficients)
    - Uses DCT-III for backward transform (coefficients to grid)
    - Proper scaling factors for unit-amplitude normalization
    - Supports padding/truncation for different grid/coefficient sizes
    """
    
    transform = ChebyshevTransform(basis)
    
    grid_size = basis.meta.size
    coeff_size = basis.meta.size  # Can be different for dealiasing
    
    setup_chebyshev_cpu_transform!(transform, grid_size, coeff_size, axis)
    
    push!(dist.transforms, transform)
    
    @debug "Chebyshev transform setup: grid_size=$grid_size, coeff_size=$coeff_size, Kmax=$(transform.Kmax)"
end

function setup_chebyshev_cpu_transform!(transform::ChebyshevTransform, grid_size::Int, coeff_size::Int, axis::Int)
    """Setup CPU Chebyshev transform using FFTW DCT"""
    
    try
        # Try to use FFTW for fast DCT transforms
        # Create DCT-II plan (forward: grid to coefficients)
        forward_plan = FFTW.plan_r2r(zeros(grid_size), FFTW.REDFT10, flags=FFTW.MEASURE)
        
        # Create DCT-III plan (backward: coefficients to grid)  
        backward_plan = FFTW.plan_r2r(zeros(grid_size), FFTW.REDFT01, flags=FFTW.MEASURE)
        
        transform.forward_plan = forward_plan
        transform.backward_plan = backward_plan
        
        # Set scaling factors following Tarang FastCosineTransform
        transform.forward_rescale_zero = 1.0 / grid_size / 2.0   # DC component
        transform.forward_rescale_pos = 1.0 / grid_size          # AC components
        transform.backward_rescale_zero = 1.0                    # DC component
        transform.backward_rescale_pos = 0.5                     # AC components
        
        @info "Setup FFTW-based Chebyshev transform for axis $axis, N=$grid_size"

    catch e
        @warn "FFTW not available or failed ($e), falling back to matrix-based DCT"
        setup_chebyshev_matrix_transform!(transform, grid_size, coeff_size, axis)
    end
    
    # Store grid and coefficient sizes
    transform.grid_size = grid_size
    transform.coeff_size = coeff_size
    transform.Kmax = min(grid_size - 1, coeff_size - 1)
    transform.axis = axis
end

function setup_chebyshev_matrix_transform!(transform::ChebyshevTransform, grid_size::Int, coeff_size::Int, axis::Int)
    """Setup CPU matrix-based Chebyshev transform"""

    # DCT-II matrix for forward transform (grid to coefficients)
    forward_matrix = zeros(coeff_size, grid_size)
    for i in 0:coeff_size-1, j in 0:grid_size-1
        if i == 0
            forward_matrix[i+1, j+1] = 1.0 / grid_size / 2.0
        else
            forward_matrix[i+1, j+1] = cos(π * i * j / (grid_size-1)) / grid_size
        end
    end

    # DCT-III matrix for backward transform (coefficients to grid)
    backward_matrix = zeros(grid_size, coeff_size)
    for i in 0:grid_size-1, j in 0:coeff_size-1
        if j == 0
            backward_matrix[i+1, j+1] = 1.0
        else
            backward_matrix[i+1, j+1] = 2.0 * cos(π * j * i / (grid_size-1)) * 0.5
        end
    end

    transform.matrices["forward"] = sparse(forward_matrix)
    transform.matrices["backward"] = sparse(backward_matrix)

    @info "Setup CPU matrix-based Chebyshev transform for axis $axis, N=$grid_size"
end

function setup_legendre_transform!(dist::Distributor, basis::Legendre, axis::Int)
    """
    Setup Legendre transform using JacobiMMT implementation.

    - Uses Gauss-Legendre quadrature (Jacobi with a=0, b=0)
    - Forward transform: integration using quadrature weights
    - Backward transform: polynomial evaluation at quadrature points
    - Proper normalization for orthogonal Legendre polynomials
    """
    
    transform = LegendreTransform(basis)
    
    grid_size = basis.meta.size
    coeff_size = basis.meta.size  # Can be different for dealiasing
    
    try
        # Get Gauss-Legendre quadrature points and weights
        try
            grid_points, quad_weights = FastGaussQuadrature.gausslegendre(grid_size)
            @info "Using FastGaussQuadrature for Legendre transform, N=$grid_size"
        catch e
            @warn "FastGaussQuadrature not available ($e), using manual implementation"
            grid_points, quad_weights = compute_legendre_quadrature(grid_size)
        end
        
        # Build Legendre polynomials at quadrature points
        poly_matrix = build_legendre_polynomials(coeff_size, grid_points)
        
        # Build forward transform matrix
        forward_matrix = zeros(coeff_size, grid_size)
        for n in 0:coeff_size-1, i in 1:grid_size
            normalization = sqrt((2*n + 1) / 2.0)
            forward_matrix[n+1, i] = poly_matrix[n+1, i] * quad_weights[i] * normalization
        end
        
        # Build backward transform matrix
        backward_matrix = zeros(grid_size, coeff_size)
        for i in 1:grid_size, n in 0:coeff_size-1
            normalization = sqrt((2*n + 1) / 2.0)
            backward_matrix[i, n+1] = poly_matrix[n+1, i] * normalization
        end
        
        transform.matrices["forward"] = sparse(forward_matrix)
        transform.matrices["backward"] = sparse(backward_matrix)
        transform.grid_points = grid_points
        transform.quad_weights = quad_weights
        
        transform.grid_size = grid_size
        transform.coeff_size = coeff_size
        transform.axis = axis
        
        @info "Setup Legendre transform for axis $axis: grid_size=$grid_size, coeff_size=$coeff_size"
        
    catch e
        @error "Legendre transform setup failed: $e, using identity fallback"
        
        # Fallback to identity matrices
        N = basis.meta.size
        I_matrix = Matrix{Float64}(I, N, N)
        
        transform.matrices["forward"] = sparse(I_matrix')
        transform.matrices["backward"] = sparse(I_matrix)
        transform.axis = axis
        
        @warn "Using identity matrix fallback for Legendre transform"
    end
    
    push!(dist.transforms, transform)
    
    @debug "Legendre transform setup completed for axis $axis"
end


# Helper functions for Legendre transform
function compute_legendre_quadrature(N::Int)
    """
    Compute Gauss-Legendre quadrature points and weights manually.
    Based on standard algorithms for Legendre polynomial roots.
    """
    
    if N == 1
        return [0.0], [2.0]
    end
    
    # Initial guess for roots using Chebyshev points  
    k = 1:N
    x = cos.(π * (k .- 0.5) / N)
    
    # Newton-Raphson iteration to find Legendre polynomial roots
    max_iter = 100
    tol = 1e-14
    
    for iter in 1:max_iter
        # Evaluate Legendre polynomial P_N(x) and its derivative P'_N(x)
        P, Pprime = evaluate_legendre_and_derivative(x, N)
        
        # Newton-Raphson update
        dx = P ./ Pprime
        x = x - dx
        
        if maximum(abs.(dx)) < tol
            break
        end
        
        if iter == max_iter
            @warn "Legendre quadrature did not converge"
        end
    end
    
    # Compute quadrature weights
    # w_i = 2 / [(1 - x_i^2) * (P'_N(x_i))^2]
    _, Pprime = evaluate_legendre_and_derivative(x, N)
    weights = 2.0 ./ ((1.0 .- x.^2) .* Pprime.^2)
    
    return x, weights
end

function evaluate_legendre_and_derivative(x::Vector{Float64}, N::Int)
    """
    Evaluate Legendre polynomial P_N(x) and its derivative P'_N(x) using recurrence relations.
    """
    
    if N == 0
        return ones(length(x)), zeros(length(x))
    elseif N == 1
        return copy(x), ones(length(x))
    end
    
    # Use three-term recurrence: (n+1)P_{n+1}(x) = (2n+1)x P_n(x) - n P_{n-1}(x)
    P_prev = ones(length(x))      # P_0(x) = 1
    P_curr = copy(x)              # P_1(x) = x
    Pprime_prev = zeros(length(x)) # P'_0(x) = 0  
    Pprime_curr = ones(length(x))  # P'_1(x) = 1
    
    for n in 1:N-1
        # Compute P_{n+1}(x)
        P_next = ((2*n + 1) * x .* P_curr - n * P_prev) / (n + 1)
        
        # Compute P'_{n+1}(x) using: (1-x^2)P'_n(x) = n[P_{n-1}(x) - x P_n(x)]
        # Rearranged: P'_{n+1}(x) = ((2n+1)[P_n(x) + x P'_n(x)] - n P'_{n-1}(x)) / (n+1)
        Pprime_next = ((2*n + 1) * (P_curr + x .* Pprime_curr) - n * Pprime_prev) / (n + 1)
        
        # Update for next iteration
        P_prev = P_curr
        P_curr = P_next
        Pprime_prev = Pprime_curr  
        Pprime_curr = Pprime_next
    end
    
    return P_curr, Pprime_curr
end

function build_legendre_polynomials(M::Int, grid_points::Vector{Float64})
    """
    Build matrix of Legendre polynomial values P_n(x_i) for n=0,...,M-1 and grid points x_i.
    """
    
    N_grid = length(grid_points)
    poly_matrix = zeros(M, N_grid)
    
    # P_0(x) = 1
    if M > 0
        poly_matrix[1, :] .= 1.0
    end
    
    # P_1(x) = x
    if M > 1
        poly_matrix[2, :] = grid_points
    end
    
    # Use three-term recurrence for higher orders
    for n in 2:M-1
        for i in 1:N_grid
            x = grid_points[i]
            poly_matrix[n+1, i] = ((2*n - 1) * x * poly_matrix[n, i] - (n - 1) * poly_matrix[n-1, i]) / n
        end
    end
    
    return poly_matrix
end

# Transform execution functions
function forward_transform!(field::ScalarField, target_layout::Symbol=:c)
    """Apply forward transform to field"""

    if field.domain === nothing
        return
    end

    ensure_layout!(field, :g)  # Start in grid space

    # Find appropriate transform
    for transform in field.dist.transforms
        if isa(transform, PencilFFTs.PencilFFTPlan)
            # Use PencilFFT for parallel 2D FFT
            field.data_c = transform * field.data_g
            field.current_layout = :c
            return
        elseif isa(transform, ParallelChebyshevTransform)
            # Use parallel Chebyshev transform
            if field.data_c === nothing
                field.data_c = similar(field.data_g, ComplexF64)
            end
            # Copy to temporary real array for transform
            temp_data = copy(real.(field.data_g))
            apply_parallel_chebyshev_forward!(temp_data, transform, field.dist)
            field.data_c .= temp_data
            field.current_layout = :c
            return
        elseif isa(transform, FourierTransform)
            # Apply FFT
            apply_fourier_forward!(field, transform)
            field.current_layout = :c
            return
        elseif isa(transform, ChebyshevTransform)
            # Apply Chebyshev transform
            apply_chebyshev_forward!(field, transform)
            field.current_layout = :c
            return
        elseif isa(transform, LegendreTransform)
            # Apply Legendre transform
            apply_legendre_forward!(field, transform)
            field.current_layout = :c
            return
        end
    end

    # Fallback for other transforms
    field.data_c .= field.data_g
    field.current_layout = :c
end

function apply_fourier_forward!(field::ScalarField, transform::FourierTransform)
    """Apply forward Fourier transform """
    
    # Ensure field data is on correct device
    field.data_g = field.data_g
    
    if isa(transform.basis, RealFourier)
        field.data_c = transform.plan_forward * field.data_g
    else
        field.data_c = transform.plan_forward * field.data_g
    end
end

function backward_transform!(field::ScalarField, target_layout::Symbol=:g)
    """Apply backward transform to field """

    if field.domain === nothing
        return
    end

    ensure_layout!(field, :c)  # Start in coefficient space

    # Find appropriate transform
    for transform in field.dist.transforms
        if isa(transform, PencilFFTs.PencilFFTPlan)
            # Use PencilFFT for parallel 2D FFT
            field.data_g = inv(transform) * field.data_c
            field.current_layout = :g
            return
        elseif isa(transform, ParallelChebyshevTransform)
            # Use parallel Chebyshev transform
            if field.data_g === nothing
                field.data_g = similar(field.data_c, Float64)
            end
            # Copy to temporary real array for transform
            temp_data = copy(real.(field.data_c))
            apply_parallel_chebyshev_backward!(temp_data, transform, field.dist)
            field.data_g .= temp_data
            field.current_layout = :g
            return
        elseif isa(transform, FourierTransform)
            # Apply FFT
            apply_fourier_backward!(field, transform)
            field.current_layout = :g
            return
        elseif isa(transform, ChebyshevTransform)
            # Apply Chebyshev transform
            apply_chebyshev_backward!(field, transform)
            field.current_layout = :g
            return
        elseif isa(transform, LegendreTransform)
            # Apply Legendre transform
            apply_legendre_backward!(field, transform)
            field.current_layout = :g
            return
        end
    end

    # Fallback for other transforms
    field.data_g .= field.data_c
    field.current_layout = :g
end

function apply_fourier_backward!(field::ScalarField, transform::FourierTransform)
    """Apply backward Fourier transform """
    
    # Ensure field data is on correct device
    field.data_c = field.data_c
    
    if isa(transform.basis, RealFourier)
        field.data_g = transform.plan_backward * field.data_c
    else
        field.data_g = transform.plan_backward * field.data_c
    end
end

# Chebyshev transform application functions following Tarang patterns
function apply_chebyshev_forward!(field::ScalarField, transform::ChebyshevTransform)
    """
    Apply forward Chebyshev transform (grid to coefficients) with in-place operations.

    Based on Tarang ScipyDCT.forward and FFTWDCT.forward methods:
    - Uses DCT-II with proper scaling for unit-amplitude normalization
    - Handles padding/truncation for different grid/coefficient sizes
    - Follows resize_rescale_forward pattern
    - OPTIMIZED: Uses workspace buffers to minimize allocations
    """

    if transform.forward_plan !== nothing
        # Use FFTW DCT-II plan for CPU with workspace buffer
        try
            # Get workspace buffer instead of allocating
            wm = get_global_workspace()
            temp_data = get_workspace!(wm, Float64, (transform.grid_size,))

            # In-place DCT using mul!
            mul!(temp_data, transform.forward_plan, field.data_g)

            # Ensure output array exists
            if field.data_c === nothing
                field.data_c = zeros(ComplexF64, transform.coeff_size)
            end

            # Apply scaling factors for unit-amplitude normalization (in-place)
            @inbounds field.data_c[1] = temp_data[1] * transform.forward_rescale_zero

            if transform.Kmax > 0
                scale_pos = transform.forward_rescale_pos
                @inbounds @simd for k in 1:min(transform.Kmax, transform.coeff_size-1)
                    field.data_c[k+1] = temp_data[k+1] * scale_pos
                end
            end

            # Zero padding if coeff_size > Kmax+1
            if transform.coeff_size > transform.Kmax + 1
                @inbounds @simd for k in (transform.Kmax + 2):transform.coeff_size
                    field.data_c[k] = 0.0
                end
            end

            # Return workspace buffer
            release_workspace!(wm, temp_data)

        catch e
            @warn "DCT forward transform failed: $e, falling back to matrix method"
            apply_chebyshev_matrix_forward!(field, transform)
        end

    else
        # Use matrix-based transform
        apply_chebyshev_matrix_forward!(field, transform)
    end
end

function apply_chebyshev_backward!(field::ScalarField, transform::ChebyshevTransform)
    """
    Apply backward Chebyshev transform (coefficients to grid) with in-place operations.

    Based on Tarang ScipyDCT.backward and FFTWDCT.backward methods:
    - Uses DCT-III with proper scaling for unit-amplitude normalization
    - Handles padding/truncation for different coefficient/grid sizes
    - OPTIMIZED: Uses workspace buffers and in-place operations
    """

    if transform.backward_plan !== nothing
        # Use FFTW DCT-III plan for CPU with workspace
        try
            # Get workspace buffer instead of allocating
            wm = get_global_workspace()
            temp_data = get_workspace!(wm, Float64, (transform.grid_size,))

            # Zero the workspace (needed for correct transform)
            fill!(temp_data, 0.0)

            # Apply scaling factors following Tarang resize_rescale_backward
            if length(field.data_c) > 0
                @inbounds temp_data[1] = real(field.data_c[1]) * transform.backward_rescale_zero
            end

            if transform.Kmax > 0
                scale_pos = transform.backward_rescale_pos
                @inbounds @simd for k in 1:min(transform.Kmax, length(field.data_c)-1)
                    temp_data[k+1] = real(field.data_c[k+1]) * scale_pos
                end
            end

            # Ensure output array exists
            if field.data_g === nothing
                field.data_g = zeros(Float64, transform.grid_size)
            end

            # In-place backward DCT
            mul!(field.data_g, transform.backward_plan, temp_data)

            # Return workspace buffer
            release_workspace!(wm, temp_data)

        catch e
            @warn "DCT backward transform failed: $e, falling back to matrix method"
            apply_chebyshev_matrix_backward!(field, transform)
        end

    else
        # Use matrix-based transform
        apply_chebyshev_matrix_backward!(field, transform)
    end
end

function apply_chebyshev_matrix_forward!(field::ScalarField, transform::ChebyshevTransform)
    """Apply forward Chebyshev transform using in-place matrix multiplication"""

    if haskey(transform.matrices, "forward")
        mat = transform.matrices["forward"]
        # Ensure output array exists with correct size
        out_size = size(mat, 1)
        if field.data_c === nothing || length(field.data_c) != out_size
            field.data_c = zeros(ComplexF64, out_size)
        end
        # In-place matrix-vector multiply
        mul!(field.data_c, mat, field.data_g)
    else
        @warn "No forward matrix available for Chebyshev transform"
        if field.data_c === nothing
            field.data_c = copy(field.data_g)
        else
            copyto!(field.data_c, field.data_g)
        end
    end
end

function apply_chebyshev_matrix_backward!(field::ScalarField, transform::ChebyshevTransform)
    """Apply backward Chebyshev transform using in-place matrix multiplication"""

    if haskey(transform.matrices, "backward")
        mat = transform.matrices["backward"]
        out_size = size(mat, 1)

        # Get workspace for real part extraction
        wm = get_global_workspace()
        real_coeffs = get_workspace!(wm, Float64, (length(field.data_c),))

        # Extract real part in-place
        @inbounds @simd for i in eachindex(real_coeffs, field.data_c)
            real_coeffs[i] = real(field.data_c[i])
        end

        # Ensure output array exists
        if field.data_g === nothing || length(field.data_g) != out_size
            field.data_g = zeros(Float64, out_size)
        end

        # In-place matrix-vector multiply
        mul!(field.data_g, mat, real_coeffs)

        release_workspace!(wm, real_coeffs)
    else
        @warn "No backward matrix available for Chebyshev transform"
        if field.data_g === nothing
            field.data_g = real.(field.data_c)
        else
            @inbounds @simd for i in eachindex(field.data_g)
                field.data_g[i] = real(field.data_c[i])
            end
        end
    end
end

# Legendre transform application functions following Tarang JacobiMMT patterns
function apply_legendre_forward!(field::ScalarField, transform::LegendreTransform)
    """
    Apply forward Legendre transform (grid to coefficients) with in-place operations.

    Based on Tarang JacobiMMT.forward_matrix:
    - Uses Gauss-Legendre quadrature integration
    - Proper normalization for orthonormal Legendre expansion
    - OPTIMIZED: In-place matrix-vector multiplication
    """

    if haskey(transform.matrices, "forward")
        mat = transform.matrices["forward"]
        out_size = size(mat, 1)

        # Ensure output array exists
        if field.data_c === nothing || length(field.data_c) != out_size
            field.data_c = zeros(ComplexF64, out_size)
        end

        # In-place forward transform: c_n = ∫ f(x) P_n(x) dx ≈ Σ f(x_i) P_n(x_i) w_i
        mul!(field.data_c, mat, field.data_g)

        @debug "Applied Legendre forward transform: grid_size=$(transform.grid_size), coeff_size=$(transform.coeff_size)"

    else
        @warn "No forward matrix available for Legendre transform, using identity"
        if field.data_c === nothing
            field.data_c = copy(field.data_g)
        else
            copyto!(field.data_c, field.data_g)
        end
    end
end

function apply_legendre_backward!(field::ScalarField, transform::LegendreTransform)
    """
    Apply backward Legendre transform (coefficients to grid) with in-place operations.

    Based on Tarang polynomial evaluation:
    - Evaluates f(x) = Σ c_n P_n(x) at Gauss-Legendre quadrature points
    - OPTIMIZED: Uses workspace buffer and in-place operations
    """

    if haskey(transform.matrices, "backward")
        mat = transform.matrices["backward"]
        out_size = size(mat, 1)

        # Get workspace for real part extraction
        wm = get_global_workspace()
        real_coeffs = get_workspace!(wm, Float64, (length(field.data_c),))

        # Extract real part in-place
        @inbounds @simd for i in eachindex(real_coeffs, field.data_c)
            real_coeffs[i] = real(field.data_c[i])
        end

        # Ensure output array exists
        if field.data_g === nothing || length(field.data_g) != out_size
            field.data_g = zeros(Float64, out_size)
        end

        # In-place backward transform: f(x_i) = Σ c_n P_n(x_i)
        mul!(field.data_g, mat, real_coeffs)

        release_workspace!(wm, real_coeffs)

        @debug "Applied Legendre backward transform: coeff_size=$(transform.coeff_size), grid_size=$(transform.grid_size)"

    else
        @warn "No backward matrix available for Legendre transform, using identity"
        if field.data_g === nothing
            field.data_g = real.(field.data_c)
        else
            @inbounds @simd for i in eachindex(field.data_g)
                field.data_g[i] = real(field.data_c[i])
            end
        end
    end
end

# Dealiasing operations following Tarang patterns
function dealias!(field::ScalarField, scales::Union{Real, Vector{Real}})
    """
    Apply dealiasing to field following Tarang field.change_scales and low_pass_filter implementation.
    
    Based on Tarang field:
    - Ensures field is in coefficient space for mode truncation
    - Applies scale-based truncation for each basis type
    - Handles multi-dimensional tensor product bases properly
    - Uses basis-specific dealiasing patterns (Fourier, Chebyshev, Legendre)
    
    Parameters
    ----------
    scales : Real or Vector{Real}
        Scale factors for each basis dimension (0 < scale <= 1)
        Values < 1 remove high-frequency modes for dealiasing
    """
    
    # Ensure we're in coefficient space for dealiasing
    ensure_layout!(field, :c)
    
    if field.domain === nothing
        @debug "Field has no domain, skipping dealiasing"
        return
    end
    
    # Convert scales to vector format
    if isa(scales, Real)
        scale_vec = fill(scales, length(field.domain.bases))
    else
        scale_vec = scales
    end
    
    if length(scale_vec) != length(field.domain.bases)
        throw(ArgumentError("Number of scales must match number of bases"))
    end
    
    @debug "Applying dealiasing with scales: $scale_vec"
    
    # Apply dealiasing for each basis dimension following Tarang patterns
    for (axis, (basis, scale)) in enumerate(zip(field.domain.bases, scale_vec))
        if scale >= 1.0
            continue  # No dealiasing needed for this axis
        end
        
        apply_basis_dealiasing!(field, basis, axis, scale)
    end
    
    @debug "Dealiasing completed"
end

function apply_basis_dealiasing!(field::ScalarField, basis::Union{RealFourier, ComplexFourier}, axis::Int, scale::Real)
    """Apply Fourier basis dealiasing following Tarang patterns"""
    
    # Calculate cutoff mode for Fourier basis
    total_modes = basis.meta.size
    kept_modes = Int(floor(total_modes * scale))
    
    @debug "Fourier dealiasing: axis=$axis, total_modes=$total_modes, kept_modes=$kept_modes, scale=$scale"
    
    # Get coefficient array dimensions
    data_size = size(field.data_c)
    
    if axis > length(data_size)
        @warn "Axis $axis exceeds field dimensions $(length(data_size))"
        return
    end
    
    axis_size = data_size[axis]
    
    # Handle different Fourier representations
    if isa(basis, RealFourier)
        # Real Fourier: modes are [DC, cos(k), sin(k), cos(2k), sin(2k), ...]
        # Keep DC component and first kept_modes/2 frequency pairs
        if kept_modes < axis_size
            cutoff_index = kept_modes + 1
            
            # Zero out high-frequency modes
            indices = [1:i for i in data_size]
            indices[axis] = cutoff_index:axis_size
            field.data_c[indices...] .= 0
        end
        
    elseif isa(basis, ComplexFourier)  
        # Complex Fourier: modes are [..., exp(-ik), ..., DC, ..., exp(ik), ...]
        # Keep central kept_modes around DC component
        if kept_modes < axis_size
            center = axis_size ÷ 2 + 1  # DC component location
            half_kept = kept_modes ÷ 2
            
            # Zero out negative high frequencies
            if center - half_kept > 1
                indices = [1:i for i in data_size]
                indices[axis] = 1:(center - half_kept - 1)
                field.data_c[indices...] .= 0
            end
            
            # Zero out positive high frequencies  
            if center + half_kept < axis_size
                indices = [1:i for i in data_size]
                indices[axis] = (center + half_kept + 1):axis_size
                field.data_c[indices...] .= 0
            end
        end
    end
end

function apply_basis_dealiasing!(field::ScalarField, basis::ChebyshevT, axis::Int, scale::Real)
    """Apply Chebyshev basis dealiasing following Tarang patterns"""
    
    # Calculate cutoff mode for Chebyshev basis
    total_modes = basis.meta.size
    kept_modes = Int(floor(total_modes * scale))
    
    @debug "Chebyshev dealiasing: axis=$axis, total_modes=$total_modes, kept_modes=$kept_modes, scale=$scale"
    
    # Get coefficient array dimensions
    data_size = size(field.data_c)
    
    if axis > length(data_size)
        @warn "Axis $axis exceeds field dimensions $(length(data_size))"
        return
    end
    
    axis_size = data_size[axis]
    
    # Chebyshev modes are ordered [T_0, T_1, T_2, ..., T_{N-1}]
    # Keep first kept_modes coefficients, zero out the rest
    if kept_modes < axis_size
        cutoff_index = kept_modes + 1
        
        # Zero out high-order polynomial modes
        indices = [1:i for i in data_size]
        indices[axis] = cutoff_index:axis_size
        field.data_c[indices...] .= 0
    end
end

function apply_basis_dealiasing!(field::ScalarField, basis::Legendre, axis::Int, scale::Real)
    """Apply Legendre basis dealiasing following Tarang patterns"""
    
    # Calculate cutoff mode for Legendre basis
    total_modes = basis.meta.size
    kept_modes = Int(floor(total_modes * scale))
    
    @debug "Legendre dealiasing: axis=$axis, total_modes=$total_modes, kept_modes=$kept_modes, scale=$scale"
    
    # Get coefficient array dimensions
    data_size = size(field.data_c)
    
    if axis > length(data_size)
        @warn "Axis $axis exceeds field dimensions $(length(data_size))"
        return
    end
    
    axis_size = data_size[axis]
    
    # Legendre modes are ordered [P_0, P_1, P_2, ..., P_{N-1}]
    # Keep first kept_modes coefficients, zero out the rest
    if kept_modes < axis_size
        cutoff_index = kept_modes + 1
        
        # Zero out high-order polynomial modes
        indices = [1:i for i in data_size]
        indices[axis] = cutoff_index:axis_size
        field.data_c[indices...] .= 0
    end
end

# Generic fallback for unknown basis types
function apply_basis_dealiasing!(field::ScalarField, basis, axis::Int, scale::Real)
    """Generic dealiasing for unknown basis types"""
    
    @warn "Unknown basis type $(typeof(basis)) for axis $axis, using generic polynomial dealiasing"
    
    # Calculate cutoff mode generically
    total_modes = basis.meta.size
    kept_modes = Int(floor(total_modes * scale))
    
    # Get coefficient array dimensions
    data_size = size(field.data_c)
    
    if axis > length(data_size) || kept_modes >= data_size[axis]
        return
    end
    
    cutoff_index = kept_modes + 1
    
    # Zero out high modes generically
    indices = [1:i for i in data_size]
    indices[axis] = cutoff_index:data_size[axis]
    field.data_c[indices...] .= 0
end

# Convenience function for domain-based dealiasing
function dealias!(field::ScalarField)
    """Apply default dealiasing using domain.dealias scales"""
    
    if field.domain !== nothing && hasfield(typeof(field.domain), :dealias)
        dealias!(field, field.domain.dealias)
    else
        @warn "No domain dealias information available, using default 2/3 rule"
        dealias!(field, 2/3)  # Standard 2/3 rule for dealiasing
    end
end

# Utility functions
function get_transform_for_basis(transforms::Vector, basis::Basis)
    """Find transform corresponding to a basis"""
    for transform in transforms
        if hasfield(typeof(transform), :basis) && transform.basis == basis
            return transform
        end
    end
    return nothing
end

function setup_pencil_fft_transforms_3d!(dist::Distributor, domain::Domain, 
                                    global_shape::Tuple, fourier_axes::Vector{Int})

    """Setup PencilFFTs transforms for parallel 3D FFT"""
    
    if dist.pencil_config === nothing
        # Create 3D pencil configuration
        setup_pencil_arrays_3d(dist, global_shape)
    end
    
    # Determine FFT dimensions and type
    if length(fourier_axes) == 3
        # Pure 3D FFT - all dimensions are Fourier
        transform_type = PencilFFTs.Transforms.FFT()
        dims = tuple(fourier_axes...)
        @info "Setting up 3D FFT for all axes: $fourier_axes"
        
    elseif length(fourier_axes) == 2
        # 2D FFT in 3D domain - mixed spectral/physical
        transform_type = PencilFFTs.Transforms.FFT() 
        dims = tuple(fourier_axes...)
        @info "Setting up 2D FFT in 3D domain for axes: $fourier_axes"
        
    elseif length(fourier_axes) == 1
        # 1D FFT in 3D domain
        transform_type = PencilFFTs.Transforms.FFT()
        dims = tuple(fourier_axes...)
        @info "Setting up 1D FFT in 3D domain for axis: $fourier_axes"
        
    else
        throw(ArgumentError("No Fourier axes found for 3D PencilFFTs"))
    end
    
    # Create the PencilFFT plan with 3D pencil decomposition
    pencil = create_pencil_3d(dist, global_shape, 1)
    fft_plan = PencilFFTs.PencilFFTPlan(pencil, transform_type, dims)
    
    # Store the plan in the distributor
    push!(dist.transforms, fft_plan)
    
    @info "Set up 3D PencilFFT transform for axes $fourier_axes with global shape $global_shape"
    @info "3D parallel decomposition: $(dist.mesh[1]) × $(dist.mesh[2]) × $(dist.mesh[3]) processes"
end

function setup_fftw_transforms_nd!(dist::Distributor, domain::Domain, fourier_axes::Vector{Int})
    """Fallback FFTW transforms for high-dimensional problems"""
    
    @info "Using FFTW fallback for $(length(domain.bases))D problem"
    
    for (i, basis) in enumerate(domain.bases)
        if isa(basis, RealFourier) || isa(basis, ComplexFourier)
            setup_fftw_transform!(dist, basis, i)
        end
    end
end

function setup_pencil_arrays_3d(dist::Distributor, global_shape::Tuple{Vararg{Int}})
    """Setup 3D PencilArrays configuration"""
    
    if length(global_shape) != 3
        throw(ArgumentError("3D setup requires 3D global shape, got $(length(global_shape))D"))
    end
    
    if length(dist.mesh) != 3
        throw(ArgumentError("3D setup requires 3D process mesh, got $(length(dist.mesh))D"))
    end
    
    # Validate mesh
    if prod(dist.mesh) != dist.size
        throw(ArgumentError("3D process mesh $(dist.mesh) incompatible with $(dist.size) processes"))
    end
    
    # Create 3D PencilArrays configuration with full 3D decomposition
    dist.pencil_config = PencilConfig(
        global_shape, 
        dist.mesh, 
        comm=dist.comm,
        decomp_dims=(true, true, true)  # Enable decomposition along all 3 axes
    )
    
    @info "Created 3D PencilArrays configuration:"
    @info "  Global shape: $global_shape"
    @info "  Process mesh: $(dist.mesh)"
    @info "  Decomposition: 3D (all axes)"
    
    return dist.pencil_config
end

function create_pencil_3d(dist::Distributor, global_shape::Tuple{Vararg{Int}},
                        decomp_index::Int=1; dtype::Type=dist.dtype)

    """Create a 3D pencil array with specified decomposition"""

    if dist.pencil_config === nothing
        setup_pencil_arrays_3d(dist, global_shape)
    end

    config = dist.pencil_config

    if dist.size == 1
        # Serial execution - just create a regular array
        return zeros(dtype, global_shape...)
    else
        # Parallel execution - use PencilArrays properly
        try
            return PencilArrays.PencilArray{dtype}(undef, global_shape, config.comm)
        catch e
            @warn "PencilArrays creation failed, using regular array" exception=e
            return zeros(dtype, global_shape...)
        end
    end
end

# Enhanced 3D transform execution
function forward_transform_3d!(field::ScalarField, target_layout::Symbol=:c)
    """Apply 3D forward transform to field"""
    
    if field.domain === nothing || length(field.domain.bases) != 3
        forward_transform!(field, target_layout)  # Fall back to general case
        return
    end
    
    ensure_layout!(field, :g)  # Start in grid space
    
    # Find appropriate 3D transform
    for transform in field.dist.transforms
        if isa(transform, PencilFFTs.PencilFFTPlan)
            # Check if this is a 3D transform
            if hasfield(typeof(transform), :dims) && length(transform.dims) <= 3
                field.data_c = transform * field.data_g
                field.current_layout = :c
                @debug "Applied 3D PencilFFT forward transform"
                return
            end
        end
    end
    
    # Fallback to regular transform
    forward_transform!(field, target_layout)
end

function backward_transform_3d!(field::ScalarField, target_layout::Symbol=:g)
    """Apply 3D backward transform to field"""
    
    if field.domain === nothing || length(field.domain.bases) != 3
        backward_transform!(field, target_layout)  # Fall back to general case
        return
    end
    
    ensure_layout!(field, :c)  # Start in coefficient space
    
    # Find appropriate 3D transform
    for transform in field.dist.transforms
        if isa(transform, PencilFFTs.PencilFFTPlan)
            # Check if this is a 3D transform
            if hasfield(typeof(transform), :dims) && length(transform.dims) <= 3
                field.data_g = inv(transform) * field.data_c
                field.current_layout = :g
                @debug "Applied 3D PencilFFT backward transform"
                return
            end
        end
    end
    
    # Fallback to regular transform
    backward_transform!(field, target_layout)
end

"""
    dealias_3d!(field::ScalarField, scales::Union{Real, Vector{Real}})

Apply 3D dealiasing to a scalar field by zeroing out high-frequency modes.

Dealiasing is essential for nonlinear computations to avoid aliasing errors.
The 2/3 rule (scale = 2/3) is commonly used: modes with k > k_max * scale are zeroed.

For Fourier bases, this zeros modes beyond the cutoff frequency.
For Chebyshev bases, this zeros high-order polynomial coefficients.

# Arguments
- `field`: ScalarField to dealias (modified in-place)
- `scales`: Dealiasing scale(s). Can be:
  - A single Real value applied to all dimensions
  - A Vector of scales for each dimension

# Example
```julia
# Apply 2/3 dealiasing rule to all dimensions
dealias_3d!(field, 2/3)

# Apply different scales per dimension
dealias_3d!(field, [2/3, 2/3, 1.0])  # No dealiasing in z
```
"""
function dealias_3d!(field::ScalarField, scales::Union{Real, Vector{Real}})
    if field.domain === nothing
        @warn "Cannot dealias field without domain"
        return
    end

    ensure_layout!(field, :c)

    if field.data_c === nothing
        @warn "No coefficient data to dealias"
        return
    end

    ndim = ndims(field.data_c)
    nbases = length(field.domain.bases)

    # Normalize scales to a vector matching the number of bases
    if isa(scales, Real)
        scale_vec = fill(Float64(scales), nbases)
    else
        if length(scales) >= nbases
            scale_vec = Float64.(scales[1:nbases])
        else
            # Extend with the last value
            scale_vec = vcat(Float64.(scales), fill(Float64(scales[end]), nbases - length(scales)))
        end
    end

    # Get array dimensions
    data_shape = size(field.data_c)

    # Compute cutoff indices for each dimension
    cutoffs = Int[]
    for (i, (basis, scale)) in enumerate(zip(field.domain.bases, scale_vec))
        if i <= ndim
            n = data_shape[i]
            # Cutoff: keep modes 1:cutoff, zero modes (cutoff+1):n
            cutoff = max(1, Int(floor(n * scale)))
            push!(cutoffs, min(cutoff, n))
        end
    end

    # Apply dealiasing by zeroing high-frequency modes
    # This needs to handle different basis types appropriately
    for (dim, (basis, cutoff)) in enumerate(zip(field.domain.bases, cutoffs))
        if dim > ndim
            break
        end

        n = data_shape[dim]
        if cutoff >= n
            # No dealiasing needed for this dimension
            continue
        end

        if isa(basis, RealFourier) || isa(basis, ComplexFourier)
            # For Fourier: zero out high-frequency modes
            # For real FFT: positive frequencies are at indices 1:N/2+1
            # High frequencies to zero: (cutoff+1):end
            zero_fourier_modes!(field.data_c, dim, cutoff, n)
            @debug "Dealiased Fourier dimension $dim: kept modes 1:$cutoff of $n"

        elseif isa(basis, ChebyshevT) || isa(basis, ChebyshevU) ||
               isa(basis, Jacobi) || isa(basis, Ultraspherical)
            # For polynomial bases: zero high-order coefficients
            zero_polynomial_modes!(field.data_c, dim, cutoff, n)
            @debug "Dealiased polynomial dimension $dim: kept modes 1:$cutoff of $n"
        end
    end
end

"""
    zero_fourier_modes!(data::AbstractArray, dim::Int, cutoff::Int, n::Int)

Zero out high-frequency Fourier modes beyond the cutoff in dimension `dim`.

For real FFTs, the data is typically stored as complex with size N÷2+1.
Modes to keep: 1:cutoff (DC and low frequencies)
Modes to zero: (cutoff+1):n (high frequencies)
"""
function zero_fourier_modes!(data::AbstractArray, dim::Int, cutoff::Int, n::Int)
    if cutoff >= n
        return
    end

    ndim = ndims(data)

    # Build index ranges for the high-frequency modes to zero
    # Use selectdim for dimension-agnostic indexing
    for i in (cutoff + 1):n
        indices = ntuple(d -> d == dim ? i : Colon(), ndim)
        view(data, indices...) .= zero(eltype(data))
    end
end

"""
    zero_polynomial_modes!(data::AbstractArray, dim::Int, cutoff::Int, n::Int)

Zero out high-order polynomial coefficients beyond the cutoff in dimension `dim`.

For Chebyshev/Jacobi bases, coefficients are stored in order of polynomial degree.
Modes to keep: 1:cutoff (low-order polynomials)
Modes to zero: (cutoff+1):n (high-order polynomials)
"""
function zero_polynomial_modes!(data::AbstractArray, dim::Int, cutoff::Int, n::Int)
    if cutoff >= n
        return
    end

    ndim = ndims(data)

    # Zero high-order coefficients
    for i in (cutoff + 1):n
        indices = ntuple(d -> d == dim ? i : Colon(), ndim)
        view(data, indices...) .= zero(eltype(data))
    end
end

"""
    dealias_field!(field::ScalarField)

Apply dealiasing using the field's basis dealias parameters.
"""
function dealias_field!(field::ScalarField)
    if field.domain === nothing
        return
    end

    # Get dealias scales from bases
    scales = Float64[]
    for basis in field.domain.bases
        if hasfield(typeof(basis), :meta) && hasfield(typeof(basis.meta), :dealias)
            push!(scales, Float64(basis.meta.dealias))
        else
            push!(scales, 1.0)  # No dealiasing
        end
    end

    if all(s -> s >= 1.0, scales)
        # No dealiasing needed
        return
    end

    dealias_3d!(field, scales)
end

"""
    synchronize_transforms!(transforms::Vector)

Synchronize all transform operations to ensure completion before proceeding.

This is a no-op for CPU-only execution. When GPU support is added, this would
call the appropriate GPU synchronization (e.g., `CUDA.synchronize()`) to ensure
all asynchronous GPU operations have completed.

# Arguments
- `transforms`: Vector of transform objects (currently unused)
"""
function synchronize_transforms!(transforms::Vector)
    # No-op for CPU-only mode
    # For GPU: would call CUDA.synchronize() or equivalent
    return nothing
end

function is_pencil_compatible(bases::Tuple{Vararg{Basis}})
    """Check if bases are compatible with parallel transforms (PencilArrays)"""
    ndim = length(bases)

    if ndim < 2
        return false  # Parallel transforms are for multi-dimensional problems
    end

    # Count different basis types
    fourier_count = 0
    chebyshev_count = 0
    for basis in bases
        if isa(basis, RealFourier) || isa(basis, ComplexFourier)
            fourier_count += 1
        elseif isa(basis, ChebyshevT)
            chebyshev_count += 1
        end
    end

    # Parallel transforms work for:
    # 1. At least one Fourier dimension (uses PencilFFTs)
    # 2. Pure Chebyshev multi-D (uses parallel DCT with transposes)
    return (fourier_count >= 1 || chebyshev_count >= 2) && ndim >= 2
end

function is_3d_pencil_optimal(bases::Tuple{Vararg{Basis}})
    """Check if 3D PencilFFTs would be optimal for these bases"""
    if length(bases) != 3
        return false
    end

    fourier_count = count(b -> isa(b, Union{RealFourier, ComplexFourier}), bases)

    # 3D PencilFFTs is optimal when:
    # - All 3 dimensions are Fourier (best case)
    # - 2 out of 3 dimensions are Fourier (good case)
    # - Even 1 Fourier dimension can benefit from 3D decomposition
    return fourier_count >= 1
end

# ============================================================================
# Parallel Chebyshev-Chebyshev Transform Support
# ============================================================================

"""
    ParallelChebyshevTransform

Struct for managing parallel multi-dimensional Chebyshev transforms.

For 2D Chebyshev-Chebyshev domains, the transform strategy is:
1. Start with data decomposed in last dimension (first dim is local)
2. Do DCT in first direction (local operation)
3. Transpose to make second direction local
4. Do DCT in second direction (local operation)

This mirrors how PencilFFTs handles multi-dimensional FFTs but uses DCT.
"""
mutable struct ParallelChebyshevTransform <: Transform
    ndim::Int
    bases::Tuple{Vararg{ChebyshevT}}
    axis_order::Vector{Int}  # Order of axes for transforms

    # Per-axis DCT plans
    forward_plans::Vector{Any}
    backward_plans::Vector{Any}

    # Scaling factors per axis
    forward_rescale_zero::Vector{Float64}
    forward_rescale_pos::Vector{Float64}
    backward_rescale_zero::Vector{Float64}
    backward_rescale_pos::Vector{Float64}

    # Size information
    grid_sizes::Vector{Int}
    coeff_sizes::Vector{Int}

    # Pencil configurations for different decompositions
    pencils::Vector{Any}  # PencilArrays.Pencil objects for each transpose stage
    transpose_buffers::Vector{Any}  # Pre-allocated buffers for transposes

    # MPI info
    comm::MPI.Comm
    nprocs::Int
    rank::Int

    function ParallelChebyshevTransform(bases::Tuple{Vararg{ChebyshevT}}, comm::MPI.Comm)
        ndim = length(bases)
        new(
            ndim,
            bases,
            collect(1:ndim),
            Vector{Any}(undef, ndim),
            Vector{Any}(undef, ndim),
            zeros(ndim),
            zeros(ndim),
            zeros(ndim),
            zeros(ndim),
            [b.meta.size for b in bases],
            [b.meta.size for b in bases],
            Any[],
            Any[],
            comm,
            MPI.Comm_size(comm),
            MPI.Comm_rank(comm)
        )
    end
end

"""
    setup_parallel_chebyshev_transforms!(dist::Distributor, domain::Domain, chebyshev_axes::Vector{Int})

Setup parallel transforms for multi-dimensional Chebyshev domains.
Uses pencil decomposition with transposes for efficient parallel DCT.
"""
function setup_parallel_chebyshev_transforms!(dist::Distributor, domain::Domain, chebyshev_axes::Vector{Int})
    ndim = length(domain.bases)

    if ndim < 2
        @warn "Parallel Chebyshev transforms require at least 2D domain"
        return
    end

    # Extract Chebyshev bases
    cheb_bases = Tuple(domain.bases[i] for i in chebyshev_axes)

    # Create parallel transform object
    transform = ParallelChebyshevTransform(cheb_bases, dist.comm)

    # Setup global shape
    global_shape = tuple([b.meta.size for b in domain.bases]...)

    # Setup pencil decomposition for parallel operation
    if dist.size > 1
        setup_chebyshev_pencil_decomposition!(transform, dist, global_shape, chebyshev_axes)
    end

    # Setup DCT plans for each axis
    for (local_idx, global_idx) in enumerate(chebyshev_axes)
        basis = domain.bases[global_idx]
        grid_size = basis.meta.size

        # Create DCT plans
        try
            transform.forward_plans[local_idx] = FFTW.plan_r2r(
                zeros(grid_size), FFTW.REDFT10, flags=FFTW.MEASURE
            )
            transform.backward_plans[local_idx] = FFTW.plan_r2r(
                zeros(grid_size), FFTW.REDFT01, flags=FFTW.MEASURE
            )
        catch e
            @warn "FFTW DCT plan failed for axis $global_idx: $e"
            transform.forward_plans[local_idx] = nothing
            transform.backward_plans[local_idx] = nothing
        end

        # Set scaling factors (same as serial Chebyshev)
        transform.forward_rescale_zero[local_idx] = 1.0 / grid_size / 2.0
        transform.forward_rescale_pos[local_idx] = 1.0 / grid_size
        transform.backward_rescale_zero[local_idx] = 1.0
        transform.backward_rescale_pos[local_idx] = 0.5

        transform.grid_sizes[local_idx] = grid_size
        transform.coeff_sizes[local_idx] = grid_size
    end

    push!(dist.transforms, transform)

    if dist.rank == 0
        @info "Setup parallel Chebyshev transform for $(ndim)D domain"
        @info "  Global shape: $global_shape"
        @info "  Process mesh: $(dist.mesh)"
        @info "  Chebyshev axes: $chebyshev_axes"
    end
end

"""
    setup_chebyshev_pencil_decomposition!(transform::ParallelChebyshevTransform,
                                           dist::Distributor, global_shape::Tuple,
                                           chebyshev_axes::Vector{Int})

Setup pencil decomposition for parallel Chebyshev transforms.
Creates pencil configurations for each transpose stage.
"""
function setup_chebyshev_pencil_decomposition!(transform::ParallelChebyshevTransform,
                                                dist::Distributor, global_shape::Tuple,
                                                chebyshev_axes::Vector{Int})
    ndim = length(global_shape)

    # For 2D: we need two pencil configurations
    # Pencil 1: decompose in dim 2, local in dim 1 (for x-transform)
    # Pencil 2: decompose in dim 1, local in dim 2 (for y-transform)

    if ndim == 2
        # Initialize MPI topology if not done
        if dist.mpi_topology === nothing
            try
                dist.mpi_topology = PencilArrays.MPITopology(dist.comm, (dist.size,))
            catch e
                @warn "Failed to create MPI topology: $e"
            end
        end

        if dist.mpi_topology !== nothing
            try
                # Pencil 1: decompose last dimension (y), x is local
                pencil1 = PencilArrays.Pencil(dist.mpi_topology, global_shape, (2,))
                push!(transform.pencils, pencil1)

                # Pencil 2: decompose first dimension (x), y is local
                pencil2 = PencilArrays.Pencil(pencil1; decomp_dims=(1,))
                push!(transform.pencils, pencil2)

                # Pre-allocate transpose buffers
                push!(transform.transpose_buffers, PencilArrays.PencilArray{Float64}(undef, pencil1))
                push!(transform.transpose_buffers, PencilArrays.PencilArray{Float64}(undef, pencil2))

                @debug "Created pencil decomposition for 2D Chebyshev"
            catch e
                @warn "Pencil decomposition setup failed: $e, using fallback"
                setup_chebyshev_manual_decomposition!(transform, dist, global_shape)
            end
        else
            setup_chebyshev_manual_decomposition!(transform, dist, global_shape)
        end

    elseif ndim == 3
        # For 3D: need three pencil configurations
        if dist.mpi_topology !== nothing
            try
                # Pencil 1: decompose dims (2,3), x is local
                pencil1 = PencilArrays.Pencil(dist.mpi_topology, global_shape, (2, 3))
                push!(transform.pencils, pencil1)

                # Pencil 2: decompose dims (1,3), y is local
                pencil2 = PencilArrays.Pencil(pencil1; decomp_dims=(1, 3))
                push!(transform.pencils, pencil2)

                # Pencil 3: decompose dims (1,2), z is local
                pencil3 = PencilArrays.Pencil(pencil1; decomp_dims=(1, 2))
                push!(transform.pencils, pencil3)

                # Pre-allocate buffers
                for pencil in [pencil1, pencil2, pencil3]
                    push!(transform.transpose_buffers, PencilArrays.PencilArray{Float64}(undef, pencil))
                end

                @debug "Created pencil decomposition for 3D Chebyshev"
            catch e
                @warn "3D Pencil decomposition failed: $e"
                setup_chebyshev_manual_decomposition!(transform, dist, global_shape)
            end
        else
            setup_chebyshev_manual_decomposition!(transform, dist, global_shape)
        end
    end
end

"""
    setup_chebyshev_manual_decomposition!(transform::ParallelChebyshevTransform,
                                           dist::Distributor, global_shape::Tuple)

Fallback manual decomposition when PencilArrays setup fails.
Uses simple slab decomposition with MPI_Alltoall for transposes.
"""
function setup_chebyshev_manual_decomposition!(transform::ParallelChebyshevTransform,
                                                dist::Distributor, global_shape::Tuple)
    # Store global shape for manual transpose operations
    transform.pencils = Any[global_shape]

    # Allocate send/receive buffers for all-to-all
    total_size = prod(global_shape)
    local_size = div(total_size, dist.size)

    push!(transform.transpose_buffers, zeros(Float64, local_size))
    push!(transform.transpose_buffers, zeros(Float64, local_size))

    @debug "Using manual slab decomposition for parallel Chebyshev"
end

"""
    apply_parallel_chebyshev_forward!(data::AbstractArray, transform::ParallelChebyshevTransform,
                                       dist::Distributor)

Apply forward parallel Chebyshev transform (grid → coefficients).
"""
function apply_parallel_chebyshev_forward!(data::AbstractArray, transform::ParallelChebyshevTransform,
                                            dist::Distributor)
    ndim = transform.ndim

    if dist.size == 1
        # Serial: just apply DCT along each axis
        return apply_serial_chebyshev_forward!(data, transform)
    end

    # Parallel transform with transposes
    if ndim == 2
        apply_parallel_chebyshev_forward_2d!(data, transform, dist)
    elseif ndim == 3
        apply_parallel_chebyshev_forward_3d!(data, transform, dist)
    else
        @warn "Parallel Chebyshev not implemented for $(ndim)D, using serial"
        return apply_serial_chebyshev_forward!(data, transform)
    end
end

"""
    apply_parallel_chebyshev_forward_2d!(data::AbstractArray, transform::ParallelChebyshevTransform,
                                          dist::Distributor)

Forward transform for 2D parallel Chebyshev.
Strategy:
1. Data starts decomposed in y (each proc has full x)
2. DCT in x direction (local)
3. Transpose (all-to-all) to decompose in x (each proc has full y)
4. DCT in y direction (local)

Optimized with:
- Pre-allocated workspace buffers
- Non-blocking MPI communication
- SIMD-optimized loops
"""
function apply_parallel_chebyshev_forward_2d!(data::AbstractArray, transform::ParallelChebyshevTransform,
                                               dist::Distributor)
    Nx, Ny_local = size(data)

    # Pre-allocate workspace for DCT
    temp_x = zeros(Nx)
    temp_y = zeros(Ny_local * dist.size)  # Will hold full y after transpose

    # Step 1: DCT in x direction (x is fully local)
    if transform.forward_plans[1] !== nothing
        plan = transform.forward_plans[1]
        scale_zero = transform.forward_rescale_zero[1]
        scale_pos = transform.forward_rescale_pos[1]

        @inbounds for j in 1:Ny_local
            # Extract x-line
            x_line = view(data, :, j)

            # Apply DCT
            mul!(temp_x, plan, x_line)

            # Apply scaling with SIMD
            x_line[1] = temp_x[1] * scale_zero
            @simd for i in 2:Nx
                x_line[i] = temp_x[i] * scale_pos
            end
        end
    end

    # Step 2: Transpose (decompose x, make y local)
    if length(transform.pencils) >= 2 && !isempty(transform.transpose_buffers)
        try
            # Use PencilArrays transpose
            src = transform.transpose_buffers[1]
            dest = transform.transpose_buffers[2]
            copyto!(parent(src), data)
            PencilArrays.transpose!(dest, src)
            data_transposed = parent(dest)
        catch e
            @debug "PencilArrays transpose failed, using optimized MPI: $e"
            data_transposed = optimized_manual_transpose_2d!(data, dist)
        end
    else
        data_transposed = optimized_manual_transpose_2d!(data, dist)
    end

    Nx_local, Ny = size(data_transposed)

    # Step 3: DCT in y direction (y is now fully local)
    if transform.forward_plans[2] !== nothing
        plan = transform.forward_plans[2]
        scale_zero = transform.forward_rescale_zero[2]
        scale_pos = transform.forward_rescale_pos[2]

        # Resize temp buffer if needed
        if length(temp_y) < Ny
            resize!(temp_y, Ny)
        end

        # Pre-allocate contiguous buffer for row data (FFTW requires contiguous arrays)
        y_line_buf = zeros(Ny)

        @inbounds for i in 1:Nx_local
            # Copy row to contiguous buffer (row views are strided in column-major Julia)
            for j in 1:Ny
                y_line_buf[j] = data_transposed[i, j]
            end

            # Apply DCT (requires contiguous input)
            mul!(temp_y, plan, y_line_buf)

            # Apply scaling with SIMD and write back
            data_transposed[i, 1] = temp_y[1] * scale_zero
            @simd for j in 2:Ny
                data_transposed[i, j] = temp_y[j] * scale_pos
            end
        end
    end

    # Transpose back to original decomposition
    if length(transform.pencils) >= 2 && !isempty(transform.transpose_buffers)
        try
            src = transform.transpose_buffers[2]
            dest = transform.transpose_buffers[1]
            copyto!(parent(src), data_transposed)
            PencilArrays.transpose!(dest, src)
            copyto!(data, parent(dest))
        catch e
            @debug "Transpose back failed, using optimized manual: $e"
            result = optimized_manual_transpose_2d!(data_transposed, dist)
            copyto!(data, result)
        end
    else
        result = optimized_manual_transpose_2d!(data_transposed, dist)
        copyto!(data, result)
    end
end

"""
    apply_parallel_chebyshev_backward_2d!(data::AbstractArray, transform::ParallelChebyshevTransform,
                                           dist::Distributor)

Backward transform for 2D parallel Chebyshev.
Reverse of forward transform.

Optimized with:
- Pre-allocated workspace buffers
- Non-blocking MPI communication
- SIMD-optimized loops
"""
function apply_parallel_chebyshev_backward_2d!(data::AbstractArray, transform::ParallelChebyshevTransform,
                                                dist::Distributor)
    Nx, Ny_local = size(data)

    # Pre-allocate workspace
    temp_x = zeros(Nx)
    result_x = zeros(Nx)

    # Step 1: Transpose to make y local
    if length(transform.pencils) >= 2 && !isempty(transform.transpose_buffers)
        try
            src = transform.transpose_buffers[1]
            dest = transform.transpose_buffers[2]
            copyto!(parent(src), data)
            PencilArrays.transpose!(dest, src)
            data_transposed = parent(dest)
        catch e
            data_transposed = optimized_manual_transpose_2d!(data, dist)
        end
    else
        data_transposed = optimized_manual_transpose_2d!(data, dist)
    end

    Nx_local, Ny = size(data_transposed)

    # Pre-allocate y workspace
    temp_y = zeros(Ny)
    result_y = zeros(Ny)

    # Step 2: Inverse DCT in y direction
    if transform.backward_plans[2] !== nothing
        plan = transform.backward_plans[2]
        scale_zero = transform.backward_rescale_zero[2]
        scale_pos = transform.backward_rescale_pos[2]

        @inbounds for i in 1:Nx_local
            y_line = view(data_transposed, i, :)

            # Apply scaling before transform
            temp_y[1] = y_line[1] * scale_zero
            @simd for j in 2:Ny
                temp_y[j] = y_line[j] * scale_pos
            end

            # Apply inverse DCT
            mul!(result_y, plan, temp_y)
            copyto!(y_line, result_y)
        end
    end

    # Step 3: Transpose back to make x local
    if length(transform.pencils) >= 2 && !isempty(transform.transpose_buffers)
        try
            src = transform.transpose_buffers[2]
            dest = transform.transpose_buffers[1]
            copyto!(parent(src), data_transposed)
            PencilArrays.transpose!(dest, src)
            copyto!(data, parent(dest))
        catch e
            result = optimized_manual_transpose_2d!(data_transposed, dist)
            copyto!(data, result)
        end
    else
        result = optimized_manual_transpose_2d!(data_transposed, dist)
        copyto!(data, result)
    end

    Nx, Ny_local = size(data)

    # Step 4: Inverse DCT in x direction
    if transform.backward_plans[1] !== nothing
        plan = transform.backward_plans[1]
        scale_zero = transform.backward_rescale_zero[1]
        scale_pos = transform.backward_rescale_pos[1]

        @inbounds for j in 1:Ny_local
            x_line = view(data, :, j)

            # Apply scaling before transform
            temp_x[1] = x_line[1] * scale_zero
            @simd for i in 2:Nx
                temp_x[i] = x_line[i] * scale_pos
            end

            # Apply inverse DCT
            mul!(result_x, plan, temp_x)
            copyto!(x_line, result_x)
        end
    end
end

"""
    manual_transpose_2d!(data::AbstractArray, dist::Distributor)

Manual 2D transpose using MPI_Alltoall.
Redistributes data from row-decomposition to column-decomposition or vice versa.
"""
function manual_transpose_2d!(data::AbstractArray, dist::Distributor)
    Nx, Ny_local = size(data)
    nprocs = dist.size

    # Calculate global Ny
    Ny_global = Ny_local * nprocs

    # Pack data for all-to-all
    # Each process sends a chunk of its data to every other process
    chunk_x = div(Nx, nprocs)
    send_buf = zeros(eltype(data), Nx * Ny_local)
    recv_buf = zeros(eltype(data), Nx * Ny_local)

    # Pack: reorder data so each block goes to correct destination
    idx = 1
    for dest in 0:(nprocs-1)
        x_start = dest * chunk_x + 1
        x_end = min((dest + 1) * chunk_x, Nx)
        for j in 1:Ny_local
            for i in x_start:x_end
                send_buf[idx] = data[i, j]
                idx += 1
            end
        end
    end

    # All-to-all exchange
    MPI.Alltoall!(send_buf, recv_buf, dist.comm)

    # Unpack into transposed layout
    result = zeros(eltype(data), chunk_x, Ny_global)
    idx = 1
    for src in 0:(nprocs-1)
        y_start = src * Ny_local + 1
        y_end = (src + 1) * Ny_local
        for j in y_start:y_end
            for i in 1:chunk_x
                result[i, j] = recv_buf[idx]
                idx += 1
            end
        end
    end

    return result
end

"""
    apply_serial_chebyshev_forward!(data::AbstractArray, transform::ParallelChebyshevTransform)

Serial forward Chebyshev transform for all axes (no MPI).
"""
function apply_serial_chebyshev_forward!(data::AbstractArray, transform::ParallelChebyshevTransform)
    ndim = ndims(data)

    for axis in 1:ndim
        if axis > length(transform.forward_plans) || transform.forward_plans[axis] === nothing
            continue
        end

        N = size(data, axis)
        plan = transform.forward_plans[axis]
        scale_zero = transform.forward_rescale_zero[axis]
        scale_pos = transform.forward_rescale_pos[axis]

        # Apply DCT along this axis
        for idx in CartesianIndices(selectdim(data, axis, 1))
            # Extract line along axis
            line = zeros(N)
            for k in 1:N
                line[k] = data[insert_index(idx, axis, k)...]
            end

            # Transform
            temp = plan * line

            # Scale and store back
            data[insert_index(idx, axis, 1)...] = temp[1] * scale_zero
            for k in 2:N
                data[insert_index(idx, axis, k)...] = temp[k] * scale_pos
            end
        end
    end

    return data
end

"""
    apply_serial_chebyshev_backward!(data::AbstractArray, transform::ParallelChebyshevTransform)

Serial backward Chebyshev transform for all axes (no MPI).
"""
function apply_serial_chebyshev_backward!(data::AbstractArray, transform::ParallelChebyshevTransform)
    ndim = ndims(data)

    # Transform in reverse order
    for axis in ndim:-1:1
        if axis > length(transform.backward_plans) || transform.backward_plans[axis] === nothing
            continue
        end

        N = size(data, axis)
        plan = transform.backward_plans[axis]
        scale_zero = transform.backward_rescale_zero[axis]
        scale_pos = transform.backward_rescale_pos[axis]

        # Apply inverse DCT along this axis
        for idx in CartesianIndices(selectdim(data, axis, 1))
            # Extract and scale line
            temp = zeros(N)
            temp[1] = data[insert_index(idx, axis, 1)...] * scale_zero
            for k in 2:N
                temp[k] = data[insert_index(idx, axis, k)...] * scale_pos
            end

            # Transform
            line = plan * temp

            # Store back
            for k in 1:N
                data[insert_index(idx, axis, k)...] = line[k]
            end
        end
    end

    return data
end

"""
    insert_index(idx::CartesianIndex, axis::Int, k::Int)

Helper to insert index k at position axis in a CartesianIndex.
"""
function insert_index(idx::CartesianIndex, axis::Int, k::Int)
    t = Tuple(idx)
    return tuple(t[1:axis-1]..., k, t[axis:end]...)
end

"""
    apply_parallel_chebyshev_forward_3d!(data::AbstractArray, transform::ParallelChebyshevTransform,
                                          dist::Distributor)

Forward transform for 3D parallel Chebyshev.
Full parallelization with two transpose operations.

Strategy:
1. Data starts with x local (decomposed in y,z)
2. DCT in x (local)
3. Transpose to make y local (decomposed in x,z)
4. DCT in y (local)
5. Transpose to make z local (decomposed in x,y)
6. DCT in z (local)
7. Transpose back to original decomposition
"""
function apply_parallel_chebyshev_forward_3d!(data::AbstractArray, transform::ParallelChebyshevTransform,
                                               dist::Distributor)
    Nx, Ny_local, Nz_local = size(data)

    # Step 1: DCT in x direction (x is fully local in initial pencil)
    apply_dct_along_axis!(data, 1, transform.forward_plans[1],
                          transform.forward_rescale_zero[1],
                          transform.forward_rescale_pos[1])

    # Step 2: Transpose x↔y (make y local)
    data_yz = transpose_3d_axis!(data, dist, 1, 2)  # swap decomposition from y to x

    # Step 3: DCT in y direction
    Nx_local, Ny, Nz_local2 = size(data_yz)
    apply_dct_along_axis!(data_yz, 2, transform.forward_plans[2],
                          transform.forward_rescale_zero[2],
                          transform.forward_rescale_pos[2])

    # Step 4: Transpose y↔z (make z local)
    data_xz = transpose_3d_axis!(data_yz, dist, 2, 3)  # swap decomposition from z to y

    # Step 5: DCT in z direction
    Nx_local2, Ny_local2, Nz = size(data_xz)
    apply_dct_along_axis!(data_xz, 3, transform.forward_plans[3],
                          transform.forward_rescale_zero[3],
                          transform.forward_rescale_pos[3])

    # Step 6: Transpose back to original decomposition (z→y, then y→x)
    data_yz2 = transpose_3d_axis!(data_xz, dist, 3, 2)
    data_final = transpose_3d_axis!(data_yz2, dist, 2, 1)

    # Copy result back to original array
    copyto!(data, data_final)
end

"""
    apply_parallel_chebyshev_backward_3d!(data::AbstractArray, transform::ParallelChebyshevTransform,
                                           dist::Distributor)

Backward transform for 3D parallel Chebyshev.
Reverse of forward transform.
"""
function apply_parallel_chebyshev_backward_3d!(data::AbstractArray, transform::ParallelChebyshevTransform,
                                                dist::Distributor)
    Nx, Ny_local, Nz_local = size(data)

    # Transpose to make z local
    data_xz = transpose_3d_axis!(data, dist, 1, 2)
    data_xz = transpose_3d_axis!(data_xz, dist, 2, 3)

    # Inverse DCT in z
    apply_idct_along_axis!(data_xz, 3, transform.backward_plans[3],
                           transform.backward_rescale_zero[3],
                           transform.backward_rescale_pos[3])

    # Transpose to make y local
    data_yz = transpose_3d_axis!(data_xz, dist, 3, 2)

    # Inverse DCT in y
    apply_idct_along_axis!(data_yz, 2, transform.backward_plans[2],
                           transform.backward_rescale_zero[2],
                           transform.backward_rescale_pos[2])

    # Transpose to make x local
    data_final = transpose_3d_axis!(data_yz, dist, 2, 1)

    # Inverse DCT in x
    apply_idct_along_axis!(data_final, 1, transform.backward_plans[1],
                           transform.backward_rescale_zero[1],
                           transform.backward_rescale_pos[1])

    copyto!(data, data_final)
end

"""
    apply_dct_along_axis!(data::AbstractArray, axis::Int, plan, scale_zero::Float64, scale_pos::Float64)

Apply forward DCT along specified axis with proper scaling.
Optimized for in-place operation where possible.
"""
function apply_dct_along_axis!(data::AbstractArray, axis::Int, plan, scale_zero::Float64, scale_pos::Float64)
    if plan === nothing
        return
    end

    N = size(data, axis)
    ndim = ndims(data)

    # Create index iterators for the other dimensions
    other_dims = [i for i in 1:ndim if i != axis]
    other_sizes = [size(data, d) for d in other_dims]

    # Pre-allocate workspace
    line = zeros(N)
    temp = zeros(N)

    # Iterate over all lines along the specified axis
    for idx in CartesianIndices(tuple(other_sizes...))
        # Build full index
        full_idx = Vector{Any}(undef, ndim)
        other_idx = 1
        for d in 1:ndim
            if d == axis
                full_idx[d] = Colon()
            else
                full_idx[d] = idx[other_idx]
                other_idx += 1
            end
        end

        # Extract line
        line_view = view(data, full_idx...)
        copyto!(line, line_view)

        # Apply DCT
        mul!(temp, plan, line)

        # Apply scaling
        line_view[1] = temp[1] * scale_zero
        @inbounds @simd for i in 2:N
            line_view[i] = temp[i] * scale_pos
        end
    end
end

"""
    apply_idct_along_axis!(data::AbstractArray, axis::Int, plan, scale_zero::Float64, scale_pos::Float64)

Apply inverse DCT along specified axis with proper scaling.
"""
function apply_idct_along_axis!(data::AbstractArray, axis::Int, plan, scale_zero::Float64, scale_pos::Float64)
    if plan === nothing
        return
    end

    N = size(data, axis)
    ndim = ndims(data)

    other_dims = [i for i in 1:ndim if i != axis]
    other_sizes = [size(data, d) for d in other_dims]

    # Pre-allocate workspace
    temp = zeros(N)
    result = zeros(N)

    for idx in CartesianIndices(tuple(other_sizes...))
        full_idx = Vector{Any}(undef, ndim)
        other_idx = 1
        for d in 1:ndim
            if d == axis
                full_idx[d] = Colon()
            else
                full_idx[d] = idx[other_idx]
                other_idx += 1
            end
        end

        line_view = view(data, full_idx...)

        # Apply scaling before transform
        temp[1] = line_view[1] * scale_zero
        @inbounds @simd for i in 2:N
            temp[i] = line_view[i] * scale_pos
        end

        # Apply inverse DCT
        mul!(result, plan, temp)
        copyto!(line_view, result)
    end
end

"""
    transpose_3d_axis!(data::AbstractArray, dist::Distributor, from_axis::Int, to_axis::Int)

Transpose 3D data to change which axis is local.
Uses non-blocking MPI for better performance.
"""
function transpose_3d_axis!(data::AbstractArray, dist::Distributor, from_axis::Int, to_axis::Int)
    if dist.size == 1
        return data  # No transpose needed for serial
    end

    dims = size(data)
    nprocs = dist.size

    # Calculate chunk sizes for the transpose
    from_size = dims[from_axis]
    to_size_local = dims[to_axis]
    to_size_global = to_size_local * nprocs

    chunk_from = div(from_size, nprocs)
    chunk_to = to_size_local

    # Calculate new dimensions after transpose
    new_dims = collect(dims)
    new_dims[from_axis] = chunk_from
    new_dims[to_axis] = to_size_global

    # Allocate send and receive buffers
    total_elements = prod(dims)
    send_buf = zeros(eltype(data), total_elements)
    recv_buf = zeros(eltype(data), total_elements)

    # Pack data for all-to-all
    pack_for_transpose_3d!(send_buf, data, from_axis, to_axis, nprocs)

    # Non-blocking all-to-all for overlap potential
    req = MPI.Ialltoall!(send_buf, recv_buf, dist.comm)
    MPI.Wait(req)

    # Unpack into new layout
    result = zeros(eltype(data), new_dims...)
    unpack_from_transpose_3d!(result, recv_buf, from_axis, to_axis, nprocs, dims)

    return result
end

"""
    pack_for_transpose_3d!(buf::Vector, data::AbstractArray, from_axis::Int, to_axis::Int, nprocs::Int)

Pack 3D data for MPI all-to-all transpose.
"""
function pack_for_transpose_3d!(buf::Vector, data::AbstractArray, from_axis::Int, to_axis::Int, nprocs::Int)
    dims = size(data)
    chunk_from = div(dims[from_axis], nprocs)

    idx = 1
    for dest in 0:(nprocs-1)
        # Calculate range in from_axis going to this destination
        from_start = dest * chunk_from + 1
        from_end = (dest + 1) * chunk_from

        # Pack all data in this chunk
        for k in 1:dims[3]
            for j in 1:dims[2]
                for i in 1:dims[1]
                    # Check if this index falls in the chunk for dest
                    coord = (i, j, k)[from_axis]
                    if from_start <= coord <= from_end
                        buf[idx] = data[i, j, k]
                        idx += 1
                    end
                end
            end
        end
    end
end

"""
    unpack_from_transpose_3d!(result::AbstractArray, buf::Vector, from_axis::Int, to_axis::Int, nprocs::Int, orig_dims::Tuple)

Unpack 3D data after MPI all-to-all transpose.
"""
function unpack_from_transpose_3d!(result::AbstractArray, buf::Vector, from_axis::Int, to_axis::Int, nprocs::Int, orig_dims::Tuple)
    new_dims = size(result)
    chunk_to = div(orig_dims[to_axis] * nprocs, nprocs)  # to_size_local from original

    idx = 1
    for src in 0:(nprocs-1)
        # Data from src covers a range in to_axis
        to_start = src * chunk_to + 1
        to_end = (src + 1) * chunk_to

        for k in 1:new_dims[3]
            for j in 1:new_dims[2]
                for i in 1:new_dims[1]
                    coord_to = (i, j, k)[to_axis]
                    if to_start <= coord_to <= to_end
                        result[i, j, k] = buf[idx]
                        idx += 1
                    end
                end
            end
        end
    end
end

"""
    apply_parallel_chebyshev_backward!(data::AbstractArray, transform::ParallelChebyshevTransform,
                                        dist::Distributor)

Apply backward parallel Chebyshev transform (coefficients → grid).
"""
function apply_parallel_chebyshev_backward!(data::AbstractArray, transform::ParallelChebyshevTransform,
                                             dist::Distributor)
    ndim = transform.ndim

    if dist.size == 1
        return apply_serial_chebyshev_backward!(data, transform)
    end

    if ndim == 2
        apply_parallel_chebyshev_backward_2d!(data, transform, dist)
    elseif ndim == 3
        apply_parallel_chebyshev_backward_3d!(data, transform, dist)
    else
        apply_serial_chebyshev_backward!(data, transform)
    end
end

# ============================================================================
# Optimized 2D Transforms with Buffer Reuse
# ============================================================================

"""
    TransposeBuffers2D

Pre-allocated buffers for 2D transpose operations to avoid repeated allocation.
"""
mutable struct TransposeBuffers2D
    send_buf::Vector{Float64}
    recv_buf::Vector{Float64}
    temp_array::Union{Nothing, Array{Float64}}

    function TransposeBuffers2D(size::Int)
        new(zeros(Float64, size), zeros(Float64, size), nothing)
    end
end

# Global buffer cache for transpose operations
const _TRANSPOSE_BUFFERS_2D = Dict{Tuple{Int,Int}, TransposeBuffers2D}()

"""
    get_transpose_buffers_2d(Nx::Int, Ny_local::Int)

Get or create pre-allocated buffers for 2D transpose.
"""
function get_transpose_buffers_2d(Nx::Int, Ny_local::Int)
    key = (Nx, Ny_local)
    if !haskey(_TRANSPOSE_BUFFERS_2D, key)
        total_size = Nx * Ny_local
        _TRANSPOSE_BUFFERS_2D[key] = TransposeBuffers2D(total_size)
    end
    return _TRANSPOSE_BUFFERS_2D[key]
end

"""
    optimized_manual_transpose_2d!(data::AbstractArray, dist::Distributor, buffers::TransposeBuffers2D)

Optimized 2D transpose using pre-allocated buffers and non-blocking MPI.
"""
function optimized_manual_transpose_2d!(data::AbstractArray, dist::Distributor)
    Nx, Ny_local = size(data)
    nprocs = dist.size

    # Get pre-allocated buffers
    buffers = get_transpose_buffers_2d(Nx, Ny_local)

    # Calculate global Ny
    Ny_global = Ny_local * nprocs
    chunk_x = div(Nx, nprocs)

    # Ensure buffer sizes are correct
    total_size = Nx * Ny_local
    if length(buffers.send_buf) < total_size
        resize!(buffers.send_buf, total_size)
        resize!(buffers.recv_buf, total_size)
    end

    # Pack data for all-to-all (optimized loop order)
    send_buf = buffers.send_buf
    recv_buf = buffers.recv_buf

    idx = 1
    @inbounds for dest in 0:(nprocs-1)
        x_start = dest * chunk_x + 1
        x_end = min((dest + 1) * chunk_x, Nx)
        for j in 1:Ny_local
            @simd for i in x_start:x_end
                send_buf[idx] = data[i, j]
                idx += 1
            end
        end
    end

    # Non-blocking all-to-all
    req = MPI.Ialltoall!(send_buf, recv_buf, dist.comm)
    MPI.Wait(req)

    # Allocate result if needed
    result = zeros(eltype(data), chunk_x, Ny_global)

    # Unpack (optimized loop order)
    idx = 1
    @inbounds for src in 0:(nprocs-1)
        y_start = src * Ny_local + 1
        y_end = (src + 1) * Ny_local
        for j in y_start:y_end
            @simd for i in 1:chunk_x
                result[i, j] = recv_buf[idx]
                idx += 1
            end
        end
    end

    return result
end

"""
    clear_transpose_buffers_2d!()

Clear the 2D transpose buffer cache to free memory.
"""
function clear_transpose_buffers_2d!()
    empty!(_TRANSPOSE_BUFFERS_2D)
end
