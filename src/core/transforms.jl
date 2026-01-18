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

mutable struct FourierTransform <: Transform
    plan_forward::Union{Nothing, Any}
    plan_backward::Union{Nothing, Any}
    basis::Basis
    axis::Int

    function FourierTransform(basis::Basis, axis::Int)
        new(nothing, nothing, basis, axis)
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

    empty!(dist.transforms)

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

    # Use PencilFFTs only when all axes are Fourier (pure Fourier problems)
    if ndim >= 2 && length(fourier_axes) == ndim
        if ndim == 2
            setup_pencil_fft_transforms_2d!(dist, domain, gshape, fourier_axes)
        elseif ndim == 3
            setup_pencil_fft_transforms_3d!(dist, domain, gshape, fourier_axes)
        else
            @warn "PencilFFTs not configured for $(ndim)D, falling back to FFTW"
            setup_fftw_transforms_nd!(dist, domain, fourier_axes)
        end

    elseif ndim >= 2 && length(chebyshev_axes) == ndim
        setup_parallel_chebyshev_transforms!(dist, domain, chebyshev_axes)

    else
        # Mixed basis cases or 1D/serial: set up per-axis transforms
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

    # For serial execution, use regular FFTW transforms instead of PencilFFTs
    if dist.size == 1
        @info "Serial execution detected, using FFTW transforms instead of PencilFFTs"
        for axis in fourier_axes
            basis = domain.bases[axis]
            setup_fftw_transform!(dist, basis, axis)
        end
        return
    end

    if dist.pencil_config === nothing
        setup_pencil_arrays(dist, global_shape)
    end

    # Create PencilFFT plan for the Fourier dimensions
    # PencilFFTPlan expects a tuple of transforms, one per dimension
    # Use RFFT for RealFourier bases (real-to-complex), FFT for ComplexFourier

    # Build transforms tuple based on basis types
    transform_list = []
    for (i, axis) in enumerate(fourier_axes)
        basis = domain.bases[axis]
        if isa(basis, RealFourier) && i == 1
            # First RealFourier axis uses RFFT (real-to-complex)
            push!(transform_list, PencilFFTs.Transforms.RFFT())
        else
            # ComplexFourier or subsequent axes use FFT (complex-to-complex)
            push!(transform_list, PencilFFTs.Transforms.FFT())
        end
    end
    transforms = Tuple(transform_list)

    # Create the PencilFFT plan (only for parallel execution)
    pencil = create_pencil(dist, global_shape, 1)
    fft_plan = PencilFFTs.PencilFFTPlan(pencil, transforms)

    # Store the plan in the distributor
    push!(dist.transforms, fft_plan)

    @info "Set up PencilFFT transform for axes $fourier_axes with global shape $global_shape"
    mesh_str = length(dist.mesh) >= 2 ? "$(dist.mesh[1]) × $(dist.mesh[2])" : "$(dist.mesh[1])"
    @info "Parallel decomposition: $mesh_str processes"
end

function setup_fftw_transform!(dist::Distributor, basis::Union{RealFourier, ComplexFourier}, axis::Int)
    """Setup FFTW transforms for 1D case (CPU only)."""
    transform = FourierTransform(basis, axis)
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
    """Setup CPU matrix-based Chebyshev transform matching FFTW DCT-II/DCT-III conventions"""

    # DCT-II matrix for forward transform (grid to coefficients)
    #
    # FFTW REDFT10 computes: Y_k = 2 * Σ_{j=0}^{N-1} X_j * cos(π*k*(j+0.5)/N)
    # After scaling (forward_rescale_zero = 1/(2N), forward_rescale_pos = 1/N):
    #   c_0 = Y_0 / (2N) = (1/N) * Σ X_j
    #   c_k = Y_k / N = (2/N) * Σ X_j * cos(π*k*(j+0.5)/N)  for k > 0
    #
    # Matrix must directly produce these scaled coefficients:
    forward_matrix = zeros(coeff_size, grid_size)
    for k in 0:coeff_size-1, j in 0:grid_size-1
        if k == 0
            # c_0 = (1/N) * Σ X_j
            forward_matrix[k+1, j+1] = 1.0 / grid_size
        else
            # c_k = (2/N) * Σ X_j * cos(π*k*(j+0.5)/N)
            forward_matrix[k+1, j+1] = 2.0 * cos(π * k * (j + 0.5) / grid_size) / grid_size
        end
    end

    # DCT-III matrix for backward transform (coefficients to grid)
    #
    # FFTW REDFT01 computes: Y_j = X_0 + 2 * Σ_{k=1}^{N-1} X_k * cos(π*k*(j+0.5)/N)
    # After pre-scaling (backward_rescale_zero = 1, backward_rescale_pos = 0.5):
    #   x_j = c_0 + 2 * Σ_{k=1} (c_k * 0.5) * cos(...) = c_0 + Σ_{k=1} c_k * cos(...)
    #
    # Matrix must directly produce grid values from scaled coefficients:
    backward_matrix = zeros(grid_size, coeff_size)
    for j in 0:grid_size-1, k in 0:coeff_size-1
        if k == 0
            # x_j receives c_0 with weight 1
            backward_matrix[j+1, k+1] = 1.0
        else
            # x_j receives c_k with weight cos(π*k*(j+0.5)/N)
            backward_matrix[j+1, k+1] = cos(π * k * (j + 0.5) / grid_size)
        end
    end

    transform.matrices["forward"] = sparse(forward_matrix)
    transform.matrices["backward"] = sparse(backward_matrix)

    # Set scaling factors for consistency with FFTW path
    # Even though matrix includes scaling, these are needed if _chebyshev_forward uses FFTW.r2r
    transform.forward_rescale_zero = 1.0 / grid_size / 2.0
    transform.forward_rescale_pos = 1.0 / grid_size
    transform.backward_rescale_zero = 1.0
    transform.backward_rescale_pos = 0.5

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

    # Declare variables outside try block to ensure proper scoping
    local grid_points::Vector{Float64}
    local quad_weights::Vector{Float64}

    try
        # Get Gauss-Legendre quadrature points and weights
        quadrature_success = false
        try
            gp, qw = FastGaussQuadrature.gausslegendre(grid_size)
            grid_points = Vector{Float64}(gp)
            quad_weights = Vector{Float64}(qw)
            quadrature_success = true
            @info "Using FastGaussQuadrature for Legendre transform, N=$grid_size"
        catch e
            @warn "FastGaussQuadrature not available ($e), using manual implementation"
        end

        if !quadrature_success
            gp, qw = compute_legendre_quadrature(grid_size)
            grid_points = Vector{Float64}(gp)
            quad_weights = Vector{Float64}(qw)
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

# ============================================================================
# GPU Transform Support
# ============================================================================

# Note: is_gpu_array is defined in architectures.jl

# ---------------------------------------------------------------------------
# GPU transform heuristics
# ---------------------------------------------------------------------------

const GPU_FFT_MIN_ELEMENTS = Ref(32_768)

"""
    set_gpu_fft_min_elements!(n::Integer)

Set the minimum number of elements required before GPU FFTs are attempted.
"""
function set_gpu_fft_min_elements!(n::Integer)
    GPU_FFT_MIN_ELEMENTS[] = max(1, Int(n))
    return GPU_FFT_MIN_ELEMENTS[]
end

gpu_fft_min_elements() = GPU_FFT_MIN_ELEMENTS[]

function should_use_gpu_fft(field::ScalarField, data_shape::Tuple)
    mode = gpu_fft_mode(field)
    if mode === :gpu
        return true
    elseif mode === :cpu
        return false
    end
    return prod(data_shape) >= GPU_FFT_MIN_ELEMENTS[]
end

should_use_gpu_fft(field::ScalarField) = (get_grid_data(field) !== nothing) && should_use_gpu_fft(field, size(get_grid_data(field)))

"""
    gpu_forward_transform!(field::ScalarField)

GPU-specific forward transform using CUFFT.
Returns true if GPU transform was applied, false otherwise.
"""
function gpu_forward_transform!(field::ScalarField)
    # Check if we're on GPU architecture
    arch = field.dist.architecture
    if !is_gpu(arch)
        return false
    end

    # Check if data is on GPU
    data_g = get_grid_data(field)
    if !is_gpu_array(data_g)
        return false
    end

    # GPU transform will be dispatched via extension
    # The extension overrides this function when CUDA is loaded
    @warn "GPU architecture specified but CUDA extension not loaded. Falling back to CPU." maxlog=1
    return false
end

"""
    gpu_backward_transform!(field::ScalarField)

GPU-specific backward transform using CUFFT.
Returns true if GPU transform was applied, false otherwise.
"""
function gpu_backward_transform!(field::ScalarField)
    # Check if we're on GPU architecture
    arch = field.dist.architecture
    if !is_gpu(arch)
        return false
    end

    # Check if data is on GPU
    data_c = get_coeff_data(field)
    if !is_gpu_array(data_c)
        return false
    end

    # GPU transform will be dispatched via extension
    @warn "GPU architecture specified but CUDA extension not loaded. Falling back to CPU." maxlog=1
    return false
end

# -----------------------------------------------------------------------------
# Helper utilities for GPU fallbacks
# -----------------------------------------------------------------------------

"""
    _execute_on_cpu(f, data)

Ensure `f` runs on CPU memory even when `data` lives on a GPU.
Returns the result on the original device (GPU or CPU).

Note: f is the first argument to support do-block syntax:
    _execute_on_cpu(data) do host_data
        ...
    end
"""
function _execute_on_cpu(f::Function, data::AbstractArray)
    if is_gpu_array(data)
        host_data = Array(data)
        host_result = f(host_data)
        return copy_to_device(host_result, data)
    end
    return f(data)
end

# Transform execution functions
function forward_transform!(field::ScalarField, target_layout::Symbol=:c)
    """Apply forward transform to field"""

    if field.domain === nothing
        return
    end

    ensure_layout!(field, :g)  # Start in grid space

    # Try GPU transform first if on GPU architecture
    if gpu_forward_transform!(field)
        field.current_layout = :c
        return
    end

    # Find appropriate transform
    for transform in field.dist.transforms
        if isa(transform, PencilFFTs.PencilFFTPlan)
            # Use PencilFFT for parallel 2D FFT
            set_coeff_data!(field, transform * get_grid_data(field))
            field.current_layout = :c
            return
        elseif isa(transform, ParallelChebyshevTransform)
            # Use parallel Chebyshev transform
            if get_coeff_data(field) === nothing
                set_coeff_data!(field, similar(get_grid_data(field), ComplexF64))
            end
            # Copy to temporary real array for transform
            temp_data = copy(real.(get_grid_data(field)))
            apply_parallel_chebyshev_forward!(temp_data, transform, field.dist)
            get_coeff_data(field) .= temp_data
            field.current_layout = :c
            return
        end
    end

    current = get_grid_data(field)
    for transform in field.dist.transforms
        if isa(transform, FourierTransform)
            current = _fourier_forward(current, transform)
        elseif isa(transform, ChebyshevTransform)
            current = _chebyshev_forward(current, transform)
        elseif isa(transform, LegendreTransform)
            current = _legendre_forward(current, transform)
        end
    end

    # Fallback for other transforms or missing plans
    if current === get_grid_data(field)
        set_coeff_data!(field, copy(get_grid_data(field)))
    else
        set_coeff_data!(field, current)
    end
    field.current_layout = :c
end

function _fourier_forward(data::AbstractArray, transform::FourierTransform)
    return _execute_on_cpu(data) do host_data
        # Use precomputed plan only if data size matches the plan's expected size
        # and data is real (plans are type-specific)
        if ndims(host_data) == 1 && transform.plan_forward !== nothing &&
           length(host_data) == transform.basis.meta.size && eltype(host_data) <: Real
            return transform.plan_forward * host_data
        end

        dims = (transform.axis,)
        if isa(transform.basis, RealFourier)
            # rfft requires real input; if data is already complex (from prior transforms),
            # use fft instead to avoid realfloat error
            if eltype(host_data) <: Complex
                return FFTW.fft(host_data, dims)
            end
            return FFTW.rfft(host_data, dims)
        end

        return FFTW.fft(host_data, dims)
    end
end

function apply_fourier_forward!(field::ScalarField, transform::FourierTransform)
    """Apply forward Fourier transform"""
    set_coeff_data!(field, _fourier_forward(get_grid_data(field), transform))
end

function backward_transform!(field::ScalarField, target_layout::Symbol=:g)
    """Apply backward transform to field """

    if field.domain === nothing
        return
    end

    ensure_layout!(field, :c)  # Start in coefficient space

    # Try GPU transform first if on GPU architecture
    if gpu_backward_transform!(field)
        field.current_layout = :g
        return
    end

    # Find appropriate transform
    for transform in field.dist.transforms
        if isa(transform, PencilFFTs.PencilFFTPlan)
            # Use PencilFFT for parallel 2D FFT
            set_grid_data!(field, inv(transform) * get_coeff_data(field))
            field.current_layout = :g
            return
        elseif isa(transform, ParallelChebyshevTransform)
            # Use parallel Chebyshev transform
            if get_grid_data(field) === nothing
                set_grid_data!(field, similar(get_coeff_data(field), Float64))
            end
            # Copy to temporary real array for transform
            temp_data = copy(real.(get_coeff_data(field)))
            apply_parallel_chebyshev_backward!(temp_data, transform, field.dist)
            get_grid_data(field) .= temp_data
            field.current_layout = :g
            return
        end
    end

    current = get_coeff_data(field)
    for transform in reverse(field.dist.transforms)
        if isa(transform, FourierTransform)
            current = _fourier_backward(current, transform)
        elseif isa(transform, ChebyshevTransform)
            current = _chebyshev_backward(current, transform)
        elseif isa(transform, LegendreTransform)
            current = _legendre_backward(current, transform)
        end
    end

    # Fallback for other transforms
    if current === get_coeff_data(field)
        set_grid_data!(field, copy(get_coeff_data(field)))
    else
        set_grid_data!(field, current)
    end
    field.current_layout = :g
end

function _fourier_backward(data::AbstractArray, transform::FourierTransform)
    return _execute_on_cpu(data) do host_data
        # Use precomputed plan only if data size matches the plan's expected size
        # For RealFourier, coefficient size is div(N, 2) + 1 where N is the real array size
        expected_rfft_size = isa(transform.basis, RealFourier) ? div(transform.basis.meta.size, 2) + 1 : transform.basis.meta.size
        if ndims(host_data) == 1 && transform.plan_backward !== nothing && length(host_data) == expected_rfft_size
            return transform.plan_backward * host_data
        end

        dims = (transform.axis,)
        if isa(transform.basis, RealFourier)
            # Check actual size along the transform axis to determine if rfft or fft was used
            actual_size = size(host_data, transform.axis)
            expected_rfft_coeff_size = div(transform.basis.meta.size, 2) + 1

            # If size matches rfft output, use irfft; otherwise use ifft (fft was used for complex input)
            if actual_size == expected_rfft_coeff_size
                return FFTW.irfft(host_data, transform.basis.meta.size, dims)
            else
                # fft was used (complex input case), use ifft
                return FFTW.ifft(host_data, dims)
            end
        end

        return FFTW.ifft(host_data, dims)
    end
end

function apply_fourier_backward!(field::ScalarField, transform::FourierTransform)
    """Apply backward Fourier transform"""
    set_grid_data!(field, _fourier_backward(get_coeff_data(field), transform))
end

# Axis-aware Chebyshev helpers
function _scale_along_axis!(data::AbstractArray, axis::Int, scale::AbstractVector{<:Real})
    if axis > ndims(data)
        return
    end
    shape = ntuple(i -> i == axis ? length(scale) : 1, ndims(data))
    data .*= reshape(scale, shape...)
end

function _chebyshev_forward(data::AbstractArray, transform::ChebyshevTransform)
    return _execute_on_cpu(data) do host_data
        axis = transform.axis
        if axis > ndims(host_data)
            return host_data
        end

        grid_size = size(host_data, axis)
        coeff_size = transform.coeff_size

        real_type = real(eltype(host_data))

        real_data = real.(host_data)
        temp_real = FFTW.r2r(real_data, FFTW.REDFT10, (axis,))

        scale_vec = fill(real_type(transform.forward_rescale_pos), grid_size)
        scale_vec[1] = real_type(transform.forward_rescale_zero)
        _scale_along_axis!(temp_real, axis, scale_vec)

        out_shape = ntuple(i -> i == axis ? coeff_size : size(temp_real, i), ndims(temp_real))
        out_real = zeros(real_type, out_shape)
        ncopy = min(grid_size, coeff_size)
        idx = ntuple(i -> i == axis ? (1:ncopy) : Colon(), ndims(temp_real))
        out_real[idx...] .= temp_real[idx...]

        if eltype(host_data) <: Complex
            imag_data = imag.(host_data)
            temp_imag = FFTW.r2r(imag_data, FFTW.REDFT10, (axis,))
            _scale_along_axis!(temp_imag, axis, scale_vec)

            out_imag = zeros(real_type, out_shape)
            out_imag[idx...] .= temp_imag[idx...]
            return complex.(out_real, out_imag)
        end

        return out_real
    end
end

function _chebyshev_backward(data::AbstractArray, transform::ChebyshevTransform)
    return _execute_on_cpu(data) do host_data
        axis = transform.axis
        if axis > ndims(host_data)
            return host_data
        end

        coeff_size = size(host_data, axis)
        grid_size = transform.grid_size

        real_type = real(eltype(host_data))
        scale_vec = zeros(real_type, coeff_size)
        if coeff_size > 0
            scale_vec[1] = real_type(transform.backward_rescale_zero)
            if coeff_size > 1
                max_k = min(transform.Kmax + 1, coeff_size)
                scale_vec[2:max_k] .= real_type(transform.backward_rescale_pos)
            end
        end

        scale_shape = ntuple(i -> i == axis ? coeff_size : 1, ndims(host_data))
        scaled_real = real.(host_data) .* reshape(scale_vec, scale_shape...)

        padded_shape = ntuple(i -> i == axis ? grid_size : size(host_data, i), ndims(host_data))
        padded_real = zeros(real_type, padded_shape)
        ncopy = min(coeff_size, grid_size)
        idx = ntuple(i -> i == axis ? (1:ncopy) : Colon(), ndims(host_data))
        padded_real[idx...] .= scaled_real[idx...]

        temp_real = FFTW.r2r(padded_real, FFTW.REDFT01, (axis,))

        if eltype(host_data) <: Complex
            scaled_imag = imag.(host_data) .* reshape(scale_vec, scale_shape...)
            padded_imag = zeros(real_type, padded_shape)
            padded_imag[idx...] .= scaled_imag[idx...]
            temp_imag = FFTW.r2r(padded_imag, FFTW.REDFT01, (axis,))
            return complex.(temp_real, temp_imag)
        end

        return temp_real
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

    if ndims(get_grid_data(field)) != 1 || eltype(get_grid_data(field)) <: Complex || is_gpu_array(get_grid_data(field))
        set_coeff_data!(field, _chebyshev_forward(get_grid_data(field), transform))
        return
    end

    if transform.forward_plan !== nothing
        # Use FFTW DCT-II plan for CPU with workspace buffer
        try
            # Get workspace buffer instead of allocating
            wm = get_global_workspace()
            temp_data = get_workspace!(wm, Float64, (transform.grid_size,))

            # In-place DCT using mul!
            mul!(temp_data, transform.forward_plan, get_grid_data(field))

            # Ensure output array exists
            if get_coeff_data(field) === nothing
                set_coeff_data!(field, zeros(ComplexF64, transform.coeff_size))
            end

            # Apply scaling factors for unit-amplitude normalization (in-place)
            @inbounds get_coeff_data(field)[1] = temp_data[1] * transform.forward_rescale_zero

            if transform.Kmax > 0
                scale_pos = transform.forward_rescale_pos
                @inbounds @simd for k in 1:min(transform.Kmax, transform.coeff_size-1)
                    get_coeff_data(field)[k+1] = temp_data[k+1] * scale_pos
                end
            end

            # Zero padding if coeff_size > Kmax+1
            if transform.coeff_size > transform.Kmax + 1
                @inbounds @simd for k in (transform.Kmax + 2):transform.coeff_size
                    get_coeff_data(field)[k] = 0.0
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

    if ndims(get_coeff_data(field)) != 1 || eltype(get_coeff_data(field)) <: Complex || is_gpu_array(get_coeff_data(field))
        set_grid_data!(field, _chebyshev_backward(get_coeff_data(field), transform))
        return
    end

    if transform.backward_plan !== nothing
        # Use FFTW DCT-III plan for CPU with workspace
        try
            # Get workspace buffer instead of allocating
            wm = get_global_workspace()
            temp_data = get_workspace!(wm, Float64, (transform.grid_size,))

            # Zero the workspace (needed for correct transform)
            fill!(temp_data, 0.0)

            # Apply scaling factors following Tarang resize_rescale_backward
            if length(get_coeff_data(field)) > 0
                @inbounds temp_data[1] = real(get_coeff_data(field)[1]) * transform.backward_rescale_zero
            end

            if transform.Kmax > 0
                scale_pos = transform.backward_rescale_pos
                @inbounds @simd for k in 1:min(transform.Kmax, length(get_coeff_data(field))-1)
                    temp_data[k+1] = real(get_coeff_data(field)[k+1]) * scale_pos
                end
            end

            # Ensure output array exists
            if get_grid_data(field) === nothing
                set_grid_data!(field, zeros(Float64, transform.grid_size))
            end

            # In-place backward DCT
            mul!(get_grid_data(field), transform.backward_plan, temp_data)

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
        coeff_dtype = coefficient_eltype(field.dtype)
        if get_coeff_data(field) === nothing || length(get_coeff_data(field)) != out_size || eltype(get_coeff_data(field)) != coeff_dtype
            set_coeff_data!(field, zeros(coeff_dtype, out_size))
        end
        # Ensure input grid data exists
        if get_grid_data(field) === nothing
            throw(ArgumentError("Chebyshev forward transform requires grid data for field $(field.name)"))
        end
        # In-place matrix-vector multiply
        mul!(get_coeff_data(field), mat, get_grid_data(field))
    else
        @warn "No forward matrix available for Chebyshev transform"
        if get_coeff_data(field) === nothing
            set_coeff_data!(field, copy(get_grid_data(field)))
        else
            copyto!(get_coeff_data(field), get_grid_data(field))
        end
    end
end

function apply_chebyshev_matrix_backward!(field::ScalarField, transform::ChebyshevTransform)
    """Apply backward Chebyshev transform using in-place matrix multiplication"""

    if haskey(transform.matrices, "backward")
        mat = transform.matrices["backward"]
        out_size = size(mat, 1)

        if get_coeff_data(field) === nothing
            throw(ArgumentError("Chebyshev backward transform requires coefficient data for field $(field.name)"))
        end

        # Get workspace for real part extraction
        wm = get_global_workspace()
        coeff_len = length(get_coeff_data(field))
        real_type = real(eltype(get_coeff_data(field)))
        real_coeffs = get_workspace!(wm, real_type, (coeff_len,))
        imag_coeffs = eltype(get_coeff_data(field)) <: Complex ? get_workspace!(wm, real_type, (coeff_len,)) : nothing

        # Extract real part in-place
        @inbounds @simd for i in eachindex(real_coeffs, get_coeff_data(field))
            real_coeffs[i] = real(get_coeff_data(field)[i])
            if imag_coeffs !== nothing
                imag_coeffs[i] = imag(get_coeff_data(field)[i])
            end
        end

        # Ensure output array exists
        if get_grid_data(field) === nothing || length(get_grid_data(field)) != out_size || eltype(get_grid_data(field)) != field.dtype
            set_grid_data!(field, zeros(field.dtype, out_size))
        end

        if eltype(get_grid_data(field)) <: Real
            if imag_coeffs !== nothing && any(x -> !iszero(x), imag_coeffs)
                @warn "Discarding imaginary coefficients for real Chebyshev backward transform of $(field.name)"
            end
            mul!(get_grid_data(field), mat, real_coeffs)
        else
            out_real = get_workspace!(wm, real_type, (out_size,))
            out_imag = get_workspace!(wm, real_type, (out_size,))
            mul!(out_real, mat, real_coeffs)
            if imag_coeffs === nothing
                fill!(out_imag, zero(real_type))
            else
                mul!(out_imag, mat, imag_coeffs)
            end
            get_grid_data(field) .= complex.(out_real, out_imag)
            release_workspace!(wm, out_real)
            release_workspace!(wm, out_imag)
        end

        release_workspace!(wm, real_coeffs)
        if imag_coeffs !== nothing
            release_workspace!(wm, imag_coeffs)
        end
    else
        @warn "No backward matrix available for Chebyshev transform"
        if get_grid_data(field) === nothing
            set_grid_data!(field, real.(get_coeff_data(field)))
        else
            @inbounds @simd for i in eachindex(get_grid_data(field))
                get_grid_data(field)[i] = real(get_coeff_data(field)[i])
            end
        end
    end
end

# Legendre transform application functions following Tarang JacobiMMT patterns
function _legendre_forward(data::AbstractArray, transform::LegendreTransform)
    return _execute_on_cpu(data) do host_data
        axis = transform.axis
        if axis > ndims(host_data)
            return host_data
        end

        if !haskey(transform.matrices, "forward")
            return host_data
        end

        mat = transform.matrices["forward"]
        coeff_size = size(mat, 1)
        out_shape = ntuple(i -> i == axis ? coeff_size : size(host_data, i), ndims(host_data))
        real_type = real(eltype(host_data))
        out_eltype = eltype(host_data) <: Complex ? eltype(host_data) : real_type
        out = zeros(out_eltype, out_shape)

        other_dims = Tuple(filter(i -> i != axis, 1:ndims(host_data)))

        wm = get_global_workspace()
        temp_real = get_workspace!(wm, real_type, (coeff_size,))
        temp_imag = eltype(host_data) <: Complex ? get_workspace!(wm, real_type, (coeff_size,)) : nothing

        if isempty(other_dims)
            mul!(temp_real, mat, real(host_data))
            if temp_imag === nothing
                copyto!(out, temp_real)
            else
                mul!(temp_imag, mat, imag(host_data))
                out .= complex.(temp_real, temp_imag)
            end
        else
            for (slice_in, slice_out) in zip(eachslice(host_data, dims=other_dims), eachslice(out, dims=other_dims))
                mul!(temp_real, mat, real(slice_in))
                if temp_imag === nothing
                    slice_out .= temp_real
                else
                    mul!(temp_imag, mat, imag(slice_in))
                    slice_out .= complex.(temp_real, temp_imag)
                end
            end
        end

        release_workspace!(wm, temp_real)
        if temp_imag !== nothing
            release_workspace!(wm, temp_imag)
        end

        return out
    end
end

function _legendre_backward(data::AbstractArray, transform::LegendreTransform)
    return _execute_on_cpu(data) do host_data
        axis = transform.axis
        if axis > ndims(host_data)
            return host_data
        end

        if !haskey(transform.matrices, "backward")
            return host_data
        end

        mat = transform.matrices["backward"]
        grid_size = size(mat, 1)
        out_shape = ntuple(i -> i == axis ? grid_size : size(host_data, i), ndims(host_data))
        real_type = real(eltype(host_data))
        out_eltype = eltype(host_data) <: Complex ? eltype(host_data) : real_type
        out = zeros(out_eltype, out_shape)

        other_dims = Tuple(filter(i -> i != axis, 1:ndims(host_data)))

        wm = get_global_workspace()
        temp_real = get_workspace!(wm, real_type, (grid_size,))
        temp_imag = eltype(host_data) <: Complex ? get_workspace!(wm, real_type, (grid_size,)) : nothing

        if isempty(other_dims)
            mul!(temp_real, mat, real(host_data))
            if temp_imag === nothing
                copyto!(out, temp_real)
            else
                mul!(temp_imag, mat, imag(host_data))
                out .= complex.(temp_real, temp_imag)
            end
        else
            for (slice_in, slice_out) in zip(eachslice(host_data, dims=other_dims), eachslice(out, dims=other_dims))
                mul!(temp_real, mat, real(slice_in))
                if temp_imag === nothing
                    slice_out .= temp_real
                else
                    mul!(temp_imag, mat, imag(slice_in))
                    slice_out .= complex.(temp_real, temp_imag)
                end
            end
        end

        release_workspace!(wm, temp_real)
        if temp_imag !== nothing
            release_workspace!(wm, temp_imag)
        end

        return out
    end
end

function apply_legendre_forward!(field::ScalarField, transform::LegendreTransform)
    """
    Apply forward Legendre transform (grid to coefficients) with in-place operations.

    Based on Tarang JacobiMMT.forward_matrix:
    - Uses Gauss-Legendre quadrature integration
    - Proper normalization for orthonormal Legendre expansion
    - OPTIMIZED: In-place matrix-vector multiplication
    """

    if haskey(transform.matrices, "forward")
        set_coeff_data!(field, _legendre_forward(get_grid_data(field), transform))
        @debug "Applied Legendre forward transform: grid_size=$(transform.grid_size), coeff_size=$(transform.coeff_size)"
        return
    end

    @warn "No forward matrix available for Legendre transform, using identity"
    set_coeff_data!(field, copy(get_grid_data(field)))
end

function apply_legendre_backward!(field::ScalarField, transform::LegendreTransform)
    """
    Apply backward Legendre transform (coefficients to grid) with in-place operations.

    Based on Tarang polynomial evaluation:
    - Evaluates f(x) = Σ c_n P_n(x) at Gauss-Legendre quadrature points
    - OPTIMIZED: Uses workspace buffer and in-place operations
    """

    if haskey(transform.matrices, "backward")
        set_grid_data!(field, _legendre_backward(get_coeff_data(field), transform))
        @debug "Applied Legendre backward transform: coeff_size=$(transform.coeff_size), grid_size=$(transform.grid_size)"
        return
    end

    @warn "No backward matrix available for Legendre transform, using identity"
    set_grid_data!(field, real.(get_coeff_data(field)))
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

    # Get coefficient array dimensions
    data_size = size(get_coeff_data(field))

    if axis > length(data_size)
        @warn "Axis $axis exceeds field dimensions $(length(data_size))"
        return
    end

    axis_size = data_size[axis]
    grid_size = basis.meta.size

    # Handle different Fourier representations
    if isa(basis, RealFourier)
        # Real Fourier (rfft): complex coefficients [DC, freq_1, freq_2, ..., freq_{N/2}]
        # axis_size = N/2 + 1, representing frequencies 0 to N/2
        # With scale (e.g., 2/3), keep frequencies 0 to floor(N * scale / 2)
        # This means keeping floor(N * scale / 2) + 1 modes (including DC)
        kept_modes = Int(floor(grid_size * scale / 2)) + 1

        @debug "RealFourier dealiasing: axis=$axis, grid_size=$grid_size, axis_size=$axis_size, kept_modes=$kept_modes, scale=$scale"

        if kept_modes < axis_size
            cutoff_index = kept_modes + 1

            # Zero out high-frequency modes
            indices = [1:i for i in data_size]
            indices[axis] = cutoff_index:axis_size
            get_coeff_data(field)[indices...] .= 0
        end
        
    elseif isa(basis, ComplexFourier)
        # Complex Fourier (fft): FFTW layout is [k=0, 1, ..., N/2, -(N/2-1), ..., -1]
        # - Index 1: k=0 (DC)
        # - Indices 2 to N/2+1: positive frequencies k=1 to N/2
        # - Indices N/2+2 to N: negative frequencies k=-(N/2-1) to -1
        #
        # With scale (e.g., 2/3), keep |k| <= N*scale/2
        # This means keeping modes 0 to floor(N*scale/2) and -floor(N*scale/2) to -1
        half_kept = Int(floor(grid_size * scale / 2))

        @debug "ComplexFourier dealiasing: axis=$axis, grid_size=$grid_size, axis_size=$axis_size, half_kept=$half_kept, scale=$scale"

        # Positive frequency cutoff: keep indices 1 to half_kept+1, zero half_kept+2 to N/2+1
        pos_cutoff = half_kept + 2  # First index to zero (1-indexed)
        nyquist_idx = axis_size ÷ 2 + 1

        if pos_cutoff <= nyquist_idx
            indices = [1:i for i in data_size]
            indices[axis] = pos_cutoff:nyquist_idx
            get_coeff_data(field)[indices...] .= 0
        end

        # Negative frequency cutoff: keep indices N-half_kept+1 to N, zero N/2+2 to N-half_kept
        neg_start = nyquist_idx + 1  # First negative frequency index
        neg_cutoff = axis_size - half_kept + 1  # First negative index to keep

        if neg_start < neg_cutoff
            indices = [1:i for i in data_size]
            indices[axis] = neg_start:(neg_cutoff - 1)
            get_coeff_data(field)[indices...] .= 0
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
    data_size = size(get_coeff_data(field))
    
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
        get_coeff_data(field)[indices...] .= 0
    end
end

function apply_basis_dealiasing!(field::ScalarField, basis::Legendre, axis::Int, scale::Real)
    """Apply Legendre basis dealiasing following Tarang patterns"""
    
    # Calculate cutoff mode for Legendre basis
    total_modes = basis.meta.size
    kept_modes = Int(floor(total_modes * scale))
    
    @debug "Legendre dealiasing: axis=$axis, total_modes=$total_modes, kept_modes=$kept_modes, scale=$scale"
    
    # Get coefficient array dimensions
    data_size = size(get_coeff_data(field))
    
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
        get_coeff_data(field)[indices...] .= 0
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
    data_size = size(get_coeff_data(field))
    
    if axis > length(data_size) || kept_modes >= data_size[axis]
        return
    end
    
    cutoff_index = kept_modes + 1
    
    # Zero out high modes generically
    indices = [1:i for i in data_size]
    indices[axis] = cutoff_index:data_size[axis]
    get_coeff_data(field)[indices...] .= 0
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

    # For serial execution, use regular FFTW transforms instead of PencilFFTs
    if dist.size == 1
        @info "Serial execution detected, using FFTW transforms instead of PencilFFTs for 3D"
        for axis in fourier_axes
            basis = domain.bases[axis]
            if isa(basis, RealFourier) || isa(basis, ComplexFourier)
                setup_fftw_transform!(dist, basis, axis)
            end
        end
        return
    end

    if dist.pencil_config === nothing
        # Create 3D pencil configuration
        setup_pencil_arrays_3d(dist, global_shape)
    end

    # Determine FFT transforms - PencilFFTPlan expects a tuple of transforms, one per dimension
    # Use RFFT for RealFourier bases (real-to-complex), FFT for ComplexFourier

    # Build transforms tuple based on basis types
    transform_list = []
    for (i, axis) in enumerate(fourier_axes)
        basis = domain.bases[axis]
        if isa(basis, RealFourier) && i == 1
            # First RealFourier axis uses RFFT (real-to-complex)
            push!(transform_list, PencilFFTs.Transforms.RFFT())
        else
            # ComplexFourier or subsequent axes use FFT (complex-to-complex)
            push!(transform_list, PencilFFTs.Transforms.FFT())
        end
    end
    transforms = Tuple(transform_list)

    @info "Setting up $(length(fourier_axes))D FFT for axes: $fourier_axes"

    # Create the PencilFFT plan with 3D pencil decomposition (only for parallel execution)
    pencil = create_pencil(dist, global_shape, 1)
    fft_plan = PencilFFTs.PencilFFTPlan(pencil, transforms)
    
    # Store the plan in the distributor
    push!(dist.transforms, fft_plan)
    
    @info "Set up 3D PencilFFT transform for axes $fourier_axes with global shape $global_shape"
    @info "3D parallel decomposition: $(join(dist.mesh, " × ")) processes"
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
    return create_pencil(dist, global_shape, decomp_index; dtype=dtype)
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
                set_coeff_data!(field, transform * get_grid_data(field))
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
                set_grid_data!(field, inv(transform) * get_coeff_data(field))
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

    if get_coeff_data(field) === nothing
        @warn "No coefficient data to dealias"
        return
    end

    ndim = ndims(get_coeff_data(field))
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
    data_shape = size(get_coeff_data(field))

    # Apply dealiasing for each dimension using the correct basis-aware logic
    # This delegates to apply_basis_dealiasing! which handles each basis type correctly
    for (dim, (basis, scale)) in enumerate(zip(field.domain.bases, scale_vec))
        if dim > ndim
            break
        end

        if scale >= 1.0
            continue  # No dealiasing needed
        end

        # Use the correct basis-aware dealiasing function
        apply_basis_dealiasing!(field, basis, dim, scale)
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
    count = length(send_buf) ÷ nprocs
    MPI.Alltoall!(MPI.UBuffer(send_buf, count), MPI.UBuffer(recv_buf, count), dist.comm)

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

# ============================================================================
# TransposableField Transform Planning
# ============================================================================

"""
    plan_transposable_transforms!(tf)

Create FFT plans for each layout in a TransposableField.

This function sets up FFTW/CUFFT plans for the dimension that is local
in each layout:
- ZLocal: plan FFT for z-dimension
- YLocal: plan FFT for y-dimension
- XLocal: plan FFT for x-dimension

The plans are cached in tf.fft_plans for efficient reuse.
"""
function plan_transposable_transforms!(tf)
    # Access the TransposableField's internal shapes
    arch = tf.buffers.architecture

    # Plan for each layout based on which dimension is local
    ndim = length(tf.global_shape)

    if ndim >= 3
        # ZLocal layout: z-dimension is local, plan FFT in z
        z_shape = tf.local_shapes[ZLocal]
        if !haskey(tf.fft_plans, ZLocal)
            if is_gpu(arch)
                # GPU plans will be created by TarangCUDAExt
                tf.fft_plans[ZLocal] = :gpu_pending
            else
                # CPU FFTW plan for z-dimension (dim 3)
                dummy = zeros(ComplexF64, z_shape...)
                tf.fft_plans[ZLocal] = FFTW.plan_fft(dummy, 3)
            end
        end

        # YLocal layout: y-dimension is local, plan FFT in y
        y_shape = tf.local_shapes[YLocal]
        if !haskey(tf.fft_plans, YLocal)
            if is_gpu(arch)
                tf.fft_plans[YLocal] = :gpu_pending
            else
                dummy = zeros(ComplexF64, y_shape...)
                tf.fft_plans[YLocal] = FFTW.plan_fft(dummy, 2)
            end
        end

        # XLocal layout: x-dimension is local, plan FFT in x
        x_shape = tf.local_shapes[XLocal]
        if !haskey(tf.fft_plans, XLocal)
            if is_gpu(arch)
                tf.fft_plans[XLocal] = :gpu_pending
            else
                dummy = zeros(ComplexF64, x_shape...)
                tf.fft_plans[XLocal] = FFTW.plan_fft(dummy, 1)
            end
        end

    elseif ndim == 2
        # 2D case: plan FFT for x and y dimensions
        y_shape = tf.local_shapes[YLocal]
        x_shape = tf.local_shapes[XLocal]

        if !haskey(tf.fft_plans, YLocal)
            if is_gpu(arch)
                tf.fft_plans[YLocal] = :gpu_pending
            else
                dummy = zeros(ComplexF64, y_shape...)
                tf.fft_plans[YLocal] = FFTW.plan_fft(dummy, 2)
            end
        end

        if !haskey(tf.fft_plans, XLocal)
            if is_gpu(arch)
                tf.fft_plans[XLocal] = :gpu_pending
            else
                dummy = zeros(ComplexF64, x_shape...)
                tf.fft_plans[XLocal] = FFTW.plan_fft(dummy, 1)
            end
        end
    end

    return tf
end

"""
    setup_distributed_transforms!(dist::Distributor, domain::Domain)

Setup distributed transforms using TransposableField for GPU+MPI parallelism.
This is called when using GPU architecture with MPI.
"""
function setup_distributed_transforms!(dist::Distributor, domain::Domain)
    if !is_gpu(dist.architecture) || dist.size == 1
        return  # Only needed for distributed GPU
    end

    # Create DistributedGPUConfig if needed
    if dist.distributed_gpu_config === nothing
        gshape = global_shape(domain)
        config = DistributedGPUConfig(dist.comm, gshape;
                                       cuda_aware_mpi=check_cuda_aware_mpi())
        dist.distributed_gpu_config = config
    end

    @info "Distributed transforms setup for GPU+MPI" rank=dist.rank size=dist.size
end

# ============================================================================
# Exports
# ============================================================================

# Export abstract type
export Transform

# Export transform types
export PencilFFTTransform, FourierTransform, ChebyshevTransform, LegendreTransform
export ParallelChebyshevTransform

# Export main transform planning function
export plan_transforms!

# TransposableField transform planning
export plan_transposable_transforms!, setup_distributed_transforms!

# Export forward/backward transform functions
export forward_transform!, backward_transform!
export forward_transform_3d!, backward_transform_3d!

# Export setup functions
export setup_pencil_fft_transforms_2d!, setup_pencil_fft_transforms_3d!
export setup_fftw_transform!, setup_fftw_transforms_nd!
export setup_chebyshev_transform!, setup_legendre_transform!
export setup_parallel_chebyshev_transforms!
export setup_pencil_arrays_3d, create_pencil_3d

# Export dealiasing functions
export dealias!, dealias_3d!, dealias_field!
export apply_basis_dealiasing!

# Export Legendre quadrature functions
export compute_legendre_quadrature
export evaluate_legendre_and_derivative
export build_legendre_polynomials

# Export utility functions
export get_transform_for_basis
export is_pencil_compatible, is_3d_pencil_optimal
export synchronize_transforms!

# Export transpose utilities
export transpose_3d_axis!
export get_transpose_buffers_2d, clear_transpose_buffers_2d!
export optimized_manual_transpose_2d!
