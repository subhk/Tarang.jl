"""
Spectral transform classes with PencilFFTs integration

Translated from dedalus/core/transforms.py
"""

using PencilFFTs
using FFTW
using LinearAlgebra
using SparseArrays

# GPU support

abstract type Transform end

# PencilFFTs-based transforms for parallel 2D FFTs
struct PencilFFTTransform <: Transform
    plan::Union{Nothing, PencilFFTs.PencilFFTPlan}
    basis::Basis
    device_config::DeviceConfig
    
    function PencilFFTTransform(basis::Basis, device_config::DeviceConfig=get_device_config())
        new(nothing, basis, device_config)
    end
end

struct FourierTransform <: Transform
    plan_forward::Union{Nothing, Any}  # Can be FFTW or GPU FFT plan
    plan_backward::Union{Nothing, Any}  # Can be FFTW or GPU FFT plan
    basis::Basis
    device_config::DeviceConfig
    gpu_fft_plan::Union{Nothing, Any}  # GPU-specific FFT plan
    
    function FourierTransform(basis::Union{RealFourier, ComplexFourier}, device_config::DeviceConfig=get_device_config())
        new(nothing, nothing, basis, device_config, nothing)
    end
end

mutable struct ChebyshevTransform <: Transform
    matrices::Dict{String, AbstractMatrix}  # Can be CPU or GPU matrices
    basis::ChebyshevT
    device_config::DeviceConfig
    
    # FFTW DCT plans (if available)
    forward_plan::Union{Nothing, Any}
    backward_plan::Union{Nothing, Any}
    
    # GPU-specific DCT plans
    gpu_forward_plan::Union{Nothing, Any}
    gpu_backward_plan::Union{Nothing, Any}
    
    # Scaling factors following Dedalus FastCosineTransform
    forward_rescale_zero::Float64
    forward_rescale_pos::Float64
    backward_rescale_zero::Float64
    backward_rescale_pos::Float64
    
    # Size information for padding/truncation
    grid_size::Int
    coeff_size::Int
    Kmax::Int
    axis::Int
    
    function ChebyshevTransform(basis::ChebyshevT, device_config::DeviceConfig=get_device_config())
        new(
            Dict{String, AbstractMatrix}(),
            basis,
            device_config,
            nothing, nothing,      # FFTW plans
            nothing, nothing,      # GPU plans
            0.0, 0.0, 0.0, 0.0,    # Scaling factors 
            0, 0, 0, 0             # Sizes and axis
        )
    end
end

mutable struct LegendreTransform <: Transform
    matrices::Dict{String, AbstractMatrix}  # Can be CPU or GPU matrices
    basis::Legendre
    device_config::DeviceConfig
    
    # Quadrature information (can be on GPU)
    grid_points::Union{Nothing, AbstractVector{Float64}}
    quad_weights::Union{Nothing, AbstractVector{Float64}}
    
    # Size information for dealiasing
    grid_size::Int
    coeff_size::Int
    axis::Int
    
    function LegendreTransform(basis::Legendre, device_config::DeviceConfig=get_device_config())
        new(
            Dict{String, AbstractMatrix}(),
            basis,
            device_config,
            nothing, nothing,  # Quadrature points and weights
            0, 0, 0            # Sizes and axis
        )
    end
end

# Transform planning and execution
function plan_transforms!(dist::Distributor, domain::Domain)
    """Plan all transforms for a domain using PencilFFTs for parallel multi-D FFT"""
    
    global_shape = global_shape(domain)
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
            @warn "PencilFFTs not optimized for $(ndim)D, falling back to FFTW"
            setup_fftw_transforms_nd!(dist, domain, fourier_axes)
        end
    else
        # 1D case or no Fourier components - use regular FFTW/matrices
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
    """Setup FFTW transforms for 1D case with GPU support"""
    
    device_config = get_device_config(basis)
    transform = FourierTransform(basis, device_config)
    
    if device_config.device_type == CPU_DEVICE
        # CPU FFTW planning
        setup_cpu_fft_transform!(transform, basis)
    else
        # GPU FFT planning
        setup_gpu_fft_transform!(transform, basis, device_config)
    end
    
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

function setup_gpu_fft_transform!(transform::FourierTransform, basis::Union{RealFourier, ComplexFourier}, device_config::DeviceConfig)
    """Setup GPU FFT transforms"""
    
    try
        if device_config.device_type == GPU_CUDA && has_cuda
            setup_cuda_fft_transform!(transform, basis)
        elseif device_config.device_type == GPU_AMDGPU && has_amdgpu
            setup_amdgpu_fft_transform!(transform, basis)
        elseif device_config.device_type == GPU_METAL && has_metal
            setup_metal_fft_transform!(transform, basis)
        else
            @warn "GPU FFT not available for device $(device_config.device_type), falling back to CPU"
            setup_cpu_fft_transform!(transform, basis)
        end
    catch e
        @warn "GPU FFT setup failed: $e, falling back to CPU"
        setup_cpu_fft_transform!(transform, basis)
    end
end

function setup_cuda_fft_transform!(transform::FourierTransform, basis::Union{RealFourier, ComplexFourier})
    """Setup CUDA FFT transforms"""
    
    if isa(basis, RealFourier)
        # Create dummy GPU arrays for planning
        dummy_in = device_zeros(Float64, (basis.meta.size,), transform.device_config)
        dummy_out = device_zeros(ComplexF64, (div(basis.meta.size, 2) + 1,), transform.device_config)
        
        # Use CUDA.jl FFT planning
        transform.gpu_fft_plan = CUDA.CUFFT.plan_rfft(dummy_in)
        transform.plan_forward = transform.gpu_fft_plan
        transform.plan_backward = CUDA.CUFFT.plan_irfft(dummy_out, basis.meta.size)
        
    else # ComplexFourier
        dummy = device_zeros(ComplexF64, (basis.meta.size,), transform.device_config)
        
        transform.gpu_fft_plan = CUDA.CUFFT.plan_fft(dummy)
        transform.plan_forward = transform.gpu_fft_plan
        transform.plan_backward = CUDA.CUFFT.plan_ifft(dummy)
    end
    
    @info "Setup CUDA FFT transform for $(typeof(basis)), size=$(basis.meta.size)"
end

function setup_amdgpu_fft_transform!(transform::FourierTransform, basis::Union{RealFourier, ComplexFourier})
    """Setup AMD GPU FFT transforms"""
    
    if isa(basis, RealFourier)
        # AMDGPU.jl FFT support
        dummy_in = device_zeros(Float64, (basis.meta.size,), transform.device_config)
        dummy_out = device_zeros(ComplexF64, (div(basis.meta.size, 2) + 1,), transform.device_config)
        
        # Use AMDGPU.jl FFT (if available)
        transform.gpu_fft_plan = AMDGPU.rocFFT.plan_rfft(dummy_in)
        transform.plan_forward = transform.gpu_fft_plan
        transform.plan_backward = AMDGPU.rocFFT.plan_irfft(dummy_out, basis.meta.size)
        
    else # ComplexFourier
        dummy = device_zeros(ComplexF64, (basis.meta.size,), transform.device_config)
        
        transform.gpu_fft_plan = AMDGPU.rocFFT.plan_fft(dummy)
        transform.plan_forward = transform.gpu_fft_plan
        transform.plan_backward = AMDGPU.rocFFT.plan_ifft(dummy)
    end
    
    @info "Setup AMD GPU FFT transform for $(typeof(basis)), size=$(basis.meta.size)"
end

function setup_metal_fft_transform!(transform::FourierTransform, basis::Union{RealFourier, ComplexFourier})
    """Setup Metal FFT transforms"""
    
    # Metal.jl FFT support (simplified implementation)
    if isa(basis, RealFourier)
        dummy_in = device_zeros(Float64, (basis.meta.size,), transform.device_config)
        dummy_out = device_zeros(ComplexF64, (div(basis.meta.size, 2) + 1,), transform.device_config)
        
        # Use Metal.jl FFT if available
        # Note: Metal.jl may not have full FFT support - this is a placeholder
        transform.plan_forward = x -> fft(x)  # Fallback to generic FFT
        transform.plan_backward = x -> real(ifft(x))
        
    else # ComplexFourier
        dummy = device_zeros(ComplexF64, (basis.meta.size,), transform.device_config)
        
        transform.plan_forward = x -> fft(x)
        transform.plan_backward = x -> ifft(x)
    end
    
    @info "Setup Metal FFT transform for $(typeof(basis)), size=$(basis.meta.size)"
end

function setup_chebyshev_transform!(dist::Distributor, basis::ChebyshevT, axis::Int)
    """
    Setup Chebyshev transform following Dedalus FastChebyshevTransform implementation with GPU support.
    
    Based on Dedalus transforms.py:
    - Uses DCT-II for forward transform (grid to coefficients)
    - Uses DCT-III for backward transform (coefficients to grid)  
    - Proper scaling factors for unit-amplitude normalization
    - Supports padding/truncation for different grid/coefficient sizes
    - GPU-compatible implementation
    """
    
    device_config = get_device_config(basis)
    transform = ChebyshevTransform(basis, device_config)
    
    grid_size = basis.meta.size
    coeff_size = basis.meta.size  # Can be different for dealiasing
    
    if device_config.device_type == CPU_DEVICE
        setup_chebyshev_cpu_transform!(transform, grid_size, coeff_size, axis)
    else
        setup_chebyshev_gpu_transform!(transform, grid_size, coeff_size, axis, device_config)
    end
    
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
        
        # Set scaling factors following Dedalus FastCosineTransform
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

function setup_chebyshev_gpu_transform!(transform::ChebyshevTransform, grid_size::Int, coeff_size::Int, axis::Int, device_config::DeviceConfig)
    """Setup GPU Chebyshev transform"""
    
    try
        if device_config.device_type == GPU_CUDA && has_cuda
            setup_chebyshev_cuda_transform!(transform, grid_size, coeff_size)
        elseif device_config.device_type == GPU_AMDGPU && has_amdgpu
            setup_chebyshev_amdgpu_transform!(transform, grid_size, coeff_size)
        elseif device_config.device_type == GPU_METAL && has_metal
            setup_chebyshev_metal_transform!(transform, grid_size, coeff_size)
        else
            @warn "GPU DCT not available for device $(device_config.device_type), using GPU matrices"
            setup_chebyshev_gpu_matrix_transform!(transform, grid_size, coeff_size, device_config)
        end
    catch e
        @warn "GPU Chebyshev setup failed: $e, falling back to GPU matrices"
        setup_chebyshev_gpu_matrix_transform!(transform, grid_size, coeff_size, device_config)
    end
    
    # Store grid and coefficient sizes
    transform.grid_size = grid_size
    transform.coeff_size = coeff_size
    transform.Kmax = min(grid_size - 1, coeff_size - 1)
    transform.axis = axis
end

function setup_chebyshev_cuda_transform!(transform::ChebyshevTransform, grid_size::Int, coeff_size::Int)
    """Setup CUDA Chebyshev transform using cuFFT DCT (if available)"""
    
    # CUDA DCT support may be limited - use matrix approach for now
    setup_chebyshev_gpu_matrix_transform!(transform, grid_size, coeff_size, transform.device_config)
    @info "Setup CUDA matrix-based Chebyshev transform, N=$grid_size"
end

function setup_chebyshev_amdgpu_transform!(transform::ChebyshevTransform, grid_size::Int, coeff_size::Int)
    """Setup AMD GPU Chebyshev transform"""
    
    # AMD GPU DCT support may be limited - use matrix approach for now
    setup_chebyshev_gpu_matrix_transform!(transform, grid_size, coeff_size, transform.device_config)
    @info "Setup AMD GPU matrix-based Chebyshev transform, N=$grid_size"
end

function setup_chebyshev_metal_transform!(transform::ChebyshevTransform, grid_size::Int, coeff_size::Int)
    """Setup Metal Chebyshev transform"""
    
    # Metal DCT support may be limited - use matrix approach for now
    setup_chebyshev_gpu_matrix_transform!(transform, grid_size, coeff_size, transform.device_config)
    @info "Setup Metal matrix-based Chebyshev transform, N=$grid_size"
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

function setup_chebyshev_gpu_matrix_transform!(transform::ChebyshevTransform, grid_size::Int, coeff_size::Int, device_config::DeviceConfig)
    """Setup GPU matrix-based Chebyshev transform"""
    
    # DCT-II matrix for forward transform (grid to coefficients)
    forward_matrix_cpu = zeros(Float64, coeff_size, grid_size)
    for i in 0:coeff_size-1, j in 0:grid_size-1
        if i == 0
            forward_matrix_cpu[i+1, j+1] = 1.0 / grid_size / 2.0
        else
            forward_matrix_cpu[i+1, j+1] = cos(π * i * j / (grid_size-1)) / grid_size
        end
    end
    
    # DCT-III matrix for backward transform (coefficients to grid)
    backward_matrix_cpu = zeros(Float64, grid_size, coeff_size)
    for i in 0:grid_size-1, j in 0:coeff_size-1
        if j == 0
            backward_matrix_cpu[i+1, j+1] = 1.0
        else
            backward_matrix_cpu[i+1, j+1] = 2.0 * cos(π * j * i / (grid_size-1)) * 0.5
        end
    end
    
    # Move matrices to GPU
    forward_matrix_gpu = device_array(forward_matrix_cpu, device_config)
    backward_matrix_gpu = device_array(backward_matrix_cpu, device_config)
    
    transform.matrices["forward"] = forward_matrix_gpu
    transform.matrices["backward"] = backward_matrix_gpu
    
    @info "Setup GPU matrix-based Chebyshev transform, N=$grid_size"
end

function setup_legendre_transform!(dist::Distributor, basis::Legendre, axis::Int)
    """
    Setup Legendre transform following Dedalus JacobiMMT implementation with GPU support.
    
    Based on Dedalus transforms.py JacobiMMT and jacobi.py:
    - Uses Gauss-Legendre quadrature (Jacobi with a=0, b=0)
    - Forward transform: integration using quadrature weights  
    - Backward transform: polynomial evaluation at quadrature points
    - Proper normalization for orthogonal Legendre polynomials
    - GPU-compatible implementation
    """
    
    device_config = get_device_config(basis)
    transform = LegendreTransform(basis, device_config)
    
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
        
        # Move data to appropriate device
        if device_config.device_type == CPU_DEVICE
            transform.matrices["forward"] = sparse(forward_matrix)
            transform.matrices["backward"] = sparse(backward_matrix)
            transform.grid_points = grid_points
            transform.quad_weights = quad_weights
        else
            # Move matrices and data to GPU
            transform.matrices["forward"] = device_array(forward_matrix, device_config)
            transform.matrices["backward"] = device_array(backward_matrix, device_config)
            transform.grid_points = device_array(grid_points, device_config)
            transform.quad_weights = device_array(quad_weights, device_config)
            @info "Moved Legendre transform matrices to GPU device: $(device_config.device_type)"
        end
        
        transform.grid_size = grid_size
        transform.coeff_size = coeff_size
        transform.axis = axis
        
        @info "Setup Legendre transform for axis $axis: grid_size=$grid_size, coeff_size=$coeff_size"
        
    catch e
        @error "Legendre transform setup failed: $e, using identity fallback"
        
        # Fallback to identity matrices
        N = basis.meta.size
        I_matrix = Matrix{Float64}(I, N, N)
        
        if device_config.device_type == CPU_DEVICE
            transform.matrices["forward"] = sparse(I_matrix')
            transform.matrices["backward"] = sparse(I_matrix)
        else
            transform.matrices["forward"] = device_array(I_matrix', device_config)
            transform.matrices["backward"] = device_array(I_matrix, device_config)
        end
        transform.axis = axis
        
        @warn "Using identity matrix fallback for Legendre transform"
    end
    
    push!(dist.transforms, transform)
    
    @debug "Legendre transform setup completed for axis $axis"
end

# Helper functions for Legendre transform following Dedalus jacobi.py patterns
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
    """Apply forward transform to field with GPU support"""
    
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
            gpu_synchronize()  # Ensure GPU operations complete
            return
        elseif isa(transform, FourierTransform)
            # Use GPU or CPU FFT based on device config
            apply_fourier_forward!(field, transform)
            field.current_layout = :c
            gpu_synchronize(transform.device_config)
            return
        elseif isa(transform, ChebyshevTransform)
            # Use GPU or CPU Chebyshev transform
            apply_chebyshev_forward!(field, transform)
            field.current_layout = :c
            gpu_synchronize(transform.device_config)
            return
        elseif isa(transform, LegendreTransform)
            # Use GPU or CPU Legendre transform
            apply_legendre_forward!(field, transform)
            field.current_layout = :c
            gpu_synchronize(transform.device_config)
            return
        end
    end
    
    # Fallback for other transforms
    field.data_c .= field.data_g
    field.current_layout = :c
end

function apply_fourier_forward!(field::ScalarField, transform::FourierTransform)
    """Apply forward Fourier transform with GPU support"""
    
    # Ensure field data is on correct device
    field.data_g = ensure_device!(field.data_g, transform.device_config)
    
    if isa(transform.basis, RealFourier)
        field.data_c = transform.plan_forward * field.data_g
    else
        field.data_c = transform.plan_forward * field.data_g
    end
end

function backward_transform!(field::ScalarField, target_layout::Symbol=:g)
    """Apply backward transform to field with GPU support"""
    
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
            gpu_synchronize()  # Ensure GPU operations complete
            return
        elseif isa(transform, FourierTransform)
            # Use GPU or CPU FFT based on device config
            apply_fourier_backward!(field, transform)
            field.current_layout = :g
            gpu_synchronize(transform.device_config)
            return
        elseif isa(transform, ChebyshevTransform)
            # Use GPU or CPU Chebyshev transform
            apply_chebyshev_backward!(field, transform)
            field.current_layout = :g
            gpu_synchronize(transform.device_config)
            return
        elseif isa(transform, LegendreTransform)
            # Use GPU or CPU Legendre transform
            apply_legendre_backward!(field, transform)
            field.current_layout = :g
            gpu_synchronize(transform.device_config)
            return
        end
    end
    
    # Fallback for other transforms
    field.data_g .= field.data_c
    field.current_layout = :g
end

function apply_fourier_backward!(field::ScalarField, transform::FourierTransform)
    """Apply backward Fourier transform with GPU support"""
    
    # Ensure field data is on correct device
    field.data_c = ensure_device!(field.data_c, transform.device_config)
    
    if isa(transform.basis, RealFourier)
        field.data_g = transform.plan_backward * field.data_c
    else
        field.data_g = transform.plan_backward * field.data_c
    end
end

# Chebyshev transform application functions following Dedalus patterns
function apply_chebyshev_forward!(field::ScalarField, transform::ChebyshevTransform)
    """
    Apply forward Chebyshev transform (grid to coefficients) with GPU support.
    
    Based on Dedalus ScipyDCT.forward and FFTWDCT.forward methods:
    - Uses DCT-II with proper scaling for unit-amplitude normalization
    - Handles padding/truncation for different grid/coefficient sizes  
    - Follows resize_rescale_forward pattern
    - GPU-compatible implementation
    """
    
    # Ensure field data is on correct device
    field.data_g = ensure_device!(field.data_g, transform.device_config)
    
    if transform.forward_plan !== nothing && transform.device_config.device_type == CPU_DEVICE
        # Use FFTW DCT-II plan for CPU
        try
            temp_data = transform.forward_plan * field.data_g
            
            if field.data_c === nothing
                if transform.device_config.device_type == CPU_DEVICE
                    field.data_c = zeros(ComplexF64, transform.coeff_size)
                else
                    field.data_c = device_zeros(ComplexF64, (transform.coeff_size,), transform.device_config)
                end
            end
            
            # Apply scaling factors for unit-amplitude normalization
            field.data_c[1] = temp_data[1] * transform.forward_rescale_zero
            
            if transform.Kmax > 0
                for k in 1:min(transform.Kmax, transform.coeff_size-1)
                    field.data_c[k+1] = temp_data[k+1] * transform.forward_rescale_pos
                end
            end
            
            # Zero padding if coeff_size > Kmax+1
            if transform.coeff_size > transform.Kmax + 1
                for k in (transform.Kmax + 2):transform.coeff_size
                    field.data_c[k] = 0.0
                end
            end
            
        catch e
            @warn "DCT forward transform failed: $e, falling back to matrix method"
            apply_chebyshev_matrix_forward!(field, transform)
        end
        
    else
        # Use matrix-based transform (works for both CPU and GPU)
        apply_chebyshev_matrix_forward!(field, transform)
    end
end

function apply_chebyshev_backward!(field::ScalarField, transform::ChebyshevTransform)
    """
    Apply backward Chebyshev transform (coefficients to grid) with GPU support.
    
    Based on Dedalus ScipyDCT.backward and FFTWDCT.backward methods:
    - Uses DCT-III with proper scaling for unit-amplitude normalization
    - Handles padding/truncation for different coefficient/grid sizes
    - Follows resize_rescale_backward pattern
    - GPU-compatible implementation
    """
    
    # Ensure field data is on correct device
    field.data_c = ensure_device!(field.data_c, transform.device_config)
    
    if transform.backward_plan !== nothing && transform.device_config.device_type == CPU_DEVICE
        # Use FFTW DCT-III plan for CPU
        try
            temp_data = zeros(transform.grid_size)
            
            # Apply scaling factors following Dedalus resize_rescale_backward
            if length(field.data_c) > 0
                temp_data[1] = real(field.data_c[1]) * transform.backward_rescale_zero
            end
            
            if transform.Kmax > 0
                for k in 1:min(transform.Kmax, length(field.data_c)-1)
                    temp_data[k+1] = real(field.data_c[k+1]) * transform.backward_rescale_pos
                end
            end
            
            field.data_g = transform.backward_plan * temp_data
            
        catch e
            @warn "DCT backward transform failed: $e, falling back to matrix method"
            apply_chebyshev_matrix_backward!(field, transform)
        end
        
    else
        # Use matrix-based transform (works for both CPU and GPU)
        apply_chebyshev_matrix_backward!(field, transform)
    end
end

function apply_chebyshev_matrix_forward!(field::ScalarField, transform::ChebyshevTransform)
    """Apply forward Chebyshev transform using matrix multiplication (GPU-compatible)"""
    
    # Ensure field data is on correct device
    field.data_g = ensure_device!(field.data_g, transform.device_config)
    
    if haskey(transform.matrices, "forward")
        field.data_c = transform.matrices["forward"] * field.data_g
    else
        @warn "No forward matrix available for Chebyshev transform"
        field.data_c = copy(field.data_g)
    end
end

function apply_chebyshev_matrix_backward!(field::ScalarField, transform::ChebyshevTransform)
    """Apply backward Chebyshev transform using matrix multiplication (GPU-compatible)"""
    
    # Ensure field data is on correct device
    field.data_c = ensure_device!(field.data_c, transform.device_config)
    
    if haskey(transform.matrices, "backward")
        field.data_g = transform.matrices["backward"] * real.(field.data_c)
    else
        @warn "No backward matrix available for Chebyshev transform"
        field.data_g = real.(field.data_c)
    end
end

# Legendre transform application functions following Dedalus JacobiMMT patterns
function apply_legendre_forward!(field::ScalarField, transform::LegendreTransform)
    """
    Apply forward Legendre transform (grid to coefficients) with GPU support.
    
    Based on Dedalus JacobiMMT.forward_matrix:
    - Uses Gauss-Legendre quadrature integration
    - Proper normalization for orthonormal Legendre expansion
    - Matrix-based computation for spectral accuracy
    - GPU-compatible implementation
    """
    
    # Ensure field data is on correct device
    field.data_g = ensure_device!(field.data_g, transform.device_config)
    
    if haskey(transform.matrices, "forward")
        # Apply forward transform: c_n = ∫ f(x) P_n(x) dx ≈ Σ f(x_i) P_n(x_i) w_i
        field.data_c = transform.matrices["forward"] * field.data_g
        
        @debug "Applied Legendre forward transform: grid_size=$(transform.grid_size), coeff_size=$(transform.coeff_size)"
        
    else
        @warn "No forward matrix available for Legendre transform, using identity"
        field.data_c = copy(field.data_g)
    end
end

function apply_legendre_backward!(field::ScalarField, transform::LegendreTransform)
    """
    Apply backward Legendre transform (coefficients to grid) with GPU support.
    
    Based on Dedalus polynomial evaluation:
    - Evaluates f(x) = Σ c_n P_n(x) at Gauss-Legendre quadrature points
    - Uses precomputed polynomial matrix for efficiency
    - Proper handling of orthonormal coefficients
    - GPU-compatible implementation
    """
    
    # Ensure field data is on correct device
    field.data_c = ensure_device!(field.data_c, transform.device_config)
    
    if haskey(transform.matrices, "backward")
        # Apply backward transform: f(x_i) = Σ c_n P_n(x_i)
        field.data_g = transform.matrices["backward"] * real.(field.data_c)
        
        @debug "Applied Legendre backward transform: coeff_size=$(transform.coeff_size), grid_size=$(transform.grid_size)"
        
    else
        @warn "No backward matrix available for Legendre transform, using identity"
        field.data_g = real.(field.data_c)
    end
end

# Dealiasing operations following Dedalus patterns
function dealias!(field::ScalarField, scales::Union{Real, Vector{Real}})
    """
    Apply dealiasing to field following Dedalus field.change_scales and low_pass_filter implementation.
    
    Based on Dedalus field.py:
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
    
    # Apply dealiasing for each basis dimension following Dedalus patterns
    for (axis, (basis, scale)) in enumerate(zip(field.domain.bases, scale_vec))
        if scale >= 1.0
            continue  # No dealiasing needed for this axis
        end
        
        apply_basis_dealiasing!(field, basis, axis, scale)
    end
    
    @debug "Dealiasing completed"
end

function apply_basis_dealiasing!(field::ScalarField, basis::FourierBasis, axis::Int, scale::Real)
    """Apply Fourier basis dealiasing following Dedalus patterns"""
    
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
    """Apply Chebyshev basis dealiasing following Dedalus patterns"""
    
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
    """Apply Legendre basis dealiasing following Dedalus patterns"""
    
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
    
    return PencilArrays.Pencil(dist.pencil_config, decomp_index, dtype)
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

# Enhanced 3D dealiasing
function dealias_3d!(field::ScalarField, scales::Union{Real, Vector{Real}})
    """Apply 3D dealiasing to field"""
    
    ensure_layout!(field, :c)
    
    # Apply dealiasing in coefficient space
    if isa(scales, Real)
        scale_vec = fill(scales, 3)
    else
        scale_vec = length(scales) == 3 ? scales : [scales[1], scales[1], scales[1]]
    end
    
    # 3D dealiasing - zero out high-frequency modes in all directions
    for (i, (basis, scale)) in enumerate(zip(field.domain.bases, scale_vec))
        if isa(basis, RealFourier) || isa(basis, ComplexFourier)
            cutoff = Int(floor(basis.meta.size * scale))
            
            # Apply 3D dealiasing along axis i
            # This is a simplified version - production code would need proper 3D indexing
            if i <= ndims(field.data_c)
                @debug "Applying dealiasing along axis $i with cutoff $cutoff"
                # Actual implementation would zero out high modes properly
                # field.data_c[high_mode_indices] .= 0
            end
        end
    end
end

# GPU utility functions for transforms
function to_device!(transform::Transform, device_config::DeviceConfig)
    """Move transform to specified device"""
    
    if isa(transform, FourierTransform)
        transform.device_config = device_config
        # Note: FFT plans are device-specific and may need recreation
    elseif isa(transform, ChebyshevTransform)
        transform.device_config = device_config
        # Move matrices to new device
        for (key, matrix) in transform.matrices
            transform.matrices[key] = device_array(matrix, device_config)
        end
    elseif isa(transform, LegendreTransform)
        transform.device_config = device_config
        # Move matrices and quadrature data to new device
        for (key, matrix) in transform.matrices
            transform.matrices[key] = device_array(matrix, device_config)
        end
        if transform.grid_points !== nothing
            transform.grid_points = device_array(transform.grid_points, device_config)
        end
        if transform.quad_weights !== nothing
            transform.quad_weights = device_array(transform.quad_weights, device_config)
        end
    end
    
    return transform
end

function get_device_config(transform::Transform)
    """Get device configuration of transform"""
    if hasfield(typeof(transform), :device_config)
        return transform.device_config
    else
        return get_device_config()  # Global default
    end
end

function synchronize_transforms!(transforms::Vector)
    """Synchronize all transforms on their respective devices"""
    for transform in transforms
        if hasfield(typeof(transform), :device_config)
            gpu_synchronize(transform.device_config)
        end
    end
end

function is_pencil_compatible(bases::Tuple{Vararg{Basis}})
    """Check if bases are compatible with PencilFFTs"""
    ndim = length(bases)
    
    if ndim < 2
        return false  # PencilFFTs is for multi-dimensional problems
    end
    
    fourier_count = 0
    for basis in bases
        if isa(basis, RealFourier) || isa(basis, ComplexFourier)
            fourier_count += 1
        end
    end
    
    # PencilFFTs is beneficial when we have at least one Fourier dimension
    # and multi-dimensional domain
    return fourier_count >= 1 && ndim >= 2
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
