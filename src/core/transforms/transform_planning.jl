"""
    Transform Planning - Functions for planning spectral transforms

This file contains functions for setting up and planning transforms
for domains with various bases (Fourier, Chebyshev, Legendre).
"""

# Track if we've already logged transform setup (to avoid repeated messages from multiple field creations)
const _transform_setup_logged = Ref(false)

# Transform planning and execution
function plan_transforms!(dist::Distributor, domain::Domain)
    """
    Plan all transforms for a domain.

    MPI parallelization is ONLY supported for pure Fourier domains.
    Mixed-basis or non-Fourier domains require serial execution.
    """

    gshape = global_shape(domain)
    ndim = length(domain.bases)

    # Check if we already have a PencilFFT plan with matching input shape
    # If so, reuse it to ensure all fields use the same pencils
    if dist.pencil_fft_input !== nothing
        existing_gshape = dist.pencil_fft_input.size_global
        if existing_gshape == gshape && !isempty(dist.transforms)
            return  # Reuse existing plan - same configuration
        end
    end

    # Clear existing transforms for new configuration
    empty!(dist.transforms)
    dist.pencil_fft_input = nothing
    dist.pencil_fft_output = nothing
    dist.pencil_config = nothing

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

    is_pure_fourier = length(fourier_axes) == ndim

    # CRITICAL: MPI parallelization validation
    if dist.size > 1
        # 1D problems cannot be parallelized efficiently
        if ndim == 1
            error("MPI parallelization is not supported for 1D problems. " *
                  "1D FFT requires global data access. Use serial execution (1 MPI process).")
        end

        # Route based on basis composition
        has_chebyshev = !isempty(chebyshev_axes)
        has_fourier = !isempty(fourier_axes)
        has_legendre = !isempty(legendre_axes)

        if has_legendre
            error("MPI parallelization is not yet supported for Legendre bases. " *
                  "Use serial execution (1 MPI process).")
        end

        # Only 2D and 3D are supported for CPU MPI
        if ndim > 3 && !is_gpu(dist.architecture)
            error("MPI parallelization is only supported for 2D and 3D problems. " *
                  "Found $(ndim)D domain. Use serial execution or implement $(ndim)D support.")
        end

        # GPU + MPI or use_pencil_arrays=false: Use local transforms only
        # (distributed transforms via TransposableField)
        # PencilArrays/PencilFFTs doesn't support GPU, and when use_pencil_arrays=false
        # the user explicitly wants to avoid PencilArrays (e.g., for TransposableField testing)
        if is_gpu(dist.architecture) || !dist.use_pencil_arrays
            if is_gpu(dist.architecture)
                @info "GPU + MPI detected: using local transforms (use TransposableField for distributed operations)"
            else
                @info "use_pencil_arrays=false: using local transforms (use TransposableField for distributed operations)"
            end
            for (i, basis) in enumerate(domain.bases)
                if isa(basis, RealFourier) || isa(basis, ComplexFourier)
                    setup_fftw_transform!(dist, basis, i)
                end
            end
            return
        end

        if has_chebyshev && !has_fourier
            error("MPI parallelization is not supported for pure Chebyshev domains. " *
                  "Chebyshev transforms require global data (matrix multiply). " *
                  "Use serial execution (1 MPI process) for pure Chebyshev domains.")
        end

        if has_chebyshev && has_fourier
            # Mixed Fourier-Chebyshev: decompose only Fourier dimensions.
            # The Chebyshev dimension must be local (first dimension in PencilArrays convention).
            # Set up PencilFFTs for Fourier axes, local DCT for Chebyshev axes.
            @info "Mixed Fourier-Chebyshev domain with MPI: " *
                  "Fourier axes $(fourier_axes) will use PencilFFTs, " *
                  "Chebyshev axes $(chebyshev_axes) will use local DCT transforms"

            # Set up PencilFFTs for only the Fourier dimensions
            if ndim == 2
                setup_pencil_fft_transforms_2d!(dist, domain, gshape, fourier_axes)
            else
                setup_pencil_fft_transforms_3d!(dist, domain, gshape, fourier_axes)
            end

            # Set up local Chebyshev transforms for non-decomposed axes
            for axis in chebyshev_axes
                basis = domain.bases[axis]
                setup_chebyshev_transform!(dist, basis, axis)
            end
            return
        end

        # Pure Fourier: Setup PencilFFTs for 2D/3D
        if ndim == 2
            setup_pencil_fft_transforms_2d!(dist, domain, gshape, fourier_axes)
        else  # ndim == 3
            setup_pencil_fft_transforms_3d!(dist, domain, gshape, fourier_axes)
        end
        return
    end

    # Serial execution (dist.size == 1): all basis types supported
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
    # NOTE: RFFT can only be applied to the first transform dimension in PencilFFTs.
    # Case 1: First Fourier axis is RealFourier → RFFT, subsequent RealFourier → FFT (OK with warning)
    # Case 2: First Fourier axis is NOT RealFourier but later one is → ERROR (shape mismatch)
    transform_list = []
    uses_rfft = false
    first_fourier_is_real = length(fourier_axes) > 0 && isa(domain.bases[fourier_axes[1]], RealFourier)
    realfourier_warning_shown = false

    for (i, axis) in enumerate(fourier_axes)
        basis = domain.bases[axis]
        if isa(basis, RealFourier)
            if i == 1
                # First Fourier axis is RealFourier - can use RFFT (real-to-complex)
                push!(transform_list, PencilFFTs.Transforms.RFFT())
                uses_rfft = true
            elseif first_fourier_is_real
                # First axis is RealFourier (gets RFFT), subsequent RealFourier must use FFT
                # This is standard behavior - no warning needed as it's expected for multi-D RealFourier
                push!(transform_list, PencilFFTs.Transforms.FFT())
            else
                # First Fourier axis is NOT RealFourier, but this one is - ERROR
                # RFFT can only be applied to dimension 1 in PencilFFTs
                error("RealFourier basis on axis $axis cannot use RFFT because the first Fourier axis " *
                      "(axis $(fourier_axes[1])) is not RealFourier. In MPI mode with PencilFFTs, " *
                      "RFFT can only be applied to dimension 1. Using FFT would produce full complex " *
                      "arrays (size N) where RealFourier expects half-spectrum (size N/2+1). " *
                      "Please reorder your domain bases to place RealFourier first, " *
                      "or use ComplexFourier for this axis.")
            end
        else
            # ComplexFourier uses FFT (complex-to-complex)
            push!(transform_list, PencilFFTs.Transforms.FFT())
        end
    end
    transforms = Tuple(transform_list)

    # Create the PencilFFT plan (only for parallel execution)
    # RFFT expects real input, FFT expects complex input
    # If dtype is already complex, use it directly; otherwise wrap in Complex{}
    pencil_dtype = if uses_rfft
        dist.dtype
    else
        dist.dtype <: Complex ? dist.dtype : Complex{dist.dtype}
    end

    # Determine decomposition strategy based on basis types:
    # - Fourier axes: can be parallelized (PencilFFTs handles transposes)
    # - Non-Fourier axes (Chebyshev, Legendre): keep local
    #
    # Strategy:
    # - 2D all-Fourier: slab decomposition (decompose dim 2, keep dim 1 local)
    # - 2D Chebyshev-Fourier: decompose only Fourier axis
    # - 3D all-Fourier: pencil decomposition (decompose dims 2,3, keep dim 1 local)
    # - 3D with non-Fourier: decompose only Fourier axes
    #
    # For PencilFFTs, we create the pencil directly with the correct decomp_dims
    # Create pencil using default PencilArrays decomposition
    # For MPITopology{M}, PencilArrays decomposes the rightmost M dimensions by default
    # PencilFFTs handles the internal transposes needed for FFT on fully decomposed data
    #
    # For 2D data with 2D mesh: both dimensions decomposed, PencilFFTs transposes as needed
    # For 3D data with 2D mesh: dims (2,3) decomposed, dim 1 local - standard pencil FFT
    if dist.mpi_topology === nothing
        dist.mpi_topology = PencilArrays.MPITopology(dist.comm, dist.mesh)
    end
    pencil = PencilArrays.Pencil(dist.mpi_topology, global_shape)

    # Try to create PencilFFT plan
    # CRITICAL: If this fails in MPI mode, we CANNOT safely fall back to local FFTW
    # because that would compute incorrect results on decomposed data
    try
        fft_plan = PencilFFTs.PencilFFTPlan(pencil, transforms)
        push!(dist.transforms, fft_plan)

        # CRITICAL: Store the plan's input/output pencils for field allocation
        # PencilFFTs requires arrays allocated from these specific pencils
        dist.pencil_fft_input = first(fft_plan.plans).pencil_in
        dist.pencil_fft_output = last(fft_plan.plans).pencil_out
    catch e
        if dist.size > 1
            # In MPI mode, failing to create parallel FFT is a critical error
            @error "PencilFFT plan creation failed in MPI mode - cannot use local FFTW on distributed data" exception=e
            error("PencilFFT plan creation failed with $(dist.size) MPI processes. " *
                  "Local FFTW fallback would produce incorrect results. " *
                  "Please check your PencilFFTs installation or use serial execution.")
        else
            # Serial mode: local FFTW is correct
            @info "PencilFFT plan creation failed in serial mode, using FFTW transforms"
            for axis in fourier_axes
                basis = domain.bases[axis]
                setup_fftw_transform!(dist, basis, axis)
            end
            return
        end
    end

    if dist.rank == 0 && !_transform_setup_logged[]
        @info "Set up PencilFFT transform for axes $fourier_axes with global shape $global_shape"
        mesh_str = length(dist.mesh) >= 2 ? "$(dist.mesh[1]) × $(dist.mesh[2])" : "$(dist.mesh[1])"
        @info "Parallel decomposition: $mesh_str processes"
        _transform_setup_logged[] = true
    end
end

function setup_fftw_transform!(dist::Distributor, basis::Union{RealFourier, ComplexFourier}, axis::Int)
    """Setup FFTW transforms for 1D case (CPU only)."""
    transform = FourierTransform(basis, axis)
    setup_cpu_fft_transform!(transform, basis, dist.dtype)
    push!(dist.transforms, transform)
end

function setup_cpu_fft_transform!(transform::FourierTransform, basis::Union{RealFourier, ComplexFourier},
                                  dtype::Type=Float64)
    """Setup CPU FFTW transforms with the specified precision."""
    real_T = dtype <: Complex ? real(dtype) : dtype
    complex_T = Complex{real_T}

    # Create dummy arrays for planning with the correct precision
    if isa(basis, RealFourier)
        dummy_in = zeros(real_T, basis.meta.size)
        dummy_out = zeros(complex_T, div(basis.meta.size, 2) + 1)

        transform.plan_forward = FFTW.plan_rfft(dummy_in)
        transform.plan_backward = FFTW.plan_irfft(dummy_out, basis.meta.size)
    else # ComplexFourier
        dummy = zeros(complex_T, basis.meta.size)

        transform.plan_forward = FFTW.plan_fft(dummy)
        transform.plan_backward = FFTW.plan_ifft(dummy)
    end
    transform.plan_dtype = real_T
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
        # Use DCT-I (REDFT00) to match the Gauss-Lobatto grid: x_k = -cos(πk/(N-1))
        # DCT-I is its own inverse: REDFT00(REDFT00(x)) = 2(N-1)*x
        forward_plan = FFTW.plan_r2r(zeros(grid_size), FFTW.REDFT00, flags=FFTW.MEASURE)
        backward_plan = FFTW.plan_r2r(zeros(grid_size), FFTW.REDFT00, flags=FFTW.MEASURE)

        transform.forward_plan = forward_plan
        transform.backward_plan = backward_plan

        # DCT-I normalization for Gauss-Lobatto grid:
        # Forward: divide by (N-1), half-weight endpoints → c_0 = (1/(N-1)) * sum / 2
        # forward_rescale_zero handles the DC coefficient (extra /2 for endpoint)
        # forward_rescale_pos handles AC coefficients
        N = grid_size
        transform.forward_rescale_zero = N > 1 ? 0.5 / (N - 1) : 1.0   # DC: 1/(2(N-1))
        transform.forward_rescale_pos = N > 1 ? 1.0 / (N - 1) : 1.0    # AC: 1/(N-1)
        # Backward: undo endpoint halving then DCT-I, divide by 2
        transform.backward_rescale_zero = 1.0   # DC: multiply by 2 then /2 = 1
        transform.backward_rescale_pos = 1.0     # AC: no pre-scale needed, /2 after DCT

        @info "Setup FFTW-based Chebyshev transform (DCT-I, Gauss-Lobatto) for axis $axis, N=$grid_size"

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
    """Setup CPU matrix-based Chebyshev transform for Gauss-Lobatto (DCT-I) grid."""

    N = grid_size
    Nm1 = max(N - 1, 1)

    # Forward matrix: grid values at CGL nodes → Chebyshev coefficients
    # f(x) = Σ c_k T_k(x), sampled at x_j = cos(πj/(N-1))
    # c_k = (2/(N-1)) * Σ'' f_j * T_k(x_j) where '' means half-weight endpoints
    forward_matrix = zeros(coeff_size, grid_size)
    for k in 0:coeff_size-1, j in 0:grid_size-1
        val = cos(π * k * j / Nm1)
        # Endpoint half-weighting (j=0 and j=N-1)
        if j == 0 || j == N - 1
            val *= 0.5
        end
        # DC half-weighting (k=0)
        if k == 0
            forward_matrix[k+1, j+1] = val / Nm1
        else
            forward_matrix[k+1, j+1] = 2.0 * val / Nm1
        end
    end

    # Backward matrix: Chebyshev coefficients → grid values at CGL nodes
    # f_j = c_0 + Σ_{k=1}^{N-1} c_k * T_k(x_j) = Σ c_k * cos(πkj/(N-1))
    backward_matrix = zeros(grid_size, coeff_size)
    for j in 0:grid_size-1, k in 0:coeff_size-1
        backward_matrix[j+1, k+1] = cos(π * k * j / Nm1)
    end

    transform.matrices["forward"] = sparse(forward_matrix)
    transform.matrices["backward"] = sparse(backward_matrix)

    # Set scaling factors for consistency with DCT-I FFTW path
    N = grid_size
    transform.forward_rescale_zero = N > 1 ? 0.5 / (N - 1) : 1.0
    transform.forward_rescale_pos = N > 1 ? 1.0 / (N - 1) : 1.0
    transform.backward_rescale_zero = 1.0
    transform.backward_rescale_pos = 1.0

    @info "Setup CPU matrix-based Chebyshev transform (Gauss-Lobatto) for axis $axis, N=$grid_size"
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
        @error "Legendre transform setup failed" exception=(e, catch_backtrace())
        rethrow(e)
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

