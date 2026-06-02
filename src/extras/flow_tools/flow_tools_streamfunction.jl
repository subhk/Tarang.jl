# Streamfunction, velocity, and SQG helpers used by flow analysis tools.

# Flow visualization helpers
"""
    Calculate streamfunction for 2D incompressible flow.

    Solves the Poisson equation ∇²ψ = ω to obtain streamfunction from vorticity.
    Based on Tarang LBVP patterns for Poisson equation solving.

    For incompressible 2D flow: u = ∂ψ/∂y, v = -∂ψ/∂x
    Vorticity: ω = ∂v/∂x - ∂u/∂y = ∇²ψ

    Args:
        velocity: 2D VectorField
        boundary_condition: :periodic, :no_slip, or :free_slip
        gauge_condition: Apply ∫ψ dA = 0 constraint

    Returns:
        ScalarField containing stream"""
function streamfunction(velocity::VectorField; boundary_condition::Symbol=:periodic, gauge_condition::Bool=true)

    if velocity.coordsys.dim != 2
        throw(ArgumentError("Streamfunction calculation requires 2D velocity field"))
    end

    # Calculate vorticity: ω = ∇ × u (2D curl)
    vorticity_field = evaluate_operator(curl(velocity))

    # Determine solution method based on basis types
    fourier_bases = get_fourier_basis_info(velocity.bases)

    if all_periodic_fourier(fourier_bases)
        # Pure Fourier bases - use direct spectral inversion
        return streamfunction_spectral_invert(vorticity_field, gauge_condition)
    else
        # Mixed/bounded bases - use boundary value problem solver
        return streamfunction_bvp_solve(vorticity_field, boundary_condition, gauge_condition)
    end
end

"""Extract information about Fourier vs non-Fourier bases.
Accepts a Tuple (e.g. `field.bases`) or a Vector."""
function get_fourier_basis_info(bases::Union{Tuple, AbstractVector})
    fourier_info = []

    for (i, basis) in enumerate(bases)
        is_fourier = isa(basis, RealFourier) || isa(basis, ComplexFourier)
        push!(fourier_info, (index=i, is_fourier=is_fourier, basis=basis))
    end

    return fourier_info
end

"""Check if all bases are periodic Fourier"""
function all_periodic_fourier(fourier_info::Vector)
    return all(info.is_fourier for info in fourier_info)
end

"""
    Direct spectral inversion for periodic domains: ψ̂(k) = -ω̂(k)/|k|²
    Based on Tarang spectral Poisson inversion patterns.
    """
function streamfunction_spectral_invert(vorticity::ScalarField, apply_gauge::Bool=true)

    # Ensure vorticity is in spectral space
    ensure_layout!(vorticity, :c)

    # Create streamfunction field
    streamfunction_field = ScalarField(vorticity.dist, "streamfunction", vorticity.bases, vorticity.dtype)
    ensure_layout!(streamfunction_field, :c)

    # Get wavenumber information for Poisson inversion
    kx_grid, ky_grid = get_2d_wavenumber_grids(vorticity)
    k_squared = kx_grid.^2 .+ ky_grid.^2

    # Perform spectral inversion: ψ̂ = -ω̂/k²
    vorticity_spec = get_coeff_data(vorticity)
    streamfunction_spec = similar(vorticity_spec)

    for idx in CartesianIndices(vorticity_spec)
        k2 = k_squared[idx]

        if k2 > 1e-12  # Avoid division by zero
            # Spectral Poisson inversion: ψ̂ = -ω̂/k²
            streamfunction_spec[idx] = -vorticity_spec[idx] / k2
        else
            # Handle k=0 mode (constant/mean mode)
            if apply_gauge
                # Gauge condition: set mean to zero
                streamfunction_spec[idx] = 0.0
            else
                # Keep arbitrary constant (may need external constraint)
                streamfunction_spec[idx] = 0.0
            end
        end
    end

    # Set spectral coefficients
    get_coeff_data(streamfunction_field) .= streamfunction_spec

    # NOTE: MPI reduction is NOT needed here.
    # Each rank computes its local spectral coefficients from local vorticity.
    # The spectral Poisson inversion is fully local (no cross-rank dependencies).
    # Previous code incorrectly used MPI.Allreduce! which would corrupt the data.

    return streamfunction_field
end

"""
    Get 2D wavenumber grids for spectral operations.
    Returns properly scaled kx, ky grids.

    CRITICAL: For MPI/PencilArrays, uses global wavenumber indices with proper offsets.
    Each rank generates wavenumbers corresponding to its local portion of global spectrum.
    """
function get_2d_wavenumber_grids(field::ScalarField)

    # Extract domain size from field bases
    if field.domain !== nothing && length(field.domain.bases) >= 2
        Lx = field.domain.bases[1].meta.bounds[2] - field.domain.bases[1].meta.bounds[1]
        Ly = field.domain.bases[2].meta.bounds[2] - field.domain.bases[2].meta.bounds[1]
    else
        # Fallback for fields without domain info
        Lx = 2π
        Ly = 2π
        @warn "get_2d_wavenumber_grids: domain bounds not available, assuming Lx=Ly=2π" maxlog=1
    end

    # Get spectral shape and MPI offsets
    ensure_layout!(field, :c)
    coeff_data = get_coeff_data(field)
    local_nx, local_ny = size(coeff_data)[1:2]

    # CRITICAL FIX: Get global shape and local offsets for MPI
    # For distributed data, wavenumber indices must account for rank's offset in global spectrum
    use_mpi = field.dist.use_pencil_arrays && field.dist.size > 1
    if use_mpi
        # Get global shape for proper wavenumber indexing
        global_shape = _get_global_coeff_shape_internal(field)
        global_nx, global_ny = global_shape[1:2]

        # Get this rank's offset in global wavenumber space
        offsets = _get_pencil_array_offsets_internal(field)
        offset_x, offset_y = offsets[1:2]
    else
        global_nx, global_ny = local_nx, local_ny
        offset_x, offset_y = 0, 0
    end

    # Generate wavenumber grids based on basis types
    bases = field.bases

    # X-direction wavenumbers (with proper offset for MPI)
    if isa(bases[1], RealFourier)
        # RFFT: wavenumbers are 0, 1, 2, ..., N/2
        kx_indices = offset_x:(offset_x + local_nx - 1)
        kx_1d = 2π / Lx * kx_indices
    else  # ComplexFourier
        # FFT: wavenumbers are -N/2, ..., -1, 0, 1, ..., N/2-1 (after fftshift)
        # For distributed FFT, need to map local indices to global wavenumbers
        global_k_indices = fftshift(-global_nx ÷ 2:(global_nx ÷ 2 - 1))
        local_k_indices = global_k_indices[(offset_x + 1):(offset_x + local_nx)]
        kx_1d = 2π / Lx * local_k_indices
    end

    # Y-direction wavenumbers (with proper offset for MPI)
    # Only the FIRST real-Fourier axis is stored as a half-spectrum rfft
    # (monotonic 0..N/2). A RealFourier axis that is NOT the first real axis
    # (e.g. y when x is also RealFourier) is a full complex FFT and must use
    # fft-frequency ordering [0,1,…,N/2-1,-N/2,…,-1] — same as ComplexFourier.
    if isa(bases[2], RealFourier) && _is_first_real_fourier_axis(bases, 2)
        ky_indices = offset_y:(offset_y + local_ny - 1)
        ky_1d = 2π / Ly * ky_indices
    else  # ComplexFourier, or a non-first RealFourier axis (full complex FFT)
        global_k_indices = fftshift(-global_ny ÷ 2:(global_ny ÷ 2 - 1))
        local_k_indices = global_k_indices[(offset_y + 1):(offset_y + local_ny)]
        ky_1d = 2π / Ly * local_k_indices
    end

    # Create 2D grids
    kx_grid = reshape(collect(kx_1d), length(kx_1d), 1)
    ky_grid = reshape(collect(ky_1d), 1, length(ky_1d))

    return kx_grid, ky_grid
end

"""
    _get_global_coeff_shape_internal(field::ScalarField)

Internal helper to get global coefficient shape. Similar to _get_global_coeff_shape
but works with ScalarField directly.
"""
function _get_global_coeff_shape_internal(field::ScalarField)
    coeff_data = get_coeff_data(field)
    if coeff_data === nothing
        error("_get_global_coeff_shape_internal: coefficient data is nothing")
    end

    if field.dist.use_pencil_arrays && field.dist.size > 1
        try
            if applicable(PencilArrays.size_global, coeff_data)
                return Tuple(PencilArrays.size_global(coeff_data))
            end
        catch
            # Fallback
        end
    end
    return size(coeff_data)
end

"""
    _get_pencil_array_offsets_internal(field::ScalarField)

Internal helper to get PencilArray offsets. Similar to _get_pencil_array_offsets
but works with ScalarField directly.
"""
function _get_pencil_array_offsets_internal(field::ScalarField)
    coeff_data = get_coeff_data(field)
    if coeff_data === nothing
        error("_get_pencil_array_offsets_internal: coefficient data is nothing")
    end

    if field.dist.use_pencil_arrays && field.dist.size > 1
        try
            if hasproperty(coeff_data, :pencil)
                axes = PencilArrays.axes_local(coeff_data)
                return Tuple(first(ax) - 1 for ax in axes)
            elseif applicable(PencilArrays.axes_local, coeff_data)
                axes = PencilArrays.axes_local(coeff_data)
                return Tuple(first(ax) - 1 for ax in axes)
            end
        catch e
            # CRITICAL: Don't silently fall back - error for MPI runs
            error("_get_pencil_array_offsets_internal: failed to get offsets for MPI data: $e")
        end
    end
    return Tuple(zeros(Int, ndims(coeff_data)))
end

"""
    Solve streamfunction BVP for bounded/mixed domains.

    Solves ∇²ψ = ω with appropriate boundary conditions using Jacobi iteration.
    This is an iterative solver suitable for domains with physical boundaries.
    """
function streamfunction_bvp_solve(vorticity::ScalarField, bc_type::Symbol, apply_gauge::Bool=true)

    streamfunction_field = ScalarField(vorticity.dist, "streamfunction", vorticity.bases, vorticity.dtype)
    ensure_layout!(streamfunction_field, :g)

    # Solve using Jacobi iteration with boundary conditions
    psi = streamfunction_jacobi_solve(vorticity, bc_type, apply_gauge)

    get_grid_data(streamfunction_field) .= psi
    return streamfunction_field
end

function streamfunction_jacobi_solve(vorticity::ScalarField, bc_type::Symbol, apply_gauge::Bool;
                                   max_iter::Int=1000, tolerance::Float64=1e-8)
    """
    Jacobi iteration solver for the Poisson equation ∇²ψ = ω.

    Uses classical Jacobi relaxation with configurable boundary conditions.
    Converges to tolerance or max_iter, whichever comes first.
    """

    ensure_layout!(vorticity, :g)
    omega = get_grid_data(vorticity)
    nx, ny = size(omega)

    # Initialize streamfunction
    psi = zeros(Float64, nx, ny)
    psi_new = similar(psi)

    # Extract grid spacing from domain, or fall back to 2π assumption
    if vorticity.domain !== nothing && length(vorticity.domain.bases) >= 2
        Lx = vorticity.domain.bases[1].meta.bounds[2] - vorticity.domain.bases[1].meta.bounds[1]
        Ly = vorticity.domain.bases[2].meta.bounds[2] - vorticity.domain.bases[2].meta.bounds[1]
    else
        Lx = 2π
        Ly = 2π
        @warn "streamfunction_jacobi_solve: domain bounds not available, assuming Lx=Ly=2π" maxlog=1
    end
    dx = Lx / nx
    dy = Ly / ny
    dx2 = dx^2
    dy2 = dy^2

    # Jacobi iteration coefficients
    alpha = 2.0 / (dx2 + dy2)
    beta_x = 1.0 / dx2
    beta_y = 1.0 / dy2

    for iter in 1:max_iter
        # Jacobi update: psi_new[i,j] = (omega[i,j] + beta_x*(psi[i-1,j] + psi[i+1,j]) + beta_y*(psi[i,j-1] + psi[i,j+1])) / alpha

        for i in 2:(nx - 1), j in 2:(ny - 1)
            psi_new[i, j] = (omega[i, j] +
                           beta_x * (psi[i - 1, j] + psi[i + 1, j]) +
                           beta_y * (psi[i, j - 1] + psi[i, j + 1])) / alpha
        end

        # Apply boundary conditions
        apply_streamfunction_bc!(psi_new, bc_type)

        # Check convergence
        residual = maximum(abs.(psi_new - psi))
        if residual < tolerance
            break
        end

        # Update
        psi, psi_new = psi_new, psi
    end

    # Apply gauge condition if requested
    if apply_gauge
        psi .-= mean(psi)  # Set mean to zero
    end

    # NOTE: MPI reduction is NOT needed here.
    # This function operates on a local Array (not a distributed field).
    # The Jacobi iteration works on the local grid; if distributed execution is needed,
    # proper halo exchange should be implemented instead of Allreduce.
    # Previous code incorrectly used MPI.Allreduce! which would corrupt the data.

    return psi
end

"""Apply boundary conditions to streamfunction"""
function apply_streamfunction_bc!(psi::Array{Float64, 2}, bc_type::Symbol)

    nx, ny = size(psi)

    if bc_type == :no_slip
        # No-slip: ψ = 0 on boundaries
        psi[1, :] .= 0.0      # Bottom
        psi[nx, :] .= 0.0     # Top
        psi[:, 1] .= 0.0      # Left
        psi[:, ny] .= 0.0     # Right

    elseif bc_type == :free_slip
        # Free-slip: ∂ψ/∂n = 0 on boundaries (approximate)
        psi[1, :] .= psi[2, :]        # Bottom
        psi[nx, :] .= psi[nx - 1, :]  # Top
        psi[:, 1] .= psi[:, 2]        # Left
        psi[:, ny] .= psi[:, ny - 1]  # Right

    elseif bc_type == :periodic
        # Periodic: handled naturally by spectral methods
        # No explicit boundary conditions needed

    else
        throw(ArgumentError("Unknown boundary condition: $bc_type"))
    end
end

"""
    Validate streamfunction by checking if it generates the correct velocity field.

    For 2D incompressible flow: u = ∂ψ/∂y, v = -∂ψ/∂x
    """
function validate_streamfunction(velocity::VectorField, streamfunction::ScalarField; tolerance::Float64=1e-6)

    if velocity.coordsys.dim != 2
        return false
    end

    # Calculate velocity from streamfunction
    dpsi_dy = evaluate_differentiate(Differentiate(streamfunction, velocity.coordsys[2], 1), :g)
    dpsi_dx = evaluate_differentiate(Differentiate(streamfunction, velocity.coordsys[1], 1), :g)

    # Check u = ∂ψ/∂y
    ensure_layout!(velocity.components[1], :g)
    ensure_layout!(dpsi_dy, :g)
    u_error = maximum(abs.(get_grid_data(velocity.components[1]) - get_grid_data(dpsi_dy)))

    # Check v = -∂ψ/∂x
    ensure_layout!(velocity.components[2], :g)
    ensure_layout!(dpsi_dx, :g)
    v_error = maximum(abs.(get_grid_data(velocity.components[2]) + get_grid_data(dpsi_dx)))

    # MPI reduction for global error using field's communicator
    if MPI.Initialized()
        comm = velocity.components[1].dist.comm
        u_error = MPI.Allreduce(u_error, MPI.MAX, comm)
        v_error = MPI.Allreduce(v_error, MPI.MAX, comm)
    end

    max_error = max(u_error, v_error)
    is_valid = max_error < tolerance

    return (valid=is_valid, u_error=u_error, v_error=v_error, max_error=max_error)
end

"""Calculate velocity divergence ∇·u"""
function velocity_divergence(velocity::VectorField)

    divergence_op = divergence(velocity)
    return evaluate_operator(divergence_op)
end

# ============================================================================
# Surface Quasi-Geostrophic (SQG) Flow Tools
# ============================================================================

"""
    perp_grad(ψ::ScalarField)

Perpendicular gradient: ∇⊥ψ = (-∂ψ/∂y, ∂ψ/∂x)

Used in 2D flows to compute velocity from streamfunction: u = ∇⊥ψ

This gives a divergence-free velocity field: ∇·u = ∇·(∇⊥ψ) = 0

# Example
```julia
ψ = ScalarField(dist, "psi", bases, Float64)
u = perp_grad(ψ)  # Returns VectorField with u_x = -∂ψ/∂y, u_y = ∂ψ/∂x
```
"""
function perp_grad(ψ::ScalarField)
    # Get coordinates
    coords = ψ.dist.coordsys

    if length(ψ.bases) != 2
        throw(ArgumentError("perp_grad requires a 2D field"))
    end

    # Get coordinates via CoordinateSystem indexing
    coord1 = coords[1]  # x
    coord2 = coords[2]  # y

    # Create result vector field
    result = VectorField(ψ.dist, coords, "perp_grad_$(ψ.name)", ψ.bases, ψ.dtype)

    # u_x = -∂ψ/∂y (second coordinate)
    dpsi_dy = evaluate_differentiate(Differentiate(ψ, coord2, 1), :g)
    ensure_layout!(result.components[1], :g)
    get_grid_data(result.components[1]) .= -get_grid_data(dpsi_dy)

    # u_y = ∂ψ/∂x (first coordinate)
    dpsi_dx = evaluate_differentiate(Differentiate(ψ, coord1, 1), :g)
    ensure_layout!(result.components[2], :g)
    get_grid_data(result.components[2]) .= get_grid_data(dpsi_dx)

    return result
end

# Unicode alias for perpendicular gradient
const ∇⊥ = perp_grad

"""
    sqg_streamfunction(θ::ScalarField)

Compute SQG streamfunction from surface buoyancy:

    ψ = (-Δ)^(-1/2) θ

In spectral space: ψ̂(k) = θ̂(k) / |k|

This is the fundamental relation for Surface Quasi-Geostrophic dynamics.

# Example
```julia
θ = ScalarField(dist, "buoyancy", bases, Float64)
ψ = sqg_streamfunction(θ)
u = perp_grad(ψ)  # or u = ∇⊥(ψ)
```
"""
function sqg_streamfunction(θ::ScalarField)
    return evaluate_fractional_laplacian(FractionalLaplacian(θ, -0.5), :g)
end

"""
    sqg_velocity(θ::ScalarField)

Compute SQG velocity directly from surface buoyancy:

    u = ∇⊥ψ = ∇⊥(-Δ)^(-1/2) θ

This combines streamfunction inversion and perpendicular gradient in one step.

# Example
```julia
θ = ScalarField(dist, "buoyancy", bases, Float64)
u = sqg_velocity(θ)  # Directly get velocity from buoyancy
```
"""
function sqg_velocity(θ::ScalarField)
    ψ = sqg_streamfunction(θ)
    return perp_grad(ψ)
end

"""
    sqg_problem_setup(dist::Distributor, bases::Tuple; κ::Real=0.0, α::Real=0.5)

Set up an SQG initial value problem.

The SQG equation is:
    ∂θ/∂t + u·∇θ = κ(-Δ)^α θ

where:
- θ is the surface buoyancy/temperature
- u = ∇⊥(-Δ)^(-1/2)θ is the velocity
- κ is the dissipation coefficient
- α is the dissipation exponent (typically 1/2 for physical SQG)

# Arguments
- `dist`: Distributor for field allocation
- `bases`: Tuple of bases (should be 2D Fourier bases for SQG)
- `κ`: Dissipation coefficient (default: 0 for inviscid)
- `α`: Dissipation exponent (default: 0.5 for physical dissipation)

# Returns
Named tuple with:
- `problem`: IVP problem object
- `θ`: Buoyancy field
- `u`: Velocity field (computed from θ)
- `ψ`: Streamfunction field (computed from θ)

# Example
```julia
coords = CartesianCoordinates("x", "y")
dist = Distributor(coords; mesh=(2, 2))
x_basis = RealFourier(coords["x"]; size=256, bounds=(0.0, 2π))
y_basis = RealFourier(coords["y"]; size=256, bounds=(0.0, 2π))

sqg = sqg_problem_setup(dist, (x_basis, y_basis); κ=1e-4, α=0.5)

# Set initial condition
    get_grid_data(sqg.θ) .= initial_buoyancy_field

# Solve with timestepper
solver = InitialValueSolver(sqg.problem, RK443())
```
"""
function sqg_problem_setup(dist::Distributor, bases::Tuple; κ::Real=0.0, α::Real=0.5)
    coords = dist.coordsys

    # Create fields
    θ = ScalarField(dist, "θ", bases, Float64)  # Buoyancy

    # Create IVP
    problem = IVP([θ])

    # Add parameters
    problem.parameters["κ"] = Float64(κ)
    problem.parameters["α"] = Float64(α)

    # The SQG equation: ∂θ/∂t + u·∇θ = κ(-Δ)^α θ
    # where u = ∇⊥(-Δ)^(-1/2)θ
    #
    # Note: The nonlinear term u·∇θ needs special handling because
    # u depends on θ through the streamfunction inversion.
    # This is typically handled in the timestepper's nonlinear evaluation.

    if κ > 0
        add_equation!(problem, "∂t(θ) = κ*fraclap(θ, α)")
    else
        add_equation!(problem, "∂t(θ) = 0")
    end

    # Note: The advection term -u·∇θ should be added to the RHS
    # as a nonlinear term evaluated at each timestep

    @info "SQG problem setup" κ=κ α=α bases=length(bases)
    @info "Use sqg_velocity(θ) to compute velocity from buoyancy at each timestep"

    return (
        problem = problem,
        θ = θ,
        compute_velocity = () -> sqg_velocity(θ),
        compute_streamfunction = () -> sqg_streamfunction(θ)
    )
end
