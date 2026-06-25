"""
Flow analysis tools

Note: GlobalArrayReducer, reduce_scalar, global_min, global_max, and global_mean
are defined in core/evaluator.jl and reused here.
"""

using Statistics
using LinearAlgebra
using MPI
using FFTW: fftshift

include("flow_tools/flow_tools_cfl.jl")
include("flow_tools/flow_tools_diagnostics.jl")
include("flow_tools/flow_tools_spectrum_types.jl")
include("flow_tools/flow_tools_domain_utils.jl")
include("flow_tools/flow_tools_spectra.jl")
include("flow_tools/flow_tools_streamfunction.jl")
include("flow_tools/flow_tools_qg.jl")
include("flow_tools/flow_tools_boundary_advection.jl")

# Turbulence statistics
"""Calculate basic turbulence statistics"""
function turbulence_statistics(velocity::VectorField)

    stats = Dict{String, Float64}()
    use_mpi = MPI.Initialized() && velocity.dist.size > 1
    comm = velocity.dist.comm

    # RMS velocity - CRITICAL: Use global reduction for MPI
    # Compute local sum of squares and local count, then reduce globally
    vel_sum_squared = 0.0
    local_count = 0
    for component in velocity.components
        ensure_layout!(component, :g)
        data = get_grid_data(component)
        # `parent` gives the LOCAL slab; reducing the PencilArray directly is a
        # COLLECTIVE op (already global) which would then be Allreduced a second time
        # below → nprocs× inflation (and errors on non-Intel MPI). parent is a no-op
        # on plain serial arrays.
        vel_sum_squared += sum(abs2, parent(data))
        local_count += length(parent(data))
    end

    if use_mpi
        # Global reduction: sum of squares and total count
        global_sum_squared = MPI.Allreduce(vel_sum_squared, MPI.SUM, comm)
        global_count = MPI.Allreduce(local_count, MPI.SUM, comm)
        stats["velocity_rms"] = sqrt(global_sum_squared / global_count)
    else
        stats["velocity_rms"] = sqrt(vel_sum_squared / local_count)
    end

    # Maximum velocity
    max_vel = 0.0
    for component in velocity.components
        ensure_layout!(component, :g)
        max_vel = max(max_vel, maximum(abs, parent(get_grid_data(component))))
    end

    if use_mpi
        # Use field's communicator, not COMM_WORLD, to support custom communicators
        max_vel = MPI.Allreduce(max_vel, MPI.MAX, comm)
    end
    stats["velocity_max"] = max_vel

    # Calculate derived quantities
    if velocity.coordsys.dim == 2
        # Vorticity RMS for 2D flows - CRITICAL: Use global reduction for MPI
        vort = evaluate_operator(curl(velocity))
        ensure_layout!(vort, :g)
        vort_data = get_grid_data(vort)
        vort_sum_squared = sum(abs2, parent(vort_data))   # local slab (see velocity_rms note)
        vort_count = length(parent(vort_data))

        if use_mpi
            global_vort_sum = MPI.Allreduce(vort_sum_squared, MPI.SUM, comm)
            global_vort_count = MPI.Allreduce(vort_count, MPI.SUM, comm)
            stats["vorticity_rms"] = sqrt(global_vort_sum / global_vort_count)
        else
            stats["vorticity_rms"] = sqrt(vort_sum_squared / vort_count)
        end

        # Maximum vorticity
        max_vort = maximum(abs, parent(vort_data))
        if use_mpi
            # Use field's communicator, not COMM_WORLD
            max_vort = MPI.Allreduce(max_vort, MPI.MAX, comm)
        end
        stats["vorticity_max"] = max_vort
    end

    return stats
end

# ============================================================================
# Exports
# ============================================================================

# CFL adaptive timestepping
export CFL, add_velocity!, compute_timestep

# Flow diagnostics
export reynolds_number, kinetic_energy, total_kinetic_energy
export enstrophy, total_enstrophy
export energy_dissipation_rate, vorticity_transport

# Energy spectrum analysis
export energy_spectrum, WavenumberInfo
export validate_fourier_bases, get_wavenumber_info
export calculate_wavenumber_grids, calculate_k_magnitudes, calculate_kmax
export calculate_radial_energy_spectrum, calculate_spectral_kinetic_energy
export calculate_full_energy_spectrum

# Power spectra for scalar and vector fields
export power_spectrum, enstrophy_spectrum, scalar_spectrum
export SpectrumBinning, LinearBinning, LogBinning, CustomBinning
export calculate_spectral_power, calculate_radial_power_spectrum, calculate_full_power_spectrum
export calculate_radial_vector_spectrum, calculate_full_vector_spectrum
export validate_fourier_bases_scalar, get_wavenumber_info_scalar

# Domain utilities
export get_domain_size, get_domain_bounds, get_fourier_shape

# Turbulence statistics
export turbulence_statistics

# Streamfunction utilities
export streamfunction, validate_streamfunction
export get_fourier_basis_info, all_periodic_fourier
export streamfunction_spectral_invert, get_2d_wavenumber_grids
export streamfunction_bvp_solve, streamfunction_jacobi_solve
export apply_streamfunction_bc!

# Velocity utilities
export velocity_divergence, perp_grad, ∇⊥

# Surface Quasi-Geostrophic (SQG) system
export sqg_streamfunction, sqg_velocity, sqg_problem_setup

# Quasi-Geostrophic (QG) system
export QGSystem, qg_system_setup
export qg_invert!, qg_surface_velocity!, extract_surface
export qg_advection_rhs, qg_step!
export qg_step_euler!, qg_step_rk2!, qg_step_rk4!
export qg_energy

# Boundary advection-diffusion system types
export VelocitySource, PrescribedVelocity, InteriorDerivedVelocity, SelfDerivedVelocity
export DiffusionSpec, BoundarySpec
export BoundaryAdvectionDiffusion

# Boundary advection-diffusion system functions
export boundary_advection_diffusion_setup, setup_interior_coupling
export bad_compute_velocity!, bad_compute_rhs!, compute_diffusion_term
export bad_step!, bad_solve_interior!
export bad_step_euler!, bad_step_rk2!, bad_step_rk4!, bad_step_ssprk3!
export bad_add_source!, bad_energy, bad_enstrophy
export bad_max_velocity, bad_cfl_dt
