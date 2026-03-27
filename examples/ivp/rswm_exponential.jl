#=
Rotating Shallow Water Equations with Exponential Mean Filtering
================================================================

This example demonstrates wave-mean flow decomposition using first-order
exponential temporal filtering, following:

    Minz, Baker, Kafiabad & Vanneste (2025). "Efficient Lagrangian averaging
    with exponential filters". Phys. Rev. Fluids 10, 074902.

Physical Setup
--------------
Rotating shallow water equations on a β-plane (doubly periodic domain):

    ∂u/∂t + u·∇u - fv = -g ∂η/∂x + Fₓ + ν∇²u
    ∂v/∂t + u·∇v + fu = -g ∂η/∂y + Fᵧ + ν∇²v
    ∂η/∂t + H∇·(u) = -∇·(ηu)

where:
    - (u, v) = horizontal velocity components
    - η = free surface elevation (deviation from mean depth H)
    - f = f₀ + βy = Coriolis parameter (β-plane approximation)
    - g = gravitational acceleration
    - H = mean layer depth
    - ν = kinematic viscosity (hyperviscosity for numerical stability)
    - (Fₓ, Fᵧ) = external forcing

Key Parameters (from paper)
---------------------------
    - Rossby number: Ro = U/(f₀L) ~ 0.1 (weakly nonlinear)
    - Burger number: Bu = (c/f₀L)² where c = √(gH)
    - Filter parameter: α = inverse averaging timescale

The exponential filter extracts the slowly varying mean flow by solving:

    d h̄/dt = α(h - h̄)

where h is any field and h̄ is its filtered mean.

References
----------
- Minz et al. (2025), Phys. Rev. Fluids 10, 074902
- Vallis (2017), "Atmospheric and Oceanic Fluid Dynamics", Ch. 3
=#

using MPI
using Tarang
using LinearAlgebra
using Printf

# ============================================================================
# Physical Parameters (nondimensional, following Minz et al. 2025)
# ============================================================================

# Domain size (nondimensional)
const Lx = 2π          # Domain length in x
const Ly = 2π          # Domain length in y

# Physical parameters
const f₀ = 1.0         # Reference Coriolis parameter
const β = 0.5          # β-plane parameter (df/dy)
const g = 1.0          # Gravitational acceleration
const H = 1.0          # Mean layer depth
const c = sqrt(g * H)  # Gravity wave speed

# Rossby number and related scales
const Ro = 0.1         # Rossby number U/(f₀L)
const U = Ro * f₀ * Lx / (2π)  # Characteristic velocity

# Dissipation (hyperviscosity for stability)
const ν = 1e-8         # Hyperviscosity coefficient
const hyper_order = 4  # Hyperviscosity order (∇⁸)

# Filter parameter
# Choose α such that averaging time T_avg = 1/α >> T_wave
# For inertia-gravity waves: T_wave ~ 2π/f₀
const T_wave = 2π / f₀
const α = 0.1 * f₀     # Averaging time ≈ 10 × wave period

# ============================================================================
# Numerical Parameters
# ============================================================================

const Nx = 128         # Grid points in x
const Ny = 128         # Grid points in y
const dt = 0.01        # Timestep
const t_end = 100.0    # End time (many wave periods)
const output_interval = 1.0  # Output interval

# ============================================================================
# Initialize MPI
# ============================================================================

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

if rank == 0
    println("=" ^ 70)
    println("Rotating Shallow Water with Exponential Mean Filtering")
    println("=" ^ 70)
    println()
    println("Physical parameters:")
    @printf("  Rossby number Ro = %.3f\n", Ro)
    @printf("  Coriolis f₀ = %.3f, β = %.3f\n", f₀, β)
    @printf("  Wave speed c = √(gH) = %.3f\n", c)
    @printf("  Wave period T_wave ≈ %.3f\n", T_wave)
    println()
    println("Filter parameters:")
    @printf("  α = %.4f (inverse averaging time)\n", α)
    @printf("  Averaging time T_avg = 1/α = %.2f\n", 1/α)
    @printf("  T_avg / T_wave = %.1f (filter captures waves with ω > α)\n", (1/α) / T_wave)
    println()
    println("Numerical parameters:")
    @printf("  Grid: %d × %d\n", Nx, Ny)
    @printf("  Timestep dt = %.4f\n", dt)
    @printf("  End time t_end = %.1f\n", t_end)
    println("=" ^ 70)
end

# ============================================================================
# Domain Setup
# ============================================================================

# Coordinates and distributor
coords = CartesianCoordinates("x", "y")
nprocs = MPI.Comm_size(comm)
mesh = nprocs == 1 ? (1, 1) : (1, nprocs)  # Decompose in y for >1 proc
dist = Distributor(coords; mesh=mesh, dtype=Float64)

# Bases (doubly periodic Fourier)
basis_x = RealFourier(coords["x"]; size=Nx, bounds=(0.0, Lx))
basis_y = RealFourier(coords["y"]; size=Ny, bounds=(0.0, Ly))

# Domain
domain = Domain(dist, (basis_x, basis_y))

# ============================================================================
# Fields
# ============================================================================

# Velocity components
u = ScalarField(dist, "u", (basis_x, basis_y), Float64)  # x-velocity
v = ScalarField(dist, "v", (basis_x, basis_y), Float64)  # y-velocity

# Free surface elevation
η = ScalarField(dist, "η", (basis_x, basis_y), Float64)

# Get local array sizes for filters
local_shape = size(parent(u.grid_data))

# ============================================================================
# Temporal Filters (Exponential Mean)
# ============================================================================

if rank == 0
    println("\nInitializing exponential mean filters...")
end

# Create filters for each field
u_filter = ExponentialMean(local_shape; α=α)
v_filter = ExponentialMean(local_shape; α=α)
η_filter = ExponentialMean(local_shape; α=α)

# Arrays for mean and fluctuation
u_mean = zeros(local_shape)
v_mean = zeros(local_shape)
η_mean = zeros(local_shape)

u_prime = zeros(local_shape)
v_prime = zeros(local_shape)
η_prime = zeros(local_shape)

# ============================================================================
# Initial Conditions
# ============================================================================

# Coordinate arrays
x_local = parent(coords(domain, "x"))
y_local = parent(coords(domain, "y"))

# Initialize with balanced flow + wave perturbation
# Geostrophically balanced jet
function initial_jet(x, y)
    # Zonal jet with sinusoidal profile
    u0 = U * sin(2 * y)
    v0 = 0.0
    # Balanced height field (geostrophic balance: f*u = -g*∂η/∂y)
    η0 = -(f₀ * U / g) * cos(2 * y) / 2
    return u0, v0, η0
end

# Add inertia-gravity wave packet
function wave_perturbation(x, y, t=0.0)
    # Wave parameters
    kx, ky = 4.0, 4.0  # Wavenumbers
    ω = sqrt(f₀^2 + c^2 * (kx^2 + ky^2))  # Dispersion relation
    A = 0.05 * U  # Wave amplitude (5% of mean flow)

    # Wave solution (linearized)
    phase = kx * x + ky * y - ω * t
    u_wave = A * cos(phase)
    v_wave = A * sin(phase)
    η_wave = (A * H / c) * cos(phase)

    return u_wave, v_wave, η_wave
end

# Set initial conditions
u_data = parent(u.grid_data)
v_data = parent(v.grid_data)
η_data = parent(η.grid_data)

for j in axes(x_local, 2), i in axes(x_local, 1)
    x, y = x_local[i, j], y_local[i, j]

    # Balanced flow
    u0, v0, η0 = initial_jet(x, y)

    # Add waves
    uw, vw, ηw = wave_perturbation(x, y)

    u_data[i, j] = u0 + uw
    v_data[i, j] = v0 + vw
    η_data[i, j] = η0 + ηw
end

# ============================================================================
# Coriolis Parameter (β-plane)
# ============================================================================

# f = f₀ + β*y at each grid point
f_coriolis = zeros(local_shape)
for j in axes(y_local, 2), i in axes(y_local, 1)
    f_coriolis[i, j] = f₀ + β * y_local[i, j]
end

# ============================================================================
# Diagnostics
# ============================================================================

function compute_energies(u_data, v_data, η_data)
    # Kinetic energy: KE = (1/2) * H * (u² + v²)
    KE = 0.5 * H * sum(u_data.^2 + v_data.^2) / (Nx * Ny)

    # Potential energy: PE = (1/2) * g * η²
    PE = 0.5 * g * sum(η_data.^2) / (Nx * Ny)

    return KE, PE
end

function compute_wave_energy(u_prime, v_prime, η_prime)
    # Wave kinetic energy
    WKE = 0.5 * H * sum(u_prime.^2 + v_prime.^2) / (Nx * Ny)

    # Wave potential energy
    WPE = 0.5 * g * sum(η_prime.^2) / (Nx * Ny)

    return WKE, WPE
end

function compute_enstrophy(u_data, v_data)
    # Relative vorticity ζ = ∂v/∂x - ∂u/∂y (computed spectrally would be better)
    # Simplified finite difference for diagnostics
    dx = Lx / Nx
    dy = Ly / Ny

    ζ = zeros(size(u_data))
    for j in 2:size(u_data, 2)-1, i in 2:size(u_data, 1)-1
        dvdx = (v_data[i+1, j] - v_data[i-1, j]) / (2 * dx)
        dudy = (u_data[i, j+1] - u_data[i, j-1]) / (2 * dy)
        ζ[i, j] = dvdx - dudy
    end

    return sum(ζ.^2) / (Nx * Ny)
end

# ============================================================================
# Time Integration (Simple semi-implicit for demonstration)
# ============================================================================

# This is a simplified time-stepper for demonstration
# In practice, use Tarang's built-in IMEX timesteppers

function rhs_shallow_water!(du, dv, dη, u_data, v_data, η_data, f_coriolis)
    dx = Lx / Nx
    dy = Ly / Ny

    # Compute derivatives using central differences (spectral would be better)
    for j in 2:Ny-1, i in 2:Nx-1
        # Gradients
        dηdx = (η_data[mod1(i+1, Nx), j] - η_data[mod1(i-1, Nx), j]) / (2 * dx)
        dηdy = (η_data[i, mod1(j+1, Ny)] - η_data[i, mod1(j-1, Ny)]) / (2 * dy)

        dudx = (u_data[mod1(i+1, Nx), j] - u_data[mod1(i-1, Nx), j]) / (2 * dx)
        dudy = (u_data[i, mod1(j+1, Ny)] - u_data[i, mod1(j-1, Ny)]) / (2 * dy)
        dvdx = (v_data[mod1(i+1, Nx), j] - v_data[mod1(i-1, Nx), j]) / (2 * dx)
        dvdy = (v_data[i, mod1(j+1, Ny)] - v_data[i, mod1(j-1, Ny)]) / (2 * dy)

        # Coriolis
        f = f_coriolis[i, j]

        # Advection
        adv_u = u_data[i, j] * dudx + v_data[i, j] * dudy
        adv_v = u_data[i, j] * dvdx + v_data[i, j] * dvdy

        # Momentum equations
        du[i, j] = -adv_u + f * v_data[i, j] - g * dηdx
        dv[i, j] = -adv_v - f * u_data[i, j] - g * dηdy

        # Continuity equation
        div_u = dudx + dvdy
        dη[i, j] = -H * div_u
    end

    # Periodic boundaries
    du[1, :] = du[Nx-1, :]
    du[Nx, :] = du[2, :]
    du[:, 1] = du[:, Ny-1]
    du[:, Ny] = du[:, 2]

    dv[1, :] = dv[Nx-1, :]
    dv[Nx, :] = dv[2, :]
    dv[:, 1] = dv[:, Ny-1]
    dv[:, Ny] = dv[:, 2]

    dη[1, :] = dη[Nx-1, :]
    dη[Nx, :] = dη[2, :]
    dη[:, 1] = dη[:, Ny-1]
    dη[:, Ny] = dη[:, 2]
end

# Allocate RHS arrays
du = zeros(local_shape)
dv = zeros(local_shape)
dη = zeros(local_shape)

# ============================================================================
# Main Time Loop
# ============================================================================

if rank == 0
    println("\nStarting time integration...")
    println()
    println("   Time      KE_total    KE_mean    KE_wave    PE_total   Enstrophy")
    println("-" ^ 70)
end

t = 0.0
step = 0
next_output = 0.0

# Storage for time series
time_series = Float64[]
KE_total_series = Float64[]
KE_mean_series = Float64[]
KE_wave_series = Float64[]

while t < t_end
    # Compute RHS
    rhs_shallow_water!(du, dv, dη, u_data, v_data, η_data, f_coriolis)

    # Forward Euler step (simple for demonstration)
    @. u_data = u_data + dt * du
    @. v_data = v_data + dt * dv
    @. η_data = η_data + dt * dη

    # Update temporal filters
    update!(u_filter, u_data, dt)
    update!(v_filter, v_data, dt)
    update!(η_filter, η_data, dt)

    # Get filtered means
    u_mean .= get_mean(u_filter)
    v_mean .= get_mean(v_filter)
    η_mean .= get_mean(η_filter)

    # Compute fluctuations
    @. u_prime = u_data - u_mean
    @. v_prime = v_data - v_mean
    @. η_prime = η_data - η_mean

    t += dt
    step += 1

    # Output diagnostics
    if t >= next_output
        KE, PE = compute_energies(u_data, v_data, η_data)
        KE_mean, _ = compute_energies(u_mean, v_mean, η_mean)
        KE_wave, PE_wave = compute_wave_energy(u_prime, v_prime, η_prime)
        enstrophy = compute_enstrophy(u_data, v_data)

        # Store time series
        push!(time_series, t)
        push!(KE_total_series, KE)
        push!(KE_mean_series, KE_mean)
        push!(KE_wave_series, KE_wave)

        if rank == 0
            @printf("%8.2f  %10.6f  %10.6f  %10.6f  %10.6f  %10.6f\n",
                    t, KE, KE_mean, KE_wave, PE, enstrophy)
        end

        next_output += output_interval
    end
end

# ============================================================================
# Final Summary
# ============================================================================

if rank == 0
    println()
    println("=" ^ 70)
    println("Simulation Complete")
    println("=" ^ 70)
    println()

    # Compute final statistics
    KE_final, PE_final = compute_energies(u_data, v_data, η_data)
    KE_mean_final, _ = compute_energies(u_mean, v_mean, η_mean)
    KE_wave_final, _ = compute_wave_energy(u_prime, v_prime, η_prime)

    println("Final state diagnostics:")
    @printf("  Total KE: %.6f\n", KE_final)
    @printf("  Mean KE:  %.6f (%.1f%% of total)\n", KE_mean_final, 100*KE_mean_final/KE_final)
    @printf("  Wave KE:  %.6f (%.1f%% of total)\n", KE_wave_final, 100*KE_wave_final/KE_final)
    println()

    println("Filter properties:")
    @printf("  α = %.4f\n", α)
    @printf("  Effective averaging time: %.2f\n", effective_averaging_time(u_filter))
    @printf("  Frequency response at ω = f₀: |H|² = %.4f\n", filter_response(u_filter, f₀))
    @printf("  Frequency response at ω = 2f₀: |H|² = %.4f\n", filter_response(u_filter, 2*f₀))
    println()

    println("Notes:")
    println("  - Exponential filter has -20 dB/decade rolloff")
    println("  - Some wave energy leaks through the filter")
    println("  - For sharper separation, use ButterworthFilter")
    println()
end

MPI.Finalize()
