#=
Rotating Shallow Water Equations with Butterworth Filtering
============================================================

This example demonstrates wave-mean flow decomposition using the second-order
Butterworth temporal filter, following:

    Minz, Baker, Kafiabad & Vanneste (2025). "Efficient Lagrangian averaging
    with exponential filters". Phys. Rev. Fluids 10, 074902.

Key Difference from Exponential Filter
--------------------------------------
The second-order Butterworth filter provides:
  - Maximally flat frequency response at ω = 0
  - Sharper cutoff at the filter frequency α
  - Steeper rolloff: -40 dB/decade (vs -20 dB/decade for exponential)
  - Better wave-mean separation with less wave energy leakage

Mathematical Formulation
------------------------
The Butterworth filter is implemented via coupled ODEs:

    dh̃/dt = α[h - (√2-1)h̃ - (2-√2)h̄]
    dh̄/dt = α(h̃ - h̄)

where h̃ is an auxiliary variable and h̄ is the filtered mean.

Transfer function: K(s) = α² / (s² + √2·α·s + α²)

Frequency response comparison at ω = 10α:
  - Exponential: |H|² = 0.0099 (1% power passes)
  - Butterworth: |H|² = 0.0001 (0.01% power passes)

Physical Setup
--------------
Same as exponential filter example: Rotating shallow water on a β-plane
with a geostrophically balanced jet and inertia-gravity wave perturbations.

References
----------
- Minz et al. (2025), Phys. Rev. Fluids 10, 074902, Section IV
- Butterworth (1930), "On the theory of filter amplifiers"
=#

using MPI
using Tarang
using LinearAlgebra
using Printf

# ============================================================================
# Physical Parameters (nondimensional, following Minz et al. 2025)
# ============================================================================

# Domain size
const Lx = 2π
const Ly = 2π

# Physical parameters
const f₀ = 1.0         # Reference Coriolis parameter
const β = 0.5          # β-plane parameter
const g = 1.0          # Gravitational acceleration
const H = 1.0          # Mean layer depth
const c = sqrt(g * H)  # Gravity wave speed

# Rossby number and related scales
const Ro = 0.1
const U = Ro * f₀ * Lx / (2π)

# Dissipation
const ν = 1e-8
const hyper_order = 4

# Filter parameter
const T_wave = 2π / f₀
const α = 0.1 * f₀     # Same α as exponential example for comparison

# ============================================================================
# Numerical Parameters
# ============================================================================

const Nx = 128
const Ny = 128
const dt = 0.01
const t_end = 100.0
const output_interval = 1.0

# ============================================================================
# Initialize MPI
# ============================================================================

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

if rank == 0
    println("=" ^ 70)
    println("Rotating Shallow Water with Butterworth Filtering")
    println("=" ^ 70)
    println()
    println("Physical parameters:")
    @printf("  Rossby number Ro = %.3f\n", Ro)
    @printf("  Coriolis f₀ = %.3f, β = %.3f\n", f₀, β)
    @printf("  Wave speed c = √(gH) = %.3f\n", c)
    @printf("  Wave period T_wave ≈ %.3f\n", T_wave)
    println()
    println("Butterworth Filter parameters:")
    @printf("  α = %.4f (inverse averaging time)\n", α)
    @printf("  Averaging time T_avg = 1/α = %.2f\n", 1/α)
    @printf("  T_avg / T_wave = %.1f\n", (1/α) / T_wave)
    println()
    println("Filter comparison at ω = 10α:")
    @printf("  Exponential |H|² = %.6f (%.2f%% power)\n",
            α^2 / (α^2 + (10α)^2), 100 * α^2 / (α^2 + (10α)^2))
    @printf("  Butterworth |K|² = %.8f (%.4f%% power)\n",
            α^4 / ((α^2 - (10α)^2)^2 + 2*α^2*(10α)^2),
            100 * α^4 / ((α^2 - (10α)^2)^2 + 2*α^2*(10α)^2))
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

coords = CartesianCoordinates("x", "y")
nprocs = MPI.Comm_size(comm)
mesh = nprocs == 1 ? (1, 1) : (1, nprocs)
dist = Distributor(coords; mesh=mesh, dtype=Float64)

basis_x = RealFourier(coords["x"]; size=Nx, bounds=(0.0, Lx))
basis_y = RealFourier(coords["y"]; size=Ny, bounds=(0.0, Ly))

domain = Domain(dist, (basis_x, basis_y))

# ============================================================================
# Fields
# ============================================================================

u = ScalarField(dist, "u", (basis_x, basis_y), Float64)
v = ScalarField(dist, "v", (basis_x, basis_y), Float64)
η = ScalarField(dist, "η", (basis_x, basis_y), Float64)

local_shape = size(parent(u.grid_data))

# ============================================================================
# Temporal Filters (Second-Order Butterworth)
# ============================================================================

if rank == 0
    println("\nInitializing Butterworth filters...")
end

# Create Butterworth filters for each field
u_filter = ButterworthFilter(local_shape; α=α)
v_filter = ButterworthFilter(local_shape; α=α)
η_filter = ButterworthFilter(local_shape; α=α)

# Arrays for mean, auxiliary, and fluctuation
u_mean = zeros(local_shape)
v_mean = zeros(local_shape)
η_mean = zeros(local_shape)

u_aux = zeros(local_shape)  # Auxiliary field h̃ (Butterworth-specific)
v_aux = zeros(local_shape)
η_aux = zeros(local_shape)

u_prime = zeros(local_shape)
v_prime = zeros(local_shape)
η_prime = zeros(local_shape)

# ============================================================================
# Initial Conditions (same as exponential example)
# ============================================================================

x_local = parent(coords(domain, "x"))
y_local = parent(coords(domain, "y"))

function initial_jet(x, y)
    u0 = U * sin(2 * y)
    v0 = 0.0
    η0 = -(f₀ * U / g) * cos(2 * y) / 2
    return u0, v0, η0
end

function wave_perturbation(x, y, t=0.0)
    kx, ky = 4.0, 4.0
    ω = sqrt(f₀^2 + c^2 * (kx^2 + ky^2))
    A = 0.05 * U

    phase = kx * x + ky * y - ω * t
    u_wave = A * cos(phase)
    v_wave = A * sin(phase)
    η_wave = (A * H / c) * cos(phase)

    return u_wave, v_wave, η_wave
end

u_data = parent(u.grid_data)
v_data = parent(v.grid_data)
η_data = parent(η.grid_data)

for j in axes(x_local, 2), i in axes(x_local, 1)
    x, y = x_local[i, j], y_local[i, j]

    u0, v0, η0 = initial_jet(x, y)
    uw, vw, ηw = wave_perturbation(x, y)

    u_data[i, j] = u0 + uw
    v_data[i, j] = v0 + vw
    η_data[i, j] = η0 + ηw
end

# ============================================================================
# Coriolis Parameter
# ============================================================================

f_coriolis = zeros(local_shape)
for j in axes(y_local, 2), i in axes(y_local, 1)
    f_coriolis[i, j] = f₀ + β * y_local[i, j]
end

# ============================================================================
# Diagnostics
# ============================================================================

function compute_energies(u_data, v_data, η_data)
    KE = 0.5 * H * sum(u_data.^2 + v_data.^2) / (Nx * Ny)
    PE = 0.5 * g * sum(η_data.^2) / (Nx * Ny)
    return KE, PE
end

function compute_wave_energy(u_prime, v_prime, η_prime)
    WKE = 0.5 * H * sum(u_prime.^2 + v_prime.^2) / (Nx * Ny)
    WPE = 0.5 * g * sum(η_prime.^2) / (Nx * Ny)
    return WKE, WPE
end

function compute_enstrophy(u_data, v_data)
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
# Time Integration
# ============================================================================

function rhs_shallow_water!(du, dv, dη, u_data, v_data, η_data, f_coriolis)
    dx = Lx / Nx
    dy = Ly / Ny

    for j in 2:Ny-1, i in 2:Nx-1
        dηdx = (η_data[mod1(i+1, Nx), j] - η_data[mod1(i-1, Nx), j]) / (2 * dx)
        dηdy = (η_data[i, mod1(j+1, Ny)] - η_data[i, mod1(j-1, Ny)]) / (2 * dy)

        dudx = (u_data[mod1(i+1, Nx), j] - u_data[mod1(i-1, Nx), j]) / (2 * dx)
        dudy = (u_data[i, mod1(j+1, Ny)] - u_data[i, mod1(j-1, Ny)]) / (2 * dy)
        dvdx = (v_data[mod1(i+1, Nx), j] - v_data[mod1(i-1, Nx), j]) / (2 * dx)
        dvdy = (v_data[i, mod1(j+1, Ny)] - v_data[i, mod1(j-1, Ny)]) / (2 * dy)

        f = f_coriolis[i, j]

        adv_u = u_data[i, j] * dudx + v_data[i, j] * dudy
        adv_v = u_data[i, j] * dvdx + v_data[i, j] * dvdy

        du[i, j] = -adv_u + f * v_data[i, j] - g * dηdx
        dv[i, j] = -adv_v - f * u_data[i, j] - g * dηdy

        div_u = dudx + dvdy
        dη[i, j] = -H * div_u
    end

    # Periodic boundaries
    du[1, :] = du[Nx-1, :]; du[Nx, :] = du[2, :]
    du[:, 1] = du[:, Ny-1]; du[:, Ny] = du[:, 2]
    dv[1, :] = dv[Nx-1, :]; dv[Nx, :] = dv[2, :]
    dv[:, 1] = dv[:, Ny-1]; dv[:, Ny] = dv[:, 2]
    dη[1, :] = dη[Nx-1, :]; dη[Nx, :] = dη[2, :]
    dη[:, 1] = dη[:, Ny-1]; dη[:, Ny] = dη[:, 2]
end

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

    # Forward Euler step
    @. u_data = u_data + dt * du
    @. v_data = v_data + dt * dv
    @. η_data = η_data + dt * dη

    # Update Butterworth filters (coupled system)
    update!(u_filter, u_data, dt)
    update!(v_filter, v_data, dt)
    update!(η_filter, η_data, dt)

    # Get filtered means
    u_mean .= get_mean(u_filter)
    v_mean .= get_mean(v_filter)
    η_mean .= get_mean(η_filter)

    # Get auxiliary fields (Butterworth-specific)
    u_aux .= get_auxiliary(u_filter)
    v_aux .= get_auxiliary(v_filter)
    η_aux .= get_auxiliary(η_filter)

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
# Final Summary with Filter Comparison
# ============================================================================

if rank == 0
    println()
    println("=" ^ 70)
    println("Simulation Complete")
    println("=" ^ 70)
    println()

    KE_final, PE_final = compute_energies(u_data, v_data, η_data)
    KE_mean_final, _ = compute_energies(u_mean, v_mean, η_mean)
    KE_wave_final, _ = compute_wave_energy(u_prime, v_prime, η_prime)

    println("Final state diagnostics:")
    @printf("  Total KE: %.6f\n", KE_final)
    @printf("  Mean KE:  %.6f (%.1f%% of total)\n", KE_mean_final, 100*KE_mean_final/KE_final)
    @printf("  Wave KE:  %.6f (%.1f%% of total)\n", KE_wave_final, 100*KE_wave_final/KE_final)
    println()

    println("Butterworth Filter properties:")
    @printf("  α = %.4f\n", α)
    @printf("  Effective averaging time: %.2f\n", effective_averaging_time(u_filter))
    println()

    println("Frequency response comparison:")
    println("  Frequency      Exponential |H|²    Butterworth |K|²    Ratio")
    println("  " * "-" ^ 60)

    for ω_mult in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
        ω = ω_mult * α
        exp_resp = α^2 / (α^2 + ω^2)
        but_resp = α^4 / ((α^2 - ω^2)^2 + 2*α^2*ω^2)
        ratio = exp_resp / but_resp

        @printf("  ω = %4.1fα      %.6f            %.8f          %.1f×\n",
                ω_mult, exp_resp, but_resp, ratio)
    end

    println()
    println("Key observations:")
    println("  - Butterworth has MUCH sharper cutoff than exponential")
    println("  - At ω = 10α: Butterworth attenuates 100× more than exponential")
    println("  - At ω = 20α: Butterworth attenuates 400× more!")
    println("  - Result: Better wave-mean separation with less wave leakage")
    println()
    println("Trade-offs:")
    println("  - Butterworth requires 2× memory (auxiliary field h̃)")
    println("  - Slightly more computation (coupled ODE system)")
    println("  - Longer spinup time due to oscillating kernel")
    println()

    # Compare auxiliary and mean fields
    aux_mean_diff = maximum(abs.(u_aux .- u_mean))
    @printf("Max |h̃ - h̄| = %.6f (should be small at steady state)\n", aux_mean_diff)
end

MPI.Finalize()
