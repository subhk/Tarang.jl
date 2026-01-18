"""
ETDRK2 Convergence Test

This test verifies the order of accuracy for different ETDRK2 variants by solving
a problem with known analytical solution.

Test Problem:
    du/dt = λu + g(t)

where λ < 0 (stiff), with exact solution u(t) = exp(λt)

We test two variants:
1. Standard ETDRK2: u_{n+1} = exp(hL)u_n + h*φ₁(hL)*N(c)
2. Richardson ETDRK2: u_{n+1} = exp(hL)u_n + h*φ₁(hL)*[2*N(c) - N(u_n)]
"""

using LinearAlgebra
using Printf

# Phi function
function phi1(z::Float64)
    if abs(z) < 1e-8
        return 1.0 + z/2.0 + z^2/6.0 + z^3/24.0
    else
        return (exp(z) - 1.0) / z
    end
end

# Test problem: du/dt = λu + g(t)
# where g(t) is chosen so that exact solution is u(t) = exp(λt)
# Then: λ*exp(λt) + g(t) = λ*exp(λt), so g(t) = 0
# Actually, let's use a more interesting problem:
# du/dt = λu + sin(ωt), λ = -10, ω = 2π
# This has exact solution (found via integrating factor):
# u(t) = C*exp(λt) + [ω*cos(ωt) + λ*sin(ωt)] / (λ² + ω²) * exp(λt) * ∫ exp(-λs)*sin(ωs) ds

# Simpler test: du/dt = λu, u(0) = 1
# Exact: u(t) = exp(λt)
function exact_solution(t::Float64, λ::Float64)
    return exp(λ * t)
end

function etdrk2_standard(u_n::Float64, λ::Float64, h::Float64, t_n::Float64)
    """Standard ETDRK2: u_{n+1} = a_n + h*φ₁*N(c)"""

    # Linear propagator
    exp_hL = exp(h * λ)
    phi1_hL = phi1(h * λ)

    # Stage 1: Predictor
    a_n = exp_hL * u_n
    N_u_n = 0.0  # For du/dt = λu, N(u) = 0 (all linear)
    c = a_n + h * phi1_hL * N_u_n

    # Stage 2: Corrector
    N_c = 0.0
    u_new = a_n + h * phi1_hL * N_c

    return u_new
end

function etdrk2_richardson(u_n::Float64, λ::Float64, h::Float64, t_n::Float64)
    """Richardson ETDRK2: u_{n+1} = a_n + h*φ₁*[2*N(c) - N(u_n)]"""

    # Linear propagator
    exp_hL = exp(h * λ)
    phi1_hL = phi1(h * λ)

    # Stage 1: Predictor
    a_n = exp_hL * u_n
    N_u_n = 0.0  # For du/dt = λu, N(u) = 0 (all linear)
    c = a_n + h * phi1_hL * N_u_n

    # Stage 2: Corrector with Richardson extrapolation
    N_c = 0.0
    u_new = a_n + h * phi1_hL * (2.0 * N_c - N_u_n)

    return u_new
end

# Better test: Semi-linear problem
# du/dt = λu + u²  (Fisher equation type)
# This has nonlinear term N(u) = u²

function etdrk2_standard_nonlinear(u_n::Float64, λ::Float64, h::Float64, t_n::Float64)
    """Standard ETDRK2 for du/dt = λu + u²"""

    # Linear propagator
    exp_hL = exp(h * λ)
    phi1_hL = phi1(h * λ)

    # Stage 1: Predictor
    a_n = exp_hL * u_n
    N_u_n = u_n * u_n  # Nonlinear term
    c = a_n + h * phi1_hL * N_u_n

    # Stage 2: Corrector
    N_c = c * c  # Evaluate nonlinear term at c
    u_new = a_n + h * phi1_hL * N_c

    return u_new
end

function etdrk2_richardson_nonlinear(u_n::Float64, λ::Float64, h::Float64, t_n::Float64)
    """Richardson ETDRK2 for du/dt = λu + u²"""

    # Linear propagator
    exp_hL = exp(h * λ)
    phi1_hL = phi1(h * λ)

    # Stage 1: Predictor
    a_n = exp_hL * u_n
    N_u_n = u_n * u_n  # Nonlinear term
    c = a_n + h * phi1_hL * N_u_n

    # Stage 2: Corrector with Richardson extrapolation
    N_c = c * c  # Evaluate nonlinear term at c
    u_new = a_n + h * phi1_hL * (2.0 * N_c - N_u_n)

    return u_new
end

function run_convergence_test()
    println("="^70)
    println("ETDRK2 Convergence Test")
    println("="^70)

    # Test parameters
    λ = -10.0  # Stiff linear part
    T = 1.0    # Final time
    u0 = 0.1   # Initial condition

    # Note: For linear problem du/dt = λu, both methods reduce to exact solution
    # via φ₁, so we need a nonlinear term to distinguish them

    println("\nTest Problem: du/dt = λu + u²")
    println("Parameters: λ = $λ, u(0) = $u0, T = $T")
    println("\nNote: No analytical solution available, using reference solution")
    println()

    # Get reference solution with very small timestep
    h_ref = 1e-5
    u_ref = u0
    t = 0.0
    while t < T
        u_ref = etdrk2_standard_nonlinear(u_ref, λ, h_ref, t)
        t += h_ref
    end
    println("Reference solution (h = $h_ref): u(T) = $u_ref")
    println()

    # Test different timesteps
    timesteps = [0.1, 0.05, 0.025, 0.0125, 0.00625]

    println("Standard ETDRK2 (Cox-Matthews):")
    println("-"^70)
    @printf("%-12s %-20s %-15s %-15s\n", "h", "u(T)", "Error", "Rate")
    println("-"^70)

    errors_standard = Float64[]
    for h in timesteps
        u = u0
        t = 0.0
        nsteps = Int(ceil(T / h))
        h_actual = T / nsteps

        for step in 1:nsteps
            u = etdrk2_standard_nonlinear(u, λ, h_actual, t)
            t += h_actual
        end

        error = abs(u - u_ref)
        push!(errors_standard, error)

        if length(errors_standard) > 1
            rate = log(errors_standard[end-1] / errors_standard[end]) / log(2.0)
            @printf("%-12.6f %-20.12e %-15.6e %-15.4f\n", h_actual, u, error, rate)
        else
            @printf("%-12.6f %-20.12e %-15.6e %-15s\n", h_actual, u, error, "—")
        end
    end

    println()
    println("Richardson ETDRK2 (Our Implementation):")
    println("-"^70)
    @printf("%-12s %-20s %-15s %-15s\n", "h", "u(T)", "Error", "Rate")
    println("-"^70)

    errors_richardson = Float64[]
    for h in timesteps
        u = u0
        t = 0.0
        nsteps = Int(ceil(T / h))
        h_actual = T / nsteps

        for step in 1:nsteps
            u = etdrk2_richardson_nonlinear(u, λ, h_actual, t)
            t += h_actual
        end

        error = abs(u - u_ref)
        push!(errors_richardson, error)

        if length(errors_richardson) > 1
            rate = log(errors_richardson[end-1] / errors_richardson[end]) / log(2.0)
            @printf("%-12.6f %-20.12e %-15.6e %-15.4f\n", h_actual, u, error, rate)
        else
            @printf("%-12.6f %-20.12e %-15.6e %-15s\n", h_actual, u, error, "—")
        end
    end

    println()
    println("="^70)
    println("Analysis:")
    println("="^70)

    # Average convergence rates (skip first which has no rate)
    if length(errors_standard) > 1
        rates_std = [log(errors_standard[i] / errors_standard[i+1]) / log(2.0)
                     for i in 1:length(errors_standard)-1]
        avg_rate_std = sum(rates_std) / length(rates_std)
        println("Standard ETDRK2 average convergence rate: $(@sprintf("%.3f", avg_rate_std))")
    end

    if length(errors_richardson) > 1
        rates_rich = [log(errors_richardson[i] / errors_richardson[i+1]) / log(2.0)
                      for i in 1:length(errors_richardson)-1]
        avg_rate_rich = sum(rates_rich) / length(rates_rich)
        println("Richardson ETDRK2 average convergence rate: $(@sprintf("%.3f", avg_rate_rich))")
    end

    println()
    println("Expected: Both methods should show O(h²) convergence (rate ≈ 2.0)")
    println()

    # Compare accuracy
    println("Accuracy comparison at h = 0.05:")
    idx = findfirst(x -> abs(x - 0.05) < 1e-10, timesteps)
    if idx !== nothing
        println("  Standard:   error = $(@sprintf("%.6e", errors_standard[idx]))")
        println("  Richardson: error = $(@sprintf("%.6e", errors_richardson[idx]))")
        if errors_richardson[idx] < errors_standard[idx]
            factor = errors_standard[idx] / errors_richardson[idx]
            println("  → Richardson is $(@sprintf("%.2f", factor))x more accurate")
        else
            factor = errors_richardson[idx] / errors_standard[idx]
            println("  → Standard is $(@sprintf("%.2f", factor))x more accurate")
        end
    end

    println("\n" * "="^70)
end

# Run the test
run_convergence_test()
