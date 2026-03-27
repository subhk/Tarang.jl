# Minimal Example: 1D Diffusion Equation
#
# Solves the heat equation ∂T/∂t = κ ∇²T on a periodic domain [0, 2π].
# This is the simplest possible Tarang.jl simulation — just 12 lines of code.
#
#     julia --project=. examples/ivp/minimal_diffusion.jl

using Tarang

# Domain + field
domain = PeriodicDomain(64)
T = ScalarField(domain, "T")

# Problem: ∂T/∂t = κ∇²T
problem = IVP([T])
add_parameters!(problem, kappa=0.01)
add_equation!(problem, "∂t(T) - kappa*Δ(T) = 0")

# Initial condition: sine wave (one line)
set!(T, (x,) -> sin(x))

# Solve
solver = InitialValueSolver(problem, RK222(); dt=0.01)
run!(solver; stop_time=1.0, log_interval=10)

println("Final max(T) = $(maximum(abs.(grid_data(T))))")
println("Expected ≈ $(exp(-0.01 * 1.0)) (exponential decay of k=1 mode)")
