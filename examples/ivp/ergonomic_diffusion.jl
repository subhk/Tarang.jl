# Ergonomic API Demo — 1D Diffusion
#
# Same physics as minimal_diffusion.jl but using the new convenience API.
# Demonstrates: PeriodicDomain, add_parameters!, set!, grid_data,
#               on_interval, @root_only

using Tarang

# Domain + field (one-liners)
domain = PeriodicDomain(64)
T = ScalarField(domain, "T")

# Problem with bulk parameters (instead of repeated add_substitution!)
problem = IVP([T])
add_parameters!(problem, kappa=0.01)
add_equation!(problem, "∂t(T) - kappa*Δ(T) = 0")

# Initial condition — one line instead of three
set!(T, (x,) -> sin(x))

# Solver
solver = InitialValueSolver(problem, RK222(); dt=0.01)

# Rich display
@root_only println(domain)
@root_only println(solver)

# Run with readable callbacks
run!(solver; stop_time=1.0, callbacks=[
    on_interval(25) do s
        data = grid_data(T)
        max_T = maximum(abs.(data))
        @root_only @info "Step $(s.iteration): max|T| = $(round(max_T; sigdigits=5))"
    end
])

@root_only @info "Done! max|T| = $(maximum(abs.(grid_data(T))))"
@root_only @info "Expected ≈ $(exp(-0.01 * 1.0)) (exponential decay of k=1 mode)"
