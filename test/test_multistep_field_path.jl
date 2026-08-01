# Regression: the explicit-multistep FIELD path.
#
# A GPU (and MPI pure-Fourier) solver assembles no global L/M matrices, so every
# multistep method used to hit `_prepare_global_multistep_matrices!`, find
# `L_matrix === nothing`, warn once, and fall back — CNAB2/SBDF2 → CNAB1/SBDF1 →
# forward Euler, permanently. Order 2/3/4 collapsed to order 1 for the whole run.
#
# With no implicit linear operator the matrices are redundant: `L = 0` and `M = I`
# reduce the IMEX multistep solve to
#     X_new = (Σ_k c[k+1]·F[k] − Σ_k a[k+1]·X[k]) / a[1]
# which is a linear combination of stored fields. These tests drive that field
# path directly on CPU (as test_gpu_field_rk_allocations.jl does for RK) and
# require it to reproduce the global-matrix path's formal order and values.

using Test
using Tarang

# Advance `solver` with the field path only, replicating the bookkeeping `step!`
# does around the timestepper dispatch.
function _advance_field_multistep!(solver, method::Symbol, dt::Float64)
    state = Tarang._ensure_timestepper_state!(solver, dt)
    Tarang._step_explicit_multistep_field!(state, solver, method)
    Tarang._sync_solver_from_timestepper!(solver)
    solver.sim_time += dt
    solver.iteration += 1
    return nothing
end

# dt(u) = -u on a periodic domain: explicit-only (L is assembled all-zero on CPU),
# spatially exact, exact solution exp(-t).
function _decay_solver(stepper; dt=0.02)
    domain = PeriodicDomain(8)
    u = ScalarField(domain, "u")
    set!(u, (x,) -> 1.0)
    problem = IVP([u])
    add_equation!(problem, "dt(u) = -u")
    return InitialValueSolver(problem, stepper; dt)
end

function _field_path_error(stepper, method::Symbol, dt::Float64; tfinal=1.0)
    solver = _decay_solver(stepper; dt)
    for _ in 1:round(Int, tfinal / dt)
        _advance_field_multistep!(solver, method, dt)
    end
    f = solver.state[1]
    ensure_layout!(f, :g)
    return abs(first(get_grid_data(f)) - exp(-tfinal))
end

function _global_path_error(stepper, dt::Float64; tfinal=1.0)
    solver = _decay_solver(stepper; dt)
    for _ in 1:round(Int, tfinal / dt)
        step!(solver, dt)
    end
    f = solver.state[1]
    ensure_layout!(f, :g)
    return abs(first(get_grid_data(f)) - exp(-tfinal))
end

@testset "Explicit-multistep field path keeps the nominal order" begin
    # (method, stepper, expected order). The field path must not silently degrade
    # to the forward-Euler rate of 1 that the missing-matrix fallback produced.
    cases = ((:cnab2, CNAB2(), 2), (:sbdf2, SBDF2(), 2),
             (:sbdf3, SBDF3(), 3), (:sbdf4, SBDF4(), 4))
    for (method, stepper, order) in cases
        e_coarse = _field_path_error(stepper, method, 0.02)
        e_fine   = _field_path_error(stepper, method, 0.01)
        rate = log2(e_coarse / e_fine)
        @info "field-path convergence" method e_coarse e_fine rate
        @test rate > order - 0.5
    end
end

@testset "Field path matches the global-matrix path" begin
    for (method, stepper) in ((:cnab1, CNAB1()), (:cnab2, CNAB2()),
                              (:sbdf1, SBDF1()), (:sbdf2, SBDF2()),
                              (:sbdf3, SBDF3()), (:sbdf4, SBDF4()))
        dt = 0.02
        ef = _field_path_error(stepper, method, dt)
        eg = _global_path_error(stepper, dt)
        # Both discretize the same scheme; with L = 0 and M = I they are the same
        # arithmetic, so the errors must agree to well within the error itself.
        @test isapprox(ef, eg; rtol=1e-6, atol=1e-14)
    end
end

@testset "First-order members reproduce forward Euler exactly" begin
    dt = 0.1
    for (method, stepper) in ((:cnab1, CNAB1()), (:sbdf1, SBDF1()))
        solver = _decay_solver(stepper; dt)
        _advance_field_multistep!(solver, method, dt)
        f = solver.state[1]
        ensure_layout!(f, :g)
        # With L = 0 both CNAB1 and SBDF1 reduce to X + dt*F = 1 - dt.
        @test first(get_grid_data(f)) ≈ 1.0 - dt atol=1e-12
    end
end

@testset "Steady-state cost is O(1) in the grid size" begin
    # The history deques recycle their dropped tail buffers and the update is a
    # sequence of in-place broadcasts, so per-step allocation must not track the
    # field size. A regression here means a buffer is being rebuilt every step.
    allocs = map((16, 512)) do n
        domain = PeriodicDomain(n)
        u = ScalarField(domain, "u")
        set!(u, (x,) -> sin(x))
        problem = IVP([u])
        add_equation!(problem, "dt(u) = -u")
        solver = InitialValueSolver(problem, SBDF4(); dt=0.02)
        for _ in 1:20                      # past startup and deque growth
            _advance_field_multistep!(solver, :sbdf4, 0.02)
        end
        @allocated _advance_field_multistep!(solver, :sbdf4, 0.02)
    end
    @info "field-path steady-state allocation" small=allocs[1] large=allocs[2]
    @test allocs[1] == allocs[2]
    @test allocs[2] < 4096
end

@testset "Field path declines when an implicit operator is present" begin
    # The whole point of the gate: an implicit L must NOT be dropped. Only a
    # genuinely explicit problem may take the matrix-free field path.
    domain = PeriodicDomain(8)
    u = ScalarField(domain, "u")
    set!(u, (x,) -> 1.0)
    problem = IVP([u])
    add_equation!(problem, "dt(u) - 0.5*lap(u) = 0")
    solver = InitialValueSolver(problem, CNAB2(); dt=0.01)
    Tarang._ensure_timestepper_state!(solver, 0.01)
    @test Tarang._problem_has_implicit_linear_term(solver)
    @test !Tarang._explicit_multistep_field_eligible(solver)

    explicit = _decay_solver(CNAB2())
    Tarang._ensure_timestepper_state!(explicit, 0.02)
    @test !Tarang._problem_has_implicit_linear_term(explicit)
    @test Tarang._explicit_multistep_field_eligible(explicit)
end
