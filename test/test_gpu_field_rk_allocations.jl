using Test
using Tarang

function _advance_field_rk!(solver, state)
    ts = state.timestepper
    Tarang._step_explicit_rk_gpu!(
        state, solver, ts.A_explicit, ts.b_explicit, ts.c_explicit)
    Tarang._sync_solver_from_timestepper!(solver)
    solver.sim_time += state.dt
    return nothing
end

@testset "Field-native RK preserves compiled lazy RHS stages" begin
    domain = PeriodicDomain(8)
    q = ScalarField(domain, "q")
    set!(q, (x,) -> 1.0)

    problem = IVP([q])
    add_equation!(problem, "dt(q) = q")
    dt = 0.1
    solver = InitialValueSolver(problem, RK222(); dt)
    state = Tarang._ensure_timestepper_state!(solver, dt)
    ts = state.timestepper

    Tarang._step_explicit_rk_gpu!(
        state, solver, ts.A_explicit, ts.b_explicit, ts.c_explicit)

    # Compute the scalar RK stability polynomial independently. Every lazy-RHS
    # evaluation reuses one output field, so the field driver must retain each
    # stage before the next evaluation overwrites that output.
    k = zeros(ts.stages)
    for s in 1:ts.stages
        ystage = 1.0
        for j in 1:(s - 1)
            ystage += dt * ts.A_explicit[s, j] * k[j]
        end
        k[s] = ystage
    end
    expected = 1.0 + dt * sum(ts.b_explicit .* k)

    result = state.history[end][1]
    ensure_layout!(result, :g)
    @test get_grid_data(result) ≈ fill(expected, 8) atol=20eps(Float64)
end

@testset "2D field-native RK reuses bounded workspace" begin
    for ts in (RK111(), RK222(), RK443())
        @test Tarang._workspace_count(ts) == ts.stages + 1
    end

    n = 16
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; dtype=Float64, device=CPU())
    xb = RealFourier(coords["x"]; size=n, bounds=(0.0, 2pi), dealias=3/2)
    yb = RealFourier(coords["y"]; size=n, bounds=(0.0, 2pi), dealias=3/2)
    domain = Domain(dist, (xb, yb))

    zeta = ScalarField(domain, "zeta")
    psi = ScalarField(domain, "psi")
    velocity = VectorField(domain, "u")
    tau_psi = ScalarField(dist, "tau_psi", (), Float64)
    problem = IVP([zeta, psi, velocity, tau_psi])
    add_parameters!(problem; nu=1e-8, drag=1e-3)
    add_equation!(problem, "dt(zeta) = -u⋅∇(zeta) - drag*zeta - nu*Δ⁴(zeta)")
    add_equation!(problem, "Δ(psi) + tau_psi - zeta = 0")
    add_equation!(problem, "u - skew(grad(psi)) = 0")
    add_bc!(problem, "integ(psi) = 0")

    dt = 1e-3
    solver = InitialValueSolver(problem, RK222(); dt)
    x = Tarang.get_grid_coordinates(domain; on_device=false)["x"]
    y = Tarang.get_grid_coordinates(domain; on_device=false)["y"]
    zeta["g"] = 1e-3 .* (sin.(x) .* cos.(y'))
    state = Tarang._ensure_timestepper_state!(solver, dt)

    for _ in 1:6
        _advance_field_rk!(solver, state)
    end
    recycled = get(state.timestepper_data, :explicit_field_rk_recycle, nothing)
    _advance_field_rk!(solver, state)
    @test state.history[end] === recycled

    GC.gc()
    allocated = @allocated _advance_field_rk!(solver, state)
    @info "2D field-native RK warmed host allocation" bytes=allocated
    @test allocated < 100_000
end
