# Guard: per-step allocation of a 3D mixed (Chebyshev × Fourier × Fourier) IVP.
#
# The per-mode gather/scatter used to fall off a cliff in 3D:
#
#   * `_subproblem_coeff_index` returns a tuple that mixes `Int` mode indices with `Colon`, so it
#     infers as `Any`. Every `view(cd, idxt...)` splat off it dynamic-dispatched and allocated a
#     SubArray wrapper — per field, per subproblem, per gather AND scatter, per stage. A strided
#     fast path existed, but it was gated on `ndims(cd) == 2`, so 3-D never took it. In 3-D the
#     local subproblem count is Nx·Ny/nprocs, i.e. thousands of calls per step.
#
#   * The 0-D (tau) branch built two SubArray wrappers to move ONE number.
#
#   * `all(b -> b === nothing, field.bases)` — the branch CONDITION — boxed on every call, because
#     `bases::Tuple{Vararg{Basis}}` is not concretely typed. It cannot even be true unless the
#     tuple is empty (the element type excludes `nothing`), so it was an allocation to ask a
#     question whose answer was `isempty`.
#
# All three are now a single memoized `(start, step, len)` strided run. Measured: 538,832 →
# 330,192 B/step (−39%).
#
# The values are unchanged either way, so no correctness test can catch a regression here —
# only an allocation count can.
using Test
using Tarang

@testset "3D mixed subproblem gather/scatter allocation" begin
    Nz, Nx, Ny = 12, 8, 8
    coords = CartesianCoordinates("z", "x", "y")      # Chebyshev FIRST (the MPI-supported layout)
    dist   = Distributor(coords; dtype=Float64, device=CPU())
    zb = ChebyshevT(coords["z"]; size=Nz, bounds=(0.0, 1.0))
    xb = RealFourier(coords["x"]; size=Nx, bounds=(0.0, 2π))
    yb = RealFourier(coords["y"]; size=Ny, bounds=(0.0, 2π))
    domain = Domain(dist, (zb, xb, yb))

    b  = ScalarField(domain, "b")
    t1 = ScalarField(dist, "tau_b1", (), Float64)     # 0-D taus (Chebyshev-first convention)
    t2 = ScalarField(dist, "tau_b2", (), Float64)
    ez, _, _ = unit_vector_fields(coords, dist)
    lb = derivative_basis(zb, 1)
    τ_lift(A) = lift(A, lb, -1)
    grad_b = grad(b) + ez * τ_lift(t1)

    pr = IVP([b, t1, t2])
    add_parameters!(pr, kappa=0.1, ez=ez, grad_b=grad_b, τ_lift=τ_lift)
    add_equation!(pr, "∂t(b) - kappa*div(grad_b) + τ_lift(tau_b2) = -b*∂x(b)")
    add_bc!(pr, "b(z=0) = 0")
    add_bc!(pr, "b(z=1) = 0")

    s = InitialValueSolver(pr, RK222(); dt=1e-3)
    @test s.rhs_plan.is_compiled                       # else the alloc number means nothing

    ensure_layout!(b, :g)
    zf = [0.5*(1-cos(π*(k-1)/(Nz-1))) for k in 1:Nz]
    get_grid_data(b) .= [sin(π*zf[i]) for i in 1:Nz, _ in 1:Nx, _ in 1:Ny]
    ensure_layout!(b, :c)

    for _ in 1:15; step!(s, 1e-3); end                 # warm up: caches/memos settle
    a = minimum((@allocated step!(s, 1e-3)) for _ in 1:5)

    # Measured 330,192 B/step after the fix, 538,832 before. The bound is deliberately loose
    # (x64 boxes more than arm64 — see the CI alloc lesson), but tight enough that reintroducing
    # the Any-splat or the boxing branch condition fails it.
    @test a < 420_000

    # And it must still step correctly.
    ensure_layout!(b, :g)
    @test isfinite(maximum(abs.(get_grid_data(b))))
    @test maximum(abs.(get_grid_data(b))) > 0.5        # diffusing, not collapsed to zero
end
