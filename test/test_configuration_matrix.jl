"""
Configuration matrix: every supported (basis combination) × (problem type) ×
(timestepper) cell must either compute its manufactured solution correctly, or
refuse loudly. There is no third option.

WHY THIS FILE EXISTS. Every serious correctness bug found in this project has the
same shape — a configuration that is not covered by a test degrades instead of
refusing, and returns a plausible number:

  * a pure-Fourier LBVP dropped its entire RHS and returned exactly zero;
  * multistep timesteppers on GPU/MPI collapsed to forward Euler, silently
    turning order 2/3/4 into order 1;
  * a block that did not fit its slot in the operator was skipped, and the
    truncated system was solved anyway;
  * the global forcing vector wrote any non-constant RHS as zero.

None of these were unlucky. The tests that existed were written alongside the
configuration the author happened to be using — every field-RHS BVP test used a
mixed Fourier×Chebyshev basis, so the pure-Fourier cell was never exercised, and
that is precisely where the bug lived. The gaps were systematic, so the guard has
to be systematic too.

HOW TO USE IT. Each cell declares its expectation:

  `:solves`  — must produce the manufactured solution to the stated tolerance.
               Returning zero, or a wrong value, fails.
  `:refuses` — must raise. A cell that starts SOLVING also fails, which forces
               whoever implements it to come here and change the expectation
               deliberately rather than letting new behaviour appear unnoticed.

Adding a basis, a problem type or a timestepper means adding its row. A cell you
cannot make `:solves` is not a failure — `:refuses` is a legitimate, documented
answer. Silently returning a number is not.

Tolerances are loose enough to survive platform FP differences and tight enough
that a dropped forcing or a collapsed order cannot hide: every `:solves` cell also
asserts the answer is non-trivial, because a zero solution passes any comparison
against a zero reference.
"""

using Test
using Tarang

# ---------------------------------------------------------------------------
# Builders. Each returns (computed, expected) on the grid, or throws.
# ---------------------------------------------------------------------------

"""1D pure-Fourier IVP: dt(u) = κ∇²u, u₀ = sin x  ->  sin(x)e^{-κt}. Spatially exact."""
function _ivp_fourier_1d(stepper; N = 16, κ = 0.1, tfinal = 0.5, dt = 0.01)
    domain = PeriodicDomain(N)
    u = ScalarField(domain, "u"); set!(u, (x,) -> sin(x))
    prob = IVP([u]); add_parameters!(prob, kappa = κ)
    add_equation!(prob, "dt(u) = kappa*lap(u)")
    solver = InitialValueSolver(prob, stepper; dt)
    for _ in 1:round(Int, tfinal / dt); step!(solver, dt); end
    f = solver.state[1]; ensure_layout!(f, :g)
    xs = [2π * (i - 1) / N for i in 1:N]
    return real.(Array(get_grid_data(f))), sin.(xs) .* exp(-κ * tfinal)
end

"""2D pure-Fourier IVP: u₀ = sin(x)cos(y), decays at e^{-2κt}."""
function _ivp_fourier_2d(stepper; N = 16, κ = 0.1, tfinal = 0.5, dt = 0.01)
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; dtype = Float64, architecture = CPU())
    xb = RealFourier(coords["x"]; size = N, bounds = (0.0, 2π))
    yb = RealFourier(coords["y"]; size = N, bounds = (0.0, 2π))
    domain = Domain(dist, (xb, yb))
    u = ScalarField(domain, "u"); set!(u, (x, y) -> sin(x) * cos(y))
    prob = IVP([u]); add_parameters!(prob, kappa = κ)
    add_equation!(prob, "dt(u) = kappa*lap(u)")
    solver = InitialValueSolver(prob, stepper; dt)
    for _ in 1:round(Int, tfinal / dt); step!(solver, dt); end
    f = solver.state[1]; ensure_layout!(f, :g)
    g = [2π * (i - 1) / N for i in 1:N]
    return real.(Array(get_grid_data(f))),
           [sin(x) * cos(y) * exp(-2κ * tfinal) for x in g, y in g]
end

"""1D pure-Fourier LBVP with a field RHS. A periodic axis has no boundary, so the
point BC has nowhere to place a tau row and the operator cannot be assembled."""
function _lbvp_fourier_1d(; N = 16)
    domain = PeriodicDomain(N)
    u = ScalarField(domain, "u")
    f = ScalarField(domain, "f"); set!(f, (x,) -> sin(x))
    prob = LBVP([u]); add_parameters!(prob, f = f)
    add_equation!(prob, "lap(u) = f"); add_bc!(prob, "u(x=0) = 0")
    solver = BoundaryValueSolver(prob); solve!(solver)
    uu = solver.state[1]; ensure_layout!(uu, :g)
    xs = [2π * (i - 1) / N for i in 1:N]
    return real.(Array(get_grid_data(uu))), -sin.(xs)
end

"""1D pure-Chebyshev LBVP: u'' = -π²sin(πz), u(0)=u(1)=0  ->  u = sin(πz)."""
function _lbvp_cheb_1d(; Nz = 24)
    coords = CartesianCoordinates("z")
    dist = Distributor(coords; dtype = Float64, architecture = CPU())
    zb = ChebyshevT(coords["z"]; size = Nz, bounds = (0.0, 1.0))
    domain = Domain(dist, (zb,))
    u = ScalarField(domain, "u")
    fld = ScalarField(domain, "f"); set!(fld, (z,) -> -π^2 * sin(π * z))
    t1 = ScalarField(dist, "t1", (), Float64)
    t2 = ScalarField(dist, "t2", (), Float64)
    lb2 = derivative_basis(zb, 2)
    prob = LBVP([u, t1, t2]); prob.namespace["f"] = fld
    add_parameters!(prob; l1 = lift(t1, lb2, -1), l2 = lift(t2, lb2, -2))
    add_equation!(prob, "lap(u) + l1 + l2 = f")
    add_bc!(prob, "u(z=0) = 0"); add_bc!(prob, "u(z=1) = 0")
    solver = BoundaryValueSolver(prob); solve!(solver)
    ensure_layout!(u, :g)
    zs = [0.5 * (1 - cos(π * (k - 1) / (Nz - 1))) for k in 1:Nz]
    return real.(Array(get_grid_data(u))), sin.(π .* zs)
end

"""Fourier×Chebyshev LBVP with a field RHS: u = sin(πz/Lz)cos(2x)."""
function _lbvp_fourier_cheb(; Nx = 8, Nz = 24, Lz = 1.0)
    coords = CartesianCoordinates("x", "z")
    dist = Distributor(coords; dtype = Float64, device = CPU())
    xb = RealFourier(coords["x"]; size = Nx, bounds = (0.0, 2π), dealias = 1.0)
    zb = ChebyshevT(coords["z"]; size = Nz, bounds = (0.0, Lz), dealias = 1.0)
    dom = Domain(dist, (xb, zb))
    u = ScalarField(dom, "u")
    t1 = ScalarField(dist, "t1", (xb,), Float64)
    t2 = ScalarField(dist, "t2", (xb,), Float64)
    fld = ScalarField(dom, "f"); lb2 = derivative_basis(zb, 2)
    xg = vec(Array(Tarang.local_grid(xb, dist, 1)))
    zg = vec(Array(Tarang.local_grid(zb, dist, 1)))
    uex(x, z) = sin(π * z / Lz) * cos(2x)
    λ = (π / Lz)^2 + 4
    ensure_layout!(fld, :g); fd = get_grid_data(fld)
    for i in 1:Nx, k in 1:Nz; fd[i, k] = -λ * uex(xg[i], zg[k]); end
    prob = LBVP([u, t1, t2]); prob.namespace["f"] = fld
    add_parameters!(prob; Lz = Lz, l1 = lift(t1, lb2, -1), l2 = lift(t2, lb2, -2))
    add_equation!(prob, "Δ(u) + l1 + l2 = f")
    add_bc!(prob, "u(z=0) = 0"); add_bc!(prob, "u(z=Lz) = 0")
    solver = BoundaryValueSolver(prob); solve!(solver)
    ensure_layout!(u, :g)
    return Array(get_grid_data(u)), [uex(xg[i], zg[k]) for i in 1:Nx, k in 1:Nz]
end

"""Chebyshev-Fourier LBVP parameterised over the two knobs that decide whether the
configuration is supported: which coordinate comes FIRST, and whether the tau
fields carry the Fourier axis (`(xb,)`, one tau per mode) or are scalars (`()`).

Both knobs matter and neither is visible in a single-ordering test. Chebyshev-first
is the only ordering MPI accepts, so it is the ordering distributed users must
write — yet serially it is the one combination below that breaks, and it breaks on
a raw `DimensionMismatch` out of the block assembler rather than a stated refusal.
The forcing is x-dependent throughout, so a cell that quietly loses the `cos(2x)`
structure is a wrong answer rather than a coarser one.

Mirrors `cm_lbvp_cheb_fourier` in test_mpi_configuration_matrix.jl; the serial and
distributed expectations for the same four cells are meant to be read side by side.
"""
function _lbvp_cheb_fourier_ordered(; cheb_first::Bool, tau_per_mode::Bool,
                                      Nz = 16, Nx = 8, Lz = 1.0)
    names = cheb_first ? ("z", "x") : ("x", "z")
    coords = CartesianCoordinates(names...)
    dist = Distributor(coords; dtype = Float64, device = CPU())
    zb = ChebyshevT(coords["z"]; size = Nz, bounds = (0.0, Lz), dealias = 1.0)
    xb = RealFourier(coords["x"]; size = Nx, bounds = (0.0, 2π), dealias = 1.0)
    dom = Domain(dist, cheb_first ? (zb, xb) : (xb, zb))
    lb2 = derivative_basis(zb, 2)
    u = ScalarField(dom, "u")
    tau_axes = tau_per_mode ? (xb,) : ()
    t1 = ScalarField(dist, "t1", tau_axes, Float64)
    t2 = ScalarField(dist, "t2", tau_axes, Float64)
    fld = ScalarField(dom, "f")
    zg = vec(Array(Tarang.local_grid(zb, dist, 1)))
    xg = vec(Array(Tarang.local_grid(xb, dist, 1)))
    uex(z, x) = sin(π * z / Lz) * cos(2x)
    λ = (π / Lz)^2 + 4
    ensure_layout!(fld, :g); fd = get_grid_data(fld)
    if cheb_first
        for k in axes(fd, 1), i in axes(fd, 2); fd[k, i] = -λ * uex(zg[k], xg[i]); end
    else
        for i in axes(fd, 1), k in axes(fd, 2); fd[i, k] = -λ * uex(zg[k], xg[i]); end
    end
    prob = LBVP([u, t1, t2]); prob.namespace["f"] = fld
    add_parameters!(prob; Lz = Lz, l1 = lift(t1, lb2, -1), l2 = lift(t2, lb2, -2))
    add_equation!(prob, "Δ(u) + l1 + l2 = f")
    add_bc!(prob, "u(z=0) = 0"); add_bc!(prob, "u(z=Lz) = 0")
    solver = BoundaryValueSolver(prob); solve!(solver)
    ensure_layout!(u, :g)
    zs = [0.5 * Lz * (1 - cos(π * (k - 1) / (Nz - 1))) for k in 1:Nz]
    xs = [2π * (i - 1) / Nx for i in 1:Nx]
    expected = cheb_first ? [uex(z, x) for z in zs, x in xs] :
                            [uex(z, x) for x in xs, z in zs]
    return Array(get_grid_data(u)), expected
end


"""3D pure-Fourier IVP: u₀ = sin(x)cos(y)sin(z), decays at e^{-3κt}.

Three Fourier axes is the case a 1D or 2D cell cannot reach: the transform chain,
the wavenumber grid and the dealiasing all index differently once a third axis
exists, and a bug there is invisible below 3D."""
function _ivp_fourier_3d(stepper; N = 8, κ = 0.1, tfinal = 0.2, dt = 0.005)
    coords = CartesianCoordinates("x", "y", "z")
    dist = Distributor(coords; dtype = Float64, architecture = CPU())
    bases = (RealFourier(coords["x"]; size = N, bounds = (0.0, 2π)),
             RealFourier(coords["y"]; size = N, bounds = (0.0, 2π)),
             RealFourier(coords["z"]; size = N, bounds = (0.0, 2π)))
    domain = Domain(dist, bases)
    u = ScalarField(domain, "u"); set!(u, (x, y, z) -> sin(x) * cos(y) * sin(z))
    prob = IVP([u]); add_parameters!(prob, kappa = κ)
    add_equation!(prob, "dt(u) = kappa*lap(u)")
    solver = InitialValueSolver(prob, stepper; dt)
    for _ in 1:round(Int, tfinal / dt); step!(solver, dt); end
    f = solver.state[1]; ensure_layout!(f, :g)
    g = [2π * (i - 1) / N for i in 1:N]
    return real.(Array(get_grid_data(f))),
           [sin(x) * cos(y) * sin(z) * exp(-3κ * tfinal) for x in g, y in g, z in g]
end

"""1D pure-Chebyshev IVP with a tau/lift formulation: u = sin(πz)e^{-κπ²t}.

A COUPLED direction with no Fourier axis at all. Every other IVP cell has at least
one separable axis, so this is the only one exercising the per-mode machinery with
a single mode."""
function _ivp_cheb_1d(stepper; Nz = 24, κ = 0.05, tfinal = 0.1, dt = 0.002)
    coords = CartesianCoordinates("z")
    dist = Distributor(coords; dtype = Float64, architecture = CPU())
    zb = ChebyshevT(coords["z"]; size = Nz, bounds = (0.0, 1.0))
    domain = Domain(dist, (zb,))
    b = ScalarField(domain, "b"); set!(b, (z,) -> sin(π * z))
    tau1 = ScalarField(dist, "tau1", (), Float64)
    tau2 = ScalarField(dist, "tau2", (), Float64)
    lb = derivative_basis(zb, 1); tau_lift(A) = lift(A, lb, -1)
    prob = IVP([b, tau1, tau2])
    add_parameters!(prob, kappa = κ, tau_lift = tau_lift)
    add_equation!(prob, "dt(b) - kappa*lap(b) + tau_lift(tau1) + tau_lift(tau2) = 0")
    add_bc!(prob, "b(z=0) = 0"); add_bc!(prob, "b(z=1) = 0")
    solver = InitialValueSolver(prob, stepper; dt)
    for _ in 1:round(Int, tfinal / dt); step!(solver, dt); end
    f = solver.state[1]; ensure_layout!(f, :g)
    zs = [0.5 * (1 - cos(π * (k - 1) / (Nz - 1))) for k in 1:Nz]
    return real.(Array(get_grid_data(f))), sin.(π .* zs) .* exp(-κ * π^2 * tfinal)
end

"""Fourier×Chebyshev IVP with tau/lift: u = sin(πz)cos(x)e^{-κ(π²+1)t}.

The production geometry — a separable axis plus a coupled one — driven as an
INITIAL-value problem. The existing coupled cell is a steady BVP, so the per-mode
timestepping path was uncovered."""
function _ivp_fourier_cheb(stepper; Nx = 8, Nz = 20, κ = 0.05, tfinal = 0.1, dt = 0.002)
    coords = CartesianCoordinates("x", "z")
    dist = Distributor(coords; dtype = Float64, architecture = CPU())
    xb = RealFourier(coords["x"]; size = Nx, bounds = (0.0, 2π))
    zb = ChebyshevT(coords["z"]; size = Nz, bounds = (0.0, 1.0))
    domain = Domain(dist, (xb, zb))
    b = ScalarField(domain, "b"); set!(b, (x, z) -> sin(π * z) * cos(x))
    tau1 = ScalarField(dist, "tau1", (xb,), Float64)
    tau2 = ScalarField(dist, "tau2", (xb,), Float64)
    lb = derivative_basis(zb, 1); tau_lift(A) = lift(A, lb, -1)
    _, ez = unit_vector_fields(coords, dist)
    grad_b = grad(b) + ez * tau_lift(tau1)
    prob = IVP([b, tau1, tau2])
    add_parameters!(prob, kappa = κ, ez = ez, grad_b = grad_b, tau_lift = tau_lift)
    add_equation!(prob, "dt(b) - kappa*div(grad_b) + tau_lift(tau2) = 0")
    add_bc!(prob, "b(z=0) = 0"); add_bc!(prob, "b(z=1) = 0")
    solver = InitialValueSolver(prob, stepper; dt)
    for _ in 1:round(Int, tfinal / dt); step!(solver, dt); end
    f = solver.state[1]; ensure_layout!(f, :g)
    xs = [2π * (i - 1) / Nx for i in 1:Nx]
    zs = [0.5 * (1 - cos(π * (k - 1) / (Nz - 1))) for k in 1:Nz]
    λ = κ * (π^2 + 1)
    return Array(get_grid_data(f)),
           [sin(π * z) * cos(x) * exp(-λ * tfinal) for x in xs, z in zs]
end

# ---------------------------------------------------------------------------
# The matrix. (label, builder, expectation, tolerance)
# ---------------------------------------------------------------------------

const MATRIX = [
    # --- IVP across timesteppers. The multistep rows are the collapse-to-Euler
    #     regression: on a path with no assembled global matrix these silently fell
    #     back to their first-order member, which a fixed-dt value check catches.
    ("IVP  1D Fourier  RK222",   () -> _ivp_fourier_1d(RK222()),   :solves, 1e-6),
    ("IVP  1D Fourier  RK443",   () -> _ivp_fourier_1d(RK443()),   :solves, 1e-6),
    ("IVP  1D Fourier  CNAB2",   () -> _ivp_fourier_1d(CNAB2()),   :solves, 1e-5),
    ("IVP  1D Fourier  SBDF2",   () -> _ivp_fourier_1d(SBDF2()),   :solves, 1e-5),
    ("IVP  1D Fourier  SBDF3",   () -> _ivp_fourier_1d(SBDF3()),   :solves, 1e-5),
    ("IVP  1D Fourier  SBDF4",   () -> _ivp_fourier_1d(SBDF4()),   :solves, 1e-5),
    ("IVP  2D Fourier  RK222",   () -> _ivp_fourier_2d(RK222()),   :solves, 1e-6),
    ("IVP  2D Fourier  SBDF2",   () -> _ivp_fourier_2d(SBDF2()),   :solves, 1e-5),

    # --- Coupled and 3-D IVP cells. Every pre-existing IVP row is pure Fourier in
    #     1D or 2D, so the per-mode TIMESTEPPING path (as opposed to the steady BVP
    #     one) and everything above two axes were uncovered. Tolerances are the
    #     measured error rounded up, not round numbers.
    ("IVP  3D Fourier  RK222",        () -> _ivp_fourier_3d(RK222()),   :solves, 1e-6),
    ("IVP  3D Fourier  SBDF2",        () -> _ivp_fourier_3d(SBDF2()),   :solves, 1e-5),
    ("IVP  1D Chebyshev+tau RK222",   () -> _ivp_cheb_1d(RK222()),      :solves, 1e-7),
    ("IVP  1D Chebyshev+tau SBDF2",   () -> _ivp_cheb_1d(SBDF2()),      :solves, 1e-5),
    ("IVP  Fourier×Chebyshev RK222",  () -> _ivp_fourier_cheb(RK222()), :solves, 1e-7),
    ("IVP  Fourier×Chebyshev RK443",  () -> _ivp_fourier_cheb(RK443()), :solves, 1e-9),
    ("IVP  Fourier×Chebyshev SBDF2",  () -> _ivp_fourier_cheb(SBDF2()), :solves, 1e-5),

    # --- LBVP across basis combinations.
    ("LBVP 1D Chebyshev",        _lbvp_cheb_1d,                    :solves, 1e-8),
    ("LBVP Fourier×Chebyshev",   _lbvp_fourier_cheb,               :solves, 1e-8),
    # A periodic direction has no boundary, so a point BC has nowhere to place its
    # tau row. This must REFUSE. It used to return exactly zero.
    ("LBVP 1D Fourier",          _lbvp_fourier_1d,                 :refuses, 0.0),

    # --- LBVP across (coordinate order) × (tau shape). Ordering was an invisible
    #     axis here: every mixed-basis cell above happens to put Fourier first, so
    #     nothing exercised the Chebyshev-first ordering that MPI *requires*. Three
    #     of these four solve; the fourth is the one distributed users must write.
    ("LBVP Fourier-first scalar tau",
        () -> _lbvp_cheb_fourier_ordered(cheb_first = false, tau_per_mode = false), :solves, 1e-8),
    ("LBVP Fourier-first per-mode tau",
        () -> _lbvp_cheb_fourier_ordered(cheb_first = false, tau_per_mode = true),  :solves, 1e-8),
    ("LBVP Cheb-first scalar tau",
        () -> _lbvp_cheb_fourier_ordered(cheb_first = true,  tau_per_mode = false), :solves, 1e-8),
    # KNOWN DEFECT, pinned rather than hidden. Chebyshev-first with per-mode taus
    # is the natural way to write a distributed mixed-basis BVP, and it fails out of
    # the block assembler with a bare `DimensionMismatch` — an internal shape error,
    # not a refusal that names the unsupported combination. The identical
    # construction is accepted Fourier-first (row above), so this is an ordering
    # inconsistency rather than a real restriction. Scalar taus solve the same
    # x-dependent problem, so there is a working alternative and no silent wrong
    # answer; when the assembler learns this case, change this row to `:solves`.
    ("LBVP Cheb-first per-mode tau",
        () -> _lbvp_cheb_fourier_ordered(cheb_first = true,  tau_per_mode = true),  :refuses, 0.0),
]

@testset "Configuration matrix: every cell solves correctly or refuses" begin
    for (label, build, expectation, tol) in MATRIX
        @testset "$label" begin
            if expectation === :refuses
                # A cell that starts solving must be re-declared here deliberately.
                @test_throws Exception build()
            else
                got, want = build()
                # A zero solution matches a zero reference, so assert non-triviality
                # first — this is what turns "dropped the forcing" from a pass into
                # a failure.
                @test maximum(abs, want) > 0.1
                @test maximum(abs, got) > 0.1
                @test maximum(abs, got .- want) < tol
            end
        end
    end
end

@testset "Every matrix cell is exercised exactly once" begin
    # A duplicated or silently-dropped label would weaken the guard without failing
    # anything, so pin the inventory itself.
    labels = [row[1] for row in MATRIX]
    @test length(unique(labels)) == length(labels)
    @test length(MATRIX) >= 15
    # Both expectations must be represented: an all-`:solves` table would mean the
    # refusal side of the contract is untested, and an all-`:refuses` table would
    # mean nothing is verified numerically.
    @test any(row -> row[3] === :solves, MATRIX)
    @test any(row -> row[3] === :refuses, MATRIX)
end
