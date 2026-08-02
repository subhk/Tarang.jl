# Configuration matrix, DISTRIBUTED half: every supported (basis combination) ×
# (problem type) × (timestepper) cell must either compute its manufactured
# solution correctly, or refuse loudly. There is no third option.
#
# WHY THIS FILE EXISTS. test_configuration_matrix.jl pins the serial-CPU cells and
# explains the reasoning: every serious correctness bug in this project came from a
# configuration nobody had a test for, which degraded instead of refusing and
# returned a plausible number. That argument applies with more force here, because
# the backend is the axis the serial matrix cannot see. The bug history is almost
# entirely backend-specific:
#
#   * explicit multistep on distributed pure-Fourier collapsed to forward Euler,
#     turning order 2/3/4 into order 1 with no warning;
#   * the GPU implicit guard read IR that only the CPU assembly path builds, so a
#     heat equation ran inviscid on device;
#   * the steady BVP per-mode gather skipped the Cheb-Fourier solve-layout
#     transpose, which only misbehaves at np>=2.
#
# Each of those lived in a cell that the serial suite covered and the distributed
# suite did not. The MPI tests that existed were written one bug at a time, each
# pinning the single configuration its author was holding — so the coverage was
# shaped like the bug list, not like the feature matrix. This file is shaped like
# the feature matrix.
#
# HOW TO USE IT. Same contract as the serial matrix:
#
#   `:solves`  — must reproduce the manufactured solution to the stated tolerance,
#                and must be non-trivial (a zero field matches a zero reference,
#                so every solving cell also asserts the answer is not ~0).
#   `:refuses` — must raise. A cell that starts SOLVING fails too, which forces
#                whoever implements it to come here and change the expectation
#                deliberately instead of letting new behaviour appear unnoticed.
#
# The `:refuses` rows are not placeholders — they are the documented distributed
# limitations, and pinning them is what stops a future change from converting a
# loud refusal into a silent degradation. That conversion is precisely how the
# multistep collapse and the GPU L-drop happened.
#
# Run with:  mpiexec -n 2 julia --project test/test_mpi_configuration_matrix.jl
# (also exercised at 4 ranks by test/run_mpi_ci.jl)

using Test
using MPI
MPI.Initialized() || MPI.Init()
using Tarang
using PencilArrays

const CM_COMM = MPI.COMM_WORLD
const CM_RANK = MPI.Comm_rank(CM_COMM)
const CM_NP   = MPI.Comm_size(CM_COMM)

if CM_NP < 2
    CM_RANK == 0 && @warn "Distributed configuration matrix needs >= 2 ranks; got $CM_NP"
    MPI.Finalize(); exit(0)
end

const CM_KAPPA = 0.1
const CM_TFIN  = 0.5

# ---------------------------------------------------------------------------
# Distributed helpers. A solved field is compared on each rank's OWN slice and
# the error is reduced, so the check covers every rank rather than rank 0's
# corner of the domain.
# ---------------------------------------------------------------------------

"Write the local slice of a global array into a (possibly decomposed) field."
function cm_set_local!(field, full)
    ensure_layout!(field, :g)
    gd = get_grid_data(field)
    if gd isa PencilArrays.PencilArray
        parent(gd) .= full[PencilArrays.pencil(gd).axes_local...]
    else
        gd .= full
    end
    ensure_layout!(field, :c)
    return field
end

"Max |field - full| over the local slice, reduced across all ranks."
function cm_error(field, full)
    ensure_layout!(field, :g)
    gd = get_grid_data(field)
    local_err = if gd isa PencilArrays.PencilArray
        maximum(abs.(parent(gd) .- full[PencilArrays.pencil(gd).axes_local...]))
    else
        maximum(abs.(Array(gd) .- full))
    end
    return MPI.Allreduce(local_err, MPI.MAX, CM_COMM)
end

"Global max |field|. Guards the zero solution, which matches any zero reference."
function cm_magnitude(field)
    ensure_layout!(field, :g)
    gd = get_grid_data(field)
    local_mag = gd isa PencilArrays.PencilArray ? maximum(abs, parent(gd)) :
                                                  maximum(abs, Array(gd))
    return MPI.Allreduce(local_mag, MPI.MAX, CM_COMM)
end

fourier_grid(N) = [2π * (i - 1) / N for i in 1:N]

# ---------------------------------------------------------------------------
# Builders. Each returns (field, expected_global_array) or throws.
# ---------------------------------------------------------------------------

"""2D distributed pure-Fourier IVP: dt(u) = κ∇²u, u₀ = sin(x)cos(y).

A single eigenmode, so this is spatially exact and decays as exp(-2κt): any error
is the time integration alone. That is what makes it a collapse detector — a
scheme that quietly degrades to its first-order member fails the tolerance
instead of merely converging more slowly.
"""
function cm_ivp_fourier_2d(stepper; N = 16, dt = 0.01)
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; dtype = Float64, architecture = CPU())
    xb = RealFourier(coords["x"]; size = N, bounds = (0.0, 2π))
    yb = RealFourier(coords["y"]; size = N, bounds = (0.0, 2π))
    dom = Domain(dist, (xb, yb))
    u = ScalarField(dom, "u")
    g = fourier_grid(N)
    u0 = [sin(x) * cos(y) for x in g, y in g]
    cm_set_local!(u, u0)
    prob = IVP([u]); add_parameters!(prob, kappa = CM_KAPPA)
    add_equation!(prob, "dt(u) = kappa*lap(u)")
    solver = InitialValueSolver(prob, stepper; dt)
    for _ in 1:round(Int, CM_TFIN / dt); step!(solver, dt); end
    return solver.state[1], u0 .* exp(-2 * CM_KAPPA * CM_TFIN)
end

"""3D distributed pure-Fourier IVP. At 4 ranks this is a 2D pencil mesh, which is
a different decomposition code path from the 2D slab above."""
function cm_ivp_fourier_3d(stepper; N = 8, dt = 0.01)
    coords = CartesianCoordinates("x", "y", "z")
    dist = Distributor(coords; dtype = Float64, architecture = CPU())
    bases = ntuple(i -> RealFourier(coords[("x", "y", "z")[i]]; size = N, bounds = (0.0, 2π)), 3)
    dom = Domain(dist, bases)
    u = ScalarField(dom, "u")
    g = fourier_grid(N)
    u0 = [sin(x) * cos(y) * sin(z) for x in g, y in g, z in g]
    cm_set_local!(u, u0)
    prob = IVP([u]); add_parameters!(prob, kappa = CM_KAPPA)
    add_equation!(prob, "dt(u) = kappa*lap(u)")
    solver = InitialValueSolver(prob, stepper; dt)
    for _ in 1:round(Int, CM_TFIN / dt); step!(solver, dt); end
    return solver.state[1], u0 .* exp(-3 * CM_KAPPA * CM_TFIN)
end

"""Chebyshev×Fourier LBVP, Chebyshev axis FIRST: Δu + lift(τ₁) + lift(τ₂) = f,
u(z=0)=u(z=Lz)=0, manufactured u = sin(πz/Lz)cos(2x).

`cheb_first` and `tau_per_mode` are the two knobs that decide whether this
configuration is supported at all; the matrix rows below pin each combination.
The forcing is x-dependent in every case, so a cell that loses the cos(2x)
structure is a wrong answer, not a coarser one.
"""
function cm_lbvp_cheb_fourier(; cheb_first::Bool, tau_per_mode::Bool,
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
    zfull = [0.5 * Lz * (1 - cos(π * (k - 1) / (Nz - 1))) for k in 1:Nz]
    xfull = fourier_grid(Nx)
    expected = cheb_first ? [uex(z, x) for z in zfull, x in xfull] :
                            [uex(z, x) for x in xfull, z in zfull]
    return u, expected
end

"""2D distributed pure-Fourier LBVP with a field RHS.

Both axes are periodic, so the point BC `u(x=0)=0` has no boundary to sit on and
its tau row does not fit the operator. This must REFUSE: the block-mismatch fix
(commit 8ac8f80f3) exists because the assembler used to skip the misfitting block
and solve the truncated system anyway, returning a confident wrong answer.
"""
function cm_lbvp_fourier_2d(; N = 8)
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; dtype = Float64, architecture = CPU())
    xb = RealFourier(coords["x"]; size = N, bounds = (0.0, 2π))
    yb = RealFourier(coords["y"]; size = N, bounds = (0.0, 2π))
    dom = Domain(dist, (xb, yb))
    u = ScalarField(dom, "u"); f = ScalarField(dom, "f")
    g = fourier_grid(N)
    cm_set_local!(f, [sin(x) * cos(y) for x in g, y in g])
    prob = LBVP([u]); add_parameters!(prob, f = f)
    add_equation!(prob, "lap(u) = f"); add_bc!(prob, "u(x=0) = 0")
    solver = BoundaryValueSolver(prob); solve!(solver)
    return u, [-sin(x) * cos(y) / 2 for x in g, y in g]
end

# ---------------------------------------------------------------------------
# The matrix. (label, builder, expectation, tolerance)
# ---------------------------------------------------------------------------

const CM_MATRIX = [
    # --- IVP across timesteppers on a decomposed pure-Fourier domain. This is the
    #     configuration where multistep silently became forward Euler: it builds no
    #     per-mode subproblems and cannot factorize a global matrix, so every scheme
    #     is served by the matrix-free field path. A fixed-dt value check catches an
    #     order collapse that a convergence-rate check can miss.
    ("IVP  2D Fourier  RK222",  () -> cm_ivp_fourier_2d(RK222()), :solves, 1e-6),
    ("IVP  2D Fourier  RK443",  () -> cm_ivp_fourier_2d(RK443()), :solves, 1e-6),
    ("IVP  2D Fourier  CNAB2",  () -> cm_ivp_fourier_2d(CNAB2()), :solves, 1e-5),
    ("IVP  2D Fourier  SBDF2",  () -> cm_ivp_fourier_2d(SBDF2()), :solves, 1e-5),
    ("IVP  2D Fourier  SBDF3",  () -> cm_ivp_fourier_2d(SBDF3()), :solves, 1e-5),
    ("IVP  2D Fourier  SBDF4",  () -> cm_ivp_fourier_2d(SBDF4()), :solves, 1e-5),

    # --- 3D, which is a 2D pencil mesh at 4 ranks rather than a slab.
    ("IVP  3D Fourier  RK222",  () -> cm_ivp_fourier_3d(RK222()), :solves, 1e-6),
    ("IVP  3D Fourier  SBDF2",  () -> cm_ivp_fourier_3d(SBDF2()), :solves, 1e-5),

    # --- LBVP across (coordinate order) × (tau shape). Only one of the four is
    #     supported distributed, and the other three must say so rather than
    #     produce a number. Each refusal has a distinct cause, noted per row.
    #
    # Supported: Chebyshev first, scalar taus. The decomposed trailing axis is the
    # Fourier one, and the solve-layout transpose covers the per-mode gather.
    ("LBVP Cheb-first  scalar tau",
        () -> cm_lbvp_cheb_fourier(cheb_first = true,  tau_per_mode = false), :solves, 1e-8),
    # Chebyshev first, per-mode taus: rejected. A tau field carrying only the
    # Fourier axis is a 1D field, and a 1D distributed FFT needs global data.
    # Serially this same construction dies on a raw DimensionMismatch out of the
    # block assembler, so it is unsupported everywhere, not merely under MPI —
    # scalar taus solve the identical x-dependent problem, so nothing is lost.
    ("LBVP Cheb-first  per-mode tau",
        () -> cm_lbvp_cheb_fourier(cheb_first = true,  tau_per_mode = true),  :refuses, 0.0),
    # Fourier first: rejected because the decomposed trailing axis is then the
    # Chebyshev one, which has no distributed transform. Solves serially, which is
    # exactly why it needs a row here — the serial matrix covers it and would not
    # notice this becoming a silent CPU fallback.
    ("LBVP Fourier-first scalar tau",
        () -> cm_lbvp_cheb_fourier(cheb_first = false, tau_per_mode = false), :refuses, 0.0),
    ("LBVP Fourier-first per-mode tau",
        () -> cm_lbvp_cheb_fourier(cheb_first = false, tau_per_mode = true),  :refuses, 0.0),

    # A point BC on an all-periodic domain has nowhere to put its tau row.
    ("LBVP 2D Fourier",         cm_lbvp_fourier_2d,               :refuses, 0.0),
]

@testset "Distributed configuration matrix: solves correctly or refuses (np=$CM_NP)" begin
    for (label, build, expectation, tol) in CM_MATRIX
        @testset "$label" begin
            if expectation === :refuses
                # A cell that starts solving must be re-declared here deliberately.
                @test_throws Exception build()
            else
                field, expected = build()
                err = cm_error(field, expected)
                mag = cm_magnitude(field)
                CM_RANK == 0 && @info "matrix cell" label err mag
                # Non-triviality first: this is what turns "dropped the forcing"
                # from a pass into a failure.
                @test maximum(abs, expected) > 0.1
                @test mag > 0.1
                @test err < tol
            end
            MPI.Barrier(CM_COMM)
        end
    end
end

@testset "Every matrix cell is exercised exactly once (np=$CM_NP)" begin
    # A duplicated or silently-dropped label would weaken the guard without failing
    # anything, so pin the inventory itself.
    labels = [row[1] for row in CM_MATRIX]
    @test length(unique(labels)) == length(labels)
    @test length(CM_MATRIX) >= 13
    # Both expectations must be represented: an all-`:solves` table would mean the
    # refusal side of the contract is untested, and an all-`:refuses` table would
    # mean nothing is verified numerically.
    @test count(row -> row[3] === :solves,  CM_MATRIX) >= 9
    @test count(row -> row[3] === :refuses, CM_MATRIX) >= 4
end

@testset "Distributed multistep is not forward Euler in disguise (np=$CM_NP)" begin
    # The matrix rows above check each scheme against the exact solution at one dt.
    # This adds the differential check that named the original bug: with L = 0 on a
    # pure-Fourier problem, CNAB1 IS forward Euler, so it is the exact "collapsed"
    # reference every higher-order scheme must beat by a wide margin. A collapse
    # that stayed inside a loose tolerance above would still fail here.
    err_of(stepper, dt) = begin
        field, expected = cm_ivp_fourier_2d(stepper; dt)
        cm_error(field, expected)
    end
    dt = 0.025
    euler = err_of(CNAB1(), dt)
    for stepper in (CNAB2(), SBDF3(), SBDF4())
        e = err_of(stepper, dt)
        CM_RANK == 0 && @info "vs forward Euler" stepper e euler ratio = euler / e
        @test e < euler / 10
    end
end

MPI.Barrier(CM_COMM)
MPI.Finalized() || MPI.Finalize()
