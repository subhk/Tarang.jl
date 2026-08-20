# Serial tests for field I/O and solver checkpoint/restart.
#
# Both `save_field` and `load_field!` were exported, untested, and broken:
# `save_field` called `ncwrite` with no preceding `nccreate` and threw on every
# call; `load_field!` called `rethrow` outside a catch block, which Julia rejects,
# destroying the NetCDF error it meant to surface.

using Test
using Tarang
using InteractiveUtils: subtypes

@testset "save_field / load_field! round-trip" begin
    dir = mktempdir()
    path = joinpath(dir, "field.nc")

    domain = PeriodicDomain(16)
    u = ScalarField(domain, "u")
    set!(u, (x,) -> sin(x) + 0.5cos(3x))
    ensure_layout!(u, :g)
    original = copy(get_grid_data(u))

    written = save_field(u, path, "u")
    @test isfile(written)

    v = ScalarField(domain, "v")
    load_field!(v, written, "u")
    ensure_layout!(v, :g)
    @test get_grid_data(v) == original
end

@testset "save_field accepts a path without the .nc suffix" begin
    dir = mktempdir()
    domain = PeriodicDomain(8)
    u = ScalarField(domain, "u")
    set!(u, (x,) -> cos(x))
    written = save_field(u, joinpath(dir, "nosuffix"), "u")
    @test endswith(written, ".nc")
    @test isfile(written)
end

@testset "load_field! surfaces the real error, not a rethrow failure" begin
    dir = mktempdir()
    domain = PeriodicDomain(8)
    v = ScalarField(domain, "v")
    err = try
        load_field!(v, joinpath(dir, "absent.nc"), "u")
        nothing
    catch e
        e
    end
    @test err !== nothing
    msg = sprint(showerror, err)
    @test !occursin("rethrow", msg)
    @test occursin("absent", msg) || occursin("no NetCDF files", msg)
end

@testset "load_field! rejects a shape mismatch instead of loading garbage" begin
    dir = mktempdir()
    path = joinpath(dir, "small.nc")
    small = ScalarField(PeriodicDomain(8), "u")
    set!(small, (x,) -> sin(x))
    save_field(small, path, "u")

    big = ScalarField(PeriodicDomain(16), "u")
    @test_throws ErrorException load_field!(big, path, "u")
end

# --- Fix round: save_field must not destroy previously-written variables ---
#
# `save_field` used to run `isfile(target) && rm(target)` on every call, which
# contradicts write_local_slab's own "additive" contract (see its docstring in
# src/tools/netcdf_slab_io.jl). This is not hypothetical: save_field_netcdf in
# src/extras/plot_tools.jl loops over a VectorField's components calling
# `save_field(component, filename, "component_$i")` with the SAME filename
# every iteration. Before this fix, iteration 2 deleted iteration 1's data, so
# only the last component survived even though save_field_netcdf logged
# success.

@testset "save_field appends: three components share one file (plot_tools pattern)" begin
    dir = mktempdir()
    path = joinpath(dir, "vector.nc")
    domain = PeriodicDomain(8)

    # Visibly different content per component so a mix-up cannot pass.
    generators = [(x,) -> sin(x), (x,) -> 2 + cos(x), (x,) -> x^2]
    originals = Vector{Any}(undef, 3)

    for i in 1:3
        comp = ScalarField(domain, "component_$i")
        set!(comp, generators[i])
        ensure_layout!(comp, :g)
        originals[i] = copy(get_grid_data(comp))
        written = save_field(comp, path, "component_$i")
        @test isfile(written)
    end

    @test originals[1] != originals[2]
    @test originals[2] != originals[3]
    @test originals[1] != originals[3]

    for i in 1:3
        loaded = ScalarField(domain, "loaded_$i")
        load_field!(loaded, path, "component_$i")
        ensure_layout!(loaded, :g)
        @test get_grid_data(loaded) == originals[i]
    end
end

@testset "save_field overwrites the same variable when it is the only one in the file" begin
    dir = mktempdir()
    path = joinpath(dir, "overwrite.nc")
    domain = PeriodicDomain(8)

    u1 = ScalarField(domain, "u")
    set!(u1, (x,) -> sin(x))
    ensure_layout!(u1, :g)
    save_field(u1, path, "u")

    u2 = ScalarField(domain, "u")
    set!(u2, (x,) -> cos(x) + 5)
    ensure_layout!(u2, :g)
    second = copy(get_grid_data(u2))

    written = save_field(u2, path, "u")
    @test isfile(written)

    v = ScalarField(domain, "v")
    load_field!(v, written, "u")
    ensure_layout!(v, :g)
    @test get_grid_data(v) == second
end

@testset "save_field refuses to silently discard other variables when re-saving one" begin
    dir = mktempdir()
    path = joinpath(dir, "mixed.nc")
    domain = PeriodicDomain(8)

    a = ScalarField(domain, "a")
    set!(a, (x,) -> sin(x))
    ensure_layout!(a, :g)
    save_field(a, path, "component_1")

    b = ScalarField(domain, "b")
    set!(b, (x,) -> cos(x))
    ensure_layout!(b, :g)
    save_field(b, path, "component_2")

    # Re-saving component_1 would need to delete the file, which would also
    # destroy component_2 -- must throw, not silently discard component_2.
    c = ScalarField(domain, "c")
    set!(c, (x,) -> x)
    ensure_layout!(c, :g)
    err = try
        save_field(c, path, "component_1")
        nothing
    catch e
        e
    end
    @test err !== nothing
    msg = sprint(showerror, err)
    @test occursin("component_1", msg)
    @test occursin("component_2", msg)

    # component_2's data must still be intact on disk, untouched.
    loaded2 = ScalarField(domain, "loaded2")
    load_field!(loaded2, path, "component_2")
    ensure_layout!(loaded2, :g)
    ensure_layout!(b, :g)
    @test get_grid_data(loaded2) == get_grid_data(b)
end

# --- Fix round: _slab_output_path! must not deadlock other ranks ---
#
# Under MPI, if rank 0's mkpath throws, it must not leave every other rank
# blocked in MPI.Barrier forever. That path needs >1 rank to exercise (owned
# by Task 5's MPI rank-count matrix); this serial test guards the path that
# IS reachable here -- resolving a path whose parent directory already
# exists, which must keep working unchanged and without requiring MPI.

@testset "save_field resolves the correct serial path when its parent directory already exists" begin
    dir = mktempdir()
    @test isdir(dir)  # the parent directory already exists before save_field runs
    path = joinpath(dir, "already_there.nc")

    domain = PeriodicDomain(8)
    u = ScalarField(domain, "u")
    set!(u, (x,) -> sin(2x))
    ensure_layout!(u, :g)

    written = save_field(u, path, "u")
    @test written == path
    @test isfile(written)
end

function _decay_solver(stepper; dt=0.02)
    domain = PeriodicDomain(16)
    u = ScalarField(domain, "u")
    set!(u, (x,) -> sin(x) + 0.25cos(2x))
    problem = IVP([u])
    add_equation!(problem, "dt(u) = -u")
    return InitialValueSolver(problem, stepper; dt)
end

@testset "save_state / load_state! restores fields and clock" begin
    dir = mktempdir()
    path = joinpath(dir, "chk")

    a = _decay_solver(RK222())
    for _ in 1:10
        step!(a, 0.02)
    end
    written = save_state(a, path)
    @test isfile(written)

    ensure_layout!(a.state[1], :g)
    expected = copy(get_grid_data(a.state[1]))

    b = _decay_solver(RK222())
    load_state!(b, path)
    @test b.sim_time ≈ a.sim_time
    @test b.iteration == a.iteration
    @test b.dt ≈ a.dt
    ensure_layout!(b.state[1], :g)
    @test get_grid_data(b.state[1]) == expected
end

@testset "a one-step scheme continues exactly across a restart" begin
    dir = mktempdir()
    path = joinpath(dir, "chk")

    uninterrupted = _decay_solver(RK222())
    for _ in 1:20
        step!(uninterrupted, 0.02)
    end
    ensure_layout!(uninterrupted.state[1], :g)
    reference = copy(get_grid_data(uninterrupted.state[1]))

    first_half = _decay_solver(RK222())
    for _ in 1:10
        step!(first_half, 0.02)
    end
    save_state(first_half, path)

    second_half = _decay_solver(RK222())
    load_state!(second_half, path)
    for _ in 1:10
        step!(second_half, 0.02)
    end
    ensure_layout!(second_half.state[1], :g)
    @test get_grid_data(second_half.state[1]) ≈ reference atol=1e-13
end

@testset "a multistep restart warns that it re-seeds" begin
    dir = mktempdir()
    path = joinpath(dir, "chk")
    a = _decay_solver(SBDF4())
    for _ in 1:12
        step!(a, 0.02)
    end
    save_state(a, path)

    b = _decay_solver(SBDF4())
    @test_logs (:warn, r"SBDF4") match_mode=:any load_state!(b, path)
    @test b.iteration == a.iteration
end

# --- Fix round: Finding 1 -- the multistep restart warning must fire once
# PER SCHEME, not once per process.
#
# `@warn ... maxlog=1` with no `_id` dedups by macro-expansion CALLSITE, not
# by message content: there is exactly one `@warn` callsite in
# `_warn_multistep_restart` serving every scheme, so the first multistep
# restart a process performs was the only one that would ever warn -- a
# different scheme, or the same scheme on a different solver, restarted
# silently for the rest of the session. The fix tags the warning with
# `_id=Symbol(:multistep_restart_, scheme)` so each scheme gets its own
# dedup identity.
#
# This test deliberately uses schemes not exercised by any earlier testset
# in this file (RK222 above is not multistep; SBDF4 above already used its
# one warning) so that a warning here is real evidence of per-scheme
# identity working, not a stale pass left over from another test's restart.
@testset "different multistep schemes each warn once on restart (per-scheme identity)" begin
    dir = mktempdir()

    path2 = joinpath(dir, "chk_sbdf2")
    a2 = _decay_solver(SBDF2())
    for _ in 1:12
        step!(a2, 0.02)
    end
    save_state(a2, path2)
    b2 = _decay_solver(SBDF2())
    @test_logs (:warn, r"SBDF2") match_mode=:any load_state!(b2, path2)

    path3 = joinpath(dir, "chk_sbdf3")
    a3 = _decay_solver(SBDF3())
    for _ in 1:12
        step!(a3, 0.02)
    end
    save_state(a3, path3)
    b3 = _decay_solver(SBDF3())
    @test_logs (:warn, r"SBDF3") match_mode=:any load_state!(b3, path3)
end

# --- Final review wave: the clock attributes are REQUIRED ---
#
# `load_state!` used to apply each clock attribute only `haskey(attrs, ...)`.
# With none present every assignment was skipped and the call RETURNED
# SUCCESSFULLY: fields correctly restored, clock silently left at
# sim_time = 0.0, iteration = 0, no warning. The restart then integrates for
# the wrong duration and evaluates any time-dependent forcing at the wrong
# time, with correct-looking field values — this repo's dominant bug class.
#
# Two reachable producers of a clock-less file, one testset each below.

@testset "load_state! refuses a checkpoint whose write died before the clock attributes" begin
    dir = mktempdir()
    path = joinpath(dir, "truncated.nc")

    # save_state writes every field first and stamps the three global
    # attributes LAST, so a run that died during the (slow) field writes leaves
    # exactly this: correct field data, no clock. Reproduce it by writing the
    # field slab the same way save_state does and stopping before the ncputatt.
    donor = _decay_solver(RK222())
    for _ in 1:5
        step!(donor, 0.02)
    end
    field = donor.state[1]
    @test field.name == "u"
    ensure_layout!(field, :g)
    gshape, _, lstart = Tarang._field_slab_geometry(field)
    Tarang.write_local_slab(path, field.name,
                            Array(Tarang.get_local_data(get_grid_data(field))), lstart, gshape)

    target = _decay_solver(RK222())
    err = try
        load_state!(target, path)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    msg = sprint(showerror, err)
    @test occursin("sim_time", msg)
    @test occursin("iteration", msg)
    @test occursin("dt", msg)
    @test occursin("truncated", msg)
    @test occursin("incomplete", msg)
    # It must not have quietly succeeded with a zero clock.
    @test target.iteration == 0 || target.iteration == donor.iteration
end

@testset "load_state! refuses save_field output, which carries no clock at all" begin
    dir = mktempdir()
    path = joinpath(dir, "fieldonly")

    donor = _decay_solver(RK222())
    for _ in 1:5
        step!(donor, 0.02)
    end
    # save_field accepts the same suffix-less path form load_state! does and
    # writes no global attributes whatsoever.
    save_field(donor.state[1], path, "u")

    target = _decay_solver(RK222())
    err = try
        load_state!(target, path)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    msg = sprint(showerror, err)
    @test occursin("save_field", msg)
    @test occursin("clock attribute", msg)
    @test target.sim_time == 0.0   # not silently continued from a bogus clock
end

# --- Final review wave: the reseed roster must cover the whole timestepper
# roster ---
#
# `_MULTISTEP_RESEED_STEPS` listed only CNAB2/SBDF2/SBDF3/SBDF4, so MCNAB2,
# CNLF2, ETD_CNAB2, ETD_SBDF2 and DiagonalIMEX_SBDF2 all restarted with a
# silently re-seeded history and no warning — and the warning text and
# docs/src/api/io.md both asserted that DiagonalIMEX_* restarts exactly, which
# is affirmatively false for DiagonalIMEX_SBDF2 (its own docstring calls it a
# multi-step method).
#
# This test is deliberately STATIC: it performs no restart, so it spends none
# of the per-scheme `@warn ... _id` dedup identities that the testsets above
# rely on (that dedup is process-global — see the note on
# `_MULTISTEP_RESEED_STEPS`).

@testset "every timestepper is classified as multistep-reseeding or one-step" begin
    all_schemes = sort(String[string(nameof(T)) for T in subtypes(Tarang.TimeStepper)
                             if isconcretetype(T)])
    @test length(all_schemes) >= 20   # the scanner must actually be finding types

    classified = union(Set(keys(Tarang._MULTISTEP_RESEED_STEPS)), Tarang._ONE_STEP_SCHEMES)
    unclassified = sort([s for s in all_schemes if !(s in classified)])
    # A new scheme added to types.jl lands here and must be put in one table or
    # the other; omission is what makes a restart silently lossy.
    @test isempty(unclassified)

    # And neither table may name a scheme that no longer exists.
    stale = sort([s for s in classified if !(s in all_schemes)])
    @test isempty(stale)

    # The schemes the roster was missing, named explicitly so a revert is loud.
    for scheme in ("MCNAB2", "CNLF2", "ETD_CNAB2", "ETD_SBDF2", "DiagonalIMEX_SBDF2")
        @test haskey(Tarang._MULTISTEP_RESEED_STEPS, scheme)
        @test Tarang._MULTISTEP_RESEED_STEPS[scheme] >= 1
    end
    @test Tarang._MULTISTEP_RESEED_STEPS["SBDF3"] == 2
    @test Tarang._MULTISTEP_RESEED_STEPS["SBDF4"] == 3

end

@testset "a newly-rostered multistep scheme warns, and the warning stops crediting DiagonalIMEX" begin
    # CNLF2 (leapfrog: needs X^{n-1}) was absent from the roster and restarted
    # in silence. It is used by no other testset in this file, so its
    # process-global `_id` dedup identity is unspent here.
    dir = mktempdir()
    path = joinpath(dir, "chk_cnlf2")
    a = _decay_solver(Tarang.CNLF2())
    for _ in 1:12
        step!(a, 0.02)
    end
    save_state(a, path)

    b = _decay_solver(Tarang.CNLF2())
    logger = Test.TestLogger()
    Base.CoreLogging.with_logger(logger) do
        load_state!(b, path)
    end
    warnings = [string(r.message) for r in logger.logs if r.level == Base.CoreLogging.Warn]
    @test any(w -> occursin("CNLF2", w), warnings)
    # The old text told the user "DiagonalIMEX_*" restarts exactly, which is
    # false for DiagonalIMEX_SBDF2.
    @test !any(w -> occursin("DiagonalIMEX_*", w), warnings)
    @test any(w -> occursin("DiagonalIMEX_SBDF2 does NOT", w), warnings)
    @test b.iteration == a.iteration
end

@testset "load_state! rejects a checkpoint from a different resolution" begin
    dir = mktempdir()
    path = joinpath(dir, "chk")
    save_state(_decay_solver(RK222()), path)

    domain = PeriodicDomain(32)
    u = ScalarField(domain, "u")
    set!(u, (x,) -> sin(x))
    problem = IVP([u])
    add_equation!(problem, "dt(u) = -u")
    wrong = InitialValueSolver(problem, RK222(); dt=0.02)
    @test_throws ErrorException load_state!(wrong, path)
end

# --- Fix round: Finding 3 -- the tau/algebraic-field skip path had zero
# test coverage.
#
# `save_state`/`load_state!` both skip fields with `isempty(field.bases)`
# (zero-dimensional tau/Lagrange-multiplier variables that carry no spatial
# data -- see the docstrings). Every test above uses `PeriodicDomain`, which
# is pure Fourier and never has a tau variable in `solver.state`, so that
# skip branch has never executed anywhere in this suite. This repo's audit
# history records shape/skip mismatches resolving to a plausible zero as its
# dominant bug class, and only a value assertion catches them.
#
# Build a genuine Chebyshev x Fourier diffusion IVP with lift-based tau
# terms and boundary conditions -- the same problem shape as
# test/test_mpi_sbdf_high_order.jl's `_sbdf_diffusion_error`, adapted to
# serial and to a checkpoint round-trip instead of an MPI convergence-rate
# check. `tau_b1`/`tau_b2` are declared with `()` bases (zero-dimensional),
# so they hit the exact skip branch under test.

function _cheb_tau_solver(stepper; dt=0.02, Nz=12, Nx=8, κ=0.1)
    coords = CartesianCoordinates("z", "x")
    dist   = Distributor(coords; dtype=Float64, device=CPU())
    zbasis = ChebyshevT(coords["z"]; size=Nz, bounds=(0.0, 1.0))
    xbasis = RealFourier(coords["x"]; size=Nx, bounds=(0.0, 2pi))
    domain = Domain(dist, (zbasis, xbasis))

    b = ScalarField(domain, "b")
    set!(b, (z, x) -> sin(pi*z) * (1 + 0.5cos(2x)))

    τ1 = ScalarField(dist, "tau_b1", (), Float64)   # zero-dim: isempty(bases) == true
    τ2 = ScalarField(dist, "tau_b2", (), Float64)   # zero-dim: isempty(bases) == true

    _, ez = unit_vector_fields(coords, dist)
    lift_basis = derivative_basis(zbasis, 1)
    τ_lift(A) = lift(A, lift_basis, -1)
    grad_b = grad(b) + ez * τ_lift(τ1)

    problem = IVP([b, τ1, τ2])
    add_parameters!(problem, kappa=κ, ez=ez, grad_b=grad_b, τ_lift=τ_lift)
    add_equation!(problem, "dt(b) - kappa*div(grad_b) + τ_lift(tau_b2) = 0")
    add_bc!(problem, "b(z=0) = 0")
    add_bc!(problem, "b(z=1) = 0")

    return InitialValueSolver(problem, stepper; dt=dt)
end

@testset "checkpoint/restart round-trips a Chebyshev tau problem (zero-dim tau skip path)" begin
    dir = mktempdir()
    path = joinpath(dir, "chk")

    # Confirm the skip path is actually exercised: the tau fields in
    # solver.state really are zero-dimensional.
    probe = _cheb_tau_solver(RK222())
    @test isempty(probe.state[2].bases)
    @test isempty(probe.state[3].bases)

    uninterrupted = _cheb_tau_solver(RK222())
    for _ in 1:20
        step!(uninterrupted, 0.02)
    end
    ensure_layout!(uninterrupted.state[1], :g)
    reference = copy(get_grid_data(uninterrupted.state[1]))
    # Oracle sanity: the field must carry real signal, not have decayed (or
    # been silently zeroed) into a trivial all-zero match.
    @test maximum(abs, reference) > 0.05

    first_half = _cheb_tau_solver(RK222())
    for _ in 1:10
        step!(first_half, 0.02)
    end
    save_state(first_half, path)

    second_half = _cheb_tau_solver(RK222())
    load_state!(second_half, path)
    for _ in 1:10
        step!(second_half, 0.02)
    end
    ensure_layout!(second_half.state[1], :g)
    restarted = get_grid_data(second_half.state[1])

    @test restarted ≈ reference atol=1e-9
end

# --- Fix round (2026-08-20 MPI review, O2): a stale MPI slab DIRECTORY of the
# same stem must not shadow a newer serial checkpoint. `_slab_files` prefers a
# directory over `<stem>.nc`, and the serial save path never removed one — so
# `load_state!` silently restored the OLD multi-rank state including its clock.

@testset "serial save_state removes a stale MPI slab directory" begin
    dir = mktempdir()
    path = joinpath(dir, "chk")

    # Simulate a leftover np=2 checkpoint directory with rank slab files.
    mkpath(path)
    for r in 0:1
        write(joinpath(path, "chk_p$r.nc"), "stale")
    end

    a = _decay_solver(RK222())
    for _ in 1:5
        step!(a, 0.02)
    end
    written = save_state(a, path)
    @test isfile(written)
    @test !isdir(path)                      # stale slab dir cleaned up

    b = _decay_solver(RK222())
    load_state!(b, path)
    @test b.sim_time ≈ a.sim_time           # pre-fix: stale dir shadowed the file
    @test b.iteration == a.iteration
end

@testset "serial save_state refuses a conflicting non-slab directory" begin
    dir = mktempdir()
    path = joinpath(dir, "chk2")
    mkpath(path)
    write(joinpath(path, "unrelated.txt"), "keep me")

    a = _decay_solver(RK222())
    step!(a, 0.02)
    # Deleting arbitrary user files would be worse than failing: refuse loudly.
    @test_throws Exception save_state(a, path)
    @test isfile(joinpath(path, "unrelated.txt"))
end
