# Guard: the distributed diagonal-IMEX SBDF2 and ETD-RK2 steppers are O(1)-alloc
# per step AND behavior-preserving.
#
# Background: the pure-Fourier + MPI (no-subproblems) diagonal-IMEX RK stepper was
# made O(1)-alloc by reusing history-dropped field-sets (_ddirk_acquire_xnew! /
# _ddirk_push_recycle!) + a persistent per-stage F cache, dropping the S+1 per-step
# copy_state allocations. That treatment was NOT carried over to its multistep
# SBDF2 sibling (step_distributed_diagonal_imex_sbdf2!) or its ETD sibling
# (step_distributed_diagonal_etd_rk222!), which each performed 2 full copy_state of
# the distributed state every step (~1.8x the RK floor at N=128). Ported 2026-07-09.
#
# The distributed diagonal path is taken ONLY at nprocs>1 (dist.size>1); at np1 the
# serial diagonal/ETD code runs instead. So the allocation bound and the tight
# distributed reference are gated on NP>1. The reference sumsq values are pinned
# from the verified pre-fix run (the fix is behavior-preserving, so post-fix must
# reproduce them to roundoff); np2 and np4 are bit-identical to the reference.
using Test
using Tarang
import MPI
MPI.Initialized() || MPI.Init()
const NP = MPI.Comm_size(MPI.COMM_WORLD)
const RANK = MPI.Comm_rank(MPI.COMM_WORLD)

_raw(f) = (d = get_grid_data(f); d isa Tarang.PencilArrays.PencilArray ? parent(d) : d)
_mkdist(coords) = NP > 1 ? Distributor(coords; mesh=(NP,), dtype=Float64, architecture=CPU()) :
                           Distributor(coords; dtype=Float64, architecture=CPU())
function _ic!(f, N, fn)
    ensure_layout!(f, :g)
    dd = get_grid_data(f)
    ax = dd isa Tarang.PencilArrays.PencilArray ? Tarang.PencilArrays.pencil(dd).axes_local : (1:N, 1:N)
    xs = [2π*(i-1)/N for i in ax[1]]; ys = [2π*(j-1)/N for j in ax[2]]
    _raw(f) .= fn.(xs, ys')
    ensure_layout!(f, :c)
end
_sumsq(f) = (ensure_layout!(f, :g); MPI.Allreduce(sum(abs2, _raw(f)), MPI.SUM, MPI.COMM_WORLD))

# Single-field pure diffusion: measure peak per-step @allocated (worst rank) after warmup.
function _alloc_per_step(stepper; N=128, warmup=15, measure=20, dt=1e-3)
    coords = CartesianCoordinates("x", "y"); dist = _mkdist(coords)
    xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
    yb = RealFourier(coords["y"]; size=N, bounds=(0.0, 2π))
    domain = Domain(dist, (xb, yb)); q = ScalarField(domain, "q")
    problem = IVP([q]); add_parameters!(problem, nu=0.02); add_equation!(problem, "dt(q) - nu*lap(q) = 0")
    solver = InitialValueSolver(problem, stepper; dt=dt)
    _ic!(q, N, (x,y)->sin(2x-y)+0.5cos(x+3y)+0.3sin(3x+2y))
    for _ in 1:warmup; step!(solver, dt); end
    peak = 0
    for _ in 1:measure; peak = max(peak, @allocated step!(solver, dt)); end
    MPI.Allreduce(peak, MPI.MAX, MPI.COMM_WORLD)
end

# Two-field linearly-coupled diffusive system. The cross-field advection terms are
# NOT diagonal in a pure-Fourier basis, so they land in the explicit RHS F (F != 0):
# this exercises the F-recycling (SBDF2 F-history ring, ETD N(Xn) cache) that a
# pure-diffusion problem (F == 0) would not. Bounded (diffusion decays energy).
function _coupled_sumsq(stepper; N=64, steps=40, dt=1e-3)
    coords = CartesianCoordinates("x", "y"); dist = _mkdist(coords)
    xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
    yb = RealFourier(coords["y"]; size=N, bounds=(0.0, 2π))
    domain = Domain(dist, (xb, yb))
    u = ScalarField(domain, "u"); v = ScalarField(domain, "v")
    problem = IVP([u, v]); add_parameters!(problem, nu=0.03, c=0.7)
    add_equation!(problem, "dt(u) - nu*lap(u) = -c*d(v,x)")
    add_equation!(problem, "dt(v) - nu*lap(v) = c*d(u,x)")
    solver = InitialValueSolver(problem, stepper; dt=dt)
    _ic!(u, N, (x,y)->sin(x)+0.4cos(2x+y)); _ic!(v, N, (x,y)->cos(x-2y)+0.3sin(3x))
    for _ in 1:steps; step!(solver, dt); end
    (_sumsq(u), _sumsq(v))
end

# Pinned distributed reference (np2 == np4 bit-identical; the behavior-preserving
# fix reproduces these to roundoff). SBDF2's serial diagonal path also matches
# these (verified np1), but ETD's serial path is a different algorithm, so the
# tight reference is only asserted on the distributed path (NP>1).
const REF_SBDF2 = (2372.3408492102935, 2209.430928365213)
const REF_ETD   = (2372.343302290148, 2209.4335176939358)
const REF_RTOL  = 1e-9

@testset "distributed diagonal-IMEX SBDF2/ETD: O(1) alloc + behavior-preserving (NP=$NP)" begin
    if NP > 1
        @testset "O(1) allocation (no per-step copy_state)" begin
            rk    = _alloc_per_step(RK222())          # already-O(1) sibling → shared per-step residual floor
            sbdf2 = _alloc_per_step(SBDF2())
            etd   = _alloc_per_step(ETD_RK222())
            RANK == 0 && @info "diagonal-IMEX per-step alloc (KB, worst rank, NP=$NP)" rk=rk/1024 sbdf2=sbdf2/1024 etd=etd/1024
            # Pre-fix SBDF2/ETD carry 2 extra full-state copies/step (≈1.8× the floor);
            # post-fix they allocate like the RK sibling. 1.5× cleanly separates.
            @test sbdf2 < 1.5 * rk
            @test etd   < 1.5 * rk
        end
        @testset "distributed result bit-identical to pinned reference" begin
            su, sv = _coupled_sumsq(SBDF2())
            @test isapprox(su, REF_SBDF2[1]; rtol=REF_RTOL)
            @test isapprox(sv, REF_SBDF2[2]; rtol=REF_RTOL)
            eu, ev = _coupled_sumsq(ETD_RK222())
            @test isapprox(eu, REF_ETD[1]; rtol=REF_RTOL)
            @test isapprox(ev, REF_ETD[2]; rtol=REF_RTOL)
        end
    else
        @testset "serial sanity (distributed diagonal path not taken at np1)" begin
            su, sv = _coupled_sumsq(SBDF2())
            @test isfinite(su) && isfinite(sv)
            eu, ev = _coupled_sumsq(ETD_RK222())
            @test isfinite(eu) && isfinite(ev)
        end
    end
end
