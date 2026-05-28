# Regression tests for the distributed diagonal-IMEX operator parser.
#
# Bug (2026-05-28): pure-Fourier MPI silently FROZE any field whose implicit linear
# operator contained a non-Laplacian term (e.g. first-derivative advection), because
# _diagonal_Lhat_from_expr returned `nothing` → the field was skipped every step.
#
# Unit/safety-net tests run SERIALLY (they exercise the parser directly, no MPI
# launcher needed). The end-to-end "field is not frozen" check is gated on nprocs>1.
using Test
using Tarang
import Tarang: _diagonal_Lhat_from_expr, d, wavenumbers_rfft, wavenumbers_fft

import MPI
MPI.Initialized() || MPI.Init()
const NP = MPI.Comm_size(MPI.COMM_WORLD)
const RANK = MPI.Comm_rank(MPI.COMM_WORLD)

function _mk_field(; np=1)
    coords = CartesianCoordinates("x", "y")
    dist = np > 1 ? Distributor(coords; mesh=(np,), dtype=Float64, architecture=CPU()) :
                    Distributor(coords; dtype=Float64, architecture=CPU())
    N = 8
    xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
    yb = RealFourier(coords["y"]; size=N, bounds=(0.0, 2π))
    domain = Domain(dist, (xb, yb))
    q = ScalarField(domain, "q"); ensure_layout!(q, :c)
    return coords, dist, xb, yb, domain, q, N
end

if NP == 1
    @testset "diagonal L̂ parser: derivative multipliers (i·k)" begin
        coords, dist, xb, yb, domain, q, N = _mk_field()
        cd = get_coeff_data(q)

        # ∂x(q): first RealFourier axis is rfft layout, k = [0,1,…,N/2]. Multiplier i*k_x.
        Lhat = _diagonal_Lhat_from_expr(d(q, coords["x"]), q)
        @test Lhat !== nothing
        kx = wavenumbers_rfft(xb)
        expected = ComplexF64[im * kx[I[1]] for I in CartesianIndices(size(cd))]
        @test maximum(abs.(ComplexF64.(Lhat) .- expected)) < 1e-10

        # ∂y(q): second axis is full-FFT layout. Multiplier i*k_y on axis 2.
        Lhat_y = _diagonal_Lhat_from_expr(d(q, coords["y"]), q)
        @test Lhat_y !== nothing
        ky = wavenumbers_fft(yb)
        expected_y = ComplexF64[im * ky[I[2]] for I in CartesianIndices(size(cd))]
        @test maximum(abs.(ComplexF64.(Lhat_y) .- expected_y)) < 1e-10
    end

    @testset "diagonal L̂ parser: real equation mixes diffusion + advection" begin
        coords, dist, xb, yb, domain, q, N = _mk_field()
        problem = IVP([q])
        add_parameters!(problem, nu=0.05)
        add_equation!(problem, "dt(q) - nu*lap(q) - 0.3*d(q,x) = 0")
        solver = InitialValueSolver(problem, RK222(); dt=1e-3)  # populates equation_data
        Lexpr = get(solver.problem.equation_data[1], "L", nothing)
        @test Lexpr !== nothing
        Lhat = _diagonal_Lhat_from_expr(Lexpr, q)
        @test Lhat !== nothing                         # was `nothing` (freeze) before fix
        ch = ComplexF64.(Lhat)
        @test any(!iszero, real.(ch))                  # diffusion (−k²) present
        @test any(!iszero, imag.(ch))                  # advection (i·k_x) present
    end

    @testset "safety net: derivative of a DIFFERENT field is not diagonal" begin
        coords, dist, xb, yb, domain, q, N = _mk_field()
        p = ScalarField(domain, "p"); ensure_layout!(p, :c)
        # ∂x(p) inside q's implicit operator is a coupling term, NOT a scalar diagonal L̂.
        @test _diagonal_Lhat_from_expr(d(p, coords["x"]), q) === nothing
    end
else
    @testset "MPI: implicit advection no longer freezes the field (rank=$RANK)" begin
        coords, dist, xb, yb, domain, q, N = _mk_field(np=NP)
        problem = IVP([q])
        add_parameters!(problem, nu=0.05)
        add_equation!(problem, "dt(q) - nu*lap(q) - 0.3*d(q,x) - 0.2*d(q,y) = 0")
        solver = InitialValueSolver(problem, RK222(); dt=2e-3)
        ensure_layout!(q, :g)
        dat = get_grid_data(q); raw = dat isa Tarang.PencilArrays.PencilArray ? parent(dat) : dat
        ax = dat isa Tarang.PencilArrays.PencilArray ? Tarang.PencilArrays.pencil(dat).axes_local : (1:N, 1:N)
        xs = [2π*(i-1)/N for i in ax[1]]; ys = [2π*(j-1)/N for j in ax[2]]
        raw .= @. sin(2xs - ys') + 0.5cos(xs + 3ys')
        ensure_layout!(q, :c)
        l2() = (ensure_layout!(q,:g); r=get_grid_data(q); rr=r isa Tarang.PencilArrays.PencilArray ? parent(r) : r;
                MPI.Allreduce(sum(abs2, rr), MPI.SUM, MPI.COMM_WORLD))

        n0 = l2()
        cd = get_coeff_data(q); snap = copy(cd isa Tarang.PencilArrays.PencilArray ? parent(cd) : cd)
        step!(solver, 2e-3)
        ensure_layout!(q, :c)
        cd2 = get_coeff_data(q); cur = cd2 isa Tarang.PencilArrays.PencilArray ? parent(cd2) : cd2
        dmax = MPI.Allreduce(maximum(abs.(cur .- snap)), MPI.MAX, MPI.COMM_WORLD)
        @test dmax > 1e-6                              # field actually evolves (was frozen → 0)
        for _ in 1:9; step!(solver, 2e-3); end
        @test l2() < n0 - 1e-6                         # diffusion decays L2 (frozen → unchanged)
    end
end
