"""
GPU timestepper correctness — the first GPU CI coverage of `step!` itself.
Previously GPU_TEST_FILES exercised transforms/DCT only, so every GPU stepping
bug was invisible to CI (the audit of 2026-07-18 found two silent-wrong paths).

Value-asserted (exact solutions, not smoke):
  1. DiagonalIMEX_{RK222,RK443,SBDF2} + SpectralLinearOperator on GPU: viscous
     decay at exp(−νk²T) — the only fully on-device implicit path.
  2. Explicit RK convergence on GPU: dt(u) = −u², u(0)=1 → 1/(1+t); RK222 must
     show 2nd order. REGRESSION GUARD for the stage-RHS aliasing bug: the
     compiled lazy plan returns the SAME output buffer every evaluate_rhs call,
     and `_step_explicit_rk_gpu!` used to store it uncopied in k_stages — all
     stages aliased, degrading RK222 to ~1st order.
  3. Standard IMEX RK with a nonzero implicit operator on GPU must THROW (loud
     refusal), not silently integrate without the linear terms. REGRESSION
     GUARD for the silent L-drop: before the fix this path ran the heat
     equation WITHOUT diffusion at @debug verbosity.

CUDA-guarded: skips cleanly when CUDA is unavailable. Authored on a GPU-less
machine — parse/skip-verified only; assertions first exercised on a GPU node.
"""

using Test
using Tarang
try
    using CUDA
catch
end

const _HAS_CUDA_TS = (@isdefined CUDA) && CUDA.functional()

if !_HAS_CUDA_TS
    @testset "GPU timesteppers (skipped: no functional CUDA)" begin
        @test_skip "CUDA not functional on this host"
    end
else
    _gpu_grid(u) = Array(Tarang.get_grid_data(u))

    function _gpu_fourier_field(N)
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64, device=GPU())
        xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
        u = ScalarField(dist, "u", (xb,), Float64)
        return coords, dist, xb, u
    end

    @testset "GPU DiagonalIMEX viscous decay (exact rate)" begin
        # dt(u) = −ν k² u via attached SpectralLinearOperator; u0 = cos(2x),
        # ν = 0.5 → λ = 2; after T = 1.0 amplitude = exp(−2).
        for ts in (DiagonalIMEX_RK222(), DiagonalIMEX_RK443(), DiagonalIMEX_SBDF2())
            N = 16
            coords, dist, xb, u = _gpu_fourier_field(N)
            xs = collect(range(0, 2π, length=N+1))[1:N]
            ensure_layout!(u, :g)
            copyto!(Tarang.get_grid_data(u), cos.(2 .* xs))
            L = SpectralLinearOperator(dist, (xb,), :laplacian; ν=0.5)
            problem = IVP([u]); add_equation!(problem, "dt(u) = 0")
            solver = InitialValueSolver(problem, ts; dt=0.005)
            Tarang.set_spectral_linear_operator!(solver, L)
            u0 = maximum(abs, _gpu_grid(u))
            for _ in 1:200
                step!(solver)
            end
            ensure_layout!(u, :g)
            uf = maximum(abs, _gpu_grid(u))
            @test isfinite(uf)
            @test isapprox(uf / u0, exp(-2.0); rtol=0.05)
        end
    end

    @testset "GPU explicit RK order (stage-aliasing regression)" begin
        # dt(u) = −u², u0 = 1 → u(T) = 1/(1+T). Aliased stages drop RK222 to ~1.
        function nl_err(dt, T)
            N = 16
            coords, dist, xb, u = _gpu_fourier_field(N)
            ensure_layout!(u, :g)
            fill!(Tarang.get_grid_data(u), 1.0)
            problem = IVP([u]); add_equation!(problem, "dt(u) = -u*u")
            solver = InitialValueSolver(problem, RK222(); dt=dt)
            for _ in 1:round(Int, T/dt)
                step!(solver)
            end
            ensure_layout!(u, :g)
            return maximum(abs, _gpu_grid(u) .- 1/(1+T))
        end
        e1 = nl_err(0.02, 0.4)
        e2 = nl_err(0.01, 0.4)
        @test e1 < 1e-3                    # sane absolute accuracy
        @test log2(e1 / e2) > 1.7          # 2nd order (aliasing gives ~1.0)
    end

    @testset "GPU non-diagonal schemes refuse to drop the implicit operator" begin
        # Heat equation with an implicit LHS operator on a pure-Fourier GPU: these
        # schemes have no per-mode implicit path there (the solver builds no global
        # matrix/subproblem), so step! must error loudly instead of integrating
        # without diffusion. The old behavior was a silent @debug + explicit
        # fallback — a heat equation ran inviscid with no error.
        for ts in (RK222(), RK443(), SBDF2(), CNAB2(), ETD_RK222())
            N = 16
            coords, dist, xb, u = _gpu_fourier_field(N)
            xs = collect(range(0, 2π, length=N+1))[1:N]
            ensure_layout!(u, :g)
            copyto!(Tarang.get_grid_data(u), cos.(2 .* xs))
            problem = IVP([u]); add_equation!(problem, "dt(u) - lap(u) = 0")
            solver = InitialValueSolver(problem, ts; dt=0.01)
            @test_throws ErrorException step!(solver)
        end
    end

    @testset "GPU DiagonalIMEX equation-derived viscous decay (no attached op)" begin
        # The implicit operator comes from the EQUATION (`dt(u) - ν lap(u) = 0`),
        # not an attached SpectralLinearOperator. This exercises the device L̂ build
        # (_diagonal_Lhat_from_expr on GPU coefficient arrays); before the device
        # allocation fix it threw a scalar-indexing error at L̂ construction.
        # ν = 0.5, u0 = cos(2x) → λ = ν·k² = 2; after T = 1.0 amplitude = exp(−2).
        for ts in (DiagonalIMEX_RK222(), DiagonalIMEX_RK443(), DiagonalIMEX_SBDF2())
            N = 16
            coords, dist, xb, u = _gpu_fourier_field(N)
            xs = collect(range(0, 2π, length=N+1))[1:N]
            ensure_layout!(u, :g)
            copyto!(Tarang.get_grid_data(u), cos.(2 .* xs))
            problem = IVP([u]); add_equation!(problem, "dt(u) - 0.5*lap(u) = 0")
            solver = InitialValueSolver(problem, ts; dt=0.005)
            u0 = maximum(abs, _gpu_grid(u))
            for _ in 1:200
                step!(solver)
            end
            ensure_layout!(u, :g)
            uf = maximum(abs, _gpu_grid(u))
            @test isfinite(uf)
            @test isapprox(uf / u0, exp(-2.0); rtol=0.05)   # actually diffused
        end
    end
end
