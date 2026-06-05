"""
GPU transform correctness — validates the single-GPU transform DISPATCH
(`gpu_forward_transform!` / `gpu_backward_transform!`) against the CPU result,
which is the trusted oracle. This is the gap the existing GPU tests
(`test_dct_reorder`, `test_optimized_dct`) leave open: they pin the low-level DCT
kernels, but nothing checked that a `forward_transform!(field)` on a CuArray-backed
field matches the CPU path for Fourier / Chebyshev / mixed / complex bases.

It also exercises the single-GPU efficiency fixes added 2026-06-04/05:
  - cached DCT plans (`get_gpu_dct_plan` / `get_gpu_dct_dim_plan`)
  - reusable scratch + ping-pong in the complex / multi-dim branches
A correct GPU-vs-CPU match here is the regression guard for those.

CUDA-guarded: skips (no failure) when CUDA is unavailable, so it is harmless in
the default CPU suite. Runs for real on a CUDA host (GPU_TEST_FILES, the JuliaGPU
Buildkite pipeline). NOTE: authored on a GPU-less machine — parse/skip-verified
only; the GPU assertions themselves are first exercised on a GPU node.

Oracle: build the SAME grid data on a CPU field and a GPU field; assert
  (1) forward:  Array(coeff_gpu) ≈ coeff_cpu
  (2) backward: Array(grid_gpu)  ≈ grid_cpu
  (3) round-trip: grid → coeff → grid recovers the input (both devices).
"""

using Test
using Tarang
using LinearAlgebra
try
    using CUDA
catch
end

const _HAS_CUDA = (@isdefined CUDA) && CUDA.functional()

if !_HAS_CUDA
    @testset "GPU transform correctness (skipped: no functional CUDA)" begin
        @test_skip "CUDA not functional on this host"
    end
else
    # Build a ScalarField on `device` with bases from makebases(coords); element
    # type `T` (Float64 or ComplexF64). Returns the field.
    function _build(device, coordnames, makebases, T)
        coords = CartesianCoordinates(coordnames...)
        dist   = Distributor(coords; dtype=T, device=device)
        dom    = Domain(dist, makebases(coords))
        return ScalarField(dom, "u")
    end

    # CPU/GPU matched pair + identical initial grid data; returns (cpu_u, gpu_u, data).
    function _pair(coordnames, makebases, T)
        cpu_u = _build(CPU(),  coordnames, makebases, T)
        gpu_u = _build(GPU(),  coordnames, makebases, T)
        ensure_layout!(cpu_u, :g); ensure_layout!(gpu_u, :g)
        data = (T <: Complex) ? (rand(T, size(get_grid_data(cpu_u))...)) :
                                 rand(size(get_grid_data(cpu_u))...)
        get_grid_data(cpu_u) .= data
        get_grid_data(gpu_u) .= CuArray(data)
        return cpu_u, gpu_u, data
    end

    # Run forward on both, compare coeffs; backward on both, compare grid + round-trip.
    function _check(cpu_u, gpu_u, data; rtol=1e-9, atol=1e-11)
        forward_transform!(cpu_u);  forward_transform!(gpu_u)
        ensure_layout!(cpu_u, :c);  ensure_layout!(gpu_u, :c)
        @test isapprox(Array(get_coeff_data(gpu_u)), get_coeff_data(cpu_u); rtol=rtol, atol=atol)
        backward_transform!(cpu_u); backward_transform!(gpu_u)
        ensure_layout!(cpu_u, :g);  ensure_layout!(gpu_u, :g)
        @test isapprox(Array(get_grid_data(gpu_u)), get_grid_data(cpu_u); rtol=rtol, atol=atol)
        @test isapprox(Array(get_grid_data(gpu_u)), data; rtol=rtol, atol=atol)   # round-trip
    end

    @testset "GPU transform correctness vs CPU" begin
        @testset "2D RealFourier" begin
            mk(c) = (RealFourier(c["x"]; size=16, bounds=(0.0, 2π)),
                     RealFourier(c["y"]; size=16, bounds=(0.0, 2π)))
            _check(_pair(("x","y"), mk, Float64)...)
        end

        @testset "1D ChebyshevT (FFT-based DCT path + plan cache)" begin
            mk(c) = (ChebyshevT(c["z"]; size=24, bounds=(0.0, 1.0)),)
            cpu_u, gpu_u, data = _pair(("z",), mk, Float64)
            _check(cpu_u, gpu_u, data)
            # second transform must reuse the cached plan (no error / same result)
            _check(_pair(("z",), mk, Float64)...)
        end

        @testset "2D ChebyshevT (multi-dim DCT, ping-pong scratch)" begin
            mk(c) = (ChebyshevT(c["x"]; size=16, bounds=(0.0, 1.0)),
                     ChebyshevT(c["y"]; size=16, bounds=(0.0, 1.0)))
            _check(_pair(("x","y"), mk, Float64)...)
        end

        @testset "2D mixed RealFourier × ChebyshevT" begin
            mk(c) = (RealFourier(c["x"]; size=16, bounds=(0.0, 2π)),
                     ChebyshevT(c["y"]; size=16, bounds=(0.0, 1.0)))
            _check(_pair(("x","y"), mk, Float64)...)
        end

        @testset "2D ComplexFourier (complex split/scratch path)" begin
            mk(c) = (ComplexFourier(c["x"]; size=16, bounds=(0.0, 2π)),
                     ComplexFourier(c["y"]; size=16, bounds=(0.0, 2π)))
            _check(_pair(("x","y"), mk, ComplexF64)...)
        end
    end

    # End-to-end: integrate 2D periodic diffusion (dt u = ν Δu) on CPU and GPU
    # from the same IC and compare the final state. This validates the WHOLE GPU
    # pipeline — forward/backward transforms, implicit RHS, timestep — not just an
    # isolated transform, so it is the strongest single GPU correctness check.
    @testset "End-to-end GPU vs CPU: 2D periodic diffusion" begin
        ν = 0.1; dt = 1e-3; nsteps = 25       # t_final = 0.025; mode k=(1,1) → k²=2
        function run_diffusion(device)
            coords = CartesianCoordinates("x", "y")
            dist   = Distributor(coords; dtype=Float64, device=device)
            xb = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
            yb = RealFourier(coords["y"]; size=16, bounds=(0.0, 2π))
            dom = Domain(dist, (xb, yb))
            u = ScalarField(dom, "u")
            prob = IVP([u]); add_parameters!(prob; nu=ν)
            add_equation!(prob, "∂t(u) - nu*Δ(u) = 0")
            ensure_layout!(u, :g)
            mesh = Tarang.get_grid_coordinates(dom; on_device=false)
            ic = @. sin(mesh["x"]) * cos(mesh["y"]')      # the k=(1,1) mode, (Nx,Ny) host
            get_grid_data(u) .= (device isa CPU ? ic : CuArray(ic))
            solver = InitialValueSolver(prob, RK222(); dt=dt)
            for _ in 1:nsteps
                step!(solver, dt)
            end
            ensure_layout!(u, :g)
            return Array(get_grid_data(u)), ic
        end
        cpu_final, ic = run_diffusion(CPU())
        gpu_final, _  = run_diffusion(GPU())
        # 1. GPU pipeline matches the CPU oracle:
        @test isapprox(gpu_final, cpu_final; rtol=1e-8, atol=1e-10)
        # 2. CPU result matches the analytic diffusion of the (1,1) mode (so it's a
        #    real solve, not a trivial no-op match): u(t) = exp(-ν·k²·t)·u₀.
        analytic = exp(-ν * 2 * (nsteps * dt)) .* ic
        @test isapprox(cpu_final, analytic; rtol=1e-4, atol=1e-8)
    end
end
