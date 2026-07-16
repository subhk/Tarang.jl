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
using Random
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
            cpu_u, gpu_u, data = _pair(("x","y"), mk, Float64)
            @test Tarang.gpu_forward_transform!(gpu_u)
            @test Tarang.gpu_backward_transform!(gpu_u)
            _check(cpu_u, gpu_u, data)
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


    @testset "End-to-end forced RealFourier × ChebyshevT IVP" begin
        CUDA.allowscalar(false)
        nx, nz = 8, 10
        dt = 1e-3

        function build_forced_mixed_solver(device)
            coords = CartesianCoordinates("x", "z")
            dist = Distributor(coords; dtype=Float64, device)
            xbasis = RealFourier(coords["x"]; size=nx, bounds=(0.0, 2pi))
            zbasis = ChebyshevT(coords["z"]; size=nz, bounds=(0.0, 1.0))
            domain = Domain(dist, (xbasis, zbasis))

            b = ScalarField(domain, "b")
            tau1 = ScalarField(dist, "tau1", (xbasis,), Float64)
            tau2 = ScalarField(dist, "tau2", (xbasis,), Float64)
            _, ez = unit_vector_fields(coords, dist)
            lift_basis = derivative_basis(zbasis, 1)
            tau_lift(A) = lift(A, lift_basis, -1)
            grad_b = grad(b) + ez * tau_lift(tau1)

            problem = IVP([b, tau1, tau2])
            add_parameters!(problem; kappa=0.1, grad_b, tau_lift)
            add_equation!(problem,
                          "∂t(b) - kappa*div(grad_b) + tau_lift(tau2) = 0")
            add_bc!(problem, "b(z=0) = 0")
            add_bc!(problem, "b(z=1) = 0")

            forcing = SeparableStochasticForcing(
                fourier_size=(nx,),
                chebyshev_basis=zbasis,
                chebyshev_profile=z -> z * (1 - z),
                domain_size=(2pi,),
                energy_injection_rate=0.05,
                k_forcing=2.0,
                dk_forcing=0.5,
                dt=dt,
                architecture=device,
                rng=MersenneTwister(1234),
            )
            add_stochastic_forcing!(problem, :b, forcing)
            solver = InitialValueSolver(problem, RK222(); dt)
            return solver, b, forcing
        end

        cpu_solver, cpu_b, _ = build_forced_mixed_solver(CPU())
        gpu_solver, gpu_b, gpu_forcing = build_forced_mixed_solver(GPU())
        @test gpu_solver.base.matsolver === CuSparseLU
        @test gpu_forcing.cached_forcing isa CUDA.CuArray
        @test gpu_forcing.random_phases isa CUDA.CuArray
        @test get_coeff_data(gpu_b) isa CUDA.CuArray

        for _ in 1:3
            step!(cpu_solver, dt)
            step!(gpu_solver, dt)
        end
        CUDA.synchronize()

        ensure_layout!(cpu_b, :c)
        ensure_layout!(gpu_b, :c)
        @test Array(get_coeff_data(gpu_b)) ≈ get_coeff_data(cpu_b) rtol=2e-7 atol=1e-9

        subproblems = gpu_solver.problem.parameters["subproblems"]
        @test !isempty(subproblems)
        for sp in subproblems
            for buffer in values(sp.runtime.rk_buffers)
                @test buffer isa CUDA.CuArray
            end
            for matrix_solver in values(sp.LHS_solvers)
                @test matrix_solver isa CuSparseLU
                @test matrix_solver.A_csr.nzVal isa CUDA.CuArray
                @test matrix_solver.rhs_buffer isa CUDA.CuArray
                @test matrix_solver.solution_buffer isa CUDA.CuArray
                @test matrix_solver.temp_buffer isa CUDA.CuArray
            end
        end

        forcing_before_stage = copy(gpu_forcing.cached_forcing)
        @test generate_forcing!(gpu_forcing, gpu_forcing.last_update_time, 2) ===
              gpu_forcing.cached_forcing
        @test gpu_forcing.cached_forcing == forcing_before_stage
        step!(gpu_solver, dt)
        @test gpu_forcing.cached_forcing != forcing_before_stage

        for _ in 1:4
            step!(gpu_solver, dt)
        end
        CUDA.synchronize()
        allocation_stats = getfield(CUDA, :alloc_stats)
        allocated_before = allocation_stats.alloc_bytes
        step!(gpu_solver, dt)
        CUDA.synchronize()
        @test allocation_stats.alloc_bytes - allocated_before == 0
    end


    @testset "End-to-end GPU vs CPU: forced 3D periodic IVP" begin
        CUDA.allowscalar(false)
        n = 8
        dt = 1e-3

        function build_periodic_3d(device; forced=false)
            coords = CartesianCoordinates("x", "y", "z")
            dist = Distributor(coords; dtype=Float64, device)
            bases = ntuple(3) do d
                name = ("x", "y", "z")[d]
                RealFourier(coords[name]; size=n, bounds=(0.0, 2pi))
            end
            domain = Domain(dist, bases)
            q = ScalarField(domain, "q3")
            problem = IVP([q])
            add_parameters!(problem; nu=0.05)
            add_equation!(problem, "∂t(q3) - nu*Δ(q3) = 0")

            forcing = nothing
            if forced
                forcing = StochasticForcing(
                    field_size=(n, n, n),
                    domain_size=(2pi, 2pi, 2pi),
                    energy_injection_rate=0.02,
                    k_forcing=2.0,
                    dk_forcing=0.5,
                    dt=dt,
                    architecture=device,
                    rng=MersenneTwister(77),
                )
                add_stochastic_forcing!(problem, :q3, forcing)
            end

            mesh = Tarang.get_grid_coordinates(domain; on_device=false)
            x = reshape(mesh["x"], n, 1, 1)
            y = reshape(mesh["y"], 1, n, 1)
            z = reshape(mesh["z"], 1, 1, n)
            initial = @. sin(x) * cos(y) * sin(z)
            ensure_layout!(q, :g)
            get_grid_data(q) .= device isa CPU ? initial : CUDA.CuArray(initial)
            solver = InitialValueSolver(problem, RK222(); dt)
            return solver, q, forcing
        end

        cpu_solver, cpu_q, _ = build_periodic_3d(CPU())
        gpu_solver, gpu_q, _ = build_periodic_3d(GPU())
        for _ in 1:4
            step!(cpu_solver, dt)
            step!(gpu_solver, dt)
        end
        ensure_layout!(cpu_q, :c)
        ensure_layout!(gpu_q, :c)
        @test Array(get_coeff_data(gpu_q)) ≈ get_coeff_data(cpu_q) rtol=2e-8 atol=1e-10
        @test get_coeff_data(gpu_q) isa CUDA.CuArray
        @test get(gpu_solver.problem.parameters, "L_matrix", nothing) === nothing
        @test get(gpu_solver.problem.parameters, "M_matrix", nothing) === nothing

        forced_solver, forced_q, forcing = build_periodic_3d(GPU(); forced=true)
        step!(forced_solver, dt)
        @test forcing.cached_forcing isa CUDA.CuArray
        @test get_coeff_data(forced_q) isa CUDA.CuArray
        @test all(isfinite, Array(get_coeff_data(forced_q)))
        periodic_state = forced_solver.timestepper_state
        for field in Iterators.flatten(periodic_state.workspace_fields)
            @test get_grid_data(field) isa CUDA.CuArray
            @test get_coeff_data(field) isa CUDA.CuArray
        end
        first_forcing = copy(forcing.cached_forcing)
        @test generate_forcing!(forcing, forcing.last_update_time, 2) ===
              forcing.cached_forcing
        @test forcing.cached_forcing == first_forcing
        step!(forced_solver, dt)
        @test forcing.cached_forcing != first_forcing

        ensure_layout!(forced_q, :c)
        coeff_before_write = copy(get_coeff_data(forced_q))
        ensure_layout!(forced_q, :g)
        grid_before_write = Array(get_grid_data(forced_q))
        output_root = mktempdir()
        output = add_file_handler(joinpath(output_root, "periodic3d"), forced_solver;
                                  iter=1, max_writes=1)
        add_task!(output, forced_q; name="q3_grid", layout="g")
        add_task!(output, forced_q; name="q3_coeff", layout="c")
        process!(output)

        output_file = Tarang.current_file(output)
        written_grid = dropdims(Tarang.group_ncread(output_file, "vars", "q3_grid"); dims=1)
        written_coeff_parts = Tarang.group_ncread(output_file, "vars", "q3_coeff")
        written_coeff = complex.(dropdims(selectdim(written_coeff_parts, 2, 1); dims=1),
                                 dropdims(selectdim(written_coeff_parts, 2, 2); dims=1))
        @test written_grid ≈ grid_before_write rtol=2e-8 atol=1e-10
        @test written_coeff ≈ Array(coeff_before_write) rtol=2e-8 atol=1e-10
        ensure_layout!(forced_q, :c)
        @test get_coeff_data(forced_q) ≈ coeff_before_write rtol=2e-8 atol=1e-10
        @test get_grid_data(forced_q) isa CUDA.CuArray
        @test get_coeff_data(forced_q) isa CUDA.CuArray

        for _ in 1:4
            step!(forced_solver, dt)
        end
        CUDA.synchronize()
        allocation_stats = getfield(CUDA, :alloc_stats)
        allocated_before = allocation_stats.alloc_bytes
        step!(forced_solver, dt)
        CUDA.synchronize()
        @test allocation_stats.alloc_bytes - allocated_before == 0
    end


    @testset "End-to-end forced 3D Fourier × Fourier × ChebyshevT IVP" begin
        CUDA.allowscalar(false)
        nx, ny, nz = 8, 8, 10
        dt = 1e-3

        function build_forced_mixed_3d(device)
            coords = CartesianCoordinates("x", "y", "z")
            dist = Distributor(coords; dtype=Float64, device)
            xbasis = RealFourier(coords["x"]; size=nx, bounds=(0.0, 2pi))
            ybasis = RealFourier(coords["y"]; size=ny, bounds=(0.0, 2pi))
            zbasis = ChebyshevT(coords["z"]; size=nz, bounds=(0.0, 1.0))
            domain = Domain(dist, (xbasis, ybasis, zbasis))

            b = ScalarField(domain, "b3")
            tau1 = ScalarField(dist, "tau31", (xbasis, ybasis), Float64)
            tau2 = ScalarField(dist, "tau32", (xbasis, ybasis), Float64)
            _, _, ez = unit_vector_fields(coords, dist)
            lift_basis = derivative_basis(zbasis, 1)
            tau_lift3(A) = lift(A, lift_basis, -1)
            grad_b3 = grad(b) + ez * tau_lift3(tau1)

            problem = IVP([b, tau1, tau2])
            add_parameters!(problem; kappa=0.1, grad_b3, tau_lift3)
            add_equation!(problem,
                          "∂t(b3) - kappa*div(grad_b3) + tau_lift3(tau32) = 0")
            add_bc!(problem, "b3(z=0) = 0")
            add_bc!(problem, "b3(z=1) = 0")

            forcing = SeparableStochasticForcing(
                fourier_size=(nx, ny),
                chebyshev_basis=zbasis,
                chebyshev_profile=z -> z * (1 - z),
                domain_size=(2pi, 2pi),
                energy_injection_rate=0.05,
                k_forcing=2.0,
                dk_forcing=0.5,
                dt=dt,
                architecture=device,
                rng=MersenneTwister(2026),
            )
            add_stochastic_forcing!(problem, :b3, forcing)
            solver = InitialValueSolver(problem, RK222(); dt)
            return solver, b, forcing
        end

        cpu_solver, cpu_b, _ = build_forced_mixed_3d(CPU())
        gpu_solver, gpu_b, gpu_forcing = build_forced_mixed_3d(GPU())
        @test gpu_solver.base.matsolver === CuSparseLU
        @test gpu_forcing.cached_forcing isa CUDA.CuArray
        @test gpu_forcing.fourier_realization isa CUDA.CuArray
        ensure_layout!(gpu_b, :g)
        @test Tarang.gpu_forward_transform!(gpu_b)
        @test Tarang.gpu_backward_transform!(gpu_b)

        for _ in 1:3
            step!(cpu_solver, dt)
            step!(gpu_solver, dt)
        end
        CUDA.synchronize()
        ensure_layout!(cpu_b, :c)
        ensure_layout!(gpu_b, :c)
        @test Array(get_coeff_data(gpu_b)) ≈ get_coeff_data(cpu_b) rtol=3e-7 atol=2e-9

        subproblems = gpu_solver.problem.parameters["subproblems"]
        @test !isempty(subproblems)
        mixed_state = gpu_solver.timestepper_state
        for field in Iterators.flatten(mixed_state.workspace_fields)
            @test get_grid_data(field) isa CUDA.CuArray
            @test get_coeff_data(field) isa CUDA.CuArray
        end
        for sp in subproblems
            for buffer in values(sp.runtime.rk_buffers)
                @test buffer isa CUDA.CuArray
            end
            for matrix_solver in values(sp.LHS_solvers)
                @test matrix_solver isa CuSparseLU
                @test matrix_solver.A_csr.nzVal isa CUDA.CuArray
                @test matrix_solver.rhs_buffer isa CUDA.CuArray
                @test matrix_solver.solution_buffer isa CUDA.CuArray
                @test matrix_solver.temp_buffer isa CUDA.CuArray
            end
        end

        forcing_before_stage = copy(gpu_forcing.cached_forcing)
        @test generate_forcing!(gpu_forcing, gpu_forcing.last_update_time, 2) ===
              gpu_forcing.cached_forcing
        @test gpu_forcing.cached_forcing == forcing_before_stage
        step!(gpu_solver, dt)
        @test gpu_forcing.cached_forcing != forcing_before_stage

        ensure_layout!(gpu_b, :c)
        coeff_before_write = copy(get_coeff_data(gpu_b))
        ensure_layout!(gpu_b, :g)
        grid_before_write = Array(get_grid_data(gpu_b))
        output_root = mktempdir()
        output = add_file_handler(joinpath(output_root, "mixed3d"), gpu_solver;
                                  iter=1, max_writes=1)
        add_task!(output, gpu_b; name="b3", layout="g")
        process!(output)

        written = dropdims(
            Tarang.group_ncread(Tarang.current_file(output), "vars", "b3"); dims=1)
        @test written ≈ grid_before_write rtol=3e-7 atol=2e-9
        ensure_layout!(gpu_b, :c)
        @test get_coeff_data(gpu_b) ≈ coeff_before_write rtol=3e-7 atol=2e-9
        @test get_grid_data(gpu_b) isa CUDA.CuArray
        @test get_coeff_data(gpu_b) isa CUDA.CuArray

        for _ in 1:4
            step!(gpu_solver, dt)
        end
        CUDA.synchronize()
        allocation_stats = getfield(CUDA, :alloc_stats)
        allocated_before = allocation_stats.alloc_bytes
        step!(gpu_solver, dt)
        CUDA.synchronize()
        @test allocation_stats.alloc_bytes - allocated_before == 0
    end
end
