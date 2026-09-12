"""
Every timestepper on the real single-GPU dispatch path, WITHOUT a GPU.

`test_gpu_timesteppers.jl` is the value-asserted, CUDA-gated GPU stepper suite.
This file drives the same
`step!` dispatch on `JLArray` device fields (GPUArrays' CPU-backed reference
device array): `plan_is_gpu`, `_field_uses_gpu` and
`_distributed_field_path_required` are all true, `allowscalar(false)` makes any
scalar indexing throw, and every stepper takes exactly the branch it takes on a
`CuArray` — the field-native explicit RK, the matrix-free multistep field path,
the serial diagonal-IMEX broadcasts, and the loud implicit-operator refusals.

What JLArray lacks is a device FFT. The transform backend below stands in for
cuFFT with a CPU twin field: it copies the device buffer to a host twin of the
same domain, runs Tarang's own CPU transform, and copies the result back. The
transform is therefore bit-identical to the CPU chain, which is what makes the
CPU-vs-device comparisons below exact rather than approximate.

Found by this harness before it was a test: MCNAB2 and CNLF2 silently ran as
first-order CNAB1 on every GPU/MPI explicit problem (their startup gate read
`length(state.history) < 2`, but the field path keeps a one-entry history).

Uniquely-prefixed names (gtj_*) — the full suite shares the Main namespace.
"""

using Test
using Tarang
using Printf
using Random

const _GTJ_OK = try
    @eval using JLArrays
    @eval using GPUArrays
    true
catch err
    @info "JLArrays/GPUArrays unavailable; skipping JLArray timestepper test" err
    false
end

if _GTJ_OK
    const _GTJ = JLArrays.JLArray
    const _GTJ_ARCH = Tarang.GPU(JLArrays.JLBackend())
    # Test-scoped only; JLArray is used by nothing else in the package. The same
    # five methods as the other JLArray test files, plus the identity/upload
    # methods the lazy RHS and dealiasing paths call on device data.
    Tarang.is_gpu_array(::_GTJ) = true
    Tarang.architecture(::_GTJ) = _GTJ_ARCH
    Tarang.on_architecture(::Tarang.GPU{JLArrays.JLBackend}, a::Array) = _GTJ(a)
    Tarang.on_architecture(::Tarang.GPU{JLArrays.JLBackend}, a::_GTJ) = a
    Tarang.on_architecture(::Tarang.GPU{JLArrays.JLBackend}, a::AbstractArray) = _GTJ(Array(a))
    Tarang.copy_to_device(a::AbstractArray, ::_GTJ) = _GTJ(Array(a))
    Tarang.copy_to_device(a::_GTJ, ::_GTJ) = copy(a)
    Tarang.array_type(::Tarang.GPU{JLArrays.JLBackend}) = _GTJ
    Tarang.array_type(::Tarang.GPU{JLArrays.JLBackend}, T::Type) = _GTJ{T}

    # ---- cuFFT stand-in: a CPU twin field transformed by Tarang's CPU chain ----
    const _GTJ_TWINS = Dict{Any, Any}()
    function gtj_twin(field)
        get!(_GTJ_TWINS, (objectid(field.bases), field.dtype)) do
            cdist = Distributor(field.dist.coordsys; dtype=field.dtype, device=CPU())
            ScalarField(Domain(cdist, field.bases), "gtj_twin_" * field.name)
        end
    end
    function gtj_sync_scales!(twin, field)
        twin.scales == field.scales && return
        twin.current_layout = :c
        Tarang.preset_scales!(twin, field.scales)
    end
    # Copy `src` into the buffer selected by `getter`/`setter` on `dst`,
    # reallocating (via `make`) only when the shape or eltype differ.
    function gtj_copy_into!(getter, setter, make, dst, src)
        buf = getter(dst)
        if buf === nothing || size(buf) != size(src) || eltype(buf) != eltype(src)
            setter(dst, make(src))
        else
            copyto!(buf, src)
        end
    end
    function Tarang._gpu_forward_transform_backend!(::Tarang.GPU{JLArrays.JLBackend},
                                                    field::Tarang.ScalarField)
        twin = gtj_twin(field)
        gtj_sync_scales!(twin, field)
        gtj_copy_into!(Tarang.get_grid_data, Tarang.set_grid_data!, copy, twin,
                       Array(Tarang.get_grid_data(field)))
        twin.current_layout = :g
        Tarang.forward_transform!(twin)
        gtj_copy_into!(Tarang.get_coeff_data, Tarang.set_coeff_data!, x -> _GTJ(copy(x)),
                       field, Tarang.get_coeff_data(twin))
        return true
    end
    function Tarang._gpu_backward_transform_backend!(::Tarang.GPU{JLArrays.JLBackend}, field)
        twin = gtj_twin(field)
        gtj_sync_scales!(twin, field)
        gtj_copy_into!(Tarang.get_coeff_data, Tarang.set_coeff_data!, copy, twin,
                       Array(Tarang.get_coeff_data(field)))
        twin.current_layout = :c
        Tarang.backward_transform!(twin)
        gtj_copy_into!(Tarang.get_grid_data, Tarang.set_grid_data!, x -> _GTJ(copy(x)),
                       field, Tarang.get_grid_data(twin))
        return true
    end
end

@testset "GPU timestepper dispatch on JLArray device fields" begin
    if !_GTJ_OK
        @test_skip "JLArrays not available"
    else
        GPUArrays.allowscalar(false)

        gtj_grid(u) = (ensure_layout!(u, :g); Array(Tarang.get_grid_data(u)))

        function gtj_solver(arch, eqn, ts, dt, u0; N=16)
            coords = CartesianCoordinates("x")
            dist = Distributor(coords; dtype=Float64, device=arch)
            xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
            u = ScalarField(Domain(dist, (xb,)), "u")
            xs = collect(range(0, 2π, length=N + 1))[1:N]
            ensure_layout!(u, :g)
            copyto!(Tarang.get_grid_data(u), u0.(xs))
            prob = InitialValueProblem([u])
            add_equation!(prob, eqn)
            return InitialValueSolver(prob, ts; dt), u, xs
        end

        function gtj_run(arch, eqn, ts, dts, u0)
            s, u, xs = gtj_solver(arch, eqn, ts, first(dts), u0)
            for d in dts
                step!(s, d)
            end
            return gtj_grid(u), xs
        end

        all_steppers = (RK111(), RK222(), RK443(), RKSMR(), Tarang.RKGFY(), Tarang.RK443_IMEX(),
                        CNAB1(), CNAB2(), SBDF1(), SBDF2(), SBDF3(), SBDF4(),
                        ETD_RK222(), ETD_CNAB2(), ETD_SBDF2(), Tarang.MCNAB2(), Tarang.CNLF2(),
                        Tarang.DiagonalIMEX_RK222(), Tarang.DiagonalIMEX_RK443(), Tarang.DiagonalIMEX_SBDF2())
        diagonal = (Tarang.DiagonalIMEX_RK222(), Tarang.DiagonalIMEX_RK443(), Tarang.DiagonalIMEX_SBDF2())
        implicit_supported = (RK222(), RK443(), SBDF2(), diagonal...)
        gtj_name(ts) = nameof(typeof(ts))

        @testset "Reference steps on CPU and device" begin
            for arch in (CPU(), _GTJ_ARCH)
                for ts in (RK222(), Tarang.DiagonalIMEX_RK222())
                    values, _ = gtj_run(arch, "dt(u) = -u", ts, [0.1], x -> 1.0)
                    @test values ≈ fill(0.905, 16) atol=1e-13
                end
                for ts in (SBDF3(), SBDF4())
                    s, u, _ = gtj_solver(arch, "dt(u) = -u", ts, 0.1, x -> 1.0)
                    step!(s)
                    @test gtj_grid(u) ≈ fill(0.9, 16) atol=1e-13
                    step!(s)
                    @test gtj_grid(u) ≈ fill(0.8133333333333334, 16) atol=1e-13
                end
            end
        end

        # Which schemes are the same arithmetic on both architectures for an
        # explicit problem. ETD_* legitimately substitute (with a warning) an RK/
        # multistep step when there is no linear operator, so they are checked for
        # accuracy but not bit parity.
        parity_exact = (RK111(), RK222(), RK443(), RKSMR(), Tarang.RKGFY(), Tarang.RK443_IMEX(),
                        CNAB1(), CNAB2(), SBDF1(), SBDF2(), SBDF3(), SBDF4(),
                        Tarang.MCNAB2(), Tarang.CNLF2(), diagonal...)

        @testset "explicit nonlinear problem: device == CPU, and startup-limited order" begin
            # dt(u) = -u², u0 ≡ 1 → 1/(1+t). Spatially exact, so the error is the
            # time-discretization error alone.
            expected_order = Dict(:RK111 => 1, :CNAB1 => 1, :SBDF1 => 1,
                                  :RK443 => 3, :RKSMR => 3, :RK443_IMEX => 3,
                                  :SBDF3 => 2, :SBDF4 => 2)
            for ts in all_steppers
                T = 0.4
                errs = map((0.02, 0.01)) do dt
                    vals, _ = gtj_run(_GTJ_ARCH, "dt(u) = -u*u", ts, fill(dt, round(Int, T / dt)), x -> 1.0)
                    maximum(abs, vals .- 1 / (1 + T))
                end
                order = get(expected_order, gtj_name(ts), 2)
                rate = log2(errs[1] / errs[2])
                @test errs[2] < 5e-3
                # MCNAB2/CNLF2 used to fall to CNAB1 (rate 1) on every device step.
                @test rate > order - 0.5

                if any(p -> typeof(p) === typeof(ts), parity_exact)
                    dts = fill(0.01, 20)
                    dev, _ = gtj_run(_GTJ_ARCH, "dt(u) = -u*u", ts, dts, x -> 1 + 0.1cos(x))
                    cpu, _ = gtj_run(CPU(), "dt(u) = -u*u", ts, dts, x -> 1 + 0.1cos(x))
                    @test maximum(abs, dev .- cpu) < 1e-12
                end
            end
        end

        @testset "implicit operator: automatic diagonal dispatch" begin
            # dt(u) - 0.5 lap(u) = 0, u0 = cos 2x → e^{-2t} cos 2x.
            T = 0.2; dt = 0.005
            for ts in all_steppers
                if any(p -> typeof(p) === typeof(ts), implicit_supported)
                    dev, xs = gtj_run(_GTJ_ARCH, "dt(u) - 0.5*lap(u) = 0", ts, fill(dt, round(Int, T / dt)), x -> cos(2x))
                    cpu, _ = gtj_run(CPU(), "dt(u) - 0.5*lap(u) = 0", ts, fill(dt, round(Int, T / dt)), x -> cos(2x))
                    @test maximum(abs, dev .- exp(-2T) .* cos.(2 .* xs)) < 1e-4
                    @test dev ≈ cpu atol=1e-12 rtol=1e-12
                else
                    s, _, _ = gtj_solver(_GTJ_ARCH, "dt(u) - 0.5*lap(u) = 0", ts, dt, x -> cos(2x))
                    @test_throws ErrorException step!(s)
                end
            end
        end

        @testset "Attached operators use the internal diagonal path" begin
            for arch in (CPU(), _GTJ_ARCH), ts in (RK222(), RK443(), SBDF2())
                s, u, xs = gtj_solver(arch, "dt(u) = 0", ts, 0.005, x -> cos(2x))
                L = SpectralLinearOperator(u.dist, u.bases, :laplacian; ν=0.5)
                set_spectral_linear_operator!(s, L)
                for _ in 1:40
                    step!(s)
                end
                @test typeof(s.timestepper_state.timestepper) === typeof(ts)
                @test haskey(s.timestepper_state.timestepper_data, :sdi_Lmap)
                @test gtj_grid(u) ≈ exp(-0.4) .* cos.(2 .* xs) atol=1e-4
            end
        end

        @testset "GPU public schemes reject cross-field implicit coupling" begin
            for ts in (RK222(), RK443(), SBDF2())
                _, u, _ = gtj_solver(_GTJ_ARCH, "dt(u) = 0", ts, 0.01, x -> 1.0)
                v = ScalarField(Domain(u.dist, u.bases), "v")
                copyto!(grid_data!(v), ones(16))
                problem = InitialValueProblem([u, v])
                add_equation!(problem, "dt(u) + v = 0")
                add_equation!(problem, "dt(v) = 0")
                s = InitialValueSolver(problem, ts; dt=0.01)
                @test_throws ArgumentError step!(s)
                @test gtj_grid(u) == ones(16)
                @test s.sim_time == 0
                @test s.iteration == 0
            end
        end

        @testset "SBDF2 can switch from a matrix solve to an attached operator" begin
            s, u, xs = gtj_solver(CPU(), "dt(u) = 0", SBDF2(), 0.005, x -> cos(2x))
            step!(s)
            L = SpectralLinearOperator(u.dist, u.bases, :laplacian; ν=0.5)
            set_spectral_linear_operator!(s, L)
            step!(s)
            @test gtj_grid(u) ≈ cos.(2 .* xs) ./ 1.01 atol=1e-12
            @test s.iteration == 2
        end

        @testset "DiagonalIMEX: derivative-of-self implicit term on device" begin
            # dt(u) + 0.3 d(u,x) - 0.1 lap(u) = 0 → e^{-0.4t} cos(2(x - 0.3t)).
            # Exercises the device upload of the (ik)^n multiplier.
            T = 0.2; dt = 0.005
            for ts in implicit_supported
                dev, xs = gtj_run(_GTJ_ARCH, "dt(u) + 0.3*d(u,x) - 0.1*lap(u) = 0", ts, fill(dt, round(Int, T / dt)), x -> cos(2x))
                @test maximum(abs, dev .- exp(-0.4T) .* cos.(2 .* (xs .- 0.3T))) < 1e-4
            end
        end

        @testset "nonidentity mass is refused before device state advances" begin
            for ts in all_steppers
                s, u, _ = gtj_solver(_GTJ_ARCH, "2*dt(u) = -u", ts, 0.01, x -> 1.0)
                @test Tarang.compiled_problem(s.problem).mass_matrix === nothing
                before = gtj_grid(u)
                err = try
                    step!(s)
                    nothing
                catch ex
                    ex
                end
                @test err isa ArgumentError
                @test occursin("non-identity mass", err === nothing ? "" : sprint(showerror, err))
                @test gtj_grid(u) == before
                @test s.sim_time == 0
                @test s.iteration == 0
            end
            for ts in diagonal
                s, u, _ = gtj_solver(_GTJ_ARCH, "2*dt(u) + u = 0", ts, 0.01, x -> 1.0)
                @test_throws ArgumentError step!(s)
                @test gtj_grid(u) == ones(16)
            end
        end

        @testset "variable dt: device == CPU" begin
            dts = repeat([0.01, 0.02], 10)
            for (ts, eqn) in ((CNAB2(), "dt(u) = -u*u"), (SBDF2(), "dt(u) = -u*u"),
                              (SBDF3(), "dt(u) = -u*u"), (SBDF4(), "dt(u) = -u*u"),
                              (Tarang.MCNAB2(), "dt(u) = -u*u"), (Tarang.CNLF2(), "dt(u) = -u*u"),
                              (SBDF2(), "dt(u) - 0.1*lap(u) = -u*u"),
                              (RK443(), "dt(u) - 0.1*lap(u) = -u*u"),
                              (Tarang.DiagonalIMEX_SBDF2(), "dt(u) - 0.1*lap(u) = -u*u"),
                              (Tarang.DiagonalIMEX_RK443(), "dt(u) - 0.1*lap(u) = -u*u"))
                dev, _ = gtj_run(_GTJ_ARCH, eqn, ts, dts, x -> 1 + 0.5cos(x))
                cpu, _ = gtj_run(CPU(), eqn, ts, dts, x -> 1 + 0.5cos(x))
                @test maximum(abs, dev .- cpu) < 1e-12
            end
        end

        @testset "registered forcings on device: stochastic == CPU, deterministic exact" begin
            # Stochastic: same seed ⇒ the counter-based phase kernel gives the same
            # realization on device and host, and the field paths agree to roundoff.
            function gtj_forced(arch, forcing_ctor, ts; N=16, dt=0.01, nst=20)
                coords = CartesianCoordinates("x")
                dist = Distributor(coords; dtype=Float64, device=arch)
                xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
                u = ScalarField(Domain(dist, (xb,)), "u")
                ensure_layout!(u, :g); fill!(Tarang.get_grid_data(u), 0.0)
                prob = InitialValueProblem([u]); add_equation!(prob, "dt(u) = 0")
                add_stochastic_forcing!(prob, :u, forcing_ctor(arch))
                s = InitialValueSolver(prob, ts; dt)
                for _ in 1:nst; step!(s); end
                return gtj_grid(u), s.sim_time
            end
            stoch(arch) = StochasticForcing(field_size=(16,), domain_size=(2π,), energy_injection_rate=0.1,
                                            k_forcing=4.0, dk_forcing=1.0, dt=0.01,
                                            rng=Random.MersenneTwister(3), architecture=arch)
            dev, _ = gtj_forced(_GTJ_ARCH, stoch, RK222())
            cpu, _ = gtj_forced(CPU(), stoch, RK222())
            @test maximum(abs, dev .- cpu) < 1e-12
            @test maximum(abs, dev) > 0

            # Deterministic: dt(u) = 0 with F = cos x integrates exactly to t·cos x; the
            # device path transforms the physical-space realization through the
            # device field (set_local_data! on device arrays + the device transform).
            det(arch) = DeterministicForcing((x, t, p) -> cos.(x), (16,); architecture=arch)
            dev, T = gtj_forced(_GTJ_ARCH, det, RK222())
            xs = collect(range(0, 2π, length=17))[1:16]
            @test maximum(abs, dev .- T .* cos.(xs)) < 1e-12
        end
    end
end
