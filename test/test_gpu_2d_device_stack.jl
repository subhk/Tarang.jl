"""
2D GPU stack device-safety + CPU/GPU value parity WITHOUT a GPU.

`test_2d_gpu_domain_compat.jl` covers the 2D pure-Fourier algebraic-constraint
refresh; this file covers the rest of the 2D GPU stack that had no coverage at
all: the stochastic-forcing device kernels (Hermitian symmetry, zero mode,
diagnostics), device-backed field construction/reductions, and the host→device
staging contract the coupled-subproblem `gather_alg_F!` relies on.

Method (same as test_2d_gpu_domain_compat.jl): run the REAL functions on
`JLArray` — GPUArrays' CPU-backed reference device array — with
`allowscalar(false)`, so any scalar `getindex`/`setindex!` throws exactly as it
would on a `CuArray`. A kernel that completes here is scalar-index-free; each
result is then compared against the CPU implementation, which is the oracle.

Beyond the two methods the older files patch, this one also supplies
`Tarang.array_type` for the JLArray backend — that is all it takes to build a
GPU-ARCHITECTURE `ScalarField` whose buffers are device arrays, which is what
lets the forcing/reduction paths be exercised end to end here.
"""

using Test
using Random
using Tarang
using LinearAlgebra

const _JLSTACK_OK = try
    @eval using JLArrays
    @eval using GPUArrays
    @eval using FFTW
    true
catch err
    @info "JLArrays/GPUArrays unavailable; skipping 2D GPU device-stack test" err
    false
end

if _JLSTACK_OK
    const _JLS = JLArrays.JLArray
    const _JLS_ARCH = Tarang.GPU(JLArrays.JLBackend())
    # Test-scoped only; JLArray is used by nothing else in the package.
    Tarang.is_gpu_array(::_JLS) = true
    Tarang.architecture(::_JLS) = _JLS_ARCH
    Tarang.on_architecture(::Tarang.GPU{JLArrays.JLBackend}, a::Array) = _JLS(a)
    Tarang.copy_to_device(a::AbstractArray, ::_JLS) = _JLS(Array(a))
    Tarang.copy_to_device(a::_JLS, ::_JLS) = copy(a)
    Tarang.array_type(::Tarang.GPU{JLArrays.JLBackend}) = _JLS
    Tarang.array_type(::Tarang.GPU{JLArrays.JLBackend}, T::Type) = _JLS{T}
end

@testset "2D GPU stack device-safety (JLArray)" begin
    if !_JLSTACK_OK
        @test_skip "JLArrays not available"
    else
        GPUArrays.allowscalar(false)

        @testset "Hermitian symmetry: device kernel == CPU loop" begin
            # The forcing spectrum is a FULL complex spectrum (field_size, not an
            # rfft half), so the conjugate partner is index `n+2-i`. The GPU kernel
            # visits every (i,j) concurrently while the CPU version walks them in
            # order — if the two disagreed (or the kernel raced), the forcing would
            # stop being real-valued in grid space and the physics would be junk.
            for (nx, ny) in ((8, 8), (9, 7), (16, 12))
                Random.seed!(4)
                a = randn(ComplexF64, nx, ny)
                cpu = copy(a)
                dev = _JLS(copy(a))
                Tarang._enforce_hermitian_symmetry!(cpu, CPU())
                Tarang._enforce_hermitian_symmetry!(dev, _JLS_ARCH)
                @test Array(dev) == cpu
            end
        end

        @testset "Hermitian symmetry makes the grid field real" begin
            Random.seed!(5)
            dev = _JLS(randn(ComplexF64, 8, 8))
            Tarang._enforce_hermitian_symmetry!(dev, _JLS_ARCH)
            Tarang._set_zero_mode!(dev, _JLS_ARCH)
            @test maximum(abs, imag.(ifft(Array(dev)))) < 1e-12
        end

        @testset "_set_zero_mode! device == CPU" begin
            Random.seed!(6)
            a = randn(ComplexF64, 8, 8)
            cpu = copy(a)
            dev = _JLS(copy(a))
            Tarang._set_zero_mode!(cpu, CPU())
            Tarang._set_zero_mode!(dev, _JLS_ARCH)
            @test Array(dev) == cpu
        end

        @testset "GPU-architecture 2D field: buffers are device arrays" begin
            coords = CartesianCoordinates("x", "y")
            dist = Distributor(coords; dtype=Float64, device=_JLS_ARCH)
            dom = Domain(dist, (RealFourier(coords["x"]; size=16, bounds=(0.0, 2π)),
                                RealFourier(coords["y"]; size=12, bounds=(0.0, 2π))))
            u = ScalarField(dom, "u")
            @test get_grid_data(u) isa _JLS
            @test get_coeff_data(u) isa _JLS
            @test size(get_grid_data(u)) == (16, 12)
            @test size(get_coeff_data(u)) == (9, 12)   # canonical rfft layout

            Random.seed!(7)
            host = randn(16, 12)
            get_grid_data(u) .= _JLS(host)
            # CFL-style reduction on device data — must not scalar-index
            @test maximum(abs, get_grid_data(u)) ≈ maximum(abs, host)
            @test sum(get_grid_data(u)) ≈ sum(host)
        end

        @testset "StochasticForcing 2D on a device architecture" begin
            fg = Tarang.StochasticForcing(; field_size=(16, 16), architecture=_JLS_ARCH,
                                          dtype=Float64, k_forcing=4.0, dk_forcing=1.0,
                                          energy_injection_rate=0.1, dt=0.01)
            fc = Tarang.StochasticForcing(; field_size=(16, 16), architecture=CPU(),
                                          dtype=Float64, k_forcing=4.0, dk_forcing=1.0,
                                          energy_injection_rate=0.1, dt=0.01)
            # The forcing spectrum is architecture-independent data
            @test Array(fg.spectrum) ≈ Array(fc.spectrum)

            Tarang.generate_forcing!(fg, 0.0)          # device kernels, no scalar indexing
            cf = fg.cached_forcing
            @test cf isa _JLS
            @test all(isfinite, Array(cf))
            # generate_forcing! ends with Hermitian symmetry + zero mode: the
            # realization must be a spectrum of a REAL field with no mean.
            @test abs(Array(cf)[1, 1]) == 0
            @test maximum(abs, imag.(ifft(Array(cf)))) < 1e-10
        end

        @testset "Forcing diagnostics: device path == CPU path" begin
            Random.seed!(8)
            fc = Tarang.StochasticForcing(; field_size=(8, 8), architecture=CPU(),
                                          dtype=Float64, k_forcing=3.0, dk_forcing=1.0,
                                          energy_injection_rate=0.1, dt=0.01)
            Tarang.generate_forcing!(fc, 0.0)
            fg = Tarang.StochasticForcing(; field_size=(8, 8), architecture=_JLS_ARCH,
                                          dtype=Float64, k_forcing=3.0, dk_forcing=1.0,
                                          energy_injection_rate=0.1, dt=0.01)
            copyto!(fg.cached_forcing, _JLS(Array(fc.cached_forcing)))

            sol = randn(ComplexF64, 8, 8)
            prev = randn(ComplexF64, 8, 8)
            # one-argument form (Itô-style) and two-argument form (Stratonovich)
            @test Tarang._diagnostic_inner_product(fg, _JLS(copy(sol))) ≈
                  Tarang._diagnostic_inner_product(fc, copy(sol))
            @test Tarang._diagnostic_inner_product(fg, _JLS(copy(sol)), _JLS(copy(prev))) ≈
                  Tarang._diagnostic_inner_product(fc, copy(sol), copy(prev))
        end

        @testset "gather_alg_F! host→device staging contract" begin
            # `gather_alg_F!` builds the sparse BC/constraint vector on the host
            # (scalar writes must stay off device arrays) and uploads it with a
            # direct `copyto!`. Routing it through `_assign_to_buffer!` instead
            # killed every GPU coupled-subproblem step at the pre-stage gather,
            # because that helper's same-architecture guard refuses exactly this
            # transfer. Pin both halves of that contract.
            host = ComplexF64[1, 2, 3, 4]
            dev = _JLS(zeros(ComplexF64, 4))
            copyto!(dev, host)
            @test Array(dev) == host
            @test_throws Exception Tarang._assign_to_buffer!(_JLS(zeros(ComplexF64, 4)), host)
        end
    end
end
