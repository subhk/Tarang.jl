"""
Device-safety of the 2D pure-Fourier GPU algebraic-constraint refresh WITHOUT a GPU.

A pure-Fourier GPU IVP (e.g. the 2D vorticity–streamfunction turbulence example)
builds no global matrix; it refreshes its algebraic constraints — the
streamfunction Poisson solve `ψ̂ = -q̂/k²`, the |k|² grid, the spectral
`skew(grad)` velocity — spectrally on-device every RHS. The transforms between
those steps need cuFFT, but the refresh kernels themselves are broadcasts and
reductions on coefficient arrays.

This runs those real kernels on `JLArray` (GPUArrays' CPU-backed reference GPU
array) with `allowscalar(false)`, so any scalar getindex/setindex! throws exactly
as on a CuArray — a kernel that completes here is scalar-index-free — and checks
each against a CPU reference. Runs on any machine; no GPU needed.
"""

using Test
using Random
using Tarang
using Statistics: mean

const _JL2D_OK = try
    @eval using JLArrays
    @eval using GPUArrays
    @eval using FFTW
    true
catch err
    @info "JLArrays/GPUArrays unavailable; skipping 2D GPU refresh device-safety test" err
    false
end

if _JL2D_OK
    const _JLA2D = JLArrays.JLArray
    # Wire JLArray into Tarang's GPU dispatch so the REAL refresh functions treat
    # it as device memory — mirrors ext/cuda/architecture.jl and ext/cuda/utils.jl.
    # Test-scoped; JLArray is used by nothing else in the suite.
    Tarang.is_gpu_array(::_JLA2D) = true
    Tarang.architecture(::_JLA2D) = Tarang.GPU{Symbol}(:jl)
    Tarang.copy_to_device(a::AbstractArray, ::_JLA2D) = _JLA2D(Array(a))
    Tarang.copy_to_device(a::_JLA2D, ::_JLA2D) = copy(a)
end

@testset "2D pure-Fourier GPU refresh device-safety (JLArray)" begin
    if !_JL2D_OK
        @test_skip "JLArrays not available"
    else
        GPUArrays.allowscalar(false)
        Random.seed!(1)

        @testset "Poisson invert ψ̂ = -q̂/k² (k=0 pinned, no scalar write)" begin
            N = 8
            q = ComplexF64.(randn(N, N)) .+ im .* randn(N, N)
            k2 = abs.(randn(N, N)); k2[1, 1] = 0.0        # k=0 (mean) mode
            ref = ifelse.(k2 .> 1e-12, .-q ./ k2, zero(ComplexF64))

            qd = _JLA2D(copy(q)); k2d = _JLA2D(copy(k2))
            Tarang._poisson_invert!(qd, k2d)              # real function, on device
            @test Array(qd) ≈ ref rtol=1e-12
            @test Array(qd)[1, 1] == 0.0 + 0.0im          # gauge / k=0 mode exactly zero
        end

        @testset "2D k² grid (rfft first axis × full-fft second axis)" begin
            N = 8; k0 = 1.0
            kx = k0 .* collect(0:(N ÷ 2))                  # rfft: length N/2+1
            ky = k0 .* Float64.([0:(N ÷ 2 - 1); -(N ÷ 2):-1])  # fft: length N
            ref = [kx[i]^2 + ky[j]^2 for i in 1:length(kx), j in 1:length(ky)]

            ksq = _JLA2D(zeros(Float64, length(kx), length(ky)))
            Tarang.add_wavenumber_squared_contribution!(ksq, kx, 1, 2)   # real function
            Tarang.add_wavenumber_squared_contribution!(ksq, ky, 2, 2)
            @test Array(ksq) ≈ ref rtol=1e-12
        end

        @testset "Poisson-solve-into broadcast (both sign conventions)" begin
            # verbatim from _spectral_poisson_solve_into! (state_utils.jl):
            #   dest .= ifelse.(k2 .> 1e-12, sign .* src ./ k2, 0)
            N = 8
            src = ComplexF64.(randn(N, N ÷ 2 + 1)) .+ im .* randn(N, N ÷ 2 + 1)
            k2 = abs.(randn(N, N ÷ 2 + 1)); k2[1, 1] = 0.0
            for was_negated in (true, false)
                s = was_negated ? -1.0 : 1.0
                ref = ifelse.(k2 .> 1e-12, s .* src ./ k2, zero(ComplexF64))
                dest = _JLA2D(similar(src)); srcd = _JLA2D(copy(src)); k2d = _JLA2D(copy(k2))
                dest .= ifelse.(k2d .> 1e-12, s .* srcd ./ k2d, zero(eltype(dest)))
                @test Array(dest) ≈ ref rtol=1e-12
                @test Array(dest)[1, 1] == 0.0 + 0.0im
            end
        end

        @testset "spectral derivative multiply (skew/grad core)" begin
            # verbatim from _apply_lazy_fourier_diff! (lazy_rhs.jl):
            #   data .*= reshape(deriv_mult, mult_shape)
            # ∂/∂x sin(3x) = 3 cos(3x) via full FFT along axis 1.
            N = 16; L = 2π; k0 = 2π / L; m = 3
            x = (0:N-1) .* (L / N)
            f = [sin(m * x[i]) for i in 1:N, j in 1:N]
            fhat = fft(f, 1)
            kfft = k0 .* Float64.([0:(N ÷ 2 - 1); -(N ÷ 2):-1])
            deriv_mult = ComplexF64.(im .* kfft)           # (ik)^1

            dhat = _JLA2D(copy(fhat))
            dhat .*= reshape(_JLA2D(deriv_mult), N, 1)     # device ×(ik) along axis 1
            got = real.(ifft(Array(dhat), 1))
            @test got ≈ [m * cos(m * x[i]) for i in 1:N, j in 1:N] rtol=1e-10
        end

        @testset "device reductions (energy/enstrophy diagnostics)" begin
            ux = randn(16, 16); uy = randn(16, 16); q = randn(16, 16)
            E_ref = 0.5 * mean(@. ux^2 + uy^2)
            Z_ref = 0.5 * mean(q .^ 2)
            uxd = _JLA2D(ux); uyd = _JLA2D(uy); qd = _JLA2D(q)
            @test 0.5 * mean(@. uxd^2 + uyd^2) ≈ E_ref rtol=1e-12
            @test 0.5 * mean(qd .^ 2) ≈ Z_ref rtol=1e-12
        end

        GPUArrays.allowscalar(true)   # restore for later tests in the process
    end
end
