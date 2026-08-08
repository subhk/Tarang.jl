"""
Value tests for the GPU DCT-I / Chebyshev-derivative kernels — WITHOUT a GPU.

The kernels in `ext/cuda/cheb_deriv.jl` are KernelAbstractions kernels, so the
very same kernel objects the CUDA path launches on a `CUDABackend()` also run on
`KernelAbstractions.CPU()` over plain `Array`s (CUDA.jl loads fine on machines
with no NVIDIA GPU). Only the cuFFT `mul!` is device-specific; FFTW's
`plan_rfft` has identical semantics, so composing the REAL kernels with FFTW
reproduces the device pipeline value-for-value.

These pins matter because the kernels were rewritten from one-thread-per-COLUMN
(with a serial inner loop) to one-thread-per-ELEMENT, four pairs were fused
(reverse+extension, prescale+extension, extract+normalize, extract+finalize),
and the complex path switched from `real./imag./complex.` splits to a packed
re/im batch. A wrong index in any of them is a silent value bug on hardware
this CI does not have.
"""

using Test
using Tarang
using KernelAbstractions
using FFTW
using LinearAlgebra: mul!

const _CUDA_LOADED = try
    @eval using CUDA
    true
catch err
    @info "CUDA.jl unavailable; skipping GPU DCT-I kernel value tests" err
    false
end

# Independent statement of Tarang's Chebyshev DCT-I convention
# (transform_chebyshev.jl): forward = REDFT00 · 1/(N-1) · ½-endpoints · odd-flip.
function _ref_cheb_forward(x::Vector{Float64})
    n = length(x)
    c = FFTW.r2r(x, FFTW.REDFT00) ./ (n - 1)
    c[1] /= 2
    c[end] /= 2
    return c .* [iseven(k) ? 1.0 : -1.0 for k in 0:n-1]
end

# Backward = odd-flip · ×2-endpoints · REDFT00 · ½ (the inverse of the above).
function _ref_cheb_backward(c::Vector{Float64})
    y = c .* [iseven(k) ? 1.0 : -1.0 for k in 0:n_ref_len(c)-1]
    y[1] *= 2
    y[end] *= 2
    return FFTW.r2r(y, FFTW.REDFT00) ./ 2
end
n_ref_len(c) = length(c)

@testset "GPU DCT-I kernels on the KA CPU backend" begin
    if !_CUDA_LOADED
        @test_skip "CUDA.jl not loadable in this environment"
    else
        ext = Base.get_extension(Tarang, :TarangCUDAExt)
        @test ext !== nothing
        backend = KernelAbstractions.CPU()

        runk(kern, args...; ndrange) = begin
            kern(backend, 64)(args...; ndrange=ndrange)
            KernelAbstractions.synchronize(backend)
        end

        # Compose the actual ext kernels with FFTW standing in for cuFFT —
        # mirrors `_dct1_batch_mat!` exactly (in-place capable: out may === in).
        function emu_dct1!(out::Matrix{Float64}, mat::Matrix{Float64}, direction::Symbol)
            n, batch = size(mat)
            M = 2 * (n - 1)
            work_ext = zeros(M, batch)
            work_cx = zeros(ComplexF64, div(M, 2) + 1, batch)
            @assert size(work_cx, 1) == n
            p = FFTW.plan_rfft(work_ext, 1)
            if direction === :forward
                runk(ext._dct1_reverse_ext_kernel!, work_ext, mat, n, batch; ndrange=(M, batch))
                mul!(work_cx, p, work_ext)
                runk(ext._dct1_extract_normalize_kernel!, out, work_cx, n, batch,
                     1.0 / (n - 1); ndrange=(n, batch))
            else
                runk(ext._dct1_prescale_ext_kernel!, work_ext, mat, n, batch; ndrange=(M, batch))
                mul!(work_cx, p, work_ext)
                runk(ext._dct1_extract_finalize_kernel!, out, work_cx, n, batch; ndrange=(n, batch))
            end
            return out
        end
        emu_dct1(mat, direction) = emu_dct1!(similar(mat), mat, direction)

        @testset "forward DCT-I matches the CPU convention (n=$n)" for n in (9, 16)
            batch = 5
            mat = randn(n, batch)
            got = emu_dct1(mat, :forward)
            for j in 1:batch
                @test got[:, j] ≈ _ref_cheb_forward(mat[:, j]) atol = 1e-13
            end
        end

        @testset "backward inverts forward, and matches the convention (n=$n)" for n in (9, 16)
            batch = 4
            mat = randn(n, batch)
            fwd = emu_dct1(mat, :forward)
            @test emu_dct1(fwd, :backward) ≈ mat atol = 1e-12
            for j in 1:batch
                @test emu_dct1!(similar(mat), fwd, :backward)[:, j] ≈
                      _ref_cheb_backward(fwd[:, j]) atol = 1e-12
            end
        end

        @testset "in-place: out === in is safe" begin
            n, batch = 12, 3
            mat = randn(n, batch)
            expect = emu_dct1(mat, :forward)
            work = copy(mat)
            emu_dct1!(work, work, :forward)
            @test work ≈ expect atol = 1e-13
        end

        @testset "plain extension kernel (derivative step 5) still mirrors correctly" begin
            n, batch = 9, 3
            M = 2 * (n - 1)
            inp = randn(n, batch)
            work = fill(NaN, M, batch)
            runk(ext._dct1_ext_kernel!, work, inp, n, batch; ndrange=(M, batch))
            for j in 1:batch
                @test work[1:n, j] == inp[:, j]
                for k in 1:n-2
                    @test work[n+k, j] == inp[n-k, j]
                end
            end
        end

        @testset "complex pack/unpack kernels" begin
            n, batch = 8, 6
            cx = randn(ComplexF64, n, batch)
            packed = fill(NaN, n, 2batch)
            runk(ext._cheb_pack_reim_kernel!, packed, cx, n, batch; ndrange=(n, 2batch))
            @test packed[:, 1:batch] == real.(cx)
            @test packed[:, batch+1:2batch] == imag.(cx)
            back = zeros(ComplexF64, n, batch)
            runk(ext._cheb_unpack_reim_kernel!, back, packed, n, batch; ndrange=(n, batch))
            @test back == cx
        end

        # Full derivative pipeline — the ext kernels composed exactly as
        # `_apply_gpu_cheb_deriv_1!` does, FFTW in place of cuFFT.
        function emu_cheb_deriv_1!(out::Matrix{Float64}, mat::Matrix{Float64}, scale::Float64)
            n, batch = size(mat)
            M = 2 * (n - 1)
            work_ext = zeros(M, batch)
            work_cx = zeros(ComplexF64, n, batch)
            work_real = zeros(n, batch)
            work_deriv = zeros(n, batch)
            p = FFTW.plan_rfft(work_ext, 1)
            runk(ext._dct1_reverse_ext_kernel!, work_ext, mat, n, batch; ndrange=(M, batch))
            mul!(work_cx, p, work_ext)
            runk(ext._extract_real_kernel!, work_real, work_cx, n, batch; ndrange=(n, batch))
            runk(ext._cheb_coeff_to_deriv_kernel!, work_deriv, work_real, n, batch,
                 1.0 / (n - 1), scale; ndrange=batch)
            runk(ext._dct1_ext_kernel!, work_ext, work_deriv, n, batch; ndrange=(M, batch))
            mul!(work_cx, p, work_ext)
            runk(ext._dct1_extract_finalize_kernel!, out, work_cx, n, batch; ndrange=(n, batch))
            return out
        end

        @testset "derivative pipeline matches chebyshev_derivative_1d! (n=$n)" for n in (9, 17)
            batch = 4
            scale = 2.0 / 3.7
            mat = randn(n, batch)
            got = emu_cheb_deriv_1!(similar(mat), mat, scale)
            for j in 1:batch
                expect = similar(mat[:, j])
                Tarang.chebyshev_derivative_1d!(expect, mat[:, j], scale)
                @test got[:, j] ≈ expect atol = 1e-11
            end
            # order 2 == applying the 1-pass twice (what _apply_gpu_cheb_deriv_nth! does)
            got2 = emu_cheb_deriv_1!(similar(mat), got, scale)
            for j in 1:batch
                e1 = similar(mat[:, j]); Tarang.chebyshev_derivative_1d!(e1, mat[:, j], scale)
                e2 = similar(e1);        Tarang.chebyshev_derivative_1d!(e2, e1, scale)
                @test got2[:, j] ≈ e2 atol = 1e-9
            end
        end

        # End-to-end: emulate the REWRITTEN mixed 2D RealFourier×Chebyshev
        # forward driver (rfft on dim 1, packed-complex DCT-I on dim 2, last
        # stage writing the coefficient buffer) and compare against the CPU
        # transform chain, bit-for-bit at machine precision.
        @testset "2D RF×Cheb forward: emulated device driver == CPU chain" begin
            coords = CartesianCoordinates("x", "z")
            dist = Distributor(coords; mesh=(1,), dtype=Float64)
            xb = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
            zb = ChebyshevT(coords["z"]; size=9, bounds=(-1.0, 1.0))
            u = ScalarField(dist, "u", (xb, zb), Float64)
            g = Tarang.get_grid_data(u)
            g .= randn(size(g))
            grid = Array(copy(g))
            u.current_layout = :g
            Tarang.forward_transform!(u)
            cpu_coeffs = Array(copy(Tarang.get_coeff_data(u)))

            # Stage 1: rfft along dim 1 (unnormalized, like cuFFT)
            spec = FFTW.rfft(grid, 1)              # (9, 9) complex
            # Stage 2: packed-complex DCT-I along dim 2 via the ext kernels
            n2 = size(spec, 2)
            b2 = size(spec, 1)
            cperm = permutedims(spec, (2, 1))       # (n2, b2)
            packed = zeros(n2, 2b2)
            runk(ext._cheb_pack_reim_kernel!, packed, cperm, n2, b2; ndrange=(n2, 2b2))
            emu_dct1!(packed, packed, :forward)
            runk(ext._cheb_unpack_reim_kernel!, cperm, packed, n2, b2; ndrange=(n2, b2))
            coeffs = permutedims(cperm, (2, 1))

            @test coeffs ≈ cpu_coeffs atol = 1e-12
        end
    end
end
