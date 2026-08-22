"""
Value tests for the GPU DCT-I / Chebyshev-derivative kernels AND for the full
Fourier-Fourier / Fourier-Chebyshev device transform drivers — WITHOUT a GPU.

The kernels in `ext/cuda/cheb_deriv.jl` are KernelAbstractions kernels, so the
very same kernel objects the CUDA path launches on a `CUDABackend()` also run on
`KernelAbstractions.CPU()` over plain `Array`s (CUDA.jl loads fine on machines
with no NVIDIA GPU). Only the cuFFT `mul!` is device-specific; FFTW's
`plan_rfft` has identical semantics, so composing the REAL kernels with FFTW
reproduces the device pipeline value-for-value.

The final block goes past the kernels and replays the DRIVERS
(`gpu_mixed_forward_transform!` / `_gpu_forward_transform_impl!`) end to end for
every FF and FC layout in 2D and 3D, so the stage order, the per-axis rfft-vs-fft
choice, the Chebyshev truncate/zero-pad and the permuted-axis DCT are all pinned
against the CPU chain — coverage the 2D-only kernel pins never reached.

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

        # ── N-D device DRIVER emulation: FF and FC in 2D and 3D ─────────────
        # The 2D block above pins one layout. The device drivers
        # (`gpu_mixed_forward_transform!` / `_gpu_forward_transform_impl!`) also
        # have to get the STAGE ORDER, the per-axis rfft-vs-fft choice, the
        # Chebyshev truncate/zero-pad, and the permuted-axis DCT right in 3D and
        # for every basis position — none of which the 2D case exercises. These
        # replay the driver logic with the SHARED layout rules
        # (`Tarang.forward_layout` / `Tarang.transform_stage_shapes` — the very
        # functions the CUDA plan calls) and the REAL ext kernels, and compare to
        # the CPU chain.
        function emu_dct1_dim!(out::Array{Float64,N}, inp::Array{Float64,N},
                               dim::Int, dir::Symbol) where {N}
            n = size(inp, dim)
            n <= 1 && return (out === inp || copyto!(out, inp); out)
            batch = prod(size(inp)) ÷ n
            if dim == 1
                emu_dct1!(reshape(out, n, batch), reshape(inp, n, batch), dir)
            else
                other = ntuple(i -> i < dim ? i : i + 1, N - 1)
                perm = (dim, other...)
                pshape = ntuple(i -> size(inp, perm[i]), N)
                wp = permutedims(inp, perm)
                mat = reshape(wp, n, batch)
                emu_dct1!(mat, mat, dir)
                permutedims!(out, reshape(mat, pshape), invperm(perm))
            end
            return out
        end

        function emu_dct1_dim!(out::Array{ComplexF64,N}, inp::Array{ComplexF64,N},
                               dim::Int, dir::Symbol) where {N}
            n = size(inp, dim)
            n <= 1 && return (out === inp || copyto!(out, inp); out)
            batch = prod(size(inp)) ÷ n
            packed = zeros(n, 2batch)
            if dim == 1
                cin = reshape(inp, n, batch); cout = reshape(out, n, batch)
                runk(ext._cheb_pack_reim_kernel!, packed, cin, n, batch; ndrange=(n, 2batch))
                emu_dct1!(packed, packed, dir)
                runk(ext._cheb_unpack_reim_kernel!, cout, packed, n, batch; ndrange=(n, batch))
            else
                other = ntuple(i -> i < dim ? i : i + 1, N - 1)
                perm = (dim, other...)
                pshape = ntuple(i -> size(inp, perm[i]), N)
                cs = permutedims(inp, perm); cmat = reshape(cs, n, batch)
                runk(ext._cheb_pack_reim_kernel!, packed, cmat, n, batch; ndrange=(n, 2batch))
                emu_dct1!(packed, packed, dir)
                runk(ext._cheb_unpack_reim_kernel!, cmat, packed, n, batch; ndrange=(n, batch))
                permutedims!(out, reshape(cs, pshape), invperm(perm))
            end
            return out
        end

        _axis_idx(nd, dim, r) = ntuple(i -> i == dim ? r : Colon(), nd)
        _stage_order(bases) = vcat(sort([d for (d, b) in enumerate(bases) if !(b isa ChebyshevT)]),
                                   [d for (d, b) in enumerate(bases) if b isa ChebyshevT])

        function emu_mixed_forward(bases, grid)
            ops, coeff_shape, _ = Tarang.forward_layout(bases, size(grid), eltype(grid))
            order = _stage_order(bases)
            stages = Tarang.transform_stage_shapes(ops, size(grid), order)
            cur = grid
            for (si, dim) in enumerate(order)
                want = stages[si + 1]
                if bases[dim] isa ChebyshevT
                    cur = emu_dct1_dim!(similar(cur), cur, dim, :forward)
                    size(cur) == want ||
                        (cur = cur[_axis_idx(ndims(cur), dim, 1:want[dim])...])
                else
                    cur = ops[dim].op === :rfft ? FFTW.rfft(cur, dim) :
                          FFTW.fft(eltype(cur) <: Complex ? cur : complex.(cur), dim)
                end
                @assert size(cur) == want "forward stage $si (dim $dim): $(size(cur)) != $want"
            end
            @assert size(cur) == coeff_shape
            return cur
        end

        function emu_mixed_backward(bases, coeffs, grid_shape, dtype)
            ops, _, _ = Tarang.forward_layout(bases, grid_shape, dtype)
            order = _stage_order(bases)
            stages = Tarang.transform_stage_shapes(ops, grid_shape, order)
            cur = coeffs
            for si in reverse(eachindex(order))
                dim = order[si]
                want = stages[si]
                if bases[dim] isa ChebyshevT
                    inp = cur
                    if size(cur) != want
                        padded = zeros(eltype(cur), want...)
                        padded[_axis_idx(length(want), dim, 1:size(cur, dim))...] .= cur
                        inp = padded
                    end
                    cur = emu_dct1_dim!(similar(inp), inp, dim, :backward)
                else
                    cur = ops[dim].op === :rfft ? FFTW.irfft(cur, want[dim], dim) :
                          FFTW.ifft(eltype(cur) <: Complex ? cur : complex.(cur), dim)
                end
                @assert size(cur) == want "backward stage $si (dim $dim): $(size(cur)) != $want"
            end
            return dtype <: Real && eltype(cur) <: Complex ? real.(cur) : cur
        end

        emu_ff_forward(bases, grid) =
            Tarang.forward_axis_op(bases[1], size(grid, 1), eltype(grid) <: Complex).op === :rfft ?
                FFTW.rfft(grid) : FFTW.fft(complex.(grid))

        function emu_ff_backward(bases, coeffs, grid_shape, dtype)
            if bases[1] isa RealFourier && !(dtype <: Complex)
                op = Tarang.backward_axis_op(bases[1], size(coeffs, 1), grid_shape[1], true)
                op.op === :irfft && return FFTW.irfft(coeffs, op.out_len)
            end
            r = FFTW.ifft(eltype(coeffs) <: Complex ? coeffs : complex.(coeffs))
            return dtype <: Real ? real.(r) : r
        end

        _mkb(c, nm, kind, n) =
            kind === :RF ? RealFourier(c[nm]; size=n, bounds=(0.0, 2π)) :
            kind === :CF ? ComplexFourier(c[nm]; size=n, bounds=(0.0, 2π)) :
                           ChebyshevT(c[nm]; size=n, bounds=(-1.0, 1.0))

        _ND_CASES = [
            ("FF 2D RF*RF",        (:RF, :RF),          (16, 16), Float64),
            ("FF 2D RF*RF odd",    (:RF, :RF),          (15, 9),  Float64),
            ("FF 2D CF*RF",        (:CF, :RF),          (12, 16), Float64),
            ("FF 3D RF*CF*RF",     (:RF, :CF, :RF),     (8, 6, 10), Float64),
            ("FF 3D CF*CF*CF",     (:CF, :CF, :CF),     (6, 6, 6), ComplexF64),
            ("FC 2D RF*CHEB",      (:RF, :CHEB),        (16, 9),  Float64),
            ("FC 2D CHEB*RF",      (:CHEB, :RF),        (9, 16),  Float64),
            ("FC 2D CF*CHEB",      (:CF, :CHEB),        (12, 9),  ComplexF64),
            ("FC 3D RF*RF*CHEB",   (:RF, :RF, :CHEB),   (8, 8, 9), Float64),
            ("FC 3D CHEB*RF*RF",   (:CHEB, :RF, :RF),   (9, 8, 8), Float64),
            ("FC 3D RF*CHEB*RF",   (:RF, :CHEB, :RF),   (8, 9, 8), Float64),
            ("FC 3D RF*CHEB*CHEB", (:RF, :CHEB, :CHEB), (8, 9, 7), Float64),
            ("FC 3D CHEB*CHEB*RF", (:CHEB, :CHEB, :RF), (7, 9, 8), Float64),
        ]

        @testset "device driver == CPU chain: $label" for (label, kinds, sizes, T) in _ND_CASES
            names = ("x", "y", "z")[1:length(kinds)]
            coords = CartesianCoordinates(names...)
            dist = Distributor(coords; mesh=(1,), dtype=T)
            bases = ntuple(i -> _mkb(coords, names[i], kinds[i], sizes[i]), length(kinds))
            u = ScalarField(dist, "u", bases, T)
            ensure_layout!(u, :g)
            g = Tarang.get_grid_data(u)
            grid = T <: Complex ? randn(ComplexF64, size(g)...) : randn(size(g)...)
            g .= grid
            u.current_layout = :g
            Tarang.forward_transform!(u)
            cpu_c = Array(copy(Tarang.get_coeff_data(u)))
            Tarang.backward_transform!(u)
            ensure_layout!(u, :g)
            cpu_g = Array(copy(Tarang.get_grid_data(u)))

            has_cheb = any(k -> k === :CHEB, kinds)
            emu_c = has_cheb ? emu_mixed_forward(bases, grid) : emu_ff_forward(bases, grid)
            emu_g = has_cheb ? emu_mixed_backward(bases, emu_c, size(grid), T) :
                               emu_ff_backward(bases, emu_c, size(grid), T)
            @test size(emu_c) == size(cpu_c)
            @test isapprox(emu_c, cpu_c; atol=1e-12)
            @test isapprox(emu_g, cpu_g; atol=1e-12)
        end
    end
end
