"""
Value tests for the GPU element-wise / fused / spectral-padding kernels — WITHOUT
a GPU.

Same route as `test_gpu_dct1_kernels_cpu.jl` and `test_gpu_transpose_kernels_cpu.jl`:
the kernels in `ext/cuda/kernels.jl` are KernelAbstractions kernels, so the very
same kernel objects the CUDA path launches on a `CUDABackend()` run on
`KernelAbstractions.CPU()` over plain `Array`s, and CUDA.jl loads on machines
with no NVIDIA GPU.

Why this file exists: an audit of the extension's export surface found that a
large majority of the names `TarangCUDAExt` exports were referenced by NO test
file at all — including every one of the element-wise launchers below — and that
the GPU CI which would have exercised the rest is inert (`.buildkite/pipeline.yml`
says so itself: "Until then this file is inert (no CI consumes it)"). So these
kernels had never executed anywhere. An off-by-one or a swapped operand in any of
them is a silent value bug: it produces a plausible number, not an error.

Coverage here is the KERNEL BODIES (index math and arithmetic), which is what the
CPU backend can reach. The `CuArray`-typed launcher wrappers (`gpu_add!` and
friends) and the cuFFT plumbing still need real hardware.
"""

using Test
using Tarang
using KernelAbstractions

const _CUDA_LOADED_K = try
    @eval using CUDA
    true
catch err
    @info "CUDA.jl unavailable; skipping GPU kernel value tests" err
    false
end

@testset "GPU element-wise / fused kernels on the KA CPU backend" begin
    if !_CUDA_LOADED_K
        @test_skip "CUDA.jl not loadable in this environment"
    else
        ext = Base.get_extension(Tarang, :TarangCUDAExt)
        @test ext !== nothing
        backend = KernelAbstractions.CPU()

        run!(kern, args...; ndrange) = begin
            kern(backend, 8)(args...; ndrange=ndrange)
            KernelAbstractions.synchronize(backend)
            nothing
        end

        n = 37                      # deliberately not a multiple of the workgroup
        a = rand(n); b = rand(n); d = rand(n)
        ca = rand(ComplexF64, n); cb = rand(ComplexF64, n)

        @testset "element-wise" begin
            c = zeros(n); run!(ext.add_kernel!, c, a, b; ndrange=n)
            @test c ≈ a .+ b

            c = zeros(n); run!(ext.sub_kernel!, c, a, b; ndrange=n)
            @test c ≈ a .- b

            c = zeros(n); run!(ext.mul_kernel!, c, a, b; ndrange=n)
            @test c ≈ a .* b

            c = zeros(n); run!(ext.scale_kernel!, c, a, 2.5; ndrange=n)
            @test c ≈ 2.5 .* a

            y = copy(b); run!(ext.axpy_kernel!, y, 3.0, a; ndrange=n)
            @test y ≈ 3.0 .* a .+ b

            c = zeros(n); run!(ext.linear_combination_kernel!, c, 2.0, a, -0.5, b; ndrange=n)
            @test c ≈ 2.0 .* a .- 0.5 .* b

            c = copy(d); run!(ext.multiply_add_kernel!, c, a, b; ndrange=n)
            @test c ≈ d .+ a .* b

            mask = Float64.(rand(Bool, n))
            v = copy(a); run!(ext.spectral_cutoff_kernel!, v, mask; ndrange=n)
            @test v ≈ a .* mask
        end

        @testset "fused timestepping / products" begin
            c = zeros(n); run!(ext.rk_stage_kernel!, c, a, b, 0.1, 0.75, 2.0; ndrange=n)
            @test c ≈ 0.75 .* a .+ 2.0 .* 0.1 .* b

            y = copy(b); run!(ext.axpby_inplace_kernel!, y, a, 0.25, 3.0; ndrange=n)
            @test y ≈ 0.25 .* b .+ 3.0 .* a

            c = zeros(n); run!(ext.fma_kernel!, c, a, b, d; ndrange=n)
            @test c ≈ a .* b .+ d

            c = zeros(n); run!(ext.scale_multiply_kernel!, c, a, b, 1.5; ndrange=n)
            @test c ≈ 1.5 .* a .* b

            mask = Float64.(rand(Bool, n))
            c = zeros(n); run!(ext.dealias_multiply_kernel!, c, a, b, mask; ndrange=n)
            @test c ≈ mask .* a .* b

            c = zeros(n); run!(ext.triple_product_kernel!, c, a, b, d; ndrange=n)
            @test c ≈ a .* b .* d

            # conj(a) * b, NOT a * conj(b): the two differ by a sign on the
            # imaginary part, and both "look right".
            cc = zeros(ComplexF64, n)
            run!(ext.conj_multiply_kernel!, cc, ca, cb; ndrange=n)
            @test cc ≈ conj.(ca) .* cb
            @test !(cc ≈ ca .* conj.(cb))

            r = zeros(n); run!(ext.squared_magnitude_kernel!, r, ca; ndrange=n)
            @test r ≈ abs2.(ca)
        end

        @testset "diagnostics" begin
            e = zeros(n); run!(ext.kinetic_energy_2d_kernel!, e, ca, cb; ndrange=n)
            @test e ≈ 0.5 .* (abs2.(ca) .+ abs2.(cb))

            cw = rand(ComplexF64, n)
            e = zeros(n); run!(ext.kinetic_energy_3d_kernel!, e, ca, cb, cw; ndrange=n)
            @test e ≈ 0.5 .* (abs2.(ca) .+ abs2.(cb) .+ abs2.(cw))

            r = zeros(n); run!(ext.grad_mag_sq_2d_kernel!, r, ca, cb; ndrange=n)
            @test r ≈ abs2.(ca) .+ abs2.(cb)

            k2 = rand(n) .* 10
            f = copy(a); run!(ext.viscous_damping_kernel!, f, k2, 0.05, 0.2; ndrange=n)
            @test f ≈ a .* exp.(-0.05 .* k2 .* 0.2)
        end

        @testset "spectral pad / truncate index map" begin
            # `_gpu_padded_idx` is the 3/2-rule dealiasing index map: positive
            # frequencies keep their index, negative frequencies move to the end
            # of the padded axis, and the gap stays zero. A wrong branch here
            # aliases energy into the wrong mode — a plausible wrong answer.
            N1, N2 = 8, 6
            M1, M2 = 12, 10
            spec = rand(ComplexF64, N1, N2)

            padded = zeros(ComplexF64, M1, M2)
            run!(ext.pad_spectral_2d_kernel!, padded, spec, N1, N2, M1, M2, true, true;
                 ndrange=(N1, N2))

            # Independent reference: build the same embedding with explicit slices.
            ref = zeros(ComplexF64, M1, M2)
            posrange(N) = 1:(N ÷ 2 + 1)
            negsrc(N) = (N ÷ 2 + 2):N
            negdst(N, M) = (M - (N - (N ÷ 2 + 2))):M
            ref[posrange(N1), posrange(N2)] = spec[posrange(N1), posrange(N2)]
            ref[negdst(N1, M1), posrange(N2)] = spec[negsrc(N1), posrange(N2)]
            ref[posrange(N1), negdst(N2, M2)] = spec[posrange(N1), negsrc(N2)]
            ref[negdst(N1, M1), negdst(N2, M2)] = spec[negsrc(N1), negsrc(N2)]
            @test padded ≈ ref
            @test count(!iszero, padded) == count(!iszero, spec)   # nothing dropped

            back = zeros(ComplexF64, N1, N2)
            run!(ext.truncate_spectral_2d_kernel!, back, padded, N1, N2, M1, M2, true, true;
                 ndrange=(N1, N2))
            @test back ≈ spec        # pad then truncate is the identity

            # 3-D, and with a NON-Fourier (Chebyshev-like) axis, where the map is
            # a plain leading copy rather than a two-sided split.
            N = (6, 4, 5); M = (10, 8, 5)
            spec3 = rand(ComplexF64, N...)
            padded3 = zeros(ComplexF64, M...)
            run!(ext.pad_spectral_3d_kernel!, padded3, spec3, N..., M..., true, true, false;
                 ndrange=N)
            back3 = zeros(ComplexF64, N...)
            run!(ext.truncate_spectral_3d_kernel!, back3, padded3, N..., M..., true, true, false;
                 ndrange=N)
            @test back3 ≈ spec3
            @test count(!iszero, padded3) == count(!iszero, spec3)
        end
    end
end

@testset "workgroup_size splits a multi-dimensional ndrange" begin
    # Regression guard for the occupancy bug: `workgroup_size` used to collapse
    # every ndrange to the scalar 256 via `prod`, and KernelAbstractions pads a
    # scalar workgroup with ONES — so a 2-D launch got a (256, 1) block. When the
    # first dimension is shorter than 256 (every practical Chebyshev axis: the
    # DCT-I kernels launch over `(2(n-1), batch)`) most threads in every block
    # were masked off.
    for dims in [(126, 65536), (62, 4096), (254, 1024), (64, 64), (512, 8), (32, 32, 32)]
        wg = Tarang._split_workgroup(256, dims)
        @test length(wg) == length(dims)
        @test prod(wg) <= 256                       # never exceeds the budget
        @test all(wg .>= 1)
        @test all(wg .<= dims)                      # never wider than the work
        # Every thread in every block lands on real work.
        blocks = map((n, w) -> cld(n, w), dims, wg)
        @test prod(blocks) * prod(wg) == prod(dims)
    end

    # Scalar ndranges keep the old scalar policy.
    @test Tarang.workgroup_size(Tarang.CPU(), 4096) == 64
    @test Tarang.workgroup_size(Tarang.CPU(), 10) == 10

    # Tuple ndranges now come back as tuples, one entry per dimension.
    @test Tarang.workgroup_size(Tarang.CPU(), (32, 32, 32)) isa Tuple
    @test length(Tarang.workgroup_size(Tarang.CPU(), (32, 32, 32))) == 3

    # Degenerate extents must not produce a zero-wide workgroup.
    @test all(Tarang._split_workgroup(256, (0, 8)) .>= 1)
    @test Tarang._split_workgroup(256, ()) == ()
end
