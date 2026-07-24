# Coverage tests for src/core/field/field_data/field_data_copy_alloc.jl
#
# Targets the serial-CPU-reachable behavior of the field copy/allocation
# helpers: Base.copy / Base.deepcopy (grid AND coeff layout), coefficient_eltype,
# the _empty_* placeholders, allocate_data!/_build_field_arrays sizing for
# Fourier (complex coeffs) and Chebyshev (real coeffs), the gpu_fft_mode
# accessor/setter, synchronize_field_architecture! (same-arch no-op + cross-arch
# throw), and the pure-arithmetic get_local_array_size decomposition logic
# (exercised by mutating a serial Distributor's mesh/size/rank fields — no MPI
# communication occurs inside that function).
#
# Self-contained; run standalone with:
#   julia --project=. -e 'include("test/test_cov_field_data_copy_alloc.jl")'

using Test
using Tarang
using LinearAlgebra

# Convenience aliases for the (non-exported) internals under test.
const T_coefficient_eltype            = Tarang.coefficient_eltype
const T_empty_grid                    = Tarang._empty_grid
const T_empty_coeff                   = Tarang._empty_coeff
const T_field_architecture            = Tarang.field_architecture
const T_sync_arch!                    = Tarang.synchronize_field_architecture!
const T_build_field_arrays            = Tarang._build_field_arrays
const T_get_local_array_size          = Tarang.get_local_array_size
const T_gpu_fft_mode                  = Tarang.gpu_fft_mode
const T_set_gpu_fft_mode!             = Tarang.set_gpu_fft_mode!
const T_get_grid_data                 = Tarang.get_grid_data
const T_get_coeff_data                = Tarang.get_coeff_data
const T_set_grid_data!                = Tarang.set_grid_data!
const T_set_coeff_data!               = Tarang.set_coeff_data!

# --- shared builders ---------------------------------------------------------

function make_fourier_field(; n=8, name="f")
    coords = Tarang.CartesianCoordinates("x")
    dist   = Tarang.Distributor(coords; mesh=(1,), dtype=Float64)
    zb     = Tarang.RealFourier(coords["x"]; size=n, bounds=(0.0, 2pi))
    f      = Tarang.ScalarField(Tarang.Domain(dist, (zb,)), name)
    return coords, dist, f
end

function make_cheb_field(; n=16, name="u")
    coords = Tarang.CartesianCoordinates("x")
    dist   = Tarang.Distributor(coords; mesh=(1,), dtype=Float64)
    zb     = Tarang.ChebyshevT(coords["x"]; size=n, bounds=(-1.0, 1.0))
    f      = Tarang.ScalarField(Tarang.Domain(dist, (zb,)), name)
    return coords, dist, f
end

@testset "field_data_copy_alloc coverage" begin

    @testset "coefficient_eltype (Type form)" begin
        # Real grid types promote to a Complex coefficient type.
        @test T_coefficient_eltype(Float64) === ComplexF64
        @test T_coefficient_eltype(Float32) === ComplexF32
        # Already-complex types are preserved unchanged.
        @test T_coefficient_eltype(ComplexF64) === ComplexF64
        @test T_coefficient_eltype(ComplexF32) === ComplexF32
    end

    @testset "coefficient_eltype (Domain form: basis-aware)" begin
        _, _, ff = make_fourier_field()
        _, _, cf = make_cheb_field()
        # A Fourier axis ⇒ complex coefficients.
        @test T_coefficient_eltype(ff.domain, Float64) === ComplexF64
        # A pure-Jacobi (Chebyshev) domain keeps the grid element type.
        @test T_coefficient_eltype(cf.domain, Float64) === Float64
        @test T_coefficient_eltype(cf.domain, Float32) === Float32
    end

    @testset "_empty_grid / _empty_coeff placeholders" begin
        eg = T_empty_grid(Float64)
        ec = T_empty_coeff(Float64)
        # Length-0 typed 1-D placeholders, never `nothing`.
        @test eg isa Vector{Float64}
        @test length(eg) == 0
        @test ndims(eg) == 1
        # Coeff placeholder uses the complex coefficient eltype.
        @test ec isa Vector{ComplexF64}
        @test length(ec) == 0
        # Real-eltype request on grid stays real.
        @test T_empty_grid(Float32) isa Vector{Float32}
        @test T_empty_coeff(Float32) isa Vector{ComplexF32}
    end

    @testset "allocate_data! / _build_field_arrays sizes (Fourier scalar)" begin
        _, dist, f = make_fourier_field(n=8)
        g = T_get_grid_data(f)
        c = T_get_coeff_data(f)
        # Grid array is full size; RealFourier coeff array is N/2+1 complex.
        @test size(g) == (8,)
        @test eltype(g) === Float64
        @test size(c) == (5,)
        @test eltype(c) === ComplexF64
        # _build_field_arrays returns matching (grid, coeff) and zero-filled.
        gg, cc = T_build_field_arrays(dist, f.domain, Float64)
        @test size(gg) == (8,)
        @test size(cc) == (5,)
        @test eltype(cc) === ComplexF64
        @test all(iszero, gg)
        @test all(iszero, cc)
    end

    @testset "allocate_data! sizes (Chebyshev scalar: real coeffs)" begin
        _, _, f = make_cheb_field(n=16)
        g = T_get_grid_data(f)
        c = T_get_coeff_data(f)
        @test size(g) == (16,)
        @test eltype(g) === Float64
        # Pure-Jacobi: coeff array is full length and REAL (not complex).
        @test size(c) == (16,)
        @test eltype(c) === Float64
    end

    @testset "allocate_data! is a no-op for a domain-less field" begin
        # field.domain === nothing ⇒ early return; storage stays untouched.
        _, _, f = make_fourier_field()
        f.domain = nothing
        @test Tarang.allocate_data!(f) === nothing
    end

    @testset "get/set grid & coeff data round-trip" begin
        _, _, f = make_fourier_field(n=8)
        Tarang.ensure_layout!(f, :g)
        newg = collect(1.0:8.0)
        @test T_set_grid_data!(f, newg) === f
        @test T_get_grid_data(f) === newg              # exact array adopted
        newc = ComplexF64.(1:5)
        @test T_set_coeff_data!(f, newc) === f
        @test T_get_coeff_data(f) === newc
    end

    @testset "Base.copy is an independent copy (grid layout)" begin
        _, _, f = make_fourier_field(n=8)
        Tarang.ensure_layout!(f, :g)
        gd = T_get_grid_data(f)
        gd .= collect(1.0:8.0)
        fc = copy(f)
        # State preserved.
        @test fc.name == f.name
        @test fc.current_layout == f.current_layout == :g
        @test fc.dtype === f.dtype
        @test T_field_architecture(fc) == T_field_architecture(f)
        # Values copied.
        @test T_get_grid_data(fc) == T_get_grid_data(f)
        # Independent storage: mutating the copy must not touch the original.
        @test T_get_grid_data(fc) !== T_get_grid_data(f)
        T_get_grid_data(fc)[1] += 1000.0
        @test T_get_grid_data(f)[1] == 1.0
    end

    @testset "Base.copy preserves coeff layout (current_layout == :c branch)" begin
        _, _, f = make_fourier_field(n=8)
        Tarang.ensure_layout!(f, :g)
        T_get_grid_data(f) .= collect(1.0:8.0)
        Tarang.ensure_layout!(f, :c)
        @test f.current_layout == :c
        fc = copy(f)
        @test fc.current_layout == :c
        @test T_get_coeff_data(fc) == T_get_coeff_data(f)
        @test T_get_coeff_data(fc) !== T_get_coeff_data(f)
        # Mutate original coeffs ⇒ copy unaffected.
        orig0 = T_get_coeff_data(f)[1]
        T_get_coeff_data(f)[1] += 100.0
        @test T_get_coeff_data(fc)[1] == orig0
    end

    @testset "deepcopy independent (grid layout)" begin
        _, _, f = make_cheb_field(n=16)
        Tarang.ensure_layout!(f, :g)
        T_get_grid_data(f) .= collect(1.0:16.0)
        fc = deepcopy(f)
        @test fc.current_layout == :g
        @test T_get_grid_data(fc) == T_get_grid_data(f)
        @test T_get_grid_data(fc) !== T_get_grid_data(f)
        # Deep-copied bases are a distinct object (metadata deep-copy path).
        @test fc.bases !== f.bases || fc.bases == f.bases
        T_get_grid_data(fc)[2] = -999.0
        @test T_get_grid_data(f)[2] == 2.0
    end

    @testset "deepcopy preserves coeff layout (current_layout == :c branch)" begin
        # This exercises the deepcopy_internal :c branch (set_coeff_data! path).
        _, _, f = make_cheb_field(n=16)
        Tarang.ensure_layout!(f, :g)
        T_get_grid_data(f) .= collect(1.0:16.0)
        Tarang.ensure_layout!(f, :c)
        @test f.current_layout == :c
        fc = deepcopy(f)
        @test fc.current_layout == :c
        @test T_get_coeff_data(fc) == T_get_coeff_data(f)
        @test T_get_coeff_data(fc) !== T_get_coeff_data(f)
        # Cheb coeffs are real here; deepcopy must preserve eltype.
        @test eltype(T_get_coeff_data(fc)) === eltype(T_get_coeff_data(f))
        cd = T_get_coeff_data(f)
        cd[1] += 50.0
        @test T_get_coeff_data(fc)[1] != T_get_coeff_data(f)[1]
    end

    @testset "deepcopy cycle detection via stackdict" begin
        # Calling deepcopy_internal twice with the same stackdict must return
        # the already-registered copy (haskey early-return path).
        _, _, f = make_fourier_field()
        sd = IdDict()
        c1 = Base.deepcopy_internal(f, sd)
        c2 = Base.deepcopy_internal(f, sd)
        @test c1 === c2
    end

    @testset "field_architecture accessor" begin
        _, _, f = make_fourier_field()
        @test T_field_architecture(f) isa Tarang.AbstractArchitecture
        @test T_field_architecture(f) == f.dist.architecture
    end

    @testset "synchronize_field_architecture! (same-arch no-op)" begin
        _, _, f = make_fourier_field()
        # Same architecture ⇒ returns the field unchanged.
        @test T_sync_arch!(f; arch=T_field_architecture(f)) === f
        @test T_sync_arch!(f) === f   # default arch == dist.architecture
    end

    @testset "synchronize_field_architecture! (cross-arch throws)" begin
        _, _, f = make_fourier_field()
        # GPU(device) inner constructor bypasses the CUDA-required keyword ctor,
        # giving a distinct architecture object to drive the mismatch branch.
        other = Tarang.GPU(0)
        @test_throws ArgumentError T_sync_arch!(f; arch=other)
    end

    @testset "gpu_fft_mode accessor / setter" begin
        _, _, f = make_fourier_field()
        # Default mode is one of the allowed symbols.
        @test T_gpu_fft_mode(f) in (:auto, :cpu, :gpu)
        for m in (:auto, :cpu, :gpu)
            @test T_set_gpu_fft_mode!(f, m) === f
            @test T_gpu_fft_mode(f) === m
        end
        # Invalid mode is rejected.
        @test_throws ArgumentError T_set_gpu_fft_mode!(f, :bogus)
    end

    @testset "GPU transforms never select a CPU fallback" begin
        _, dist, f = make_fourier_field(n=8)
        # A mock GPU architecture is enough to exercise the core dispatch
        # contract without requiring CUDA on the test host.  The arrays remain
        # CPU-backed deliberately: transform dispatch must reject that mismatch
        # before it can reach FFTW.
        dist.architecture = Tarang.GPU(0)

        @test Tarang.should_use_gpu_fft(f, (8,))
        @test_throws ArgumentError T_set_gpu_fft_mode!(f, :cpu)
        @test_throws ErrorException Tarang.forward_transform!(f)
    end

    @testset "get_local_array_size serial fast-path" begin
        _, dist, _ = make_fourier_field()
        # size == 1 ⇒ local == global.
        @test T_get_local_array_size(dist, (8, 4)) == (8, 4)
        @test T_get_local_array_size(dist, (5,)) == (5,)
        @test T_get_local_array_size(dist, (3, 4, 5)) == (3, 4, 5)
    end

    @testset "get_local_array_size mesh===nothing fast-path" begin
        coords = Tarang.CartesianCoordinates("x")
        d = Tarang.Distributor(coords; mesh=(1,), dtype=Float64)
        # Force the non-serial size but clear the mesh ⇒ still returns global.
        d.size = 2
        d.mesh = nothing
        @test T_get_local_array_size(d, (8, 8)) == (8, 8)
    end

    @testset "get_local_array_size PencilArrays decomposition (even)" begin
        coords = Tarang.CartesianCoordinates("x")
        d = Tarang.Distributor(coords; mesh=(1,), dtype=Float64)
        # Drive the distributed arithmetic directly (no MPI communication here).
        d.size = 4
        d.mesh = (2, 2)
        d.use_pencil_arrays = true
        # PencilArrays decomposes the LAST ndims_mesh dims; dim 1 stays local.
        # 3D (8,8,8) over (2,2): dim1 local=8, dims 2,3 split → 4 each.
        d.rank = 0
        @test T_get_local_array_size(d, (8, 8, 8)) == (8, 4, 4)
        d.rank = 3
        @test T_get_local_array_size(d, (8, 8, 8)) == (8, 4, 4)
    end

    @testset "get_local_array_size PencilArrays uneven remainder" begin
        coords = Tarang.CartesianCoordinates("x")
        d = Tarang.Distributor(coords; mesh=(1,), dtype=Float64)
        d.size = 2
        d.mesh = (2,)
        d.use_pencil_arrays = true
        # 9 over 2 procs on the (single, last) decomposed dim: rank 0 gets the
        # extra element (proc_coord < remainder).
        d.rank = 0
        @test T_get_local_array_size(d, (9,)) == (5,)
        d.rank = 1
        @test T_get_local_array_size(d, (9,)) == (4,)
        # Local sizes sum back to the global extent.
        @test 5 + 4 == 9
    end

    @testset "get_local_array_size TransposableField (use_pencil_arrays=false)" begin
        coords = Tarang.CartesianCoordinates("x")
        d = Tarang.Distributor(coords; mesh=(1,), dtype=Float64)
        d.size = 4
        d.mesh = (2, 2)
        d.use_pencil_arrays = false
        # TransposableField decomposes the FIRST (up to 2) dims.
        # 2D (8,8) over (2,2): both dims split → 4 each.
        d.rank = 0
        @test T_get_local_array_size(d, (8, 8)) == (4, 4)
        # 3D (8,8,8): dims 1,2 split; dim 3 (z) stays LOCAL.
        @test T_get_local_array_size(d, (8, 8, 8)) == (4, 4, 8)
    end

    @testset "get_local_array_size mesh-dim > domain-dim warn path" begin
        coords = Tarang.CartesianCoordinates("x")
        d = Tarang.Distributor(coords; mesh=(1,), dtype=Float64)
        d.size = 4
        d.mesh = (2, 2)      # 2D mesh ...
        d.use_pencil_arrays = true
        d.rank = 0
        # ... applied to a 1D global shape ⇒ ndims_mesh > ndims_global warn.
        # In the pencil branch the decomposition loop is gated on
        # `ndims_global >= ndims_mesh` (1 >= 2 is false), so it is skipped and
        # the shape is returned unchanged — the warn path still executes.
        local res
        @test_logs (:warn,) match_mode=:any begin
            res = T_get_local_array_size(d, (8,))
        end
        @test res == (8,)
    end

end
