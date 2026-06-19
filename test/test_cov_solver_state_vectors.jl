using Test
using Tarang
using LinearAlgebra

# Coverage-focused tests for src/core/solvers/solver_state_vectors.jl
#
# This module owns the field <-> flat-solver-vector conversion boundary
# (pack/unpack). All functions live in the Tarang module namespace but are not
# exported, so they are referenced via the `Tarang.` prefix.
#
# Every assertion here checks a real invariant: round-trips (pack then unpack
# reproduces field data), known sizes, error/warning conditions, and the
# layout/eltype bookkeeping the conversion path is responsible for.

# ---------------------------------------------------------------------------
# Helpers to build serial CPU fields.
# ---------------------------------------------------------------------------

function cheb_field(name="f"; size=16, dtype=Float64)
    coords = CartesianCoordinates("x")
    dist = Distributor(coords; mesh=(1,), dtype=dtype)
    zb = Tarang.ChebyshevT(coords["x"]; size=size, bounds=(-1.0, 1.0))
    f = ScalarField(Domain(dist, (zb,)), name)
    return f, dist
end

function realfourier_field(name="f"; size=16, dtype=Float64)
    coords = CartesianCoordinates("x")
    dist = Distributor(coords; mesh=(1,), dtype=dtype)
    xb = Tarang.RealFourier(coords["x"]; size=size, bounds=(0.0, 2π))
    f = ScalarField(Domain(dist, (xb,)), name)
    return f, dist
end

function complexfourier_field(name="f"; size=16)
    coords = CartesianCoordinates("x")
    dist = Distributor(coords; mesh=(1,), dtype=ComplexF64)
    xb = Tarang.ComplexFourier(coords["x"]; size=size, bounds=(0.0, 2π))
    f = ScalarField(Domain(dist, (xb,)), name)
    return f, dist
end

# A 0-D / no-bases field (empty bases tuple) — the tau-style sentinel field.
function empty_field(name="tau")
    coords = CartesianCoordinates("x")
    dist = Distributor(coords; mesh=(1,), dtype=Float64)
    f = ScalarField(dist, name)   # no bases
    return f, dist
end

# Duck-typed basis stand-ins for the get_basis_size fallback branches.
# These have no `.meta` field, so they fall through to .shape / .size / .N /
# default. Defined at top level (struct defs are illegal in local scope).
struct _ShapeTupleBasis; shape::Tuple{Int,Int}; end
struct _ShapeScalarBasis; shape::Int; end
struct _SizeBasis; size::Int; end
struct _NBasis; N::Int; end
struct _UnknownBasis; foo::Int; end

@testset "solver_state_vectors coverage" begin

    @testset "compute_field_vector_size: empty bases" begin
        f, _ = empty_field()
        # 0-D field has no spatial DOF: returns the length-1 sentinel.
        @test Tarang.compute_field_vector_size(f) == 1
    end

    @testset "compute_field_vector_size: coeff-data path" begin
        f, _ = cheb_field(; size=16)
        ensure_layout!(f, :c)
        @test Tarang.compute_field_vector_size(f) == length(get_coeff_data(f))
        @test Tarang.compute_field_vector_size(f) == 16
    end

    @testset "compute_field_vector_size: RealFourier half-spectrum length" begin
        # A coeff-backed RealFourier field stores the half-spectrum, so the
        # vector length is div(N,2)+1 (this is what the coeff-data path returns).
        f, _ = realfourier_field(; size=16)
        ensure_layout!(f, :c)
        @test Tarang.compute_field_vector_size(f) == length(get_coeff_data(f))
        @test Tarang.compute_field_vector_size(f) == (div(16, 2) + 1)
    end

    @testset "get_basis_size: .meta.size path (real bases)" begin
        coords = CartesianCoordinates("x")
        cheb = Tarang.ChebyshevT(coords["x"]; size=24, bounds=(-1.0, 1.0))
        @test Tarang.get_basis_size(cheb) == 24
        fourier = Tarang.RealFourier(coords["x"]; size=32, bounds=(0.0, 2π))
        @test Tarang.get_basis_size(fourier) == 32
    end

    @testset "get_basis_size: fallback branches via duck-typed structs" begin
        @test Tarang.get_basis_size(_ShapeTupleBasis((4, 5))) == 20   # .shape Tuple
        @test Tarang.get_basis_size(_ShapeScalarBasis(7)) == 7        # .shape scalar
        @test Tarang.get_basis_size(_SizeBasis(11)) == 11             # .size
        @test Tarang.get_basis_size(_NBasis(13)) == 13                # .N
        # default branch -> 64 (warns)
        local sz
        @test_logs (:warn,) (sz = Tarang.get_basis_size(_UnknownBasis(1)))
        @test sz == 64
    end

    @testset "_state_vector_transport_mode" begin
        f, _ = cheb_field()
        @test Tarang._state_vector_transport_mode([f]) === :local
        @test Tarang._state_vector_transport_mode(ScalarField[]) === :empty
    end

    @testset "_fields_vector_size sums field sizes" begin
        f1, _ = cheb_field("a"; size=16)
        f2, _ = cheb_field("b"; size=8)
        ensure_layout!(f1, :c); ensure_layout!(f2, :c)
        @test Tarang._fields_vector_size([f1, f2]) == 16 + 8
    end

    @testset "_ensure_coeff_layout! converts to :c and returns arch" begin
        f, dist = cheb_field()
        ensure_layout!(f, :g)
        arch = Tarang._ensure_coeff_layout!([f])
        @test f.current_layout == :c
        @test arch === dist.architecture
        @test Tarang._ensure_coeff_layout!(ScalarField[]) === nothing
    end

    @testset "fields_to_vector round-trip (Chebyshev)" begin
        f, _ = cheb_field(; size=16)
        ensure_layout!(f, :c)
        cd = get_coeff_data(f)
        cd .= ComplexF64.(1:16)            # write known coeffs
        v = Tarang.fields_to_vector([f])
        @test length(v) == 16
        @test v == ComplexF64.(1:16)

        # Unpack into a fresh state and confirm the data round-trips.
        new_state = Tarang.vector_to_fields(v, [f])
        @test length(new_state) == 1
        rd = real.(vec(get_coeff_data(new_state[1])))  # cheb coeffs are real-typed
        @test rd ≈ Float64.(1:16)
    end

    @testset "fields_to_vector! in-place round-trip (multi-field)" begin
        f1, _ = cheb_field("a"; size=16)
        f2, _ = cheb_field("b"; size=8)
        ensure_layout!(f1, :c); ensure_layout!(f2, :c)
        get_coeff_data(f1) .= ComplexF64.(1:16)
        get_coeff_data(f2) .= ComplexF64.(101:108)

        total = Tarang.compute_field_vector_size(f1) + Tarang.compute_field_vector_size(f2)
        v = Vector{ComplexF64}(undef, total)
        ret = Tarang.fields_to_vector!(v, [f1, f2])
        @test ret === v
        @test v[1:16] == ComplexF64.(1:16)
        @test v[17:24] == ComplexF64.(101:108)
    end

    @testset "fields_to_vector! DimensionMismatch on wrong-size vector" begin
        f, _ = cheb_field(; size=16)
        ensure_layout!(f, :c)
        bad = Vector{ComplexF64}(undef, 5)
        @test_throws DimensionMismatch Tarang.fields_to_vector!(bad, [f])
    end

    @testset "fields_to_vector! empty fields requires empty vector" begin
        empty_v = Vector{ComplexF64}()
        @test Tarang.fields_to_vector!(empty_v, ScalarField[]) === empty_v
        nonempty = Vector{ComplexF64}(undef, 3)
        @test_throws ArgumentError Tarang.fields_to_vector!(nonempty, ScalarField[])
    end

    @testset "fields_to_vector empty -> empty vector" begin
        v = Tarang.fields_to_vector(ScalarField[])
        @test v isa Vector{ComplexF64}
        @test isempty(v)
    end

    @testset "fields_to_vector with empty-bases (tau) field" begin
        # _copy_field_data_to_vector! isempty(field.bases) branch: writes a 0.
        tau, _ = empty_field()
        v = Tarang.fields_to_vector([tau])
        @test length(v) == 1
        @test v[1] == 0
    end

    @testset "_copy_field_data_to_vector!: coeff path writes data at offset" begin
        f, _ = cheb_field(; size=16)
        ensure_layout!(f, :c)
        get_coeff_data(f) .= ComplexF64.(1:16)
        v = fill(ComplexF64(99), 20)
        newoff = Tarang._copy_field_data_to_vector!(v, 3, f, 16)   # offset 3
        @test newoff == 3 + 16
        @test v[3:18] == ComplexF64.(1:16)
        @test v[1] == ComplexF64(99) && v[2] == ComplexF64(99)
    end

    @testset "_copy_field_data_to_vector!: size-mismatch error" begin
        f, _ = cheb_field(; size=16)
        ensure_layout!(f, :c)
        v = Vector{ComplexF64}(undef, 20)
        # Claimed field_size (10) disagrees with actual coeff length (16).
        @test_throws ErrorException Tarang._copy_field_data_to_vector!(v, 1, f, 10)
    end

    @testset "_copy_field_data_to_vector!: field_size==0 early return" begin
        f, _ = cheb_field()
        @test Tarang._copy_field_data_to_vector!(Vector{ComplexF64}(undef, 4), 3, f, 0) == 3
    end

    @testset "_copy_field_data_to_vector!: vector-too-small error" begin
        f, _ = cheb_field(; size=16)
        ensure_layout!(f, :c)
        tiny = Vector{ComplexF64}(undef, 4)
        @test_throws ErrorException Tarang._copy_field_data_to_vector!(tiny, 1, f, 16)
    end

    @testset "extract_field_data_for_vector: coeff path" begin
        f, _ = cheb_field(; size=16)
        ensure_layout!(f, :c)
        get_coeff_data(f) .= ComplexF64.(1:16)
        d = Tarang.extract_field_data_for_vector(f)
        @test length(d) == 16
        @test real.(d) ≈ Float64.(1:16)
    end

    @testset "extract_field_data_for_vector: empty-bases path" begin
        tau, _ = empty_field()
        d = Tarang.extract_field_data_for_vector(tau)
        @test d == zeros(ComplexF64, 1)
    end

    @testset "extract_field_data_for_vector: matches fields_to_vector packing" begin
        # extract_field_data_for_vector forces :c then returns vec(coeff). It must
        # agree with the per-field slice produced by the gather path.
        f, _ = cheb_field(; size=16)
        ensure_layout!(f, :c)
        get_coeff_data(f) .= ComplexF64.(1:16)
        d = Tarang.extract_field_data_for_vector(f)
        @test d == Tarang.fields_to_vector([f])
    end

    @testset "set_field_data_from_vector!: empty-bases is a no-op" begin
        tau, _ = empty_field()
        @test Tarang.set_field_data_from_vector!(tau, ComplexF64[1.0]) === nothing
        # layout must not be flipped to :c by the no-op.
        @test tau.current_layout == :g
    end

    @testset "set_field_data_from_vector!: exact-size coeff write (real field)" begin
        f, _ = cheb_field(; size=16)
        ensure_layout!(f, :c)
        data = ComplexF64.(1:16)
        Tarang.set_field_data_from_vector!(f, data)
        @test f.current_layout == :c
        @test real.(vec(get_coeff_data(f))) ≈ Float64.(1:16)
    end

    @testset "set_field_data_from_vector!: complex field same-eltype path" begin
        f, _ = complexfourier_field(; size=16)
        ensure_layout!(f, :c)
        n = length(get_coeff_data(f))
        data = ComplexF64.((1:n) .+ im .* (1:n))
        Tarang.set_field_data_from_vector!(f, data)
        @test vec(get_coeff_data(f)) ≈ data
        @test f.current_layout == :c
    end

    @testset "set_field_data_from_vector!: short data is zero-padded" begin
        f, _ = cheb_field(; size=16)
        ensure_layout!(f, :c)
        short = ComplexF64.(1:4)            # < 16
        Tarang.set_field_data_from_vector!(f, short)
        cd = real.(vec(get_coeff_data(f)))
        @test cd[1:4] ≈ Float64.(1:4)
        @test all(iszero, cd[5:16])
    end

    @testset "set_field_data_from_vector!: long data is truncated" begin
        f, _ = cheb_field(; size=16)
        ensure_layout!(f, :c)
        long = ComplexF64.(1:32)           # > 16
        Tarang.set_field_data_from_vector!(f, long)
        cd = real.(vec(get_coeff_data(f)))
        @test cd ≈ Float64.(1:16)
    end

    @testset "set_field_data_from_vector!: imaginary part discarded for real field (warns)" begin
        f, _ = cheb_field(; size=16)
        ensure_layout!(f, :c)
        data = ComplexF64.(1:16) .+ im       # nonzero imag into a real-typed field
        @test_logs (:warn,) Tarang.set_field_data_from_vector!(f, data)
        cd = real.(vec(get_coeff_data(f)))
        @test cd ≈ Float64.(1:16)
    end

    @testset "set_field_data_from_vector!: convert-eltype path (real input into complex field)" begin
        f, _ = complexfourier_field(; size=16)
        ensure_layout!(f, :c)
        n = length(get_coeff_data(f))
        realdata = Float64.(1:n)             # eltype Float64, target ComplexF64
        Tarang.set_field_data_from_vector!(f, realdata)
        @test vec(get_coeff_data(f)) ≈ ComplexF64.(1:n)
    end

    @testset "copy_solution_to_fields! round-trip (multi-field)" begin
        f1, _ = cheb_field("a"; size=16)
        f2, _ = cheb_field("b"; size=8)
        ensure_layout!(f1, :c); ensure_layout!(f2, :c)
        sol = ComplexF64.(1:24)
        Tarang.copy_solution_to_fields!([f1, f2], sol)
        @test real.(vec(get_coeff_data(f1))) ≈ Float64.(1:16)
        @test real.(vec(get_coeff_data(f2))) ≈ Float64.(17:24)
        @test f1.current_layout == :c
        @test f2.current_layout == :c
    end

    @testset "copy_solution_to_fields!: empty fields is a no-op" begin
        @test Tarang.copy_solution_to_fields!(ScalarField[], ComplexF64[]) === nothing
    end

    @testset "_solution_vector_start_offset is 1 for local mode" begin
        f, _ = cheb_field()
        @test Tarang._solution_vector_start_offset([f], :local) == 1
    end

    @testset "vector_to_fields builds new state matching template" begin
        f, _ = cheb_field(; size=16)
        ensure_layout!(f, :c)
        v = ComplexF64.(1:16)
        new_state = Tarang.vector_to_fields(v, [f])
        @test length(new_state) == 1
        @test new_state[1].name == f.name
        @test real.(vec(get_coeff_data(new_state[1]))) ≈ Float64.(1:16)
        # New fields must be distinct objects from the template.
        @test new_state[1] !== f
    end

    @testset "vector_to_fields! in-place round-trip" begin
        f, _ = cheb_field(; size=16)
        ensure_layout!(f, :c)
        out, _ = cheb_field("out"; size=16)
        ensure_layout!(out, :c)
        v = ComplexF64.(1:16)
        ret = Tarang.vector_to_fields!([out], v, [f])
        @test ret[1] === out
        @test out.current_layout == :c
        @test real.(vec(get_coeff_data(out))) ≈ Float64.(1:16)
    end

    @testset "vector_to_fields! from a :g-layout output (coeff buffer present)" begin
        # Output left in :g layout: the in-place write must still land in the
        # coeff buffer and flip the layout to :c.
        f, _ = cheb_field(; size=16)
        ensure_layout!(f, :c)
        out, _ = cheb_field("out"; size=16)
        ensure_layout!(out, :g)
        v = ComplexF64.(1:16)
        Tarang.vector_to_fields!([out], v, [f])
        cd = get_coeff_data(out)
        @test cd !== nothing
        @test real.(vec(cd)) ≈ Float64.(1:16)
        @test out.current_layout == :c
    end

    @testset "full pack/unpack round-trip preserves data (RealFourier)" begin
        f, _ = realfourier_field(; size=16)
        ensure_layout!(f, :c)
        cd = get_coeff_data(f)
        original = copy(cd)
        cd .= ComplexF64.(reshape(1:length(cd), size(cd))) .+ im
        v = Tarang.fields_to_vector([f])
        # Round-trip back into a fresh field.
        new_state = Tarang.vector_to_fields(v, [f])
        @test vec(get_coeff_data(new_state[1])) ≈ vec(get_coeff_data(f))
        @test length(v) == length(cd)
    end

end
