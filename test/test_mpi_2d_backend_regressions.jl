using Test
using MPI
using PencilArrays
using Tarang

MPI.Initialized() || MPI.Init()

const BACKEND_2D_COMM = MPI.COMM_WORLD
const BACKEND_2D_RANK = MPI.Comm_rank(BACKEND_2D_COMM)
const BACKEND_2D_NPROCS = MPI.Comm_size(BACKEND_2D_COMM)

if BACKEND_2D_NPROCS < 2
    BACKEND_2D_RANK == 0 &&
        @warn "2D backend regressions require at least two MPI ranks"
    exit(0)
end

_backend_local(data::PencilArrays.PencilArray) = parent(data)
_backend_local(data::AbstractArray) = data

_backend_global_size(data::PencilArrays.PencilArray) =
    Tuple(PencilArrays.size_global(data))
_backend_global_size(data::AbstractArray) = size(data)

function _backend_roundtrip_matches!(field; complex_values::Bool=false)
    ensure_layout!(field, :g)
    local_grid = _backend_local(get_grid_data(field))
    for I in CartesianIndices(local_grid)
        s = sum(Tuple(I)) + 3 * BACKEND_2D_RANK
        real_part = 0.125 * s - 0.375
        value = complex_values ? complex(real_part, 0.0625 * s + 0.25) : real_part
        local_grid[I] = value
    end
    original = copy(local_grid)

    forward_transform!(field)
    backward_transform!(field)

    restored = _backend_local(get_grid_data(field))
    tol = field.dtype === Float32 || field.dtype === ComplexF32 ? 2e-5 : 1e-11
    return field.current_layout == :g &&
           isapprox(restored, original; rtol=tol, atol=tol)
end

function _backend_snapshot(field)
    grid = get_grid_data(field)
    coeff = get_coeff_data(field)
    return (
        scales=field.scales,
        layout=field.current_layout,
        grid_type=typeof(grid),
        coeff_type=typeof(coeff),
        grid_size=size(grid),
        coeff_size=size(coeff),
        grid=copy(_backend_local(grid)),
        coeff=copy(_backend_local(coeff)),
    )
end

function _backend_fill_matching_grids!(field_a, field_b)
    ensure_layout!(field_a, :g)
    ensure_layout!(field_b, :g)
    grid_a = _backend_local(get_grid_data(field_a))
    grid_b = _backend_local(get_grid_data(field_b))
    @assert axes(grid_a) == axes(grid_b)
    for I in CartesianIndices(grid_a)
        s = sum(Tuple(I)) + 5 * BACKEND_2D_RANK
        grid_a[I] = 0.0625 * s^2 - 0.375 * s + 0.25
    end
    copyto!(grid_b, grid_a)
    return copy(grid_a)
end

function _backend_test_unchanged(field, before)
    grid = get_grid_data(field)
    coeff = get_coeff_data(field)
    @test field.scales == before.scales
    @test field.current_layout == before.layout
    @test typeof(grid) === before.grid_type
    @test typeof(coeff) === before.coeff_type
    @test size(grid) == before.grid_size
    @test size(coeff) == before.coeff_size
    @test isequal(_backend_local(grid), before.grid)
    @test isequal(_backend_local(coeff), before.coeff)
end

@testset "2D backend regressions (rank=$BACKEND_2D_RANK, np=$BACKEND_2D_NPROCS)" begin

@testset "2D MPI coefficient geometry (rank=$BACKEND_2D_RANK, np=$BACKEND_2D_NPROCS)" begin
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(
        coords;
        mesh=(BACKEND_2D_NPROCS,),
        dtype=Float64,
        architecture=CPU(),
    )
    xbasis = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
    ybasis = RealFourier(coords["y"]; size=10, bounds=(0.0, 2π))
    domain = Domain(dist, (xbasis, ybasis))
    field = ScalarField(domain, "u")

    coeff_data = get_coeff_data(field)
    @test coeff_data isa PencilArrays.PencilArray

    coeff_ranges = Tuple(PencilArrays.range_local(
        coeff_data,
        PencilArrays.LogicalOrder(),
    ))
    expected_global_shape = (
        length(Tarang.wavenumbers_rfft(xbasis)),
        length(Tarang.wavenumbers_fft(ybasis)),
    )

    @test Tuple(PencilArrays.size_global(coeff_data)) == expected_global_shape
    @test size(coeff_data) == Tuple(length.(coeff_ranges))
    @test local_shape(domain, :c) == Tuple(length.(coeff_ranges))

    viscosity = 0.25
    operator = SpectralLinearOperator(field, :laplacian; ν=viscosity)
    kx = Tarang.wavenumbers_rfft(xbasis)
    ky = Tarang.wavenumbers_fft(ybasis)
    expected_coefficients = [
        viscosity * (kx[i]^2 + ky[j]^2)
        for i in coeff_ranges[1], j in coeff_ranges[2]
    ]

    @test size(operator.coefficients) == size(coeff_data)
    @test operator.coefficients ≈ expected_coefficients
end

@testset "2D MPI plans are field-specific (rank=$BACKEND_2D_RANK, np=$BACKEND_2D_NPROCS)" begin
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(
        coords;
        mesh=(BACKEND_2D_NPROCS,),
        dtype=Float64,
        architecture=CPU(),
    )

    bases_a = (
        RealFourier(coords["x"]; size=8, bounds=(0.0, 2π)),
        RealFourier(coords["y"]; size=10, bounds=(0.0, 2π)),
    )
    bases_b = (
        RealFourier(coords["x"]; size=12, bounds=(0.0, 2π)),
        RealFourier(coords["y"]; size=14, bounds=(0.0, 2π)),
    )

    field_a = ScalarField(dist, "shape_a", bases_a, Float64)
    field_b = ScalarField(dist, "shape_b", bases_b, Float64)

    # Constructing or transforming either field must not invalidate the other.
    @test _backend_roundtrip_matches!(field_b)
    @test _backend_roundtrip_matches!(field_a)
    @test _backend_roundtrip_matches!(field_b)
end


@testset "2D MPI plans honor field dtype and domain (rank=$BACKEND_2D_RANK, np=$BACKEND_2D_NPROCS)" begin
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(
        coords;
        mesh=(BACKEND_2D_NPROCS,),
        dtype=Float64,
        architecture=CPU(),
    )
    bases = (
        RealFourier(coords["x"]; size=10, bounds=(0.0, 2π)),
        RealFourier(coords["y"]; size=12, bounds=(0.0, 2π)),
    )

    field32 = ScalarField(dist, "float32", bases, Float32)
    field64 = ScalarField(dist, "float64", bases, Float64)

    @test eltype(get_grid_data(field32)) === Float32
    @test eltype(get_coeff_data(field32)) === ComplexF32
    @test eltype(get_grid_data(field64)) === Float64
    @test eltype(get_coeff_data(field64)) === ComplexF64

    # Alternate the dtypes to catch a mutable distributor-wide active plan.
    @test _backend_roundtrip_matches!(field32)
    @test _backend_roundtrip_matches!(field64)
    @test _backend_roundtrip_matches!(field32)
end


@testset "2D MPI batched plans follow bundle lifecycle (rank=$BACKEND_2D_RANK, np=$BACKEND_2D_NPROCS)" begin
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(
        coords;
        mesh=(BACKEND_2D_NPROCS,),
        dtype=Float64,
        architecture=CPU(),
    )
    bases = (
        RealFourier(coords["x"]; size=12, bounds=(0.0, 2π)),
        RealFourier(coords["y"]; size=10, bounds=(0.0, 2π)),
    )
    field = ScalarField(dist, "batched_cache", bases, Float64)

    old_bundle = Tarang._field_transform_bundle(field)
    @test isempty(old_bundle.batched_plan_cache)
    old_entry = Tarang._get_batched_backward_plan!(field, 2)
    @test old_bundle.batched_plan_cache[2] === old_entry

    clear_distributor_cache!(dist)
    @test Tarang._field_transform_bundle(field) === old_bundle
    @test old_bundle.batched_plan_cache[2] === old_entry
    @test _backend_roundtrip_matches!(field)

    field_copy = deepcopy(field)
    @test Tarang._field_transform_bundle(field_copy) === old_bundle
    @test PencilArrays.pencil(get_grid_data(field_copy)) === old_bundle.pencil_fft_input
    @test PencilArrays.pencil(get_coeff_data(field_copy)) === old_bundle.pencil_fft_output
    @test _backend_roundtrip_matches!(field_copy)

    new_field = ScalarField(dist, "batched_cache_new", bases, Float64)
    new_bundle = Tarang._field_transform_bundle(new_field)
    @test new_bundle !== old_bundle
    @test isempty(new_bundle.batched_plan_cache)
    @test Tarang._get_batched_backward_plan!(new_field, 2) !== old_entry

    Tarang.clear_domain_cache!()
    @test Tarang._field_transform_bundle(field) === old_bundle
    @test Tarang._field_transform_bundle(new_field) === new_bundle
    @test _backend_roundtrip_matches!(field)
    @test _backend_roundtrip_matches!(new_field)
end


@testset "2D MPI complex fields use a full spectrum (rank=$BACKEND_2D_RANK, np=$BACKEND_2D_NPROCS)" begin
    n = 10
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(
        coords;
        mesh=(BACKEND_2D_NPROCS,),
        dtype=Float64,
        architecture=CPU(),
    )
    bases = (
        RealFourier(coords["x"]; size=n, bounds=(0.0, 2π)),
        RealFourier(coords["y"]; size=n, bounds=(0.0, 2π)),
    )
    field = ScalarField(dist, "complex", bases, ComplexF64)

    @test eltype(get_grid_data(field)) === ComplexF64
    @test eltype(get_coeff_data(field)) === ComplexF64
    @test _backend_global_size(get_grid_data(field)) == (n, n)
    @test _backend_global_size(get_coeff_data(field)) == (n, n)
    @test _backend_roundtrip_matches!(field; complex_values=true)
end


@testset "2D MPI ComplexFourier promotes a real grid (rank=$BACKEND_2D_RANK, np=$BACKEND_2D_NPROCS)" begin
    n = 10
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(
        coords;
        mesh=(BACKEND_2D_NPROCS,),
        dtype=Float64,
        architecture=CPU(),
    )
    bases = (
        ComplexFourier(coords["x"]; size=n, bounds=(0.0, 2π)),
        ComplexFourier(coords["y"]; size=n, bounds=(0.0, 2π)),
    )
    field = ScalarField(dist, "real_complex_fourier", bases, Float64)

    @test eltype(get_grid_data(field)) === Float64
    @test eltype(get_coeff_data(field)) === ComplexF64
    @test _backend_global_size(get_grid_data(field)) == (n, n)
    @test _backend_global_size(get_coeff_data(field)) == (n, n)
    @test _backend_roundtrip_matches!(field)
end


@testset "2D MPI public layout helpers use canonical transforms (rank=$BACKEND_2D_RANK, np=$BACKEND_2D_NPROCS)" begin
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(
        coords;
        mesh=(BACKEND_2D_NPROCS,),
        dtype=Float64,
        architecture=CPU(),
    )

    for (label, bases) in (
        (
            "real_complex_fourier",
            (
                ComplexFourier(coords["x"]; size=10, bounds=(0.0, 2π)),
                ComplexFourier(coords["y"]; size=10, bounds=(0.0, 2π)),
            ),
        ),
        (
            "mixed_chebyshev_fourier",
            (
                ChebyshevT(coords["x"]; size=10, bounds=(-1.0, 1.0)),
                RealFourier(coords["y"]; size=10, bounds=(0.0, 2π)),
            ),
        ),
    )
        canonical = ScalarField(dist, "$(label)_canonical", bases, Float64)
        via_layout = ScalarField(dist, "$(label)_layout", bases, Float64)
        original = _backend_fill_matching_grids!(canonical, via_layout)

        forward_transform!(canonical)
        require_coeff_space!(via_layout, 1)

        @test canonical.current_layout == :c
        @test via_layout.current_layout == :c
        @test _backend_local(get_coeff_data(via_layout)) ≈
              _backend_local(get_coeff_data(canonical))

        # Start both inverse paths from the same canonical coefficients. This
        # independently checks that the public layout helper performs any
        # required promotion and coupled inverse transform.
        copyto!(
            _backend_local(get_coeff_data(via_layout)),
            _backend_local(get_coeff_data(canonical)),
        )
        backward_transform!(canonical)
        require_grid_space!(via_layout, 2)

        @test canonical.current_layout == :g
        @test via_layout.current_layout == :g
        @test _backend_local(get_grid_data(via_layout)) ≈
              _backend_local(get_grid_data(canonical))
        @test _backend_local(get_grid_data(via_layout)) ≈ original
    end
end


@testset "2D MPI nonlinear scratch follows field dtype (rank=$BACKEND_2D_RANK, np=$BACKEND_2D_NPROCS)" begin
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(
        coords;
        mesh=(BACKEND_2D_NPROCS,),
        dtype=Float64,
        architecture=CPU(),
    )
    bases = (
        # Full-spectrum bases keep four coefficient rows for four-rank
        # coverage while N=4 still selects the truncation fallback.
        ComplexFourier(coords["x"]; size=4, bounds=(0.0, 2π)),
        ComplexFourier(coords["y"]; size=4, bounds=(0.0, 2π)),
    )
    field32 = ScalarField(dist, "nonlinear_float32", bases, Float32)
    field64 = ScalarField(dist, "nonlinear_float64", bases, Float64)

    for field in (field32, field64)
        global_grid = PencilArrays.global_view(get_grid_data(field))
        for I in CartesianIndices(global_grid)
            x = 2π * (I[1] - 1) / 4
            y = 2π * (I[2] - 1) / 4
            global_grid[I] =
                exp(0.17 * sin(x) + 0.11 * cos(y)) +
                0.123456789 * sin(x + y)
        end
    end

    reused = NonlinearEvaluator(dist; dealiasing_factor=1.5)
    evaluate_transform_multiply(field32, field32, reused)
    got = evaluate_transform_multiply(field64, field64, reused)

    fresh = NonlinearEvaluator(dist; dealiasing_factor=1.5)
    expected = evaluate_transform_multiply(field64, field64, fresh)
    @test isapprox(
        _backend_local(get_grid_data(got)),
        _backend_local(get_grid_data(expected));
        rtol=1e-12,
        atol=1e-12,
    )
end


@testset "2D MPI mixed batched inverse uses coupled DCT (rank=$BACKEND_2D_RANK, np=$BACKEND_2D_NPROCS)" begin
    coords = CartesianCoordinates("z", "x")
    dist = Distributor(
        coords;
        mesh=(BACKEND_2D_NPROCS,),
        dtype=Float64,
        architecture=CPU(),
    )
    bases = (
        ChebyshevT(coords["z"]; size=8, bounds=(-1.0, 1.0)),
        # Keep at least one Fourier coefficient per rank. The helper is called
        # directly, so its mixed-basis correctness does not depend on N<=4.
        RealFourier(coords["x"]; size=8, bounds=(0.0, 2π)),
    )
    fields = [
        ScalarField(dist, "mixed_batched_$i", bases, Float64)
        for i in 1:2
    ]
    originals = Matrix{Float64}[]
    for (i, field) in enumerate(fields)
        global_grid = PencilArrays.global_view(get_grid_data(field))
        for I in CartesianIndices(global_grid)
            global_grid[I] =
                0.137 * I[1] +
                sinpi((I[2] - 1) / 2) +
                0.019 * i * I[1] * I[2]
        end
        push!(originals, copy(_backend_local(get_grid_data(field))))
        forward_transform!(field)
    end

    Tarang._pencil_batched_backward!(fields)
    for i in eachindex(fields)
        @test isapprox(
            _backend_local(get_grid_data(fields[i])),
            originals[i];
            rtol=1e-11,
            atol=1e-11,
        )
    end
end


@testset "2D MPI derivatives alternate domain and dtype safely (rank=$BACKEND_2D_RANK, np=$BACKEND_2D_NPROCS)" begin
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(
        coords;
        mesh=(BACKEND_2D_NPROCS,),
        dtype=Float64,
        architecture=CPU(),
    )
    bases32 = (
        RealFourier(coords["x"]; size=8, bounds=(0.0, 2π)),
        RealFourier(coords["y"]; size=10, bounds=(0.0, 2π)),
    )
    bases64 = (
        RealFourier(coords["x"]; size=12, bounds=(0.0, 2π)),
        RealFourier(coords["y"]; size=14, bounds=(0.0, 2π)),
    )
    field32 = ScalarField(dist, "derivative_float32", bases32, Float32)
    field64 = ScalarField(dist, "derivative_complex64", bases64, ComplexF64)

    grid32 = PencilArrays.global_view(get_grid_data(field32))
    for I in CartesianIndices(grid32)
        y = 2π * (I[2] - 1) / 10
        grid32[I] = sin(2y)
    end
    grid64 = PencilArrays.global_view(get_grid_data(field64))
    for I in CartesianIndices(grid64)
        x = 2π * (I[1] - 1) / 12
        y = 2π * (I[2] - 1) / 14
        grid64[I] = (1 + 0.2 * cos(x)) * cis(2y)
    end

    for field in (field32, field64, field32)
        result = evaluate_differentiate(
            Differentiate(field, coords["y"], 1),
            :g,
        )
        result_grid = PencilArrays.global_view(get_grid_data(result))
        if field === field32
            error_max = maximum(CartesianIndices(result_grid)) do I
                y = 2π * (I[2] - 1) / 10
                abs(result_grid[I] - 2cos(2y))
            end
            @test error_max < 2e-5
        else
            error_max = maximum(CartesianIndices(result_grid)) do I
                x = 2π * (I[1] - 1) / 12
                y = 2π * (I[2] - 1) / 14
                expected = 2im * (1 + 0.2 * cos(x)) * cis(2y)
                abs(result_grid[I] - expected)
            end
            @test error_max < 1e-11
        end
    end
end


@testset "2D MPI mixed solve layouts alternate bundles safely (rank=$BACKEND_2D_RANK, np=$BACKEND_2D_NPROCS)" begin
    coords = CartesianCoordinates("z", "x")
    dist = Distributor(
        coords;
        mesh=(BACKEND_2D_NPROCS,),
        dtype=Float64,
        architecture=CPU(),
    )
    bases32 = (
        ChebyshevT(coords["z"]; size=8, bounds=(-1.0, 1.0)),
        RealFourier(coords["x"]; size=8, bounds=(0.0, 2π)),
    )
    bases64 = (
        ChebyshevT(coords["z"]; size=10, bounds=(-1.0, 1.0)),
        RealFourier(coords["x"]; size=12, bounds=(0.0, 2π)),
    )
    field32 = ScalarField(dist, "solve_float32", bases32, Float32)
    field64 = ScalarField(dist, "solve_float64", bases64, Float64)

    for field in (field32, field64)
        global_grid = PencilArrays.global_view(get_grid_data(field))
        nx = field === field32 ? 8 : 12
        for I in CartesianIndices(global_grid)
            x = 2π * (I[2] - 1) / nx
            global_grid[I] = 0.1 * I[1] + sin(x) + 0.025 * I[1] * cos(2x)
        end
    end

    for field in (field32, field64, field32)
        original = copy(_backend_local(get_grid_data(field)))
        stash = Tarang.to_solve_layout!([field], dist; fuse_from_grid=true)
        bundle = Tarang._field_transform_bundle(field)
        @test PencilArrays.pencil(get_coeff_data(field)) === bundle.pencil_solve

        Tarang.from_solve_layout!(stash, dist; to_grid=true)
        tol = field.dtype === Float32 ? 2e-5 : 1e-11
        @test isapprox(
            _backend_local(get_grid_data(field)),
            original;
            rtol=tol,
            atol=tol,
        )
    end
end


@testset "2D MPI resolution changes fail before mutation (rank=$BACKEND_2D_RANK, np=$BACKEND_2D_NPROCS)" begin
    function fourier_field(name; use_pencil_arrays=true)
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(
            coords;
            mesh=(BACKEND_2D_NPROCS,),
            dtype=Float64,
            architecture=CPU(),
            use_pencil_arrays=use_pencil_arrays,
        )
        FourierBasis = use_pencil_arrays ? RealFourier : ComplexFourier
        bases = (
            FourierBasis(coords["x"]; size=8, bounds=(0.0, 2π)),
            FourierBasis(coords["y"]; size=8, bounds=(0.0, 2π)),
        )
        return ScalarField(dist, name, bases, Float64)
    end

    function mixed_field(name)
        coords = CartesianCoordinates("z", "x")
        dist = Distributor(
            coords;
            mesh=(BACKEND_2D_NPROCS,),
            dtype=Float64,
            architecture=CPU(),
        )
        bases = (
            ChebyshevT(coords["z"]; size=8, bounds=(-1.0, 1.0)),
            RealFourier(coords["x"]; size=8, bounds=(0.0, 2π)),
        )
        return ScalarField(dist, name, bases, Float64)
    end

    for (operation, label) in ((set_scales!, "set_scales!"),
                               (preset_scales!, "preset_scales!"))
        @testset "$label rejects coefficient-space Fourier resize" begin
            field = fourier_field("coeff_$(label)")
            _backend_local(get_grid_data(field)) .= BACKEND_2D_RANK + 0.5
            ensure_layout!(field, :c)
            before = _backend_snapshot(field)

            @test_throws ErrorException operation(field, 1.5)
            _backend_test_unchanged(field, before)
        end

        @testset "$label rejects mixed grid-space resize" begin
            field = mixed_field("mixed_$(label)")
            _backend_local(get_grid_data(field)) .= BACKEND_2D_RANK + 0.25
            before = _backend_snapshot(field)

            @test_throws ErrorException operation(field, 1.5)
            _backend_test_unchanged(field, before)
        end

        @testset "$label rejects use_pencil_arrays=false resize" begin
            field = fourier_field("local_$(label)"; use_pencil_arrays=false)
            _backend_local(get_grid_data(field)) .= BACKEND_2D_RANK + 0.75
            before = _backend_snapshot(field)

            @test_throws ErrorException operation(field, 1.5)
            _backend_test_unchanged(field, before)
        end
    end
end

end
