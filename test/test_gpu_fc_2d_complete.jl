using Test
using Tarang

const _FC_REQUIRE_CUDA = lowercase(get(ENV, "TARANG_REQUIRE_CUDA", "false")) in
                         ("1", "true", "yes")
const _FC_HAS_CUDA = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end

if !_FC_HAS_CUDA
    _FC_REQUIRE_CUDA && error(
        "2D Fourier--Chebyshev validation requires CUDA.jl and a functional CUDA device")
    @testset "Complete 2D Fourier--Chebyshev GPU path" begin
        @test_skip "CUDA not functional on this host"
    end
else
    CUDA.allowscalar(false)

    function _fc_scaled_field_with_coords(device, coord_names, make_bases,
                                          ::Type{T}, scales) where {T}
        coords = CartesianCoordinates(coord_names...)
        dist = Distributor(coords; dtype=T, device=device)
        field = ScalarField(Domain(dist, make_bases(coords)), "u")
        preset_scales!(field, scales)
        ensure_layout!(field, :g)
        return field, coords
    end

    _fc_scaled_field(device, coord_names, make_bases, T, scales) =
        first(_fc_scaled_field_with_coords(device, coord_names, make_bases, T, scales))

    function _check_fc_transform(cpu_field, gpu_field, data;
                                 rtol=2e-10, atol=2e-11, roundtrip=true)
        @test size(get_grid_data(cpu_field)) == size(data)
        @test size(get_grid_data(gpu_field)) == size(data)
        copyto!(get_grid_data(cpu_field), data)
        copyto!(get_grid_data(gpu_field), CuArray(data))

        forward_transform!(cpu_field)
        forward_transform!(gpu_field)
        ensure_layout!(cpu_field, :c)
        ensure_layout!(gpu_field, :c)

        _, expected_shape, _ = Tarang.forward_layout(
            gpu_field.bases, size(data), eltype(data))
        @test size(get_coeff_data(gpu_field)) == expected_shape
        @test isapprox(Array(get_coeff_data(gpu_field)), get_coeff_data(cpu_field);
                       rtol, atol)

        saved_coefficients = copy(get_coeff_data(gpu_field))
        backward_transform!(cpu_field)
        backward_transform!(gpu_field)
        ensure_layout!(cpu_field, :g)
        ensure_layout!(gpu_field, :g)

        @test get_coeff_data(gpu_field) == saved_coefficients
        @test isapprox(Array(get_grid_data(gpu_field)), get_grid_data(cpu_field);
                       rtol, atol)
        roundtrip && @test isapprox(Array(get_grid_data(gpu_field)), data; rtol, atol)
        return nothing
    end

    @testset "2D FC device axis resizing" begin
        source_host = reshape(collect(Float64, 1:54), 6, 9)
        source = CuArray(source_host)
        truncated = CUDA.zeros(Float64, 6, 5)

        @test Tarang._copy_axis_prefix!(truncated, source, 2) === truncated
        @test Array(truncated) == source_host[:, 1:5]

        restored = CUDA.fill(-1.0, 6, 9)
        @test Tarang._zero_pad_axis_prefix!(restored, truncated, 2) === restored
        @test Array(restored[:, 1:5]) == source_host[:, 1:5]
        @test iszero(Array(restored[:, 6:9]))

        complex_host = complex.(reshape(collect(Float64, 1:24), 6, 4),
                                reshape(collect(Float64, 25:48), 6, 4))
        complex_source = CuArray(complex_host)
        complex_truncated = CUDA.zeros(ComplexF64, 3, 4)
        Tarang._copy_axis_prefix!(complex_truncated, complex_source, 1)
        @test Array(complex_truncated) == complex_host[1:3, :]

        complex_restored = CUDA.fill(ComplexF64(-1, -1), 6, 4)
        Tarang._zero_pad_axis_prefix!(complex_restored, complex_truncated, 1)
        @test Array(complex_restored[1:3, :]) == complex_host[1:3, :]
        @test iszero(Array(complex_restored[4:6, :]))
    end

    @testset "2D FC scaled transforms match CPU" begin
        @testset "RealFourier x ChebyshevT, both axes scaled" begin
            make_bases(c) = (
                RealFourier(c["x"]; size=8, bounds=(0.0, 2π)),
                ChebyshevT(c["z"]; size=9, bounds=(0.0, 1.0)),
            )
            scales = (3 / 2, 14 / 9)
            cpu = _fc_scaled_field(CPU(), ("x", "z"), make_bases, Float64, scales)
            gpu = _fc_scaled_field(GPU(), ("x", "z"), make_bases, Float64, scales)
            nx, nz = size(get_grid_data(cpu))
            x = (0:nx-1) .* (2π / nx)
            z = (1 .- cos.(π .* (0:nz-1) ./ (nz - 1))) ./ 2
            data = [sin(2xi) * (1 + zj + 0.25 * (2zj^2 - 1)) for xi in x, zj in z]
            _check_fc_transform(cpu, gpu, data)
        end

        @testset "ChebyshevT x RealFourier, both axes scaled" begin
            make_bases(c) = (
                ChebyshevT(c["z"]; size=9, bounds=(0.0, 1.0)),
                RealFourier(c["x"]; size=8, bounds=(0.0, 2π)),
            )
            scales = (14 / 9, 3 / 2)
            cpu = _fc_scaled_field(CPU(), ("z", "x"), make_bases, Float64, scales)
            gpu = _fc_scaled_field(GPU(), ("z", "x"), make_bases, Float64, scales)
            nz, nx = size(get_grid_data(cpu))
            z = (1 .- cos.(π .* (0:nz-1) ./ (nz - 1))) ./ 2
            x = (0:nx-1) .* (2π / nx)
            data = [(1 + zi - 0.5zi^2) * cos(2xj) for zi in z, xj in x]
            _check_fc_transform(cpu, gpu, data)
        end

        @testset "ComplexFourier x ChebyshevT with independent scales" begin
            make_bases(c) = (
                ComplexFourier(c["x"]; size=8, bounds=(0.0, 2π)),
                ChebyshevT(c["z"]; size=10, bounds=(-1.0, 1.0)),
            )
            scales = (5 / 4, 3 / 2)
            cpu = _fc_scaled_field(CPU(), ("x", "z"), make_bases, ComplexF64, scales)
            gpu = _fc_scaled_field(GPU(), ("x", "z"), make_bases, ComplexF64, scales)
            nx, nz = size(get_grid_data(cpu))
            x = (0:nx-1) .* (2π / nx)
            z = -cos.(π .* (0:nz-1) ./ (nz - 1))
            data = [exp(2im * xi) * (1 + zj + 0.2zj^3) for xi in x, zj in z]
            _check_fc_transform(cpu, gpu, data; rtol=5e-10, atol=5e-11)
        end

        @testset "unscaled mixed transform remains unchanged" begin
            make_bases(c) = (
                RealFourier(c["x"]; size=8, bounds=(0.0, 2π)),
                ChebyshevT(c["z"]; size=9, bounds=(0.0, 1.0)),
            )
            cpu = _fc_scaled_field(CPU(), ("x", "z"), make_bases, Float64, (1, 1))
            gpu = _fc_scaled_field(GPU(), ("x", "z"), make_bases, Float64, (1, 1))
            nx, nz = size(get_grid_data(cpu))
            x = (0:nx-1) .* (2π / nx)
            z = (1 .- cos.(π .* (0:nz-1) ./ (nz - 1))) ./ 2
            data = [cos(xi) * (1 - zj^2) for xi in x, zj in z]
            _check_fc_transform(cpu, gpu, data)
        end
    end

    @testset "2D FC plan cache distinguishes basis sizes" begin
        cuda_ext = Base.get_extension(Tarang, :TarangCUDAExt)
        cuda_ext.clear_gpu_mixed_transform_cache!()

        make_bases_9(c) = (
            RealFourier(c["x"]; size=8, bounds=(0.0, 2π)),
            ChebyshevT(c["z"]; size=9, bounds=(0.0, 1.0)),
        )
        make_bases_10(c) = (
            RealFourier(c["x"]; size=8, bounds=(0.0, 2π)),
            ChebyshevT(c["z"]; size=10, bounds=(0.0, 1.0)),
        )
        first = _fc_scaled_field(GPU(), ("x", "z"), make_bases_9,
                                 Float64, (1, 14 / 9))
        second = _fc_scaled_field(GPU(), ("x", "z"), make_bases_10,
                                  Float64, (1, 14 / 10))
        first_data = reshape(sin.(collect(1.0:112.0)), 8, 14)
        second_data = reshape(cos.(collect(1.0:112.0)), 8, 14)
        copyto!(get_grid_data(first), CuArray(first_data))
        copyto!(get_grid_data(second), CuArray(second_data))

        forward_transform!(first)
        forward_transform!(second)
        @test size(get_coeff_data(first)) == (5, 9)
        @test size(get_coeff_data(second)) == (5, 10)
    end

    @testset "2D FC derivatives match CPU and analytic values" begin
        function check_derivatives(; chebyshev_first::Bool, complex_fourier::Bool=false)
            coord_names = chebyshev_first ? ("z", "x") : ("x", "z")
            dtype = complex_fourier ? ComplexF64 : Float64
            make_bases = if chebyshev_first
                c -> (
                    ChebyshevT(c["z"]; size=11, bounds=(0.0, 1.0)),
                    (complex_fourier ? ComplexFourier : RealFourier)(
                        c["x"]; size=16, bounds=(0.0, 2π)),
                )
            else
                c -> (
                    (complex_fourier ? ComplexFourier : RealFourier)(
                        c["x"]; size=16, bounds=(0.0, 2π)),
                    ChebyshevT(c["z"]; size=11, bounds=(0.0, 1.0)),
                )
            end

            cpu, cpu_coords = _fc_scaled_field_with_coords(
                CPU(), coord_names, make_bases, dtype, (1, 1))
            gpu, gpu_coords = _fc_scaled_field_with_coords(
                GPU(), coord_names, make_bases, dtype, (1, 1))
            z = (1 .- cos.(π .* (0:10) ./ 10)) ./ 2
            x = (0:15) .* (2π / 16)
            profile = @. z^2 * (1 - z)
            dprofile = @. 2z - 3z^2
            wave = complex_fourier ? exp.(2im .* x) : sin.(2 .* x)
            dwave = complex_fourier ? 2im .* exp.(2im .* x) : 2 .* cos.(2 .* x)

            if chebyshev_first
                data = [profile[iz] * wave[ix] for iz in eachindex(z), ix in eachindex(x)]
                exact_dx = [profile[iz] * dwave[ix] for iz in eachindex(z), ix in eachindex(x)]
                exact_dz = [dprofile[iz] * wave[ix] for iz in eachindex(z), ix in eachindex(x)]
            else
                data = [wave[ix] * profile[iz] for ix in eachindex(x), iz in eachindex(z)]
                exact_dx = [dwave[ix] * profile[iz] for ix in eachindex(x), iz in eachindex(z)]
                exact_dz = [wave[ix] * dprofile[iz] for ix in eachindex(x), iz in eachindex(z)]
            end
            copyto!(get_grid_data(cpu), data)
            copyto!(get_grid_data(gpu), CuArray(data))
            cpu_snapshot = copy(get_grid_data(cpu))
            gpu_snapshot = copy(get_grid_data(gpu))

            for (name, exact) in (("x", exact_dx), ("z", exact_dz))
                cpu_g = Tarang.evaluate_differentiate(
                    Tarang.Differentiate(cpu, cpu_coords[name], 1), :g)
                gpu_g = Tarang.evaluate_differentiate(
                    Tarang.Differentiate(gpu, gpu_coords[name], 1), :g)
                @test isapprox(Array(get_grid_data(gpu_g)), get_grid_data(cpu_g);
                               rtol=2e-9, atol=2e-10)
                @test isapprox(get_grid_data(cpu_g), exact; rtol=2e-9, atol=2e-10)

                cpu_c = Tarang.evaluate_differentiate(
                    Tarang.Differentiate(cpu, cpu_coords[name], 1), :c)
                gpu_c = Tarang.evaluate_differentiate(
                    Tarang.Differentiate(gpu, gpu_coords[name], 1), :c)
                @test isapprox(Array(get_coeff_data(gpu_c)), get_coeff_data(cpu_c);
                               rtol=3e-9, atol=3e-10)
            end

            @test get_grid_data(cpu) == cpu_snapshot
            @test get_grid_data(gpu) == gpu_snapshot
        end

        check_derivatives(chebyshev_first=false)
        check_derivatives(chebyshev_first=true)
        check_derivatives(chebyshev_first=false, complex_fourier=true)
    end

    @testset "2D FC 3/2-dealiased products match CPU" begin
        function check_product(; chebyshev_first::Bool, nyquist::Bool)
            coord_names = chebyshev_first ? ("z", "x") : ("x", "z")
            make_bases = if chebyshev_first
                c -> (
                    ChebyshevT(c["z"]; size=9, bounds=(0.0, 1.0)),
                    RealFourier(c["x"]; size=12, bounds=(0.0, 2π), dealias=3 / 2),
                )
            else
                c -> (
                    RealFourier(c["x"]; size=12, bounds=(0.0, 2π), dealias=3 / 2),
                    ChebyshevT(c["z"]; size=9, bounds=(0.0, 1.0)),
                )
            end

            cpu_u = _fc_scaled_field(CPU(), coord_names, make_bases, Float64, (1, 1))
            cpu_v = ScalarField(cpu_u.domain, "v")
            gpu_u = _fc_scaled_field(GPU(), coord_names, make_bases, Float64, (1, 1))
            gpu_v = ScalarField(gpu_u.domain, "v")
            ensure_layout!(cpu_v, :g)
            ensure_layout!(gpu_v, :g)

            z = (1 .- cos.(π .* (0:8) ./ 8)) ./ 2
            x = (0:11) .* (2π / 12)
            ku, kv = nyquist ? (6, 6) : (2, 3)
            wave_u = cos.(ku .* x)
            wave_v = nyquist ? cos.(kv .* x) : sin.(kv .* x)
            if chebyshev_first
                data_u = [(1 + z[iz]) * wave_u[ix] for iz in eachindex(z), ix in eachindex(x)]
                data_v = [(1 - z[iz]) * wave_v[ix] for iz in eachindex(z), ix in eachindex(x)]
            else
                data_u = [wave_u[ix] * (1 + z[iz]) for ix in eachindex(x), iz in eachindex(z)]
                data_v = [wave_v[ix] * (1 - z[iz]) for ix in eachindex(x), iz in eachindex(z)]
            end
            copyto!(get_grid_data(cpu_u), data_u)
            copyto!(get_grid_data(cpu_v), data_v)
            copyto!(get_grid_data(gpu_u), CuArray(data_u))
            copyto!(get_grid_data(gpu_v), CuArray(data_v))
            cpu_u_before, cpu_v_before = copy(get_grid_data(cpu_u)), copy(get_grid_data(cpu_v))
            gpu_u_before, gpu_v_before = copy(get_grid_data(gpu_u)), copy(get_grid_data(gpu_v))

            cpu_ev = Tarang.NonlinearEvaluator(cpu_u.dist; dealiasing_factor=3 / 2)
            gpu_ev = Tarang.NonlinearEvaluator(gpu_u.dist; dealiasing_factor=3 / 2)
            for layout in (:g, :c)
                cpu_product = Tarang.evaluate_transform_multiply(
                    cpu_u, cpu_v, cpu_ev; result_layout=layout)
                gpu_product = Tarang.evaluate_transform_multiply(
                    gpu_u, gpu_v, gpu_ev; result_layout=layout)
                if layout === :g
                    @test isapprox(Array(get_grid_data(gpu_product)),
                                   get_grid_data(cpu_product); rtol=3e-9, atol=3e-10)
                    if !nyquist
                        @test isapprox(get_grid_data(cpu_product), data_u .* data_v;
                                       rtol=3e-9, atol=3e-10)
                    end
                else
                    @test isapprox(Array(get_coeff_data(gpu_product)),
                                   get_coeff_data(cpu_product); rtol=4e-9, atol=4e-10)
                end
            end

            @test get_grid_data(cpu_u) == cpu_u_before
            @test get_grid_data(cpu_v) == cpu_v_before
            @test get_grid_data(gpu_u) == gpu_u_before
            @test get_grid_data(gpu_v) == gpu_v_before
        end

        for chebyshev_first in (false, true), nyquist in (false, true)
            check_product(; chebyshev_first, nyquist)
        end
    end
end
