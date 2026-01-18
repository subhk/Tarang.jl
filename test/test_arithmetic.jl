using Test
using Tarang

import Tarang: is_gpu_array, copy_to_device

const _CUDA_AVAILABLE_ARITH = try
    Tarang.has_cuda() && begin
        using CUDA
        CUDA.functional()
    end
catch
    false
end

function set_field_data!(field::ScalarField, values)
    ensure_layout!(field, :g)
    data = reshape(values, size(field["g"]))
    if is_gpu_array(field["g"])
        field["g"] .= copy_to_device(data, field["g"])
    else
        field["g"] .= data
    end
    return field
end

function scalar_field_fixture(arch)
    coords = CartesianCoordinates("x")
    dist = Distributor(coords; mesh=(1,), architecture=arch, dtype=Float64)
    basis = RealFourier(coords["x"]; size=4, bounds=(0.0, 2π))
    u = ScalarField(dist, "u", (basis,), Float64)
    v = ScalarField(dist, "v", (basis,), Float64)
    set_field_data!(u, Float64.(1:4))
    set_field_data!(v, Float64.(4:-1:1))
    return u, v
end

function vector_field_fixture(arch)
    coords = CartesianCoordinates("x", "y", "z")
    dist = Distributor(coords; mesh=(1, 1, 1), architecture=arch, dtype=Float64)
    bases = (
        RealFourier(coords["x"]; size=2, bounds=(0.0, 2π)),
        RealFourier(coords["y"]; size=2, bounds=(0.0, 2π)),
        RealFourier(coords["z"]; size=2, bounds=(0.0, 2π))
    )
    a = VectorField(dist, coords, "a", bases, Float64)
    b = VectorField(dist, coords, "b", bases, Float64)
    shape = size(a[1]["g"])
    data_base = reshape(Float64.(1:prod(shape)), shape)
    set_field_data!(a[1], data_base)
    set_field_data!(a[2], 2 .* data_base)
    set_field_data!(a[3], 3 .* data_base)
    set_field_data!(b[1], 0.5 .* data_base)
    set_field_data!(b[2], 0.25 .* data_base)
    set_field_data!(b[3], 0.125 .* data_base)
    return a, b
end

function field_data(field::ScalarField)
    ensure_layout!(field, :g)
    return Array(field["g"])
end

function run_scalar_tests(arch)
    u, v = scalar_field_fixture(arch)
    add_result = evaluate(Add(u, v))
    @test field_data(add_result) == field_data(u) .+ field_data(v)
    mul_result = evaluate(Multiply(u, 2.0))
    @test field_data(mul_result) == 2.0 .* field_data(u)
    div_result = evaluate(Divide(u, 2.0))
    @test field_data(div_result) == field_data(u) ./ 2.0
    pow_result = evaluate(Power(u, 2.0))
    @test field_data(pow_result) == field_data(u) .^ 2
    neg_result = evaluate(Negate(u))
    @test field_data(neg_result) == .-field_data(u)
    sub_result = evaluate(Subtract(u, v))
    @test field_data(sub_result) == field_data(u) .- field_data(v)
    ratio_result = evaluate(Divide(u, v))
    @test field_data(ratio_result) == field_data(u) ./ field_data(v)
    if arch isa GPU
        @test is_gpu_array(add_result["g"])
        @test is_gpu_array(mul_result["g"])
    end
end

function run_vector_tests(arch)
    a, b = vector_field_fixture(arch)
    dot_result = evaluate(DotProduct(a, b))
    expected_dot = Array(a[1]["g"]) .* Array(b[1]["g"]) .+
                   Array(a[2]["g"]) .* Array(b[2]["g"]) .+
                   Array(a[3]["g"]) .* Array(b[3]["g"])
    @test field_data(dot_result) == expected_dot
    cross_result = evaluate(CrossProduct(a, b))
    expected1 = Array(a[2]["g"]) .* Array(b[3]["g"]) .- Array(a[3]["g"]) .* Array(b[2]["g"])
    expected2 = Array(a[3]["g"]) .* Array(b[1]["g"]) .- Array(a[1]["g"]) .* Array(b[3]["g"])
    expected3 = Array(a[1]["g"]) .* Array(b[2]["g"]) .- Array(a[2]["g"]) .* Array(b[1]["g"])
    @test field_data(cross_result[1]) == expected1
    @test field_data(cross_result[2]) == expected2
    @test field_data(cross_result[3]) == expected3
    if arch isa GPU
        @test all(is_gpu_array(cross_result[i]["g"]) for i in 1:3)
    end
end

@testset "Arithmetic CPU" begin
    run_scalar_tests(CPU())
    run_vector_tests(CPU())
end

if _CUDA_AVAILABLE_ARITH
    using CUDA
    @testset "Arithmetic GPU" begin
        CUDA.allowscalar(false)
        run_scalar_tests(GPU())
        run_vector_tests(GPU())
    end
else
    @testset "Arithmetic GPU" begin
        @test_skip "CUDA not available"
    end
end
