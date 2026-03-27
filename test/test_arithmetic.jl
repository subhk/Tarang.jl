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

function vector_field_fixture(arch; right_handed::Bool=true)
    coords = CartesianCoordinates("x", "y", "z"; right_handed=right_handed)
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

# Helper: resolve a result that may be a Future or an already-evaluated field
_resolve(x::Future) = evaluate(x)
_resolve(x) = x

function run_scalar_tests(arch)
    u, v = scalar_field_fixture(arch)

    # --- Basic binary operations ---
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

    # --- Field * Field ---
    mul_ff = evaluate(Multiply(u, v))
    @test field_data(mul_ff) == field_data(u) .* field_data(v)

    # --- Mixed scalar-field operations ---
    @test field_data(evaluate(Add(u, 3.0)))      ≈ field_data(u) .+ 3.0
    @test field_data(evaluate(Add(3.0, u)))       ≈ field_data(u) .+ 3.0
    @test field_data(evaluate(Subtract(u, 1.0)))  ≈ field_data(u) .- 1.0
    @test field_data(evaluate(Subtract(10.0, u))) ≈ 10.0 .- field_data(u)
    @test field_data(evaluate(Multiply(3.0, u)))  ≈ 3.0 .* field_data(u)
    @test field_data(evaluate(Divide(1.0, v)))    ≈ 1.0 ./ field_data(v)

    # --- Chained / composed operations ---
    chained = evaluate(Add(Multiply(u, 2.0), v))
    @test field_data(chained) ≈ 2.0 .* field_data(u) .+ field_data(v)

    nested = evaluate(Negate(Add(u, v)))
    @test field_data(nested) ≈ .- (field_data(u) .+ field_data(v))

    # --- Operator overload syntax ---
    @test field_data(_resolve(u + v))   == field_data(u) .+ field_data(v)
    @test field_data(_resolve(u - v))   == field_data(u) .- field_data(v)
    @test field_data(_resolve(u * v))   == field_data(u) .* field_data(v)
    @test field_data(_resolve(u * 2.0)) == 2.0 .* field_data(u)
    @test field_data(_resolve(2.0 * u)) == 2.0 .* field_data(u)
    @test field_data(_resolve(u / 2.0)) ≈  field_data(u) ./ 2.0
    @test field_data(_resolve(u ^ 3))   == field_data(u) .^ 3
    @test field_data(_resolve(-u))      == .-field_data(u)

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

    # Left-handed Cartesian coordinates should flip the cross product sign
    a_lh, b_lh = vector_field_fixture(arch; right_handed=false)
    cross_lh = evaluate(CrossProduct(a_lh, b_lh))
    expected1_lh = - (Array(a_lh[2]["g"]) .* Array(b_lh[3]["g"]) .- Array(a_lh[3]["g"]) .* Array(b_lh[2]["g"]))
    expected2_lh = - (Array(a_lh[3]["g"]) .* Array(b_lh[1]["g"]) .- Array(a_lh[1]["g"]) .* Array(b_lh[3]["g"]))
    expected3_lh = - (Array(a_lh[1]["g"]) .* Array(b_lh[2]["g"]) .- Array(a_lh[2]["g"]) .* Array(b_lh[1]["g"]))
    @test field_data(cross_lh[1]) == expected1_lh
    @test field_data(cross_lh[2]) == expected2_lh
    @test field_data(cross_lh[3]) == expected3_lh

    # Mismatched coordinate systems should error
    a_rh, _ = vector_field_fixture(arch; right_handed=true)
    _, b_mh = vector_field_fixture(arch; right_handed=false)
    @test_throws ArgumentError evaluate(CrossProduct(a_rh, b_mh))

    # --- VectorField arithmetic ---
    va = field_data.(a.components)
    vb = field_data.(b.components)

    # Add / Subtract
    vec_add = evaluate(Add(a, b))
    for i in 1:3
        @test field_data(vec_add[i]) == va[i] .+ vb[i]
    end

    vec_sub = evaluate(Subtract(a, b))
    for i in 1:3
        @test field_data(vec_sub[i]) == va[i] .- vb[i]
    end

    # Negate
    vec_neg = evaluate(Negate(a))
    for i in 1:3
        @test field_data(vec_neg[i]) == .-va[i]
    end

    # Scale by number
    vec_scale = evaluate(Multiply(a, 3.0))
    for i in 1:3
        @test field_data(vec_scale[i]) == 3.0 .* va[i]
    end

    vec_scale_rev = evaluate(Multiply(0.5, a))
    for i in 1:3
        @test field_data(vec_scale_rev[i]) == 0.5 .* va[i]
    end

    # Divide by number
    vec_div = evaluate(Divide(a, 2.0))
    for i in 1:3
        @test field_data(vec_div[i]) ≈ va[i] ./ 2.0
    end

    # VectorField * VectorField should error (ambiguous)
    @test_throws ArgumentError evaluate(Multiply(a, b))

    # VectorField + ScalarField should error
    u_scalar, _ = scalar_field_fixture(arch)
    @test_throws ArgumentError evaluate(Add(a, u_scalar))

    # --- Unicode operator syntax ---
    dot_unicode = _resolve(a ⋅ b)
    @test field_data(dot_unicode) == expected_dot

    cross_unicode = _resolve(a × b)
    for i in 1:3
        @test field_data(cross_unicode[i]) == field_data(cross_result[i])
    end

    # --- Operator overloads on VectorFields ---
    vec_add_op = _resolve(a + b)
    for i in 1:3
        @test field_data(vec_add_op[i]) == va[i] .+ vb[i]
    end

    vec_sub_op = _resolve(a - b)
    for i in 1:3
        @test field_data(vec_sub_op[i]) == va[i] .- vb[i]
    end

    vec_neg_op = _resolve(-a)
    for i in 1:3
        @test field_data(vec_neg_op[i]) == .-va[i]
    end

    vec_scale_op = _resolve(a * 2.0)
    for i in 1:3
        @test field_data(vec_scale_op[i]) == 2.0 .* va[i]
    end

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

@testset "Cartesian unit vectors" begin
    coords = CartesianCoordinates("x", "y", "z")
    dist = Distributor(coords; mesh=(1, 1, 1), dtype=Float64)
    ex, ey, ez = unit_vector_fields(coords, dist)

    @test get_grid_data(ex[1])[] == 1.0
    @test get_grid_data(ex[2])[] == 0.0
    @test get_grid_data(ex[3])[] == 0.0

    @test get_grid_data(ey[1])[] == 0.0
    @test get_grid_data(ey[2])[] == 1.0
    @test get_grid_data(ey[3])[] == 0.0

    @test get_grid_data(ez[1])[] == 0.0
    @test get_grid_data(ez[2])[] == 0.0
    @test get_grid_data(ez[3])[] == 1.0
end
