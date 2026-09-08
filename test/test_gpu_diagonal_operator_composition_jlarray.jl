using Test
using Tarang

const _DOCJL_AVAILABLE = try
    @eval using JLArrays
    @eval using GPUArrays
    true
catch
    false
end

if _DOCJL_AVAILABLE
    const _DOCJL = JLArrays.JLArray
    const _DOCJL_ARCH = Tarang.GPU(JLArrays.JLBackend())
    Tarang.is_gpu_array(::_DOCJL) = true
    Tarang.architecture(::_DOCJL) = _DOCJL_ARCH
    Tarang.on_architecture(::Tarang.GPU{JLArrays.JLBackend}, a::Array) = _DOCJL(a)
    Tarang.on_architecture(::Tarang.GPU{JLArrays.JLBackend}, a::_DOCJL) = a
    Tarang.on_architecture(::Tarang.GPU{JLArrays.JLBackend}, a::AbstractArray) = _DOCJL(Array(a))
    Tarang.copy_to_device(a::AbstractArray, ::_DOCJL) = _DOCJL(Array(a))
    Tarang.copy_to_device(a::_DOCJL, ::_DOCJL) = copy(a)
    Tarang.array_type(::Tarang.GPU{JLArrays.JLBackend}) = _DOCJL
    Tarang.array_type(::Tarang.GPU{JLArrays.JLBackend}, T::Type) = _DOCJL{T}
end

@testset "Diagonal operator composition on JLArray device coefficients" begin
    if !_DOCJL_AVAILABLE
        @test_skip "JLArrays/GPUArrays unavailable"
    else
        GPUArrays.allowscalar(false)
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; dtype=Float64, device=_DOCJL_ARCH)
        xb = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
        domain = Domain(dist, (xb,))
        u, v = ScalarField(domain, "u"), ScalarField(domain, "v")
        # The parser only needs coefficient geometry. Starting in :c exercises
        # its real device allocation/broadcast path without a mock GPU FFT.
        u.current_layout = :c
        v.current_layout = :c
        namespace = Dict{String, Any}("u" => u, "v" => v)
        k = Float64.(0:8)
        for (expression, expected) in (
            ("lap(lap(u))", k.^4),
            ("lap(-2*u)", 2 .* k.^2),
            ("lap(2*(u + u))", -4 .* k.^2),
            ("fraclap(fraclap(u, 0.5), 1.5)", k.^4),
            ("lap(fraclap(u, 0.5))", -k.^3),
            ("fraclap(fraclap(u, -0.5), 0.5)", Float64.(k .> 0)),
        )
            op = Tarang.parse_expression(expression, namespace)
            actual = Tarang._diagonal_Lhat_from_expr(op, u)
            @test actual isa _DOCJL
            @test Array(actual) ≈ expected
        end
        for expression in ("lap(v)", "fraclap(v, 0.5)", "lap(u + v)", "lap(u*v)")
            op = Tarang.parse_expression(expression, namespace)
            @test Tarang._diagonal_Lhat_from_expr(op, u) === nothing
        end
    end
end
