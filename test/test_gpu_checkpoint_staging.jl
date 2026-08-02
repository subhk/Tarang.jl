"""
Device staging for checkpoint loads, with no GPU hardware.

NetCDF reads into host memory, so a device-resident field has to be uploaded.
The upload is one explicit `copyto!` — deliberate one-shot I/O staging, not a
silent CPU fallback, because file I/O cannot read into device memory at all.

JLArray provides device-like arrays with no driver. Everything here stays in grid
layout, so no FFT is needed and the JLArray transform limitation does not apply.
"""

using Test
using Tarang

const _JL_LOADED = try
    @eval using JLArrays
    @eval using GPUArrays
    true
catch err
    @info "JLArrays unavailable; skipping device staging tests" err
    false
end

@testset "checkpoint load stages onto the device" begin
    if !_JL_LOADED
        @test_skip "JLArrays not available in this environment"
    else
        # Test-scoped: teaches Tarang to build fields backed by JLArray.
        @eval Tarang.array_type(::Tarang.GPU{<:JLArrays.JLBackend}) = JLArrays.JLArray
        @eval Tarang.array_type(::Tarang.GPU{<:JLArrays.JLBackend}, ::Type{T}) where {T} =
            JLArrays.JLArray{T}

        coords = CartesianCoordinates("x")
        dist = Distributor(coords; dtype=Float64, device=Tarang.GPU(JLArrays.JLBackend()))
        xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        domain = Domain(dist, (xb,))

        u = ScalarField(domain, "u")
        ensure_layout!(u, :g)
        gd = get_grid_data(u)
        @test gd isa JLArrays.JLArray
        gd .= collect(1.0:8.0)

        dir = mktempdir()
        written = save_field(u, joinpath(dir, "dev"), "u")
        @test isfile(written)

        v = ScalarField(domain, "v")
        ensure_layout!(v, :g)
        load_field!(v, written, "u")

        loaded = get_grid_data(v)
        # The data must land in DEVICE storage, not be replaced by a host array.
        @test loaded isa JLArrays.JLArray
        @test Array(loaded) == collect(1.0:8.0)
    end
end

@testset "_store_local_grid_data! rejects a wrong-sized slab" begin
    domain = PeriodicDomain(8)
    u = ScalarField(domain, "u")
    ensure_layout!(u, :g)
    @test_throws ErrorException Tarang._store_local_grid_data!(u, zeros(Float64, 4))
end
