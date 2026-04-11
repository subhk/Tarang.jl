using Test
using Tarang

@testset "Fields" begin
    # Shared setup
    domain = PeriodicDomain(16, 16)
    dist = domain.dist
    bases = domain.bases

    # 1. ScalarField construction
    @testset "ScalarField construction" begin
        f = ScalarField(domain, "f")
        @test f isa ScalarField
        @test f.name == "f"
        @test f.dtype == Float64
        @test length(f.bases) == 2
        @test f.domain !== nothing
        @test f.current_layout == :g
        @test Tarang.get_grid_data(f) !== nothing

        # Explicit constructor from Distributor + bases
        g = ScalarField(dist, "g", bases, Float64)
        @test g isa ScalarField
        @test g.name == "g"
        @test g.bases === bases
    end

    # 2. VectorField construction and component access
    @testset "VectorField construction and access" begin
        u = VectorField(domain, "u")
        @test u isa VectorField
        @test u.name == "u"
        @test length(u.components) == 2
        @test u.components isa Vector{<:ScalarField}

        # From Distributor + bases
        v = VectorField(dist, "v", bases, Float64)
        @test v isa VectorField
        @test length(v.components) == 2

        # Integer indexing
        @test u[1] isa ScalarField
        @test u[2] isa ScalarField
        @test u[1] === u.components[1]
        @test u[2] === u.components[2]
    end

    # 3. TensorField construction
    @testset "TensorField construction" begin
        S = TensorField(domain, "S")
        @test S isa TensorField
        @test S.name == "S"
        @test size(S.components) == (2, 2)

        # Component access and naming
        @test S[1, 1] isa ScalarField
        @test S[1, 2] isa ScalarField
        @test S[1, 1].name == "S_xx"
        @test S[1, 2].name == "S_xy"
        @test S[2, 1].name == "S_yx"
        @test S[2, 2].name == "S_yy"
    end

    # 4. Field data access
    @testset "Data access" begin
        f = ScalarField(domain, "f")

        # Raw accessors
        gd = Tarang.get_grid_data(f)
        @test gd isa AbstractArray
        @test ndims(gd) == 2
        cd = Tarang.get_coeff_data(f)
        @test cd isa AbstractArray

        # grid_data auto-transforms to grid
        f2 = ScalarField(domain, "f2")
        set!(f2, 1.0)
        forward_transform!(f2)
        @test f2.current_layout == :c
        gd2 = grid_data(f2)
        @test f2.current_layout == :g
        @test gd2 isa AbstractArray

        # coeff_data auto-transforms to coeff
        f3 = ScalarField(domain, "f3")
        set!(f3, 1.0)
        @test f3.current_layout == :g
        cd3 = coeff_data(f3)
        @test f3.current_layout == :c
        @test cd3 isa AbstractArray

        # VectorField data access
        u = VectorField(domain, "u")
        gds = grid_data(u)
        @test length(gds) == 2
        @test all(d -> d isa AbstractArray, gds)
        cds = coeff_data(u)
        @test length(cds) == 2
        @test all(d -> d isa AbstractArray, cds)
    end

    # 5. Field layout management
    @testset "Layout management" begin
        f = ScalarField(domain, "f")
        set!(f, 1.0)
        @test f.current_layout == :g

        # grid -> coeff
        ensure_layout!(f, :c)
        @test f.current_layout == :c

        # coeff -> grid
        ensure_layout!(f, :g)
        @test f.current_layout == :g

        # Idempotent: calling again does not corrupt data
        gd_before = copy(Tarang.get_grid_data(f))
        ensure_layout!(f, :g)
        @test Tarang.get_grid_data(f) == gd_before

        # VectorField: applies to all components
        u = VectorField(domain, "u")
        set!(u, ((x, y) -> sin(x), (x, y) -> cos(y)))
        ensure_layout!(u, :c)
        @test all(c -> c.current_layout == :c, u.components)
        ensure_layout!(u, :g)
        @test all(c -> c.current_layout == :g, u.components)
    end

    # 6. Forward / backward transforms
    @testset "Transforms" begin
        f = ScalarField(domain, "f")
        set!(f, (x, y) -> sin(x) * cos(y))

        # Forward sets layout to :c
        forward_transform!(f)
        @test f.current_layout == :c

        # Backward sets layout to :g
        backward_transform!(f)
        @test f.current_layout == :g

        # Roundtrip preserves data
        set!(f, (x, y) -> sin(x) * cos(y))
        original = copy(Tarang.get_grid_data(f))
        forward_transform!(f)
        backward_transform!(f)
        @test isapprox(Tarang.get_grid_data(f), original; rtol=1e-10, atol=1e-12)

        # VectorField roundtrip
        u = VectorField(domain, "u")
        set!(u, ((x, y) -> sin(y), (x, y) -> -sin(x)))
        originals = [copy(Tarang.get_grid_data(c)) for c in u.components]
        forward_transform!(u)
        backward_transform!(u)
        for (i, c) in enumerate(u.components)
            @test isapprox(Tarang.get_grid_data(c), originals[i]; rtol=1e-10, atol=1e-12)
        end
    end

    # 7. Copy and deepcopy
    @testset "Copy and deepcopy" begin
        f = ScalarField(domain, "f")
        set!(f, (x, y) -> sin(x))

        # copy: shared dist, independent data
        fc = copy(f)
        @test fc.name == f.name
        @test fc.dist === f.dist
        @test fc.current_layout == f.current_layout
        @test Tarang.get_grid_data(fc) == Tarang.get_grid_data(f)
        Tarang.get_grid_data(fc) .= 0.0
        @test !all(Tarang.get_grid_data(f) .== 0.0)

        # deepcopy: fully independent
        set!(f, (x, y) -> cos(y))
        fd = deepcopy(f)
        @test fd.name == f.name
        @test fd.current_layout == f.current_layout
        @test Tarang.get_grid_data(fd) == Tarang.get_grid_data(f)
        Tarang.get_grid_data(fd) .= 99.0
        @test !all(Tarang.get_grid_data(f) .== 99.0)
    end

    # 8. Bracket notation: field["g"], field["c"]
    @testset "Bracket notation" begin
        f = ScalarField(domain, "f")
        set!(f, 1.0)

        # field["g"] returns grid-space data
        gd = f["g"]
        @test gd isa AbstractArray
        @test all(gd .== 1.0)

        # field["c"] returns coeff-space data
        set!(f, 0.0)
        cd = f["c"]
        @test cd isa AbstractArray

        # Auto-transform from coeff to grid via bracket
        set!(f, (x, y) -> sin(x))
        original = copy(f["g"])
        forward_transform!(f)
        @test f.current_layout == :c
        recovered = f["g"]
        @test isapprox(recovered, original; rtol=1e-10, atol=1e-12)

        # Auto-transform from grid to coeff via bracket
        set!(f, (x, y) -> sin(x))
        forward_transform!(f)
        ref_coeff = copy(f["c"])
        backward_transform!(f)
        @test f.current_layout == :g
        @test isapprox(f["c"], ref_coeff; rtol=1e-10, atol=1e-12)

        # Unknown layout string throws
        @test_throws ArgumentError f["x"]

        # VectorField bracket returns array of component data
        u = VectorField(domain, "u")
        gds = u["g"]
        @test length(gds) == 2
        @test all(d -> d isa AbstractArray, gds)
    end

    # 9. set! for initial conditions
    @testset "set! initial conditions" begin
        # Scalar constant
        f = ScalarField(domain, "f")
        result = set!(f, 3.14)
        @test all(Tarang.get_grid_data(f) .== 3.14)
        @test f.current_layout == :g
        @test result === f  # returns the field

        # Scalar function 2D
        set!(f, (x, y) -> sin(x) * cos(y))
        @test f.current_layout == :g
        @test !all(Tarang.get_grid_data(f) .== 0.0)

        # VectorField with tuple of functions
        u = VectorField(domain, "u")
        set!(u, ((x, y) -> sin(y), (x, y) -> -sin(x)))
        for c in u.components
            @test c.current_layout == :g
            @test !all(Tarang.get_grid_data(c) .== 0.0)
        end

        # Wrong number of functions throws
        @test_throws ArgumentError set!(u, ((x, y) -> sin(y),))

        # 1D scalar function
        domain1d = PeriodicDomain(16)
        f1d = ScalarField(domain1d, "f1d")
        set!(f1d, (x,) -> sin(x))
        @test f1d.current_layout == :g
        @test !all(Tarang.get_grid_data(f1d) .== 0.0)
    end

    # 10. VectorField getproperty access
    @testset "VectorField getproperty" begin
        # 2D periodic: u.x and u.y
        u = VectorField(domain, "u")
        @test u.x isa ScalarField
        @test u.y isa ScalarField
        @test u.x === u[1]
        @test u.y === u[2]

        # Channel domain: u.x and u.z
        ch = ChannelDomain(8, 8; Lx=2pi, Lz=1.0, dealias=1.0)
        v = VectorField(ch, "v")
        @test v.x isa ScalarField
        @test v.z isa ScalarField
        @test v.x === v[1]
        @test v.z === v[2]

        # Invalid component name throws
        @test_throws ArgumentError u.w

        # Struct fields still accessible
        @test u.name == "u"
        @test u.dist === dist
        @test u.dtype == Float64
    end

    # Bonus: 3D field sanity checks
    @testset "3D fields" begin
        domain3d = PeriodicDomain(8, 8, 8)

        f = ScalarField(domain3d, "f3d")
        @test ndims(Tarang.get_grid_data(f)) == 3
        set!(f, (x, y, z) -> sin(x) * cos(y) * sin(z))
        @test !all(Tarang.get_grid_data(f) .== 0.0)

        u = VectorField(domain3d, "u3d")
        @test length(u.components) == 3
        @test u.x === u[1]
        @test u.y === u[2]
        @test u.z === u[3]

        S = TensorField(domain3d, "S3d")
        @test size(S.components) == (3, 3)

        # 3D roundtrip
        set!(f, (x, y, z) -> cos(x) * sin(y + z))
        original = copy(Tarang.get_grid_data(f))
        forward_transform!(f)
        backward_transform!(f)
        @test isapprox(Tarang.get_grid_data(f), original; rtol=1e-10, atol=1e-12)
    end
end
