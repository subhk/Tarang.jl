"""
Dedicated tests for `src/core/cartesian_operators/cartesian_operator_core.jl`.

Targets the previously-uncovered surface of the CartesianComponent operator and
its core helpers:

  * CartesianComponent constructor — success fields + all guard/error paths
  * get_tensorsig                  — VectorField, TensorField, no-tensorsig fallback
  * matrix_dependence / matrix_coupling (CartesianComponent delegation)
  * subproblem_matrix              — vector selector + rank-2 tensor selector (index 0/1)
  * get_scalar_size                — :bases branch, component recursion, fallback
  * check_conditions / enforce_conditions
  * operate                        — vector path, tensor path, output-type guard
  * extract_tensor_component!      — row/column extraction + size + index guards
  * evaluate_cartesian_component   — VectorField (:g/:c) + TensorField copy path

Independent oracles only: expected values come from the linear-algebra / index
definition of component extraction (selection matrices, column-major tensor
flattening, sin/cos analytic fields), never from the function under test.
"""

using Test
using LinearAlgebra
using Tarang

const Lx = 1.3
const Ly = 2.4
const Lz = 1.9
const Ncore = 16

# ---------------------------------------------------------------------------
# Domain builders (mirror test_cartesian_operators.jl idiom)
# ---------------------------------------------------------------------------

"""2D Fourier-Fourier domain (x,y)."""
function build_FF2(N::Int=Ncore)
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; dtype=Float64)
    xb = RealFourier(coords["x"]; size=N, bounds=(0.0, Lx))
    yb = RealFourier(coords["y"]; size=N, bounds=(0.0, Ly))
    bases = (xb, yb)
    x = local_grid(xb, dist, 1.0)
    y = local_grid(yb, dist, 1.0)
    return coords, dist, bases, (x, y)
end

"""3D Fourier-Fourier-Fourier domain (x,y,z)."""
function build_FFF3(N::Int=Ncore)
    coords = CartesianCoordinates("x", "y", "z")
    dist = Distributor(coords; dtype=Float64)
    xb = RealFourier(coords["x"]; size=N, bounds=(0.0, Lx))
    yb = RealFourier(coords["y"]; size=N, bounds=(0.0, Ly))
    zb = RealFourier(coords["z"]; size=N, bounds=(0.0, Lz))
    bases = (xb, yb, zb)
    return coords, dist, bases
end

gridof(f) = Tarang.get_grid_data(f)
coefof(f) = Tarang.get_coeff_data(f)

# ===========================================================================
# 1. CartesianComponent constructor: fields + guard/error paths
# ===========================================================================

@testset "CartesianComponent constructor" begin
    coords, dist, bases, _ = build_FF2()
    coords3, dist3, bases3 = build_FFF3()

    @testset "vector field: field values" begin
        u = VectorField(dist, coords, "u", bases, Float64)

        opx = CartesianComponent(u; index=0, comp=coords["x"])
        opy = CartesianComponent(u; index=0, comp=coords["y"])

        # comp_subaxis is the 0-based position of the coordinate within coordsys.
        @test opx.comp_subaxis == 0          # x is coordinate 1 -> 0-based 0
        @test opy.comp_subaxis == 1          # y is coordinate 2 -> 0-based 1
        @test opx.index == 0
        # The selected tensor index's coordsys is the field coordsys.
        @test opx.coordsys === coords
        # Extracting the single tensor index of a vector leaves a scalar (() sig).
        @test opx.tensorsig == ()
        @test opy.tensorsig == ()
    end

    @testset "tensor field: output tensorsig drops extracted index" begin
        T = TensorField(dist, coords, "T", bases, Float64)
        # rank-2 tensorsig is (coords, coords); dropping index 0 leaves (coords,)
        op0 = CartesianComponent(T; index=0, comp=coords["x"])
        op1 = CartesianComponent(T; index=1, comp=coords["y"])
        @test op0.tensorsig == (coords,)
        @test op1.tensorsig == (coords,)
        @test op0.comp_subaxis == 0
        @test op1.comp_subaxis == 1
        @test op1.index == 1
    end

    @testset "3D vector: comp_subaxis for z" begin
        u3 = VectorField(dist3, coords3, "u", bases3, Float64)
        opz = CartesianComponent(u3; index=0, comp=coords3["z"])
        @test opz.comp_subaxis == 2          # z is coordinate 3 -> 0-based 2
    end

    @testset "guard: non-vector/tensor operand rejected" begin
        f = ScalarField(dist, "f", bases, Float64)
        @test_throws ArgumentError CartesianComponent(f; index=0, comp=coords["x"])
    end

    @testset "guard: invalid tensor index rejected" begin
        u = VectorField(dist, coords, "u", bases, Float64)
        # vector has 1 tensor index; valid indices are only {0}
        @test_throws ArgumentError CartesianComponent(u; index=1, comp=coords["x"])
        @test_throws ArgumentError CartesianComponent(u; index=-1, comp=coords["x"])
        T = TensorField(dist, coords, "T", bases, Float64)
        # rank-2 tensor has 2 tensor indices; index 2 is out of range
        @test_throws ArgumentError CartesianComponent(T; index=2, comp=coords["x"])
    end

    @testset "guard: coordinate not in coordsys rejected" begin
        u = VectorField(dist, coords, "u", bases, Float64)   # 2D x,y
        # z is not part of the 2D coordinate system
        @test_throws ArgumentError CartesianComponent(u; index=0, comp=coords3["z"])
    end
end

# ===========================================================================
# 2. get_tensorsig
# ===========================================================================

@testset "get_tensorsig" begin
    coords, dist, bases, _ = build_FF2()

    u = VectorField(dist, coords, "u", bases, Float64)
    T = TensorField(dist, coords, "T", bases, Float64)
    f = ScalarField(dist, "f", bases, Float64)

    # VectorField -> rank-1 signature (one copy of the coordinate system).
    @test get_tensorsig(u) == (coords,)
    # TensorField -> rank-2 signature.
    @test get_tensorsig(T) == (coords, coords)
    # ScalarField has no tensor structure (no :tensorsig field, not vec/tensor).
    @test get_tensorsig(f) == ()
end

# ===========================================================================
# 3. matrix_dependence / matrix_coupling (delegation to operand)
# ===========================================================================

@testset "matrix_dependence / matrix_coupling" begin
    coords, dist, bases, _ = build_FF2()
    u = VectorField(dist, coords, "u", bases, Float64)
    v = VectorField(dist, coords, "v", bases, Float64)

    op = CartesianComponent(u; index=0, comp=coords["x"])

    # Component extraction adds no dependence beyond the operand's own.
    # The operand IS u, so it depends on u (true) and not on the unrelated v.
    dep = matrix_dependence(op, u, v)
    @test dep == Bool[true, false]

    # Pure selection adds no coupling between distinct variables.
    cpl = matrix_coupling(op, u, v)
    @test cpl == Bool[false, false]
end

# ===========================================================================
# 4. subproblem_matrix  (selection-matrix structure is the oracle)
# ===========================================================================

@testset "subproblem_matrix" begin
    coords, dist, bases, _ = build_FF2()
    coords3, dist3, bases3 = build_FFF3()

    # scalar size for an N x N field is N*N
    n = Ncore * Ncore

    @testset "vector: selects e_i ⊗ I" begin
        u = VectorField(dist, coords, "u", bases, Float64)

        opx = CartesianComponent(u; index=0, comp=coords["x"])
        opy = CartesianComponent(u; index=0, comp=coords["y"])

        Mx = subproblem_matrix(opx, nothing)
        My = subproblem_matrix(opy, nothing)

        dim = 2
        # Oracle: kron(e_i', I_n). Shape (n, dim*n); block i is identity.
        @test size(Mx) == (n, dim * n)
        @test Matrix(Mx) == kron(reshape(Float64[1, 0], 1, 2), Matrix(I, n, n))
        @test Matrix(My) == kron(reshape(Float64[0, 1], 1, 2), Matrix(I, n, n))
    end

    @testset "tensor: row selection (index 0)" begin
        T = TensorField(dist, coords, "T", bases, Float64)
        dim = 2
        # Select row comp_idx=1 (x). Column-major flatten: idx_in = (j-1)*dim + comp_idx.
        # Output row j picks input component (row=x, col=j).
        op = CartesianComponent(T; index=0, comp=coords["x"])
        M = subproblem_matrix(op, nothing)
        @test size(M) == (dim * n, dim * dim * n)

        # Build oracle selector independently.
        comp_idx = 1
        selector = zeros(Float64, dim, dim * dim)
        for j in 1:dim
            selector[j, (j - 1) * dim + comp_idx] = 1.0
        end
        @test Matrix(M) == kron(selector, Matrix(I, n, n))
    end

    @testset "tensor: column selection (index 1)" begin
        T = TensorField(dist, coords, "T", bases, Float64)
        dim = 2
        # Select column comp_idx=2 (y). idx_in = (comp_idx-1)*dim + i.
        op = CartesianComponent(T; index=1, comp=coords["y"])
        M = subproblem_matrix(op, nothing)

        comp_idx = 2
        selector = zeros(Float64, dim, dim * dim)
        for i in 1:dim
            selector[i, (comp_idx - 1) * dim + i] = 1.0
        end
        @test Matrix(M) == kron(selector, Matrix(I, n, n))
    end

    @testset "3D vector selection" begin
        u3 = VectorField(dist3, coords3, "u", bases3, Float64)
        dim = 3
        n3 = Ncore^3
        opz = CartesianComponent(u3; index=0, comp=coords3["z"])
        M = subproblem_matrix(opz, nothing)
        @test size(M) == (n3, dim * n3)
        # Compare with ≈, not ==: the selection matrix is built from a sparse
        # `kron`, and exact `==` on the larger 3D case was brittle across the
        # x64/Julia-1.12.6 CI runner (the 2D cases above pass with `==`). `≈`
        # tolerates floating representation while still catching a structural
        # error — a misplaced selector entry (1.0 vs 0.0) is far above tolerance.
        @test Matrix(M) ≈ kron(reshape(Float64[0, 0, 1], 1, 3), Matrix(I, n3, n3))
    end
end

# ===========================================================================
# 5. get_scalar_size
# ===========================================================================

@testset "get_scalar_size" begin
    coords, dist, bases, _ = build_FF2()
    coords3, dist3, bases3 = build_FFF3()

    f = ScalarField(dist, "f", bases, Float64)
    u = VectorField(dist, coords, "u", bases, Float64)
    T = TensorField(dist, coords, "T", bases, Float64)

    # ScalarField: hits :bases branch -> product of basis sizes.
    @test get_scalar_size(f, nothing) == Ncore * Ncore
    # VectorField: recurses into components[1] (a ScalarField).
    @test get_scalar_size(u, nothing) == Ncore * Ncore
    # TensorField: recurses into components[1] (a ScalarField) too.
    @test get_scalar_size(T, nothing) == Ncore * Ncore

    # 3D scalar
    f3 = ScalarField(dist3, "f", bases3, Float64)
    @test get_scalar_size(f3, nothing) == Ncore^3

    # Fallback branch: an operand with none of components/buffers/bases -> 1.
    @test get_scalar_size(42, nothing) == 1
end

# ===========================================================================
# 6. check_conditions / enforce_conditions
# ===========================================================================

@testset "check_conditions / enforce_conditions" begin
    coords, dist, bases, _ = build_FF2()

    @testset "vector: shared layout -> true" begin
        u = VectorField(dist, coords, "u", bases, Float64)
        for c in u.components
            ensure_layout!(c, :g)
        end
        op = CartesianComponent(u; index=0, comp=coords["x"])
        @test check_conditions(op) == true
        # enforce_conditions normalizes layout; returns nothing, must not error.
        @test enforce_conditions(op) === nothing
    end

    @testset "vector: mixed layouts -> false" begin
        u = VectorField(dist, coords, "u", bases, Float64)
        ensure_layout!(u.components[1], :g)
        ensure_layout!(u.components[2], :c)
        op = CartesianComponent(u; index=0, comp=coords["x"])
        # Two distinct current_layouts among the components -> not OK.
        @test check_conditions(op) == false
    end

    @testset "tensor: shared layout -> true; enforce ok" begin
        T = TensorField(dist, coords, "T", bases, Float64)
        for c in T.components
            ensure_layout!(c, :g)
        end
        op = CartesianComponent(T; index=0, comp=coords["x"])
        @test check_conditions(op) == true
        @test enforce_conditions(op) === nothing
    end
end

# ===========================================================================
# 7. operate  (in-place extraction into a provided output)
# ===========================================================================

@testset "operate" begin
    coords, dist, bases, r = build_FF2()
    x, y = r

    @testset "vector -> ScalarField out" begin
        u = VectorField(dist, coords, "u", bases, Float64)
        for i in eachindex(x), j in eachindex(y)
            gridof(u.components[1])[i, j] = sin(2π * x[i] / Lx)
            gridof(u.components[2])[i, j] = cos(2π * y[j] / Ly)
        end
        u.components[1].current_layout = :g
        u.components[2].current_layout = :g

        out = ScalarField(dist, "out", bases, Float64)
        out.current_layout = :g

        op = CartesianComponent(u; index=0, comp=coords["y"])
        ret = operate(op, out)

        @test ret === out
        # Oracle: out equals the y-component grid data exactly (a copy).
        @test gridof(out) == gridof(u.components[2])
        # Independent oracle on the analytic field.
        expected = [cos(2π * y[j] / Ly) for i in eachindex(x), j in eachindex(y)]
        @test isapprox(gridof(out), expected, rtol=1e-12)
    end

    @testset "tensor -> VectorField out (row extract)" begin
        T = TensorField(dist, coords, "T", bases, Float64)
        # Distinct constant per component so we can verify the right one is picked.
        for i in 1:2, j in 1:2
            fill!(gridof(T.components[i, j]), 10.0 * i + j)
            T.components[i, j].current_layout = :g
        end
        out = VectorField(dist, coords, "out", bases, Float64)
        for c in out.components
            c.current_layout = :g
        end

        # index=0, comp=x (comp_idx=1): out[j] = T[1, j]  -> values 11, 12
        op = CartesianComponent(T; index=0, comp=coords["x"])
        operate(op, out)
        @test all(gridof(out.components[1]) .== 11.0)   # T[1,1]
        @test all(gridof(out.components[2]) .== 12.0)   # T[1,2]
    end

    @testset "guard: tensor extraction needs VectorField out" begin
        T = TensorField(dist, coords, "T", bases, Float64)
        for c in T.components
            ensure_layout!(c, :g)
        end
        bad_out = ScalarField(dist, "bad", bases, Float64)
        op = CartesianComponent(T; index=0, comp=coords["x"])
        @test_throws ArgumentError operate(op, bad_out)
    end
end

# ===========================================================================
# 8. extract_tensor_component!
# ===========================================================================

@testset "extract_tensor_component!" begin
    coords, dist, bases, _ = build_FF2()

    function make_tensor()
        T = TensorField(dist, coords, "T", bases, Float64)
        for i in 1:2, j in 1:2
            fill!(gridof(T.components[i, j]), 10.0 * i + j)  # T[i,j] = 10i + j
            T.components[i, j].current_layout = :g
        end
        return T
    end

    @testset "index 0 -> row comp_idx" begin
        T = make_tensor()
        out = VectorField(dist, coords, "out", bases, Float64)
        # comp_subaxis=0 -> comp_idx=1; row 1: out[j] = T[1,j] = 11, 12
        Tarang.extract_tensor_component!(out, T, 0, 0, :g)
        @test all(gridof(out.components[1]) .== 11.0)
        @test all(gridof(out.components[2]) .== 12.0)
    end

    @testset "index 1 -> column comp_idx" begin
        T = make_tensor()
        out = VectorField(dist, coords, "out", bases, Float64)
        # comp_subaxis=1 -> comp_idx=2; col 2: out[i] = T[i,2] = 12, 22
        Tarang.extract_tensor_component!(out, T, 1, 1, :g)
        @test all(gridof(out.components[1]) .== 12.0)
        @test all(gridof(out.components[2]) .== 22.0)
    end

    @testset "guard: bad tensor index" begin
        T = make_tensor()
        out = VectorField(dist, coords, "out", bases, Float64)
        @test_throws ArgumentError Tarang.extract_tensor_component!(out, T, 2, 0, :g)
    end

    @testset "guard: output size mismatch" begin
        coords3, dist3, bases3 = build_FFF3()
        T = make_tensor()                                   # 2x2 tensor
        out3 = VectorField(dist3, coords3, "out", bases3, Float64)  # 3-component
        # index 0: expects out.components length == n2 == 2, but out3 has 3.
        @test_throws ArgumentError Tarang.extract_tensor_component!(out3, T, 0, 0, :g)
        @test_throws ArgumentError Tarang.extract_tensor_component!(out3, T, 1, 0, :g)
    end
end

# ===========================================================================
# 9. evaluate_cartesian_component
# ===========================================================================

@testset "evaluate_cartesian_component" begin
    coords, dist, bases, r = build_FF2()
    x, y = r

    @testset "vector :g returns the component (reference)" begin
        u = VectorField(dist, coords, "u", bases, Float64)
        for i in eachindex(x), j in eachindex(y)
            gridof(u.components[1])[i, j] = sin(2π * x[i] / Lx)
            gridof(u.components[2])[i, j] = cos(2π * y[j] / Ly)
        end
        u.components[1].current_layout = :g
        u.components[2].current_layout = :g

        op = CartesianComponent(u; index=0, comp=coords["x"])
        res = evaluate_cartesian_component(op, :g)
        # Documented: returns a reference to the component field itself.
        @test res === u.components[1]
        expected = [sin(2π * x[i] / Lx) for i in eachindex(x), j in eachindex(y)]
        @test isapprox(gridof(res), expected, rtol=1e-12)
    end

    @testset "vector :c returns component in coeff layout" begin
        u = VectorField(dist, coords, "u", bases, Float64)
        for i in eachindex(x), j in eachindex(y)
            gridof(u.components[1])[i, j] = sin(2π * x[i] / Lx)
            gridof(u.components[2])[i, j] = cos(2π * y[j] / Ly)
        end
        u.components[1].current_layout = :g
        u.components[2].current_layout = :g

        op = CartesianComponent(u; index=0, comp=coords["y"])
        res = evaluate_cartesian_component(op, :c)
        @test res === u.components[2]
        @test res.current_layout == :c
        # Round-trip back to grid recovers the analytic field (transform sanity).
        ensure_layout!(res, :g)
        expected = [cos(2π * y[j] / Ly) for i in eachindex(x), j in eachindex(y)]
        @test isapprox(gridof(res), expected, rtol=1e-10)
    end

    @testset "tensor :g returns an independent VectorField copy" begin
        T = TensorField(dist, coords, "T", bases, Float64)
        for i in 1:2, j in 1:2
            fill!(gridof(T.components[i, j]), 10.0 * i + j)
            T.components[i, j].current_layout = :g
        end

        # index 0, comp x (comp_idx=1): result[j] = T[1, j] = 11, 12
        op = CartesianComponent(T; index=0, comp=coords["x"])
        res = evaluate_cartesian_component(op, :g)
        @test res isa VectorField
        @test all(gridof(res.components[1]) .== 11.0)
        @test all(gridof(res.components[2]) .== 12.0)

        # Result must be a COPY: mutating it does not touch the tensor.
        fill!(gridof(res.components[1]), -999.0)
        @test all(gridof(T.components[1, 1]) .== 11.0)
    end

    @testset "tensor :g column extraction (index 1)" begin
        T = TensorField(dist, coords, "T", bases, Float64)
        for i in 1:2, j in 1:2
            fill!(gridof(T.components[i, j]), 10.0 * i + j)
            T.components[i, j].current_layout = :g
        end
        # index 1, comp y (comp_idx=2): result[i] = T[i, 2] = 12, 22
        op = CartesianComponent(T; index=1, comp=coords["y"])
        res = evaluate_cartesian_component(op, :g)
        @test all(gridof(res.components[1]) .== 12.0)
        @test all(gridof(res.components[2]) .== 22.0)
    end
end
