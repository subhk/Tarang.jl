"""
`EquationIR` is a struct wearing an `AbstractDict` interface, and the seam between
those two is where bugs live. This pins the seam.

`EquationIR` holds the six common equation slots — `M`, `L`, `F`, `F_expr`,
`lhs`, `equation_size` — as real struct fields, and everything else in a
`metadata` Dict. The `AbstractDict` shim exists so older code can keep indexing
it with strings. The shim is not a faithful `Dict`, and both differences have
already produced live defects:

  1. `haskey` answers "is this a known slot", not "was anything stored". Every
     canonical key reports `true` forever, so any guard of the form
     `if haskey(eq_data, "lhs")` is unconditional and `if !haskey(...)` is dead —
     for `EquationIR` input. `is_equation_valid` still carries such a guard; it
     is meaningful only for the plain-`Dict` callers it also serves, and is
     documented in place rather than changed, because a present-but-`nothing`
     LHS is accepted downstream.

  2. Keys are case-sensitive and unaliased. `"lhs"` is a field, `"rhs"` is
     metadata, and `"LHS"`/`"RHS"` are neither — nothing writes them.
     `_get_residual_expression` read `"LHS"`/`"RHS"`, so its subtraction branch
     was unreachable and every Newton Jacobian fell back to `F` alone, dropping
     the implicit `L` contribution with no error.

Both are the same underlying shape: a string key that does not exist yields a
default instead of an error. That is why new code should read the fields
directly — `ir.linear`, not `get(ir, "L", nothing)` — and why the tests below
assert the shim's exact behaviour rather than assuming Dict semantics.
"""

using Test
using Tarang

@testset "EquationIR canonical slots are fields" begin
    ir = Tarang.EquationIR()

    # Unset canonical slots read as `nothing` through both routes.
    @test ir.linear === nothing
    @test ir.mass === nothing
    @test get(ir, "L", nothing) === nothing
    @test ir["L"] === nothing

    ir["L"] = 42
    @test ir.linear == 42          # the string key wrote the FIELD
    @test get(ir, "L", nothing) == 42

    ir.mass = 7                     # and the field is visible through the shim
    @test ir["M"] == 7

    # equation_size is an Int slot with a typed setter.
    ir["equation_size"] = 3
    @test ir.equation_size == 3
    @test_throws ArgumentError ir["equation_size"] = "not an integer"
end

@testset "EquationIR haskey means 'known slot', not 'assigned'" begin
    ir = Tarang.EquationIR()

    # THE TRAP. Every canonical key is present on an entirely empty IR, because
    # the slots are struct fields. A guard written as `haskey` is therefore
    # unconditional, which is how the "Missing LHS expression" validation became
    # unreachable. Pin it so the behaviour is a documented decision, not a
    # surprise to the next person who writes such a guard.
    for key in ("M", "L", "F", "F_expr", "lhs", "equation_size")
        @test haskey(ir, key)
        @test key in keys(ir)
    end
    @test get(ir, "lhs", nothing) === nothing   # ... yet nothing was ever stored

    # metadata keys answer honestly, the way a Dict does.
    @test !haskey(ir, "equation_index")
    ir["equation_index"] = 2
    @test haskey(ir, "equation_index")
    @test ir.metadata["equation_index"] == 2

    # So the correct existence test for a canonical slot is on the VALUE.
    @test get(ir, "lhs", nothing) === nothing
    ir["lhs"] = 1
    @test get(ir, "lhs", nothing) !== nothing
end

@testset "EquationIR keys are case-sensitive and unaliased" begin
    ir = Tarang.EquationIR()
    ir["lhs"] = "the lhs"

    # "LHS" is NOT the canonical slot: it lands in metadata, so reading it back
    # returns the default and reports absent. This is exactly what made
    # `_get_residual_expression` silently useless.
    @test get(ir, "LHS", nothing) === nothing
    @test !haskey(ir, "LHS")
    @test ir.lhs == "the lhs"

    # "rhs" is legitimate metadata, not a canonical field — the pair is
    # deliberately asymmetric, so neither name should be assumed.
    @test !haskey(ir, "rhs")
    ir["rhs"] = "the rhs"
    @test ir.metadata["rhs"] == "the rhs"
    @test ir.lhs == "the lhs"       # writing rhs did not disturb the lhs field
end

@testset "residual expression uses the keys the parser writes" begin
    # Regression for the defect above: `build_matrix_expressions!` writes "lhs"
    # and "rhs", so the residual must be built from those. Reading the uppercase
    # spellings returned `0`-or-`F` forever.
    ir = Tarang.EquationIR()
    ir["lhs"] = 5.0
    ir["rhs"] = 2.0
    res = Tarang._get_residual_expression(ir)
    @test isa(res, Tarang.SubtractOperator)
    @test res.left == 5.0
    @test res.right == 2.0

    # With no lhs/rhs it falls back to F, and with nothing at all it is 0 —
    # tested by value, since `haskey(ir, "F")` is true on an empty IR.
    only_f = Tarang.EquationIR()
    only_f["F"] = 9.0
    @test Tarang._get_residual_expression(only_f) == 9.0
    @test Tarang._get_residual_expression(Tarang.EquationIR()) == 0
end

@testset "EquationIR round-trips a legacy Dict" begin
    # Older code paths push plain Dicts into `Vector{EquationIR}`, which converts.
    ir = convert(Tarang.EquationIR, Dict("L" => 1, "condition" => "always"))
    @test ir isa Tarang.EquationIR
    @test ir.linear == 1                       # canonical key -> field
    @test ir.metadata["condition"] == "always" # unknown key -> metadata
end
