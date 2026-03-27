"""
Test suite for the Problem and Solver system.

Tests IVP/LBVP/EVP creation, equation and BC management, parameter handling,
BC helper functions, parse_bc_string, namespace management, and basic solver
construction with diagnose.
"""

using Test
using Tarang

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

function make_periodic_field(name="u")
    coords = CartesianCoordinates("x")
    dist   = Distributor(coords; mesh=(1,), dtype=Float64)
    basis  = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
    ScalarField(dist, name, (basis,), Float64)
end

function make_channel_fields()
    domain = ChannelDomain(8, 8; Lx=2π, Lz=1.0, dealias=1.0)
    p = ScalarField(domain, "p")
    T = ScalarField(domain, "T")
    u = VectorField(domain, "u")
    return domain, p, T, u
end

# ===========================================================================
@testset "Problem & Solver System" begin
# ===========================================================================

# ---- 1. IVP creation -----------------------------------------------------
@testset "IVP creation" begin
    f = make_periodic_field("u")
    prob = IVP([f])
    @test prob isa IVP
    @test length(prob.variables) == 1
    @test prob.variables[1] === f
    @test isempty(prob.equations)
    @test isempty(prob.boundary_conditions)

    # IVP with multiple variables
    domain, p, T, u = make_channel_fields()
    ux, uz = u.components
    prob2 = IVP(Tarang.Operand[p, T, ux, uz])
    @test length(prob2.variables) == 4
end

# ---- 2. LBVP creation ----------------------------------------------------
@testset "LBVP creation" begin
    f = make_periodic_field("phi")
    prob = LBVP([f])
    @test prob isa LBVP
    @test length(prob.variables) == 1
    @test isempty(prob.equations)
end

# ---- 3. EVP creation -----------------------------------------------------
@testset "EVP creation" begin
    f = make_periodic_field("psi")
    prob = EVP([f]; eigenvalue=:sigma)
    @test prob isa EVP
    @test prob.eigenvalue === :sigma

    # EVP without eigenvalue keyword
    prob2 = EVP([f])
    @test prob2.eigenvalue === nothing
end

# ---- 4. add_equation! with string equations -------------------------------
@testset "add_equation! with string equations" begin
    f = make_periodic_field("u")
    prob = IVP([f])

    add_equation!(prob, "dt(u) = 0")
    @test length(prob.equations) == 1
    @test prob.equations[1] == "dt(u) = 0"

    # Multiple equations
    f2 = make_periodic_field("v")
    prob2 = IVP(Tarang.Operand[f, f2])
    add_equation!(prob2, "dt(u) - nu*lap(u) = 0")
    add_equation!(prob2, "dt(v) = -u")
    @test length(prob2.equations) == 2
end

# ---- 5. add_parameters! (kwargs style) -----------------------------------
@testset "add_parameters!" begin
    f = make_periodic_field("u")
    prob = IVP([f])

    add_parameters!(prob, nu=1e-3, kappa=1e-4, Ra=1e6)
    @test prob.namespace["nu"] == 1e-3
    @test prob.namespace["kappa"] == 1e-4
    @test prob.namespace["Ra"] == 1e6

    # Overwrite existing parameter
    add_parameters!(prob, nu=2e-3)
    @test prob.namespace["nu"] == 2e-3
end

# ---- 6. add_substitution! (deprecated) -----------------------------------
@testset "add_substitution! (deprecated)" begin
    f = make_periodic_field("u")
    prob = IVP([f])

    # add_substitution! uses Base.depwarn internally, so it may emit a
    # deprecation warning depending on --depwarn flag.  We only verify that
    # the function still works correctly (sets the namespace value).
    Tarang.add_substitution!(prob, "nu", 1e-3)
    @test prob.namespace["nu"] == 1e-3

    # Calling it again overwrites
    Tarang.add_substitution!(prob, "nu", 2e-3)
    @test prob.namespace["nu"] == 2e-3

    # add_parameters! calls add_substitution! with _internal=true (no depwarn)
    add_parameters!(prob, kappa=1e-4)
    @test prob.namespace["kappa"] == 1e-4
end

# ---- 7. add_bc! with string BCs ------------------------------------------
@testset "add_bc! with string BCs" begin
    domain, p, T, u = make_channel_fields()
    ux, uz = u.components
    prob = IVP(Tarang.Operand[p, T, ux, uz])

    add_bc!(prob, "u(z=0) = 0")
    @test length(prob.boundary_conditions) == 1
    @test prob.boundary_conditions[1] == "u(z=0) = 0"

    add_bc!(prob, "integ(p) = 0")
    @test length(prob.boundary_conditions) == 2
end

# ---- 8. add_bc! with BC objects (dirichlet_bc, neumann_bc) ----------------
@testset "add_bc! with BC objects" begin
    domain, p, T, u = make_channel_fields()
    ux, uz = u.components
    prob = IVP(Tarang.Operand[p, T, ux, uz])

    # Dirichlet BC via object
    bc_d = dirichlet_bc("T", "z", 0.0, 1.0)
    @test bc_d isa DirichletBC
    @test bc_d.field == "T"
    @test bc_d.value == 1.0
    ret = add_bc!(prob, bc_d)
    @test ret === bc_d
    @test length(prob.bc_manager.conditions) >= 1

    # Neumann BC via object
    bc_n = neumann_bc("T", "z", 1.0, 0.0)
    @test bc_n isa NeumannBC
    @test bc_n.coordinate == "z"
    @test bc_n.position == 1.0
    add_bc!(prob, bc_n)
    @test length(prob.bc_manager.conditions) >= 2

    # Both should also appear in legacy string list
    @test length(prob.boundary_conditions) >= 2
end

# ---- 9. BC helper functions -----------------------------------------------
@testset "BC helpers on ChannelDomain" begin
    domain, p, T, u = make_channel_fields()
    ux, uz = u.components
    prob = IVP(Tarang.Operand[p, T, ux, uz])

    # no_slip! returns DirichletBC with value 0
    bc1 = no_slip!(prob, "u", "z", 0.0)
    @test bc1 isa DirichletBC
    @test bc1.value == 0.0
    @test bc1.field == "u"

    bc2 = no_slip!(prob, "u", "z", 1.0)
    @test bc2.position == 1.0

    # fixed_value! returns DirichletBC with specified value
    bc3 = fixed_value!(prob, "T", "z", 0.0, 1.0)
    @test bc3 isa DirichletBC
    @test bc3.value == 1.0

    bc4 = fixed_value!(prob, "T", "z", 1.0, 0.0)
    @test bc4.value == 0.0

    # free_slip! returns NeumannBC with value 0
    bc5 = free_slip!(prob, "u", "z", 0.0)
    @test bc5 isa NeumannBC
    @test bc5.value == 0.0

    # insulating! returns NeumannBC with value 0
    bc6 = insulating!(prob, "T", "z", 1.0)
    @test bc6 isa NeumannBC
    @test bc6.value == 0.0
    @test bc6.coordinate == "z"

    # All BCs registered
    @test length(prob.bc_manager.conditions) == 6
end

# ---- 10. parse_bc_string --------------------------------------------------
@testset "parse_bc_string" begin
    # Standard Dirichlet: field(coord=pos) = value
    fname, coord, pos, val = Tarang.parse_bc_string("u(z=0) = 0")
    @test fname == "u"
    @test coord == "z"
    @test pos == 0.0
    @test val == 0.0

    # Float position and value
    fname, coord, pos, val = Tarang.parse_bc_string("T(z=1.0) = 0.5")
    @test fname == "T"
    @test pos == 1.0
    @test val == 0.5

    # Expression value kept as string
    fname, coord, pos, val = Tarang.parse_bc_string("T(z=0) = sin(x)")
    @test val == "sin(x)"

    # Negative position
    fname, coord, pos, val = Tarang.parse_bc_string("u(z=-1) = 0")
    @test pos == -1.0

    # Invalid format throws
    @test_throws ArgumentError Tarang.parse_bc_string("nonsense string")
    @test_throws ArgumentError Tarang.parse_bc_string("u = 0")
end

# ---- 11. Problem namespace management ------------------------------------
@testset "Namespace management" begin
    domain, p, T, u = make_channel_fields()
    ux, uz = u.components
    prob = IVP(Tarang.Operand[p, T, ux, uz])

    # Variables are registered by name
    @test haskey(prob.namespace, "p")
    @test haskey(prob.namespace, "T")

    # Scalar components of vector field registered by their component name
    @test haskey(prob.namespace, ux.name)
    @test haskey(prob.namespace, uz.name)
    @test prob.namespace["p"] === p

    # Parameters added later appear in namespace
    add_parameters!(prob, Ra=1e6)
    @test prob.namespace["Ra"] == 1e6

    # Variable names take highest priority (cannot be overwritten by parameters)
    add_parameters!(prob, p=42.0)
    # After add_parameters! the namespace key "p" holds 42.0 because
    # add_substitution! just assigns -- but the original field is still in variables
    @test prob.namespace["p"] == 42.0  # most-recent write wins
end

# ---- 12. InitialValueSolver creation -------------------------------------
@testset "InitialValueSolver creation" begin
    # Minimal: single periodic field
    f = make_periodic_field("u")
    prob = IVP([f])
    add_equation!(prob, "dt(u) = 0")

    solver = InitialValueSolver(prob, RK111(); dt=1e-3)
    @test solver isa InitialValueSolver
    @test solver.problem === prob
    @test solver.dt ≈ 1e-3
    @test solver.sim_time == 0.0
    @test solver.iteration == 0
    @test length(solver.state) >= 1

    # With RK222 timestepper
    f2 = make_periodic_field("v")
    prob2 = IVP([f2])
    add_equation!(prob2, "dt(v) = 0")

    solver2 = InitialValueSolver(prob2, RK222(); dt=5e-4)
    @test solver2.dt ≈ 5e-4
    ts = solver2.timestepper
    @test ts isa RK222
end

# ---- 13. diagnose(solver) does not error ----------------------------------
@testset "diagnose(solver)" begin
    f = make_periodic_field("u")
    prob = IVP([f])
    add_equation!(prob, "dt(u) = 0")
    solver = InitialValueSolver(prob, RK111(); dt=1e-3)

    # diagnose prints to stdout; capture it and verify key sections appear
    output = let buf = IOBuffer()
        redirect_stdout(buf) do
            diagnose(solver)
        end
        String(take!(buf))
    end
    @test occursin("Solver Diagnostics", output)
    @test occursin("Timestepper", output)
    @test occursin("State fields", output)
end

# ===========================================================================
end  # top-level testset
# ===========================================================================
