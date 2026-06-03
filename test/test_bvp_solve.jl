"""
End-to-end boundary-value-problem (BVP) solve tests with manufactured
(analytic) oracle solutions.

Goal: exercise the BVP solver path that was previously never tested against a
real answer:
    - solver_stepping.jl       : solve!/solve_linear!/solve_nonlinear!
    - solver_compiled_rhs.jl   : forcing/expression evaluation
    - subsystems/subproblem_modes.jl + global matrix assembly

Method (independent oracle / method of manufactured solutions):
    Pick u_exact, derive the forcing + BCs analytically, solve the BVP, and
    compare the solved grid to u_exact. Expected values come ONLY from the
    manufactured solution, never from the solver's own output.

------------------------------------------------------------------------------
STATUS (2026-06): The BVP solver path is BROKEN at three independent levels and
cannot solve any manufactured BVP. This file asserts the CORRECT manufactured
answers and wraps the failing reality in `@test_broken`, recording the captured
evidence inline so the bugs can be fixed and the broken markers flipped to
passing. No `src/` files were edited; the module is never mutated.

Confirmed bugs (all in the BVP/EVP solver build + global-matrix assembly):

  Bug A — UNCONDITIONAL CONSTRUCTION CRASH (once validation passes).
    `_build_boundary_value_solver` (src/core/solvers/solver_types.jl:705) and
    `_build_eigenvalue_solver` (:812) call `_solver_type(base.matsolver)`, but
    `_solver_type` is NOT DEFINED anywhere in the module. Construction throws
    `UndefVarError: _solver_type`. The call should be
    `MatSolvers.solver_instance(base.matsolver, L_sparse)` directly:
    `base.matsolver` is already a solver Type from `MatSolvers.get_solver` in
    `SolverBaseData`, and `solver_instance` already calls `get_solver`.

  Bug B — `add_bc!` BCs ARE NEVER MERGED into a BVP equation system.
    `_build_initial_value_solver` calls `_merge_boundary_conditions!(problem)`
    BEFORE `validate_problem`; `_build_boundary_value_solver`
    (solver_types.jl:682-683) only calls `setup_domain!` then `validate_problem`
    — it never merges BCs. So `add_bc!` BCs never become matrix rows for a BVP,
    and validation fails with "equations < variables".

  Bug C — GLOBAL TAU-METHOD MATRIX/FORCING ASSEMBLY IS WRONG (singular system).
    Even after locally working around A + B with a square tau-method system, the
    assembled global L is rank-deficient and `build_forcing_vector`
    (problem_matrices_legacy.jl:8-33) only handles Constant/Zero RHS — and for a
    constant RHS it writes the raw constant into EVERY spectral coefficient slot
    (it should map to the zeroth Chebyshev coefficient of the k=0 Fourier mode
    only). A spatially varying forcing is silently zeroed. Net effect:
    `L \\ F` / SPQR returns nonsense (residual ~2.5e4, |u| ~3.6e20 vs analytic
    max 0.25), so the solve never matches the oracle.
"""

using Test
using Tarang
using LinearAlgebra
using SparseArrays

# ---------------------------------------------------------------------------
# Manufactured oracle for the LINEAR Poisson BVP.
#
#   PDE : Δu = -2     on  z ∈ [0, Lz]   (x-independent; only k=0 Fourier mode)
#   BCs : u(z=0) = 0,  u(z=Lz) = 0
#   Exact:  u_exact(z) = z (Lz - z)
#       Δu_exact = u_exact'' = -2  ✓ ;  u_exact(0) = u_exact(Lz) = 0  ✓
#   Peak: max_z z(Lz - z) = Lz^2 / 4.
# ---------------------------------------------------------------------------
const BVP_LZ = 1.0
const BVP_NX = 4
const BVP_NZ = 16

bvp_u_exact(z) = z * (BVP_LZ - z)

# ===========================================================================
# Bug A: BoundaryValueSolver construction crashes once validation passes.
#
# Dedalus-style over-determined formulation (single variable `u`, BCs added via
# `add_equation!` so they enter `problem.equations`). This passes
# `validate_problem` (equations 3 > variable 1) and reaches the build step that
# calls the undefined `_solver_type`. We capture the REAL exception.
# ===========================================================================
@testset "Linear BVP construction (Bug A: _solver_type undefined)" begin
    coords = CartesianCoordinates("x", "z")
    dist   = Distributor(coords; dtype=Float64, device=CPU())
    xbasis = RealFourier(coords["x"]; size=BVP_NX, bounds=(0.0, 2π))
    zbasis = ChebyshevT(coords["z"];  size=BVP_NZ, bounds=(0.0, BVP_LZ))
    domain = Domain(dist, (xbasis, zbasis))
    u = ScalarField(domain, "u")

    problem = Tarang.LBVP([u])
    add_parameters!(problem; Lz=BVP_LZ)
    Tarang.add_equation!(problem, "Δ(u) = -2")
    Tarang.add_equation!(problem, "u(z=0) = 0")
    Tarang.add_equation!(problem, "u(z=Lz) = 0")

    err = try
        Tarang.BoundaryValueSolver(problem)
        nothing
    catch e
        e
    end

    # Self-validating assertions documenting the exact failure as shipped.
    # (If the source is fixed so `_solver_type` exists, this branch no longer
    #  throws UndefVarError and these guards relax accordingly.)
    bug_a_present = !isdefined(Tarang, :_solver_type)
    if bug_a_present
        @test err isa UndefVarError
        @test err.var === :_solver_type
    else
        @test err === nothing || !(err isa UndefVarError)
    end

    # CORRECT behavior: a usable solver is constructed. Currently broken.
    constructed_ok = !(err isa Exception)
    @test_broken constructed_ok
end

# ===========================================================================
# Bug B: `add_bc!` BCs never reach the equation system for a BVP build, so
# validation fails ("equations < variables") before any matrix is assembled.
# The proper tau-method square system (1 PDE + 2 tau vars, 2 BCs) is what we
# want; with add_bc! the BCs are dropped.
# ===========================================================================
@testset "Linear BVP add_bc! merge (Bug B: BCs dropped for BVP)" begin
    coords = CartesianCoordinates("x", "z")
    dist   = Distributor(coords; dtype=Float64, device=CPU())
    xbasis = RealFourier(coords["x"]; size=BVP_NX, bounds=(0.0, 2π))
    zbasis = ChebyshevT(coords["z"];  size=BVP_NZ, bounds=(0.0, BVP_LZ))
    domain = Domain(dist, (xbasis, zbasis))

    u      = ScalarField(domain, "u")
    tau_u1 = ScalarField(dist, "tau_u1", (xbasis,), Float64)
    tau_u2 = ScalarField(dist, "tau_u2", (xbasis,), Float64)
    _, ez = unit_vector_fields(coords, dist)
    lift_basis = derivative_basis(zbasis, 1)
    τ_lift(A) = lift(A, lift_basis, -1)
    grad_u = grad(u) + ez * τ_lift(tau_u1)

    problem = Tarang.LBVP([u, tau_u1, tau_u2])
    add_parameters!(problem; Lz=BVP_LZ, grad_u=grad_u, τ_lift=τ_lift)
    Tarang.add_equation!(problem, "div(grad_u) + τ_lift(tau_u2) = -2")
    Tarang.add_bc!(problem, "u(z=0) = 0")
    Tarang.add_bc!(problem, "u(z=Lz) = 0")

    # add_bc! recorded the BCs as raw strings, but they are NOT yet equations.
    @test length(problem.boundary_conditions) == 2
    @test length(problem.equations) == 1                 # only the PDE so far
    @test length(problem.variables) == 3                 # u, tau_u1, tau_u2
    @test isdefined(Tarang, :_merge_boundary_conditions!)  # the helper exists

    # Construction SHOULD merge BCs (as the IVP build does) and succeed.
    # Currently it throws an ArgumentError at validation because BCs were not
    # merged into `problem.equations`.
    err = try
        Tarang.BoundaryValueSolver(problem)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("less than number of variables", sprint(showerror, err))

    bvp_build_succeeds = !(err isa Exception)
    @test_broken bvp_build_succeeds
end

# ===========================================================================
# Bug C: with Bugs A + B worked around (BCs merged manually; matrices solved
# directly without `BoundaryValueSolver` so no module mutation is needed), the
# global tau-method matrix is singular and the forcing vector is mis-assembled,
# so the linear solve does NOT recover the manufactured solution.
#
# Exercises the real assembly/solve paths:
#   build_matrices -> build_forcing_vector -> (rank-revealing) solve ->
#   copy_solution_to_fields! -> grid transform, then compares to u_exact.
# ===========================================================================
@testset "Linear BVP Poisson manufactured oracle (Bug C: singular system)" begin
    coords = CartesianCoordinates("x", "z")
    dist   = Distributor(coords; dtype=Float64, device=CPU())
    xbasis = RealFourier(coords["x"]; size=BVP_NX, bounds=(0.0, 2π))
    zbasis = ChebyshevT(coords["z"];  size=BVP_NZ, bounds=(0.0, BVP_LZ))
    domain = Domain(dist, (xbasis, zbasis))

    u      = ScalarField(domain, "u")
    tau_u1 = ScalarField(dist, "tau_u1", (xbasis,), Float64)
    tau_u2 = ScalarField(dist, "tau_u2", (xbasis,), Float64)
    _, ez = unit_vector_fields(coords, dist)
    lift_basis = derivative_basis(zbasis, 1)
    τ_lift(A) = lift(A, lift_basis, -1)
    grad_u = grad(u) + ez * τ_lift(tau_u1)

    problem = Tarang.LBVP([u, tau_u1, tau_u2])
    add_parameters!(problem; Lz=BVP_LZ, grad_u=grad_u, τ_lift=τ_lift)
    Tarang.add_equation!(problem, "div(grad_u) + τ_lift(tau_u2) = -2")
    Tarang.add_bc!(problem, "u(z=0) = 0")
    Tarang.add_bc!(problem, "u(z=Lz) = 0")

    # Bug-B workaround: merge BCs into the equation system exactly as the IVP
    # build path does, so we can reach matrix assembly.
    Tarang.setup_domain!(problem)
    Tarang._merge_boundary_conditions!(problem)
    @test length(problem.equations) == length(problem.variables)  # 3 == 3

    # Assemble the global system the way the BVP build does.
    L, M, F = Tarang.build_matrices(problem)
    Lsp = sparse(L)
    Fc  = ComplexF64.(F)

    # --- Independent-oracle target on the actual Chebyshev grid ---
    _, z = local_grids(dist, xbasis, zbasis)
    u_exact_grid = [bvp_u_exact(zk) for _ in 1:BVP_NX, zk in z]
    exact_peak = maximum(abs, u_exact_grid)
    # The Chebyshev-Gauss-Lobatto grid (even N) does not include z = Lz/2, so
    # the discrete max is just below the continuous max Lz^2/4 = 0.25.
    @test exact_peak < BVP_LZ^2 / 4
    @test isapprox(exact_peak, BVP_LZ^2 / 4; rtol=2e-2)

    # --- The CORRECT solve: L x = F should reproduce u_exact. ---
    # The global L is rank-deficient (singular), so `L \\ F` blows up. Use the
    # codebase's rank-revealing SPQR (minimum-norm) solver, then measure error.
    spqr  = Tarang.MatSolvers.SPQRSolver(Lsp)
    x     = Tarang.MatSolvers.solve(spqr, Fc)
    resid = norm(Lsp * x - Fc)

    Tarang.copy_solution_to_fields!([u, tau_u1, tau_u2], x)
    ensure_layout!(u, :g)
    u_num = real.(Tarang.get_grid_data(u))

    solved_peak = maximum(abs, u_num)
    max_err     = maximum(abs, u_num .- u_exact_grid)
    L_rank      = rank(Matrix(Lsp))

    @info "BVP Poisson oracle evidence" L_size=size(Lsp) L_rank=L_rank residual=resid exact_peak=exact_peak solved_peak=solved_peak max_err=max_err

    # Positive assertions documenting the brokenness (self-validating evidence):
    @test L_rank < size(Lsp, 1)      # global L is singular (rank-deficient)
    @test resid > 1.0                # residual is huge, not ~0
    @test solved_peak > 1e6          # solution blows up vs analytic peak ~0.25

    # CORRECT manufactured-solution assertion (currently broken):
    bvp_matches_oracle = max_err < 1e-6
    @test_broken bvp_matches_oracle
end

# ===========================================================================
# Nonlinear BVP (solve_nonlinear! + Frechet Jacobian path).
#
#   PDE : Δu - u^2 = g(z),  manufactured u_exact = z(Lz - z):
#         g = Δu_exact - u_exact^2 = -2 - (z(Lz - z))^2
#   BCs : u(z=0) = u(z=Lz) = 0
#
# This sits on top of the same broken construction/assembly path (Bug A at
# build, Bug B for add_bc!, Bug C for the linear/Jacobian assembly), so the
# NLBVP cannot be solved to the oracle. We document the intended oracle and
# mark the converged-solution result broken, capturing the construction error.
# ===========================================================================
@testset "Nonlinear BVP manufactured oracle (blocked by Bug A/C)" begin
    coords = CartesianCoordinates("x", "z")
    dist   = Distributor(coords; dtype=Float64, device=CPU())
    xbasis = RealFourier(coords["x"]; size=BVP_NX, bounds=(0.0, 2π))
    zbasis = ChebyshevT(coords["z"];  size=BVP_NZ, bounds=(0.0, BVP_LZ))
    domain = Domain(dist, (xbasis, zbasis))

    u = ScalarField(domain, "u")

    # Over-determined Dedalus-style formulation so validation passes and we
    # reach the build step (which then hits Bug A: _solver_type).
    problem = Tarang.NLBVP([u])
    add_parameters!(problem; Lz=BVP_LZ)
    Tarang.add_equation!(problem, "Δ(u) - u*u = -2")
    Tarang.add_equation!(problem, "u(z=0) = 0")
    Tarang.add_equation!(problem, "u(z=Lz) = 0")

    # Good Newton initial guess: seed u ≈ exact so convergence isn't the issue.
    set!(u, (x, z) -> bvp_u_exact(z))

    err = try
        Tarang.BoundaryValueSolver(problem)
        nothing
    catch e
        e
    end
    bug_a_present = !isdefined(Tarang, :_solver_type)
    if bug_a_present
        @test err isa UndefVarError       # NLBVP build hits the same crash
        @test err.var === :_solver_type
    end

    # CORRECT behavior: Newton converges to the manufactured solution. Broken.
    nlbvp_solves_oracle = false
    if !(err isa Exception)
        solver = Tarang.BoundaryValueSolver(problem)
        nlbvp_solves_oracle = try
            Tarang.solve!(solver)
            ensure_layout!(u, :g)
            _, z = local_grids(dist, xbasis, zbasis)
            u_num = real.(Tarang.get_grid_data(u))
            u_exact_grid = [bvp_u_exact(zk) for _ in 1:BVP_NX, zk in z]
            maximum(abs, u_num .- u_exact_grid) < 1e-5
        catch
            false
        end
    end
    @test_broken nlbvp_solves_oracle
end
