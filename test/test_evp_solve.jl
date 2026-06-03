"""
End-to-end eigenvalue-problem (EVP) solve tests with an analytic oracle.

The EVP solver was rehabilitated 2026-06-04 to solve PER-FOURIER-MODE (one square
tau subproblem per separable mode), mirroring the BVP solver and Dedalus. Fixes:
add_bc! BCs merged in the EVP build (same Bug B as the BVP); matrix-coupling
configured so build_subsystems creates per-mode subproblems; `solve!` rewritten to
solve the generalized eigenproblem `L_sp v = λ M_sp v` on the SQUARE per-subproblem
matrices (the global L/M are rank-deficient → Arpack threw SingularException) and
to filter spurious eigenvalues from the singular (zero-row) mass matrix.

Oracle — the 1D diffusion eigenproblem on a Chebyshev interval:
    dt(u) = Δu,   u(z=0) = u(z=Lz) = 0
The growth rates are σ_n = -(nπ/Lz)², i.e. |σ_n| = (nπ/Lz)² (n = 1,2,3,...), with
eigenfunctions sin(nπz/Lz). These are the eigenvalues of the Dirichlet Laplacian.

Uniquely-prefixed names (evp_*) — the full suite shares the Main namespace.
"""

using Test
using Tarang

const evp_Lz = 1.0
const evp_Nz = 32
evp_oracle(n) = (n * π / evp_Lz)^2          # |σ_n| for the Dirichlet Laplacian

function evp_build_problem()
    coords = CartesianCoordinates("z")
    dist   = Distributor(coords; dtype=Float64, device=CPU())
    zb = ChebyshevT(coords["z"]; size=evp_Nz, bounds=(0.0, evp_Lz))
    dom = Domain(dist, (zb,))
    u    = ScalarField(dom, "u")
    tau1 = ScalarField(dist, "tau1", (), Float64)
    tau2 = ScalarField(dist, "tau2", (), Float64)
    lb2  = derivative_basis(zb, 2)
    prob = Tarang.EVP([u, tau1, tau2]; eigenvalue=:σ)
    add_parameters!(prob; Lz=evp_Lz, l1=lift(tau1, lb2, -1), l2=lift(tau2, lb2, -2))
    Tarang.add_equation!(prob, "dt(u) - Δ(u) - l1 - l2 = 0")
    Tarang.add_bc!(prob, "u(z=0) = 0")
    Tarang.add_bc!(prob, "u(z=Lz) = 0")
    return prob
end

@testset "EVP solver" begin
    @testset "per-mode build: square tau subproblems, finite-rank M" begin
        prob = evp_build_problem()
        solver = Tarang.EigenvalueSolver(prob; nev=4, which=:SM)
        @test !isempty(solver.subproblems)
        for sp in solver.subproblems
            sp.L_min === nothing && continue
            # square (Nz + n_tau); M loses the 2 algebraic BC rows (zero mass).
            @test size(sp.L_min, 1) == size(sp.L_min, 2)
            @test size(sp.M_min) == size(sp.L_min)
        end
    end

    @testset "Diffusion eigenvalues match Dirichlet-Laplacian oracle" begin
        prob = evp_build_problem()
        solver = Tarang.EigenvalueSolver(prob; nev=5, which=:SM)
        λ, v = Tarang.solve!(solver)

        @test length(λ) == 5
        # smallest-magnitude eigenvalues are the lowest Dirichlet modes (nπ/Lz)².
        mags = sort(abs.(λ))
        for n in 1:5
            @test isapprox(mags[n], evp_oracle(n); rtol=1e-6)
        end
        # eigenvectors returned in the (single) subproblem space, one column each.
        @test size(v, 2) == 5
    end

    @testset "spurious eigenvalues filtered (no Inf/NaN returned)" begin
        prob = evp_build_problem()
        solver = Tarang.EigenvalueSolver(prob; nev=8, which=:SM)
        λ, _ = Tarang.solve!(solver)
        @test all(isfinite, λ)
        @test all(x -> abs(x) < 1e10, λ)
    end
end
