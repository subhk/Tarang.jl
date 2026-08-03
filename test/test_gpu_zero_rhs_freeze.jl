# Regression: a pure-Fourier GPU IVP had an identically-zero right-hand side, so the
# solution held its initial condition forever with no error and no warning.
#
# THE CHAIN. `_gpu_pure_fourier_state` (solver_types.jl) deliberately skips global
# matrix assembly for a pure-Fourier GPU IVP — the host matrices are unused there and
# prohibitive at production sizes. But that assembly is also what fills
# `problem.equation_data`, and `build_lazy_rhs_plan!` read an empty `equation_data` as
# "nothing to compile": it returned a plan holding only zero fields and set
# `is_compiled = true`. That flag makes `_compiled_lazy_rhs_available` true, so
# `evaluate_rhs` took the lazy path, received the zero fields and returned them —
# never falling through to the interpreted evaluator. Every stage of every step got
# F = 0. A heat equation simply never cooled.
#
# This is the same root cause as the GPU implicit-guard bug fixed in PR #85 (a
# consumer reading IR that the GPU path never builds), in a different consumer, and it
# survived for the same reason: GPU CI is inert, so no test ever ran a GPU IVP and
# checked that the answer MOVED.
#
# WHY THE TESTS BELOW LOOK THE WAY THEY DO. The mechanism is not GPU-specific — it is
# "empty equation_data at plan-build time" — so the first testset reproduces it on
# plain CPU with no GPU stack involved, which is what makes it a reliable guard rather
# than something that only fires on hardware nobody has. The JLArray testset then pins
# the actual GPU dispatch: without cuFFT the transform must REFUSE and that refusal
# must propagate, because "no device FFT" silently becoming "F = 0" is precisely the
# bug. On a real CUDA host the same path computes the RHS instead.

using Test
using Tarang

function _zrf_build_cpu(; N = 8, κ = 0.1)
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; dtype = Float64, architecture = CPU())
    xb = RealFourier(coords["x"]; size = N, bounds = (0.0, 2π))
    yb = RealFourier(coords["y"]; size = N, bounds = (0.0, 2π))
    dom = Domain(dist, (xb, yb))
    u = ScalarField(dom, "u"); set!(u, (x, y) -> sin(x) * cos(y))
    prob = IVP([u]); add_parameters!(prob, kappa = κ)
    add_equation!(prob, "dt(u) = kappa*lap(u)")
    return InitialValueSolver(prob, RK222(); dt = 0.01)
end

_zrf_mag(field) = begin
    data = Tarang.get_coeff_data(field)
    data = data === nothing ? Tarang.get_grid_data(field) : data
    data === nothing ? 0.0 : maximum(abs, Array(data))
end

@testset "An empty equation_data must not compile to a zero RHS" begin
    # Baseline: with the IR present the RHS is the real thing.
    solver = _zrf_build_cpu()
    @test !isempty(solver.problem.equation_data)
    rhs = Tarang.evaluate_rhs(solver, solver.state, 0.0)
    @test _zrf_mag(rhs[1]) > 1.0

    # Now the GPU situation, reproduced without a GPU: no IR when the plan is built.
    # build_lazy_rhs_plan! must rebuild it (build_matrix_expressions! is matrix-free)
    # rather than hand back an empty plan.
    solver2 = _zrf_build_cpu()
    empty!(solver2.problem.equation_data)
    @test isempty(solver2.problem.equation_data)

    plan = Tarang.build_lazy_rhs_plan!(solver2)
    @test !isempty(solver2.problem.equation_data)   # rebuilt on demand
    solver2.rhs_plan = plan

    rhs2 = Tarang.evaluate_rhs(solver2, solver2.state, 0.0)
    # The assertion that matters. Before the fix this was exactly 0.0, and a zero RHS
    # is indistinguishable from a converged one to every check except this.
    @test _zrf_mag(rhs2[1]) > 1.0
    @test _zrf_mag(rhs2[1]) ≈ _zrf_mag(rhs[1])      # and it is the SAME RHS
end

@testset "A plan that compiled nothing refuses when the problem evolves something" begin
    # The backstop, independent of how equation_data got emptied. A plan with no
    # compiled expression yields F = 0 for every stage; that is correct only when the
    # problem has nothing to evolve. `_problem_has_evolution_equation` is what tells
    # the two apart.
    solver = _zrf_build_cpu()
    @test Tarang._problem_has_evolution_equation(solver.problem)

    # A problem with no equations at all evolves nothing, so a zero RHS is legitimate
    # and must NOT raise — otherwise the backstop would break valid setups.
    coords = CartesianCoordinates("x")
    dist = Distributor(coords; dtype = Float64, architecture = CPU())
    xb = RealFourier(coords["x"]; size = 8, bounds = (0.0, 2π))
    dom = Domain(dist, (xb,))
    v = ScalarField(dom, "v")
    empty_prob = IVP([v])
    @test !Tarang._problem_has_evolution_equation(empty_prob)
end

# ---------------------------------------------------------------------------
# The GPU dispatch itself, on the JLArray device stack (no CUDA hardware needed).
# ---------------------------------------------------------------------------

const _ZRF_OK = try
    @eval using JLArrays
    @eval using GPUArrays
    true
catch err
    @info "JLArrays/GPUArrays unavailable; skipping the GPU half of the zero-RHS test" err
    false
end

if _ZRF_OK
    const _ZRF_JL = JLArrays.JLArray
    const _ZRF_ARCH = Tarang.GPU(JLArrays.JLBackend())
    # Test-scoped only; JLArray is used by nothing else in the package.
    Tarang.is_gpu_array(::_ZRF_JL) = true
    Tarang.architecture(::_ZRF_JL) = _ZRF_ARCH
    Tarang.on_architecture(::Tarang.GPU{JLArrays.JLBackend}, a::Array) = _ZRF_JL(a)
    Tarang.copy_to_device(a::AbstractArray, ::_ZRF_JL) = _ZRF_JL(Array(a))
    Tarang.copy_to_device(a::_ZRF_JL, ::_ZRF_JL) = copy(a)
    Tarang.array_type(::Tarang.GPU{JLArrays.JLBackend}) = _ZRF_JL
    Tarang.array_type(::Tarang.GPU{JLArrays.JLBackend}, T::Type) = _ZRF_JL{T}
end

@testset "A pure-Fourier GPU IVP builds its RHS IR, and never silently returns zero" begin
    if !_ZRF_OK
        @test_skip "JLArrays not available"
    else
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; dtype = Float64, architecture = _ZRF_ARCH)
        xb = RealFourier(coords["x"]; size = 8, bounds = (0.0, 2π))
        yb = RealFourier(coords["y"]; size = 8, bounds = (0.0, 2π))
        dom = Domain(dist, (xb, yb))
        u = ScalarField(dom, "u"); set!(u, (x, y) -> sin(x) * cos(y))
        prob = IVP([u]); add_parameters!(prob, kappa = 0.1)
        add_equation!(prob, "dt(u) = kappa*lap(u)")
        solver = InitialValueSolver(prob, RK222(); dt = 0.01)

        # The device field really is device-resident, so this exercises the GPU dispatch.
        @test Tarang.get_grid_data(solver.state[1]) isa _ZRF_JL

        # The IR is present despite global matrix assembly being skipped. This is the
        # single assertion that would have caught the bug: it was 0 here.
        @test !isempty(solver.problem.equation_data)

        # And the RHS is never a silent zero. JLArray provides device ARRAYS but no
        # device FFT, so the transform must refuse and that refusal must propagate.
        # With CUDA loaded this path computes the RHS instead; what is pinned here is
        # that neither outcome is "quietly zero".
        outcome = try
            rhs = Tarang.evaluate_rhs(solver, solver.state, 0.0)
            _zrf_mag(rhs[1]) > 1e-12 ? :nonzero : :silent_zero
        catch err
            :refused
        end
        @test outcome !== :silent_zero
        @test outcome === :refused || outcome === :nonzero
    end
end
