"""
The single-GPU implicit-operator guard must actually FIRE on device-resident fields.

`_check_gpu_implicit_compatibility!` (dispatch.jl) exists because a pure-Fourier GPU IVP
builds no global matrix and no subproblems, so a standard IMEX / multistep / ETD scheme
falls through to a fully-explicit step and drops the implicit `L` — a heat equation runs
inviscid, with no error.

Its final gate is `_problem_has_implicit_linear_term`, which reads `problem.equation_data`.
That IR is produced by `build_matrix_expressions!`, which runs as part of the global-matrix
assembly — *the very step the GPU path skips*. So on GPU `equation_data` was EMPTY, the
detector returned false, the guard exempted the problem, and the operator was dropped
silently. Measured before the fix, on the JLArray device stack:

    n(equations) = 1   n(equation_data) = 0   _problem_has_implicit_linear_term = false
    RK222      + `∂t(u) - 0.5*lap(u) = 0`  -> completed, no warning, no error
    ETD_RK222  + same                      -> "@warn requires L_matrix … falling back to
                                              RK222", then completed

test_gpu_implicit_guard.jl pins the CPU-side halves and notes that the guard firing "needs
a GPU field … asserted in test_gpu_timesteppers.jl on a GPU node" — which never runs,
because there is no GPU CI. `JLArray` (GPUArrays' CPU-backed reference device array) makes
`is_gpu_array` true and drives the identical dispatch, so the firing half is testable here.

Uniquely-prefixed names (gijl_*) — the full suite shares the Main namespace.
"""

using Test
using Tarang

const _GIJL_OK = try
    @eval using JLArrays
    @eval using GPUArrays
    true
catch err
    @info "JLArrays/GPUArrays unavailable; skipping GPU implicit-guard firing test" err
    false
end

if _GIJL_OK
    const _GIJL = JLArrays.JLArray
    const _GIJL_ARCH = Tarang.GPU(JLArrays.JLBackend())
    # Test-scoped only; JLArray is used by nothing else in the package.
    Tarang.is_gpu_array(::_GIJL) = true
    Tarang.architecture(::_GIJL) = _GIJL_ARCH
    Tarang.on_architecture(::Tarang.GPU{JLArrays.JLBackend}, a::Array) = _GIJL(a)
    Tarang.copy_to_device(a::AbstractArray, ::_GIJL) = _GIJL(Array(a))
    Tarang.copy_to_device(a::_GIJL, ::_GIJL) = copy(a)
    Tarang.array_type(::Tarang.GPU{JLArrays.JLBackend}) = _GIJL
    Tarang.array_type(::Tarang.GPU{JLArrays.JLBackend}, T::Type) = _GIJL{T}
end

@testset "GPU implicit-operator guard fires on device fields (JLArray)" begin
    if !_GIJL_OK
        @test_skip "JLArrays not available"
    else
        GPUArrays.allowscalar(false)

        function gijl_solver(eqn, ts)
            coords = CartesianCoordinates("x")
            dist = Distributor(coords; dtype=Float64, device=_GIJL_ARCH)
            dom = Domain(dist, (RealFourier(coords["x"]; size=32, bounds=(0.0, 2π)),))
            u = ScalarField(dom, "u")
            ensure_layout!(u, :g)
            prob = IVP([u])
            Tarang.add_equation!(prob, eqn)
            return InitialValueSolver(prob, ts; dt=1e-3)
        end

        @testset "implicit term is detected even though the GPU path builds no matrices" begin
            s = gijl_solver("∂t(u) - 0.5*lap(u) = 0", RK222())
            @test Tarang._distributed_field_path_reason(s.state) === :gpu
            @test Tarang.compiled_subproblems(s.problem) === nothing
            @test Tarang._problem_has_implicit_linear_term(s)
        end

        @testset "non-diagonal schemes refuse instead of dropping the operator" begin
            for ts in (RK222(), SBDF2(), ETD_RK222())
                s = gijl_solver("∂t(u) - 0.5*lap(u) = 0", ts)
                @test_throws ErrorException step!(s, 1e-3)
            end
        end

        # JLArray emulates device ARRAYS, not a device FFT: `_gpu_forward_transform_backend!`
        # demands CUDA.jl, so a step that reaches a transform raises "GPU forward transform
        # backend is unavailable". That is the emulation's limit, not the guard's verdict —
        # so for the cases that must NOT be refused, assert on which error appears rather
        # than on the step completing.
        gijl_guard_refused(s) = begin
            msg = try
                step!(s, 1e-3)
                ""
            catch e
                sprint(showerror, e)
            end
            occursin("cannot treat an implicit", msg)
        end

        @testset "a genuinely explicit equation is not refused" begin
            # Diffusion on the RHS is explicit; there is nothing to drop, so the guard
            # must stay quiet. A guard that fired here would break correct GPU runs.
            s = gijl_solver("∂t(u) = 0.5*lap(u)", RK222())
            @test !Tarang._problem_has_implicit_linear_term(s)
            @test !gijl_guard_refused(s)
        end

        @testset "diagonal-IMEX schemes remain exempt" begin
            # These solve the diagonal Fourier operator per mode on-device, so they are
            # the answer the guard's error message points users to. The implicit term is
            # present, but the guard must not refuse them.
            s = gijl_solver("∂t(u) - 0.5*lap(u) = 0", DiagonalIMEX_RK222())
            @test Tarang._problem_has_implicit_linear_term(s)
            @test !gijl_guard_refused(s)
        end
    end
end
