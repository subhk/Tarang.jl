# JLArray boundary-test support. FFT and LU explicitly use host stand-ins;
# field storage, boundary operations, stage arithmetic, and scatter stay on JLArray.
if _BCJL_OK
    const _BCJL = JLArrays.JLArray
    const _BCJL_ARCH = Tarang.GPU(JLArrays.JLBackend())
    # Test-scoped only; JLArray is used by nothing else in the package. The same
    # five methods as the other JLArray test files, plus the identity/upload
    # methods the lazy RHS and dealiasing paths call on device data.
    Tarang.is_gpu_array(::_BCJL) = true
    Tarang.architecture(::_BCJL) = _BCJL_ARCH
    Tarang.on_architecture(::Tarang.GPU{JLArrays.JLBackend}, a::Array) = _BCJL(a)
    Tarang.on_architecture(::Tarang.GPU{JLArrays.JLBackend}, a::_BCJL) = a
    Tarang.on_architecture(::Tarang.GPU{JLArrays.JLBackend}, a::AbstractArray) = _BCJL(Array(a))
    Tarang.copy_to_device(a::AbstractArray, ::_BCJL) = _BCJL(Array(a))
    Tarang.copy_to_device(a::_BCJL, ::_BCJL) = copy(a)
    Tarang.array_type(::Tarang.GPU{JLArrays.JLBackend}) = _BCJL
    Tarang.array_type(::Tarang.GPU{JLArrays.JLBackend}, T::Type) = _BCJL{T}

    # ---- cuFFT stand-in: a CPU twin field transformed by Tarang's CPU chain ----
    const _BCJL_TWINS = Dict{Any, Any}()
    function bcjl_twin(field)
        get!(_BCJL_TWINS, (objectid(field.bases), field.dtype)) do
            cdist = Distributor(field.dist.coordsys; dtype=field.dtype, device=CPU())
            ScalarField(Domain(cdist, field.bases), "bcjl_twin_" * field.name)
        end
    end
    function bcjl_sync_scales!(twin, field)
        twin.scales == field.scales && return
        twin.current_layout = :c
        Tarang.preset_scales!(twin, field.scales)
    end
    # Copy `src` into the buffer selected by `getter`/`setter` on `dst`,
    # reallocating (via `make`) only when the shape or eltype differ.
    function bcjl_copy_into!(getter, setter, make, dst, src)
        buf = getter(dst)
        if buf === nothing || size(buf) != size(src) || eltype(buf) != eltype(src)
            setter(dst, make(src))
        else
            copyto!(buf, src)
        end
    end
    function Tarang._gpu_forward_transform_backend!(::Tarang.GPU{JLArrays.JLBackend},
                                                    field::Tarang.ScalarField)
        twin = bcjl_twin(field)
        bcjl_sync_scales!(twin, field)
        bcjl_copy_into!(Tarang.get_grid_data, Tarang.set_grid_data!, copy, twin,
                       Array(Tarang.get_grid_data(field)))
        twin.current_layout = :g
        Tarang.forward_transform!(twin)
        bcjl_copy_into!(Tarang.get_coeff_data, Tarang.set_coeff_data!, x -> _BCJL(copy(x)),
                       field, Tarang.get_coeff_data(twin))
        return true
    end
    function Tarang._gpu_backward_transform_backend!(::Tarang.GPU{JLArrays.JLBackend}, field)
        twin = bcjl_twin(field)
        bcjl_sync_scales!(twin, field)
        bcjl_copy_into!(Tarang.get_coeff_data, Tarang.set_coeff_data!, copy, twin,
                       Array(Tarang.get_coeff_data(field)))
        twin.current_layout = :c
        Tarang.backward_transform!(twin)
        bcjl_copy_into!(Tarang.get_grid_data, Tarang.set_grid_data!, x -> _BCJL(copy(x)),
                       field, Tarang.get_grid_data(twin))
        return true
    end
end

using LinearAlgebra, SparseArrays
GPUArrays.allowscalar(false)
Tarang.is_gpu_array(::SubArray{T,N,<:JLArrays.JLArray}) where {T,N} = true
Tarang.architecture(::SubArray{T,N,<:JLArrays.JLArray}) where {T,N} = _BCJL_ARCH

# Test-only stand-ins: FFTs use the CPU twin above; sparse factorization and
# solves use host LU with explicit transfers. All boundary gather/override,
# stage arithmetic, field storage, and scatter execute unchanged on JLArray.
struct BoundaryJLHostLU <: Tarang.MatSolvers.AbstractMatSolver
    factor::Any
end
BoundaryJLHostLU(A::AbstractMatrix; kwargs...) = BoundaryJLHostLU(lu(Matrix{ComplexF64}(A)))
function Tarang.MatSolvers.solve!(dest, s::BoundaryJLHostLU, rhs)
    copyto!(dest, s.factor \ Array(rhs))
    return dest
end
function Tarang._subproblem_backend_matrix!(sp::Tarang.Subproblem, matrix, which::Symbol,
                                           data::JLArrays.JLArray{T,1}) where T
    matrix === nothing && return nothing
    field = Tarang._subproblem_backend_field(which)
    cached = getfield(sp.runtime, field)
    cached !== nothing && return cached
    backend = JLArrays.JLArray(Matrix(matrix))
    setfield!(sp.runtime, field, backend)
    return backend
end
