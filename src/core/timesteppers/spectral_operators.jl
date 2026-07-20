# ============================================================================
# Spectral Linear Operator for Diagonal IMEX Methods (GPU-native)
# ============================================================================

"""
    SpectralLinearOperator{T, N, A<:AbstractArray{T,N}}

Diagonal linear operator in spectral space for GPU-native IMEX timestepping.

For pseudospectral methods, linear operators like viscosity and hyperviscosity
are diagonal in Fourier space:
- Viscosity `-ν∇²`:     L̂(k) = ν(kx² + ky²)
- Hyperviscosity `-ν∇⁴`: L̂(k) = ν(kx² + ky²)²
- Hyper⁸ `-ν∇⁸`:        L̂(k) = ν(kx² + ky²)⁴

The implicit step `(I + dt*γ*L)⁻¹ * RHS` becomes element-wise division:
    û_new = RHS ./ (1 .+ dt * γ .* L_diagonal)

This avoids sparse matrix solves and stays 100% on GPU.

# Fields
- `coefficients::A`: Diagonal operator values L̂(k) for each wavenumber
- `architecture::AbstractArchitecture`: CPU() or GPU()

# Example
```julia
# Create Laplacian operator for 2D field
L = SpectralLinearOperator(dist, bases, :laplacian; ν=1e-3)

# Create hyperviscosity operator
L = SpectralLinearOperator(dist, bases, :hyperviscosity; ν=1e-10, order=4)

# Apply implicit step: û_new = û_old / (1 + dt*γ*L)
apply_implicit_diagonal!(u_new, u_old, L, dt, γ)
```
"""
struct SpectralLinearOperator{T<:AbstractFloat, N, A<:AbstractArray{T,N}}
    coefficients::A                          # L̂(k) diagonal values
    architecture::AbstractArchitecture
    operator_type::Symbol                    # one of SPECTRAL_OPERATOR_TYPES
    ν::Float64                               # viscosity/diffusion coefficient
    order::Int                               # power of Laplacian (1 = ∇², 2 = ∇⁴, …)
end

"""
Valid `operator_type` symbols accepted by [`SpectralLinearOperator`](@ref).

Kept as a single source of truth so the constructor's validation and the
per-mode kernel `_spectral_operator_value` cannot drift apart. An unrecognised
symbol used to fall through to `zero(T)`, producing an ALL-ZERO operator: the
implicit term then vanished with no diagnostic (e.g. a viscous run silently
became inviscid). Unknown symbols now raise instead.
"""
const SPECTRAL_OPERATOR_TYPES = (:laplacian, :hyperviscosity, :biharmonic, :custom)

"""
    SpectralLinearOperator(dist::Distributor, bases::Tuple, operator_type::Symbol; kwargs...)

Create a spectral linear operator for diagonal IMEX methods.

# Arguments
- `dist`: Distributor (determines architecture and local grid)
- `bases`: Tuple of basis objects (e.g., `(ComplexFourier, ComplexFourier)`)
- `operator_type`: Type of operator — one of `$(SPECTRAL_OPERATOR_TYPES)`
  - `:laplacian` - `-ν∇²` with coefficient `ν`, i.e. `L̂ = ν k²`
  - `:hyperviscosity` - `-ν∇^(2p)` with coefficient `ν` and `order=p`, i.e. `L̂ = ν (k²)^p`
  - `:biharmonic` - `ν∇⁴` with coefficient `ν`, i.e. `L̂ = ν k⁴`
  - `:custom` - provide `coefficients` directly (required for this type)

Any other symbol raises an `ArgumentError`.

# Keyword Arguments
- `ν::Real=1.0`: Viscosity/diffusion coefficient. Deliberately a SCALAR: a
  diagonal operator multiplies each coefficient by a number that depends on the
  wavenumber alone, so a spatially varying `ν(x)` is not representable (it
  couples modes). Passing an array raises a `MethodError` here; a variable
  coefficient written into an equation's implicit `L` term is rejected by the
  diagonal-IMEX steppers (see `_serial_diagonal_imex_Lmap!`). Use a
  non-diagonal IMEX scheme (`RK222`, `SBDF2`, …) for variable coefficients.
- `order::Int=1`: Power of Laplacian (1 for ∇², 2 for ∇⁴, 4 for ∇⁸)
- `coefficients::AbstractArray`: Custom diagonal coefficients (required for `:custom`)
- `dtype::Type=Float64`: Element type

# Example
```julia
# Laplacian: L = ν(kx² + ky²)
L_visc = SpectralLinearOperator(dist, bases, :laplacian; ν=1e-3)

# Hyperviscosity: L = ν(kx² + ky²)²
L_hyper4 = SpectralLinearOperator(dist, bases, :hyperviscosity; ν=1e-10, order=2)

# Hyper-8: L = ν(kx² + ky²)⁴
L_hyper8 = SpectralLinearOperator(dist, bases, :hyperviscosity; ν=1e-16, order=4)
```
"""
function SpectralLinearOperator(
    dist::Distributor,
    bases::Tuple,
    operator_type::Symbol;
    ν::Real=1.0,
    order::Int=1,
    coefficients::Union{Nothing, AbstractArray}=nothing,
    dtype::Type{T}=Float64
) where {T<:AbstractFloat}

    # Reject an unrecognised operator_type up front. Falling through used to yield
    # an all-zero L̂ — the implicit term silently disappeared and the run was, e.g.,
    # inviscid without any complaint. Validate here (before any allocation) so the
    # message names the offending symbol and the full valid set.
    _validate_spectral_operator_type(operator_type)
    if operator_type === :custom && coefficients === nothing
        throw(ArgumentError(
            "SpectralLinearOperator: operator_type=:custom requires the `coefficients` " *
            "keyword (the diagonal L̂ values). Without it the operator would be all zeros, " *
            "silently dropping the implicit term. Pass `coefficients=...`, or use one of " *
            "$(join(filter(!=(:custom), SPECTRAL_OPERATOR_TYPES), ", ")) with `ν`/`order`."))
    end

    arch = dist.architecture
    N = length(bases)

    # Coefficient-space shape the operator must broadcast against — i.e. the
    # field's local coefficient array. The first RealFourier/Fourier axis is
    # rfft-reduced to N÷2+1; other axes keep full size N (and MPI yields the
    # per-rank slice). Reuse the canonical domain logic rather than re-deriving
    # the rfft layout: using the basis *grid* size here was the bug that made
    # L.coefficients (e.g. 16) mismatch the field's coeff array (e.g. 9) and
    # throw DimensionMismatch in the diagonal IMEX step.
    coeff_shape = local_shape(get_or_build_domain(dist, bases), :c)

    # Per-axis 1-based range of GLOBAL coefficient modes this rank owns. Under MPI a
    # decomposed Fourier axis is a sub-slab, so the operator must use THIS rank's
    # wavenumbers (sliced by the global offset), not the first `local_len` global modes
    # — otherwise rank>0 gets the wrong per-mode decay rate. Serial ⇒ full range
    # (identical to the old behaviour). Mirrors the slicing in step_diagonal_imex.jl.
    local_ranges = ntuple(N) do d
        basis = bases[d]
        gsize = basis isa RealFourier ? (basis.meta.size ÷ 2 + 1) :
                (basis isa ComplexFourier ? basis.meta.size : coeff_shape[d])
        (dist.size > 1 && coeff_shape[d] != gsize) ? local_indices(dist, d, gsize) :
                                                     (1:coeff_shape[d])
    end

    if operator_type == :custom && coefficients !== nothing
        # Use provided coefficients
        L_coeffs = on_architecture(arch, T.(coefficients))
    else
        # Build wavenumber-based operator on CPU first
        L_coeffs_cpu = _build_spectral_operator(bases, coeff_shape, operator_type, ν, order, T, local_ranges)
        # Move to target architecture
        L_coeffs = on_architecture(arch, L_coeffs_cpu)
    end

    SpectralLinearOperator{T, N, typeof(L_coeffs)}(
        L_coeffs, arch, operator_type, Float64(ν), order
    )
end

"""
    SpectralLinearOperator(field::ScalarField, operator_type::Symbol; kwargs...)

Convenience constructor that extracts distributor and bases from a ScalarField.

# Example
```julia
q = ScalarField(dist, "q", bases, ComplexF64)
L = SpectralLinearOperator(q, :hyperviscosity; ν=1e-10, order=4)
```
"""
function SpectralLinearOperator(
    field::ScalarField,
    operator_type::Symbol;
    kwargs...
)
    SpectralLinearOperator(field.dist, field.bases, operator_type; kwargs...)
end

"""
Build spectral operator coefficients on CPU.
"""
function _build_spectral_operator(
    bases::Tuple,
    coeff_shape::Tuple,
    operator_type::Symbol,
    ν::Real,
    order::Int,
    dtype::Type{T},
    local_ranges::Tuple = ntuple(d -> 1:coeff_shape[d], length(bases))
) where {T}

    N = length(bases)

    # Wavenumbers must match the coefficient-array layout the field's spectral
    # transforms/derivatives use — NOT the native cos/sin packing that
    # `wavenumbers(basis)` returns ([0,1,1,2,2,…]), which is misaligned with the
    # rfft coefficient array and was assigning each mode the wrong decay rate.
    # The reduced axis (coeff length N÷2+1) uses the rfft layout [0,1,…,N/2];
    # full-length Fourier axes use the fft layout [0,…,N/2-1,-N/2,…,-1].
    k_arrays = ntuple(N) do d
        basis = bases[d]
        Nb = hasfield(typeof(basis), :meta) ? basis.meta.size : coeff_shape[d]
        if basis isa RealFourier
            full = coeff_shape[d] == (Nb ÷ 2 + 1) ? T.(wavenumbers_rfft(basis)) :
                                                    T.(wavenumbers_fft(basis))
            full[local_ranges[d]]                       # this rank's owned global modes
        elseif basis isa ComplexFourier
            T.(wavenumbers(basis))[local_ranges[d]]
        else
            # Non-spectral basis (e.g., Chebyshev): no wavenumber damping.
            zeros(T, coeff_shape[d])
        end
    end

    # Compute k² = sum of kᵢ² over all dimensions
    L_coeffs = zeros(T, coeff_shape)

    if N == 1
        for i in 1:coeff_shape[1]
            k2 = k_arrays[1][i]^2
            L_coeffs[i] = _spectral_operator_value(k2, operator_type, ν, order, T)
        end
    elseif N == 2
        for j in 1:coeff_shape[2]
            for i in 1:coeff_shape[1]
                k2 = k_arrays[1][i]^2 + k_arrays[2][j]^2
                L_coeffs[i, j] = _spectral_operator_value(k2, operator_type, ν, order, T)
            end
        end
    elseif N == 3
        for k in 1:coeff_shape[3]
            for j in 1:coeff_shape[2]
                for i in 1:coeff_shape[1]
                    k2 = k_arrays[1][i]^2 + k_arrays[2][j]^2 + k_arrays[3][k]^2
                    L_coeffs[i, j, k] = _spectral_operator_value(k2, operator_type, ν, order, T)
                end
            end
        end
    else
        error("SpectralLinearOperator: dimension $N not supported")
    end

    return L_coeffs
end

"""
    _validate_spectral_operator_type(operator_type::Symbol)

Throw an `ArgumentError` naming the valid symbols unless `operator_type` is one
of [`SPECTRAL_OPERATOR_TYPES`](@ref).
"""
function _validate_spectral_operator_type(operator_type::Symbol)
    operator_type in SPECTRAL_OPERATOR_TYPES && return nothing
    throw(ArgumentError(
        "SpectralLinearOperator: unknown operator_type `:$(operator_type)`. " *
        "Valid options are: $(join(map(s -> ":$s", SPECTRAL_OPERATOR_TYPES), ", ")). " *
        "(`:laplacian` → L̂ = ν k²; `:hyperviscosity` → L̂ = ν (k²)^order; " *
        "`:biharmonic` → L̂ = ν k⁴; `:custom` → supply `coefficients` yourself.) " *
        "An unrecognised symbol previously produced an all-zero operator, silently " *
        "dropping the implicit term."))
end

"""
Compute operator value L(k²) for a given k².
"""
function _spectral_operator_value(k2::T, operator_type::Symbol, ν::Real, order::Int, ::Type{T}) where T
    if operator_type == :laplacian
        # L = ν * k²
        return T(ν * k2)
    elseif operator_type == :hyperviscosity
        # L = ν * (k²)^order
        return T(ν * k2^order)
    elseif operator_type == :biharmonic
        # L = ν * k⁴
        return T(ν * k2^2)
    else
        # Unreachable via the constructor (which validates first); a direct call
        # with a bad symbol must still fail loudly rather than return zero.
        _validate_spectral_operator_type(operator_type)
        throw(ArgumentError(
            "SpectralLinearOperator: operator_type `:$(operator_type)` has no per-mode " *
            "value function (`:custom` operators must supply `coefficients` directly)."))
    end
end

"""
    apply_implicit_diagonal!(u_new, u_old, L::SpectralLinearOperator, dt, γ)

Apply implicit step using diagonal spectral operator (GPU-compatible).

Computes: û_new = û_old ./ (1 .+ dt * γ .* L)

This is the key operation that replaces sparse matrix solves with
element-wise division in spectral space.
"""
function apply_implicit_diagonal!(
    u_new::ScalarField,
    u_old::ScalarField,
    L::SpectralLinearOperator,
    dt::Real,
    γ::Real
)
    ensure_layout!(u_old, :c)
    ensure_layout!(u_new, :c)

    # û_new = û_old / (1 + dt*γ*L)
    # Broadcasting works on both CPU and GPU arrays
    get_coeff_data(u_new) .= get_coeff_data(u_old) ./ (1 .+ dt .* γ .* L.coefficients)

    return u_new
end

"""
    apply_implicit_diagonal_inplace!(u::ScalarField, L::SpectralLinearOperator, dt, γ)

In-place implicit step: û = û ./ (1 .+ dt * γ .* L)
"""
function apply_implicit_diagonal_inplace!(
    u::ScalarField,
    L::SpectralLinearOperator,
    dt::Real,
    γ::Real
)
    ensure_layout!(u, :c)
    get_coeff_data(u) ./= (1 .+ dt .* γ .* L.coefficients)
    return u
end

"""
    _get_spectral_linear_operator(solver::InitialValueSolver)

Get the spectral linear operator from the solver or problem.
Returns nothing if not configured.
"""
function _get_spectral_linear_operator(solver::InitialValueSolver)
    # Check solver's timestepper_state first
    if solver.timestepper_state !== nothing &&
       haskey(solver.timestepper_state.timestepper_data, :spectral_linear_operator)
        return solver.timestepper_state.timestepper_data[:spectral_linear_operator]
    end

    # Check problem parameters
    if haskey(solver.problem.parameters, "spectral_linear_operator")
        return solver.problem.parameters["spectral_linear_operator"]
    end

    return nothing
end

"""
    set_spectral_linear_operator!(solver::InitialValueSolver, L::SpectralLinearOperator)

Set the spectral linear operator for diagonal IMEX methods.

# Example
```julia
L = SpectralLinearOperator(dist, bases, :hyperviscosity; ν=1e-10, order=4)
set_spectral_linear_operator!(solver, L)
```
"""
function set_spectral_linear_operator!(solver::InitialValueSolver, L::SpectralLinearOperator)
    if solver.timestepper_state !== nothing
        solver.timestepper_state.timestepper_data[:spectral_linear_operator] = L
    else
        solver.problem.parameters["spectral_linear_operator"] = L
    end
    return solver
end
