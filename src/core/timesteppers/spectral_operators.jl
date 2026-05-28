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
    operator_type::Symbol                    # :laplacian, :hyperviscosity, :custom
    ν::Float64                               # viscosity/diffusion coefficient
    order::Int                               # power of Laplacian (1 = ∇², 2 = ∇⁴, …)
end

"""
    SpectralLinearOperator(dist::Distributor, bases::Tuple, operator_type::Symbol; kwargs...)

Create a spectral linear operator for diagonal IMEX methods.

# Arguments
- `dist`: Distributor (determines architecture and local grid)
- `bases`: Tuple of basis objects (e.g., `(ComplexFourier, ComplexFourier)`)
- `operator_type`: Type of operator
  - `:laplacian` - `-ν∇²` with coefficient `ν`
  - `:hyperviscosity` - `-ν∇^(2p)` with coefficient `ν` and `order=p`
  - `:custom` - provide `coefficients` directly

# Keyword Arguments
- `ν::Real=1.0`: Viscosity/diffusion coefficient
- `order::Int=1`: Power of Laplacian (1 for ∇², 2 for ∇⁴, 4 for ∇⁸)
- `coefficients::AbstractArray`: Custom diagonal coefficients (for :custom)
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

    arch = dist.architecture
    N = length(bases)

    # Coefficient-space shape the operator must broadcast against — i.e. the
    # field's local coefficient array. The first RealFourier/Fourier axis is
    # rfft-reduced to N÷2+1; other axes keep full size N (and MPI yields the
    # per-rank slice). Reuse the canonical domain logic rather than re-deriving
    # the rfft layout: using the basis *grid* size here was the bug that made
    # L.coefficients (e.g. 16) mismatch the field's coeff array (e.g. 9) and
    # throw DimensionMismatch in the diagonal IMEX step.
    coeff_shape = local_shape(Domain(dist, bases), :c)

    if operator_type == :custom && coefficients !== nothing
        # Use provided coefficients
        L_coeffs = on_architecture(arch, T.(coefficients))
    else
        # Build wavenumber-based operator on CPU first
        L_coeffs_cpu = _build_spectral_operator(bases, coeff_shape, operator_type, ν, order, T)
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
    dtype::Type{T}
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
            coeff_shape[d] == (Nb ÷ 2 + 1) ? T.(wavenumbers_rfft(basis)) :
                                             T.(wavenumbers_fft(basis))
        elseif basis isa ComplexFourier
            T.(wavenumbers(basis))
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
        return zero(T)
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
