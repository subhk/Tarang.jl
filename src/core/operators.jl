"""
Operator classes for spectral operations

Translated from dedalus/core/operators.py
"""

using LinearAlgebra
using LinearAlgebra: BLAS
using SparseArrays
using LoopVectorization  # For SIMD-optimized loops

# Operator registration tables (mirroring Dedalus parsing registries)
const OPERATOR_ALIASES = Dict{String, Any}()
const OPERATOR_PARSEABLES = Dict{String, Any}()
const OPERATOR_PREFIXES = Dict{String, Any}()

function register_operator_alias!(op, names::AbstractString...)
    for name in names
        OPERATOR_ALIASES[name] = op
    end
    return op
end

function register_operator_parseable!(op, names::AbstractString...)
    for name in names
        OPERATOR_PARSEABLES[name] = op
    end
    return op
end

function register_operator_prefix!(op, names::AbstractString...)
    for name in names
        OPERATOR_PREFIXES[name] = op
    end
    return op
end

# CPU-only (CPU support removed)

abstract type Operator <: Operand end

# Basic differential operators
struct Gradient <: Operator
    operand::Operand
    coordsys::CoordinateSystem
end
const _Gradient_constructor = Gradient

struct Divergence <: Operator
    operand::Operand
end
const _Divergence_constructor = Divergence

struct Curl <: Operator
    operand::Operand
    coordsys::CoordinateSystem
end
const _Curl_constructor = Curl

struct Laplacian <: Operator
    operand::Operand
end
const _Laplacian_constructor = Laplacian

struct Trace <: Operator
    operand::Operand
end
const _Trace_constructor = Trace

struct Skew <: Operator
    operand::Operand
end
const _Skew_constructor = Skew

struct TransposeComponents <: Operator
    operand::Operand
end
const _TransposeComponents_constructor = TransposeComponents

# Time derivative
struct TimeDerivative <: Operator
    operand::Operand
    order::Int
    
    function TimeDerivative(operand::Operand, order::Int=1)
        new(operand, order)
    end
end

# Interpolation and integration
struct Interpolate <: Operator
    operand::Operand
    coord::Coordinate
    position::Real
end
const _Interpolate_constructor = Interpolate

struct Integrate <: Operator
    operand::Operand
    coord::Union{Coordinate, Tuple{Vararg{Coordinate}}}
end
const _Integrate_constructor = Integrate

struct Average <: Operator
    operand::Operand
    coord::Coordinate
end
const _Average_constructor = Average

# Conversion operators
struct Convert <: Operator
    operand::Operand
    basis::Basis
end
const _Convert_constructor = Convert

struct Grid <: Operator
    operand::Operand
end
const _Grid_constructor = Grid

struct Coeff <: Operator
    operand::Operand
end
const _Coeff_constructor = Coeff

# Lifting operator for boundary conditions
struct Lift <: Operator
    operand::Operand
    basis::Basis
    n::Int
end
const _Lift_constructor = Lift

# Component extraction
struct Component <: Operator
    operand::Operand
    index::Int
end
const _Component_constructor = Component

struct RadialComponent <: Operator
    operand::Operand
end
const _RadialComponent_constructor = RadialComponent

struct AngularComponent <: Operator
    operand::Operand
end
const _AngularComponent_constructor = AngularComponent

struct AzimuthalComponent <: Operator
    operand::Operand
end
const _AzimuthalComponent_constructor = AzimuthalComponent

# Spectral differentiation
struct Differentiate <: Operator
    operand::Operand
    coord::Coordinate
    order::Int
end
const _Differentiate_constructor = Differentiate

# Constructor functions (following Dedalus API)
function grad(operand::Operand, coordsys::CoordinateSystem=operand.dist.coordsys)
    """Gradient operator"""
    return multiclass_new(Gradient, operand, coordsys)
end

function div(operand::Operand)
    """Divergence operator"""
    return multiclass_new(Divergence, operand)
end

function curl(operand::Operand, coordsys::CoordinateSystem=operand.dist.coordsys)
    """Curl operator"""
    return multiclass_new(Curl, operand, coordsys)
end

function lap(operand::Operand)
    """Laplacian operator"""
    return multiclass_new(Laplacian, operand)
end

function trace(operand::Operand)
    """Trace operator"""
    return multiclass_new(Trace, operand)
end

function skew(operand::Operand)
    """Skew-symmetric part of tensor"""
    return multiclass_new(Skew, operand)
end

function transpose_components(operand::Operand)
    """Transpose tensor components"""
    return multiclass_new(TransposeComponents, operand)
end

function interpolate(operand::Operand, coord::Coordinate, position::Real)
    """Interpolate operand along a coordinate at a given position"""
    return multiclass_new(Interpolate, operand, coord, position)
end

function integrate(operand::Operand, coord::Union{Coordinate, Tuple{Vararg{Coordinate}}})
    """Integrate operand over the specified coordinate(s)"""
    return multiclass_new(Integrate, operand, coord)
end

function average(operand::Operand, coord::Coordinate)
    """Average operand along a coordinate"""
    return multiclass_new(Average, operand, coord)
end

function convert(operand::Operand, basis::Basis)
    """Convert operand to a different basis"""
    return multiclass_new(Convert, operand, basis)
end

function dt(operand::Operand, order::Int=1)
    """Time derivative"""
    return TimeDerivative(operand, order)
end

# Differentiation functions
function d(operand::Operand, coord::Coordinate, order::Int=1)
    """Differentiate with respect to coordinate"""
    return multiclass_new(Differentiate, operand, coord, order)
end

function lift(operand::Operand, basis::Basis, n::Int)
    """Apply lifting operator for boundary conditions"""
    return multiclass_new(Lift, operand, basis, n)
end

# Register core operators for parsing namespace consistency
register_operator_alias!(grad, "grad", "gradient")
register_operator_parseable!(grad, "grad", "gradient")

register_operator_alias!(div, "div", "divergence")
register_operator_parseable!(div, "div", "divergence")

register_operator_alias!(curl, "curl")
register_operator_parseable!(curl, "curl")

register_operator_alias!(lap, "lap", "laplacian")
register_operator_parseable!(lap, "lap", "laplacian")

register_operator_alias!(trace, "trace")
register_operator_parseable!(trace, "trace")

register_operator_alias!(skew, "skew")
register_operator_parseable!(skew, "skew")

register_operator_alias!(transpose_components, "transpose_components", "transpose")
register_operator_parseable!(transpose_components, "transpose_components", "transpose")

register_operator_alias!(interpolate, "interpolate")
register_operator_parseable!(interpolate, "interpolate")

register_operator_alias!(integrate, "integrate", "integ")
register_operator_parseable!(integrate, "integrate", "integ")

register_operator_alias!(average, "average", "avg")
register_operator_parseable!(average, "average", "avg")

register_operator_alias!(convert, "convert")
register_operator_parseable!(convert, "convert")

register_operator_alias!(dt, "dt")
register_operator_parseable!(dt, "dt")

register_operator_alias!(d, "d", "differentiate")
register_operator_parseable!(d, "d", "differentiate")

register_operator_alias!(lift, "lift")
register_operator_parseable!(lift, "lift")

# Helper functions for creating common operators
function ∇(operand::Operand, coordsys::CoordinateSystem=operand.dist.coordsys)
    """Unicode gradient operator"""
    return grad(operand, coordsys)
end

function ∇²(operand::Operand)
    """Unicode Laplacian operator"""
    return lap(operand)
end

# Include nonlinear terms for integration
# This will be included after nonlinear_terms.jl is loaded

# Operator evaluation functions
function evaluate_gradient(grad_op::Gradient, layout::Symbol=:g)
    """Evaluate gradient operator"""
    operand = grad_op.operand
    coordsys = grad_op.coordsys
    
    if isa(operand, ScalarField)
        # Create vector field for result
        result = VectorField(operand.dist, coordsys, "grad_$(operand.name)", operand.bases, operand.dtype)
        
        # Compute partial derivatives for each component
        for (i, coord_name) in enumerate(coordsys.names)
            coord = coordsys[coord_name]
            # Apply differentiation operator
            result.components[i] = evaluate_differentiate(Differentiate(operand, coord, 1), layout)
        end
        
        return result
    else
        throw(ArgumentError("Gradient not implemented for operand type $(typeof(operand))"))
    end
end

function evaluate_divergence(div_op::Divergence, layout::Symbol=:g)
    """Evaluate divergence operator"""
    operand = div_op.operand
    
    if isa(operand, VectorField)
        # Sum partial derivatives of components
        coordsys = operand.coordsys
        result = ScalarField(operand.dist, "div_$(operand.name)", operand.bases, operand.dtype)
        
        # Initialize result to zero
        ensure_layout!(result, layout)
        fill!(result[string(layout)], 0)
        
        for (i, coord_name) in enumerate(coordsys.names)
            coord = coordsys[coord_name]
            # Add ∂u_i/∂x_i
            component_deriv = evaluate_differentiate(Differentiate(operand.components[i], coord, 1), layout)
            result = result + component_deriv
        end
        
        return result
    else
        throw(ArgumentError("Divergence not implemented for operand type $(typeof(operand))"))
    end
end

function evaluate_differentiate(diff_op::Differentiate, layout::Symbol=:g)
    """Evaluate differentiation operator"""
    operand = diff_op.operand
    coord = diff_op.coord
    order = diff_op.order
    
    if !isa(operand, ScalarField)
        throw(ArgumentError("Differentiation currently only supports scalar fields"))
    end
    
    # Find which basis corresponds to this coordinate
    basis_index = nothing
    for (i, basis) in enumerate(operand.bases)
        if basis.meta.element_label == coord.name
            basis_index = i
            break
        end
    end
    
    if basis_index === nothing
        throw(ArgumentError("Coordinate $(coord.name) not found in field bases"))
    end
    
    basis = operand.bases[basis_index]
    result = ScalarField(operand.dist, "d$(order)_$(operand.name)_d$(coord.name)$(order)", operand.bases, operand.dtype)
    
    # Apply differentiation based on basis type
    if isa(basis, RealFourier) || isa(basis, ComplexFourier)
        evaluate_fourier_derivative!(result, operand, basis_index, order, layout)
    elseif isa(basis, ChebyshevT)
        evaluate_chebyshev_derivative!(result, operand, basis_index, order, layout)
    elseif isa(basis, Legendre)
        evaluate_legendre_derivative!(result, operand, basis_index, order, layout)
    else
        throw(ArgumentError("Differentiation not implemented for basis type $(typeof(basis))"))
    end
    
    return result
end

function evaluate_fourier_derivative!(result::ScalarField, operand::ScalarField, axis::Int, order::Int, layout::Symbol)
    """Evaluate Fourier derivative following Dedalus conventions """
    ensure_layout!(operand, :c)  # Work in coefficient space
    ensure_layout!(result, :c)
    
    # Get the basis for the specified axis
    basis = operand.bases[axis]
    N = basis.meta.size
    L = basis.meta.bounds[2] - basis.meta.bounds[1]
    
    # Get device configuration from the field
    
    if isa(basis, RealFourier)
        # Real Fourier case: use Dedalus approach with 2x2 group matrices
        evaluate_real_fourier_derivative_dedalus!(result, operand, axis, order, N, L)
    elseif isa(basis, ComplexFourier)
        # Complex Fourier case: simple multiplication by (ik)^order
        evaluate_complex_fourier_derivative!(result, operand, axis, order, N, L)
    else
        throw(ArgumentError("Fourier derivative only applicable to Fourier bases"))
    end
    
        
    if layout == :g
        backward_transform!(result)
    end
end


function evaluate_real_fourier_derivative_dedalus!(result::ScalarField, operand::ScalarField, axis::Int, order::Int, N::Int, L::Float64)
    """Real Fourier derivative following Dedalus 2x2 group matrix approach """
    
    # Dedalus stores RealFourier as [cos_0, cos_1, sin_1, cos_2, sin_2, ..., cos_nyq]
    # Each wavenumber k>0 has a 2x2 group matrix:
    # dx [cos(kx)]   [0  -k] [cos(kx)]   [-k*sin(kx)]
    #    [sin(kx)] = [k   0] [sin(kx)] = [ k*cos(kx)]
    
    # Initialize result to zero
    fill!(result.data_c, 0.0)
    
    # DC component (k=0): derivative is always 0
    result.data_c[1] = 0.0
    
    # Process each wavenumber group
    k_max = N ÷ 2
    is_even = (N % 2 == 0)
    
    # Choose optimized implementation based on device and size
    if true  # CPU-only && order == 1 && length(operand.data_c) > 100
        # CPU path without LoopVectorization to avoid macro issues during precompilation
        for k in 1:k_max-(is_even ? 1 : 0)  # k=1 to k_max-1 (excluding Nyquist)
            # Physical wavenumber
            k_phys = 2π * k / L
            
            # Indices in coefficient array
            cos_idx = 2*k        # cos(kx) coefficient
            sin_idx = 2*k + 1    # sin(kx) coefficient
            
            if cos_idx <= length(operand.data_c) && sin_idx <= length(operand.data_c)
                cos_coeff = operand.data_c[cos_idx]
                sin_coeff = operand.data_c[sin_idx]
                
                # Apply 2x2 matrix: [0 -k; k 0]
                result.data_c[cos_idx] = -k_phys * sin_coeff   # d/dx[cos] = -k*sin
                result.data_c[sin_idx] =  k_phys * cos_coeff   # d/dx[sin] =  k*cos
            end
        end
    elseif false  # Not used; legacy GPU path removed
        # Optimized implementation using broadcasting
        evaluate_real_fourier_derivative_optimized!(result, operand, N, L, k_max, is_even)
    else
        # Fallback implementation for higher orders or other cases
        evaluate_real_fourier_derivative_fallback!(result, operand, N, L, k_max, is_even, order)
    end
    
    # Handle Nyquist frequency for even N (real-valued, derivative = 0)  
    if is_even
        nyquist_idx = N  # Last coefficient is Nyquist
        if nyquist_idx <= length(result.data_c)
            if order % 2 == 1
                result.data_c[nyquist_idx] = 0.0  # Odd derivatives of cos(π*x/L) = 0 at boundaries
            else
                k_nyquist = 2π * k_max / L
                result.data_c[nyquist_idx] = ((-1)^(order÷2)) * (k_nyquist^order) * operand.data_c[nyquist_idx]
            end
        end
    end
end

function evaluate_real_fourier_derivative_optimized!(result::ScalarField, operand::ScalarField, N::Int, L::Float64, k_max::Int, is_even::Bool)
    """Optimized Real Fourier derivative implementation"""

    # Create wavenumber arrays
    k_range = 1:(k_max-(is_even ? 1 : 0))
    if !isempty(k_range)
        # Vectorized vectorized operations
        for k in k_range
            k_phys = 2π * k / L
            cos_idx = 2*k
            sin_idx = 2*k + 1
            
            if cos_idx <= length(operand.data_c) && sin_idx <= length(operand.data_c)
                cos_coeff = operand.data_c[cos_idx]
                sin_coeff = operand.data_c[sin_idx]
                
                # Apply 2x2 matrix: [0 -k; k 0]
                result.data_c[cos_idx] = -k_phys * sin_coeff
                result.data_c[sin_idx] =  k_phys * cos_coeff
            end
        end
    end
end

function evaluate_real_fourier_derivative_fallback!(result::ScalarField, operand::ScalarField, N::Int, L::Float64, k_max::Int, is_even::Bool, order::Int)
    """Fallback implementation for higher-order derivatives"""
    
    for k in 1:k_max-(is_even ? 1 : 0)
        k_phys = 2π * k / L
        cos_idx = 2*k
        sin_idx = 2*k + 1
        
        if cos_idx <= length(operand.data_c) && sin_idx <= length(operand.data_c)
            cos_coeff = operand.data_c[cos_idx] 
            sin_coeff = operand.data_c[sin_idx]
            
            if order == 1
                result.data_c[cos_idx] = -k_phys * sin_coeff
                result.data_c[sin_idx] =  k_phys * cos_coeff
            elseif order == 2
                result.data_c[cos_idx] = -k_phys^2 * cos_coeff
                result.data_c[sin_idx] = -k_phys^2 * sin_coeff
            else
                # General case: (ik)^order applied to cos+i*sin representation
                complex_coeff = complex(cos_coeff, sin_coeff)
                factor = (im * k_phys)^order
                result_complex = factor * complex_coeff
                result.data_c[cos_idx] = real(result_complex)
                result.data_c[sin_idx] = imag(result_complex)
            end
        end
    end
end

function evaluate_complex_fourier_derivative!(result::ScalarField, operand::ScalarField, axis::Int, order::Int, N::Int, L::Float64)
    """Complex Fourier derivative: simple multiplication by (ik)^order """
    
    # Complex FFT wavenumbers: [0, 1, ..., N/2-1, -N/2, -(N/2-1), ..., -1]
    k_cpu = 2π/L * [0:(N÷2-1); -(N÷2):(-1)]
    
    # Move wavenumbers to appropriate device
    k = k_cpu
    
    if true  # CPU-only && length(operand.data_c) > 100 && order <= 3
        # CPU path without LoopVectorization to avoid macro errors
        for i in eachindex(result.data_c, operand.data_c)
            k_val = k[i]
            factor = (im * k_val)^order
            result.data_c[i] = operand.data_c[i] * factor
        end
    else
        # Vectorized or standard version using broadcasting
        factors = (im .* k).^order
        result.data_c .= operand.data_c .* factors
    end
end

function evaluate_chebyshev_derivative!(result::ScalarField, operand::ScalarField, axis::Int, order::Int, layout::Symbol)
    """Evaluate Chebyshev derivative using Dedalus-compatible differentiation matrix """
    ensure_layout!(operand, :c)  # Work in coefficient space
    ensure_layout!(result, :c)
    
    # Get device configuration from the field
    
    # Ensure data is on correct device
    operand.data_c = operand.data_c
    result.data_c = result.data_c
    
    basis = operand.bases[axis]
    N = basis.meta.size
    a, b = basis.meta.bounds
    
    # Domain transformation scale factor
    scale = 2.0 / (b - a)
    
    # Apply multiple derivatives if order > 1
    if order == 1
        # Single derivative with optimized implementation
        evaluate_chebyshev_single_derivative!(result, operand, N, scale)
    else
        # Multiple derivatives: apply single derivative 'order' times
        temp_field = ScalarField(operand.dist, "temp_deriv", operand.bases, operand.dtype)
        current_operand = operand

        for i in 1:order
            if i == order
                # Last iteration: store in result
                evaluate_chebyshev_single_derivative!(result, current_operand, N, scale)
            else
                # Intermediate iterations: use temp field
                evaluate_chebyshev_single_derivative!(temp_field, current_operand, N, scale)
                current_operand = temp_field
            end
        end
    end
    
        
    if layout == :g
        backward_transform!(result)
    end
end

function evaluate_chebyshev_single_derivative!(result::ScalarField, operand::ScalarField, N::Int, scale::Float64)
    """
    Single Chebyshev derivative using correct backward recurrence .
    
    The standard Chebyshev derivative formula is:
    c'_k = sum_{j=k+1, j-k odd} 2*j*c_j  for k >= 0
    
    This implements the backward recurrence efficiently.
    """
    
    # Initialize result to zero
    fill!(result.data_c, 0.0)
    
    if true  # CPU-only && N > 100 && length(operand.data_c) > 100
        # CPU path without LoopVectorization to keep precompilation simple
        for k in 1:min(N, length(result.data_c))
            deriv_sum = 0.0
            for j in (k+1):min(N, length(operand.data_c))
                if (j - k) % 2 == 1  # j-k is odd
                    deriv_sum += 2.0 * (j - 1) * operand.data_c[j]
                end
            end
            result.data_c[k] = deriv_sum * scale
        end
    else
        # Vectorized or standard implementation
        for k in 1:min(N, length(result.data_c))
            deriv_sum = 0.0
            
            # Apply the standard Chebyshev derivative recurrence:
            # c'_k = sum_{j=k+1, j-k odd} 2*j*c_j
            
            for j in (k+1):min(N, length(operand.data_c))
                if (j - k) % 2 == 1  # j-k is odd
                    # Coefficient j corresponds to T_{j-1} polynomial (1-based indexing)
                    deriv_sum += 2.0 * (j - 1) * operand.data_c[j]
                end
            end
            
            result.data_c[k] = deriv_sum * scale
        end
    end
end

function build_chebyshev_differentiation_matrix(N::Int)
    """
    Build the Chebyshev differentiation matrix using the correct backward recurrence.
    
    Uses the standard formula: c'_k = sum_{j=k+1, j-k odd} 2*j*c_j
    """
    D = zeros(Float64, N, N)
    
    # Apply the standard Chebyshev derivative recurrence:
    # result_coeff[k] = sum_j D[k,j] * input_coeff[j]
    # where D[k,j] = 2*j if j > k and (j-k) is odd, 0 otherwise
    
    for k in 1:N      # Output coefficient index  
        for j in 1:N  # Input coefficient index
            if j > k && (j - k) % 2 == 1  # j > k and j-k is odd
                # Coefficient j corresponds to T_{j-1} polynomial (1-based indexing)
                D[k, j] = 2.0 * (j - 1)
            end
        end
    end
    
    return sparse(D)
end

function evaluate_legendre_derivative!(result::ScalarField, operand::ScalarField, axis::Int, order::Int, layout::Symbol)
    """Evaluate Legendre derivative using Dedalus-compatible Jacobi implementation """
    ensure_layout!(operand, :c)  # Work in coefficient space
    ensure_layout!(result, :c)
    
    # Get device configuration from the field
    
    # Ensure data is on correct device
    operand.data_c = operand.data_c
    result.data_c = result.data_c
    
    basis = operand.bases[axis]
    N = basis.meta.size
    a, b = basis.meta.bounds
    
    # Domain transformation scale factor
    scale = 2.0 / (b - a)
    
    # Apply multiple derivatives if order > 1
    if order == 1
        # Single derivative with optimized implementation
        evaluate_legendre_single_derivative!(result, operand, N, scale)
    else
        # Multiple derivatives: apply single derivative 'order' times
        temp_field = ScalarField(operand.dist, "temp_deriv", operand.bases, operand.dtype)
        current_operand = operand

        for i in 1:order
            if i == order
                # Last iteration: store in result
                evaluate_legendre_single_derivative!(result, current_operand, N, scale)
            else
                # Intermediate iterations: use temp field
                evaluate_legendre_single_derivative!(temp_field, current_operand, N, scale)
                current_operand = temp_field
            end
        end
    end
    
        
    if layout == :g
        backward_transform!(result)
    end
end

function evaluate_legendre_single_derivative!(result::ScalarField, operand::ScalarField, N::Int, scale::Float64)
    """
    Single Legendre derivative using Dedalus Jacobi approach .
    
    Legendre polynomials are Jacobi polynomials with a=0, b=0.
    From Dedalus Jacobi D(+1): bands = [(N + a + b + 1) * 2^(-1)]
    For Legendre: bands = [(N + 1) * 0.5]
    
    The standard Legendre derivative recurrence relation is:
    P'_n = (2n-1)*P_{n-1} + (2n-5)*P_{n-3} + (2n-9)*P_{n-5} + ...
    
    This can be written in backward form for coefficient transformation.
    """
    
    # Initialize result to zero
    fill!(result.data_c, 0.0)
    
    if true  # CPU-only && N > 100 && length(operand.data_c) > 100
        # CPU path without LoopVectorization for simplicity
        for k in 1:min(N, length(result.data_c))
            deriv_sum = 0.0
            for j in (k+1):min(N, length(operand.data_c))
                if (j - k) % 2 == 1  # j-k is odd
                    deriv_sum += (2.0 * (j - 1) - 1.0) * operand.data_c[j]
                end
            end
            result.data_c[k] = deriv_sum * scale
        end
    else
        # Vectorized or standard implementation
        for k in 1:min(N, length(result.data_c))
            deriv_sum = 0.0
            
            # Apply the Legendre derivative recurrence:
            # Based on Jacobi D(+1) with a=b=0
            
            for j in (k+1):min(N, length(operand.data_c))
                if (j - k) % 2 == 1  # j-k is odd
                    # For Legendre: coefficient factor is (2j-1) instead of 2j
                    # This comes from the Jacobi D(+1) formula with a=b=0
                    deriv_sum += (2.0 * (j - 1) - 1.0) * operand.data_c[j]  # 2j-1 pattern
                end
            end
            
            result.data_c[k] = deriv_sum * scale
        end
    end
end

# Helper functions - the build_chebyshev_differentiation_matrix function above 
# replaces the old simplified chebyshev_derivative_matrix implementation

function apply_matrix_along_axis(matrix::AbstractMatrix, array::AbstractArray, axis::Int; out=nothing)
    """
    Apply matrix along any axis of an array.
    Following Dedalus array.py:77-82 and apply_dense:104-126 implementation.
    """
    if issparse(matrix)
        return apply_sparse_along_axis(matrix, array, axis; out=out)
    else
        return apply_dense_along_axis(matrix, array, axis; out=out)
    end
end

function apply_dense_along_axis(matrix::AbstractMatrix, array::AbstractArray, axis::Int; out=nothing)
    """
    Apply dense matrix along any axis of an array .
    Following Dedalus apply_dense implementation in array.py:104-126.
    """
    ndim = ndims(array)
    
    # Resolve wraparound axis (convert to 1-based indexing)
    axis = mod1(axis, ndim)
    
    # Move target axis to position 1 (Julia's first dimension)
    if axis != 1
        perm = collect(1:ndim)
        perm[1] = axis
        perm[axis] = 1
        array = permutedims(array, perm)
    end
    
    # Get array shape after permutation
    array_shape = size(array)
    
    # Flatten later axes for matrix multiplication
    if ndim > 2
        array = reshape(array, (array_shape[1], prod(array_shape[2:end])))
    end
    
    # Apply matrix multiplication (CPU-compatible)
    temp = matrix * array
    
    # Unflatten later axes
    if ndim > 2
        new_shape = (size(temp, 1), array_shape[2:end]...)
        temp = reshape(temp, new_shape)
    end
    
    # Move axis back from position 1
    if axis != 1
        perm = collect(1:ndim)
        perm[1] = axis
        perm[axis] = 1
        temp = permutedims(temp, perm)
    end
    
    # Handle output
    if out === nothing
        return temp
    else
        copyto!(out, temp)
        return out
    end
end

function apply_sparse_along_axis(matrix::SparseMatrixCSC, array::AbstractArray, axis::Int; out=nothing, check_shapes=false)
    """
    Apply sparse matrix along any axis of an array.
    Following Dedalus apply_sparse implementation in array.py:171-203.
    Note: Uses SparseMatrixCSC (Julia's sparse format) instead of CSR.
    """
    ndim = ndims(array)
    
    # Resolve wraparound axis
    axis = mod1(axis, ndim)
    
    # Check output allocation
    if out === nothing
        out_shape = collect(size(array))
        out_shape[axis] = size(matrix, 1)
        out = zeros(eltype(array), out_shape...)
    elseif out === array
        throw(ArgumentError("Cannot apply in place"))
    end
    
    # Check shapes if requested
    if check_shapes
        if !(1 <= axis <= ndim)
            throw(BoundsError("Axis out of bounds"))
        end
        if size(matrix, 2) != size(array, axis) || size(matrix, 1) != size(out, axis)
            throw(DimensionMismatch("Matrix shape mismatch"))
        end
    end
    
    # Move target axis to position 1
    if axis != 1
        perm = collect(1:ndim)
        perm[1] = axis
        perm[axis] = 1
        array = permutedims(array, perm)
    end
    
    # Get array shape after permutation
    array_shape = size(array)
    
    # Flatten later axes
    if ndim > 2
        array = reshape(array, (array_shape[1], prod(array_shape[2:end])))
    end
    
    # Apply sparse matrix multiplication
    temp = matrix * array
    
    # Unflatten later axes
    if ndim > 2
        new_shape = (size(temp, 1), array_shape[2:end]...)
        temp = reshape(temp, new_shape)
    end
    
    # Move axis back from position 1
    if axis != 1
        perm = collect(1:ndim)
        perm[1] = axis
        perm[axis] = 1
        temp = permutedims(temp, perm)
    end
    
    # Copy to output
    copyto!(out, temp)
    return out
end

# Operator arithmetic
Base.:+(op1::Operator, op2::Operator) = AddOperator(op1, op2)
Base.:-(op1::Operator, op2::Operator) = SubtractOperator(op1, op2)  
Base.:*(op1::Operator, op2::Union{Real, Operator}) = MultiplyOperator(op1, op2)

# ---------------------------------------------------------------------------
# Dispatch integration for core operators
# ---------------------------------------------------------------------------

function dispatch_preprocess(::Type{Gradient}, args::Tuple, kwargs::NamedTuple)
    if length(args) == 1
        operand = args[1]
        coordsys = hasfield(typeof(operand), :dist) ? operand.dist.coordsys : operand.coordsys
        return ((operand, coordsys), kwargs)
    elseif length(args) == 2
        return (args, kwargs)
    else
        throw(ArgumentError("Gradient expects 1 or 2 arguments"))
    end
end

function dispatch_check(::Type{Gradient}, args::Tuple, kwargs::NamedTuple)
    operand = args[1]
    if !isa(operand, Operand)
        throw(ArgumentError("Gradient requires an Operand"))
    end
    return true
end

function invoke_constructor(::Type{Gradient}, args::Tuple, kwargs::NamedTuple)
    operand, coordsys = args
    return _Gradient_constructor(operand, coordsys)
end

function dispatch_check(::Type{Divergence}, args::Tuple, kwargs::NamedTuple)
    operand = args[1]
    if !isa(operand, VectorField)
        throw(ArgumentError("Divergence requires a VectorField"))
    end
    return true
end

function invoke_constructor(::Type{Divergence}, args::Tuple, kwargs::NamedTuple)
    return _Divergence_constructor(args[1])
end

function dispatch_preprocess(::Type{Curl}, args::Tuple, kwargs::NamedTuple)
    if length(args) == 1
        operand = args[1]
        coordsys = hasfield(typeof(operand), :dist) ? operand.dist.coordsys : operand.coordsys
        return ((operand, coordsys), kwargs)
    elseif length(args) == 2
        return (args, kwargs)
    else
        throw(ArgumentError("Curl expects 1 or 2 arguments"))
    end
end

function dispatch_check(::Type{Curl}, args::Tuple, kwargs::NamedTuple)
    operand = args[1]
    if !isa(operand, VectorField)
        throw(ArgumentError("Curl requires a VectorField"))
    end
    return true
end

function invoke_constructor(::Type{Curl}, args::Tuple, kwargs::NamedTuple)
    operand, coordsys = args
    return _Curl_constructor(operand, coordsys)
end

function dispatch_check(::Type{Laplacian}, args::Tuple, kwargs::NamedTuple)
    operand = args[1]
    if !isa(operand, Operand)
        throw(ArgumentError("Laplacian requires an Operand"))
    end
    return true
end

function invoke_constructor(::Type{Laplacian}, args::Tuple, kwargs::NamedTuple)
    return _Laplacian_constructor(args[1])
end

function dispatch_check(::Type{Trace}, args::Tuple, kwargs::NamedTuple)
    operand = args[1]
    if !isa(operand, TensorField)
        throw(ArgumentError("Trace requires a TensorField"))
    end
    return true
end

function invoke_constructor(::Type{Trace}, args::Tuple, kwargs::NamedTuple)
    return _Trace_constructor(args[1])
end

function dispatch_check(::Type{Interpolate}, args::Tuple, kwargs::NamedTuple)
    operand, coord, position = args
    if !isa(operand, Operand)
        throw(ArgumentError("Interpolate requires an Operand"))
    end
    if !isa(coord, Coordinate)
        throw(ArgumentError("Interpolate requires a Coordinate"))
    end
    if !isa(position, Real)
        throw(ArgumentError("Interpolate position must be real"))
    end
    return true
end

function invoke_constructor(::Type{Interpolate}, args::Tuple, kwargs::NamedTuple)
    operand, coord, position = args
    return _Interpolate_constructor(operand, coord, position)
end

function dispatch_check(::Type{Integrate}, args::Tuple, kwargs::NamedTuple)
    operand, coord = args
    if !isa(operand, Operand)
        throw(ArgumentError("Integrate requires an Operand"))
    end
    return true
end

function invoke_constructor(::Type{Integrate}, args::Tuple, kwargs::NamedTuple)
    return _Integrate_constructor(args[1], args[2])
end

function dispatch_check(::Type{Average}, args::Tuple, kwargs::NamedTuple)
    operand, coord = args
    if !isa(operand, Operand)
        throw(ArgumentError("Average requires an Operand"))
    end
    if !isa(coord, Coordinate)
        throw(ArgumentError("Average requires a Coordinate"))
    end
    return true
end

function invoke_constructor(::Type{Average}, args::Tuple, kwargs::NamedTuple)
    return _Average_constructor(args[1], args[2])
end

function dispatch_preprocess(::Type{Differentiate}, args::Tuple, kwargs::NamedTuple)
    if length(args) == 2
        operand, coord = args
        return ((operand, coord, 1), kwargs)
    elseif length(args) == 3
        return (args, kwargs)
    else
        throw(ArgumentError("Differentiate expects 2 or 3 arguments"))
    end
end

function dispatch_check(::Type{Differentiate}, args::Tuple, kwargs::NamedTuple)
    operand, coord, order = args
    if !isa(operand, Operand)
        throw(ArgumentError("Differentiate requires an Operand"))
    end
    if !isa(coord, Coordinate)
        throw(ArgumentError("Differentiate requires a Coordinate"))
    end
    if !(order isa Integer) || order < 0
        throw(ArgumentError("Differentiate order must be a non-negative integer"))
    end
    return true
end

function invoke_constructor(::Type{Differentiate}, args::Tuple, kwargs::NamedTuple)
    operand, coord, order = args
    return _Differentiate_constructor(operand, coord, order)
end

function dispatch_check(::Type{Convert}, args::Tuple, kwargs::NamedTuple)
    operand, basis = args
    if !isa(operand, Operand)
        throw(ArgumentError("Convert requires an Operand"))
    end
    if !isa(basis, Basis)
        throw(ArgumentError("Convert requires a Basis"))
    end
    return true
end

function invoke_constructor(::Type{Convert}, args::Tuple, kwargs::NamedTuple)
    return _Convert_constructor(args[1], args[2])
end

function dispatch_check(::Type{Grid}, args::Tuple, kwargs::NamedTuple)
    if !isa(args[1], Operand)
        throw(ArgumentError("Grid conversion requires an Operand"))
    end
    return true
end

function dispatch_check(::Type{Coeff}, args::Tuple, kwargs::NamedTuple)
    if !isa(args[1], Operand)
        throw(ArgumentError("Coeff conversion requires an Operand"))
    end
    return true
end

function invoke_constructor(::Type{Grid}, args::Tuple, kwargs::NamedTuple)
    return _Grid_constructor(args[1])
end

function invoke_constructor(::Type{Coeff}, args::Tuple, kwargs::NamedTuple)
    return _Coeff_constructor(args[1])
end

function dispatch_check(::Type{Skew}, args::Tuple, kwargs::NamedTuple)
    operand = args[1]
    if !isa(operand, TensorField)
        throw(ArgumentError("Skew requires a TensorField"))
    end
    return true
end

function invoke_constructor(::Type{Skew}, args::Tuple, kwargs::NamedTuple)
    return _Skew_constructor(args[1])
end

function dispatch_check(::Type{TransposeComponents}, args::Tuple, kwargs::NamedTuple)
    operand = args[1]
    if !isa(operand, TensorField)
        throw(ArgumentError("Transpose requires a TensorField"))
    end
    return true
end

function invoke_constructor(::Type{TransposeComponents}, args::Tuple, kwargs::NamedTuple)
    return _TransposeComponents_constructor(args[1])
end

function dispatch_check(::Type{Lift}, args::Tuple, kwargs::NamedTuple)
    operand, basis, n = args
    if !isa(operand, Operand)
        throw(ArgumentError("Lift requires an Operand"))
    end
    if !isa(basis, Basis)
        throw(ArgumentError("Lift requires a Basis"))
    end
    if !(n isa Integer) || n < 0
        throw(ArgumentError("Lift order must be a non-negative integer"))
    end
    return true
end

function invoke_constructor(::Type{Lift}, args::Tuple, kwargs::NamedTuple)
    operand, basis, n = args
    return _Lift_constructor(operand, basis, n)
end

function dispatch_check(::Type{Component}, args::Tuple, kwargs::NamedTuple)
    operand, index = args
    if !isa(operand, Operand)
        throw(ArgumentError("Component extraction requires an Operand"))
    end
    if !(index isa Integer) || index < 1
        throw(ArgumentError("Component index must be a positive integer"))
    end
    return true
end

function invoke_constructor(::Type{Component}, args::Tuple, kwargs::NamedTuple)
    operand, index = args
    return _Component_constructor(operand, index)
end

function dispatch_check(::Type{RadialComponent}, args::Tuple, kwargs::NamedTuple)
    operand = args[1]
    if !isa(operand, VectorField)
        throw(ArgumentError("RadialComponent requires a VectorField"))
    end
    return true
end

function invoke_constructor(::Type{RadialComponent}, args::Tuple, kwargs::NamedTuple)
    return _RadialComponent_constructor(args[1])
end

function dispatch_check(::Type{AngularComponent}, args::Tuple, kwargs::NamedTuple)
    operand = args[1]
    if !isa(operand, VectorField)
        throw(ArgumentError("AngularComponent requires a VectorField"))
    end
    return true
end

function invoke_constructor(::Type{AngularComponent}, args::Tuple, kwargs::NamedTuple)
    return _AngularComponent_constructor(args[1])
end

function dispatch_check(::Type{AzimuthalComponent}, args::Tuple, kwargs::NamedTuple)
    operand = args[1]
    if !isa(operand, VectorField)
        throw(ArgumentError("AzimuthalComponent requires a VectorField"))
    end
    return true
end

function invoke_constructor(::Type{AzimuthalComponent}, args::Tuple, kwargs::NamedTuple)
    return _AzimuthalComponent_constructor(args[1])
end

struct AddOperator <: Operator
    left::Operator
    right::Operator
end

struct SubtractOperator <: Operator
    left::Operator
    right::Operator
end

struct MultiplyOperator <: Operator
    left::Operator
    right::Union{Real, Operator}
end

