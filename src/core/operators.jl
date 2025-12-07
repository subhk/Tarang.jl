"""
Operator classes for spectral operations
"""

using LinearAlgebra
using LinearAlgebra: BLAS
using SparseArrays
using LoopVectorization  # For SIMD loops

# Operator registration tables (for parsing registries)
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

# Outer product (tensor product of vectors)
struct Outer <: Operator
    left::Operand
    right::Operand
end
const _Outer_constructor = Outer

# AdvectiveCFL operator - computes grid-crossing frequency field
struct AdvectiveCFL <: Operator
    operand::Operand  # Velocity vector field
    coords::CoordinateSystem
end
const _AdvectiveCFL_constructor = AdvectiveCFL

# Constructor functions (following API)
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

function component(operand::Operand, index::Int)
    """Extract component from vector/tensor field"""
    return multiclass_new(Component, operand, index)
end

function radial(operand::Operand)
    """Extract radial component from vector field"""
    return multiclass_new(RadialComponent, operand)
end

function angular(operand::Operand)
    """Extract angular component from vector field"""
    return multiclass_new(AngularComponent, operand)
end

function azimuthal(operand::Operand)
    """Extract azimuthal component from vector field"""
    return multiclass_new(AzimuthalComponent, operand)
end

function grid(operand::Operand)
    """Convert operand to grid space"""
    return multiclass_new(Grid, operand)
end

function coeff(operand::Operand)
    """Convert operand to coefficient space"""
    return multiclass_new(Coeff, operand)
end

function outer(left::Operand, right::Operand)
    """
    Outer product (tensor product) of two vector fields.

    Returns a rank-2 tensor field T where T_ij = u_i * v_j.

    Following operators pattern for tensor construction.

    Arguments:
    - left: First vector field (u)
    - right: Second vector field (v)

    Returns:
    - TensorField with components T_ij = u_i * v_j
    """
    return multiclass_new(Outer, left, right)
end

function advective_cfl(velocity::Operand, coords::CoordinateSystem)
    """
    Compute advective CFL grid-crossing frequency field.

    Returns a scalar field representing the local grid-crossing frequency:
    f = |u|/Δx + |v|/Δy + |w|/Δz

    This is useful for adaptive timestepping where Δt < 1/max(f).

    Following operators:4342 AdvectiveCFL pattern.

    Arguments:
    - velocity: Vector field representing fluid velocity
    - coords: Coordinate system for the domain

    Returns:
    - ScalarField with CFL frequency at each grid point
    """
    return multiclass_new(AdvectiveCFL, velocity, coords)
end

# Alias for convenience
const cfl = advective_cfl

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

register_operator_alias!(average, "average", "avg", "ave")
register_operator_parseable!(average, "average", "avg", "ave")

register_operator_alias!(convert, "convert")
register_operator_parseable!(convert, "convert")

register_operator_alias!(dt, "dt")
register_operator_parseable!(dt, "dt")

register_operator_alias!(d, "d", "differentiate")
register_operator_parseable!(d, "d", "differentiate")

register_operator_alias!(lift, "lift")
register_operator_parseable!(lift, "lift")

register_operator_alias!(component, "component")
register_operator_parseable!(component, "component")

register_operator_alias!(radial, "radial")
register_operator_parseable!(radial, "radial")

register_operator_alias!(angular, "angular")
register_operator_parseable!(angular, "angular")

register_operator_alias!(azimuthal, "azimuthal")
register_operator_parseable!(azimuthal, "azimuthal")

register_operator_alias!(grid, "grid")
register_operator_parseable!(grid, "grid")

register_operator_alias!(coeff, "coeff")
register_operator_parseable!(coeff, "coeff")

register_operator_alias!(outer, "outer", "tensor_product")
register_operator_parseable!(outer, "outer", "tensor_product")

register_operator_alias!(advective_cfl, "advective_cfl", "cfl")
register_operator_parseable!(advective_cfl, "advective_cfl", "cfl")

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
    """Evaluate Fourier derivative following conventions """
    ensure_layout!(operand, :c)  # Work in coefficient space
    ensure_layout!(result, :c)
    
    # Get the basis for the specified axis
    basis = operand.bases[axis]
    N = basis.meta.size
    L = basis.meta.bounds[2] - basis.meta.bounds[1]
    
    # Get device configuration from the field
    
    if isa(basis, RealFourier)
        # Real Fourier case: use approach with 2x2 group matrices
        evaluate_real_fourier_derivative_groups!(result, operand, axis, order, N, L)
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


function evaluate_real_fourier_derivative_groups!(result::ScalarField, operand::ScalarField, axis::Int, order::Int, N::Int, L::Float64)
    """Real Fourier derivative following 2x2 group matrix approach """
    
    # Stores RealFourier as [cos_0, cos_1, sin_1, cos_2, sin_2, ..., cos_nyq]
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
    
    # Choose implementation based on size
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
        # Vectorized implementation using broadcasting
        evaluate_real_fourier_derivative_vectorized!(result, operand, N, L, k_max, is_even)
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

function evaluate_real_fourier_derivative_vectorized!(result::ScalarField, operand::ScalarField, N::Int, L::Float64, k_max::Int, is_even::Bool)
    """Vectorized Real Fourier derivative implementation"""

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
    """Evaluate Chebyshev derivative using compatible differentiation matrix """
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
        # Single derivative
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
    """Evaluate Legendre derivative using compatible Jacobi implementation """
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
        # Single derivative
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
    Single Legendre derivative using Jacobi approach .
    
    Legendre polynomials are Jacobi polynomials with a=0, b=0.
    From Jacobi D(+1): bands = [(N + a + b + 1) * 2^(-1)]
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
    Following array:77-82 and apply_dense:104-126 implementation.
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
    Following apply_dense implementation in array:104-126.
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
    Following apply_sparse implementation in array:171-203.
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

# ============================================================================
# Expression matrices for matrix assembly (following operators)
# ============================================================================

"""
    expression_matrices(op::Operator, sp, vars; kwargs...)

Build expression matrices for operator applied to each variable.
Following operators expression_matrices method.

Returns Dict mapping variables to sparse matrices.
"""
function expression_matrices(op::Operator, sp, vars; kwargs...)
    # Default: return empty dict (override for specific operators)
    return Dict{Any, SparseMatrixCSC}()
end

"""
    expression_matrices(op::TimeDerivative, sp, vars; kwargs...)

Time derivative matrices: returns M matrix contribution.
Following operators TimeDerivative.expression_matrices.
"""
function expression_matrices(op::TimeDerivative, sp, vars; kwargs...)
    operand = op.operand
    result = Dict{Any, SparseMatrixCSC}()

    for var in vars
        if var === operand || (hasfield(typeof(var), :name) && hasfield(typeof(operand), :name) && var.name == operand.name)
            # Identity matrix for time derivative term
            n = field_dofs(var)
            result[var] = sparse(I, n, n) * Float64(op.order)
        end
    end

    return result
end

"""
    expression_matrices(op::Differentiate, sp, vars; kwargs...)

Spatial differentiation matrices.
Following operators Differentiate.expression_matrices.
"""
function expression_matrices(op::Differentiate, sp, vars; kwargs...)
    operand = op.operand
    coord = op.coord
    order = op.order
    result = Dict{Any, SparseMatrixCSC}()

    for var in vars
        if var === operand || (hasfield(typeof(var), :name) && hasfield(typeof(operand), :name) && var.name == operand.name)
            # Build differentiation matrix for this variable
            D = build_operator_differentiation_matrix(var, coord, order; kwargs...)
            if D !== nothing
                result[var] = D
            end
        end
    end

    return result
end

"""
    expression_matrices(op::Laplacian, sp, vars; kwargs...)

Laplacian matrices: sum of second derivatives.
Following operators Laplacian.expression_matrices.
"""
function expression_matrices(op::Laplacian, sp, vars; kwargs...)
    operand = op.operand
    result = Dict{Any, SparseMatrixCSC}()

    for var in vars
        if var === operand || (hasfield(typeof(var), :name) && hasfield(typeof(operand), :name) && var.name == operand.name)
            # Build Laplacian matrix = sum of D_i^2 for each coordinate
            lap_mat = nothing

            if hasfield(typeof(var), :bases)
                for basis in var.bases
                    if basis !== nothing
                        coord = get_coord_for_basis(basis)
                        D2 = build_operator_differentiation_matrix(var, coord, 2; kwargs...)
                        if D2 !== nothing
                            if lap_mat === nothing
                                lap_mat = D2
                            else
                                lap_mat = lap_mat + D2
                            end
                        end
                    end
                end
            end

            if lap_mat !== nothing
                result[var] = lap_mat
            end
        end
    end

    return result
end

"""
    expression_matrices(op::Gradient, sp, vars; kwargs...)

Gradient matrices for scalar -> vector.
Following operators Gradient.expression_matrices.
"""
function expression_matrices(op::Gradient, sp, vars; kwargs...)
    operand = op.operand
    coordsys = op.coordsys
    result = Dict{Any, SparseMatrixCSC}()

    for var in vars
        if var === operand || (hasfield(typeof(var), :name) && hasfield(typeof(operand), :name) && var.name == operand.name)
            # For Cartesian: gradient is vector of partial derivatives
            # Build block diagonal matrix with D_i for each component
            n = field_dofs(var)
            ndim = coordsys.dim

            blocks = SparseMatrixCSC[]
            for coord in coordsys.coords
                D = build_operator_differentiation_matrix(var, coord, 1; kwargs...)
                if D !== nothing
                    push!(blocks, D)
                else
                    push!(blocks, spzeros(Float64, n, n))
                end
            end

            # Stack blocks vertically for vector output
            if !isempty(blocks)
                result[var] = vcat(blocks...)
            end
        end
    end

    return result
end

"""
    expression_matrices(op::Divergence, sp, vars; kwargs...)

Divergence matrices for vector -> scalar.
Following operators Divergence.expression_matrices.
"""
function expression_matrices(op::Divergence, sp, vars; kwargs...)
    operand = op.operand
    result = Dict{Any, SparseMatrixCSC}()

    if !isa(operand, VectorField)
        return result
    end

    coordsys = operand.coordsys

    for var in vars
        if var === operand || (hasfield(typeof(var), :name) && var.name == operand.name)
            # For Cartesian: divergence is sum of partial derivatives of components
            # Build row of blocks [D_x, D_y, D_z]
            n_comp = length(operand.components)
            n_per_comp = n_comp > 0 ? field_dofs(operand.components[1]) : 0

            blocks = SparseMatrixCSC[]
            for (i, coord) in enumerate(coordsys.coords)
                comp = operand.components[i]
                D = build_operator_differentiation_matrix(comp, coord, 1; kwargs...)
                if D !== nothing
                    push!(blocks, D)
                else
                    push!(blocks, spzeros(Float64, n_per_comp, n_per_comp))
                end
            end

            # Concatenate blocks horizontally for scalar output
            if !isempty(blocks)
                result[var] = hcat(blocks...)
            end
        end
    end

    return result
end

"""
    expression_matrices(op::Lift, sp, vars; kwargs...)

Lift matrices for boundary conditions (tau method).
Following operators Lift.expression_matrices.
"""
function expression_matrices(op::Lift, sp, vars; kwargs...)
    operand = op.operand
    basis = op.basis
    n = op.n
    result = Dict{Any, SparseMatrixCSC}()

    for var in vars
        if var === operand || (hasfield(typeof(var), :name) && hasfield(typeof(operand), :name) && var.name == operand.name)
            # Lift matrix places tau values at specific spectral modes
            # Following tau method
            lift_mat = build_lift_matrix(var, basis, n; kwargs...)
            if lift_mat !== nothing
                result[var] = lift_mat
            end
        end
    end

    return result
end

"""
    expression_matrices(op::Convert, sp, vars; kwargs...)

Basis conversion matrices.
Following operators Convert.expression_matrices.
"""
function expression_matrices(op::Convert, sp, vars; kwargs...)
    operand = op.operand
    out_basis = op.basis
    result = Dict{Any, SparseMatrixCSC}()

    for var in vars
        if var === operand || (hasfield(typeof(var), :name) && hasfield(typeof(operand), :name) && var.name == operand.name)
            # Get input basis and build conversion matrix
            if hasfield(typeof(var), :bases) && !isempty(var.bases)
                in_basis = var.bases[1]
                if in_basis !== nothing && isa(in_basis, JacobiBasis) && isa(out_basis, JacobiBasis)
                    conv_mat = conversion_matrix(in_basis, out_basis)
                    result[var] = conv_mat
                end
            end
        end
    end

    return result
end

# ============================================================================
# Helper functions for building operator matrices
# ============================================================================

"""
    build_operator_differentiation_matrix(var, coord, order; kwargs...)

Build differentiation matrix for variable with respect to coordinate.
"""
function build_operator_differentiation_matrix(var, coord::Coordinate, order::Int; kwargs...)
    if !hasfield(typeof(var), :bases)
        return nothing
    end

    # Find the basis corresponding to this coordinate
    basis_idx = nothing
    target_basis = nothing

    for (i, basis) in enumerate(var.bases)
        if basis !== nothing && basis.meta.element_label == coord.name
            basis_idx = i
            target_basis = basis
            break
        end
    end

    if target_basis === nothing
        return nothing
    end

    n_total = field_dofs(var)
    n_basis = target_basis.meta.size

    # Build 1D differentiation matrix based on basis type
    D1d = nothing

    if isa(target_basis, JacobiBasis)
        D1d = differentiation_matrix(target_basis, order)
    elseif isa(target_basis, FourierBasis)
        D1d = fourier_differentiation_matrix(target_basis, order)
    end

    if D1d === nothing
        return nothing
    end

    # For multi-dimensional fields, apply Kronecker product
    if length(var.bases) == 1
        return D1d
    else
        # Build identity matrices for other dimensions
        matrices = AbstractMatrix[]
        for (i, basis) in enumerate(var.bases)
            if basis === nothing
                continue
            end
            if i == basis_idx
                push!(matrices, D1d)
            else
                push!(matrices, sparse(I, basis.meta.size, basis.meta.size))
            end
        end

        # Kronecker product in reverse order (column-major)
        result = matrices[end]
        for i in (length(matrices)-1):-1:1
            result = kron(result, matrices[i])
        end

        return result
    end
end

"""
    fourier_differentiation_matrix(basis::FourierBasis, order::Int)

Build Fourier differentiation matrix.
"""
function fourier_differentiation_matrix(basis::RealFourier, order::Int)
    N = basis.meta.size
    L = basis.meta.bounds[2] - basis.meta.bounds[1]
    k0 = 2π / L

    # RealFourier differentiation follows Dedalus convention:
    # Modes: [cos(0x), cos(1x), -sin(1x), cos(2x), -sin(2x), ...]
    # Note: Using -sin (msin) convention
    #
    # For differentiation with (ik)^order factor:
    # d/dx cos(kx) = -k sin(kx) = k * (-sin(kx))  -> k * msin
    # d/dx (-sin(kx)) = -k cos(kx)                -> -k * cos
    #
    # The 2x2 block for each wavenumber k is:
    # | 0  -k |   (maps: cos <- -k*msin, msin <- k*cos)
    # | k   0 |
    #
    # For order n, we apply this matrix n times, or equivalently compute (ik)^n
    # and extract the real/imaginary parts for the rotation.

    I_list = Int[]
    J_list = Int[]
    V_list = Float64[]

    # DC mode (k=0) -> 0 for any derivative order
    push!(I_list, 1)
    push!(J_list, 1)
    push!(V_list, 0.0)

    k_max = (N - 1) ÷ 2

    for k in 1:k_max
        cos_idx = 2*k      # 1-indexed: mode 2 is cos(1x), mode 4 is cos(2x), etc.
        sin_idx = 2*k + 1  # 1-indexed: mode 3 is -sin(1x), mode 5 is -sin(2x), etc.
        k_phys = k0 * k

        if cos_idx <= N && sin_idx <= N
            # Compute (ik)^order = k^order * i^order
            # i^0 = 1, i^1 = i, i^2 = -1, i^3 = -i, i^4 = 1, ...
            # For real representation: (ik)^n = k^n * (cos(nπ/2) + i*sin(nπ/2))
            #
            # The 2x2 derivative block D^n for the (cos, msin) pair:
            # D^1 = | 0  -k |    D^2 = |-k²  0  |    D^3 = | 0   k³ |   D^4 = |k⁴  0 |
            #       | k   0 |          | 0  -k² |          |-k³  0  |         | 0  k⁴|
            #
            # Pattern: D^n has form k^n * |cos(nπ/2)  -sin(nπ/2)|
            #                             |sin(nπ/2)   cos(nπ/2)|

            kn = k_phys^order
            phase = order * π / 2
            c = cos(phase)
            s = sin(phase)

            # Matrix entries for the 2x2 block:
            # cos_out = c * k^n * cos_in - s * k^n * msin_in
            # msin_out = s * k^n * cos_in + c * k^n * msin_in

            # (cos_idx, cos_idx): c * k^n
            if abs(c * kn) > 1e-15
                push!(I_list, cos_idx); push!(J_list, cos_idx); push!(V_list, c * kn)
            end
            # (cos_idx, sin_idx): -s * k^n
            if abs(s * kn) > 1e-15
                push!(I_list, cos_idx); push!(J_list, sin_idx); push!(V_list, -s * kn)
            end
            # (sin_idx, cos_idx): s * k^n
            if abs(s * kn) > 1e-15
                push!(I_list, sin_idx); push!(J_list, cos_idx); push!(V_list, s * kn)
            end
            # (sin_idx, sin_idx): c * k^n
            if abs(c * kn) > 1e-15
                push!(I_list, sin_idx); push!(J_list, sin_idx); push!(V_list, c * kn)
            end
        end
    end

    if isempty(I_list)
        return spzeros(Float64, N, N)
    end

    return sparse(I_list, J_list, V_list, N, N)
end

function fourier_differentiation_matrix(basis::ComplexFourier, order::Int)
    N = basis.meta.size
    L = basis.meta.bounds[2] - basis.meta.bounds[1]
    k0 = 2π / L

    # ComplexFourier: diagonal matrix with (ik)^order
    k_native = [0:(N÷2-1); -(N÷2):-1]
    k_phys = k0 .* k_native

    diag_vals = (im .* k_phys).^order

    return spdiagm(0 => diag_vals)
end

"""
    build_lift_matrix(var, basis, n; kwargs...)

Build lifting matrix for tau method boundary conditions.
Following the standard basis LiftJacobi implementation (lines 790-814).

the standard convention:
- n >= 0: sets mode n directly (0-indexed convention, 1-indexed in Julia)
- n < 0: wraps around (n = -1 means last mode, n = -2 means second-to-last, etc.)

Example: For N=10 modes
- Lift(tau, basis, 0) → sets mode 1 (Julia 1-indexed)
- Lift(tau, basis, -1) → sets mode N (last mode)
- Lift(tau, basis, -2) → sets mode N-1 (second-to-last mode)
"""
function build_lift_matrix(var, basis, n::Int; kwargs...)
    if !hasfield(typeof(var), :bases)
        return nothing
    end

    # Find the basis index
    basis_idx = nothing
    for (i, b) in enumerate(var.bases)
        if b === basis
            basis_idx = i
            break
        end
    end

    if basis_idx === nothing
        # Try matching by name
        for (i, b) in enumerate(var.bases)
            if b !== nothing && basis.meta.element_label == b.meta.element_label
                basis_idx = i
                break
            end
        end
    end

    if basis_idx === nothing
        return nothing
    end

    N = basis.meta.size

    # Following the standard convention: direct indexing with negative wrap-around
    # In the standard (0-indexed) convention: P['c'][axslice(axis, n, n+1)] = 1
    # In Julia (1-indexed): we add 1 to convert from 0-indexed to 1-indexed
    lift_mode = n
    if lift_mode < 0
        lift_mode = N + lift_mode  # Wrap negative indices (e.g., -1 → N-1 in 0-indexed → N in 1-indexed)
    end
    lift_mode += 1  # Convert from 0-indexed to 1-indexed

    I_list = Int[]
    J_list = Int[]
    V_list = Float64[]

    if lift_mode >= 1 && lift_mode <= N
        push!(I_list, lift_mode)
        push!(J_list, 1)
        push!(V_list, 1.0)
    end

    if isempty(I_list)
        return spzeros(Float64, N, 1)
    end

    return sparse(I_list, J_list, V_list, N, 1)
end

"""
    get_coord_for_basis(basis::Basis)

Get coordinate associated with a basis.
"""
function get_coord_for_basis(basis::Basis)
    if hasfield(typeof(basis), :meta) && hasfield(typeof(basis.meta), :coordsys)
        coordsys = basis.meta.coordsys
        coord_name = basis.meta.element_label
        if hasfield(typeof(coordsys), :coords)
            for coord in coordsys.coords
                if coord.name == coord_name
                    return coord
                end
            end
        end
    end
    return nothing
end

"""
    field_dofs(field)

Get total degrees of freedom for a field.
"""
function field_dofs(field)
    if hasfield(typeof(field), :data_c) && field.data_c !== nothing
        return length(field.data_c)
    elseif hasfield(typeof(field), :data_g) && field.data_g !== nothing
        return length(field.data_g)
    elseif hasfield(typeof(field), :bases)
        total = 1
        for basis in field.bases
            if basis !== nothing
                total *= basis.meta.size
            end
        end
        return total
    elseif hasfield(typeof(field), :components)
        # VectorField or TensorField
        return sum(field_dofs(comp) for comp in field.components)
    end
    return 0
end

# ============================================================================
# Utility Operator Evaluation Functions
# Following operators implementation
# ============================================================================

"""
    evaluate_interpolate(interp_op::Interpolate, layout::Symbol=:g)

Evaluate interpolation operator at a specific position along a coordinate.
Following operators Interpolate implementation.

For Fourier bases: uses spectral interpolation (sum of modes)
For Jacobi bases: uses barycentric interpolation or Clenshaw algorithm
"""
function evaluate_interpolate(interp_op::Interpolate, layout::Symbol=:g)
    operand = interp_op.operand
    coord = interp_op.coord
    position = interp_op.position

    if !isa(operand, ScalarField)
        throw(ArgumentError("Interpolate currently only supports scalar fields"))
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

    # Work in coefficient space for spectral interpolation
    ensure_layout!(operand, :c)

    if isa(basis, RealFourier) || isa(basis, ComplexFourier)
        return interpolate_fourier(operand, basis, basis_index, position, layout)
    elseif isa(basis, JacobiBasis)
        return interpolate_jacobi(operand, basis, basis_index, position, layout)
    else
        throw(ArgumentError("Interpolation not implemented for basis type $(typeof(basis))"))
    end
end

"""
    interpolate_fourier(field, basis, axis, position, layout)

Interpolate Fourier field at a specific position using spectral reconstruction.
f(x) = Σ c_k exp(i k x) for ComplexFourier
f(x) = a_0 + Σ (a_k cos(kx) + b_k sin(kx)) for RealFourier
"""
function interpolate_fourier(field::ScalarField, basis::FourierBasis, axis::Int, position::Real, layout::Symbol)
    N = basis.meta.size
    L = basis.meta.bounds[2] - basis.meta.bounds[1]
    x0 = basis.meta.bounds[1]

    # Normalize position to [0, L)
    x = mod(position - x0, L)

    # Get coefficient data
    coeffs = field.data_c

    if isa(basis, RealFourier)
        # RealFourier: [a_0, a_1, b_1, a_2, b_2, ..., a_nyq]
        result = coeffs[1]  # DC component

        k_max = N ÷ 2
        is_even = (N % 2 == 0)

        for k in 1:(k_max - (is_even ? 1 : 0))
            k_phys = 2π * k / L
            cos_idx = 2*k
            sin_idx = 2*k + 1

            if cos_idx <= length(coeffs) && sin_idx <= length(coeffs)
                result += coeffs[cos_idx] * cos(k_phys * x)
                result += coeffs[sin_idx] * sin(k_phys * x)
            end
        end

        # Nyquist component for even N
        if is_even && N <= length(coeffs)
            k_nyq = 2π * k_max / L
            result += coeffs[N] * cos(k_nyq * x)
        end

    else  # ComplexFourier
        # ComplexFourier: standard FFT ordering [0, 1, ..., N/2-1, -N/2, ..., -1]
        result = complex(0.0, 0.0)

        for i in 1:N
            if i <= N÷2
                k = i - 1
            else
                k = i - N - 1
            end
            k_phys = 2π * k / L
            result += coeffs[i] * exp(im * k_phys * x)
        end

        result = real(result)
    end

    return result
end

"""
    interpolate_jacobi(field, basis, axis, position, layout)

Interpolate Jacobi-type field (Chebyshev, Legendre) using Clenshaw algorithm.
"""
function interpolate_jacobi(field::ScalarField, basis::JacobiBasis, axis::Int, position::Real, layout::Symbol)
    N = basis.meta.size
    a, b = basis.meta.bounds

    # Map position to native [-1, 1] interval
    x_native = 2.0 * (position - a) / (b - a) - 1.0

    # Clamp to valid range
    x_native = clamp(x_native, -1.0, 1.0)

    # Get coefficient data
    coeffs = field.data_c

    if isa(basis, ChebyshevT)
        # Clenshaw algorithm for Chebyshev T_n
        return clenshaw_chebyshev_t(coeffs, x_native)
    elseif isa(basis, ChebyshevU)
        # Clenshaw algorithm for Chebyshev U_n
        return clenshaw_chebyshev_u(coeffs, x_native)
    elseif isa(basis, Legendre)
        # Clenshaw algorithm for Legendre P_n
        return clenshaw_legendre(coeffs, x_native)
    else
        # General Jacobi: use direct polynomial evaluation
        return clenshaw_jacobi(coeffs, x_native, basis.a, basis.b)
    end
end

"""
    clenshaw_chebyshev_t(coeffs, x)

Clenshaw algorithm for evaluating Chebyshev T polynomial sum.
T_n(x) satisfies: T_{n+1}(x) = 2x T_n(x) - T_{n-1}(x)
"""
function clenshaw_chebyshev_t(coeffs::AbstractVector, x::Real)
    n = length(coeffs)
    if n == 0
        return 0.0
    elseif n == 1
        return coeffs[1]
    end

    # Backward recurrence: b_k = c_k + 2x b_{k+1} - b_{k+2}
    b_k2 = 0.0  # b_{n+1}
    b_k1 = 0.0  # b_{n}

    for k in n:-1:2
        b_k = coeffs[k] + 2x * b_k1 - b_k2
        b_k2 = b_k1
        b_k1 = b_k
    end

    # Final step: f(x) = c_0 + x*b_1 - b_2
    return coeffs[1] + x * b_k1 - b_k2
end

"""
    clenshaw_chebyshev_u(coeffs, x)

Clenshaw algorithm for evaluating Chebyshev U polynomial sum.
U_n(x) satisfies: U_{n+1}(x) = 2x U_n(x) - U_{n-1}(x)
"""
function clenshaw_chebyshev_u(coeffs::AbstractVector, x::Real)
    n = length(coeffs)
    if n == 0
        return 0.0
    elseif n == 1
        return coeffs[1]
    end

    # Same recurrence as T_n
    b_k2 = 0.0
    b_k1 = 0.0

    for k in n:-1:2
        b_k = coeffs[k] + 2x * b_k1 - b_k2
        b_k2 = b_k1
        b_k1 = b_k
    end

    # For U_n: f(x) = c_0 * U_0(x) + b_1 * U_1(x) - b_2 * U_0(x)
    # where U_0(x) = 1, U_1(x) = 2x
    return coeffs[1] + 2x * b_k1 - b_k2
end

"""
    clenshaw_legendre(coeffs, x)

Clenshaw algorithm for evaluating Legendre polynomial sum.
P_n(x) satisfies: (n+1) P_{n+1}(x) = (2n+1) x P_n(x) - n P_{n-1}(x)
"""
function clenshaw_legendre(coeffs::AbstractVector, x::Real)
    n = length(coeffs)
    if n == 0
        return 0.0
    elseif n == 1
        return coeffs[1]
    end

    # Backward recurrence for Legendre
    b_k2 = 0.0
    b_k1 = 0.0

    for k in n:-1:2
        # Recurrence: b_k = c_k + ((2k-1)/(k)) x b_{k+1} - (k/(k+1)) b_{k+2}
        # Note: k here is 0-indexed polynomial degree, so adjust
        deg = k - 1  # 0-indexed degree
        alpha = (2*deg + 1) / (deg + 1) * x
        beta = deg / (deg + 1)
        b_k = coeffs[k] + alpha * b_k1 - beta * b_k2
        b_k2 = b_k1
        b_k1 = b_k
    end

    # Final step
    return coeffs[1] + x * b_k1 - 0.5 * b_k2
end

"""
    clenshaw_jacobi(coeffs, x, a, b)

Clenshaw algorithm for evaluating general Jacobi polynomial sum.
"""
function clenshaw_jacobi(coeffs::AbstractVector, x::Real, a::Float64, b::Float64)
    n = length(coeffs)
    if n == 0
        return 0.0
    elseif n == 1
        return coeffs[1]
    end

    # Use direct evaluation for general Jacobi (less efficient but correct)
    result = 0.0
    for k in 1:n
        result += coeffs[k] * jacobi_polynomial(k-1, a, b, x)
    end
    return result
end

"""
    jacobi_polynomial(n, a, b, x)

Evaluate Jacobi polynomial P_n^{(a,b)}(x) using three-term recurrence.
"""
function jacobi_polynomial(n::Int, a::Float64, b::Float64, x::Real)
    if n == 0
        return 1.0
    elseif n == 1
        return 0.5 * (a - b + (a + b + 2) * x)
    end

    p_km2 = 1.0
    p_km1 = 0.5 * (a - b + (a + b + 2) * x)

    for k in 2:n
        # Three-term recurrence for Jacobi polynomials
        k_f = Float64(k)
        a1 = 2 * k_f * (k_f + a + b) * (2*k_f + a + b - 2)
        a2 = (2*k_f + a + b - 1) * (a^2 - b^2)
        a3 = (2*k_f + a + b - 2) * (2*k_f + a + b - 1) * (2*k_f + a + b)
        a4 = 2 * (k_f + a - 1) * (k_f + b - 1) * (2*k_f + a + b)

        p_k = ((a2 + a3 * x) * p_km1 - a4 * p_km2) / a1
        p_km2 = p_km1
        p_km1 = p_k
    end

    return p_km1
end

"""
    evaluate_integrate(int_op::Integrate, layout::Symbol=:g)

Evaluate integration operator over specified coordinate(s).
Following operators Integrate implementation.

Uses appropriate quadrature weights for each basis type:
- Fourier: trapezoidal rule (uniform weights)
- Chebyshev: Clenshaw-Curtis quadrature
- Legendre: Gauss-Legendre quadrature
"""
function evaluate_integrate(int_op::Integrate, layout::Symbol=:g)
    operand = int_op.operand
    coord = int_op.coord

    if !isa(operand, ScalarField)
        throw(ArgumentError("Integrate currently only supports scalar fields"))
    end

    # Handle single coordinate or tuple of coordinates
    coords = isa(coord, Coordinate) ? (coord,) : coord

    # Start with the operand
    result_field = copy(operand)

    # Integrate over each coordinate in sequence
    for c in coords
        result_field = integrate_along_coord(result_field, c)
    end

    # If all dimensions integrated, return scalar
    if length(coords) == length(operand.bases)
        ensure_layout!(result_field, :g)
        return sum(result_field.data_g)
    end

    if layout == :g
        ensure_layout!(result_field, :g)
    else
        ensure_layout!(result_field, :c)
    end

    return result_field
end

"""
    integrate_along_coord(field, coord)

Integrate field along a single coordinate using appropriate quadrature.
"""
function integrate_along_coord(field::ScalarField, coord::Coordinate)
    # Find which basis corresponds to this coordinate
    basis_index = nothing
    for (i, basis) in enumerate(field.bases)
        if basis.meta.element_label == coord.name
            basis_index = i
            break
        end
    end

    if basis_index === nothing
        throw(ArgumentError("Coordinate $(coord.name) not found in field bases"))
    end

    basis = field.bases[basis_index]

    # Work in grid space for integration
    ensure_layout!(field, :g)

    # Get quadrature weights for this basis
    weights = get_integration_weights(basis)

    # Apply weighted sum along the axis
    data = field.data_g

    # Sum along the specified axis with weights
    result_data = integrate_weighted_sum(data, weights, basis_index)

    # Create result field with reduced dimensionality
    # For now, return scalar for 1D or the summed array
    if ndims(data) == 1
        return sum(data .* weights)
    else
        # Create new field without the integrated dimension
        new_bases = [b for (i, b) in enumerate(field.bases) if i != basis_index]
        if isempty(new_bases)
            return sum(data .* reshape(weights, size_for_axis(length(weights), basis_index, ndims(data))))
        end

        result = ScalarField(field.dist, "int_$(field.name)", tuple(new_bases...), field.dtype)
        result.data_g = result_data
        result.current_layout = :g
        return result
    end
end

"""
    get_integration_weights(basis)

Get quadrature weights for integration over a basis.
"""
function get_integration_weights(basis::Basis)
    N = basis.meta.size
    a, b = basis.meta.bounds
    L = b - a

    if isa(basis, RealFourier) || isa(basis, ComplexFourier)
        # Uniform weights for periodic Fourier
        return fill(L / N, N)

    elseif isa(basis, ChebyshevT)
        # Clenshaw-Curtis quadrature weights
        return clenshaw_curtis_weights(N, L)

    elseif isa(basis, Legendre)
        # Gauss-Legendre quadrature weights
        _, weights = gauss_legendre_quadrature(N)
        return weights .* (L / 2)  # Scale from [-1,1] to [a,b]

    else
        # Default: uniform weights
        return fill(L / N, N)
    end
end

"""
    clenshaw_curtis_weights(N, L)

Compute Clenshaw-Curtis quadrature weights for Chebyshev integration.
"""
function clenshaw_curtis_weights(N::Int, L::Float64)
    weights = zeros(N)

    if N == 1
        return [L]
    end

    # Clenshaw-Curtis weights on [-1, 1]
    for j in 1:N
        theta_j = π * (j - 1) / (N - 1)
        w = 0.0
        for k in 0:(N÷2)
            if k == 0 || k == N÷2
                factor = 1.0
            else
                factor = 2.0
            end
            if 2*k != N - 1
                w += factor * cos(2*k * theta_j) / (1 - 4*k^2)
            end
        end
        weights[j] = 2 * w / (N - 1)
    end

    # Handle endpoints
    weights[1] *= 0.5
    weights[N] *= 0.5

    # Scale to interval [a, b]
    return weights .* (L / 2)
end

"""
    gauss_legendre_quadrature(N)

Compute Gauss-Legendre quadrature points and weights on [-1, 1].
"""
function gauss_legendre_quadrature(N::Int)
    points = zeros(N)
    weights = zeros(N)

    # Initial guesses for roots using Chebyshev nodes
    for i in 1:N
        points[i] = -cos(π * (i - 0.25) / (N + 0.5))
    end

    # Newton-Raphson iteration for roots
    for i in 1:N
        x = points[i]
        for _ in 1:100  # Max iterations
            # Evaluate P_N and P_{N-1} using recurrence
            p_km2 = 1.0
            p_km1 = x

            for k in 2:N
                p_k = ((2*k - 1) * x * p_km1 - (k - 1) * p_km2) / k
                p_km2 = p_km1
                p_km1 = p_k
            end

            # Derivative: P'_N = N(x P_N - P_{N-1}) / (x^2 - 1)
            dp = N * (x * p_km1 - p_km2) / (x^2 - 1)

            # Newton update
            dx = -p_km1 / dp
            x += dx

            if abs(dx) < 1e-14
                break
            end
        end

        points[i] = x

        # Compute weight
        p_km2 = 1.0
        p_km1 = x
        for k in 2:N
            p_k = ((2*k - 1) * x * p_km1 - (k - 1) * p_km2) / k
            p_km2 = p_km1
            p_km1 = p_k
        end
        dp = N * (x * p_km1 - p_km2) / (x^2 - 1)
        weights[i] = 2 / ((1 - x^2) * dp^2)
    end

    return points, weights
end

"""
    integrate_weighted_sum(data, weights, axis)

Apply weighted sum along specified axis.
"""
function integrate_weighted_sum(data::AbstractArray, weights::AbstractVector, axis::Int)
    nd = ndims(data)

    if nd == 1
        return sum(data .* weights)
    end

    # Reshape weights to broadcast along the correct axis
    shape = ones(Int, nd)
    shape[axis] = length(weights)
    w_shaped = reshape(weights, shape...)

    # Weighted sum along axis
    return dropdims(sum(data .* w_shaped, dims=axis), dims=axis)
end

"""
    size_for_axis(n, axis, ndims)

Create a size tuple with n in the specified axis position and 1 elsewhere.
"""
function size_for_axis(n::Int, axis::Int, nd::Int)
    shape = ones(Int, nd)
    shape[axis] = n
    return tuple(shape...)
end

"""
    evaluate_average(avg_op::Average, layout::Symbol=:g)

Evaluate averaging operator along a coordinate.
Average = Integrate / (interval length)
"""
function evaluate_average(avg_op::Average, layout::Symbol=:g)
    operand = avg_op.operand
    coord = avg_op.coord

    if !isa(operand, ScalarField)
        throw(ArgumentError("Average currently only supports scalar fields"))
    end

    # Find the basis for this coordinate
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
    L = basis.meta.bounds[2] - basis.meta.bounds[1]

    # Integrate and divide by interval length
    int_result = integrate_along_coord(operand, coord)

    if isa(int_result, Real)
        return int_result / L
    else
        # Scale the field data
        if int_result.data_g !== nothing
            int_result.data_g ./= L
        end
        if int_result.data_c !== nothing
            int_result.data_c ./= L
        end
        return int_result
    end
end

"""
    evaluate_lift(lift_op::Lift, layout::Symbol=:g)

Evaluate lifting operator for tau method boundary conditions.
Following the standard basis LiftJacobi implementation (lines 790-814).

the standard convention:
- n >= 0: sets mode n directly (0-indexed convention, 1-indexed in Julia)
- n < 0: wraps around (n = -1 means last mode, n = -2 means second-to-last, etc.)

This creates a polynomial field P with coefficient n set to 1, then returns P * operand.
"""
function evaluate_lift(lift_op::Lift, layout::Symbol=:g)
    operand = lift_op.operand
    basis = lift_op.basis
    n = lift_op.n

    if !isa(operand, ScalarField)
        throw(ArgumentError("Lift currently only supports scalar fields"))
    end

    # Find the basis index
    basis_index = nothing
    for (i, b) in enumerate(operand.bases)
        if b === basis || b.meta.element_label == basis.meta.element_label
            basis_index = i
            break
        end
    end

    if basis_index === nothing
        throw(ArgumentError("Basis not found in operand"))
    end

    # Create result field with same structure
    result = ScalarField(operand.dist, "lift_$(operand.name)", operand.bases, operand.dtype)

    # Work in coefficient space
    ensure_layout!(operand, :c)
    ensure_layout!(result, :c)

    N = basis.meta.size

    # Zero out result
    fill!(result.data_c, 0.0)

    # Following the standard convention: direct indexing with negative wrap-around
    # In the standard (0-indexed) convention: P['c'][axslice(axis, n, n+1)] = 1
    # In Julia (1-indexed): we add 1 to convert from 0-indexed to 1-indexed
    lift_mode = n
    if lift_mode < 0
        lift_mode = N + lift_mode  # Wrap negative indices (e.g., -1 → N-1 in 0-indexed → N in 1-indexed)
    end
    lift_mode += 1  # Convert from 0-indexed to 1-indexed

    if lift_mode >= 1 && lift_mode <= N
        if ndims(operand.data_c) == 1
            result.data_c[lift_mode] = operand.data_c[1]
        else
            # For multi-dimensional fields, apply along the appropriate axis
            apply_lift_nd!(result.data_c, operand.data_c, basis_index, lift_mode)
        end
    end

    if layout == :g
        backward_transform!(result)
    end

    return result
end

"""
    apply_lift_nd!(result, operand, axis, lift_mode)

Apply lift operation along specified axis for multi-dimensional arrays.
"""
function apply_lift_nd!(result::AbstractArray, operand::AbstractArray, axis::Int, lift_mode::Int)
    nd = ndims(result)

    # Use selectdim to access the lift mode slice
    selectdim(result, axis, lift_mode) .= selectdim(operand, axis, 1)
end

"""
    evaluate_convert(conv_op::Convert, layout::Symbol=:g)

Evaluate basis conversion operator.
Following operators Convert implementation.

Converts field from one basis representation to another using
spectral conversion matrices.
"""
function evaluate_convert(conv_op::Convert, layout::Symbol=:g)
    operand = conv_op.operand
    out_basis = conv_op.basis

    if !isa(operand, ScalarField)
        throw(ArgumentError("Convert currently only supports scalar fields"))
    end

    # Find the input basis to convert
    in_basis_index = nothing
    in_basis = nothing

    for (i, b) in enumerate(operand.bases)
        if b !== nothing && isa(b, JacobiBasis) && isa(out_basis, JacobiBasis)
            # Check if bases are on same coordinate
            if b.meta.element_label == out_basis.meta.element_label
                in_basis_index = i
                in_basis = b
                break
            end
        end
    end

    if in_basis === nothing
        # No conversion needed or not applicable
        return copy(operand)
    end

    # Work in coefficient space
    ensure_layout!(operand, :c)

    # Build or retrieve conversion matrix
    conv_mat = conversion_matrix(in_basis, out_basis)

    # Create result field
    new_bases = collect(operand.bases)
    new_bases[in_basis_index] = out_basis
    result = ScalarField(operand.dist, "conv_$(operand.name)", tuple(new_bases...), operand.dtype)
    ensure_layout!(result, :c)

    # Apply conversion matrix
    if ndims(operand.data_c) == 1
        result.data_c .= conv_mat * operand.data_c
    else
        result.data_c .= apply_matrix_along_axis(conv_mat, operand.data_c, in_basis_index)
    end

    if layout == :g
        backward_transform!(result)
    end

    return result
end

# ============================================================================
# Power and General Function Operators
# ============================================================================

"""
    Power <: Operator

Nonlinear power operator: f^p
"""
struct Power <: Operator
    operand::Operand
    power::Real
end
const _Power_constructor = Power

"""
    power(operand::Operand, p::Real)

Raise operand to power p.
"""
function power(operand::Operand, p::Real)
    return multiclass_new(Power, operand, p)
end

register_operator_alias!(power, "power", "pow")
register_operator_parseable!(power, "power", "pow")

"""
    evaluate_power(pow_op::Power, layout::Symbol=:g)

Evaluate power operator in grid space.
"""
function evaluate_power(pow_op::Power, layout::Symbol=:g)
    operand = pow_op.operand
    p = pow_op.power

    if !isa(operand, ScalarField)
        throw(ArgumentError("Power currently only supports scalar fields"))
    end

    # Work in grid space for nonlinear operations
    ensure_layout!(operand, :g)

    # Create result field
    result = ScalarField(operand.dist, "pow_$(operand.name)_$p", operand.bases, operand.dtype)
    ensure_layout!(result, :g)

    # Apply power operation
    result.data_g .= operand.data_g .^ p

    if layout == :c
        forward_transform!(result)
    end

    return result
end

function dispatch_check(::Type{Power}, args::Tuple, kwargs::NamedTuple)
    operand, p = args
    if !isa(operand, Operand)
        throw(ArgumentError("Power requires an Operand"))
    end
    if !isa(p, Real)
        throw(ArgumentError("Power exponent must be real"))
    end
    return true
end

function invoke_constructor(::Type{Power}, args::Tuple, kwargs::NamedTuple)
    operand, p = args
    return _Power_constructor(operand, p)
end

"""
    GeneralFunction <: Operator

Apply general function to field in grid space.
"""
struct GeneralFunction <: Operator
    operand::Operand
    func::Function
    name::String
end
const _GeneralFunction_constructor = GeneralFunction

"""
    apply_function(operand::Operand, f::Function, name::String="func")

Apply arbitrary function to operand in grid space.
"""
function apply_function(operand::Operand, f::Function, name::String="func")
    return multiclass_new(GeneralFunction, operand, f, name)
end

register_operator_alias!(apply_function, "apply_function", "apply_func")
register_operator_parseable!(apply_function, "apply_function", "apply_func")

"""
    evaluate_general_function(gf_op::GeneralFunction, layout::Symbol=:g)

Evaluate general function operator in grid space.
"""
function evaluate_general_function(gf_op::GeneralFunction, layout::Symbol=:g)
    operand = gf_op.operand
    f = gf_op.func
    name = gf_op.name

    if !isa(operand, ScalarField)
        throw(ArgumentError("GeneralFunction currently only supports scalar fields"))
    end

    # Work in grid space
    ensure_layout!(operand, :g)

    # Create result field
    result = ScalarField(operand.dist, "$(name)_$(operand.name)", operand.bases, operand.dtype)
    ensure_layout!(result, :g)

    # Apply function element-wise
    result.data_g .= f.(operand.data_g)

    if layout == :c
        forward_transform!(result)
    end

    return result
end

function dispatch_check(::Type{GeneralFunction}, args::Tuple, kwargs::NamedTuple)
    operand, f, name = args
    if !isa(operand, Operand)
        throw(ArgumentError("GeneralFunction requires an Operand"))
    end
    if !isa(f, Function)
        throw(ArgumentError("GeneralFunction requires a Function"))
    end
    return true
end

function invoke_constructor(::Type{GeneralFunction}, args::Tuple, kwargs::NamedTuple)
    operand, f, name = args
    return _GeneralFunction_constructor(operand, f, name)
end

"""
    UnaryGridFunction <: Operator

Apply unary numpy-style function (sin, cos, exp, etc.) to field.
"""
struct UnaryGridFunction <: Operator
    operand::Operand
    func::Function
    name::String
end
const _UnaryGridFunction_constructor = UnaryGridFunction

# Common unary functions following
function sin_field(operand::Operand)
    return UnaryGridFunction(operand, sin, "sin")
end

function cos_field(operand::Operand)
    return UnaryGridFunction(operand, cos, "cos")
end

function tan_field(operand::Operand)
    return UnaryGridFunction(operand, tan, "tan")
end

function exp_field(operand::Operand)
    return UnaryGridFunction(operand, exp, "exp")
end

function log_field(operand::Operand)
    return UnaryGridFunction(operand, log, "log")
end

function sqrt_field(operand::Operand)
    return UnaryGridFunction(operand, sqrt, "sqrt")
end

function abs_field(operand::Operand)
    return UnaryGridFunction(operand, abs, "abs")
end

function tanh_field(operand::Operand)
    return UnaryGridFunction(operand, tanh, "tanh")
end

# Register unary functions
register_operator_parseable!(sin_field, "sin")
register_operator_parseable!(cos_field, "cos")
register_operator_parseable!(tan_field, "tan")
register_operator_parseable!(exp_field, "exp")
register_operator_parseable!(log_field, "log")
register_operator_parseable!(sqrt_field, "sqrt")
register_operator_parseable!(abs_field, "abs")
register_operator_parseable!(tanh_field, "tanh")

"""
    evaluate_unary_grid_function(ugf_op::UnaryGridFunction, layout::Symbol=:g)

Evaluate unary grid function operator.
"""
function evaluate_unary_grid_function(ugf_op::UnaryGridFunction, layout::Symbol=:g)
    return evaluate_general_function(
        GeneralFunction(ugf_op.operand, ugf_op.func, ugf_op.name),
        layout
    )
end

# ============================================================================
# Grid and Coeff conversion evaluation
# ============================================================================

"""
    evaluate_grid(grid_op::Grid)

Convert operand to grid space.
"""
function evaluate_grid(grid_op::Grid)
    operand = grid_op.operand

    if isa(operand, ScalarField)
        ensure_layout!(operand, :g)
        return operand
    else
        throw(ArgumentError("Grid conversion not implemented for $(typeof(operand))"))
    end
end

"""
    evaluate_coeff(coeff_op::Coeff)

Convert operand to coefficient space.
"""
function evaluate_coeff(coeff_op::Coeff)
    operand = coeff_op.operand

    if isa(operand, ScalarField)
        ensure_layout!(operand, :c)
        return operand
    else
        throw(ArgumentError("Coeff conversion not implemented for $(typeof(operand))"))
    end
end

# ============================================================================
# Component extraction evaluation
# ============================================================================

"""
    evaluate_component(comp_op::Component)

Extract specific component from vector/tensor field.
"""
function evaluate_component(comp_op::Component)
    operand = comp_op.operand
    index = comp_op.index

    if isa(operand, VectorField)
        if index < 1 || index > length(operand.components)
            throw(BoundsError("Component index $index out of bounds"))
        end
        return operand.components[index]

    elseif isa(operand, TensorField)
        # For tensors, index could be linear or we need (i,j)
        # Use linear indexing for now
        if index < 1 || index > length(operand.components)
            throw(BoundsError("Component index $index out of bounds"))
        end
        return operand.components[index]

    else
        throw(ArgumentError("Component extraction requires VectorField or TensorField"))
    end
end

"""
    evaluate_radial_component(rc_op::RadialComponent)

Extract radial component from vector field.
For Cartesian coordinates, this is the x-component.
"""
function evaluate_radial_component(rc_op::RadialComponent)
    operand = rc_op.operand

    if !isa(operand, VectorField)
        throw(ArgumentError("RadialComponent requires a VectorField"))
    end

    # In Cartesian, "radial" is typically the first component
    # For proper polar/spherical, would need coordinate system info
    return operand.components[1]
end

"""
    evaluate_angular_component(ac_op::AngularComponent)

Extract angular component from vector field.
For Cartesian 2D, this is the y-component.
"""
function evaluate_angular_component(ac_op::AngularComponent)
    operand = ac_op.operand

    if !isa(operand, VectorField)
        throw(ArgumentError("AngularComponent requires a VectorField"))
    end

    if length(operand.components) < 2
        throw(ArgumentError("VectorField must have at least 2 components"))
    end

    return operand.components[2]
end

"""
    evaluate_azimuthal_component(az_op::AzimuthalComponent)

Extract azimuthal component from vector field.
For Cartesian 3D, this is the z-component.
"""
function evaluate_azimuthal_component(az_op::AzimuthalComponent)
    operand = az_op.operand

    if !isa(operand, VectorField)
        throw(ArgumentError("AzimuthalComponent requires a VectorField"))
    end

    if length(operand.components) < 3
        throw(ArgumentError("VectorField must have at least 3 components"))
    end

    return operand.components[3]
end

# ============================================================================
# Trace and Skew evaluation for tensors
# ============================================================================

"""
    evaluate_trace(trace_op::Trace, layout::Symbol=:g)

Evaluate trace of a tensor field.
trace(T) = Σ_i T_ii
"""
function evaluate_trace(trace_op::Trace, layout::Symbol=:g)
    operand = trace_op.operand

    if !isa(operand, TensorField)
        throw(ArgumentError("Trace requires a TensorField"))
    end

    # Create result scalar field
    result = ScalarField(operand.dist, "trace_$(operand.name)", operand.bases, operand.dtype)

    # Ensure diagonal components are in correct layout
    dim = size(operand.components, 1)

    for i in 1:dim
        ensure_layout!(operand.components[i,i], layout)
    end

    ensure_layout!(result, layout)

    # Sum diagonal components
    data_field = layout == :g ? :data_g : :data_c
    fill!(getfield(result, data_field), 0.0)

    for i in 1:dim
        comp_data = getfield(operand.components[i,i], data_field)
        result_data = getfield(result, data_field)
        result_data .+= comp_data
    end

    return result
end

"""
    evaluate_skew(skew_op::Skew, layout::Symbol=:g)

Evaluate skew-symmetric part of a tensor field.
skew(T) = (T - T^T) / 2
"""
function evaluate_skew(skew_op::Skew, layout::Symbol=:g)
    operand = skew_op.operand

    if !isa(operand, TensorField)
        throw(ArgumentError("Skew requires a TensorField"))
    end

    coordsys = operand.coordsys
    result = TensorField(operand.dist, coordsys, "skew_$(operand.name)", operand.bases, operand.dtype)

    dim = size(operand.components, 1)
    data_field = layout == :g ? :data_g : :data_c

    for i in 1:dim
        for j in 1:dim
            ensure_layout!(operand.components[i,j], layout)
            ensure_layout!(operand.components[j,i], layout)
            ensure_layout!(result.components[i,j], layout)

            T_ij = getfield(operand.components[i,j], data_field)
            T_ji = getfield(operand.components[j,i], data_field)
            result_data = getfield(result.components[i,j], data_field)

            result_data .= 0.5 .* (T_ij .- T_ji)
        end
    end

    return result
end

"""
    evaluate_transpose_components(trans_op::TransposeComponents, layout::Symbol=:g)

Evaluate transpose of tensor field components.
"""
function evaluate_transpose_components(trans_op::TransposeComponents, layout::Symbol=:g)
    operand = trans_op.operand

    if !isa(operand, TensorField)
        throw(ArgumentError("TransposeComponents requires a TensorField"))
    end

    coordsys = operand.coordsys
    result = TensorField(operand.dist, coordsys, "trans_$(operand.name)", operand.bases, operand.dtype)

    dim = size(operand.components, 1)
    data_field = layout == :g ? :data_g : :data_c

    for i in 1:dim
        for j in 1:dim
            ensure_layout!(operand.components[j,i], layout)
            ensure_layout!(result.components[i,j], layout)

            src_data = getfield(operand.components[j,i], data_field)
            dst_data = getfield(result.components[i,j], data_field)

            copyto!(dst_data, src_data)
        end
    end

    return result
end

# ============================================================================
# Curl evaluation for 2D and 3D
# ============================================================================

"""
    evaluate_curl(curl_op::Curl, layout::Symbol=:g)

Evaluate curl of a vector field.
2D: curl(v) = ∂v_y/∂x - ∂v_x/∂y (scalar)
3D: curl(v) = (∂v_z/∂y - ∂v_y/∂z, ∂v_x/∂z - ∂v_z/∂x, ∂v_y/∂x - ∂v_x/∂y)
"""
function evaluate_curl(curl_op::Curl, layout::Symbol=:g)
    operand = curl_op.operand
    coordsys = curl_op.coordsys

    if !isa(operand, VectorField)
        throw(ArgumentError("Curl requires a VectorField"))
    end

    dim = length(operand.components)

    if dim == 2
        # 2D curl: returns scalar
        return evaluate_curl_2d(operand, coordsys, layout)
    elseif dim == 3
        # 3D curl: returns vector
        return evaluate_curl_3d(operand, coordsys, layout)
    else
        throw(ArgumentError("Curl only implemented for 2D and 3D"))
    end
end

function evaluate_curl_2d(operand::VectorField, coordsys::CoordinateSystem, layout::Symbol)
    # curl(v) = ∂v_y/∂x - ∂v_x/∂y
    vx = operand.components[1]
    vy = operand.components[2]

    coord_x = coordsys.coords[1]
    coord_y = coordsys.coords[2]

    # ∂v_y/∂x
    dvy_dx = evaluate_differentiate(Differentiate(vy, coord_x, 1), layout)

    # ∂v_x/∂y
    dvx_dy = evaluate_differentiate(Differentiate(vx, coord_y, 1), layout)

    # Result = dvy_dx - dvx_dy
    result = ScalarField(operand.dist, "curl_$(operand.name)", operand.bases, operand.dtype)
    ensure_layout!(result, layout)

    data_field = layout == :g ? :data_g : :data_c
    result_data = getfield(result, data_field)
    dvy_data = getfield(dvy_dx, data_field)
    dvx_data = getfield(dvx_dy, data_field)

    result_data .= dvy_data .- dvx_data

    return result
end

function evaluate_curl_3d(operand::VectorField, coordsys::CoordinateSystem, layout::Symbol)
    vx = operand.components[1]
    vy = operand.components[2]
    vz = operand.components[3]

    coord_x = coordsys.coords[1]
    coord_y = coordsys.coords[2]
    coord_z = coordsys.coords[3]

    # Component 1: ∂v_z/∂y - ∂v_y/∂z
    dvz_dy = evaluate_differentiate(Differentiate(vz, coord_y, 1), layout)
    dvy_dz = evaluate_differentiate(Differentiate(vy, coord_z, 1), layout)

    # Component 2: ∂v_x/∂z - ∂v_z/∂x
    dvx_dz = evaluate_differentiate(Differentiate(vx, coord_z, 1), layout)
    dvz_dx = evaluate_differentiate(Differentiate(vz, coord_x, 1), layout)

    # Component 3: ∂v_y/∂x - ∂v_x/∂y
    dvy_dx = evaluate_differentiate(Differentiate(vy, coord_x, 1), layout)
    dvx_dy = evaluate_differentiate(Differentiate(vx, coord_y, 1), layout)

    result = VectorField(operand.dist, coordsys, "curl_$(operand.name)", operand.bases, operand.dtype)

    data_field = layout == :g ? :data_g : :data_c

    for comp in result.components
        ensure_layout!(comp, layout)
    end

    # Set component data
    getfield(result.components[1], data_field) .= getfield(dvz_dy, data_field) .- getfield(dvy_dz, data_field)
    getfield(result.components[2], data_field) .= getfield(dvx_dz, data_field) .- getfield(dvz_dx, data_field)
    getfield(result.components[3], data_field) .= getfield(dvy_dx, data_field) .- getfield(dvx_dy, data_field)

    return result
end

# ============================================================================
# Laplacian evaluation
# ============================================================================

"""
    evaluate_laplacian(lap_op::Laplacian, layout::Symbol=:g)

Evaluate Laplacian operator.
∇²f = Σ_i ∂²f/∂x_i²
"""
function evaluate_laplacian(lap_op::Laplacian, layout::Symbol=:g)
    operand = lap_op.operand

    if isa(operand, ScalarField)
        return evaluate_scalar_laplacian(operand, layout)
    elseif isa(operand, VectorField)
        return evaluate_vector_laplacian(operand, layout)
    else
        throw(ArgumentError("Laplacian not implemented for $(typeof(operand))"))
    end
end

function evaluate_scalar_laplacian(operand::ScalarField, layout::Symbol)
    result = ScalarField(operand.dist, "lap_$(operand.name)", operand.bases, operand.dtype)
    ensure_layout!(result, layout)

    data_field = layout == :g ? :data_g : :data_c
    fill!(getfield(result, data_field), 0.0)

    for (i, basis) in enumerate(operand.bases)
        # Find coordinate for this basis
        coord = Coordinate(basis.meta.coordsys, basis.meta.element_label)

        # Second derivative
        d2f = evaluate_differentiate(Differentiate(operand, coord, 2), layout)

        result_data = getfield(result, data_field)
        d2f_data = getfield(d2f, data_field)
        result_data .+= d2f_data
    end

    return result
end

function evaluate_vector_laplacian(operand::VectorField, layout::Symbol)
    result = VectorField(operand.dist, operand.coordsys, "lap_$(operand.name)",
                        operand.bases, operand.dtype)

    for (i, comp) in enumerate(operand.components)
        lap_comp = evaluate_scalar_laplacian(comp, layout)

        data_field = layout == :g ? :data_g : :data_c
        ensure_layout!(result.components[i], layout)
        copyto!(getfield(result.components[i], data_field), getfield(lap_comp, data_field))
    end

    return result
end

# ============================================================================
# Outer Product (Tensor Product) evaluation
# ============================================================================

"""
    evaluate_outer(outer_op::Outer, layout::Symbol=:g)

Evaluate outer product (tensor product) of two vector fields.

For vectors u and v, returns tensor T where T_ij = u_i * v_j.

Following pattern for tensor construction from vector products.

Arguments:
- outer_op: Outer operator containing left and right operands
- layout: :g for grid space, :c for coefficient space

Returns:
- TensorField with components T_ij = u_i * v_j
"""
function evaluate_outer(outer_op::Outer, layout::Symbol=:g)
    left = outer_op.left
    right = outer_op.right

    # Validate operands are vector fields
    if !isa(left, VectorField) || !isa(right, VectorField)
        throw(ArgumentError("Outer product requires two VectorFields"))
    end

    # Get dimensions
    dim_left = length(left.components)
    dim_right = length(right.components)

    # Create result tensor field
    dist = left.dist
    coordsys = left.coordsys
    bases = left.bases
    dtype = promote_type(left.dtype, right.dtype)

    result = TensorField(dist, coordsys, "outer_$(left.name)_$(right.name)", bases, dtype)

    # Ensure proper dimensions
    if size(result.components) != (dim_left, dim_right)
        # Reinitialize with correct dimensions
        result.components = Matrix{ScalarField}(undef, dim_left, dim_right)
        for i in 1:dim_left
            for j in 1:dim_right
                result.components[i,j] = ScalarField(dist, "T_$(i)$(j)", bases, dtype)
            end
        end
    end

    data_field = layout == :g ? :data_g : :data_c

    # Compute T_ij = u_i * v_j for each component
    for i in 1:dim_left
        for j in 1:dim_right
            # Ensure operands are in correct layout
            ensure_layout!(left.components[i], layout)
            ensure_layout!(right.components[j], layout)
            ensure_layout!(result.components[i,j], layout)

            # Get data arrays
            u_data = getfield(left.components[i], data_field)
            v_data = getfield(right.components[j], data_field)
            result_data = getfield(result.components[i,j], data_field)

            # Compute outer product component: T_ij = u_i * v_j
            result_data .= u_data .* v_data
        end
    end

    return result
end

# ============================================================================
# AdvectiveCFL evaluation
# ============================================================================

"""
    evaluate_advective_cfl(cfl_op::AdvectiveCFL, layout::Symbol=:g)

Evaluate advective CFL grid-crossing frequency field.

Computes the local grid-crossing frequency:
    f = |u|/Δx + |v|/Δy + |w|/Δz

where u, v, w are velocity components and Δx, Δy, Δz are local grid spacings.

This field can be used for adaptive timestepping: Δt < 1/max(f).

Following operators:4342-4411 AdvectiveCFL pattern.

Arguments:
- cfl_op: AdvectiveCFL operator containing velocity field and coordinate system
- layout: Must be :g (grid space) for CFL computation

Returns:
- ScalarField with CFL frequency at each grid point
"""
function evaluate_advective_cfl(cfl_op::AdvectiveCFL, layout::Symbol=:g)
    velocity = cfl_op.operand
    coords = cfl_op.coords

    # CFL must be computed in grid space
    if layout != :g
        @warn "AdvectiveCFL requires grid space; converting to :g"
        layout = :g
    end

    # Validate operand is a vector field
    if !isa(velocity, VectorField)
        throw(ArgumentError("AdvectiveCFL requires a VectorField (velocity)"))
    end

    dim = length(velocity.components)

    # Create result scalar field
    dist = velocity.dist
    bases = velocity.bases
    dtype = velocity.dtype

    result = ScalarField(dist, "cfl_freq", bases, dtype)
    ensure_layout!(result, :g)

    # Initialize result to zero
    result.data_g .= 0

    # Compute CFL frequency for each coordinate direction
    for i in 1:dim
        # Ensure velocity component is in grid space
        ensure_layout!(velocity.components[i], :g)
        vel_data = velocity.components[i].data_g

        # Get grid spacing for this coordinate
        # Grid spacing depends on the basis type
        basis = bases[i]
        grid_spacing = compute_grid_spacing(basis, dist, i)

        # Add contribution: |u_i| / Δx_i
        if isa(grid_spacing, AbstractArray)
            # Variable grid spacing (e.g., Chebyshev)
            result.data_g .+= abs.(vel_data) ./ grid_spacing
        else
            # Uniform grid spacing (e.g., Fourier)
            result.data_g .+= abs.(vel_data) ./ grid_spacing
        end
    end

    return result
end

"""
    compute_grid_spacing(basis::Basis, dist, axis::Int)

Compute local grid spacing for a basis.

For Fourier bases: uniform spacing Δx = L/N
For Chebyshev bases: variable spacing based on Chebyshev nodes

Following basis CartesianAdvectiveCFL.grid_spacing pattern.
"""
function compute_grid_spacing(basis::Basis, dist, axis::Int)
    if basis === nothing
        return 1.0  # Fallback for undefined basis
    end

    N = basis.N

    if isa(basis, FourierBasis)
        # Uniform spacing for Fourier
        L = basis.bounds[2] - basis.bounds[1]
        return L / N
    elseif isa(basis, ChebyshevT) || isa(basis, ChebyshevU) || isa(basis, Chebyshev)
        # Chebyshev grid spacing (clustered at boundaries)
        # For Chebyshev-Gauss-Lobatto points: x_j = cos(π*j/N)
        # Local spacing: Δx_j ≈ π * sqrt(1 - x_j²) / N
        L = basis.bounds[2] - basis.bounds[1]

        # Compute grid points
        j = 0:(N-1)
        x_j = cos.(π .* j ./ (N - 1))

        # Local spacing (avoiding singularity at boundaries)
        spacing = (π / N) .* sqrt.(max.(1 .- x_j.^2, 1e-10))

        # Scale to physical domain
        return spacing .* (L / 2)
    elseif isa(basis, Legendre)
        # Legendre-Gauss-Lobatto spacing (similar to Chebyshev)
        L = basis.bounds[2] - basis.bounds[1]
        return L / N  # Approximate with uniform for simplicity
    elseif isa(basis, Jacobi)
        # General Jacobi basis
        L = basis.bounds[2] - basis.bounds[1]
        return L / N  # Approximate with uniform
    else
        # Default uniform spacing
        return 1.0
    end
end

# ============================================================================
# Unified operator evaluation dispatcher
# ============================================================================

"""
    evaluate(op::Operator, layout::Symbol=:g)

Unified evaluation function that dispatches to specific operator evaluators.
"""
function evaluate(op::Operator, layout::Symbol=:g)
    if isa(op, Gradient)
        return evaluate_gradient(op, layout)
    elseif isa(op, Divergence)
        return evaluate_divergence(op, layout)
    elseif isa(op, Curl)
        return evaluate_curl(op, layout)
    elseif isa(op, Laplacian)
        return evaluate_laplacian(op, layout)
    elseif isa(op, Differentiate)
        return evaluate_differentiate(op, layout)
    elseif isa(op, Interpolate)
        return evaluate_interpolate(op, layout)
    elseif isa(op, Integrate)
        return evaluate_integrate(op, layout)
    elseif isa(op, Average)
        return evaluate_average(op, layout)
    elseif isa(op, Lift)
        return evaluate_lift(op, layout)
    elseif isa(op, Convert)
        return evaluate_convert(op, layout)
    elseif isa(op, Power)
        return evaluate_power(op, layout)
    elseif isa(op, GeneralFunction)
        return evaluate_general_function(op, layout)
    elseif isa(op, UnaryGridFunction)
        return evaluate_unary_grid_function(op, layout)
    elseif isa(op, Grid)
        return evaluate_grid(op)
    elseif isa(op, Coeff)
        return evaluate_coeff(op)
    elseif isa(op, Component)
        return evaluate_component(op)
    elseif isa(op, RadialComponent)
        return evaluate_radial_component(op)
    elseif isa(op, AngularComponent)
        return evaluate_angular_component(op)
    elseif isa(op, AzimuthalComponent)
        return evaluate_azimuthal_component(op)
    elseif isa(op, Trace)
        return evaluate_trace(op, layout)
    elseif isa(op, Skew)
        return evaluate_skew(op, layout)
    elseif isa(op, TransposeComponents)
        return evaluate_transpose_components(op, layout)
    elseif isa(op, Outer)
        return evaluate_outer(op, layout)
    elseif isa(op, AdvectiveCFL)
        return evaluate_advective_cfl(op, layout)
    elseif isa(op, TimeDerivative)
        # TimeDerivative is handled by solvers, not direct evaluation
        throw(ArgumentError("TimeDerivative cannot be directly evaluated; use solver"))
    else
        throw(ArgumentError("Evaluation not implemented for operator type $(typeof(op))"))
    end
end

