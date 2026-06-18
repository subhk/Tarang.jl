# Integration and averaging evaluation along spectral coordinates.

# ============================================================================
# Integration Evaluation
# ============================================================================

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

    # Accept composite expressions (e.g. integrate(0.5*(u⋅u), coords)): reduce the
    # operand tree to a scalar field first, then integrate (Dedalus-style).
    if !isa(operand, ScalarField)
        operand = evaluate(operand)
    end
    if !isa(operand, ScalarField)
        throw(ArgumentError("integrate: operand must reduce to a scalar field, got $(typeof(operand))"))
    end

    # Handle single coordinate or tuple of coordinates
    coords = isa(coord, Coordinate) ? (coord,) : coord

    # Start with the operand
    result_field = copy(operand)

    # MPI-distributed grid data needs its own paths: the serial per-axis loop
    # below applies GLOBAL-length quadrature weights to LOCAL slabs, and
    # PencilArrays' generic mapreduce is unavailable on non-Intel MPI builds.
    # Full integration reduces to a scalar Allreduce; partial integration
    # gathers the (analysis-sized) field and reuses the serial reduction.
    ensure_layout!(result_field, :g)
    gdata = get_grid_data(result_field)
    if isa(gdata, PencilArrays.PencilArray) && length(coords) == length(operand.bases)
        return _integrate_full_distributed(result_field, gdata)
    end

    # Integrate over each coordinate in sequence
    for c in coords
        result_field = integrate_along_coord(result_field, c)
    end

    # If all dimensions integrated, return scalar
    if length(coords) == length(operand.bases)
        # integrate_along_coord may already return a scalar for 1D fields
        if result_field isa Number
            return result_field
        end
        ensure_layout!(result_field, :g)
        return sum(get_grid_data(result_field))
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

    # Apply weighted sum along the axis. Distributed fields are gathered to a
    # replicated global array first (analysis-sized; cold path), so the serial
    # reduction below applies and the reduced result is identical on all ranks
    # — mirroring the serial return semantics. The reduced field is built on a
    # COMM_SELF distributor: each rank holds the full result, and reduced
    # sub-domains (e.g. 1D) cannot be MPI-transform-planned anyway.
    data = get_grid_data(field)
    result_dist = field.dist
    if isa(data, PencilArrays.PencilArray)
        data = _allgather_global_grid(data, field.dist.comm)
        result_dist = _serial_replica_distributor(field.dist, field.dtype)
    end

    # Sum along the specified axis with weights
    result_data = integrate_weighted_sum(data, weights, basis_index)

    # Create result field with reduced dimensionality
    if ndims(data) == 1
        return result_data
    else
        # Create new field without the integrated dimension
        nb = length(field.bases)
        new_bases = ntuple(i -> field.bases[i < basis_index ? i : i + 1], nb - 1)
        if nb == 1
            return sum(result_data)
        end

        result = ScalarField(result_dist, "int_$(field.name)", new_bases, field.dtype)
        set_grid_data!(result, result_data)
        result.current_layout = :g
        return result
    end
end

# Serial (COMM_SELF) distributors for rank-replicated reduction results,
# cached so repeated integrate/average calls don't re-plan transforms.
const _SERIAL_REPLICA_DISTS = Dict{Tuple{UInt, DataType}, Distributor}()

function _serial_replica_distributor(dist::Distributor, dtype::DataType)
    key = (objectid(dist.coordsys), dtype)
    return get!(_SERIAL_REPLICA_DISTS, key) do
        Distributor(dist.coordsys; comm=MPI.COMM_SELF, dtype=dtype)
    end
end

"""
    _allgather_global_grid(pdata, comm)

Replicate a distributed grid array on every rank: each rank writes its local
slab into the right global positions (via `global_view`, permutation-aware) of
a zero array, then a single `Allreduce(+)` fills in everyone's pieces. Built-in
reduction op, so safe on all architectures (PencilArrays' `gather` and generic
mapreduce are not).
"""
function _allgather_global_grid(pdata::PencilArrays.PencilArray, comm)
    g = zeros(eltype(pdata), PencilArrays.size_global(pdata))
    gv = PencilArrays.global_view(pdata)
    for I in CartesianIndices(gv)
        g[I] = gv[I]
    end
    MPI.Allreduce!(g, +, comm)
    return g
end

"""
    _integrate_full_distributed(field, pdata)

Integral over ALL coordinates of an MPI-distributed field: weight each rank's
local slab with its slice of the global quadrature weights, sum locally, then
`MPI.Allreduce` the scalar (built-in `+` op — safe on all architectures, unlike
PencilArrays' generic mapreduce).

Pencil grid arrays can be axis-permuted: `axes_local` is indexed by LOGICAL
axis, while `parent(pdata)` is stored in PHYSICAL order, so each weight vector
is reshaped along `findfirst(==(logical), permutation)` (same convention as
`_apply_spectral_derivative_distributed!`).
"""
function _integrate_full_distributed(field::ScalarField, pdata::PencilArrays.PencilArray)
    local_data = parent(pdata)
    pencil = PencilArrays.pencil(pdata)
    local_axes = pencil.axes_local
    nd = ndims(local_data)
    # Tuple(NoPermutation()) is `nothing`, not the identity tuple
    perm_raw = Tuple(PencilArrays.permutation(pdata))
    perm_tuple = perm_raw === nothing ? ntuple(identity, nd) : perm_raw

    weighted = local_data
    for l in 1:length(field.bases)
        w_local = get_integration_weights(field.bases[l])[local_axes[l]]
        if is_gpu_array(local_data)
            w_local = copy_to_device(w_local, local_data)
        end
        p = findfirst(==(l), perm_tuple)
        p === nothing && (p = l)
        shape = ntuple(i -> i == p ? length(w_local) : 1, nd)
        weighted = weighted .* reshape(w_local, shape)
    end

    local_sum = sum(weighted)
    return MPI.Allreduce(local_sum, +, field.dist.comm)
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

    elseif isa(basis, ChebyshevU) || isa(basis, ChebyshevV) ||
           isa(basis, Ultraspherical) || isa(basis, Jacobi)
        # These Jacobi-family bases collocate on NON-uniform Gauss / Gauss-Jacobi
        # nodes (see _native_grid), so uniform L/N weights do NOT integrate them.
        # Derive plain-integral weights from the basis's own reference nodes —
        # exact for polynomials up to degree N-1, the span of the basis.
        return _nodal_integration_weights(_native_grid(basis, 1.0), L)

    else
        # Default: uniform weights (only valid for a uniform grid; kept as a
        # defensive fallback for any future basis type).
        return fill(L / N, N)
    end
end

"""
    _nodal_integration_weights(nodes_ref, L)

Quadrature weights for an arbitrary set of nodal points `nodes_ref` on the
reference interval [-1, 1], exact for polynomials up to degree N-1. Solves the
(well-conditioned) Legendre Vandermonde system Vᵀ w = m, where V[k,j] = P_{j-1}(x_k)
and m_j = ∫_{-1}^1 P_{j-1} dx = 2·δ_{j1}. Scaled by L/2 to map onto an interval of
length L. Works for any Jacobi-family basis regardless of which Gauss/Gauss-Jacobi
nodes it collocates on (the weights depend only on the nodes, not the basis).
"""
function _nodal_integration_weights(nodes_ref::AbstractVector{<:Real}, L::Real)
    N = length(nodes_ref)
    if N == 1
        return [float(L)]
    end
    # V[k,j] = P_{j-1}(x_k) via the Legendre three-term recurrence.
    V = Matrix{Float64}(undef, N, N)
    @inbounds for k in 1:N
        x = float(nodes_ref[k])
        V[k, 1] = 1.0
        V[k, 2] = x
        for j in 2:(N - 1)
            V[k, j + 1] = ((2j - 1) * x * V[k, j] - (j - 1) * V[k, j - 1]) / j
        end
    end
    m = zeros(Float64, N)
    m[1] = 2.0                    # ∫_{-1}^1 P_0 dx = 2; ∫ P_{j≥1} dx = 0 (orthogonality)
    w_ref = transpose(V) \ m      # weights on [-1, 1]
    return w_ref .* (L / 2)
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

    # Clenshaw-Curtis weights on [-1, 1] via DCT-I approach
    # Sum goes to floor((N-1)/2) — the max harmonic of the DCT-I on N points
    M = (N - 1) ÷ 2  # maximum harmonic index
    for j in 1:N
        theta_j = pi * (j - 1) / (N - 1)
        w = 0.0
        for k in 0:M
            # DCT-I boundary factors: half-weight at k=0 and k=M (when N-1 is even)
            if k == 0 || (k == M && iseven(N - 1))
                factor = 1.0
            else
                factor = 2.0
            end
            w += factor * cos(2 * k * theta_j) / (1 - 4 * k^2)
        end
        weights[j] = 2 * w / (N - 1)
    end

    # Endpoint halving for Gauss-Lobatto grid (endpoints count half)
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
        points[i] = -cos(pi * (i - 0.25) / (N + 0.5))
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

    # Move weights to same device as data if needed
    if is_gpu_array(data)
        weights_device = copy_to_device(weights, data)
    else
        weights_device = weights
    end

    if nd == 1
        # dot conjugates its first argument; weights are real so this equals
        # sum(data .* weights) without materializing the weighted array
        return dot(weights_device, data)
    end

    # Reshape weights to broadcast along the correct axis
    shape = ones(Int, nd)
    shape[axis] = length(weights_device)
    w_shaped = reshape(weights_device, shape...)

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

# ============================================================================
# Average Evaluation
# ============================================================================

"""
    evaluate_average(avg_op::Average, layout::Symbol=:g)

Evaluate averaging operator along a coordinate.
Average = Integrate / (interval length)
"""
function evaluate_average(avg_op::Average, layout::Symbol=:g)
    operand = avg_op.operand
    coord = avg_op.coord

    # Accept composite expressions: reduce the operand tree to a scalar field first.
    if !isa(operand, ScalarField)
        operand = evaluate(operand)
    end
    if !isa(operand, ScalarField)
        throw(ArgumentError("average: operand must reduce to a scalar field, got $(typeof(operand))"))
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
        # Scale only the active layout's data to avoid corrupting stale buffers
        if int_result.current_layout == :g
            data = get_grid_data(int_result)
            if data !== nothing
                data ./= L
            end
        else
            data = get_coeff_data(int_result)
            if data !== nothing
                data ./= L
            end
        end
        return int_result
    end
end
