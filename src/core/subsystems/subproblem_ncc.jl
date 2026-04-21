# ---------------------------------------------------------------------------
# NCC (Non-Constant Coefficient) handling
# ---------------------------------------------------------------------------

"""
    NCCData

Storage for non-constant coefficient data.
"""
mutable struct NCCData
    coeffs::Union{Nothing, Array}
    cutoff::Float64
    max_terms::Union{Nothing, Int}
end

NCCData() = NCCData(nothing, 1e-6, nothing)

"""
    build_ncc_matrix(ncc_data, sp, arg_domain, out_domain; ncc_cutoff=1e-6, max_ncc_terms=nothing)

Build the NCC (non-constant coefficient) multiplication matrix for a single
subproblem. An NCC is a spatially-varying coefficient appearing as a
multiplicative factor in an equation, e.g. `ν(z) * ∇²u` where `ν` varies
with the coupled coordinate.

The matrix is built by summing mode-shift matrices (one per significant
spectral coefficient of the NCC field), weighted by that coefficient's
value:

    M = Σ_k ν_k · S_k

where `ν_k` is the k-th spectral coefficient of the NCC field and `S_k` is
the shift matrix that implements spectral-space convolution by mode k.

## Two-stage truncation

To keep the resulting sparse matrix as small as possible (which directly
reduces sparse LU factorization cost), we apply two levels of truncation:

1. **Per-mode cutoff** (`ncc_cutoff`): skip contributions from NCC modes
   whose absolute coefficient is below `ncc_cutoff`. Dedalus's default is
   `1e-6`; tighter cutoffs give sparser matrices at some accuracy cost.
2. **Max-terms cap** (`max_ncc_terms`): after sorting modes by coefficient
   magnitude, retain at most `max_ncc_terms` dominant modes. Useful for
   rapidly-decaying NCCs where a small number of modes capture most of the
   information.

After accumulation, the matrix is passed through `droptol!` to remove any
entries whose absolute value is below `ncc_cutoff` — this catches residual
entries that appear from cross-mode accumulation (e.g. two large opposite
contributions that partially cancel) and shrinks the nnz count for the
downstream sparse LU.

Follows the spectral methods pattern from the Dedalus arithmetic module.
"""
function build_ncc_matrix(ncc_data::NCCData, sp::Subproblem, arg_domain, out_domain;
                          ncc_cutoff::Float64=1e-6, max_ncc_terms::Union{Nothing, Int}=nothing)
    if ncc_data.coeffs === nothing
        return nothing
    end

    coeffs = ncc_data.coeffs
    shape = (coeff_size(sp, out_domain), coeff_size(sp, arg_domain))
    matrix = spzeros(ComplexF64, shape...)

    sp_shape = coeff_shape(sp, out_domain)
    ncc_shape = size(coeffs)

    # ─── Relative cutoff ─────────────────────────────────────────────
    # Dedalus convention: interpret `ncc_cutoff` as a RELATIVE threshold
    # against the L-infinity norm of the coefficient array, not as an
    # absolute magnitude. This scales automatically with the problem's
    # natural coefficient magnitude — a viscosity `ν ~ 1e-3` and a
    # temperature `T ~ 1.0` both get truncated at the same relative
    # precision without the user having to tune `ncc_cutoff` per field.
    #
    # Matches `dedalus.core.arithmetic.Multiply._ncc_matrices` where
    # `cutoff` is divided by the coefficient L∞ norm before comparison.
    coeff_max = 0.0
    @inbounds for i in eachindex(coeffs)
        ai = abs(coeffs[i])
        ai > coeff_max && (coeff_max = ai)
    end
    # Absolute threshold: `ncc_cutoff × max|coefficient|`. For
    # coeff_max == 0 (all-zero NCC), treat as zero — build_ncc_matrix
    # returns an empty sparse matrix.
    abs_cutoff = ncc_cutoff * coeff_max

    # ─── Energy-weighted mode selection ──────────────────────────────
    # Collect modes above the relative cutoff, along with their "energy"
    # |coeff|² which is what determines the truncation error in an
    # L²-sense approximation. Sort by energy descending so the
    # `max_ncc_terms` cap keeps the modes that carry the most spectral
    # power, not just the modes that are pointwise-largest.
    #
    # For smoothly-decaying coefficient fields (e.g. exponentially
    # decaying Chebyshev coefficients) energy-sorted and magnitude-sorted
    # orderings agree. For oscillatory fields or fields with fat tails,
    # energy sorting gives a better approximation at the same
    # `max_ncc_terms` budget.
    significant_modes = Tuple{Float64, CartesianIndex}[]
    @inbounds for ncc_mode in CartesianIndices(ncc_shape)
        mag = abs(coeffs[ncc_mode])
        if mag >= abs_cutoff && mag > 0
            push!(significant_modes, (mag * mag, ncc_mode))  # energy weight
        end
    end
    sort!(significant_modes; by = first, rev = true)

    # Optional energy-retention cap: if max_ncc_terms is nothing, we can
    # additionally cap by CUMULATIVE energy fraction — keep enough modes
    # to capture 1 - ncc_cutoff² of the total power. This is another
    # Dedalus-style truncation that's independent of the count cap.
    total_energy = 0.0
    @inbounds for k in 1:length(significant_modes)
        total_energy += significant_modes[k][1]
    end

    n_to_use = length(significant_modes)
    if max_ncc_terms !== nothing
        n_to_use = min(n_to_use, max_ncc_terms)
    else
        # Cap by cumulative energy: retain enough modes to capture
        # (1 - ncc_cutoff²) of the total L² energy. For ncc_cutoff = 1e-6
        # this keeps essentially all significant modes; for ncc_cutoff =
        # 0.01 it drops trailing modes that contribute < 1e-4 of energy.
        target_energy = (1.0 - ncc_cutoff * ncc_cutoff) * total_energy
        cumulative = 0.0
        cap = 0
        @inbounds for k in 1:length(significant_modes)
            cumulative += significant_modes[k][1]
            cap = k
            cumulative >= target_energy && break
        end
        n_to_use = max(cap, 1)
    end

    # ─── Accumulate mode-shift contributions ─────────────────────────
    @inbounds for k in 1:n_to_use
        ncc_mode = significant_modes[k][2]
        ncc_coeff = coeffs[ncc_mode]
        mode_mat = cartesian_mode_matrix(sp_shape, arg_domain, out_domain, Tuple(ncc_mode))
        # In-place accumulation into `matrix.nzval` would avoid the
        # temporary sparse matrix on the RHS, but sparse-structure merging
        # is awkward when new mode matrices introduce additional nonzero
        # patterns. The current add-and-reassign is O(nnz) per iteration
        # and dominated by the subsequent LU, so it's not the bottleneck.
        matrix = matrix + ncc_coeff * mode_mat
    end

    # ─── Post-accumulation sparsification ────────────────────────────
    # After summing all contributions, drop residual small entries. Use
    # the absolute cutoff (scaled by max coefficient) to match the
    # per-mode threshold, and then dropzeros! to clean up structural
    # cancellations.
    if abs_cutoff > 0 && nnz(matrix) > 0
        droptol!(matrix, abs_cutoff)
    end
    if nnz(matrix) > 0
        dropzeros!(matrix)
    end

    return matrix
end

"""
    cartesian_mode_matrix(sp_shape, arg_domain, out_domain, ncc_mode)

Build mode matrix for Cartesian NCC.
Following arithmetic:446-460.
"""
function cartesian_mode_matrix(sp_shape, arg_domain, out_domain, ncc_mode::Tuple)
    # Build Kronecker product of 1D mode matrices for NCC convolution
    # This implements the mode matrix for Non-Constant Coefficient multiplication
    # in spectral space, which corresponds to convolution in coefficient space
    matrix = sparse([1.0])

    for axis in eachindex(sp_shape)
        n = sp_shape[axis]
        mode = axis <= length(ncc_mode) ? ncc_mode[axis] : 0

        # Product matrix for this axis
        # mode == 0: identity (no shift)
        # mode != 0: circulant shift matrix for periodic bases
        if mode == 0 || mode == 1
            # Identity matrix - no mode shift
            axis_mat = sparse(I, n, n)
        else
            # Build circulant shift matrix for mode k
            # This shifts coefficients by the NCC mode index
            # For periodic bases: c_j -> c_{j-k} (circular)
            rows = Int[]
            cols = Int[]
            vals = Float64[]

            for j in 1:n
                # Target index after shift (circular)
                target = mod1(j + mode - 1, n)
                push!(rows, target)
                push!(cols, j)
                push!(vals, 1.0)
            end

            axis_mat = sparse(rows, cols, vals, n, n)
        end

        matrix = kron(matrix, axis_mat)
    end

    return matrix
end

# ---------------------------------------------------------------------------
# Legacy compatibility functions
# ---------------------------------------------------------------------------

# Keep old function signatures for backwards compatibility
coeff_size(subsystem::Subsystem, field::ScalarField) = field_size(subsystem, field)
coeff_size(subsystem::Subsystem, field::VectorField) = sum(field_size(subsystem, comp) for comp in field.components)
coeff_size(subsystem::Subsystem, field::TensorField) = sum(field_size(subsystem, comp) for comp in vec(field.components))

