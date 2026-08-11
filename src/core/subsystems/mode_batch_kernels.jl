# ── Batched Fourier-mode kernels ─────────────────────────────────────────────
#
# Each kernel replaces one per-mode operation in the coupled stage loop with a
# single launch over all modes. Column `m` of every `(n, nmodes)` argument is
# mode `batch.sp_indices[m]`.
#
# WRITE-ONCE CONTRACT: every kernel below writes each output element exactly
# once and never reads it back. The KernelAbstractions CPU backend wraps kernel
# bodies in an ivdep/no-alias workitem loop, which licenses reordering of
# same-slot read-modify-writes around inner loops — that miscompiled the
# Chebyshev recurrence kernel in `ext/cuda/cheb_deriv.jl`. Accumulation is done
# into a register and stored once, never into the output slot in a loop.

using KernelAbstractions

@kernel function _batched_gather_kernel!(X, @Const(cd), @Const(starts),
                                         step_, row_offset)
    i, m = @index(Global, NTuple)
    @inbounds X[row_offset + i, m] = cd[starts[m] + (i - 1) * step_]
end

"""
    batched_gather!(X, cd, starts, step_, len, row_offset) -> X

Gather one strided run per mode out of the coefficient array `cd` into rows
`row_offset+1 : row_offset+len` of `X`. `starts[m]` is mode `m`'s linear start
offset; `step_` and `len` are shared, since every mode selects the same coupled
axis. This is `_gather_strided!` for all modes at once.
"""
function batched_gather!(X::AbstractMatrix{ComplexF64}, cd::AbstractArray,
                         starts::AbstractVector{Int}, step_::Int, len::Int,
                         row_offset::Int)
    backend = get_backend(X)
    _batched_gather_kernel!(backend)(X, cd, starts, step_, row_offset;
                                     ndrange=(len, size(X, 2)))
    KernelAbstractions.synchronize(backend)
    return X
end

@kernel function _batched_scatter_kernel!(cd, @Const(X), @Const(starts),
                                          step_, row_offset)
    i, m = @index(Global, NTuple)
    @inbounds cd[starts[m] + (i - 1) * step_] = X[row_offset + i, m]
end

"""
    batched_scatter!(cd, X, starts, step_, len, row_offset) -> cd

The mirror of `batched_gather!`. Writes rows `row_offset+1 : row_offset+len` of
`X` back into each mode's strided run of `cd`.
"""
function batched_scatter!(cd::AbstractArray, X::AbstractMatrix{ComplexF64},
                          starts::AbstractVector{Int}, step_::Int, len::Int,
                          row_offset::Int)
    backend = get_backend(X)
    _batched_scatter_kernel!(backend)(cd, X, starts, step_, row_offset;
                                      ndrange=(len, size(X, 2)))
    KernelAbstractions.synchronize(backend)
    return cd
end

# One thread per (row, mode). Each thread accumulates that row's dot product in
# a REGISTER and stores once — a CSC-column loop writing into Y[row, m]
# repeatedly is exactly the same-slot RMW shape that miscompiled before.
# Iterating rows requires the CSR view of the pattern, so callers pass the
# TRANSPOSED CSC pattern (equivalently, the CSR pattern of the original).
@kernel function _batched_spmv_kernel!(Y, @Const(rowptr), @Const(colval),
                                       @Const(nzval), @Const(X))
    r, m = @index(Global, NTuple)
    acc = zero(ComplexF64)
    @inbounds for k in rowptr[r]:(rowptr[r + 1] - 1)
        acc += nzval[k, m] * X[colval[k], m]
    end
    @inbounds Y[r, m] = acc
end

"""
    batched_spmv!(Y, rowptr, colval, nzval, X) -> Y

`Y[:, m] = A_m * X[:, m]` for every mode, where all `A_m` share the CSR pattern
`(rowptr, colval)` and `nzval[:, m]` holds mode `m`'s values in that order.
"""
function batched_spmv!(Y::AbstractMatrix{ComplexF64},
                       rowptr::AbstractVector{Int}, colval::AbstractVector{Int},
                       nzval::AbstractMatrix{ComplexF64},
                       X::AbstractMatrix{ComplexF64})
    backend = get_backend(Y)
    _batched_spmv_kernel!(backend)(Y, rowptr, colval, nzval, X;
                                   ndrange=size(Y))
    KernelAbstractions.synchronize(backend)
    return Y
end

@kernel function _batched_bc_override_kernel!(RHS, @Const(ALG_F),
                                              @Const(bc_rows), coeff)
    b, m = @index(Global, NTuple)
    @inbounds r = bc_rows[b]
    @inbounds RHS[r, m] = coeff * ALG_F[r, m]
end

"""
    batched_bc_override!(RHS, ALG_F, bc_rows, coeff) -> RHS

Overwrite the algebraic/BC rows of every mode's stage RHS with
`coeff * ALG_F`, enforcing `L_row * X = F_alg` at each stage. `bc_rows` is
shared across the batch — the bucket signature guarantees it.
"""
function batched_bc_override!(RHS::AbstractMatrix{ComplexF64},
                              ALG_F::AbstractMatrix{ComplexF64},
                              bc_rows::AbstractVector{Int}, coeff::Number)
    isempty(bc_rows) && return RHS
    backend = get_backend(RHS)
    _batched_bc_override_kernel!(backend)(RHS, ALG_F, bc_rows,
                                          ComplexF64(coeff);
                                          ndrange=(length(bc_rows), size(RHS, 2)))
    KernelAbstractions.synchronize(backend)
    return RHS
end

# Two passes, each write-once: zero the dense workspace, then place the stored
# values. A single pass cannot do both without reading back what it wrote.
# Zeroing is mandatory — the workspace is reused across dt changes, and touching
# only the stored nonzeros would leave the previous factorization's values
# sitting in the structural-zero slots.
@kernel function _batched_lhs_zero_kernel!(dense)
    i, j, m = @index(Global, NTuple)
    @inbounds dense[i, j, m] = zero(ComplexF64)
end

@kernel function _batched_lhs_place_kernel!(dense, @Const(colptr), @Const(rowval),
                                            @Const(M_nzval), @Const(L_nzval),
                                            coeff, ncols)
    k, m = @index(Global, NTuple)
    # Locate the column owning stored index k by binary search over colptr.
    lo, hi = 1, ncols
    @inbounds while lo < hi
        mid = (lo + hi + 1) >> 1
        if colptr[mid] <= k
            lo = mid
        else
            hi = mid - 1
        end
    end
    @inbounds dense[rowval[k], lo, m] = M_nzval[k, m] + coeff * L_nzval[k, m]
end

"""
    batched_assemble_lhs!(dense, colptr, rowval, M_nzval, L_nzval, coeff) -> dense

Build every mode's dense stage LHS as `M_exp + coeff * L_exp`, on-device, from
values that live on the device permanently. This is what removes the per-mode
host `LHS.nzval` rebuild and its upload under adaptive dt.

Not bit-exact against the host expression: the backend may contract
`M + coeff*L` into an FMA.
"""
function batched_assemble_lhs!(dense::AbstractArray{ComplexF64, 3},
                               colptr::AbstractVector{Int},
                               rowval::AbstractVector{Int},
                               M_nzval::AbstractMatrix{ComplexF64},
                               L_nzval::AbstractMatrix{ComplexF64},
                               coeff::Number)
    backend = get_backend(dense)
    n, _, nmodes = size(dense)
    _batched_lhs_zero_kernel!(backend)(dense; ndrange=(n, n, nmodes))
    KernelAbstractions.synchronize(backend)
    _batched_lhs_place_kernel!(backend)(dense, colptr, rowval, M_nzval, L_nzval,
                                        ComplexF64(coeff), n;
                                        ndrange=(size(M_nzval, 1), nmodes))
    KernelAbstractions.synchronize(backend)
    return dense
end

# One thread per (column, mode). Each output element is written exactly once
# and never read back — including the null columns, which must be WRITTEN to
# zero rather than skipped, because the destination buffer is reused across
# stages and steps.
@kernel function _batched_mass_apply_kernel!(X, @Const(B), @Const(src),
                                             @Const(scale))
    j, m = @index(Global, NTuple)
    @inbounds begin
        s = src[j]
        X[j, m] = s == 0 ? zero(ComplexF64) : B[s, m] / scale[j]
    end
end

"""
    batched_mass_apply!(X, B, src, scale) -> X

Apply the pseudo-inverse of a scaled partial-permutation mass matrix to every
mode at once: `X[j, m] = B[src[j], m] / scale[j]`, and zero where `src[j] == 0`.

This is the minimum-norm least-squares solution of `M x = b` when `M` is a
scaled partial permutation — see `mass_selection_plan`, which verifies that
structure and produces `src`/`scale`. Callers must not reach here without a
plan from it.
"""
function batched_mass_apply!(X::AbstractMatrix{ComplexF64},
                             B::AbstractMatrix{ComplexF64},
                             src::AbstractVector{Int},
                             scale::AbstractVector{ComplexF64})
    backend = get_backend(X)
    _batched_mass_apply_kernel!(backend)(X, B, src, scale; ndrange=size(X))
    KernelAbstractions.synchronize(backend)
    return X
end
