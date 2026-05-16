"""
    Transform Legendre — forward and backward Legendre transform execution.

Uses Gauss-Legendre quadrature via a precomputed forward/backward matrix
(`transform.forward_matrix` / `transform.backward_matrix`, built by
`setup_legendre_transform!` in transform_planning.jl).

Only the generic (multi-dim, out-of-place) path is kept here. The in-place
multi-dim path hasn't been written for Legendre yet — the dispatch via
`_apply_forward!` / `_apply_backward!` falls through to this file's
out-of-place functions through the GPU-guard fallback in the Fourier
in-place variants. If you add a Legendre-heavy benchmark that becomes
allocation-bound, write `_apply_forward!(::LegendreTransform)` following
the pattern in `transform_fourier.jl`.
"""

function _legendre_forward(data::AbstractArray, transform::LegendreTransform)
    # Legendre transforms are matrix-backed and CPU-only today. GPU inputs are
    # staged through `_execute_on_cpu` and returned by the caller's transform
    # orchestration path.
    return _execute_on_cpu(data) do host_data
        axis = transform.axis
        if axis > ndims(host_data)
            return host_data
        end

        if transform.forward_matrix === nothing
            return host_data
        end

        mat = transform.forward_matrix
        coeff_size = size(mat, 1)
        out_shape = ntuple(i -> i == axis ? coeff_size : size(host_data, i), ndims(host_data))
        real_type = real(eltype(host_data))
        out_eltype = eltype(host_data) <: Complex ? eltype(host_data) : real_type
        out = zeros(out_eltype, out_shape)

        other_dims = Tuple(filter(i -> i != axis, 1:ndims(host_data)))

        # Reuse global work buffers for each one-dimensional slice so repeated
        # multi-dimensional transforms do not allocate scratch vectors.
        wm = get_global_workspace()
        temp_real = get_workspace!(wm, real_type, (coeff_size,))
        temp_imag = eltype(host_data) <: Complex ? get_workspace!(wm, real_type, (coeff_size,)) : nothing

        if isempty(other_dims)
            # One-dimensional case: apply the quadrature matrix directly.
            mul!(temp_real, mat, real(host_data))
            if temp_imag === nothing
                copyto!(out, temp_real)
            else
                mul!(temp_imag, mat, imag(host_data))
                out .= complex.(temp_real, temp_imag)
            end
        else
            # Multi-dimensional case: transform along the selected axis by
            # applying the same matrix to each orthogonal slice.
            for (slice_in, slice_out) in zip(eachslice(host_data, dims=other_dims), eachslice(out, dims=other_dims))
                mul!(temp_real, mat, real(slice_in))
                if temp_imag === nothing
                    slice_out .= temp_real
                else
                    mul!(temp_imag, mat, imag(slice_in))
                    slice_out .= complex.(temp_real, temp_imag)
                end
            end
        end

        release_workspace!(wm, temp_real)
        if temp_imag !== nothing
            release_workspace!(wm, temp_imag)
        end

        return out
    end
end

function _legendre_backward(data::AbstractArray, transform::LegendreTransform)
    # Backward transform mirrors `_legendre_forward`, using the grid-space
    # reconstruction matrix built during planning.
    return _execute_on_cpu(data) do host_data
        axis = transform.axis
        if axis > ndims(host_data)
            return host_data
        end

        if transform.backward_matrix === nothing
            return host_data
        end

        mat = transform.backward_matrix
        grid_size = size(mat, 1)
        out_shape = ntuple(i -> i == axis ? grid_size : size(host_data, i), ndims(host_data))
        real_type = real(eltype(host_data))
        out_eltype = eltype(host_data) <: Complex ? eltype(host_data) : real_type
        out = zeros(out_eltype, out_shape)

        other_dims = Tuple(filter(i -> i != axis, 1:ndims(host_data)))

        # Separate real/imaginary scratch keeps real-valued Legendre matrices
        # usable for complex coefficient arrays without complex matrix storage.
        wm = get_global_workspace()
        temp_real = get_workspace!(wm, real_type, (grid_size,))
        temp_imag = eltype(host_data) <: Complex ? get_workspace!(wm, real_type, (grid_size,)) : nothing

        if isempty(other_dims)
            mul!(temp_real, mat, real(host_data))
            if temp_imag === nothing
                copyto!(out, temp_real)
            else
                mul!(temp_imag, mat, imag(host_data))
                out .= complex.(temp_real, temp_imag)
            end
        else
            for (slice_in, slice_out) in zip(eachslice(host_data, dims=other_dims), eachslice(out, dims=other_dims))
                mul!(temp_real, mat, real(slice_in))
                if temp_imag === nothing
                    slice_out .= temp_real
                else
                    mul!(temp_imag, mat, imag(slice_in))
                    slice_out .= complex.(temp_real, temp_imag)
                end
            end
        end

        release_workspace!(wm, temp_real)
        if temp_imag !== nothing
            release_workspace!(wm, temp_imag)
        end

        return out
    end
end

# Dispatch for transform loop
_apply_forward(current, t::LegendreTransform) = _legendre_forward(current, t)
_apply_backward(current, t::LegendreTransform) = _legendre_backward(current, t)
