# O(N log N) FFT-based DCT-II / DCT-III along one dimension of an N-D array.
#
# Replaces the O(N²) naive cos-sum used by the GPU multi-dimensional DCT
# (`gpu_dct_dim!`). DEVICE-AGNOSTIC: `fft`/`ifft` dispatch to FFTW (CPU `Array`)
# or CUFFT (`CuArray`) via AbstractFFTs; the even–odd reorder is an index gather
# and the twiddle/scale are broadcasts — all of which run on CPU and GPU. This
# lets the algorithm be validated on the CPU (see test/test_fft_dct.jl) and run
# unchanged on the GPU.
#
# Convention matched EXACTLY to Tarang's DCT (so it is a drop-in for the cos-sum):
#   forward (DCT-II):  X[k] = scale_k · Σ_n x_n cos(π k (2n+1) / 2N),
#                      scale_0 = 1/(2N),  scale_{k≥1} = 1/N
#   backward (DCT-III): x_n = 2 Σ_k X[k] cos(π k (2n+1) / 2N)
#
# FFT identity used: with the even–odd reorder v = [x_0,x_2,…,x_{N-1},…,x_3,x_1]
# and V = FFT(v), the unnormalized DCT-II is Re(2·e^{-iπk/2N}·V[k]); applying the
# scale above reproduces Tarang's convention. The DCT-III is the matching inverse.

"""Even–odd reorder permutation `v[j] = x[perm[j]]` (1-based) for the N-point DCT."""
_dct_reorder_perm(N::Int) = Int[(j - 1) < cld(N, 2) ? 2 * (j - 1) + 1 : 2 * (N - (j - 1) - 1) + 2 for j in 1:N]

"""
    fft_dct_forward_dim(x, dim) -> X

O(N log N) forward DCT-II along `dim` (Tarang convention). `x` must be real-valued
(complex fields transform their real/imaginary parts separately upstream).
Works on `Array` (FFTW) and `CuArray` (CUFFT).
"""
function fft_dct_forward_dim(x::AbstractArray{T}, dim::Int) where {T<:Real}
    N  = size(x, dim)
    nd = ndims(x)
    RT = float(T)
    perm = _dct_reorder_perm(N)
    idx  = ntuple(i -> i == dim ? perm : Colon(), nd)
    V    = fft(complex(x)[idx...], dim)
    # twiddle/scale built ON x's device (so the broadcast stays on-device for CuArray)
    kr = _dct_kgrid(x, N, dim, nd, RT)
    tw = cis.(-RT(π) .* kr ./ (2N))
    sc = ifelse.(kr .== 0, one(RT) / (2N), one(RT) / N)
    return real.(tw .* V) .* sc
end

# Reshape `0:N-1` along `dim` as an array on the same device as `x`.
function _dct_kgrid(x::AbstractArray, N::Int, dim::Int, nd::Int, ::Type{RT}) where {RT}
    k = similar(x, RT, N)
    copyto!(k, RT.(0:N-1))
    return reshape(k, ntuple(i -> i == dim ? N : 1, nd))
end

"""
    fft_dct_backward_dim(X, dim) -> x

O(N log N) backward DCT-III along `dim` (inverse of `fft_dct_forward_dim`).
"""
function fft_dct_backward_dim(X::AbstractArray{T}, dim::Int) where {T<:Real}
    N  = size(X, dim)
    nd = ndims(X)
    RT = float(T)
    kr = _dct_kgrid(X, N, dim, nd, RT)
    tw_b = cis.(RT(π) .* kr ./ (2N))
    w = real.(ifft(complex(X) .* tw_b, dim)) .* (2N)
    perm = _dct_reorder_perm(N)
    idx  = ntuple(i -> i == dim ? perm : Colon(), nd)
    out  = similar(w)
    out[idx...] = w          # inverse reorder: out[perm[j]] = w[j]
    return out
end
