# Fourier wavenumber storage and layout helpers.

# ============================================================================
# Wavenumber computation
# ============================================================================

"""
    wavenumbers(basis::RealFourier)

Get wavenumbers for RealFourier basis.
"""
function wavenumbers(basis::RealFourier)
    N = basis.meta.size
    L = basis.meta.bounds[2] - basis.meta.bounds[1]
    if abs(L) < 1e-14
        throw(ArgumentError("wavenumbers: domain length is zero"))
    end
    k0 = 2π / L
    # Build wavenumber sequence matching RealFourier storage:
    # [cos_0, cos_1, sin_1, cos_2, sin_2, ..., (optional) cos_nyquist]
    if iseven(N)
        kmax = N ÷ 2
        # cos_0, (cos_k, sin_k) for k=1..kmax-1, then cos_kmax (Nyquist)
        k_native = vcat([0], vec(repeat(1:(kmax - 1), inner=2)), [kmax])
    else
        kmax = (N - 1) ÷ 2
        # cos_0, (cos_k, sin_k) for k=1..kmax
        k_native = vcat([0], vec(repeat(1:kmax, inner=2)))
    end
    return k0 .* k_native
end

"""
    wavenumbers(basis::ComplexFourier)

Get wavenumbers for ComplexFourier basis.
"""
function wavenumbers(basis::ComplexFourier)
    N = basis.meta.size
    L = basis.meta.bounds[2] - basis.meta.bounds[1]
    if abs(L) < 1e-14
        throw(ArgumentError("wavenumbers: domain length is zero"))
    end
    k0 = 2π / L
    # FFT ordering:
    # Even N: [0, 1, 2, ..., N/2-1, -N/2, ..., -2, -1]
    # Odd N:  [0, 1, 2, ..., (N-1)/2, -(N-1)/2, ..., -2, -1]
    if iseven(N)
        k_native = [0:(N ÷ 2 - 1); -(N ÷ 2):-1]
    else
        kmax = (N - 1) ÷ 2
        k_native = [0:kmax; -kmax:-1]
    end
    return k0 .* k_native
end

"""
    wavenumbers_rfft(basis::RealFourier)

Get wavenumbers for RealFourier basis in RFFT (real-to-complex FFT) layout.

RFFT output has size N/2+1 complex values representing non-negative frequencies:
[k=0, k=1, k=2, ..., k=N/2]

This is different from the native RealFourier storage (cos/sin interleaved, length N).
Use this function when working with coefficient arrays from RFFT transforms,
such as in MPI mode with PencilFFTs.
"""
function wavenumbers_rfft(basis::RealFourier)
    N = basis.meta.size
    L = basis.meta.bounds[2] - basis.meta.bounds[1]
    if abs(L) < 1e-14
        throw(ArgumentError("wavenumbers_rfft: domain length is zero"))
    end
    k0 = 2π / L
    # RFFT output: [0, 1, 2, ..., N/2] (N/2+1 values)
    kmax = N ÷ 2
    return k0 .* collect(0:kmax)
end

"""
    wavenumbers_fft(basis::RealFourier)

Get wavenumbers for RealFourier basis in standard FFT layout.

In multi-dimensional transforms, `rfft` is applied to the first axis (producing
N/2+1 complex values), while subsequent RealFourier axes use `fft` (because their
input is already complex). The `fft` output uses standard FFT ordering:

    [k=0, k=1, ..., k=N/2-1, k=-N/2, k=-N/2+1, ..., k=-1]

This function returns wavenumbers matching that layout.
"""
function wavenumbers_fft(basis::RealFourier)
    N = basis.meta.size
    L = basis.meta.bounds[2] - basis.meta.bounds[1]
    if abs(L) < 1e-14
        throw(ArgumentError("wavenumbers_fft: domain length is zero"))
    end
    k0 = 2π / L
    if iseven(N)
        k_fft = Float64.([0:(N ÷ 2 - 1); -(N ÷ 2):-1])
    else
        kmax = (N - 1) ÷ 2
        k_fft = Float64.([0:kmax; -kmax:-1])
    end
    return k0 .* k_fft
end

"""
    wavenumbers_for_coefficients(basis, use_rfft::Bool=false)

Get wavenumbers matching the coefficient array layout.

- For RealFourier with use_rfft=true: returns RFFT layout [0, 1, ..., N/2] (length N/2+1)
- For RealFourier with use_rfft=false: returns native cos/sin layout (length N)
- For ComplexFourier: returns FFT layout (length N)
- For other bases: returns nothing (no wavenumber concept)

Use use_rfft=true when:
- Working with PencilFFTs transforms (RFFT on first axis)
- Working with FFTW.rfft output
- Computing spectral operators on RFFT coefficient arrays
"""
function wavenumbers_for_coefficients(basis::RealFourier; use_rfft::Bool=false)
    if use_rfft
        return wavenumbers_rfft(basis)
    else
        return wavenumbers(basis)
    end
end

function wavenumbers_for_coefficients(basis::ComplexFourier; use_rfft::Bool=false)
    return wavenumbers(basis)
end

function wavenumbers_for_coefficients(basis::Basis; use_rfft::Bool=false)
    # Non-Fourier bases don't have wavenumber concept
    return nothing
end
