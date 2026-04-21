"""Spectrum binning configuration and shared spectral metadata types."""

"""
    SpectrumBinning

Configuration for spectrum binning to smooth power spectra.

# Fields
- `mode::Symbol`: Binning mode - `:linear`, `:log`, or `:custom`
- `bin_width::Float64`: Bin width for linear binning (default: 1.0)
- `num_bins::Int`: Number of bins for logarithmic binning
- `bin_edges::Vector{Float64}`: Custom bin edges (for `:custom` mode)
"""
struct SpectrumBinning
    mode::Symbol
    bin_width::Float64
    num_bins::Int
    bin_edges::Vector{Float64}
end

# Constructors for different binning modes
"""
    LinearBinning(; bin_width=1.0)

Create linear binning configuration with specified bin width.
Default bin_width=1.0 gives integer wavenumber bins (k=0,1,2,...).
Larger bin_width smooths the spectrum by averaging over wider k ranges.

# Example
```julia
# Standard integer bins
P = power_spectrum(field, binning=LinearBinning())

# Smoothed with Δk=2
P = power_spectrum(field, binning=LinearBinning(bin_width=2.0))
```
"""
LinearBinning(; bin_width::Real=1.0) = SpectrumBinning(:linear, Float64(bin_width), 0, Float64[])

"""
    LogBinning(; num_bins=nothing, bins_per_decade=10)

Create logarithmic binning configuration for spectra spanning many decades.
Useful for turbulence spectra where equal spacing in log(k) is desired.

# Arguments
- `num_bins::Int`: Total number of bins (overrides bins_per_decade if specified)
- `bins_per_decade::Int=10`: Number of bins per decade of wavenumber

# Example
```julia
# Logarithmic bins, 10 per decade
P = power_spectrum(field, binning=LogBinning())

# Coarser log bins
P = power_spectrum(field, binning=LogBinning(bins_per_decade=5))

# Fixed number of log bins
P = power_spectrum(field, binning=LogBinning(num_bins=50))
```
"""
function LogBinning(; num_bins::Union{Int,Nothing}=nothing, bins_per_decade::Int=10)
    if num_bins === nothing
        # Will be computed based on kmax later
        return SpectrumBinning(:log, 0.0, -bins_per_decade, Float64[])  # Negative = bins_per_decade
    else
        return SpectrumBinning(:log, 0.0, num_bins, Float64[])
    end
end

"""
    CustomBinning(bin_edges::Vector{<:Real})

Create custom binning with user-specified bin edges.

# Arguments
- `bin_edges`: Vector of bin edges [k0, k1, k2, ...]. Bins are [k0,k1), [k1,k2), etc.

# Example
```julia
# Custom bins: [1,2), [2,4), [4,8), [8,16), ...
edges = [1, 2, 4, 8, 16, 32, 64, 128]
P = power_spectrum(field, binning=CustomBinning(edges))
```
"""
CustomBinning(bin_edges::Vector{<:Real}) = SpectrumBinning(:custom, 0.0, length(bin_edges)-1, Float64.(bin_edges))

struct WavenumberInfo
    kmax::Int
    k_magnitudes::Array{Float64}
    kx_grid::Array{Float64}
    ky_grid::Array{Float64}
    kz_grid::Union{Array{Float64}, Nothing}
    domain_size::Tuple{Vararg{Float64}}
    fourier_shape::Tuple{Vararg{Int}}
end
