### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 03bf01fb-6e7a-454e-9776-83d972d11f3c
begin
    using Tarang
    using NetCDF
    using CairoMakie
    using FFTW
end

# ╔═╡ 409d1644-7bc8-4ebb-98bb-334a6519a462
begin
    nc_path = "snapshots/snapshots_s1.nc"
    var_name = "q"
    streamfunction_var = "psi"
    time_index = 1
end

# ╔═╡ 4cb5fbf9-8755-4193-a446-d01b338b9cd8
function read_output_var(path, name)
    try
        Tarang.group_ncread(path, "vars", name)
    catch
        ncread(path, name)
    end
end

# ╔═╡ 89f75ca8-96af-490f-9a4c-88150aa057a7
begin
    x_name = "x"
    y_name = "y"
end

# ╔═╡ 7b412973-7e1a-4bd2-aaf4-0e52f0785973
function read_grid_var(path, name)
    try
        Tarang.group_ncread(path, "grids", name)
    catch
        ncread(path, name)
    end
end

# ╔═╡ 6fb73321-a15b-471e-a0a5-f891ef9c4bff
begin
    q = read_output_var(nc_path, var_name)
    x = read_grid_var(nc_path, x_name)
    y = read_grid_var(nc_path, y_name)
end

# ╔═╡ f177e0d1-1bda-45be-9dd4-6f26794476c7
z = ndims(q) == 3 ? Matrix(q[time_index, :, :]) : Matrix(q)

# ╔═╡ 15f52eb4-d6b2-4a8a-8f96-7f4f31249e32
ψ_data = read_output_var(nc_path, streamfunction_var)

# ╔═╡ 72ca16c9-32d3-4b10-9d63-d2bbf54bbd15
ψ = ndims(ψ_data) == 3 ? Matrix(ψ_data[time_index, :, :]) : Matrix(ψ_data)

# ╔═╡ 116c7806-51dc-4079-965f-5fe85f7811fe
function angular_wavenumbers(coord)
    n = length(coord)
    dx = n > 1 ? coord[2] - coord[1] : 1.0
    L = dx * n
    k0 = 2π / L
    return [((j <= n ÷ 2) ? j : j - n) * k0 for j in 0:(n - 1)]
end

# ╔═╡ 5ed87737-5045-4ffc-b748-71e707f6f593
function kinetic_energy_spectrum_from_streamfunction(ψ, x, y)
    nx, ny = size(ψ)
    kx = angular_wavenumbers(x)
    ky = angular_wavenumbers(y)
    dk_candidates = Float64[]
    length(kx) > 1 && push!(dk_candidates, abs(kx[2] - kx[1]))
    length(ky) > 1 && push!(dk_candidates, abs(ky[2] - ky[1]))
    dk = minimum(filter(>(0), dk_candidates))

    ψ̂ = fft(Float64.(ψ)) ./ length(ψ)
    kmax = maximum(hypot(kxi, kyj) for kxi in kx, kyj in ky)
    nbins = floor(Int, kmax / dk) + 1
    energy = zeros(Float64, nbins)
    counts = zeros(Int, nbins)

    for i in 1:nx, j in 1:ny
        kmag = hypot(kx[i], ky[j])
        kmag == 0 && continue
        bin = floor(Int, kmag / dk) + 1
        energy[bin] += 0.5 * (kx[i]^2 + ky[j]^2) * abs2(ψ̂[i, j])
        counts[bin] += 1
    end

    k = collect(0:(nbins - 1)) .* dk
    keep = counts .> 0
    return (k=k[keep], energy=energy[keep], counts=counts[keep])
end

# ╔═╡ c826698b-9e88-4e4b-a96b-a495793c85ff
ke_spectrum = kinetic_energy_spectrum_from_streamfunction(ψ, x, y)

# ╔═╡ b4a84cde-0ff7-46a9-94d5-207265fa16a8
begin
    fig = Figure(size=(800, 600))
    ax = Axis(
        fig[1, 1],
        xlabel=x_name,
        ylabel=y_name,
        title="$(var_name), time index $(time_index)",
        xgridvisible=true,
        ygridvisible=true,
    )
    hm = heatmap!(ax, x, y, z; colormap=:viridis)
    Colorbar(fig[1, 2], hm, label=var_name)
    fig
end

# ╔═╡ 33e08a1f-dd7e-4041-af90-5d29c9790601
begin
    fig = Figure(size=(800, 600))
    ax = Axis(
        fig[1, 1],
        xlabel="k",
        ylabel="E(k)",
        title="Kinetic energy spectrum from $(streamfunction_var), time index $(time_index)",
        xscale=log10,
        yscale=log10,
        xgridvisible=true,
        ygridvisible=true,
    )
    positive = (ke_spectrum.k .> 0) .& (ke_spectrum.energy .> 0)
    if any(positive)
        lines!(ax, ke_spectrum.k[positive], ke_spectrum.energy[positive])
        scatter!(ax, ke_spectrum.k[positive], ke_spectrum.energy[positive]; markersize=6)
    end
    fig
end

# ╔═╡ Cell order:
# ╠═03bf01fb-6e7a-454e-9776-83d972d11f3c
# ╠═409d1644-7bc8-4ebb-98bb-334a6519a462
# ╠═4cb5fbf9-8755-4193-a446-d01b338b9cd8
# ╠═89f75ca8-96af-490f-9a4c-88150aa057a7
# ╠═7b412973-7e1a-4bd2-aaf4-0e52f0785973
# ╠═6fb73321-a15b-471e-a0a5-f891ef9c4bff
# ╠═f177e0d1-1bda-45be-9dd4-6f26794476c7
# ╠═15f52eb4-d6b2-4a8a-8f96-7f4f31249e32
# ╠═72ca16c9-32d3-4b10-9d63-d2bbf54bbd15
# ╠═116c7806-51dc-4079-965f-5fe85f7811fe
# ╠═5ed87737-5045-4ffc-b748-71e707f6f593
# ╠═c826698b-9e88-4e4b-a96b-a495793c85ff
# ╠═b4a84cde-0ff7-46a9-94d5-207265fa16a8
# ╠═33e08a1f-dd7e-4041-af90-5d29c9790601
