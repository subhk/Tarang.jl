"""
Diffusive (parabolic) stability limit of the adaptive CFL controller.

`compute_timestep` is advective-only until a diffusivity is registered with
`add_diffusivity!`. These tests pin down

  * the advection-only path is byte-for-byte the behaviour of `test_cfl.jl`,
  * a constant diffusivity reproduces `dt = safety / (2 ν Σᵢ Δxᵢ⁻²)`,
  * an array / `ScalarField` diffusivity uses its GLOBAL maximum,
  * the tighter of the advective and diffusive limits wins,
  * anisotropic (and Chebyshev) spacings are summed as Σᵢ Δxᵢ⁻², not maxed,
  * every registered entry flows through the single batched `Allreduce(MAX)`.
"""

using Test
using Tarang
using MPI

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

const _CFL_AXES = ("x", "y", "z")

"""
Periodic Fourier IVP carrying a uniform velocity `vels` on a grid of `sizes`
over `bounds`. Returns `(u, solver, dist, bases)`.
"""
function _cfl_diffusive_setup(sizes::Tuple, bounds::Tuple, vels::Tuple)
    names = _CFL_AXES[1:length(sizes)]
    coords = CartesianCoordinates(names...)
    dist = Distributor(coords; dtype=Float64)
    bases = ntuple(i -> RealFourier(coords[names[i]]; size=sizes[i],
                                    bounds=bounds[i], dtype=Float64), length(sizes))

    u = VectorField(dist, coords, "u", bases, Float64)
    for (i, component) in enumerate(u.components)
        Tarang.ensure_layout!(component, :g)
        fill!(Tarang.get_grid_data(component), vels[i])
    end

    problem = IVP([u]; namespace=Dict("u" => u))
    Tarang.add_equation!(problem, "∂t(u) = 0")
    solver = InitialValueSolver(problem, RK111(); device="cpu")
    return u, solver, dist, bases
end

"""Advective frequency `Σᵢ |uᵢ|/Δxᵢ` for a spatially uniform velocity."""
function _advective_frequency(domain, vels::Tuple)
    spacings = Tarang.grid_spacing(domain)
    return sum(abs(vels[i]) / spacings[i] for i in eachindex(spacings))
end

"""Diffusive frequency `2 ν Σᵢ Δxᵢ⁻²`."""
function _diffusive_frequency(domain, ν::Real)
    spacings = Tarang.grid_spacing(domain)
    return 2 * ν * sum(inv(dx^2) for dx in spacings)
end

# ---------------------------------------------------------------------------
# 1. The advection-only path must not move.
# ---------------------------------------------------------------------------

@testset "CFL diffusive: advection-only path unchanged" begin
    # Mirrors the assertion in test_cfl.jl exactly: with nothing registered via
    # add_diffusivity!, dt is still safety * Δx / |u|.
    for N in (16, 32), L in (1.0, 2.0), safety in (0.2, 0.4), velocity_mag in (0.5, 3.0)
        u, solver, _, _ = _cfl_diffusive_setup((N,), ((0.0, L),), (velocity_mag,))

        cfl = CFL(solver; initial_dt=1.0, safety=safety, cadence=1)
        add_velocity!(cfl, u)

        dt = compute_timestep(cfl)

        spacing = Tarang.grid_spacing(u.domain)[1]
        @test isapprox(dt, safety * spacing / abs(velocity_mag); rtol=1e-6)
        @test isempty(cfl.diffusivities)
    end
end

@testset "CFL diffusive: opt-in only" begin
    u, solver, _, _ = _cfl_diffusive_setup((16,), ((0.0, 1.0),), (1.0,))

    # Two controllers on the same state: one bare, one with a diffusivity that
    # is deliberately negligible. They must agree.
    bare = CFL(solver; initial_dt=1.0, safety=0.4, cadence=1, threshold=0.0)
    add_velocity!(bare, u)
    dt_bare = compute_timestep(bare)

    for negligible in (0.0, 1e-12, -1.0)   # zero, tiny, and (clamped) negative
        cfl = CFL(solver; initial_dt=1.0, safety=0.4, cadence=1, threshold=0.0)
        add_velocity!(cfl, u)
        add_diffusivity!(cfl, negligible)
        @test isapprox(compute_timestep(cfl), dt_bare; rtol=1e-12)
    end

    # A negative coefficient is anti-diffusion, not a dt limit: it must never
    # produce a negative or infinite step.
    cfl_neg = CFL(solver; initial_dt=1.0, safety=0.4, cadence=1, threshold=0.0)
    add_velocity!(cfl_neg, u)
    add_diffusivity!(cfl_neg, -1e6)
    dt_neg = compute_timestep(cfl_neg)
    @test dt_neg > 0
    @test isapprox(dt_neg, dt_bare; rtol=1e-12)
end

# ---------------------------------------------------------------------------
# 2. Constant diffusivity: analytic dt on a known grid.
# ---------------------------------------------------------------------------

@testset "CFL diffusive: constant ν analytic dt" begin
    # N = 16 over [0, 1] ⇒ Δx = 1/16 = 0.0625, Δx⁻² = 256.
    # ν = 1, safety = 0.4 ⇒ f_diff = 2·1·256 = 512 ⇒ dt = 0.4/512 = 7.8125e-4.
    # The velocity is tiny so the diffusive limit is the binding one.
    safety = 0.4
    ν = 1.0
    u, solver, _, _ = _cfl_diffusive_setup((16,), ((0.0, 1.0),), (1e-6,))

    cfl = CFL(solver; initial_dt=1.0, safety=safety, cadence=1, threshold=0.0)
    add_velocity!(cfl, u)
    @test add_diffusivity!(cfl, ν) === cfl        # returns the controller
    @test length(cfl.diffusivities) == 1

    dt = compute_timestep(cfl)

    dx = Tarang.grid_spacing(u.domain)[1]
    @test isapprox(dx, 0.0625; rtol=1e-12)
    @test isapprox(dt, safety * dx^2 / (2 * ν); rtol=1e-10)
    @test isapprox(dt, 7.8125e-4; rtol=1e-10)

    # Equivalent 1-D statement of the same limit: dt ≤ Δx²/(2 d ν) with d = 1.
    @test isapprox(dt, safety / _diffusive_frequency(u.domain, ν); rtol=1e-10)
end

@testset "CFL diffusive: isotropic 2-D reduces to Δx²/(2dν)" begin
    # 16×16 over [0,1]² ⇒ Δx = Δy = 1/16, d = 2.
    # Σᵢ Δxᵢ⁻² = 512 ⇒ dt = safety·Δx²/(2·2·ν) = 0.5·(1/256)/(4·0.5) = 9.765625e-4.
    safety = 0.5
    ν = 0.5
    u, solver, _, _ = _cfl_diffusive_setup((16, 16), ((0.0, 1.0), (0.0, 1.0)), (1e-8, 1e-8))

    cfl = CFL(solver; initial_dt=1.0, safety=safety, cadence=1, threshold=0.0)
    add_velocity!(cfl, u)
    add_diffusivity!(cfl, ν)

    dt = compute_timestep(cfl)

    dx = Tarang.grid_spacing(u.domain)[1]
    d = 2
    @test isapprox(dt, safety * dx^2 / (2 * d * ν); rtol=1e-10)
    @test isapprox(dt, 9.765625e-4; rtol=1e-10)
end

# ---------------------------------------------------------------------------
# 3. Array / field diffusivity uses its maximum.
# ---------------------------------------------------------------------------

@testset "CFL diffusive: array ν uses its maximum" begin
    safety = 0.4
    ν_max = 2.0
    u, solver, _, _ = _cfl_diffusive_setup((16,), ((0.0, 1.0),), (1e-6,))

    # Shaped like an LES eddy-viscosity slab: a plain per-rank Array.
    slab = fill(0.05, size(parent(Tarang.get_grid_data(u.components[1]))))
    slab[1] = ν_max
    slab[end] = 0.5 * ν_max

    cfl = CFL(solver; initial_dt=1.0, safety=safety, cadence=1, threshold=0.0)
    add_velocity!(cfl, u)
    add_diffusivity!(cfl, slab)
    dt_array = compute_timestep(cfl)

    # Must equal the constant-ν answer at ν = maximum(slab), NOT the mean.
    reference = CFL(solver; initial_dt=1.0, safety=safety, cadence=1, threshold=0.0)
    add_velocity!(reference, u)
    add_diffusivity!(reference, ν_max)
    dt_constant = compute_timestep(reference)

    @test isapprox(dt_array, dt_constant; rtol=1e-12)
    @test isapprox(dt_array, safety / _diffusive_frequency(u.domain, ν_max); rtol=1e-10)

    mean_ν = sum(slab) / length(slab)
    @test !isapprox(dt_array, safety / _diffusive_frequency(u.domain, mean_ν); rtol=1e-3)
end

@testset "CFL diffusive: array is held by reference (in-place LES refresh)" begin
    # An LES model overwrites νₑ in place every step; registering once must be
    # enough for compute_timestep to see the new values.
    safety = 0.4
    u, solver, _, _ = _cfl_diffusive_setup((16,), ((0.0, 1.0),), (1e-6,))

    slab = fill(1.0, size(parent(Tarang.get_grid_data(u.components[1]))))

    cfl = CFL(solver; initial_dt=1.0, safety=safety, cadence=1, threshold=0.0,
              max_change=100.0, min_change=0.01)
    add_velocity!(cfl, u)
    add_diffusivity!(cfl, slab)

    dt_before = compute_timestep(cfl)
    fill!(slab, 4.0)                       # in-place refresh, no re-registration
    dt_after = compute_timestep(cfl)

    @test isapprox(dt_after, dt_before / 4; rtol=1e-10)
end

@testset "CFL diffusive: ScalarField ν" begin
    safety = 0.4
    ν_max = 0.5
    u, solver, dist, bases = _cfl_diffusive_setup((16,), ((0.0, 1.0),), (1e-6,))

    ν_field = ScalarField(dist, "nu", bases, Float64)
    Tarang.ensure_layout!(ν_field, :g)
    ν_data = Tarang.get_grid_data(ν_field)
    fill!(ν_data, 0.1)
    parent(ν_data)[1] = ν_max

    cfl = CFL(solver; initial_dt=1.0, safety=safety, cadence=1, threshold=0.0)
    add_velocity!(cfl, u)
    add_diffusivity!(cfl, ν_field)

    dt = compute_timestep(cfl)

    # Δx = 1/16 ⇒ f_diff = 2·0.5·256 = 256 ⇒ dt = 0.4/256 = 1.5625e-3.
    @test isapprox(dt, safety / _diffusive_frequency(u.domain, ν_max); rtol=1e-10)
    @test isapprox(dt, 1.5625e-3; rtol=1e-10)

    # The field carries its own domain, so none had to be supplied.
    @test cfl.diffusivities[1].domain === ν_field.domain
end

# ---------------------------------------------------------------------------
# 4. The tighter limit wins.
# ---------------------------------------------------------------------------

@testset "CFL diffusive: diffusive limit wins and loses correctly" begin
    # Δx = 1/16, |u| = 1 ⇒ f_adv = 16. The two limits cross at
    # ν_crit = |u|·Δx/2 = 0.03125, where f_diff = 2·ν·256 = 16.
    safety = 0.4
    velocity_mag = 1.0
    u, solver, _, _ = _cfl_diffusive_setup((16,), ((0.0, 1.0),), (velocity_mag,))

    dx = Tarang.grid_spacing(u.domain)[1]
    f_adv = _advective_frequency(u.domain, (velocity_mag,))
    @test isapprox(f_adv, 16.0; rtol=1e-12)

    ν_crit = velocity_mag * dx / 2
    @test isapprox(ν_crit, 0.03125; rtol=1e-12)

    dt_advective_only = safety / f_adv        # 0.025

    for factor in (0.25, 4.0)
        ν = factor * ν_crit
        cfl = CFL(solver; initial_dt=1.0, safety=safety, cadence=1, threshold=0.0)
        add_velocity!(cfl, u)
        add_diffusivity!(cfl, ν)

        dt = compute_timestep(cfl)
        f_diff = _diffusive_frequency(u.domain, ν)

        # The controller keeps the SMALLER of the two limits, i.e. safety over
        # the LARGER frequency.
        @test isapprox(dt, safety / max(f_adv, f_diff); rtol=1e-10)

        if factor < 1
            @test f_diff < f_adv
            @test isapprox(dt, dt_advective_only; rtol=1e-10)   # diffusion loses
        else
            @test f_diff > f_adv
            @test dt < dt_advective_only                        # diffusion wins
            @test isapprox(dt, safety / f_diff; rtol=1e-10)
            @test isapprox(dt, 0.00625; rtol=1e-10)             # 0.4 / 64
        end
    end

    # Right at the crossover the two agree.
    cfl = CFL(solver; initial_dt=1.0, safety=safety, cadence=1, threshold=0.0)
    add_velocity!(cfl, u)
    add_diffusivity!(cfl, ν_crit)
    @test isapprox(compute_timestep(cfl), dt_advective_only; rtol=1e-10)
end

# ---------------------------------------------------------------------------
# 5. Anisotropic spacing: Σᵢ Δxᵢ⁻², not maxᵢ.
# ---------------------------------------------------------------------------

@testset "CFL diffusive: anisotropic spacing sums Δxᵢ⁻²" begin
    # 16 × 8 over [0,1]² ⇒ Δx = 1/16 (Δx⁻² = 256), Δy = 1/8 (Δy⁻² = 64).
    # Σ = 320 ⇒ f_diff = 2·1·320 = 640 ⇒ dt = 0.4/640 = 6.25e-4.
    safety = 0.4
    ν = 1.0
    u, solver, _, _ = _cfl_diffusive_setup((16, 8), ((0.0, 1.0), (0.0, 1.0)), (0.0, 0.0))

    cfl = CFL(solver; initial_dt=1.0, safety=safety, cadence=1, threshold=0.0)
    add_velocity!(cfl, u)
    add_diffusivity!(cfl, ν)

    dt = compute_timestep(cfl)

    spacings = Tarang.grid_spacing(u.domain)
    @test isapprox(spacings[1], 1 / 16; rtol=1e-12)
    @test isapprox(spacings[2], 1 / 8; rtol=1e-12)

    inv_dx2 = [inv(dx^2) for dx in spacings]
    @test isapprox(sum(inv_dx2), 320.0; rtol=1e-12)

    @test isapprox(dt, safety / (2 * ν * sum(inv_dx2)); rtol=1e-10)
    @test isapprox(dt, 6.25e-4; rtol=1e-10)

    # Explicitly NOT the max-over-axes form (which would give 0.4/512 = 7.8125e-4).
    @test !isapprox(dt, safety / (2 * ν * maximum(inv_dx2)); rtol=1e-3)

    # A zero velocity field contributes no advective frequency, so the diffusive
    # limit alone sets dt — it is not swallowed by the velocity path.
    @test dt < 1.0
end

@testset "CFL diffusive: Chebyshev near-wall spacing via explicit domain" begin
    # grid_spacing uses the MINIMUM Chebyshev-Gauss-Lobatto spacing,
    # Δz = L(1 - cos(π/(N-1)))/2, which is far below L/N near the wall.
    safety = 0.4
    ν = 1e-3
    u, solver, _, _ = _cfl_diffusive_setup((16,), ((0.0, 1.0),), (1e-6,))

    zcoords = CartesianCoordinates("z")
    zdist = Distributor(zcoords; dtype=Float64)
    zb = ChebyshevT(zcoords["z"]; size=16, bounds=(0.0, 1.0))
    zdomain = Tarang.Domain(zdist, (zb,))

    dz = Tarang.grid_spacing(zdomain)[1]
    @test isapprox(dz, (1 - cos(π / 15)) / 2; rtol=1e-12)
    @test dz < 1 / 16                      # much tighter than the uniform guess

    cfl = CFL(solver; initial_dt=1.0, safety=safety, cadence=1, threshold=0.0)
    add_velocity!(cfl, u)
    add_diffusivity!(cfl, ν; domain=zdomain)

    dt = compute_timestep(cfl)
    @test isapprox(dt, safety / (2 * ν / dz^2); rtol=1e-10)

    # Using L/N instead of the near-wall spacing would overestimate dt ~33×.
    @test dt < safety / (2 * ν * 16.0^2)
end

# ---------------------------------------------------------------------------
# 6. Registration bookkeeping and the batched MPI reduction.
# ---------------------------------------------------------------------------

@testset "CFL diffusive: diffusivity without any velocity" begin
    # Registering only a diffusivity must not hit the "nothing registered"
    # early return — the limit would be silently dropped.
    safety = 0.4
    ν = 1.0
    u, solver, _, _ = _cfl_diffusive_setup((16,), ((0.0, 1.0),), (1.0,))

    cfl = CFL(solver; initial_dt=1.0, safety=safety, cadence=1, threshold=0.0)
    add_diffusivity!(cfl, ν)               # no add_velocity!

    dt = compute_timestep(cfl)

    @test isempty(cfl.velocities)
    @test isapprox(dt, safety / _diffusive_frequency(u.domain, ν); rtol=1e-10)
    @test dt != 1.0                        # not the untouched initial_dt
end

@testset "CFL diffusive: batched single Allreduce over all entries" begin
    # Every velocity and every diffusivity shares one Allreduce(MAX) buffer of
    # length n_vel + n_diff. Registering several of each and recovering the
    # single tightest limit shows they all reach the reduction.
    safety = 0.4
    u, solver, _, _ = _cfl_diffusive_setup((16,), ((0.0, 1.0),), (1.0,))

    cfl = CFL(solver; initial_dt=1.0, safety=safety, cadence=1, threshold=0.0)
    add_velocity!(cfl, u)
    add_velocity!(cfl, u)                  # 2 velocities
    add_diffusivity!(cfl, 0.01)
    add_diffusivity!(cfl, 0.25)            # 2 diffusivities; this one binds
    add_diffusivity!(cfl, 0.02)

    @test length(cfl.velocities) == 2
    @test length(cfl.diffusivities) == 3

    dt = compute_timestep(cfl)

    frequencies = [_advective_frequency(u.domain, (1.0,)),
                   _advective_frequency(u.domain, (1.0,)),
                   _diffusive_frequency(u.domain, 0.01),
                   _diffusive_frequency(u.domain, 0.25),
                   _diffusive_frequency(u.domain, 0.02)]

    @test isapprox(dt, safety / maximum(frequencies); rtol=1e-10)
    @test argmax(frequencies) == 4         # the ν = 0.25 entry is the binding one

    # Exercise the reduction primitive itself. At np = 1 an Allreduce(MAX) is the
    # identity, so the round trip must preserve the per-entry local maxima.
    buffer = copy(frequencies)
    Tarang.reduce_vector!(cfl.reducer, buffer, MPI.MAX)
    if !MPI.Initialized() || MPI.Comm_size(cfl.reducer.comm) == 1
        @test buffer == frequencies
    else
        @test all(buffer .>= frequencies)
    end
end

@testset "CFL diffusive: struct surface" begin
    u, solver, _, _ = _cfl_diffusive_setup((16,), ((0.0, 1.0),), (1.0,))

    @test hasfield(Tarang.CFL, :diffusivities)
    @test hasmethod(add_diffusivity!, Tuple{Tarang.CFL, Float64})
    @test hasmethod(add_diffusivity!, Tuple{Tarang.CFL, Array{Float64, 1}})
    @test hasmethod(add_diffusivity!, Tuple{Tarang.CFL, ScalarField})

    cfl = CFL(solver; initial_dt=1.0, safety=0.4, cadence=1)
    @test isempty(cfl.diffusivities)

    # The compact show flags a registered diffusivity but stays quiet otherwise.
    @test !contains(repr(cfl), "diffusivities")
    add_diffusivity!(cfl, 1.0)
    @test contains(repr(cfl), "1 diffusivities")
    @test contains(repr("text/plain", cfl), "Diffusivities:")
end
