# Fourier--Chebyshev GPU Forcing Design

## Goal

Add generic, physically explicit stochastic forcing for mixed Fourier--Chebyshev
domains and make the corresponding IVP subproblem path GPU-safe with bounded
steady-state device allocations. Preserve the existing all-Fourier forcing and
solver behavior.

## Public API

Introduce `SeparableStochasticForcing`. It generates stochastic phases and a
band-limited spectrum only over the periodic Fourier dimensions, then forms the
mixed coefficient forcing as a separable product with a user-supplied
Chebyshev profile.

```julia
forcing = SeparableStochasticForcing(
    fourier_size=(Nx,),
    chebyshev_basis=zbasis,
    chebyshev_profile=z -> 1 - z^2,
    domain_size=(Lx,),
    energy_injection_rate=epsilon,
    k_forcing=kf,
    dk_forcing=dk,
    injection_metric=:direct,
    architecture=GPU(),
)
```

The Chebyshev basis supplies the grid, transform convention, bounds, and
quadrature metadata. The profile may be supplied as coefficients or as a
function sampled on that grid and transformed once during construction. The
constructor normalizes the profile with the domain quadrature and rejects
incompatible lengths, zero norm, and non-finite values.

The first version supports exact normalization for `injection_metric=:direct`.
It rejects `:vorticity_kinetic` for mixed bases because the periodic `1/k^2`
weight is not the inverse channel Laplacian.

## Runtime data flow

The forcing owns persistent arrays for the Fourier spectrum, Fourier random
phases, Chebyshev profile, Fourier realization, and assembled mixed coefficient
forcing. At the start of each timestep it:

1. fills the Fourier phase buffer;
2. applies the Fourier spectrum and `1/sqrt(dt)` white-noise scaling;
3. enforces Hermitian symmetry only along real-Fourier axes;
4. writes the outer product with the Chebyshev profile into the persistent
   mixed coefficient buffer; and
5. reuses that realization for every RK substep.

No Chebyshev index is interpreted as a Fourier wavenumber.

## GPU mixed-basis IVP path

Fourier--Chebyshev problems continue to use coupled per-mode subproblems; the
pure-Fourier in-place Poisson and skew-gradient shortcut remains guarded by an
all-Fourier basis check. This preserves boundary-condition and tau-equation
semantics.

When the state is GPU-resident and the caller left the matrix solver at its
default, solver construction selects the CUDA sparse direct backend for the
coupled subproblems. An explicitly requested solver remains authoritative; an
explicit CPU-only solver with GPU fields fails early with an actionable error.

Subproblem matrices, RHS/solution vectors, and CUDA sparse-solver scratch arrays
are cached and reused. CPU and pure-Fourier dispatch are unchanged.

## Error handling

Construction fails early for:

- mixed `:vorticity_kinetic` normalization;
- missing or incompatible Chebyshev profile metadata;
- zero-norm or non-finite profiles;
- unsupported profile sizes; and
- explicitly selected CPU-only matrix solvers with GPU-resident coupled fields.

Fallbacks must never silently reinterpret a Chebyshev axis as periodic.

## Verification

CPU forcing tests compare the mixed coefficient array with an explicit Fourier
spectrum--profile product and verify reproducibility, Hermitian symmetry,
within-step caching, between-step renewal, normalization, and constructor
errors.

A small CPU Fourier--Chebyshev IVP provides a numerical reference. A conditional
CUDA version runs with `CUDA.allowscalar(false)`, verifies GPU residency of the
state, forcing, stage, matrix, and subproblem buffers, and compares several
RK222 steps with the CPU result.

After warming plans, factorizations, and buffers, the CUDA test measures another
step. The target is zero device allocation. If a CUDA sparse primitive requires
unavoidable internal scratch, the accepted ceiling must be fixed and
resolution-independent; full-field per-step allocations remain forbidden.

Existing Fourier--Fourier forcing, mixed CPU subproblem, transform, MPI, and
inventory tests remain regression gates.
