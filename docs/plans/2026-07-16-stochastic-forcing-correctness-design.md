# Stochastic Forcing Correctness Design

## Goal

Make stochastic forcing honor its documented spatial covariance and mean-energy
injection rate in Tarang's actual FFT convention, apply it to the requested flattened
state component, and refuse timestepper combinations that alter white-noise statistics.

## Public API

`StochasticForcing` gains `injection_metric`, with supported values `:direct` (default)
and `:vorticity_kinetic`. `:direct` targets the domain-mean quadratic energy
`mean(abs2(u))/2`. `:vorticity_kinetic` targets domain-mean kinetic energy when the
forced state is vorticity, using the spectral weight `1/|k|^2`. The metric is stored on
the forcing so construction and work diagnostics use the same inner product.

`add_stochastic_forcing!` resolves symbols against the flattened scalar solver state.
Scalar variables after vectors/tensors therefore receive their correct offset. Component
symbols such as `:u_x` are supported; a vector/tensor container symbol is rejected as
ambiguous instead of forcing one arbitrary component.

## Statistics and transforms

Tarang uses unnormalised FFT coefficients. For `M = prod(field_size)`, direct-field
covariance is normalized with `sum(Q)/(2M^2) = epsilon`; vorticity kinetic covariance
uses `sum(Q/k^2)/(2M^2) = epsilon`. The Gaussian ring builder produces `sqrt(Q)`, so its
amplitude exponent is half the covariance exponent. Empty positive-rate spectra raise
an `ArgumentError`.

Hermitian self-conjugate modes are multiplied by `sqrt(2)` when projected to the real
axis so their expected variance remains `Q`. Work and power diagnostics use RFFT
multiplicity weights, `M^-2` Parseval normalization, and the selected injection metric.
GPU reductions operate on array expressions rather than scalar-indexing device arrays.

## Timesteppers

Current multistep implementations store and extrapolate the whole explicit RHS. That is
valid for smooth deterministic terms but changes independent forcing into colored noise.
Until stochastic-specific multistep quadrature is implemented, registered stochastic
forcing is rejected for CNAB2, SBDF2/3/4, ETD_CNAB2, and ETD_SBDF2 with an actionable
error recommending RK222/RK443/ETD_RK222 or a first-order one-step scheme. This is safer
than returning statistically incorrect trajectories.

## Verification

Regression tests cover flattened registration, physical resolution-independent energy,
vorticity weighting, RFFT work, ring width, Nyquist variance, empty spectra, and the
multistep guard. Existing CPU and MPI forcing tests remain required. CUDA tests verify
work diagnostics when CUDA is available.
