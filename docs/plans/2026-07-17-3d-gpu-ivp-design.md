# 3D GPU IVP Design

## Scope

Tarang will support and continuously verify two single-GPU 3D initial-value
problem layouts. A fully periodic `RealFourier × RealFourier × RealFourier`
field uses the existing field-native GPU RK path and skips unused global CPU
matrix assembly. A channel-style `RealFourier × RealFourier × ChebyshevT`
field uses the existing per-Fourier-mode coupled subproblem path, with
`matsolver=:auto` selecting `CuSparseLU`.

The change is deliberately test-led. Existing transform and timestepper code is
already dimension-generic in several places, so production code will change
only where an end-to-end 3D test exposes a concrete dimension assumption,
host fallback, or warmed device allocation. The established 2D behavior and
its tests remain unchanged.

## Fully periodic data flow

The periodic test advances a small 3D scalar diffusion IVP on CPU and GPU from
identical analytic initial data. It verifies coefficient agreement, field and
timestepper workspace device residency, finite output, and bounded warmed
device allocations. A registered three-dimensional `StochasticForcing` smoke
test verifies that forcing generation, Hermitian projection, real-FFT matching,
and timestep caching work on a `CuArray` without scalar indexing.

## Mixed-basis data flow

The channel test uses a scalar diffusion equation with two tau fields and
homogeneous Dirichlet boundary conditions. `SeparableStochasticForcing` draws a
two-dimensional Fourier realization and takes its outer product with a
quadrature-normalized Chebyshev profile. Identical seeded CPU and GPU problems
run for several RK222 steps. Their coefficients must agree within GPU transform
and sparse-solver tolerance.

The GPU solver must use `CuSparseLU`. The field, forcing, RK workspaces,
subproblem gather/scatter vectors, CSR matrix values, and cached sparse-solve
buffers must all be GPU-resident. Forcing stays fixed across RK stages and
changes between timesteps. After warmup, another fixed-resolution step must
allocate zero device bytes; if a CUDA library necessarily allocates internally,
the test may instead enforce a small resolution-independent ceiling measured at
two resolutions.

## Error handling and verification

Unsupported CPU-only matrix solvers remain rejected for coupled GPU states.
CUDA scalar indexing remains disabled throughout the tests. CPU-only hosts skip
the conditional GPU assertions cleanly, while the ordinary CPU suite exercises
3D constructor, shape, and solver-selection behavior. Documentation will state
that single-GPU 3D support covers both fully periodic and two-periodic-one-bounded
domains; distributed multi-GPU behavior remains governed by the existing MPI
and distributed-DCT tests.
