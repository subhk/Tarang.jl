# Complete 2D Single-GPU Fourier--Chebyshev Design

## Goal

Make the two-dimensional single-GPU Fourier--Chebyshev path complete enough for
value-based validation on an NVIDIA cluster. Completion means that transforms,
nonlinear products, Chebyshev/Fourier derivatives, implicit channel
subproblems, and RK stepping stay on the device and agree with the CPU oracle.
Both basis orders and real/complex Fourier bases are in scope. Multi-GPU MPI is
explicitly out of scope for this first milestone.

## Transform architecture

`forward_layout` and `backward_layout` remain the single source of truth for
axis operations and intermediate shapes. The CUDA mixed plan will record the
shape before and after every transform stage rather than assuming that every
stage fits the final coefficient shape.

Fourier axes continue to execute first. A real Fourier axis uses R2C only when
the shared layout rule says the arriving data is real; all later Fourier stages
are C2C. Chebyshev stages use Tarang's existing DCT-I convention.

Scaled Chebyshev axes gain two device-only operations:

1. Forward: DCT-I on the scaled grid, then truncate the transformed axis to the
   basis coefficient length.
2. Backward: zero-pad the coefficient axis to the scaled grid length, then
   apply inverse DCT-I.

Cached scratch storage is sized from the largest intermediate shape and keyed
by device, element type, and the complete stage-shape sequence. No device-to-host
fallback is permitted. Unsupported basis types and dimensions still fail with
an actionable error.

## Nonlinear and solver data flow

The serial padded nonlinear evaluator retains its current mixed-basis rule:
only Fourier axes are expanded for 3/2-rule dealiasing; the Chebyshev axis stays
in nodal space. CUDA padding/truncation kernels therefore operate on the Fourier
axis while preserving every Chebyshev row or column. Tests will cover both basis
orders and Nyquist-sensitive inputs.

Fourier--Chebyshev IVPs continue to use coupled per-Fourier-mode subproblems and
the CUDA sparse solver. Tau fields and boundary conditions are enforced by the
existing subproblem machinery. This work does not replace that solver; it adds
an end-to-end nonlinear value oracle that exercises it together with the new
transform shapes.

The supported runtime contract is one Julia task performing transforms at a
time on one CUDA device. Existing caches share scratch for identical plans, so
concurrent same-shape transforms remain outside this milestone.

## Error handling

The GPU dispatcher will reject:

- unsupported basis families;
- fields above the implemented dimensionality;
- malformed coefficient or grid shapes that do not match the shared layout;
- direct multi-rank 2D transforms outside the distributed APIs; and
- attempts to run the cluster validation entry point without functional CUDA.

Failures must occur before partial mutation whenever practical and must never
stage through CPU memory.

## Verification

Conditional CUDA tests will compare CPU and GPU values for:

- unscaled and scaled `RealFourier x ChebyshevT` transforms;
- `ChebyshevT x RealFourier` and complex-Fourier axis order variants;
- Chebyshev and Fourier derivatives;
- 3/2-dealiased nonlinear products, including even-size Nyquist cases; and
- a nonlinear, wall-bounded RK222 IVP with tau boundary conditions and GPU
  sparse subproblem solves.

Every CUDA test enables `CUDA.allowscalar(false)`. Round trips must preserve the
coefficient buffer, and CPU/GPU coefficient and grid comparisons use explicit
tolerances appropriate for FFT/DCT ordering and sparse factorization.

A dedicated cluster runner will require `CUDA.functional()`, print device and
CUDA versions, execute the focused 2D FC suite, and exit nonzero if CUDA is
missing, a test is skipped, or an assertion fails. Existing CPU transform,
nonlinear, solver, and guard suites remain regression gates on non-GPU hosts.
