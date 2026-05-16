# Third-party and standard-library dependencies imported into the Tarang module.
#
# Keeping these imports here separates package dependency scope from the root
# module shell, implementation load order, exports, and runtime initialization.

using MPI
using PencilArrays
using PencilFFTs
using LinearAlgebra
using LinearAlgebra: BLAS
using SparseArrays
using FFTW
using AbstractFFTs: Plan as AbstractFFTPlan
using StaticArrays
using Parameters
using ChainRulesCore
using ForwardDiff
using LoopVectorization
using ExponentialUtilities  # For Krylov-based exponential integrators
using FastGaussQuadrature   # For Gauss-Legendre quadrature in Legendre transforms
using KernelAbstractions    # For backend-agnostic GPU/CPU kernels
