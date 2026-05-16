"""
Tarang.jl - Spectral PDE framework for Julia

"""

module Tarang

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

include("load_order.jl")
include("public_api.jl")
include("namespaces.jl")
include("runtime_init.jl")

end # module
