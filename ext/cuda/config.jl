# ============================================================================
# GPU Configuration
# ============================================================================

# CUDA.jl does not export deviceid() — use device().handle for the integer ordinal
_current_device_id() = Int(CUDA.device().handle)

# ============================================================================
# Tensor Core Support
# ============================================================================

"""
    enable_tensor_cores!()

Switch this process's CUDA math mode to `FAST_MATH`, which lets cuBLAS use
Tensor Cores (TF32) for `Float32`/`Float64` GEMMs on Volta+ GPUs.

!!! warning "This changes numerics, process-wide and for everything"
    This is **not** a free speedup, and it is not scoped to whatever you had in
    mind when you called it. `CUDA.math_mode!` is global CUDA.jl state: every
    subsequent cuBLAS call in this process is affected, including the batched
    stage-matrix LU that `BatchedDenseLU` runs inside the timestepper. TF32
    carries ~10 bits of mantissa, so a spectral solve that depends on
    well-conditioned Chebyshev tau rows can lose several digits, and an
    ill-conditioned stage matrix can go from "solvable" to "silently wrong".

    Enable it only for a run you are prepared to validate against a
    `DEFAULT_MATH` baseline, and call [`disable_tensor_cores!`](@ref) to restore
    strict IEEE behaviour.
"""
function enable_tensor_cores!()
    try
        CUDA.math_mode!(CUDA.FAST_MATH)
        @warn "CUDA math mode set to FAST_MATH process-wide. Every cuBLAS call — " *
              "including the batched stage-matrix LU inside the timestepper — now " *
              "runs at reduced precision. Validate against a disable_tensor_cores! " *
              "baseline before trusting these results." maxlog=1
    catch e
        @warn "Could not enable Tensor Cores: $e"
    end
end

"""
    disable_tensor_cores!()

Disable Tensor Core operations for strict IEEE compliance.
"""
function disable_tensor_cores!()
    try
        CUDA.math_mode!(CUDA.DEFAULT_MATH)
        @info "Tensor Cores disabled (DEFAULT_MATH mode)"
    catch e
        @warn "Could not disable Tensor Cores: $e"
    end
end
