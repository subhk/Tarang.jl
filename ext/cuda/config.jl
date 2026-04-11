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

Enable Tensor Core operations for compatible CUDA operations.
Provides significant speedup on Volta+ GPUs for matrix operations.
"""
function enable_tensor_cores!()
    try
        CUDA.math_mode!(CUDA.FAST_MATH)
        @info "Tensor Cores enabled (FAST_MATH mode)"
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
