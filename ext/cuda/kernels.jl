# ============================================================================
# KernelAbstractions Kernels for Element-wise Operations
# ============================================================================

"""
Element-wise addition kernel: c = a + b
"""
@kernel function add_kernel!(c, @Const(a), @Const(b))
    i = @index(Global)
    @inbounds c[i] = a[i] + b[i]
end

"""
Element-wise subtraction kernel: c = a - b
"""
@kernel function sub_kernel!(c, @Const(a), @Const(b))
    i = @index(Global)
    @inbounds c[i] = a[i] - b[i]
end

"""
Element-wise multiplication kernel: c = a * b
"""
@kernel function mul_kernel!(c, @Const(a), @Const(b))
    i = @index(Global)
    @inbounds c[i] = a[i] * b[i]
end

"""
Scalar multiplication kernel: c = α * a
"""
@kernel function scale_kernel!(c, @Const(a), α)
    i = @index(Global)
    @inbounds c[i] = α * a[i]
end

"""
AXPY kernel: y = α * x + y
"""
@kernel function axpy_kernel!(y, α, @Const(x))
    i = @index(Global)
    @inbounds y[i] = α * x[i] + y[i]
end

"""
Linear combination kernel: c = α * a + β * b
"""
@kernel function linear_combination_kernel!(c, α, @Const(a), β, @Const(b))
    i = @index(Global)
    @inbounds c[i] = α * a[i] + β * b[i]
end

"""
Complex multiply-accumulate kernel for nonlinear terms: c += a * b
"""
@kernel function multiply_add_kernel!(c, @Const(a), @Const(b))
    i = @index(Global)
    @inbounds c[i] = c[i] + a[i] * b[i]
end

"""
Apply spectral filter/cutoff kernel
"""
@kernel function spectral_cutoff_kernel!(data, @Const(mask))
    i = @index(Global)
    @inbounds data[i] = data[i] * mask[i]
end

# ============================================================================
# Fused Kernels for Time-Stepping and Common Operations
# ============================================================================

"""
Fused RK stage kernel: u_new = α * u_old + β * dt * rhs
Common in Runge-Kutta time stepping.
"""
@kernel function rk_stage_kernel!(u_new, @Const(u_old), @Const(rhs), dt, α, β)
    i = @index(Global)
    @inbounds u_new[i] = α * u_old[i] + β * dt * rhs[i]
end

"""
Fused AXPBY in-place kernel: y = α * y + β * x
More efficient than separate scale and add.
"""
@kernel function axpby_inplace_kernel!(y, @Const(x), α, β)
    i = @index(Global)
    @inbounds y[i] = α * y[i] + β * x[i]
end

"""
Fused multiply-add kernel: c = a * b + d
Useful for nonlinear terms with forcing.
"""
@kernel function fma_kernel!(c, @Const(a), @Const(b), @Const(d))
    i = @index(Global)
    @inbounds c[i] = a[i] * b[i] + d[i]
end

"""
Fused scale-multiply kernel: c = α * a * b
Useful for scaled nonlinear products.
"""
@kernel function scale_multiply_kernel!(c, @Const(a), @Const(b), α)
    i = @index(Global)
    @inbounds c[i] = α * a[i] * b[i]
end

"""
Fused dealias-multiply kernel: c = mask * a * b
Combines dealiasing with multiplication.
"""
@kernel function dealias_multiply_kernel!(c, @Const(a), @Const(b), @Const(mask))
    i = @index(Global)
    @inbounds c[i] = mask[i] * a[i] * b[i]
end

"""
Triple product kernel: d = a * b * c
Useful for cubic nonlinearities.
"""
@kernel function triple_product_kernel!(d, @Const(a), @Const(b), @Const(c))
    i = @index(Global)
    @inbounds d[i] = a[i] * b[i] * c[i]
end

"""
Conjugate multiply kernel: c = conj(a) * b
Useful for correlation/convolution operations.
"""
@kernel function conj_multiply_kernel!(c, @Const(a), @Const(b))
    i = @index(Global)
    @inbounds c[i] = conj(a[i]) * b[i]
end

"""
Squared magnitude kernel: c = |a|^2
Useful for energy calculations.
"""
@kernel function squared_magnitude_kernel!(c, @Const(a))
    i = @index(Global)
    @inbounds c[i] = abs2(a[i])
end

"""
Complex to real energy kernel: e = 0.5 * (|u|^2 + |v|^2)
Useful for kinetic energy in 2D.
"""
@kernel function kinetic_energy_2d_kernel!(e, @Const(u), @Const(v))
    i = @index(Global)
    @inbounds e[i] = 0.5 * (abs2(u[i]) + abs2(v[i]))
end

"""
3D kinetic energy kernel: e = 0.5 * (|u|^2 + |v|^2 + |w|^2)
"""
@kernel function kinetic_energy_3d_kernel!(e, @Const(u), @Const(v), @Const(w))
    i = @index(Global)
    @inbounds e[i] = 0.5 * (abs2(u[i]) + abs2(v[i]) + abs2(w[i]))
end

"""
Gradient magnitude squared kernel for 2D: |∇f|^2 = |∂f/∂x|^2 + |∂f/∂y|^2
"""
@kernel function grad_mag_sq_2d_kernel!(result, @Const(dfdx), @Const(dfdy))
    i = @index(Global)
    @inbounds result[i] = abs2(dfdx[i]) + abs2(dfdy[i])
end

"""
Apply viscous damping: f = f * exp(-ν * k² * dt)
Common in spectral viscosity.
"""
@kernel function viscous_damping_kernel!(f, @Const(k_sq), ν, dt)
    i = @index(Global)
    @inbounds f[i] = f[i] * exp(-ν * k_sq[i] * dt)
end

# ============================================================================
# GPU Kernel Launchers
# ============================================================================

# KernelOperation wrappers for reuse
const GPU_ADD_OP = KernelOperation(add_kernel!) do c, a, b
    length(c)
end

const GPU_SUB_OP = KernelOperation(sub_kernel!) do c, a, b
    length(c)
end

const GPU_MUL_OP = KernelOperation(mul_kernel!) do c, a, b
    length(c)
end

const GPU_SCALE_OP = KernelOperation(scale_kernel!) do c, a, _
    length(c)
end

const GPU_AXPY_OP = KernelOperation(axpy_kernel!) do y, _, _
    length(y)
end

const GPU_LINEAR_COMB_OP = KernelOperation(linear_combination_kernel!) do c, _, _, _, _
    length(c)
end


"""
    gpu_add!(c::CuArray, a::CuArray, b::CuArray)

Element-wise addition on GPU: c = a + b
"""
function gpu_add!(c::CuArray, a::CuArray, b::CuArray)
    launch!(architecture(c), add_kernel!, c, a, b; ndrange=length(c))
    return c
end

"""
    gpu_sub!(c::CuArray, a::CuArray, b::CuArray)

Element-wise subtraction on GPU: c = a - b
"""
function gpu_sub!(c::CuArray, a::CuArray, b::CuArray)
    launch!(architecture(c), sub_kernel!, c, a, b; ndrange=length(c))
    return c
end

"""
    gpu_mul!(c::CuArray, a::CuArray, b::CuArray)

Element-wise multiplication on GPU: c = a * b
"""
function gpu_mul!(c::CuArray, a::CuArray, b::CuArray)
    launch!(architecture(c), mul_kernel!, c, a, b; ndrange=length(c))
    return c
end

"""
    gpu_scale!(c::CuArray, a::CuArray, α::Number)

Scalar multiplication on GPU: c = α * a
"""
function gpu_scale!(c::CuArray, a::CuArray, α::Number)
    launch!(architecture(c), scale_kernel!, c, a, α; ndrange=length(c))
    return c
end

"""
    gpu_axpy!(y::CuArray, α::Number, x::CuArray)

AXPY operation on GPU: y = α * x + y
"""
function gpu_axpy!(y::CuArray, α::Number, x::CuArray)
    launch!(architecture(y), axpy_kernel!, y, α, x; ndrange=length(y))
    return y
end

"""
    gpu_linear_combination!(c::CuArray, α::Number, a::CuArray, β::Number, b::CuArray)

Linear combination on GPU: c = α * a + β * b
"""
function gpu_linear_combination!(c::CuArray, α::Number, a::CuArray, β::Number, b::CuArray)
    launch!(architecture(c), linear_combination_kernel!, c, α, a, β, b; ndrange=length(c))
    return c
end

# ============================================================================
# Fused Kernel Launchers
# ============================================================================

"""
    gpu_rk_stage!(u_new, u_old, rhs, dt, α, β)

Fused RK stage: u_new = α * u_old + β * dt * rhs
"""
function gpu_rk_stage!(u_new::CuArray, u_old::CuArray, rhs::CuArray, dt::Number, α::Number, β::Number)
    launch!(architecture(u_new), rk_stage_kernel!, u_new, u_old, rhs, dt, α, β; ndrange=length(u_new))
    return u_new
end

"""
    gpu_axpby!(y, x, α, β)

In-place AXPBY: y = α * y + β * x
"""
function gpu_axpby!(y::CuArray, x::CuArray, α::Number, β::Number)
    launch!(architecture(y), axpby_inplace_kernel!, y, x, α, β; ndrange=length(y))
    return y
end

"""
    gpu_fma!(c, a, b, d)

Fused multiply-add: c = a * b + d
"""
function gpu_fma!(c::CuArray, a::CuArray, b::CuArray, d::CuArray)
    launch!(architecture(c), fma_kernel!, c, a, b, d; ndrange=length(c))
    return c
end

"""
    gpu_scale_multiply!(c, a, b, α)

Scaled multiply: c = α * a * b
"""
function gpu_scale_multiply!(c::CuArray, a::CuArray, b::CuArray, α::Number)
    launch!(architecture(c), scale_multiply_kernel!, c, a, b, α; ndrange=length(c))
    return c
end

"""
    gpu_dealias_multiply!(c, a, b, mask)

Dealiased multiply: c = mask * a * b
"""
function gpu_dealias_multiply!(c::CuArray, a::CuArray, b::CuArray, mask::CuArray)
    launch!(architecture(c), dealias_multiply_kernel!, c, a, b, mask; ndrange=length(c))
    return c
end

"""
    gpu_triple_product!(d, a, b, c)

Triple product: d = a * b * c
"""
function gpu_triple_product!(d::CuArray, a::CuArray, b::CuArray, c::CuArray)
    launch!(architecture(d), triple_product_kernel!, d, a, b, c; ndrange=length(d))
    return d
end

"""
    gpu_conj_multiply!(c, a, b)

Conjugate multiply: c = conj(a) * b
"""
function gpu_conj_multiply!(c::CuArray, a::CuArray, b::CuArray)
    launch!(architecture(c), conj_multiply_kernel!, c, a, b; ndrange=length(c))
    return c
end

"""
    gpu_squared_magnitude!(c, a)

Squared magnitude: c = |a|²
"""
function gpu_squared_magnitude!(c::CuArray, a::CuArray)
    launch!(architecture(c), squared_magnitude_kernel!, c, a; ndrange=length(c))
    return c
end

"""
    gpu_kinetic_energy_2d!(e, u, v)

2D kinetic energy: e = 0.5 * (|u|² + |v|²)
"""
function gpu_kinetic_energy_2d!(e::CuArray, u::CuArray, v::CuArray)
    launch!(architecture(e), kinetic_energy_2d_kernel!, e, u, v; ndrange=length(e))
    return e
end

"""
    gpu_kinetic_energy_3d!(e, u, v, w)

3D kinetic energy: e = 0.5 * (|u|² + |v|² + |w|²)
"""
function gpu_kinetic_energy_3d!(e::CuArray, u::CuArray, v::CuArray, w::CuArray)
    launch!(architecture(e), kinetic_energy_3d_kernel!, e, u, v, w; ndrange=length(e))
    return e
end

"""
    gpu_grad_mag_sq_2d!(result, dfdx, dfdy)

2D gradient magnitude squared: |∇f|² = |∂f/∂x|² + |∂f/∂y|²
"""
function gpu_grad_mag_sq_2d!(result::CuArray, dfdx::CuArray, dfdy::CuArray)
    launch!(architecture(result), grad_mag_sq_2d_kernel!, result, dfdx, dfdy; ndrange=length(result))
    return result
end

"""
    gpu_viscous_damping!(f, k_sq, ν, dt)

Apply viscous damping: f = f * exp(-ν * k² * dt)
"""
function gpu_viscous_damping!(f::CuArray, k_sq::CuArray, ν::Number, dt::Number)
    launch!(architecture(f), viscous_damping_kernel!, f, k_sq, ν, dt; ndrange=length(f))
    return f
end
