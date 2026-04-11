# field_pool.jl — Pre-allocated ScalarField recycling pool
#
# FieldPool eliminates per-timestep memory allocations by maintaining a
# keyed dictionary of available (returned) ScalarField objects that can
# be checked out and reused instead of allocating new ones.

export FieldPool, checkout!, return!, with_pool_field, prewarm!,
       get_field_pool, set_field_pool!, checkout_or_alloc, maybe_return!

# ---------------------------------------------------------------------------
# PoolKey — identifies a "shape class" of ScalarField
# ---------------------------------------------------------------------------

"""
    PoolKey(bases_hash, dtype)

Key used to bucket compatible `ScalarField` objects in a `FieldPool`.
Two fields are compatible if they have the same basis configuration (hashed)
and the same element dtype.
"""
struct PoolKey
    bases_hash::UInt64
    dtype::DataType
end

"""
    PoolKey(bases, dtype)

Construct a `PoolKey` from a tuple of bases and a data type.
"""
function PoolKey(bases::Tuple{Vararg{Basis}}, dtype::DataType)
    h = hash(bases)
    PoolKey(h, dtype)
end

# ---------------------------------------------------------------------------
# FieldPool
# ---------------------------------------------------------------------------

"""
    FieldPool(dist; max_per_key=16)

A pool of pre-allocated `ScalarField` objects keyed by `(bases_hash, dtype)`.

Fields are checked out with `checkout!` and returned with `return!`. When a
field is checked out from the pool its `_from_pool` flag is `true` and its
`_pool_generation` counter is incremented each time it is returned and reused.

# Fields
- `available`   — Dict mapping `PoolKey` to a stack (Vector) of idle fields
- `dist`        — The `Distributor` used when allocating new fields
- `in_use`      — Current number of fields that have been checked out
- `max_per_key` — Maximum fields retained per key when returning (extras are GC'd)
"""
mutable struct FieldPool
    available::Dict{PoolKey, Vector{ScalarField}}
    dist::Distributor
    in_use::Int
    max_per_key::Int

    function FieldPool(dist::Distributor; max_per_key::Int=16)
        new(Dict{PoolKey, Vector{ScalarField}}(), dist, 0, max_per_key)
    end
end

# ---------------------------------------------------------------------------
# checkout!
# ---------------------------------------------------------------------------

"""
    checkout!(pool::FieldPool, bases, dtype) -> ScalarField

Return a `ScalarField` compatible with `bases` and `dtype`.

If a recycled field is available it is popped from the pool; otherwise a new
one is allocated.  The field's `_from_pool` flag is set to `true` and its
`_pool_generation` is incremented.  The pool's `in_use` counter is also
incremented.
"""
function checkout!(pool::FieldPool,
                   bases::Tuple{Vararg{Basis}},
                   dtype::DataType=Float64)
    key = PoolKey(bases, dtype)
    stack = get!(Vector{ScalarField}, pool.available, key)

    field = if isempty(stack)
        # Allocate a fresh field — _from_pool starts false, _pool_generation=0
        ScalarField(pool.dist, "pool_field", bases, dtype)
    else
        f = pop!(stack)
        # Zero-fill recycled field to prevent stale data from previous use
        grid_data = get_grid_data(f)
        if grid_data !== nothing
            fill!(grid_data, zero(eltype(grid_data)))
        end
        coeff_data = get_coeff_data(f)
        if coeff_data !== nothing
            fill!(coeff_data, zero(eltype(coeff_data)))
        end
        f
    end

    field._from_pool = true
    field._pool_generation += 1
    pool.in_use += 1
    return field
end

# ---------------------------------------------------------------------------
# return!
# ---------------------------------------------------------------------------

"""
    return!(pool::FieldPool, field::ScalarField)

Return `field` to the pool so it can be reused by a future `checkout!` call.

The field's `current_layout` is reset to `:g` before it is stashed.

Throws `ArgumentError` if the field was not originally obtained from this pool
(i.e. `field._from_pool` is `false`).

If the per-key stack already holds `pool.max_per_key` fields the returned field
is simply dropped (allowing the GC to collect it).
"""
function return!(pool::FieldPool, field::ScalarField)
    if !field._from_pool
        throw(ArgumentError(
            "Cannot return a ScalarField to the pool that was not originally " *
            "checked out from a pool (field._from_pool == false)."))
    end

    # Reset layout so the next user always starts from a known state
    field.current_layout = :g

    key = PoolKey(field.bases, field.dtype)
    stack = get!(Vector{ScalarField}, pool.available, key)

    if length(stack) < pool.max_per_key
        push!(stack, field)
    end
    # else: drop the field and let GC collect it

    pool.in_use = max(0, pool.in_use - 1)
    return nothing
end

# ---------------------------------------------------------------------------
# with_pool_field
# ---------------------------------------------------------------------------

"""
    with_pool_field(f, pool::FieldPool, bases, dtype)

Check out a field, call `f(field)`, then automatically return the field to
the pool via a `try/finally` block.  The return value of `f` is forwarded.

# Example
```julia
result = with_pool_field(pool, bases, Float64) do tmp
    tmp .= some_computation()
    sum(tmp)
end
```
"""
function with_pool_field(f, pool::FieldPool,
                         bases::Tuple{Vararg{Basis}},
                         dtype::DataType=Float64)
    field = checkout!(pool, bases, dtype)
    try
        return f(field)
    finally
        return!(pool, field)
    end
end

# ---------------------------------------------------------------------------
# prewarm!
# ---------------------------------------------------------------------------

"""
    prewarm!(pool::FieldPool, bases, dtype, count::Int)

Pre-allocate `count` `ScalarField` objects and deposit them into the pool so
that the first `count` `checkout!` calls for this key incur no allocation.
"""
function prewarm!(pool::FieldPool,
                  bases::Tuple{Vararg{Basis}},
                  dtype::DataType,
                  count::Int)
    key = PoolKey(bases, dtype)
    stack = get!(Vector{ScalarField}, pool.available, key)
    for _ in 1:count
        field = ScalarField(pool.dist, "pool_field", bases, dtype)
        field._from_pool = true
        # _pool_generation stays 0 — it will be incremented on first checkout
        push!(stack, field)
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Global pool access
# ---------------------------------------------------------------------------

"""Global `FieldPool` reference.  `nothing` means no pool is active."""
const _FIELD_POOL = Ref{Union{Nothing, FieldPool}}(nothing)

"""
    get_field_pool() -> Union{Nothing, FieldPool}

Return the currently active global `FieldPool`, or `nothing` if none has been
set.
"""
get_field_pool() = _FIELD_POOL[]

"""
    set_field_pool!(pool::Union{Nothing, FieldPool})

Install `pool` as the active global `FieldPool`.  Pass `nothing` to disable
pooling.
"""
set_field_pool!(pool::Union{Nothing, FieldPool}) = (_FIELD_POOL[] = pool; nothing)

# ---------------------------------------------------------------------------
# Convenience helpers for callers that may or may not have a pool
# ---------------------------------------------------------------------------

"""
    checkout_or_alloc(bases, dtype, dist) -> ScalarField

If a global pool is active, check out a field from it.  Otherwise allocate a
fresh `ScalarField` directly (with `_from_pool = false`).
"""
function checkout_or_alloc(bases::Tuple{Vararg{Basis}},
                           dtype::DataType,
                           dist::Distributor)
    pool = get_field_pool()
    if pool !== nothing
        return checkout!(pool, bases, dtype)
    else
        return ScalarField(dist, "tmp_field", bases, dtype)
    end
end

"""
    maybe_return!(field::ScalarField)

If `field` was obtained from a pool (`field._from_pool == true`) and a global
pool is active, return it to the pool.  Otherwise this is a no-op.
"""
function maybe_return!(field::ScalarField)
    pool = get_field_pool()
    if pool !== nothing && field._from_pool
        return!(pool, field)
    end
    return nothing
end
