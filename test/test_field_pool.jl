using Test
using Tarang

@testset "FieldPool" begin

    # -------------------------------------------------------------------
    # Shared setup: a small 2-D periodic domain
    # -------------------------------------------------------------------
    domain = PeriodicDomain(8, 8)
    dist   = domain.dist
    bases  = domain.bases

    # -------------------------------------------------------------------
    # 1. checkout! allocates on first call; field metadata is correct
    # -------------------------------------------------------------------
    @testset "checkout! — first call allocates, sets pool metadata" begin
        pool  = FieldPool(dist)
        field = checkout!(pool, bases, Float64)

        @test field isa ScalarField
        @test field._from_pool        == true
        @test field._pool_generation  == 1
        @test pool.in_use             == 1

        return!(pool, field)
    end

    # -------------------------------------------------------------------
    # 2. return! + re-checkout reuses the same underlying memory
    # -------------------------------------------------------------------
    @testset "return! + checkout! reuses same memory" begin
        pool  = FieldPool(dist)
        f1    = checkout!(pool, bases, Float64)

        # Capture a pointer into the underlying data array to verify identity
        data_ptr = pointer(Tarang.get_grid_data(f1))

        return!(pool, f1)
        @test pool.in_use == 0

        f2 = checkout!(pool, bases, Float64)
        @test pointer(Tarang.get_grid_data(f2)) === data_ptr  # same memory
        @test f2._pool_generation == 2                         # counter incremented

        return!(pool, f2)
    end

    # -------------------------------------------------------------------
    # 3. return! resets current_layout to :g
    # -------------------------------------------------------------------
    @testset "return! resets layout to :g" begin
        pool  = FieldPool(dist)
        field = checkout!(pool, bases, Float64)
        field.current_layout = :c          # simulate coefficient-space usage
        return!(pool, field)

        field2 = checkout!(pool, bases, Float64)
        @test field2.current_layout == :g

        return!(pool, field2)
    end

    # -------------------------------------------------------------------
    # 4. Non-pool fields are rejected by return!
    # -------------------------------------------------------------------
    @testset "return! rejects non-pool fields with ArgumentError" begin
        pool        = FieldPool(dist)
        plain_field = ScalarField(dist, "plain", bases, Float64)

        @test plain_field._from_pool == false
        @test_throws ArgumentError return!(pool, plain_field)
    end

    # -------------------------------------------------------------------
    # 5. with_pool_field auto-returns the field
    # -------------------------------------------------------------------
    @testset "with_pool_field auto-returns (in_use == 0 after)" begin
        pool = FieldPool(dist)

        result = with_pool_field(pool, bases, Float64) do tmp
            @test pool.in_use == 1
            @test tmp._from_pool == true
            42  # return value forwarded
        end

        @test result      == 42
        @test pool.in_use == 0
    end

    # -------------------------------------------------------------------
    # 6. with_pool_field returns the field even when the body throws
    # -------------------------------------------------------------------
    @testset "with_pool_field returns on exception" begin
        pool = FieldPool(dist)

        @test_throws ErrorException begin
            with_pool_field(pool, bases, Float64) do _
                error("deliberate test error")
            end
        end

        @test pool.in_use == 0   # field must be back in pool despite the throw
    end

    # -------------------------------------------------------------------
    # 7. prewarm! pre-populates the pool
    # -------------------------------------------------------------------
    @testset "prewarm! fills pool without in_use side-effects" begin
        pool = FieldPool(dist)
        prewarm!(pool, bases, Float64, 3)

        key   = Tarang.PoolKey(bases, Float64)
        stack = pool.available[key]

        @test length(stack) == 3
        @test pool.in_use   == 0          # prewarm does not count as checked-out

        # All pre-warmed fields should have _from_pool == true
        for f in stack
            @test f._from_pool == true
        end

        # checkout from pre-warmed pool should NOT allocate (trivially: just test it works)
        f = checkout!(pool, bases, Float64)
        @test f._from_pool         == true
        @test f._pool_generation   == 1   # incremented on checkout
        @test length(stack)        == 2   # one was popped
        return!(pool, f)
    end

    # -------------------------------------------------------------------
    # 8. Global pool access: set / get / clear
    # -------------------------------------------------------------------
    @testset "Global pool: set_field_pool! / get_field_pool!" begin
        # Start clean
        set_field_pool!(nothing)
        @test get_field_pool() === nothing

        pool = FieldPool(dist)
        set_field_pool!(pool)
        @test get_field_pool() === pool

        # Reset global to nothing so other tests are unaffected
        set_field_pool!(nothing)
        @test get_field_pool() === nothing
    end

    # -------------------------------------------------------------------
    # 9. checkout_or_alloc — uses pool when active, allocates otherwise
    # -------------------------------------------------------------------
    @testset "checkout_or_alloc with and without active pool" begin
        # Without pool: allocates a plain (non-pool) ScalarField
        set_field_pool!(nothing)
        f_no_pool = checkout_or_alloc(bases, Float64, dist)
        @test f_no_pool isa ScalarField
        @test f_no_pool._from_pool == false

        # With pool: returns a pool field
        pool = FieldPool(dist)
        set_field_pool!(pool)
        f_pool = checkout_or_alloc(bases, Float64, dist)
        @test f_pool._from_pool == true
        @test pool.in_use       == 1

        return!(pool, f_pool)
        set_field_pool!(nothing)
    end

    # -------------------------------------------------------------------
    # 10. maybe_return! — no-op for non-pool fields; returns pool fields
    # -------------------------------------------------------------------
    @testset "maybe_return! behaviour" begin
        pool = FieldPool(dist)
        set_field_pool!(pool)

        # Non-pool field: no-op
        plain = ScalarField(dist, "plain", bases, Float64)
        @test plain._from_pool == false
        maybe_return!(plain)          # must not throw
        @test pool.in_use == 0

        # Pool field: should be returned
        pf = checkout!(pool, bases, Float64)
        @test pool.in_use == 1
        maybe_return!(pf)
        @test pool.in_use == 0

        set_field_pool!(nothing)
    end

    # -------------------------------------------------------------------
    # 11. max_per_key cap — excess returned fields are dropped
    # -------------------------------------------------------------------
    @testset "max_per_key cap: excess fields are dropped" begin
        pool = FieldPool(dist; max_per_key=2)
        prewarm!(pool, bases, Float64, 2)

        # Checkout 3 fields (all from allocations or prewarm)
        f1 = checkout!(pool, bases, Float64)
        f2 = checkout!(pool, bases, Float64)
        f3 = checkout!(pool, bases, Float64)  # allocates new one

        @test pool.in_use == 3

        # Return all three — only 2 should fit back in the stack
        return!(pool, f1)
        return!(pool, f2)
        return!(pool, f3)   # should be silently dropped

        key   = Tarang.PoolKey(bases, Float64)
        stack = pool.available[key]
        @test length(stack) == 2
        @test pool.in_use   == 0
    end

    # -------------------------------------------------------------------
    # 12. Zero-allocation integration: step! does not allocate after warmup
    # -------------------------------------------------------------------
    @testset "Zero allocation after warmup" begin
        domain = PeriodicDomain(8)

        u = ScalarField(domain, "u")
        set!(u, (x,) -> sin(x))

        # Create a simple IVP: du/dt = 0 (trivial RHS)
        problem = IVP([u])
        add_equation!(problem, "∂t(u) = 0")
        solver = InitialValueSolver(problem, RK111(); dt=0.01)

        # Warmup steps — pool fills up
        step!(solver, 0.01)
        step!(solver, 0.01)

        # Measure allocation on subsequent steps
        alloc = @allocated for _ in 1:5
            step!(solver, 0.01)
        end

        # Allow overhead for GC bookkeeping, transforms, and runtime internals.
        # The point is no MB-scale field allocations, not absolute zero.
        @test alloc < 131072  # less than 128 KB for 5 steps
    end

end
