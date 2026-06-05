using Test
using Tarang

# ============================================================================
# Tests for src/core/field/field_data/field_data_component_buffers.jl
#
# This file provides structure-of-arrays helpers that stack the components of a
# VectorField / TensorField into a single contiguous buffer (component index as
# the leading dimension(s)) and scatter them back out again:
#
#   stack_components(vf)            -> buffer of shape (n_components, data...)
#   unstack_components!(vf, buf)    -> scatter buffer back into components
#   stack_tensor_components(tf)     -> buffer of shape (dim, dim, data...)
#   unstack_tensor_components!(tf, buf)
#
# ORACLE: the clean, code-independent oracle is the ROUND-TRIP IDENTITY. If we
# fill each component with known distinct data, stack it, zero the components,
# and unstack, every component must be byte-for-byte restored. Independently,
# the leading buffer dimension(s) must index the components, so slice i of the
# buffer must equal component i's data. Neither expectation is read off the
# function's own output.
#
# NOTE ON LAYOUTS: in :g (grid) layout a RealFourier component's data is real
# (Matrix{Float64}); in :c (coefficient) layout it is the unnormalized complex
# half-spectrum (Matrix{ComplexF64}). The buffer is allocated as `field.dtype`
# (Float64 here). The :g path is therefore an exact round-trip; the :c path
# currently has bugs that are pinned below (see the @test_broken / @test_throws
# blocks and the report).
# ============================================================================

@testset "field_data_component_buffers.jl" begin

    # Helper: fill component (1-based linear index `id`) of a grid field with a
    # distinct, position-dependent pattern so components are mutually distinct.
    pattern!(g, id) = begin
        nx, ny = size(g)
        for a in 1:nx, b in 1:ny
            g[a, b] = 1000.0 * id + 10.0 * a + 0.1 * b
        end
        g
    end

    # ------------------------------------------------------------------
    # 1. VectorField :g round-trip (the clean oracle)
    # ------------------------------------------------------------------
    @testset "stack/unstack VectorField :g round-trip" begin
        dom = PeriodicDomain(8, 8)          # 2-component coordsys (x, y)
        u = VectorField(dom, "u")
        @test length(u.components) == 2

        orig = Matrix{Float64}[]
        for (i, comp) in enumerate(u.components)
            Tarang.ensure_layout!(comp, :g)
            push!(orig, copy(pattern!(Tarang.get_grid_data(comp), i)))
        end

        buf = Tarang.stack_components(u; layout=:g)

        # SHAPE: leading dim indexes components, rest is the field's local data.
        @test ndims(buf) == 3
        @test size(buf) == (2, 8, 8)
        @test size(buf, 1) == length(u.components)
        @test eltype(buf) == Float64

        # Stacking preserves per-component values: slice i == component i data.
        @test collect(selectdim(buf, 1, 1)) == orig[1]
        @test collect(selectdim(buf, 1, 2)) == orig[2]
        @test orig[1] != orig[2]            # components really are distinct

        # buffer bookkeeping is recorded on the field
        @test u.buffer_layout == :g
        @test u.component_buffer === buf

        # Clobber the components, then scatter the buffer back.
        for comp in u.components
            Tarang.get_grid_data(comp) .= 0.0
        end
        @test all(c -> all(Tarang.get_grid_data(c) .== 0.0), u.components)

        ret = Tarang.unstack_components!(u, buf; layout=:g)
        @test ret === u                     # returns the field

        # ROUND-TRIP IDENTITY: every component is byte-for-byte restored.
        for (i, comp) in enumerate(u.components)
            @test Tarang.get_grid_data(comp) == orig[i]
        end
    end

    # ------------------------------------------------------------------
    # 2. unstack default-layout inference (uses vf.buffer_layout)
    #    After a :g stack, buffer_layout is :g, so unstack without an
    #    explicit layout must still round-trip.
    # ------------------------------------------------------------------
    @testset "unstack_components! infers layout from field" begin
        dom = PeriodicDomain(8, 8)
        u = VectorField(dom, "u")
        orig = Matrix{Float64}[]
        for (i, comp) in enumerate(u.components)
            Tarang.ensure_layout!(comp, :g)
            push!(orig, copy(pattern!(Tarang.get_grid_data(comp), i)))
        end
        buf = Tarang.stack_components(u; layout=:g)
        for comp in u.components
            Tarang.get_grid_data(comp) .= 0.0
        end
        Tarang.unstack_components!(u, buf)         # no layout kwarg
        for (i, comp) in enumerate(u.components)
            @test Tarang.get_grid_data(comp) == orig[i]
        end
    end

    # ------------------------------------------------------------------
    # 3. Buffer caching / reuse and force=true
    # ------------------------------------------------------------------
    @testset "stack_components buffer reuse and force" begin
        dom = PeriodicDomain(8, 8)
        u = VectorField(dom, "u")
        for (i, comp) in enumerate(u.components)
            Tarang.ensure_layout!(comp, :g)
            pattern!(Tarang.get_grid_data(comp), i)
        end

        buf1 = Tarang.stack_components(u; layout=:g)
        buf2 = Tarang.stack_components(u; layout=:g)
        @test buf2 === buf1                 # same shape/arch -> reuse

        buf3 = Tarang.stack_components(u; layout=:g, force=true)
        @test buf3 !== buf1                 # force -> fresh allocation
        @test buf3 == buf1                  # but identical contents
    end

    # ------------------------------------------------------------------
    # 4. VectorField validation / error paths
    # ------------------------------------------------------------------
    @testset "VectorField buffer error paths" begin
        dom = PeriodicDomain(8, 8)
        u = VectorField(dom, "u")
        for (i, comp) in enumerate(u.components)
            Tarang.ensure_layout!(comp, :g)
            pattern!(Tarang.get_grid_data(comp), i)
        end

        # unsupported layout symbol
        @test_throws ArgumentError Tarang.stack_components(u; layout=:bogus)
        @test_throws ArgumentError Tarang.unstack_components!(u, zeros(2, 8, 8); layout=:bogus)

        # component-count mismatch on unstack
        @test_throws ArgumentError Tarang.unstack_components!(u, zeros(3, 8, 8); layout=:g)

        # a fresh field has no recorded buffer_layout, so an inference-only
        # unstack must refuse rather than guess.
        fresh = VectorField(dom, "fresh")
        @test fresh.buffer_layout === nothing
        @test_throws ArgumentError Tarang.unstack_components!(fresh, zeros(2, 8, 8); layout=nothing)
    end

    # ------------------------------------------------------------------
    # 5. TensorField :g round-trip
    #
    # BUG (pinned): stack_tensor_components / unstack_tensor_components! slice
    # the buffer with selectdim(selectdim(buf, 1, i), 2, j). After the first
    # selectdim drops the leading tensor dimension, the SECOND selectdim's
    # axis 2 is a DATA axis, not the second tensor index. For a buffer of
    # shape (dim, dim, nx, ny) with nx != dim this raises a BoundsError; even
    # when it does not raise it copies the wrong slab. The correct slicing is
    # selectdim(selectdim(buf, 2, j), 1, i) (or view(buf, i, j, :, :)).
    #
    # We PIN the current broken behavior with @test_throws, and capture the
    # INTENDED round-trip identity with @test_broken so it flips to a pass the
    # moment the source is fixed.
    # ------------------------------------------------------------------
    @testset "stack/unstack TensorField :g (round-trip currently broken)" begin
        dom = PeriodicDomain(8, 8)
        S = TensorField(dom, "S")
        @test size(S.components) == (2, 2)

        torig = Dict{Tuple{Int,Int},Matrix{Float64}}()
        for i in 1:2, j in 1:2
            comp = S.components[i, j]
            Tarang.ensure_layout!(comp, :g)
            # distinct linear id per (i,j) so slices are mutually distinct
            torig[(i, j)] = copy(pattern!(Tarang.get_grid_data(comp), 10 * i + j))
        end

        # CURRENT BEHAVIOR: stacking on an 8x8 grid raises BoundsError due to
        # the selectdim mis-ordering described above.
        @test_throws BoundsError Tarang.stack_tensor_components(S; layout=:g)

        # INTENDED BEHAVIOR (round-trip identity) once the slicing is fixed.
        # This must not error here, so guard it in a try and assert via
        # @test_broken on the round-trip success flag.
        roundtrip_ok = false
        try
            buf = Tarang.stack_tensor_components(S; layout=:g)
            for i in 1:2, j in 1:2
                Tarang.get_grid_data(S.components[i, j]) .= 0.0
            end
            Tarang.unstack_tensor_components!(S, buf; layout=:g)
            roundtrip_ok = all(
                Tarang.get_grid_data(S.components[i, j]) == torig[(i, j)]
                for i in 1:2, j in 1:2
            )
        catch
            roundtrip_ok = false
        end
        @test_broken roundtrip_ok
    end

    # ------------------------------------------------------------------
    # 6. TensorField buffer dimension validation on unstack
    #    (the leading-dimension check fires before the buggy slicing).
    # ------------------------------------------------------------------
    @testset "unstack_tensor_components! validation" begin
        dom = PeriodicDomain(8, 8)
        S = TensorField(dom, "S")

        @test_throws ArgumentError Tarang.unstack_tensor_components!(S, zeros(2, 2, 8, 8); layout=:bogus)
        # wrong leading tensor dims (3x3 instead of 2x2)
        @test_throws ArgumentError Tarang.unstack_tensor_components!(S, zeros(3, 3, 8, 8); layout=:g)
        # missing layout with no recorded buffer_layout
        @test S.buffer_layout === nothing
        @test_throws ArgumentError Tarang.unstack_tensor_components!(S, zeros(2, 2, 8, 8); layout=nothing)
    end

    # ------------------------------------------------------------------
    # 7. VectorField :c (coefficient) layout
    #
    # BUG (pinned): the buffer is allocated as vf.dtype (Float64), but in :c
    # layout a RealFourier component's data is the complex half-spectrum
    # (Matrix{ComplexF64}). copyto! of a complex slice into the Float64 buffer
    # raises InexactError whenever any coefficient has a nonzero imaginary part
    # (i.e. essentially always for a non-constant field). A correct buffer would
    # be allocated as complex(vf.dtype) for the :c layout.
    # ------------------------------------------------------------------
    @testset "stack_components :c layout (currently broken for complex coeffs)" begin
        dom = PeriodicDomain(8, 8)
        u = VectorField(dom, "u")
        for (i, comp) in enumerate(u.components)
            Tarang.ensure_layout!(comp, :g)
            g = Tarang.get_grid_data(comp)
            nx, ny = size(g)
            for a in 1:nx, b in 1:ny
                # a single sine mode -> nonzero imaginary Fourier coefficient
                g[a, b] = Float64(i) + sin(2pi * (a - 1) / nx)
            end
        end
        Tarang.ensure_layout!(u, :c)
        # the complex half-spectrum genuinely carries imaginary content
        @test any(abs.(imag.(Tarang.get_coeff_data(u.components[1]))) .> 1e-8)

        # CURRENT BEHAVIOR: Float64 buffer cannot hold the complex slice.
        @test_throws InexactError Tarang.stack_components(u; layout=:c)
    end

end
