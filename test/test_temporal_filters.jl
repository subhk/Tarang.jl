"""
Tests for temporal filters (ExponentialMean, ButterworthFilter, LagrangianFilter)
"""

using Test
using Tarang
using LinearAlgebra

@testset "Temporal Filters" begin

    @testset "ExponentialMean Filter" begin
        # Test 1D filter
        @testset "1D ExponentialMean" begin
            N = 64
            α = 0.5
            filter = ExponentialMean((N,); α=α)

            # Test initialization
            @test size(get_mean(filter)) == (N,)
            @test all(get_mean(filter) .== 0.0)
            @test effective_averaging_time(filter) == 1/α

            # Test filtering of constant signal
            h = ones(N)
            dt = 0.01
            for _ in 1:1000
                update!(filter, h, dt)
            end
            # Should converge to h for constant input
            @test all(abs.(get_mean(filter) .- 1.0) .< 0.01)

            # Test reset
            reset!(filter)
            @test all(get_mean(filter) .== 0.0)
        end

        # Test 2D filter
        @testset "2D ExponentialMean" begin
            Nx, Ny = 32, 32
            α = 1.0
            filter = ExponentialMean((Nx, Ny); α=α)

            @test size(get_mean(filter)) == (Nx, Ny)

            # Test with oscillating signal
            # h(t) = sin(ωt) should be filtered to near zero for ω >> α
            ω = 20.0  # High frequency
            dt = 0.001
            t = 0.0

            for _ in 1:5000
                h = sin(ω * t) .* ones(Nx, Ny)
                update!(filter, h, dt)
                t += dt
            end

            # Mean should be small for high-frequency signal
            mean_field = get_mean(filter)
            @test maximum(abs.(mean_field)) < 0.2
        end

        # Test frequency response
        @testset "Frequency Response" begin
            filter = ExponentialMean((10,); α=1.0)

            # At ω=0, response should be 1
            @test filter_response(filter, 0.0) ≈ 1.0

            # At ω=α, response should be 0.5
            @test filter_response(filter, 1.0) ≈ 0.5

            # At ω >> α, response should be small
            @test filter_response(filter, 10.0) < 0.01
        end
    end

    @testset "ButterworthFilter" begin
        @testset "1D ButterworthFilter" begin
            N = 64
            α = 0.5
            filter = ButterworthFilter((N,); α=α)

            # Test initialization
            @test size(get_mean(filter)) == (N,)
            @test size(get_auxiliary(filter)) == (N,)
            @test all(get_mean(filter) .== 0.0)
            @test all(get_auxiliary(filter) .== 0.0)

            # Test filtering of constant signal
            h = ones(N)
            dt = 0.01
            for _ in 1:2000  # Needs more iterations due to 2nd order
                update!(filter, h, dt)
            end
            # Should converge to h for constant input
            @test all(abs.(get_mean(filter) .- 1.0) .< 0.05)
        end

        @testset "2D ButterworthFilter" begin
            Nx, Ny = 32, 32
            α = 1.0
            filter = ButterworthFilter((Nx, Ny); α=α)

            # Test with high-frequency oscillation
            ω = 20.0
            dt = 0.001
            t = 0.0

            for _ in 1:5000
                h = sin(ω * t) .* ones(Nx, Ny)
                update!(filter, h, dt)
                t += dt
            end

            # Butterworth should filter better than exponential
            mean_field = get_mean(filter)
            @test maximum(abs.(mean_field)) < 0.1  # Tighter bound
        end

        @testset "Frequency Response Comparison" begin
            α = 1.0
            exp_filter = ExponentialMean((10,); α=α)
            but_filter = ButterworthFilter((10,); α=α)

            # Both should pass DC
            @test filter_response(exp_filter, 0.0) ≈ 1.0
            @test filter_response(but_filter, 0.0) ≈ 1.0

            # Butterworth should have steeper rolloff at high frequencies
            ω_high = 10.0
            exp_response = filter_response(exp_filter, ω_high)
            but_response = filter_response(but_filter, ω_high)

            # Butterworth rolls off as (α/ω)⁴, exponential as (α/ω)²
            @test but_response < exp_response
            @test but_response < 0.0001  # Very small at ω >> α
        end
    end

    @testset "LagrangianFilter" begin
        @testset "Exponential LagrangianFilter" begin
            Nx, Ny = 32, 32
            α = 0.5
            filter = LagrangianFilter((Nx, Ny); α=α, filter_type=:exponential)

            # Test initialization
            @test size(get_displacement(filter)) == (Nx, Ny, 2)
            @test size(get_mean_velocity(filter)) == (Nx, Ny, 2)

            # Test with simple velocity field
            u = zeros(Nx, Ny, 2)
            u[:, :, 1] .= 1.0  # Constant u-velocity
            u[:, :, 2] .= 0.0

            dt = 0.01
            for _ in 1:100
                update_displacement!(filter, u, dt)
            end

            # Mean velocity should approach u for constant velocity
            ū = get_mean_velocity(filter)
            @test size(ū) == (Nx, Ny, 2)
        end

        @testset "Butterworth LagrangianFilter" begin
            Nx, Ny = 32, 32
            α = 0.5
            filter = LagrangianFilter((Nx, Ny); α=α, filter_type=:butterworth)

            # Test initialization
            @test size(get_displacement(filter)) == (Nx, Ny, 2)
            @test size(get_mean_velocity(filter)) == (Nx, Ny, 2)

            # Basic functionality test
            u = zeros(Nx, Ny, 2)
            dt = 0.01
            update_displacement!(filter, u, dt)

            # With zero velocity, displacement should stay near zero
            ξ = get_displacement(filter)
            @test maximum(abs.(ξ)) < 0.1
        end

        @testset "set_α!" begin
            filter = LagrangianFilter((16, 16); α=0.5, filter_type=:butterworth)
            @test filter.α ≈ 0.5

            set_α!(filter, 1.0)
            @test filter.α ≈ 1.0
        end
    end

    @testset "Filter Convergence" begin
        # Test that both filters converge to the same value for slowly varying signals
        @testset "Slow signal convergence" begin
            N = 32
            α = 2.0  # Relatively fast filter
            dt = 0.01

            exp_filter = ExponentialMean((N,); α=α)
            but_filter = ButterworthFilter((N,); α=α)

            # Slowly varying signal (ω << α)
            ω = 0.1
            t = 0.0

            for _ in 1:3000
                h = (1.0 + 0.5 * sin(ω * t)) .* ones(N)
                update!(exp_filter, h, dt)
                update!(but_filter, h, dt)
                t += dt
            end

            exp_mean = get_mean(exp_filter)
            but_mean = get_mean(but_filter)

            # Both should be similar for slow signals
            @test maximum(abs.(exp_mean .- but_mean)) < 0.2
        end
    end

    @testset "RK2 Integration" begin
        @testset "ExponentialMean RK2" begin
            N = 32
            α = 1.0
            dt = 0.1  # Larger timestep to test stability

            filter_euler = ExponentialMean((N,); α=α)
            filter_rk2 = ExponentialMean((N,); α=α)

            h = ones(N)

            for _ in 1:100
                update!(filter_euler, h, dt)
                update!(filter_rk2, h, dt, Val(:RK2))
            end

            # RK2 should also converge to 1
            @test all(abs.(get_mean(filter_rk2) .- 1.0) .< 0.01)
        end
    end
end

println("All temporal filter tests passed!")
