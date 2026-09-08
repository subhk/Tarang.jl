using Test, Tarang, LinearAlgebra, SparseArrays, Random

# Hold every task inside its solve before allowing the RHS reads to continue.
# Subsequent reads also yield, so a workspace must belong to the active solve,
# even when another task runs on the same thread.
struct MatSolverYieldingRHS{T} <: AbstractVector{T}
    values::Vector{T}
    entered::Threads.Atomic{Int}
    release::Channel{Nothing}
end
Base.size(rhs::MatSolverYieldingRHS) = size(rhs.values)
function Base.getindex(rhs::MatSolverYieldingRHS, i::Int)
    if i == 1
        Threads.atomic_add!(rhs.entered, 1)
        take!(rhs.release)
    end
    yield()
    return rhs.values[i]
end

struct MatSolverThrowingRHS{T} <: AbstractVector{T}
    values::Vector{T}
end
Base.size(rhs::MatSolverThrowingRHS) = size(rhs.values)
function Base.getindex(rhs::MatSolverThrowingRHS, i::Int)
    i == 3 && error("intentional RHS read failure")
    return rhs.values[i]
end

function matsolver_concurrency_cases()
    rng = MersenneTwister(91)
    block_matrix = zeros(ComplexF64, 12, 12)
    for k in 0:2
        indices = (4k+1):(4k+4)
        block_matrix[indices, indices] .= randn(rng, ComplexF64, 4, 4) + 10I
    end
    qr_matrix = sprand(rng, ComplexF64, 15, 12, 0.4)
    qr_matrix[1:12, 1:12] += 10I
    return (
        ("BlockDiagonalSolver", Tarang.MatSolvers.BlockDiagonalSolver(
            block_matrix; block_sizes=[4, 4, 4]), block_matrix),
        ("SPQRSolver", Tarang.MatSolvers.SPQRSolver(qr_matrix), qr_matrix),
    )
end

@testset "CPU matrix solvers share factors with independent workspaces" begin
    for (name, solver, A) in matsolver_concurrency_cases()
        @testset "$name" begin
            truth = ComplexF64.(sin.(1:size(A, 2)), cos.(1:size(A, 2)))
            rhs = A*truth
            dest = similar(truth)

            @testset "all default-pool thread IDs" begin
                results = Vector{Any}(undef, Threads.nthreads(:default))
                Threads.@threads :static for job in eachindex(results)
                    try
                        worst = 0.0
                        for iteration in 1:100
                            expected = truth .* (job + iteration/100 + iteration/500*im)
                            actual = similar(expected)
                            Tarang.MatSolvers.solve!(actual, solver, A*expected)
                            worst = max(worst, norm(actual-expected)/norm(expected))
                        end
                        results[job] = worst
                    catch err
                        results[job] = err
                    end
                end
                for result in results
                    @test result isa Float64
                    result isa Float64 && @test result < 2e-12
                end
            end

            @testset "oversubscribed tasks yield during RHS reads" begin
                jobs = 2Threads.nthreads(:default) + 3
                entered = Threads.Atomic{Int}(0)
                release = Channel{Nothing}(jobs)
                expected = [truth .* (job + (job/7)*im) for job in 1:jobs]
                tasks = map(1:jobs) do job
                    @async begin
                        actual = similar(truth)
                        yielding_rhs = MatSolverYieldingRHS(A*expected[job], entered, release)
                        try
                            Tarang.MatSolvers.solve!(actual, solver, yielding_rhs)
                            actual
                        catch err
                            err
                        end
                    end
                end
                # A whole-solve lock would prevent all tasks entering. Release
                # them even on failure so this assertion cannot strand tasks.
                all_entered = timedwait(() -> entered[] == jobs, 10.0)
                for _ in 1:jobs
                    put!(release, nothing)
                end
                results = fetch.(tasks)
                @test all_entered === :ok
                for (actual, reference) in zip(results, expected)
                    @test actual isa Vector{ComplexF64}
                    actual isa Vector{ComplexF64} && @test actual ≈ reference atol=2e-12 rtol=2e-12
                end
            end

            @testset "exceptions return reusable workspaces" begin
                workspace_count = length(solver.workspace)
                for _ in 1:(2workspace_count+3)
                    @test_throws ErrorException Tarang.MatSolvers.solve!(
                        dest, solver, MatSolverThrowingRHS(rhs))
                    Tarang.MatSolvers.solve!(dest, solver, rhs)
                    @test dest ≈ truth atol=2e-12 rtol=2e-12
                end
                @test length(solver.workspace) == workspace_count
            end
        end
    end
end
