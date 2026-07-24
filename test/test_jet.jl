using JET
using Tarang

@testset "JET.jl" begin
    # Use report_package for informational analysis.
    # test_package is too strict for codebases with optional GPU/MPI paths
    # and Union{Nothing, T} fields (CUDA, NCCL, MPI.Ialltoallv! etc.).
    result = JET.report_package(Tarang; target_modules=(Tarang,))
    n_reports = length(JET.get_reports(result))
    @info "JET.report_package(Tarang): $n_reports reports"

    # Regression ceiling. The package carries a known backlog of JET reports
    # (mostly optional GPU/MPI `Union{Nothing,T}` paths and deep library
    # error-path noise) — ~911 as of 2026-06. A bare `@test result isa Any`
    # never noticed new instabilities; this ceiling catches a regression that
    # meaningfully grows the backlog without forcing it to zero. Lower it as the
    # backlog is paid down. JET 0.11.6 reports 958 on Julia 1.12; exact counts
    # shift slightly across compiler/JET versions, hence the small headroom.
    @test n_reports <= 975
end
