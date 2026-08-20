"""
Non-pencil (use_pencil_arrays=false) NetCDF slab metadata: `get_local_start`
and `get_local_shape` must describe the SAME partition.

Bug (2026-08-20 MPI review, finding O3/V8): the non-pencil branch of
`get_local_start` used the balanced split `div(N*coord, P)` while
`get_local_shape` gives the remainder to the FIRST coords (matching the actual
data slab from `get_local_array_size`). For N % P != 0 the written
start/count hyperslabs overlapped and left gaps — e.g. N=10, P=4 claimed
[0,3), [2,5), [5,7), [7,9): duplicate coverage at index 2, index 9 never
written. Every GPU+MPI run takes this branch (use_pencil_arrays=false there).

The functions are duck-typed on `dist`, so a fake distributor drives the
multi-rank branch from a serial test.
"""

using Test
using Tarang

struct _FakeNPDist
    size::Int
    mesh::Tuple{Vararg{Int}}
    rank::Int
    use_pencil_arrays::Bool
end

function _np_slabs(N::Int, P::Int)
    dist = _FakeNPDist(P, (P,), 0, false)
    di = Dict{String, Any}("shape" => (N,), "dist" => dist)
    return [(Tarang.get_local_start("g", di, 1.0, r)[1],
             Tarang.get_local_shape("g", di, 1.0, r)[1]) for r in 0:P-1]
end

@testset "non-pencil start/count partition exactly" begin
    for (N, P) in [(10, 4), (9, 2), (7, 3), (8, 4), (16, 4)]
        slabs = _np_slabs(N, P)
        covered = zeros(Int, N)
        for (start, count) in slabs
            covered[start+1:start+count] .+= 1
        end
        # Pre-fix N=10,P=4: index 3 covered twice, index 10 never.
        @test all(==(1), covered)
    end
end

@testset "non-pencil 2D mesh start/count partition exactly" begin
    N1, N2, P1, P2 = 10, 6, 2, 4
    dist = _FakeNPDist(P1 * P2, (P1, P2), 0, false)
    di = Dict{String, Any}("shape" => (N1, N2), "dist" => dist)
    covered = zeros(Int, N1, N2)
    for r in 0:(P1 * P2 - 1)
        s = Tarang.get_local_start("g", di, 1.0, r)
        c = Tarang.get_local_shape("g", di, 1.0, r)
        covered[s[1]+1:s[1]+c[1], s[2]+1:s[2]+c[2]] .+= 1
    end
    @test all(==(1), covered)
end
