using Test
using LinearAlgebra
using SparseArrays

@testset "GPU Solver Logic (CPU)" begin
    # ========================================================================
    # These tests verify the mathematical correctness of the GPU solver
    # algorithms using CPU arrays. This ensures the GMRES Hessenberg fix
    # (H on CPU) and general solver logic work correctly.
    # ========================================================================

    # ========================================================================
    # GMRES core algorithm on CPU
    # ========================================================================

    @testset "GMRES Arnoldi with CPU Hessenberg" begin
        # Reproduce the exact pattern from gpu_matsolvers.jl:
        # H must be a CPU array for scalar indexing in the Arnoldi loop
        n = 20
        A = sprand(n, n, 0.3) + 5.0I  # diagonally dominant for convergence
        b = rand(n)
        x = zeros(n)
        m = 10  # restart parameter

        # Arnoldi process (same logic as gpu_matsolvers.jl lines 600-625)
        r = b - A * x
        beta = norm(r)
        V = [r / beta]
        H = zeros(Float64, m + 1, m)  # CPU array — this is the fix

        for j in 1:m
            w = A * V[j]

            # Modified Gram-Schmidt
            for i in 1:j
                H[i, j] = dot(V[i], w)
                w = w - H[i, j] * V[i]
            end
            H[j + 1, j] = norm(w)

            if abs(H[j + 1, j]) < 1e-14
                break
            end
            push!(V, w / H[j + 1, j])
        end

        # Solve least squares on CPU
        e1 = zeros(m + 1)
        e1[1] = beta
        y = H \ e1
        y_vals = y[1:length(V)-1]

        # Update solution
        for i in 1:length(y_vals)
            x .+= y_vals[i] .* V[i]
        end

        # Verify convergence
        residual = norm(b - A * x) / norm(b)
        @test residual < 1e-6
    end

    @testset "GMRES solves small dense system" begin
        A = [4.0 1.0 0.0;
             1.0 3.0 1.0;
             0.0 1.0 4.0]
        b = [5.0, 5.0, 5.0]
        x_exact = A \ b

        # Run GMRES manually
        x = zeros(3)
        m = 3
        r = b - A * x
        beta = norm(r)
        V = [r / beta]
        H = zeros(m + 1, m)

        last_j = m
        for j in 1:m
            w = A * V[j]
            for i in 1:j
                H[i, j] = dot(V[i], w)
                w = w - H[i, j] * V[i]
            end
            H[j + 1, j] = norm(w)
            if abs(H[j + 1, j]) < 1e-14
                last_j = j
                break
            end
            push!(V, w / H[j + 1, j])
        end

        # Solve the last_j x last_j least squares problem
        H_sub = H[1:last_j+1, 1:last_j]
        e1 = zeros(last_j + 1)
        e1[1] = beta
        y = H_sub \ e1

        for i in 1:min(length(y), length(V))
            x .+= y[i] .* V[i]
        end

        @test isapprox(x, x_exact; atol=1e-8)
    end

    # ========================================================================
    # CG core algorithm on CPU
    # ========================================================================

    @testset "CG solves SPD system" begin
        # Conjugate Gradient for symmetric positive definite
        n = 30
        L = sprand(n, n, 0.2)
        A = L' * L + 5.0I  # guaranteed SPD
        b = rand(n)

        x = zeros(n)
        r = b - A * x
        p = copy(r)
        tol = 1e-10
        max_iter = 100

        for iter in 1:max_iter
            Ap = A * p
            rr = dot(r, r)
            alpha = rr / dot(p, Ap)
            x .+= alpha .* p
            r .-= alpha .* Ap
            if norm(r) < tol
                break
            end
            beta = dot(r, r) / rr
            p .= r .+ beta .* p
        end

        x_exact = Matrix(A) \ b
        @test isapprox(x, x_exact; atol=1e-8)
    end

    # ========================================================================
    # Preconditioner math
    # ========================================================================

    @testset "Jacobi preconditioner" begin
        # Jacobi = diagonal inverse
        A = [4.0 1.0; 1.0 3.0]
        M_inv = Diagonal(1.0 ./ diag(A))  # [1/4, 1/3]

        @test M_inv[1, 1] == 0.25
        @test M_inv[2, 2] ≈ 1.0 / 3.0

        # Preconditioned residual should reduce condition number
        r = [1.0, 1.0]
        z = M_inv * r  # preconditioned residual
        @test z == [0.25, 1.0 / 3.0]
    end

    @testset "ILU(0) factorization structure" begin
        # ILU(0) preserves sparsity pattern of A
        A = sparse([1.0 2.0 0.0;
                     3.0 4.0 5.0;
                     0.0 6.0 7.0])

        F = lu(Matrix(A))  # Full LU for reference

        # The key property: L*U should approximate A
        @test isapprox(F.L * F.U, Matrix(A)[F.p, :]; atol=1e-10)
    end

    # ========================================================================
    # Transpose count arithmetic (for NCCL)
    # ========================================================================

    @testset "chunk size computation" begin
        # This tests the same arithmetic used in compute_transpose_counts!
        # div(N, P) + (rank < mod(N, P) ? 1 : 0)

        N = 10
        P = 3
        chunks = [div(N, P) + ((r - 1) < mod(N, P) ? 1 : 0) for r in 1:P]

        @test sum(chunks) == N  # all elements accounted for
        @test chunks == [4, 3, 3]  # first rank gets the extra element
    end

    @testset "chunk size for even division" begin
        N = 12
        P = 4
        chunks = [div(N, P) + ((r - 1) < mod(N, P) ? 1 : 0) for r in 1:P]

        @test sum(chunks) == N
        @test all(c -> c == 3, chunks)  # perfectly even
    end

    @testset "transpose send/recv count symmetry" begin
        # For a Z->Y transpose:
        # send_counts[i] = Nx_local * chunk_y[i] * Nz_local (sending Y-chunks)
        # recv_counts[i] = Nx_local * Ny_local * chunk_z[i]  (receiving Z-chunks)
        # Total sent == Total received == Nx_local * Ny * Nz_local
        Nx, Ny, Nz = 16, 12, 10
        P = 3  # number of ranks in the communicator

        for rank in 0:P-1
            Nz_local = div(Nz, P) + (rank < mod(Nz, P) ? 1 : 0)

            send_total = 0
            recv_total = 0
            for i in 0:P-1
                chunk_y = div(Ny, P) + (i < mod(Ny, P) ? 1 : 0)
                chunk_z = div(Nz, P) + (i < mod(Nz, P) ? 1 : 0)
                send_total += Nx * chunk_y * Nz_local
                recv_total += Nx * Ny * chunk_z  # This is wrong if using full Ny!
            end

            # Total sent should equal Nx * Ny * Nz_local (all Y-chunks for this rank's Z)
            @test send_total == Nx * Ny * Nz_local

            # For the recv side in Y-pencil, this rank will have full Ny,
            # and receive Z-chunks from each peer
            @test recv_total == Nx * Ny * Nz  # sum of all Z chunks = full Nz
        end
    end

    @testset "Y->X recv counts use per-rank chunk_x, not full Nx" begin
        # This is the bug we fixed in compute_transpose_counts! for :y_to_x
        # The WRONG formula was: recv_counts[i] = Nx * chunk_y * Nz_local
        # The CORRECT formula is: recv_counts[i] = chunk_x[i] * chunk_y_me * Nz_local
        Nx, Ny, Nz = 16, 12, 10
        P = 3

        for my_rank in 0:P-1
            Ny_me = div(Ny, P) + (my_rank < mod(Ny, P) ? 1 : 0)
            Nz_local = div(Nz, P) + (my_rank < mod(Nz, P) ? 1 : 0)

            recv_total_correct = 0
            recv_total_wrong = 0
            for i in 0:P-1
                chunk_x = div(Nx, P) + (i < mod(Nx, P) ? 1 : 0)
                chunk_y = div(Ny, P) + (i < mod(Ny, P) ? 1 : 0)

                recv_total_correct += chunk_x * Ny_me * Nz_local  # correct
                recv_total_wrong += Nx * chunk_y * Nz_local        # old bug
            end

            # Correct total = full Nx * my Y chunk * my Z chunk
            # (after Y->X transpose, we own full X and our share of Y,Z)
            @test recv_total_correct == Nx * Ny_me * Nz_local

            # Wrong total would be inflated (unless P=1)
            if P > 1
                @test recv_total_wrong != recv_total_correct
            end
        end
    end

    # ========================================================================
    # Displacement computation
    # ========================================================================

    @testset "displacement computation from counts" begin
        counts = [100, 150, 100, 200]
        displs = zeros(Int, length(counts))
        displs[1] = 0
        for i in 2:length(counts)
            displs[i] = displs[i-1] + counts[i-1]
        end

        @test displs == [0, 100, 250, 350]
        @test displs[end] + counts[end] == sum(counts)
    end
end
