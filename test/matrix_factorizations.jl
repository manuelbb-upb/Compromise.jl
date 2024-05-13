import Compromise.RBFModels: QRWYWs, do_qr!
import LinearAlgebra as LA
using Test
import Random

Random.seed!(987654321)

@testset "Cached QR" begin
for F in (Float32, Float64, ComplexF32, ComplexF64)
for dim1 in (0, 1, 2, 3, 5, 8, 13)
for dim2 in (0, 1, 2, 3, 5, 8, 13)
dim1 == 0 && continue # remove when https://github.com/JuliaLang/julia/issues/53451 is fixed
dimmin = min(dim1, dim2)

X = rand(F, dim1, dim2)
_X = copy(X)

qr1 = LA.qr(X)
@test X == _X
@test qr1 isa LA.QRCompactWY
@test size(qr1.Q) == (dim1, dim1)
@test 0 <= size(qr1.R, 1) <= dimmin
@test prod(qr1) ≈ X

dimR1, dimR2 = size(qr1.R)

qr2 = do_qr!(nothing, X)
@test X == _X
@test qr2 isa LA.QRCompactWY
@test size(qr2.Q) == (dim1, dim1)
@test size(qr2.R) == (dimR1, dimR2)
@test prod(qr2) ≈ X

ws = QRWYWs(X)
@test ws isa QRWYWs{F, Matrix{F}}
qr3 = do_qr!(ws, X)
@test X == _X
@test qr3 isa LA.QRCompactWY
@test size(qr3.Q) == (dim1, dim1)
@test size(qr3.R) == (dimR1, dimR2)
@test prod(qr3) ≈ X

@test qr1.Q == qr2.Q == qr3.Q
@test qr1.R == qr2.R == qr3.R
end
end
end
end

import Compromise.RBFModels: find_poised_points!

@testset "Affine Sampling QR" begin
    dim_x = 10
    X = rand(dim_x, 2*dim_x)
    Ys = Y = nothing
    qr_ws = QRWYWs(X)
    QRbuff = copy(X)
    x0 = zeros(dim_x)
    Xs = Matrix{Float64}(LA.I(dim_x))
    xZ = zero(x0)
    th = 1e-3
    
    n_new, qr = find_poised_points!(X, Y, qr_ws, QRbuff, x0, Xs, Ys;)
    @test n_new == dim_x
    @test X[:, 1:dim_x] ≈ Xs[:, end:-1:1]
    _X = prod(qr)
    @test size(_X) == (dim_x, n_new-1)
    @test _X ≈ X[:, 1:n_new-1]
end
import Compromise.RBFModels: sample_along_Z!, _rbf_poly_mat!
import Compromise.RBFModels: GaussianKernel, initial_qr_for_cholesky_test!, 
    initial_cholesky_for_test!, compute_cholesky_test_value!, update_cholesky_buffers!

function test_cholesky(_Q, _NΦ, _NΦN, _Φ, _L, _Linv, n_X, dim_π; check_matprods = false)
    dim_N = n_X - dim_π
    
    N = @view(_Q[1:n_X, dim_π+1:n_X])
    Φ = LA.Symmetric(@view(_Φ[1:n_X, 1:n_X]), :U)
    L = @view(_L[1:dim_N, 1:dim_N])
    Linv = @view(_Linv[1:dim_N, 1:dim_N])
    
    if check_matprods
        NΦ = @view(_NΦ[1:dim_N, 1:n_X])
        NΦN = @view(_NΦN[1:dim_N, 1:dim_N])
        @test NΦ ≈ N' * Φ
        @test NΦN ≈ N'Φ*N
    end
    NΦN = N' * Φ * N

    @test L * Linv ≈ Linv * L
    @test L * Linv ≈ LA.I(dim_N)
    @test L * L' ≈ NΦN
end

function test_qr(_Q, _R, _X, poly_deg, dim_x, dim_π, n_X)
    X = @view(_X[1:dim_x, 1:n_X])
    Π = similar(X, n_X, dim_π)
    _rbf_poly_mat!(Π, poly_deg, X)
    R = [
        _R[1:dim_π, 1:dim_π]
        zeros(n_X-dim_π, dim_π)
    ]
    Q = @view(_Q[1:n_X, 1:n_X])
    @test Q * R ≈ Π
end
@testset "Cholesky Routine" begin 
    dim_x = 10
    X = rand(dim_x, 2*dim_x)
    Ys = Y = nothing
    qr_ws = QRWYWs(X)
    QRbuff = copy(X)
    x0 = zeros(dim_x)
    Xs = Matrix{Float64}(LA.I(dim_x))
    xZ = zero(x0)
    th = 1e-3

    X[:, 1] .= 0

    N = dim_x ÷ 2    
    n_new, qr = find_poised_points!(
        X, Y, qr_ws, QRbuff, x0, @view(Xs[:, 1:N]), Ys;
        ix1=2, ix2=1
    )
    @test N == n_new
    @test X[:, 2:N+1] ≈ Xs[:, N:-1:1]
    _X = prod(qr)
    @test size(_X) == (dim_x, N)
    @test _X ≈ X[:, 2:N+1]

    lb = fill(-1.0, dim_x)
    ub = fill(+1.0, dim_x)
    n_new, qr = sample_along_Z!(
        X, qr_ws, QRbuff, x0, lb, ub, th; qr, ix1=2, ix2=N+1)
    @test n_new == N
    for j in 2:dim_x+1
        @test mapreduce( xi -> (abs(xi) > 1e-6), +, X[:, j] ) == 1
    end

    dim_y = 5
    kernel = GaussianKernel()
    poly_deg = 1
 
    n_X = dim_x + 1
    max_points = 2*dim_x
    dim_π = n_X

    Y = rand(dim_y, 2*dim_x)
    Xs = rand(dim_x, 10 * dim_x)
    Ys = rand(dim_y, 10 * dim_x)

    qr_ws = QRWYWs(@view(X[:, 1:n_X]))
    Φ = zeros(max_points, max_points)
    Q = zeros(max_points, max_points)
    R = zeros(dim_π, dim_π)
    Qj = zeros(max_points, dim_π + 1)
    Rj = zeros(dim_π + 1, dim_π)
    v1 = zeros(max_points)
    v2 = zeros(max_points)

    ε = 1
    φ0 = 1
    initial_qr_for_cholesky_test!(Φ, Q, R, qr_ws, Qj, X; kernel, poly_deg, ε, n_X, dim_x, dim_π)
    
    test_qr(Q, R, X, poly_deg, dim_x, dim_π, n_X)

    max_dim_N = max_points - dim_π
    NΦ = zeros(max_dim_N, max_points)
    NΦN = zeros(max_dim_N, max_dim_N)
    L = zeros(max_dim_N, max_dim_N)
    Linv = zeros(max_dim_N, max_dim_N)
    initial_cholesky_for_test!(NΦ, NΦN, L, Linv, Q, Φ; n_X, dim_π)
  
    test_cholesky(Q, NΦ, NΦN, Φ, L, Linv, n_X, dim_π; check_matprods=true)
    
    j = n_X + 1
    for _xj = eachcol(Xs)
        if j > max_points 
            break
        end
        xj = @view(X[:, j])
        copyto!(xj, _xj)
        τj = compute_cholesky_test_value!(
            Φ, Rj, Qj, Linv, v1, v2, X, R, Q;
            xj, kernel, poly_deg, ε, φ0, n_X,
            dim_x, dim_π
        )
        if τj >= 1e-7
            update_cholesky_buffers!(
                Φ, Q, R, L, Linv, v1, v2, Rj, Qj;
                n_X, τj, dim_π, φ0
            )
            j += 1
            n_X += 1
            test_qr(Q, R, X, poly_deg, dim_x, dim_π, n_X)
            test_cholesky(Q, NΦ, NΦN, Φ, L, Linv, n_X, dim_π)
        end
    end
    @test n_X <= max_points
    @test n_X > dim_x + 1
end
