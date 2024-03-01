# Often times, the function arguments are denoted with a leading 
# underscore.
# This is because we work with views into those arrays
# within the functions and use the symbols there.

"""
    initial_qr_for_cholesky_test!(
        Φ, Π, Q, R, qr_ws,
        X;
        kernel, poly_deg, ε, n_X,
        dim_X = size(X, 1),
        dim_π = size(R, 2)
    )

Initialize the RBF matrix `Φ` and the polynomial 
matrix `Π` for the subsequent routine and
compute the QR decomposition of `Π`.

`X[:, 1:n_X]` contains the interpolation sites as columns.

`Φ[1:n_X, 1:n_X]` will hold the RBF basis values.
Column `j` holds the values for center `X[:, j]`, evaluated
at all columns in `X[:, 1:n_X]`.

Likewise, `Π[1:n_X, 1:dim_π]` will hold the polynomial basis
values in its columns.
Column `j` holds the values for the `j`-th basis monomial,
evaluated at all columns in `X[:, 1:n_X]`.

`Q[1:n_X, 1:n_X]` is the full Q factor and 
`R[1:dim_π, 1:dim_π]` is the truncated R factor.
"""
function initial_qr_for_cholesky_test!(
    ## modified
    _Φ :: AbstractMatrix,
    #src _Π :: AbstractMatrix,
    _Q :: AbstractMatrix,
    _R :: AbstractMatrix,
    qr_ws,
    _Qj :: AbstractMatrix,  
    # _Qj missused as a buffer here. needs at least size `min_points` × `dim_π`
    ## not modified
    _X :: AbstractMatrix;
    kernel, poly_deg, ε, n_X, 
    dim_π,
    dim_x = size(_X, 1),
    #src dim_π = size(_Π, 2)
)
    @assert size(_X, 1) == dim_x

    @assert size(_Φ, 1) >= n_X
    @assert size(_Φ, 2) >= n_X

    @assert size(_Q, 1) >= n_X
    @assert size(_Q, 2) >= n_X

    @assert size(_R, 1) >= dim_π
    @assert size(_R, 2) >= dim_π
    
    #src @assert size(_Π, 1) >= n_X
    #src @assert size(_Π, 2) >= dim_π
    @assert size(_Qj, 1) >= n_X
    @assert size(_Qj, 2) >= dim_π

    Φ = @view(_Φ[1:n_X, 1:n_X])
    X = @view(_X[1:dim_x, 1:n_X])
    Π = @view(_Qj[1:n_X, 1:dim_π])
    R = @view(_R[1:dim_π, 1:dim_π])
    Q = @view(_Q[1:n_X, 1:n_X])

    _rbf_kernel_mat!(
        Φ, kernel, X, X, ε; 
        centers_eq_features=true
    )
    _rbf_poly_mat!(Π, poly_deg, X)

    qr!(Q, R, Π, qr_ws, Π)  # the last argument is also modified in place, but we don't need Π anymore so that is okay
    return nothing
end

"""
    initial_cholesky_for_test!(
        NΦ, NΦN, L, Linv,
        Q, Φ;
        n_X, dim_π
    )

Given the initial Q factor `Q[1:n_X, 1:n_X]` of the polynomial basis matrix `Π`,
compute and set:
* `NΦ[1:dim_N, 1:n_X]` to hold the product of `N'` with `Φ[1:n_X, 1:n_X]`,
  where `N` is the columns of `Q` spanning the orthogonal complement of `Π`.
  It has `dim_n = n_X - dim_π` columns.
* `NΦN[1:dim_N, 1:din_N]` to hold the symmetric product of `NΦ[1:dim_N, 1:n_X]`
  with `N[1:n_X, 1:dim_N]`.
* `L[1:dim_N, 1:dim_N]` to hold the lower triangular Cholesky factor of `NΦ`.
* Its inverse in `Linv[1:dim_N, 1:dim_N]`.
"""
function initial_cholesky_for_test!(
    ## modified
    _NΦ,
    _NΦN,
    _L,
    _Linv,
    ## not modified
    _Q,
    _Φ; 
    n_X, dim_π
)

    dim_N = n_X - dim_π

    @assert size(_NΦ, 1) >= dim_N
    @assert size(_NΦ, 2) >= n_X
    
    @assert size(_NΦN, 1) >= dim_N
    @assert size(_NΦN, 2) >= dim_N

    @assert size(_L, 1) >= dim_N
    @assert size(_L, 2) >= dim_N
    
    @assert size(_Linv, 1) >= dim_N
    @assert size(_Linv, 2) >= dim_N

    N = @view(_Q[1:n_X, dim_π+1:n_X])
    NΦ = @view(_NΦ[1:dim_N, 1:n_X])
    NΦN = @view(_NΦN[1:dim_N, 1:dim_N])
    Φ = LA.Symmetric(@view(_Φ[1:n_X, 1:n_X]), :U)
    
    LA.mul!(NΦ, N', Φ)
    LA.mul!(NΦN, NΦ, N)

    # in place cholesky decomposition to get lower triangular factor
    L = @view(_L[1:dim_N, 1:dim_N])
    copyto!(L, NΦN)
    LA.cholesky!(LA.Symmetric(L, :L))
    copyto!(L, LA.LowerTriangular(L))

    # in place inverse (requires `LowerTriangular` view)
    Linv = @view(_Linv[1:dim_N, 1:dim_N])
    copyto!(Linv, LA.LowerTriangular(L))
    LA.inv!(LA.LowerTriangular(Linv))
    
    return nothing
end

"""
Return the test value for new site `xj`.

Assume `i = n_X + 1`.
* Evaluate the RBF with center `xj` and at all sites in `X[:, 1:n_X]` store results 
  in `_Φ[1:n_X, i]`.
* Add poly basis evaluation as `Rj[dim_π+1, 1:dim_π]` and modify `Rj` to eliminate this 
  row by applying Givens rotations. 
* Add an orthogonal column in `Qj[1:i, i]` and apply the inverse rotations.
* Correct `Linv` to account for the changes.
* `v1` must also be set.
"""
@views function compute_cholesky_test_value!(
    ## modified
    _Φ::AbstractMatrix,
    _Rj::AbstractMatrix, 
    _Qj::AbstractMatrix,
    _Linv::AbstractMatrix,
    _v1::AbstractVector,
    _v2::AbstractVector,
    ## not modified
    _X::AbstractMatrix,
    _R::AbstractMatrix, 
    _Q::AbstractMatrix,
    ;
    xj, kernel, poly_deg, ε, φ0, n_X,
    dim_x = size(_X, 1),
    dim_π = size(_R, 2)
)
    ## At the start of this function, we assume `_X` to hold `n_X` columns of interpolation 
    ## sites. The next point would have index `i = n_X + 1`.
    ## `_Φ` is at least `i` × `i` and the first `n_X`×`n_X` columns are valid RBF 
    ## evaluations for `X`. The last column and last row is modified in-place during
    ## this routine.
    ## `_Q[1:n_X, 1:n_X]` is the full Q factor for `Π[1:n_X, 1:dim_π]`, 
    ## `_R[1:dim_π, 1:dim_π]` is the truncated R factor.
    
    @assert size(_X, 1) == length(xj) == dim_x
    @assert size(_X, 2) >= n_X

    @assert length(_v1) >= n_X

    ## `dim_N` is the dimension of the orthogonal complement to Π without xj
    dim_N = n_X - dim_π
    @assert length(_v2) >= dim_N

    ## the full Q factor must have columns of length `n_X` and contain the basis vectors
    ## for Π in the first `dim_π` coloumns
    @assert size(_Q,1) >= n_X 
    @assert size(_Q, 2) >= dim_π

    ## the truncated R factor must have at least `dim_π` rows and at least `dim_π` cols.
    @assert size(_R, 1) >= dim_π
    @assert size(_R, 2) >= dim_π 
    
    ## the RBF matrix must be able to store 1 additional column
    i = n_X + 1
    k = dim_π + 1
    
    @assert size(_Φ, 2) >= i
    ## the next Q factor needs an additional row and must store at least `dim_π + 1` columns
    @assert size(_Qj, 1) >= i
    @assert size(_Qj, 2) >= k   # column `k` is the new orthogonal basis vector after adding xj
    
    ## likewise, the next R factor needs an additional row
    @assert size(_Rj, 2) >= dim_π
    @assert size(_Rj, 1) >= k
    ## `φj` holds values of RBFs with centers `X[:, 1], …, X[:, n_X]` and `xj`,
    ## evaluated at those same sites
    φj = _Φ[1:n_X, i]
    _rbf_kernel_mat!(vec2col(φj), kernel, _X[:, 1:n_X], vec2col(xj), ε)
    ## Φ[i, 1:n_X] .= φj   # Φ is symmetric, mirror the column into row `i`
    ## Φ[i, i] = φ0        # diagnoal values are always `φ0`

    ## augment R factor with additional row from polynomial basis evaluation
    ## 1) copy R values into top rows:
    _Rj[1:dim_π, 1:dim_π] .= _R[1:dim_π, 1:dim_π]  
    ## 2) evaluate into row `k`:
    _rbf_poly_mat!(vec2row(_Rj[k, 1:dim_π]), poly_deg, vec2col(xj))
    
    ## augment Q factor
    ## 1) copy first `dim_π` columns of previous factor:
    _Qj[1:n_X, 1:dim_π] .= _Q[1:n_X, 1:dim_π]  
    ## 2) add orthogonal column at position `k`:
    _Qj[i, :] .= 0
    _Qj[:, k] .= 0
    _Qj[i, k] = 1
    
    ## correct QR factors by eliminating last row from augmented R factor.
    Rj = _Rj[1:k, 1:dim_π]
    Qj = _Qj[1:i, 1:k]
    for l=1:dim_π
        ## in column `l`, use diagonal element to turn last row to zero:
        g = first(LA.givens(_Rj[l, l], _Rj[k, l], l, k))
        LA.lmul!(g, Rj)     # left multiplication of g with Rj…
        LA.rmul!(Qj, g')    # …requires right multiplicaton of Qj with inverse g⁻¹ = g'
    end

    ## new basis vector is contained in last column of Qj
    Qg = Qj[1:end-1, end]
    g = Qj[end, end]
      
    ## instantiate views for computations
    Φ = LA.Symmetric(_Φ[1:n_X, 1:n_X], :U)
    N =_Q[1:n_X, dim_π+1:n_X]
    Linv = LA.LowerTriangular(_Linv[1:dim_N, 1:dim_N])
    v1 = _v1[1:n_X]      # temporary array, will hold Linv * N'*(Φ*Qg + g*φj)
    v2 = _v2[1:dim_N]    # temporary array

    ## v1 = Φ*Qg + g*φj
    v1 .= φj
    LA.mul!(v1, Φ, Qg, 1, g)
    ## v2 = N' * v1 = N'*(Φ*Qg + g*φj)
    LA.mul!(v2, N', v1)

    ## Note: LA.dot is non-allocating
    σj = LA.dot(Qg, Φ, Qg) + 2 * g * LA.dot(Qg, φj) + g^2 * φ0

    ## `dim_L` is always <= `n_X`
    v1 = _v1[1:dim_N]
    v1 .= v2
    LA.lmul!(Linv, v1)
    τj = σj - sum(v1.^2)
    if τj <= 0
        return 0
    else
        return sqrt(τj)
    end
end

"""
update_cholesky_buffers!(
    Φ, Q, R, L, Linv, v1, v2,
    Rj, Qj;
    n_X, τj, dim_π = size(R, 2), φ0
)

Consider points indexed with `1:n_X` fixed and accept `n_X+1`.
Update the buffers used in `compute_cholesky_test_value!` to reflect the acceptance.
"""
@views function update_cholesky_buffers!(
    _Φ,
    _Q,
    _R,
    _L,
    _Linv,
    _v1, 
    _v2,
    _Rj,
    _Qj
    ;
    n_X::Integer, 
    τj::Number, 
    dim_π ::Integer = size(R, 2), 
    φ0 :: Number
)
    i = n_X + 1
    k = dim_π + 1
    dim_N = n_X - dim_π

    ## add updated column as additional row, too
    _Φ[i, 1:n_X] .= _Φ[1:n_X, i]
    _Φ[i, i] = φ0
    
    _R[1:dim_π, 1:dim_π] .= _Rj[1:dim_π, 1:dim_π]
    
    _Q[i, 1:i] .= 0 ## very very viery important!!!
    _Q[1:i, 1:dim_π] .= _Qj[1:i, 1:dim_π]
    _Q[1:i, i] .= _Qj[1:i, k]
    
    #Linv = LA.LowerTriangular(@view(_Linv[1:dim_N, 1:dim_N]))
    Linv = LA.LowerTriangular(_Linv[1:dim_N, 1:dim_N])
    v1 = _v1[1:dim_N]
    ## make `v1` hold `Linv * v2`, needed to update both Linv & L
    ## LA.mul!(v1, Linv, v2) # done in `compute_cholesky_test_value!`

    _L[1:dim_N, dim_N + 1] .= 0
    _L[dim_N + 1, 1:dim_N] .= v1
    _L[dim_N + 1, dim_N + 1 ] = τj

    LA.lmul!(Linv', v1)
    v1 .*= -1/τj
    _Linv[1:dim_N, dim_N + 1] .= 0
    _Linv[dim_N + 1, 1:dim_N] .= v1
    _Linv[dim_N + 1, dim_N + 1 ] = 1/τj


    return nothing
end

"""
    set_coefficients!(
        coeff_φ, coeff_π, FX, Linv, 
        Φ, Q, R, L;
        n_X = size(coeff_φ, 1),
        dim_y = size(coeff_φ, 2),
        dim_π = size(coeff_π, 1)
    )

Assuming valid indices `1:n_X`, compute the model coefficients and store them in 
`coeff_φ[1:n_X, 1:dim_y]` and `coeff_π[1:dim_π, 1:dim_y]`.
The columns of `FX` hold the right-hand-side values and `FX` is modified to act as a buffer.
`Linv` is also used as a buffer.
Because `FX` is modified, this routine should be called only once!
"""
function set_coefficients!(
    _coeff_φ,
    _coeff_π,
    _FX,
    _Linv,
    _Φ,
    _Q, 
    _R,
    _L,
    ;
    n_X = size(_coeff_φ, 1),
    dim_y = size(_coeff_φ, 2),
    dim_π = size(_coeff_π, 1)
)
    dim_N = n_X - dim_π
    @assert size(_Linv, 1) >= dim_N
    @assert size(_Linv, 2) >= dim_y

    Q = @view(_Q[1:n_X, 1:dim_π])
    R = LA.UpperTriangular(@view(_R[1:dim_π, 1:dim_π]))
    N = @view(_Q[1:n_X, dim_π + 1 : dim_π + dim_N])
    L = LA.LowerTriangular(@view(_L[1:dim_N, 1:dim_N])) 
    NΦN = LA.Cholesky(L) 
    FX = @view(_FX[1:dim_y, 1:n_X])

    ## RBF
    ## 1) solve NΦN * w = N' * FX'
    ## 2) set coeff_φ = N * w
    coeff_φ = @view(_coeff_φ[1:n_X, 1:dim_y])
    rhs_φ = @view(_Linv[1:dim_N, 1:dim_y])      # dim_N × dim_y
    LA.mul!(rhs_φ', FX, N)                      
    w = @view(_Linv[1:dim_N, 1:dim_y])
    LA.ldiv!(w, NΦN, rhs_φ)
    LA.mul!(coeff_φ, N, w)

    ## Polynomial
    ## RHS is Q'(FX' - Φ*coeff_φ)
    coeff_π = @view(_coeff_π[1:dim_π, 1:dim_y])
    Φ = @view(_Φ[1:n_X, 1:n_X])
    LA.mul!(FX', Φ, coeff_φ, -1, 1)     
    ## FX' is n_X × dim_y, Q' * FX' has dim_π × dim_y
    rhs_π = @view(_Linv[1:dim_π, 1:dim_y])
    LA.mul!(rhs_π, Q', FX')
    LA.ldiv!(coeff_π, R, rhs_π)

    return nothing
end