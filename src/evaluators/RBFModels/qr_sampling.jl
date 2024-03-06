function project_onto_Q_colspace!(x, qr, j1=1, j2=size(qr.Q, 2))
    dim_x = length(x)
    @assert size(qr.Q, 1) == dim_x

    if j1 > j2
        x .= 0
        return 
    end
    
    dim_Q = j2 - j1 + 1
    if dim_Q == dim_x
        # columns of Q span ℝ^dim_x entirely
        return
    end
   
    Q = qr.Q
    LA.lmul!(Q', x)
    x[1:j1-1] .= 0
    x[j2+1:end] .= 0
    LA.lmul!(Q, x)
    return
end

copy_col!(mat_trgt, mat_src, j_trgt, j_src)=nothing
function copy_col!(mat_trgt::AbstractMatrix, mat_src::AbstractMatrix, j_trgt, j_src)
    mat_trgt[:, j_trgt] .= mat_src[:, j_src]
    nothing
end

@views function find_poised_points!(
    X, Y, qr_ws, QRbuff, x0, Xs, Ys;
    xZ = zero(x0),
    th = 1e-3,
    ix1 = 1, 
    ix2 = 0, 
    norm_p = Inf, 
    chosen_index = nothing
    #log_level=Info
)

    dim_x = size(X, 1)
    @assert length(x0) == dim_x == size(Xs, 1)
    @assert isnothing(Y) && isnothing(Ys) || size(Y, 1) == size(Ys, 1)
    @assert isnothing(Y) || size(X, 2) ==  size(Y, 2)
    @assert isnothing(Ys) || size(Xs, 2) ==  size(Ys, 2)
    
    _n_X = n_X = ix2 - ix1 + 1

    if n_X >= dim_x
        return 0
    end
      
    j = ix2 + 1
    XJ = X[:, ix1:j-1]
    xj = X[:, j]
    qr = do_qr!(qr_ws, XJ, QRbuff)
    
    len_Xs = size(Xs, 2)
    for l=len_Xs:-1:1
        xl = Xs[:, l]
        xj .= xl .- x0
        copyto!(xZ, xj)
        project_onto_Q_colspace!(xZ, qr, n_X + 1, dim_x)
        if LA.norm(xZ, norm_p) >= th
            copy_col!(Y, Ys, j, l)
            n_X += 1
            if !isnothing(chosen_index)
                chosen_index[j] = l
            end
            if n_X == dim_x
                break
            end
            j += 1
            XJ = X[:, ix1:j-1]
            xj = X[:, j]
            qr = do_qr!(qr_ws, XJ, QRbuff)
        end
    end
    return n_X - _n_X, qr
end

function sample_along_Z!(
    X,      # holds sufficiently independent columns between `ix1` and `ix2`
    qr_ws,  # if `qr_ws` is not nothing, it is used as a workspace for `do_qr!`
    QRbuff,
    x0, 
    lb, 
    ub, 
    th_qr;
    n_new = nothing,
    qr=nothing, # if `qr` is not nothing, we assume it to be a valid 
                # QR decomposition of `X[:, ix1:ix2]`
    ix1=2, 
    ix2=ix1-1, 
    norm_p=Inf
)

    ix2 = max(0, ix2)
    _n_X = n_X = ix2 - ix1 + 1
    dim_x = size(X,1)

    if n_X >= dim_x
        return 0, qr
    end

    dim_Z = dim_x - n_X
    n_its = isnothing(n_new) ? dim_Z : min(n_new, dim_Z)
    n_its = min(size(X,2) - ix2, n_its)
    if isnothing(qr)
        qr = do_qr!(qr_ws, @view(X[:, ix1:ix2]), QRbuff)
    end

    for i=1:n_its
        z = @view(X[:, ix2+i])
        ## project onto orthogonal complement of `X[:, ix1:ix2+i]`:
        j_col_Z = _n_X + i
        z .= 0
        z[j_col_Z] = 1
        LA.lmul!(qr.Q, z)
        ## scale into box:
        @ignoraise fit_z_into_box!(z, x0, lb, ub; norm_p, th_qr)
        n_X += 1
    end
    return n_X - _n_X, qr 
end

function fit_z_into_box!(
    z, x0, lb, ub;
    norm_p=Inf,
    th_qr
)
    σ = stepsize_in_box(x0, z, lb, ub)
    z .*= σ
    norm_z = LA.norm(z, norm_p)
    if iszero(norm_z) 
        return RBFConstructionImpossible()
    elseif norm_z < th_qr
        ## c * norm_z = th_qr ⇔ c = th_qr / norm_z
        c = th_qr / norm_z
        @warn "Incompatible box constraints, changing σ from $σ to $(c*σ)."
        z .*= c
    end
    return z
end

function stepsize_in_box(x, z, lb, ub)
    σ_min, σ_max = intersect_box(x, z, lb, ub)
    if abs(σ_min) > abs(σ_max)
        return σ_min
    else
        return σ_max
    end
end
