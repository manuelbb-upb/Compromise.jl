function CE.process_trial_point!(
    rbf::RBFModel, xtrial, fxtrial, is_next::Bool;
    log_level=Info
)
    if !is_next
        @unpack params, database = rbf
        if params.xtrial != xtrial
            lock_write(database.rwlock) do
                add_to_database!(database, xtrial, fxtrial)
            end
            params.xtrial .= xtrial
        end
    end
    return nothing
end

function update_rbf_model!(
    rbf::RBFModel, op, Δ, x0, fx0, global_lb=nothing, global_ub=nothing; 
    norm_p=Inf, log_level=Info, force_rebuild::Bool=false, indent::Int=0
)
    indent += 1
    pad_str = lpad("", indent)

    @unpack delta_max, dim_x, dim_y, max_points, min_points = rbf
    @assert dim_x == length(x0)
    @assert dim_y == length(fx0)

    @unpack buffers, params, database = rbf
    was_fully_linear = val(params.is_fully_linear_ref)
    last_db_state = val(params.database_state_ref)
    
    @unpack rwlock = database
    
    lock_read(rwlock)
        # we need to update if
        # we are force by `force_rebuild` OR
        # if `x0` changed or the radius changed OR
        # the previous model was not fully linear and there is new points in the database

        if (    # we do not need to update if …
            !force_rebuild &&                                   # … we are not forced 
            x0 == params.x0 && Δ == val(params.delta_ref) &&    # … did the center or radius change nor
            !(
                was_fully_linear &&                             # … if the previous model can be improved
                last_db_state != db_state(database)             # … with new points from the database
            )
        )
            @logmsg log_level "$(pad_str)RBFModel: No need to update."
            unlock_read(rwlock)
            return nothing
        else
            @logmsg log_level "$(pad_str)RBFModel: Starting surrogate update."
        end
    unlock_read(rwlock)

    @ignoraise n_X_affine_sampling, n_X = lock_read(rwlock) do
        @ignoraise n_X = affine_sampling!(
            rbf, Δ, x0, fx0, global_lb, global_ub; 
            delta_max, norm_p, log_level, indent
        )
        @ignoraise n_X = eval_missing_values!(rbf, op, x0, n_X)
        
        if n_X < rbf.min_points
            @warn "$(pad_str)Cannot make a fully linear RBF model."
        end

        @unpack shape_parameter_ref = params;
        val!(shape_parameter_ref, model_shape_parameter(rbf, Δ))
    
        n_X_affine_sampling = n_X
        @unpack max_search_factor = rbf

        if n_X == min_points
            delta = delta_max * max_search_factor
            n_X = cholesky_point_search!(rbf, x0, n_X; log_level, delta, indent)
        end
       
        n_X_affine_sampling, n_X
    end
    
    new_db_state = lock_write(rwlock) do
        put_new_evals_into_db!(rbf, x0, n_X_affine_sampling, buffers.xZ)
        db_state(database)
    end

    if n_X_affine_sampling != min_points
        @ignoraise least_squares_model!(rbf, n_X; log_level, indent)
    else
        @unpack coeff_φ, coeff_π, = params 
        @unpack FX, Linv, Φ, Q, R, L = buffers
        @unpack dim_y, dim_π = rbf
        @ignoraise set_coefficients!(
            coeff_φ, coeff_π, FX, Linv, Φ, Q, R, L;
            n_X, dim_y, dim_π
        )
    end
     
    val!(params.delta_ref, Δ)
    params.x0 .= x0
    #params.xtrial .= NaN

    val!(params.n_X_ref, n_X)
    val!(buffers.x0_db_index_ref, buffers.db_index[1])
    val!(params.database_state_ref, new_db_state)

    return nothing
end

function model_shape_parameter(rbf::RBFModel, Δ)
    if rbf.shape_parameter_function isa Function
        ε = 0 
        try 
            ε = rbf.shape_parameter_function(Δ)
        catch err
            @error "Could not compute shape parameter." exception=(err, catch_backtrace())
        end
        if ε > 0
            return ε
        end
    end
    if rbf.shape_parameter_function isa Number
        ε = rbf.shape_parameter_function
        if ε > 0
            return ε
        end
    end
    return _shape_parameter(rbf.kernel)
end

# Suppose, `filter_flags` has been used to create a view into `_X` via
# `X = @view(X[:, filter_flags])`.
# From `X`, certain columns are chosen, these have integer indices `chosen_index`.
# The function `unfilter_index!` modifies `chosen_index` to contain indices
# with respect to `_X` instead of `X`.
# It is equivalent to `chosen_index = findall(filter_flags)[chosen_index]`
function unfilter_index!(
    chosen_index, filter_flags; ix1=2, ix2=length(chosen_index)
)
    j = 0
    for (i, is) = enumerate(filter_flags)
        if is
            j += 1  # flag `i` is the `j`-th true flag
            for l = ix1:ix2
                ci = chosen_index[l]
                if ci == j
                    chosen_index[l] = i
                    break
                end
            end
        end
    end
end

function eval_missing_values!(rbf, op, x0, n_X)
    @unpack buffers, params = rbf
    @unpack X = params
    @unpack db_flags, db_index, FX, xZ, fxZ = buffers

    return eval_missing_values!(X, FX, db_flags, xZ, fxZ, db_index, n_X, op, x0)
end

@views function eval_missing_values!(X, FX, db_flags, xtmp, fxtmp, db_index, n_X, op, x0)

    # `presort_eval_arrays!` will sort columns in `X`, `FX` and `db_flags` such that
    # indices `1:i0` have database values already.
    i0 = presort_eval_arrays!(X, FX, db_flags, xtmp, fxtmp, db_index, n_X)
    
    # indices `i0+1:n_X` need evaluation still
    i0 += 1
    _X = X[:, i0:n_X] 
    _FX = FX[:, i0:n_X]
    
    _X .+= x0       # for evaluation, undo shift
    _FX .= NaN
    @ignoraise func_vals!(_FX, op, _X)
    _X .-= x0       # and re-do shift afterwards    # TODO extra buffer to avoid these additions?

    # sort arrays so that valid entries are first, in columns `1:n_X`
    n_X = postsort_eval_arrays!(X, FX, db_flags, xtmp, fxtmp, n_X; i0)
    return n_X
end

function presort_eval_arrays!(
    X, FX, db_flags, xtmp, fxtmp,
    db_index, n_X
)
    make_presort_flags!(db_flags, db_index, n_X)
    return sort_with_flags!(X, FX, db_flags, xtmp, fxtmp, n_X)
end

function make_presort_flags!(db_flags, db_index, n_X)
    for i = 1:n_X
        db_flags[i] = (db_index[i] > 0)
    end
    return nothing
end

function postsort_eval_arrays!(X, FX, db_flags, xtmp, fxtmp, n_X; i0=1)
    make_postsort_flags!(db_flags, FX, n_X; i0)
    return sort_with_flags!(X, FX, db_flags, xtmp, fxtmp, n_X; i0)
end

function make_postsort_flags!(db_flags, FX, n_X; i0=1)
    for i = i0:n_X 
        db_flags[i] = !isnan(FX[1, i])
    end
    return nothing
end

function sort_with_flags!(X, FX, db_flags, xtmp, fxtmp, n_X; i0=1)
    j=i0-1 # index of last entry with db flag
    for i=i0:n_X
        has_db = db_flags[i]
       
        if has_db
            xtmp .= X[:, i]
            fxtmp .= FX[:, i]
            
            # sub-arrays with 1:i-1 are sorted
            # columns 1:j have entries with database flag 
            # make current column `i` the new column `j`
            
            jr = j + 1
            if j>0
                # shift all entries after `j` one to the right
                il = i - 1
                db_flags[jr:i] .= db_flags[j:il] 
                X[:, jr:i] .= X[:, j:il]
                FX[:, jr:i] .= FX[:, j:il]
            end
            # then place entries with index `i` into correct spot
            j = jr
            db_flags[j] = has_db
            X[:, j] .= xtmp
            FX[:, j] .= fxtmp
        end
    end
    return j
end

@views function put_new_evals_into_db!(rbf, x0, n_X, xZ)
    @unpack buffers, database, params = rbf
    @unpack db_flags, FX = buffers
    @unpack X = params
    for i=1:n_X
        if !db_flags[i]
            xZ .= X[:, i]
            xZ .+= x0
            y = FX[:, i]
            add_to_database!(database, xZ, y)
        end
    end
end
#=
function evaluate_and_update_db!(rbf, op, x0, n_X)
    @unpack database, buffers, params = rbf
    @unpack X = params
    @unpack db_flags, db_index, FX, lb, ub = buffers

    ix1 = 1
    ix2 = n_X
    return evaluate_and_update_db!(
        db_flags, db_index, FX, database,
        op, X, x0, lb, ub; 
        ix1, ix2
    )
end

@views function evaluate_and_update_db!(
    not_db_flags, db_index, FX, database,
    op, X, x0, lb, ub; 
    ix1, ix2,
)
    @assert ix1==1

    for i = ix1:ix2
        not_db_flags[i] = db_index[i] <= 0
    end
    _X = X[:, ix1:ix2][:, not_db_flags[ix1:ix2]]
    _Y = FX[:, ix1:ix2][:, not_db_flags[ix1:ix2]]
    _Y[1, :] .= NaN
    _X .+= x0
    @ignoraise op_code = func_vals!(_Y, op, _X)

    use_col_flags = not_db_flags
    n_X = 0

    for i=ix1:ix2
        if use_col_flags[i]
            xi = X[:, i]
            project_into_box!(xi, lb, ub)
            if isnan(FX[1, i])
                use_col_flags[i] = false
            else
                add_to_database!(database, xi, FX[:, i])
                n_X += 1
            end
            xi .-= x0
        else
            use_col_flags[i] = true
            n_X += 1
        end
    end
    return n_X
end
=#

function affine_sampling!(
    rbf::RBFModel, Δ, x0, fx0, global_lb=nothing, global_ub=nothing; 
    delta_max, norm_p, log_level, indent::Int=0
)
    @unpack dim_x, params, buffers, database = rbf
    @unpack X = params
    @unpack FX, db_index, lb, ub, xZ, qr_ws_dim_x = buffers
    @unpack has_z_new_ref, is_fully_linear_ref, z_new = params
    @unpack min_points, max_points, search_factor, max_search_factor = rbf 
    @unpack sampling_factor, max_sampling_factor, th_qr, enforce_fully_linear = rbf
 
    x0_db_index = val(buffers.x0_db_index_ref)

    if x0_db_index > 0
        _x0 = view(db_view_x(database), 1:dim_x, x0_db_index)
        if !isequal(x0, _x0)
            x0_db_index = 0
        end
    end
    
    #=
    if x0_db_index < 1
        x0_db_index = add_to_database!(database, x0, fx0)
    end
    =# #should happen later on anyways

    QRbuff = buffers.Q
    @unpack filter_flags = buffers
    return affine_sampling!(
        X, FX, db_index, lb, ub, xZ, qr_ws_dim_x, QRbuff,
        has_z_new_ref, is_fully_linear_ref, z_new,
        filter_flags, database,
        min_points, max_points, search_factor, max_search_factor, 
        sampling_factor, max_sampling_factor, th_qr,
        Δ, x0, fx0, global_lb, global_ub;
        norm_p, delta_max, enforce_fully_linear, log_level, x0_db_index,
        indent
    )
end

@views function affine_sampling!(
    X, FX, db_index::AbstractVector{Int}, lb, ub, xZ, qr_ws_dim_x, QRbuff,
    has_z_new_ref, is_fully_linear_ref, z_new, 
    filter_flags, database, 
    min_points, max_points, search_factor, max_search_factor,
    sampling_factor, max_sampling_factor, th_qr, 
    Δ, x0, fx0, global_lb, global_ub;
    norm_p, enforce_fully_linear, delta_max, log_level, x0_db_index=-1,
    indent::Int=0
)
    indent += 1
    pad_str = lpad("", indent)
    
    ## first column will contain `x0`, but shifted into origin, so set to zero:
    db_index .= -1

    X[:, 1] .= 0
    FX[:, 1] .= fx0
    db_index[1] = 0
    n_X = 1
    db_index[1] = x0_db_index
    
    th_qr *= Δ
    
    if n_X < min_points  
        Δz = sampling_factor .* Δ
        trust_region_bounds!(lb, ub, x0, Δz, global_lb, global_ub)

        has_z_new = val(has_z_new_ref)
        if has_z_new           
            X[:, 2] .= z_new
            fit_z_into_box!(X[:, 2], x0, lb, ub; norm_p, th_qr)
            db_index[2] = -1
            n_X += 1
        end
    end
    
    _n_X = n_X
    if n_X < min_points
        Δ1 = search_factor .* Δ
        trust_region_bounds!(lb, ub, x0, Δ1, global_lb, global_ub)

        box_search!(filter_flags, database, lb, ub)
        for ix in 1:n_X
            ci = db_index[ix]
            ci < 1 && continue
            filter_flags[ci] = false
        end

        Xs = filtered_view_x(database, filter_flags)
        Ys = filtered_view_y(database, filter_flags)
        n_new, qr = find_poised_points!(    
            X, FX, qr_ws_dim_x, QRbuff, x0, Xs, Ys;
            xZ, ix1=2, ix2=n_X, norm_p, chosen_index=db_index, th=th_qr,    
         )
        
        n_X += n_new
        @logmsg log_level "$(pad_str)RBFModel: Found $(n_new) points in radius $(Δ1)."

        ## account for offset indices in chosen_index:
        unfilter_index!(db_index, filter_flags; ix1=_n_X+1, ix2=n_X) 
    end

    if n_X < min_points && enforce_fully_linear
        ΔZ1 = sampling_factor .* Δ
        trust_region_bounds!(lb, ub, x0, ΔZ1, global_lb, global_ub)

        @ignoraise n_new, qr = sample_along_Z!(
            X, qr_ws_dim_x, QRbuff, x0, lb, ub, th_qr;
            ix1=2, ix2=n_X, norm_p, qr, n_new = min_points - n_X
        )
        n_X += n_new
        @logmsg log_level "$(pad_str)RBFModel: Sampled $(n_new) points in radius $(ΔZ1)."
    end
    
    if n_X < min_points || min_points < max_points
        Δ2 = max_search_factor .* delta_max
        trust_region_bounds!(lb, ub, x0, Δ2, global_lb, global_ub)

        box_search!(filter_flags, database, lb, ub; xor=true)
        for ix in 1:n_X # if `xor=true` works as it should, this loop should not be necessary
            ci = db_index[ix]
            ci < 1 && continue
            filter_flags[ci] = false
        end
    end

    if n_X < min_points
        if enforce_fully_linear
            @warn "$(pad_str)Cannot make model fully linear."
        end

        ## store model-improvement direction (next column of qr factor)
        z_new .= 0
        z_new[n_X + 1] = 1
        LA.lmul!(qr.Q, z_new)
        val!(has_z_new_ref, true)

        Xs = filtered_view_x(database, filter_flags)
        Ys = filtered_view_y(database, filter_flags)
        n_new, qr = find_poised_points!(    
            X, FX, qr_ws_dim_x, QRbuff, x0, Xs, Ys;
            xZ, ix1=2, ix2=n_X, norm_p, chosen_index = db_index, th = th_qr,    
        )
        _n_X = n_X + n_new
        @logmsg log_level "$(pad_str)RBFModel: Found $(n_new) points in radius $(Δ2)."
        
        ## `filter_flags` will be used in cholesky sampling.
        ## to avoid unnecessary checks, unmark chosen indices here already:
        ## 1) account for offset indices in chosen_index:
        unfilter_index!(db_index, filter_flags; ix1=n_X+1, ix2=_n_X)
        ## 2) unmark
        for ix in n_X+1:_n_X
            ci = db_index[ix]
            ci < 1 && continue
            filter_flags[ci] = false
        end
        n_X = _n_X
    else
        val!(is_fully_linear_ref, true)
        val!(has_z_new_ref, false)
    end

    if n_X < min_points
        ΔZ2 = max_sampling_factor .* delta_max
        trust_region_bounds!(lb, ub, x0, ΔZ2, global_lb, global_ub)

        @ignoraise n_new, qr = sample_along_Z!(
            X, qr_ws_dim_x, QRbuff, x0, lb, ub, th_qr;
            ix1=2, ix2=n_X, norm_p, qr, n_new = min_points - n_X
        )
        n_X += n_new
        @logmsg log_level "$(pad_str)RBFModel: Sampled $(n_new) points in radius $(ΔZ2)."
    end

    return n_X
end

function find_additional_points!(
    rbf, x0, n_X;
    log_level, delta, indent::Int=0
)
    if n_X != rbf.min_points
        least_squares_model!(rbf, n_X; log_level, indent)
        return n_X
    else
        return cholesky_point_search!(rbf, x0, n_X; log_level, delta, indent)
    end
end

function least_squares_model!(rbf, n_X; log_level, indent::Int=0)
    @unpack params, buffers, min_points, dim_y, dim_π, kernel, poly_deg = rbf
    @unpack X = params
    @unpack FX, Φ, Qj = buffers
    ε = val(params.shape_parameter_ref)

    _X = @view(X[:, 1:n_X])
    _Y = transpose(@view(FX[:, 1:n_X]))
    @unpack coeff_φ, coeff_π = params
    cφ = @view(coeff_φ[1:n_X, 1:dim_y])
    cπ = @view(coeff_π[1:dim_π, 1:dim_y])
    _Π = @view(Qj[1:n_X, 1:dim_π])
    _Φ = @view(Φ[1:n_X, 1:n_X])
    _rbf_poly_mat!(_Π, poly_deg, _X)
    _rbf_kernel_mat!(_Φ, kernel, _X, _X, ε; centers_eq_features=true)
    @ignoraise _rbf_solve_normal_eqs!(cφ, cπ, hcat(_Φ, _Π), _Y)
    return nothing
end

function cholesky_point_search!(rbf, x0, n_X; log_level, delta, indent::Int=0)
    
    @assert n_X == rbf.min_points
    @unpack params, buffers, database = rbf
    @unpack min_points, max_points, dim_x, dim_y, dim_π, kernel, poly_deg, th_cholesky = rbf
    @unpack X = params
    @unpack FX, Φ, L, Linv, Q, R, qr_ws_min_points, Qj, filter_flags = buffers
    
    ε = val(params.shape_parameter_ref)
    
    initial_qr_for_cholesky_test!(
        Φ, Q, R, qr_ws_min_points, Qj, X;
        kernel, poly_deg, ε, n_X, dim_x, dim_π
    )
    
    Xs = filtered_view_x(database, filter_flags)
    if size(Xs, 2) > 0

        Ys = filtered_view_y(database, filter_flags)

        @unpack Rj, L, Linv, v1, v2 = buffers
        j = n_X + 1
        φ0 = apply_kernel(kernel, 0, ε)
        xj = buffers.xZ
        for (_xj, yj) = zip(eachcol(Xs), eachcol(Ys))
            xj .= _xj .- x0
            if n_X >= max_points
                break
            end
            τj = compute_cholesky_test_value!(
                Φ, Rj, Qj, Linv, v1, v2, X, R, Q;
                xj, kernel, poly_deg, ε, φ0, dim_x, dim_π, n_X
            )
            if τj >= th_cholesky
                X[:, j] .= xj
                FX[:, j] .= yj
                update_cholesky_buffers!(
                    Φ, Q, R, L, Linv, v1, v2, Rj, Qj;
                    n_X, τj, dim_π, φ0
                )
                n_X += 1
                j += 1
            end
        end
    end
    
    @unpack coeff_φ, coeff_π = params
  
    n_new = n_X - rbf.min_points
    if n_new > 0
        indent += 1
        pad_str = lpad("", indent)
        @logmsg log_level "$(pad_str)RBFModel: Found $(n_new) additional points in radius $delta."
    end
    return n_X
end