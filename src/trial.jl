function test_trial_point!(
    mop, mod, scaler, lin_cons, scaled_cons, vals, vals_tmp,
    mod_vals, filter, step_vals, step_cache, crit_cache, trial_caches, 
    iteration_status, iteration_scalars, stop_crits,
    algo_opts;
    indent::Int=0,
    do_log_trial::Bool=true,
    do_log_radius::Bool=true
)
   
    @unpack log_level = algo_opts
    @logmsg log_level "$(indent_str(indent))* Checking trial point."
    
    # Use `vals_tmp` to hold the true values at `xs`
    copyto!(cached_x(vals_tmp), step_vals.xs)
    @ignoraise eval_mop!(vals_tmp, mop, scaler) indent
    
    fits_filter = is_filter_acceptable(filter, vals_tmp, vals)
    (x, fx, θx, Φx, fx_mod, xs, fxs, θxs, Φxs, fxs_mod) = _trial_point_arrays(
        vals, vals_tmp, mod_vals, step_vals)

    @ignoraise objectives!(fxs_mod, mod, xs) indent # just to be sure

    @unpack trial_mode, kappa_theta, psi_theta, nu_accept, nu_success = algo_opts
    @unpack diff_x, diff_fx, diff_fx_mod = trial_caches
    iteration_classification, rho, rho_classification = _test_trial_point!(
        diff_x, diff_fx, diff_fx_mod,
        x, xs, fx, fxs, fx_mod, fxs_mod, θx, fits_filter, 
        kappa_theta, psi_theta, nu_accept, nu_success, 
        trial_mode;
        log_level, indent=indent+1
    )

    if do_log_trial
        _log_trial_results(θxs, Φxs, iteration_classification, rho_classification; indent, log_level)
    end

    delta = iteration_scalars.delta
    @unpack gamma_grow, gamma_shrink, gamma_shrink_much, delta_max = algo_opts
    delta_new, radius_update = _update_radius(
        delta, 
        iteration_classification, rho, rho_classification,
        gamma_grow, gamma_shrink, gamma_shrink_much, delta_max,
        nu_accept, nu_success
    )
    
    if do_log_radius
        _log_radius_update(delta, delta_new, radius_update; indent, log_level)
    end

    iteration_status.rho = rho
    iteration_status.rho_classification = rho_classification
    iteration_status.iteration_classification = iteration_classification
    iteration_status.radius_update_result = radius_update
    return delta_new
end

function _trial_point_arrays(vals, vals_tmp, mod_vals, step_vals)
       
    x = cached_x(vals)
    fx = cached_fx(vals)

    xs = cached_x(vals_tmp)
    fxs = cached_fx(vals_tmp)

    θx = cached_theta(vals)
    Φx = cached_Phi(vals)
    
    θxs = cached_theta(vals_tmp)
    Φxs = cached_Phi(vals_tmp)

    fx_mod = cached_fx(mod_vals)
    fxs_mod = step_vals.fxs

    return (x, fx, θx, Φx, fx_mod, xs, fxs, θxs, Φxs, fxs_mod)
end

function _test_trial_point!(
    diff_x, diff_fx, diff_fx_mod,
    x, xs, fx, fxs, fx_mod, fxs_mod, θx, fits_filter,
    kappa_theta, psi_theta, nu_accept, nu_success,
    mode::Union{Val{:max_diff}, Val{:min_rho}, Val{:max_rho}}=Val(:max_diff);
    log_level, indent
)
    
    @. diff_x = x - xs
    @. diff_fx = fx - fxs
    @. diff_fx_mod = fx_mod .- fxs_mod

    rho = NaN
    rho_classification = RHO_NAN
    iteration_classification = IT_INITIALIZATION
    
    if !fits_filter
        iteration_classification = IT_FILTER_FAIL
    end

    if iteration_classification == IT_INITIALIZATION
        f_step_test_rhs = kappa_theta * θx^psi_theta
        rho, is_f_step = _trial_analysis(mode, fx, fxs, fx_mod, fxs_mod, diff_fx, diff_fx_mod, f_step_test_rhs)  
        
        @logmsg log_level """\n
        $(indent_str(indent))- mode     = $(mode), 
        $(indent_str(indent))- fx - fxs = $(pretty_row_vec(diff_fx)), 
        $(indent_str(indent))- mx - mxs = $(pretty_row_vec(diff_fx_mod))
        $(indent_str(indent))- rho      = $(rho)
        """
        
        if is_f_step
            iteration_classification = IT_F_STEP
            if rho >= nu_accept
                if rho >= nu_success
                    rho_classification = RHO_SUCCESS
                else
                    rho_classification = RHO_ACCEPT
                end
            else
                rho_classification = RHO_FAIL
            end
        else
            # constraint violation significant compared to predicted decrease
            iteration_classification = IT_THETA_STEP
        end
    end
    return iteration_classification, rho, rho_classification
end

function _trial_analysis(
    ::Val{:max_diff}, 
    fx, fxs, fx_mod, fxs_mod, diff_fx, diff_fx_mod,
    f_step_test_rhs
)
    ## following suggestions from “Trust Region Methods” by Conn et. al.,
    ## Subsec. 17.4.2
    eps_k = eps(eltype(fx)) * 10
    max_fx = maximum(fx)
    del_k = eps_k * max_fx

    max_fxs = maximum(fxs)
    max_fxs_mod = maximum(fxs_mod)
    objf_decrease = maximum(fx) - max_fxs - del_k
    model_decrease = maximum(fx_mod) - max_fxs_mod
    is_f_step = model_decrease >= f_step_test_rhs
    model_decrease -= del_k
    rho = if (
        max_fxs == max_fxs_mod || 
        (
            abs(objf_decrease) < eps_k && abs(model_decrease) < eps_k
        )
    )   
        1
    else
        objf_decrease / model_decrease
    end
    return rho, is_f_step
end

function _trial_analysis(
    mode::Val{:min_rho}, 
    fx, fxs, fx_mod, fxs_mod, diff_fx, diff_fx_mod,
    f_step_test_rhs
)
    return __trial_analysis(<=, Inf, fxs, fxs_mod, diff_fx, diff_fx_mod, f_step_test_rhs)
end

function _trial_analysis(
    mode::Val{:max_rho}, 
    fx, fxs, fx_mod, fxs_mod, diff_fx, diff_fx_mod,
    f_step_test_rhs
)
    return __trial_analysis(>=, -Inf, fxs, fxs_mod, diff_fx, diff_fx_mod, f_step_test_rhs)
end

function __trial_analysis(
    comp_op, rho0, 
    fxs, fxs_mod, diff_fx, diff_fx_mod,
    f_step_test_rhs
)
    ## following suggestions from “Trust Region Methods” by Conn et. al.,
    ## Subsec. 17.4.2
    eps_k = eps(eltype(fx)) * 10
    del_k = eps_k * mapreduce(abs, max, fx)
    
    rho = rho0
    _i = 0
    for i = eachindex(diff_fx)
        diff_mod = diff_fx_mod[i] - del_k
        diff_mod <= 0 && continue
        diff_func = diff_fx[i] - del_k
        rho_i = if (
            fxs[i] == fxs_mod[i] ||
            (
                abs(diff_mod) < del_k && abs(diff_func) < del_k
            )
        )
            1
        else
            diff_func / diff_mod
        end
        rho_i = fx_i / mx_i
        if comp_op(rho_i , rho)
            _i = i
            rho = rhoi
        end
    end
    
    is_f_step = false
    if _i > 0
        is_f_step = diff_fx_mod[_i] >= f_step_test_rhs
    end

    if isinf(rho) || isnan(rho)
        rho = 0
    end

    return rho, is_f_step 
end

function _log_trial_results(θxs, Φxs, iteration_classification, rho_classification; indent, log_level)
    indent += 1
    pad_str = indent_str(indent)
    
    @logmsg log_level "$(pad_str)Trial point with (θ, Φ) = $((θxs, Φxs)) does$(iteration_classification == IT_FILTER_FAIL ? " not" : "") fit filter."
    @logmsg log_level "$(pad_str)$(iteration_classification) && $(rho_classification)."
    return nothing
end

function _log_radius_update(delta, delta_new, radius_update; indent, log_level)
    indent += 1
    pad_str = indent_str(indent)
    if Int8(radius_update) > 0
        @logmsg log_level "$(pad_str)Radius will be changed from $(delta) to $(delta_new)."
    else
        @logmsg log_level "$(pad_str)Radius will stay at $(delta)."
    end
    return nothing
end

function _update_radius(
    delta,
    iteration_classification, rho, rho_classification,
    gamma_grow, gamma_shrink, gamma_shrink_much, delta_max,
    nu_accept, nu_success
)
    radius_update = nothing
    if iteration_classification == IT_F_STEP ||
        iteration_classification == IT_THETA_STEP
        if rho_classification == RHO_ACCEPT
            gamma_factor = gamma_shrink
            if nu_success > nu_accept
                rho_frac = (rho - nu_accept) / (nu_success - nu_accept)
                gamma_factor += rho_frac * (1 - gamma_shrink)
            end
            delta_new = gamma_factor * delta
            radius_update = RADIUS_SHRINK
        elseif rho_classification == RHO_SUCCESS
            if delta < delta_max
                delta_new = min(gamma_grow * delta, delta_max)
                radius_update = RADIUS_GROW
            else
                delta_new = delta
                radius_update = RADIUS_GROW_FAIL
            end
        end 
    #=
    elseif iteration_classification == IT_THETA_STEP
        delta_new = delta
        radius_update = RADIUS_NO_CHANGE
    =#
    end

    if isnothing(radius_update)
        delta_new = delta * gamma_shrink_much
        radius_update = RADIUS_SHRINK
    end

    return delta_new, radius_update
end