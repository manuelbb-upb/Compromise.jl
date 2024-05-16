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
    
    fits_filter = is_filter_acceptable(filter, vals, vals_tmp)
    (x, fx, θx, Φx, fx_mod, xs, fxs, θxs, Φxs, fxs_mod) = _trial_point_arrays(
        vals, vals_tmp, mod_vals, step_vals)

    @unpack strict_acceptance_test, kappa_theta, psi_theta, nu_accept, nu_success = algo_opts
    @unpack diff_x, diff_fx, diff_fx_mod = trial_caches
    iteration_type = _test_trial_point!(
        diff_x, diff_fx, diff_fx_mod,
        x, xs, fx, fxs, fx_mod, fxs_mod, θx, fits_filter, 
        strict_acceptance_test, kappa_theta, psi_theta, nu_accept, nu_success;
    )
    if do_log_trial
        _log_trial_results(θxs, Φxs, iteration_type; indent, log_level)
    end

    delta = iteration_scalars.delta
    @unpack gamma_grow, gamma_shrink, gamma_shrink_much, delta_max = algo_opts
    delta_new, radius_update = _update_radius(
        delta, iteration_type, gamma_grow, gamma_shrink, gamma_shrink_much, delta_max)
    
    if do_log_radius
        _log_radius_update(delta, delta_new, radius_update; indent, log_level)
    end

    iteration_status.iteration_type = iteration_type
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
    strict_acceptance_test, kappa_theta, psi_theta, nu_accept, nu_success
)
    
    @. diff_x = x - xs
    @. diff_fx = fx - fxs
    @. diff_fx_mod = fx_mod .- fxs_mod

    it_type = INITIALIZATION
    
    if !fits_filter
        it_type = FILTER_FAIL
    end

    if it_type == INITIALIZATION
        objf_decrease, model_decrease = if strict_acceptance_test
            (
                diff_fx, 
                diff_fx_mod
            )
        else
            (
                maximum(fx) - maximum(fxs),
                maximum(fx_mod) - maximum(fxs_mod) 
            )
        end
        rho = minimum( objf_decrease ./ model_decrease )            
        sufficient_decrease_condition = rho >= nu_accept
        successful_decrease_condition = rho >= nu_success
        
        if all(model_decrease .>= kappa_theta * θx^psi_theta)
            if sufficient_decrease_condition
                if successful_decrease_condition
                    it_type = F_STEP_SUCCESSFUL
                else
                    it_type = F_STEP_ACCEPTABLE
                end                
            else
                it_type = INACCEPTABLE
            end
        else
            # constraint violation significant compared to predicted decrease
            it_type = THETA_STEP
        end
    end
    return it_type
end

function _log_trial_results(θxs, Φxs, iteration_type; indent, log_level)
    indent += 1
    pad_str = indent_str(indent)
    
    @logmsg log_level "$(pad_str)Trial point with (θ, Φ) = $((θxs, Φxs)) does$(iteration_type == FILTER_FAIL ? " not" : "") fit filter."
    @logmsg log_level "$(pad_str)$(iteration_type): The trial point is$(_trial_point_accepted(iteration_type) ? "" : " not") accepted."
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
    delta, it_type,
    gamma_grow, gamma_shrink, gamma_shrink_much, delta_max
)
    if it_type == F_STEP_ACCEPTABLE
        delta_new = gamma_shrink * delta
        radius_update = SHRINK
    elseif it_type == F_STEP_SUCCESSFUL
        if delta < delta_max
            delta_new = min(gamma_grow * delta, delta_max)
            radius_update = GROW
        else
            delta_new = delta
            radius_update = GROW_FAIL
        end
    elseif it_type == THETA_STEP
        delta_new = delta
        radius_update = NO_CHANGE
    else
        delta_new = gamma_shrink_much * delta
        radius_update = SHRINK
    end
    return delta_new, radius_update
end