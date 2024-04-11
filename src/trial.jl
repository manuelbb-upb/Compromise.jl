function test_trial_point!(optimizer_caches, algo_opts; indent=0)
    @unpack (
        iteration_status, trial_caches, vals, vals_tmp, mod_vals, step_vals,
        iteration_scalars, mop, scaler, filter
    ) = optimizer_caches 
    return test_trial_point!(
        iteration_status, trial_caches, vals, vals_tmp, mod_vals, step_vals,
        iteration_scalars, mop, scaler, filter, algo_opts;
        indent
    )   
end

function test_trial_point!(
    iteration_status, trial_caches, 
    vals, vals_tmp, mod_vals, step_vals,
    # not modified:
    iteration_scalars, mop, scaler, filter, algo_opts;
    indent::Int=0
)
    @unpack log_level = algo_opts
    @logmsg log_level "$(indent_str(indent))* Checking trial point."

    # Use `vals_tmp` to hold the true values at `xs`
    copyto!(cached_x(vals_tmp), step_vals.xs)
    @ignoraise eval_mop!(vals_tmp, mop, scaler) indent
    
    (x, fx, θx, Φx, fx_mod, xs, fxs, θxs, Φxs, fxs_mod) = _trial_point_arrays(
        vals, vals_tmp, mod_vals, step_vals)

    @unpack strict_acceptance_test, kappa_theta, psi_theta, nu_accept, nu_success = algo_opts
    @unpack diff_x, diff_fx, diff_fx_mod = trial_caches
    iteration_type, step_class = _test_trial_point!(
        diff_x, diff_fx, diff_fx_mod,
        filter, x, xs, fx, fxs, fx_mod, fxs_mod, θx, θxs, Φx, Φxs,
        strict_acceptance_test, kappa_theta, psi_theta, nu_accept, nu_success;
    )
    _log_trial_results(iteration_type, step_class; indent, log_level)

    delta = iteration_scalars.delta
    @unpack gamma_grow, gamma_shrink, gamma_shrink_much, delta_max = algo_opts
    delta_new, radius_update = _update_radius(
        delta, step_class, gamma_grow, gamma_shrink, gamma_shrink_much, delta_max)
    _log_radius_update(delta, delta_new, radius_update; indent, log_level)

    iteration_status.iteration_type = iteration_type
    iteration_status.step_class = step_class
    iteration_status.radius_update = radius_update
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
    filter, x, xs, fx, fxs, fx_mod, fxs_mod, θx, θxs, Φx, Φxs,
    strict_acceptance_test, kappa_theta, psi_theta, nu_accept, nu_success
)
    
    fits_filter = is_acceptable(filter, θxs, Φxs, θx, Φx)
    
    @. diff_x = x - xs
    @. diff_fx = fx - fxs
    @. diff_fx_mod = fx_mod .- fxs_mod

    it_type = INITIALIZATION
    step_class = INACCEPTABLE
    it_type = if !fits_filter
        FILTER_FAIL
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
        
        if all(model_decrease .>= kappa_theta * θx^psi_theta)
            it_type = F_STEP
        else
            # constraint violation significant compared to predicted decrease
            it_type = THETA_STEP
        end

        if sufficient_decrease_condition
            step_class = rho < nu_success ? ACCEPTABLE : SUCCESSFUL
        end
    end
    return it_type, step_class
end

function _log_trial_results(iteration_type, step_class; indent, log_level)
    indent += 1
    pad_str = indent_str(indent)
    
    @logmsg log_level "$(pad_str)Trial point with (θ, Φ) = $((θxs, Φxs)) does$(iteration_type == FILTER_FAIL ? " not" : "") fit filter."
    @logmsg log_level "$(pad_str)The trial point is$(Int8(step_class) < 1 ? " not" : "") accepted."
    return nothing
end

function _log_trial_results(delta, delta_new, radius_update; indent, log_level)
    indent += 1
    pad_str = indent_str(indent)
    if Int8(radius_update) > 1
        @logmsg log_level "$(pad_str)Radius will be changed from $(delta) to $(delta_new)."
    else
        @logmsg log_level "$(pad_str)Radius will stay it $(delta)."
    end
    return nothing
end

function _update_radius(
    delta, step_class,
    gamma_grow, gamma_shrink, gamma_shrink_much, delta_max
)
    if step_class == ACCEPTABLE
        delta_new = gamma_shrink * Δ
        radius_update = SHRINK
    elseif step_class == SUCCESSFUL
        if delta_new < delta_max
            delta_new = min(gamma_grow * Δ, delta_max)
            radius_update = GROW
        else
            delta_new = delta
            radius_update = GROW_FAIL
        end
    else
        delta_new = gamma_shrink_much * delta
        radius_update = SHRINK
    end
    return delta_new, radius_update
end