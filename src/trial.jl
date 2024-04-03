function test_trial_point!(
    update_results, vals_tmp, Δ, mop, scaler, filter, vals, mod_vals, step_vals, algo_opts;
    it_index, indent::Int=0
)
    @unpack log_level = algo_opts
    pad_str = lpad("", indent)
    @logmsg log_level "$(pad_str)* Checking trial point."
    indent += 1

    @unpack diff_x, diff_fx, diff_fx_mod = update_results
    @unpack strict_acceptance_test, kappa_theta, psi_theta, nu_accept, nu_success = algo_opts

    # Use `vals_tmp` to hold the true values at `xs`:
    copyto!(cached_x(vals_tmp), step_vals.xs)
    @ignoraise eval_mop!(vals_tmp, mop, scaler)
    
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

    new_it_stat = test_trial_point!(
        diff_x, diff_fx, diff_fx_mod,
        filter, x, xs, fx, fxs, fx_mod, fxs_mod, θx, θxs, Φx, Φxs;
        strict_acceptance_test, kappa_theta, psi_theta, nu_accept, nu_success, log_level,
        it_index, indent
    )

    @unpack gamma_grow, gamma_shrink, gamma_shrink_much, delta_max = algo_opts
    finalize_update_results!(
        update_results, Δ, new_it_stat;
        point_has_changed = Int(new_it_stat) > 0,
        gamma_grow, gamma_shrink, gamma_shrink_much, delta_max
    )
    return nothing
end

function finalize_update_results!(
    update_results, Δ, new_it_stat, point_has_changed
)
    update_results.point_has_changed = point_has_changed
    update_results.it_stat = new_it_stat
    update_results.Δ_pre = update_results.Δ_post
    update_results.Δ_post = Δ

    update_results.norm2_x = LA.norm(update_results.diff_x)
    update_results.norm2_fx = LA.norm(update_results.diff_fx)
    update_results.norm2_fx_mod = LA.norm(update_results.diff_fx_mod)
    update_results.it_index += 1
    return nothing
end

function finalize_update_results!(
    update_results, Δ, new_it_stat
    ;
    point_has_changed::Bool, 
    gamma_grow::Real, 
    gamma_shrink::Real, 
    gamma_shrink_much::Real, 
    delta_max::Real
)
    finalize_update_results!(update_results, Δ, new_it_stat, point_has_changed)

    update_results.Δ_post = update_radius(
        Δ, new_it_stat; gamma_grow, gamma_shrink, gamma_shrink_much, delta_max)
    return nothing    
end

function test_trial_point!(
    diff_x, diff_fx, diff_fx_mod,
    # not modified
    filter, x, xs, fx, fxs, fx_mod, fxs_mod, θx, θxs, Φx, Φxs;
    strict_acceptance_test, kappa_theta, psi_theta, nu_accept, nu_success, log_level,
    it_index, indent::Int=0
)
    indent += 1
    pad_str = lpad("", indent)
    fits_filter = is_acceptable(filter, θxs, Φxs, θx, Φx)
    
    @. diff_x = x - xs
    @. diff_fx = fx - fxs
    @. diff_fx_mod = fx_mod .- fxs_mod

    new_it_stat = if !fits_filter
        @logmsg log_level "$(pad_str)Trial point with (θ, Φ) = $((θxs, Φxs)) does not fit filter."
        FILTER_FAIL
    else
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
        model_decrease_condition = all(model_decrease .>= kappa_theta * θx^psi_theta)
        sufficient_decrease_condition = rho >= nu_accept

        @logmsg log_level "$(pad_str)Trial point with (θ, Φ) = $((θxs, Φxs)) is filter-acceptable."
        if model_decrease_condition
            if !sufficient_decrease_condition
                @logmsg log_level "$(pad_str)Step INACCEPTABLE, ρ=$(rho)."
                INACCEPTABLE
            else
                if rho < nu_success
                    @logmsg log_level "$(pad_str)Step ACCEPTABLE, ρ=$(rho)."
                    ACCEPTABLE
                else
                    @logmsg log_level "$(pad_str)Step SUCCESSFUL, ρ=$(rho)."
                    SUCCESSFUL
                end
            end
        else
            if !sufficient_decrease_condition
                @logmsg log_level "$(pad_str)Model decrease insufficient and bad step, FILTER_ADD_SHRINK."
                FILTER_ADD_SHRINK
            else
                @logmsg log_level "$(pad_str)Model decrease insufficient, okay step, FILTER_ADD."
                FILTER_ADD
            end
        end
    end

    return new_it_stat
end

function update_radius(
    Δ, it_stat;
    gamma_grow, gamma_shrink, gamma_shrink_much, delta_max
)
    _Δ = if it_stat == FILTER_ADD_SHRINK || it_stat == INACCEPTABLE
        gamma_shrink_much * Δ
    elseif it_stat == ACCEPTABLE || it_stat == FILTER_FAIL
        gamma_shrink * Δ
    elseif it_stat == SUCCESSFUL
        min(gamma_grow * Δ, delta_max)
    else
        Δ
    end
    return _Δ
end