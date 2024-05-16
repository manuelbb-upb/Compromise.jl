
function optimize(
    MOP::AbstractMOP, ξ0::RVec;
    algo_opts :: AlgorithmOptions = AlgorithmOptions(),
    user_callback :: AbstractStoppingCriterion = NoUserCallback(),
)   
    return optimize_with_algo(MOP, algo_opts, ξ0; user_callback)
end

function optimize(
    MOP::AbstractMOP;
    algo_opts :: AlgorithmOptions = AlgorithmOptions(),
    user_callback :: AbstractStoppingCriterion = NoUserCallback(),
)
    ξ0 = initial_vars(MOP)
    
    @assert !isnothing(ξ0) "`optimize` called without initial variable vector."

    return optimize_with_algo(MOP, algo_opts, ξ0; user_callback)
end

function optimize_with_algo(
    MOP::AbstractMOP, algo_opts::AlgorithmOptions, ξ0; 
    user_callback :: AbstractStoppingCriterion = NoUserCallback(),
)
    optimizer_caches = initialize_structs(MOP, ξ0, algo_opts, user_callback)
    return optimize!(optimizer_caches, algo_opts, ξ0)
end

function optimize!(optimizer_caches::AbstractStoppingCriterion, algo_opts, ξ0)
    log_stop_code(optimizer_caches, algo_opts.log_level)
    return ReturnObject(ξ0, nothing, optimizer_caches)
end

function optimize!(optimizer_caches::OptimizerCaches, algo_opts, ξ0)
    @unpack (
        mop, mod, scaler, lin_cons, scaled_cons, vals, vals_tmp,
        mod_vals, filter, step_vals, step_cache, crit_cache, trial_caches, 
        iteration_status, iteration_scalars, stop_crits
    ) = optimizer_caches
    _ξ0, stop_code = optimize!(
        mop, mod, scaler, lin_cons, scaled_cons, vals, vals_tmp,
        mod_vals, filter, step_vals, step_cache, crit_cache, trial_caches, 
        iteration_status, iteration_scalars, stop_crits,
        algo_opts
    )
    return ReturnObject(_ξ0, optimizer_caches, stop_code)
end

function optimize!(
    mop, mod, scaler, lin_cons, scaled_cons, vals, vals_tmp,
    mod_vals, filter, step_vals, step_cache, crit_cache, trial_caches, 
    iteration_status, iteration_scalars, stop_crits,
    algo_opts
)
     
    _ξ0 = copy(cached_ξ(vals))    # for final ReturnObject

    stop_code = nothing
    while true
        time_start = time()
        @ignorebreak stop_code = do_iteration!(
            mop, mod, scaler, lin_cons, scaled_cons, vals, vals_tmp,
            mod_vals, filter, step_vals, step_cache, crit_cache, trial_caches, 
            iteration_status, iteration_scalars, stop_crits,
            algo_opts
        )
        time_stop = time()
        @logmsg algo_opts.log_level "Iteration was $(time_stop - time_start) sec."
    end
    log_stop_code(stop_code, algo_opts.log_level)
    return _ξ0, stop_code
end

function do_iteration!(
    mop, mod, scaler, lin_cons, scaled_cons, vals, vals_tmp,
    mod_vals, filter, step_vals, step_cache, crit_cache, trial_caches, 
    iteration_status, iteration_scalars, stop_crits,
    algo_opts
)
    indent = 0

    @unpack log_level = algo_opts
    
    iteration_scalars.it_index += 1
    @unpack it_index, delta = iteration_scalars

    @ignoraise check_stopping_criterion(
        stop_crits, CheckPreIteration(),
        mop, mod, scaler, lin_cons, scaled_cons, vals, vals_tmp,
        mod_vals, filter, step_vals, step_cache, crit_cache, trial_caches, 
        iteration_status, iteration_scalars, stop_crits,
        algo_opts
    ) indent

    @logmsg log_level """\n
    ###########################
    #  ITERATION $(it_index).
    ###########################
    Δ = $(delta)
    θ = $(cached_theta(vals))
    ξ = $(pretty_row_vec(cached_ξ(vals)))
    x = $(pretty_row_vec(cached_x(vals)))
    fx = $(pretty_row_vec(cached_fx(vals)))
    """
    ## The models are valid
    ## - the last iteration was a successfull restoration iteration.
    ## - if the point has not changed and the models do not depend on the radius or  
    ##   radius has not changed neither.
    radius_has_changed = Int8(iteration_status.radius_update_result) >= 0
    models_valid = if iteration_status.iteration_type == INITIALIZATION
        false
    else
        if iteration_status.iteration_type == RESTORATION
            true
        elseif (
            !_trial_point_accepted(iteration_status) && 
            !( depends_on_radius(mod) && radius_has_changed )
        )
            true
        else
            false
        end
    end

    if !models_valid
        @logmsg log_level "* Updating Surrogates."
        @ignoraise update_models!(mod, delta, scaler, vals, scaled_cons; log_level, indent) indent
        @ignoraise eval_and_diff_mod!(mod_vals, mod, cached_x(vals)) indent
    end
    @ignoraise do_normal_step!(
        step_cache, step_vals, delta, mop, mod, scaler, lin_cons, scaled_cons, vals, mod_vals;
        log_level, indent=0
    ) indent

    n_is_compatible = compatibility_test(step_vals.n, algo_opts, delta)

    if !n_is_compatible
        ## Try to do a restoration
        @logmsg log_level "* Normal step incompatible. Trying restoration."
        add_to_filter!(filter, vals)
        @ignoraise do_restoration(
            mop, mod, scaler, lin_cons, scaled_cons, vals, vals_tmp,
            mod_vals, filter, step_vals, step_cache, crit_cache, trial_caches, 
            iteration_status, iteration_scalars, stop_crits,
            algo_opts
        ) indent
        return nothing
    end

    @logmsg log_level "* Computing a descent step."
    @ignoraise do_descent_step!(
        step_cache, step_vals, delta, mop, mod, scaler, lin_cons, scaled_cons, vals, mod_vals;
        log_level
    ) indent
    
    @logmsg log_level " - Criticality χ=$(step_vals.crit_ref[]), ‖d‖₂=$(LA.norm(step_vals.d)), ‖s‖₂=$(LA.norm(step_vals.s))."

    @ignoraise check_stopping_criterion(
        stop_crits, CheckPostDescentStep(), 
        mop, mod, scaler, lin_cons, scaled_cons, vals, vals_tmp,
        mod_vals, filter, step_vals, step_cache, crit_cache, trial_caches, 
        iteration_status, iteration_scalars, stop_crits,
        algo_opts
    ) indent
    ## For the convergence analysis to work, we also have to have the Criticality Routine:
    @ignoraise delta_new = criticality_routine!(
        mop, mod, scaler, lin_cons, scaled_cons, vals, vals_tmp,
        mod_vals, filter, step_vals, step_cache, crit_cache, trial_caches, 
        iteration_status, iteration_scalars, stop_crits,
        algo_opts;
        indent
    ) indent
    iteration_scalars.delta = delta_new
    delta = iteration_scalars.delta
    
    ## test if trial point is acceptable
    @ignoraise delta_new = test_trial_point!(
        mop, mod, scaler, lin_cons, scaled_cons, vals, vals_tmp,
        mod_vals, filter, step_vals, step_cache, crit_cache, trial_caches, 
        iteration_status, iteration_scalars, stop_crits,
        algo_opts;
        indent
    ) indent
    iteration_scalars.delta = delta_new
    delta = iteration_scalars.delta
    
    @ignoraise check_stopping_criterion(
        stop_crits, CheckPostIteration(),
        mop, mod, scaler, lin_cons, scaled_cons, vals, vals_tmp,
        mod_vals, filter, step_vals, step_cache, crit_cache, trial_caches, 
        iteration_status, iteration_scalars, stop_crits,
        algo_opts
    ) indent

    @ignoraise process_trial_point!(mod, vals_tmp, iteration_status) indent

    ## update filter
    if iteration_status.iteration_type == THETA_STEP
        add_to_filter!(filter, vals)
    end
   
    if _trial_point_accepted(iteration_status)
        ## accept trial point, mathematically ``xₖ₊₁ ← xₖ + sₖ``, pseudo-code `copyto!(vals, vals_tmp)`: 
        universal_copy!(vals, vals_tmp)
    end
    
    return nothing
end