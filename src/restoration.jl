# As of now, the restoration procedure constitutes an entire iteration;
# we have to break/continue afterwards.
# In case of success, we first check `CheckPostIteration` stopping criteria, assuming
# a zero descent step, and afterwards set `vals` and `iteration_scalars.delta` for the 
# subsequent (non-restoration) iteration.
# In that subsequent iteration, we are guaranteed to find a compatible normal step
# with the current `mod` for `iteration_scalars.delta`.


function do_restoration(
    mop, mod, scaler, lin_cons, scaled_cons, vals, vals_tmp,
    mod_vals, filter, step_vals, step_cache, crit_cache, trial_caches, 
    iteration_status, iteration_scalars, stop_crits,
    algo_opts;
    indent =  1,
    solve_nl_subproblem=true    # TODO deprecate
)
    @unpack log_level = algo_opts
    indent += 1
    pad_str = lpad("", indent)
    @logmsg log_level "$(pad_str)Starting restoration."
    
    iteration_status.iteration_type = RESTORATION   # set this to avoid model updates in subsequent iteration of the main algo

    universal_copy!(vals_tmp, vals) # we use `cached_x(vals_tmp)` as a starting point for nonlinear restoration
                                    # this array might be changed by a prior check to see whether `step_vals.xn` 
                                    # is a suitable point.

    @ignoraise Δ = restore_with_normal_step!(
        mop, mod, scaler, lin_cons, scaled_cons, vals, vals_tmp,
        mod_vals, filter, step_vals, step_cache, crit_cache, trial_caches, 
        iteration_status, iteration_scalars, stop_crits,
        algo_opts;
        indent
    )

    if isnothing(Δ)
        if (dim_nl_eq_constraints(mop) > 0 || dim_nl_ineq_constraints(mop) > 0) &&  solve_nl_subproblem
            @ignoraise Δ = restore_nonlinearly!(
                mop, mod, scaler, lin_cons, scaled_cons, vals, vals_tmp,
                mod_vals, filter, step_vals, step_cache, crit_cache, trial_caches, 
                iteration_status, iteration_scalars, stop_crits,
                algo_opts;
                indent
            )
        else
            # there are only linear constraints, so the normal step should have been valid
            # in case of compatible constraints
            return InfeasibleStopping()
        end
    end
    
    if isnothing(Δ)
        return InfeasibleStopping()
    end
    @logmsg log_level "$(pad_str)Restoration worked!"
    # If we are here, then restoration was successfull
    iteration_scalars.delta = Δ # set next radius
    @ignoraise check_stopping_criterion(
        stop_crits, CheckPostIteration(), 
        mop, mod, scaler, lin_cons, scaled_cons, vals, vals_tmp,
        mod_vals, filter, step_vals, step_cache, crit_cache, trial_caches, 
        iteration_status, iteration_scalars, stop_crits,
        algo_opts
    ) indent
    universal_copy!(vals, vals_tmp)
    return nothing
end

function restore_with_normal_step!(
    mop, mod, scaler, lin_cons, scaled_cons, vals, vals_tmp,
    mod_vals, filter, step_vals, step_cache, crit_cache, trial_caches, 
    iteration_status, iteration_scalars, stop_crits,
    algo_opts;
    indent=0
)
    @unpack log_level = algo_opts
    check_for_compatibility = false
    xr_opt = cached_x(vals_tmp)
    if !any(isnan.(step_vals.n))
        @logmsg log_level "$(indent_str(indent)) Trying `xn` as next iterate."
        copyto!(xr_opt, step_vals.xn)
        @ignoraise eval_mop!(vals_tmp, mop, scaler)
        if is_filter_acceptable(filter, vals_tmp)
            check_for_compatibility = true
        else
            @logmsg log_level "$(indent_str(indent)) Not filter acceptable (θ = $(cached_theta(vals_tmp)))..."
        end
    end

    if check_for_compatibility
        @ignoraise Δ = postproccess_restoration(
            xr_opt, iteration_scalars.delta, trial_caches, mop, mod, scaler, lin_cons, scaled_cons,
            vals, vals_tmp, step_vals, mod_vals, filter, step_cache, algo_opts;
            indent, 
            evaluate_mop = false    # vals_tmp already is valid for xr_opt
        ) indent
        return Δ
    end
    return nothing
end

function restore_nonlinearly!(
    mop, mod, scaler, lin_cons, scaled_cons, vals, vals_tmp,
    mod_vals, filter, step_vals, step_cache, crit_cache, trial_caches, 
    iteration_status, iteration_scalars, stop_crits,
    algo_opts;
    indent=0
)
    @unpack log_level = algo_opts
    @logmsg log_level "$(indent_str(indent)) Trying to solve nonlinear subproblem."
    x0 = cached_x(vals_tmp)
    theta = cached_theta(vals_tmp)
    (θ_opt, xr_opt, ret) = solve_restoration_problem(
        mop, vals_tmp, scaler, scaled_cons, x0, theta;
    )
    if ret in (
        :SUCCESS, 
        :STOPVAL_REACHED, 
        :FTOL_REACHED, 
        :XTOL_REACHED, 
        :MAXEVAL_REACHED,
		:MAXTIME_REACHED
    )
        @ignoraise Δ = postproccess_restoration(
            xr_opt, iteration_scalars.delta, trial_caches, mop, mod, scaler, lin_cons, scaled_cons,
            vals, vals_tmp, step_vals, mod_vals, filter, step_cache, algo_opts;
            indent,
            evaluate_mop = true
        ) indent
        return Δ
    end
    return nothing
end

function restoration_objective()
    return function(tx, grads)
        return first(tx)
    end
end

function restoration_constraints(mop, vals_tmp, scaler, scaled_cons)
    @unpack A, b, E, c = scaled_cons
    
    function (constr_vec, txr::Vector, grads)
        if !isempty(grads)
            error("Restoration only supports derivative-free NLopt algorithms.")
        end
        t = first(txr)
        xr = @view(txr[2:end])

        ξ = cached_ξ(vals_tmp)
        unscale!(ξ, scaler, xr)
        stop_code = nl_eq_constraints!(cached_hx(vals_tmp), mop, ξ)
        if !isa(stop_code, AbstractStoppingCriterion)
            stop_code = nl_ineq_constraints!(cached_gx(vals_tmp), mop, ξ)
        end
        if isa(stop_code, AbstractStoppingCriterion)
            @error("Stopped in NLopt.")
            throw(NLopt.ForcedStop())
        end

        lin_cons!(cached_Ex_min_c(vals_tmp), cached_Ex(vals_tmp), E, c, xr)
        lin_cons!(cached_Ax_min_b(vals_tmp), cached_Ax(vals_tmp), A, b, xr)
        
        gx = cached_gx(vals_tmp)
        i = 1
        j = length(gx)
        constr_vec[i:j] .= gx .- t
        
        i = j + 1
        hx = cached_hx(vals_tmp)
        j += length(hx)
        constr_vec[i:j] .= hx .- t
        i = j + 1
        j += length(hx)
        constr_vec[i:j] .= -hx .- t

        rineq = cached_Ax_min_b(vals_tmp)
        i = j + 1
        j += length(rineq)
        constr_vec[i:j] .= rineq .- t

        req = cached_Ex_min_c(vals_tmp)
        i = j + 1
        j += length(req)
        constr_vec[i:j] .= req .- t
        i = j + 1
        j += length(req)
        constr_vec[i:j] .= -req .- t

       return constr_vec
    end
end

function solve_restoration_problem(mop, vals_tmp, scaler, scaled_cons, x, theta; stopval=0)
    n_vars = length(x)

	opt = NLopt.Opt(:LN_COBYLA, n_vars + 1)

	opt.min_objective = restoration_objective()

    l = min(-1, -2*abs(theta))
    u = max(1, 2*abs(theta))
    F = float_type(mop)
    lb = array(F, n_vars + 1)
    ub = array(F, n_vars + 1)
    lb .= -Inf
    ub .= Inf
    lb[1] = l
    ub[1] = u
	if !isnothing(scaled_cons.lb)
        lb[2:end] .= scaled_cons.lb
    end
    if !isnothing(scaled_cons.ub)
        ub[2:end] .= scaled_cons.ub
    end
    opt.lower_bounds = lb
    opt.upper_bounds = ub
	
    #src # TODO make some of these settings configurable ?
    tol = sqrt(eps(F))
    opt.xtol_rel = tol
    opt.xtol_abs = eps(F)
	opt.maxeval = 50 * n_vars^2
	opt.stopval = max(stopval - tol, 0)

    constr! = restoration_constraints(mop, vals_tmp, scaler, scaled_cons)
    n_constr = dim_nl_ineq_constraints(mop) + 2 * dim_nl_eq_constraints(mop) + dim_lin_ineq_constraints(mop) + 2 * dim_lin_eq_constraints(mop)
    NLopt.inequality_constraint!(opt, constr!, fill(tol, n_constr))

    txr0 = min.(ub, max.(lb, [theta; x]))
	optf, opt_tx, ret = NLopt.optimize(opt, txr0)
    return (optf, opt_tx[2:end], ret)
end

#=
function restoration_objective(mop, vals_tmp, scaler, scaled_cons)
    @unpack A, b, E, c = scaled_cons
    function objf(xr::Vector, grad::Vector)
        if !isempty(grad)
            error("Restoration only supports derivative-free NLopt algorithms.")
        end

        ξ = cached_ξ(vals_tmp)
        unscale!(ξ, scaler, xr)
        @ignoraise nl_eq_constraints!(cached_hx(vals_tmp), mop, ξ)
        @ignoraise nl_ineq_constraints!(cached_gx(vals_tmp), mop, ξ)
        lin_cons!(cached_Ex_min_c(vals_tmp), cached_Ex(vals_tmp), E, c, xr)
        lin_cons!(cached_Ax_min_b(vals_tmp), cached_Ax(vals_tmp), A, b, xr)
        
        theta = constraint_violation(
            cached_hx(vals_tmp), 
            cached_gx(vals_tmp), 
            cached_Ex_min_c(vals_tmp), 
            cached_Ax_min_b(vals_tmp)
        )
        return theta
    end
    return objf
end

function solve_restoration_problem(mop, vals_tmp, scaler, scaled_cons, x, nl_opt)
    n_vars = length(x)

	opt = NLopt.Opt(:LN_COBYLA, n_vars)

	opt.min_objective = restoration_objective(mop, vals_tmp, scaler, scaled_cons)

	if !isnothing(scaled_cons.lb)
        opt.lower_bounds = scaled_cons.lb
    end
    if !isnothing(scaled_cons.ub)
        opt.upper_bounds = scaled_cons.ub
    end
	
    #src # TODO make some of these settings configurable
    opt.xtol_rel = sqrt(eps(eltype(x)))
    opt.xtol_abs = eps(eltype(x))
	opt.maxeval = 50 * n_vars^2
	opt.stopval = 0

    xr0 = deepcopy(x)   # this allocation should not matter too much
	return NLopt.optimize(opt, xr0)
end
=#
function postproccess_restoration(
    xr_opt, Δ, trial_caches, mop, mod, scaler, lin_cons, scaled_cons,
    vals, vals_tmp, step_vals, mod_vals, filter, step_cache, algo_opts;
    indent,
    evaluate_mop = true     # make `vals_tmp` valid for `xr_opt`, `mop` and `scalar`
)
    @unpack log_level = algo_opts

    ## the nonlinear problem was solved successfully.
    ## pretend, trial point was next iterate and set values

    if evaluate_mop
        copyto!(cached_x(vals_tmp), xr_opt)
        @ignoraise eval_mop!(vals_tmp, mop, scaler)
    end

    @logmsg log_level "$(indent_str(indent))Checking restoration result with θ=$(cached_theta(vals_tmp))."

    trial_caches.diff_x .= cached_x(vals) .- cached_x(vals_tmp)
    trial_caches.diff_fx .= cached_fx(vals) .- cached_fx(vals_tmp)
    trial_caches.diff_fx_mod .= cached_fx(mod_vals)

    ## the next point should be acceptable for the filter:
    if is_filter_acceptable(filter, vals_tmp)
        ## make models valid at `x + r` and set model values
        ## also compute normal step based on models
        @logmsg log_level "$(indent_str(indent+1))Surrogates need update."
        @ignoraise update_models!(mod, Δ, scaler, vals_tmp, scaled_cons; log_level, indent=indent+1)
        @ignoraise eval_and_diff_mod!(mod_vals, mod, cached_x(vals_tmp))

        trial_caches.diff_fx_mod .-= cached_fx(mod_vals)

        @ignoraise do_normal_step!(
            step_cache, step_vals, Δ, mop, mod, scaler, lin_cons, scaled_cons, vals_tmp, mod_vals;
            log_level, indent
        )

        @unpack c_delta, c_mu, mu, delta_max = algo_opts
        n = step_vals.n
        norm_n = LA.norm(n, Inf)
        n_is_compatible = norm_n <= compatibility_test_rhs(c_delta, c_mu, mu, Δ)
        
        ## if the normal step is compatible, we are done and can use vals_tmp as the next
        ## iterate with next radius `_Δ=Δ`.
        _Δ = Δ

        ## if it is not compatible, we might try to increase the radius:
        if !n_is_compatible
            ## a) ‖n‖ ≤ c Δ         ⇔ ‖n‖/c ≤ Δ 
            ## b) ‖n‖ ≤ c κ Δ^{1+μ} ⇔ (‖n‖/(cκ))^{1/(1+μ)} ≤ Δ
            ## ⇒ Δ ≥ max(‖n‖/c, (‖n‖\(cκ))^{1/(1+μ)})).
            _Δ1 = norm_n/c_delta
            _Δ2 = (_Δ1/c_mu)^(1/(1+mu))
            _Δ = max(_Δ1, _Δ2)
            norm_n <= compatibility_test_rhs(c_delta, c_mu, mu, _Δ)
            if _Δ <= delta_max
                n_is_compatible = true
                @logmsg log_level "$(indent_str(indent))Radius is changed to $(_Δ)."
            end                
        end
        if n_is_compatible
            step_vals.n .= cached_x(vals_tmp) .- cached_x(vals)
            @. step_vals.d = 0
            @. step_vals.s = step_vals.n

            return _Δ
        end
    end
    # if we have not returned here, then no compatible normal step was found :(
    return InfeasibleStopping()
end