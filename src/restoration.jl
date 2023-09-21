function do_restoration(
    it_index, Δ, it_stat, mop, mod, scaler, lin_cons, scaled_cons,
    vals, vals_tmp, step_vals, mod_vals, filter, step_cache, algo_opts
)
    @info "Starting restoration."
	(θ_opt, xr_opt, ret) = solve_restoration_problem(mop, scaled_cons, vals.x, algo_opts.nl_opt)
    @show xr_opt, ret
    if ret in (
        :SUCCESS, 
        :STOPVAL_REACHED, 
        :FTOL_REACHED, 
        :XTOL_REACHED, 
        :MAXEVAL_REACHED,
		:MAXTIME_REACHED
    )
        return postproccess_restoration(
            it_index, Δ, it_stat, mop, mod, scaler, lin_cons, scaled_cons,
            vals, vals_tmp, step_vals, mod_vals, filter, step_cache, algo_opts;
            xr_opt
        )   
    end
    return RESTORATION, INFEASIBLE, false, Δ
end


function restoration_objective(mop, scaled_cons)
    function objf(xr::Vector, grad::Vector)
        if !isempty(grad)
            @error "Restoration only supports derivative-free NLopt algorithms."
        end

        hx = prealloc_nl_eq_constraints_vector(mop)
        gx = prealloc_nl_ineq_constraints_vector(mop)
        Eres = prealloc_lin_eq_constraints_vector(mop)
        Ares = prealloc_lin_ineq_constraints_vector(mop)
        Ex = deepcopy(Eres)
        Ax = deepcopy(Ares)
        
        nl_eq_constraints!(hx, mop, xr)
        nl_ineq_constraints!(gx, mop, xr)
        lin_cons!(Eres, Ex, scaled_cons.Ab, xr)
        lin_cons!(Ares, Ax, scaled_cons.Ec, xr)
        
        return constraint_violation(hx, gx, Eres, Ares)
    end
    return objf
end

function solve_restoration_problem(mop, scaled_cons, x, nl_opt)
    n_vars = length(x)

	opt = NLopt.Opt(nl_opt, n_vars)

	opt.min_objective = restoration_objective(mop, scaled_cons)

	if !isnothing(scaled_cons.lb)
        opt.lower_bounds = scaled_cons.lb
    end
    if !isnothing(scaled_cons.ub)
        opt.upper_bounds = scaled_cons.ub
    end
	
    #src # TODO make some of these settings configurable
    opt.xtol_rel = 1e-4
	opt.maxeval = 50 * n_vars^2
	opt.stopval = 0

    xr0 = deepcopy(x)   # this allocation should not matter too much
	return NLopt.optimize(opt, xr0)
end

function postproccess_restoration(
    it_index, Δ, it_stat, mop, mod, scaler, lin_cons, scaled_cons,
    vals, vals_tmp, step_vals, mod_vals, filter, step_cache, algo_opts;
    xr_opt
)
    ## the nonlinear problem was solved successfully.
    ## pretend, trial point was next iterate and set values
    ## (it doesn't matter if we use `vals` or `vals_tmp`: in case of successfull restoration
    ##  we have to set `vals` eventually. in case of unsuccessfull restoration, we have to 
    ##  abort anyways)
    copyto!(vals.x, xr_opt)
    eval_mop!(vals, mop, scaler)

    ## the next point should be acceptable for the filter:
    if is_acceptable(filter, vals.θ[], vals.Φ[])
        ## make models valid at `x + r` and set model values
        ## also compute normal step based on models
        update_models!(mod, Δ, mop, scaler, vals, scaled_cons, algo_opts; point_has_changed=true)
        eval_and_diff_mod!(mod_vals, mod, vals.x)

        compute_normal_step(
            it_index, Δ, it_stat, mop, mod, scaler, lin_cons, scaled_cons,
            vals, vals_tmp, step_vals, mod_vals, filter, step_cache, algo_opts
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
            _Δ2 = (_Δ1/c_mu)^(1/1+mu)
            _Δ = max(_Δ1, _Δ2)
            if _Δ <= delta_max
                n_is_compatible = true
            end                
        end
        if n_is_compatible
            accept_trial_point!(vals, vals_tmp)
            return RESTORATION, CONTINUE, true, _Δ
        end
    end
    return RESTORATION, INFEASIBLE, false, Δ
end