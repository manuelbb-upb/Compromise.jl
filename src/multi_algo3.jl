include("multi_filters.jl")

function optimize_set(
    X::RMat, 
    mop;
    algo_opts = AlgorithmOptions(), 
    user_callback = NoUserCallback(),
    filter_gamma = 1e-6,
)
    @unpack log_level, max_iter = algo_opts

    optimizer_caches = initialize_structs(mop, @view(X[:, 1]), algo_opts, user_callback; log_time=false)

    if optimizer_caches isa AbstractStoppingCriterion
        return optimizer_caches
    end
    
    @unpack (
        mop, mod, scaler, lin_cons, scaled_cons, vals, vals_tmp,
        mod_vals, step_vals, step_cache, crit_cache, trial_caches, 
        iteration_status, iteration_scalars,
    ) = optimizer_caches

    @unpack stop_delta_min = algo_opts
    stop_crits = MinimumRadiusStopping(; delta_min = stop_delta_min)
    @reset optimizer_caches.stop_crits = stop_crits

    F = float_type(mop)
    prepopulation = initialize_prepopulation(F, X, mop, scaler, lin_cons, vals; log_level)   
    
    sol, population = initialize_population(
        prepopulation, mod_vals, step_vals, step_cache, iteration_scalars, mop, mod, stop_crits
    )

    ndset = initialize_filter(F, sol, filter_gamma)

    optimize_population!(population, ndset, optimizer_caches, algo_opts)
    return population
end

function initialize_filter(F, sol, gamma)
    tmp = make_filter_element(sol)
    return _initialize_filter(F, tmp, gamma)
end
function _initialize_filter(::Type{F}, tmp::tmpType, gamma) where {F, tmpType}
    return AugmentedVectorFilter{F, AugmentedVectorElement{F}, tmpType}(;
        print_name="Filter", gamma, tmp)
end

function initialize_population(
    prepopulation, mod_vals, step_vals, step_cache, iteration_scalars, mop, mod, stop_crits
)
    ST = _status_type(mop, mod, stop_crits)
    vals = getfield(first(nondominated_elements(prepopulation)), :vals)
    sol = make_solution(
        vals, step_vals, step_cache, iteration_scalars, mod, mod_vals, ST
    )
    population = singleton_population(float_type(mop), sol)
    for elem in Iterators.drop(nondominated_elements(prepopulation), 1)
        sol = make_solution(
            elem.vals, step_vals, step_cache, iteration_scalars, mod, mod_vals, ST)
        unconditionally_add_to_set!(population, sol)
    end
    return sol, population
end

function initialize_prepopulation(F, X, mop, scaler, lin_cons, vals; log_level=Info)
    prepopulation = make_prepopulation(F, vals; log_level)
    return add_columns_to_prepulation!(prepopulation, X, mop, scaler, lin_cons, vals; log_level)
end

function make_prepopulation(::Type{F}, vals::valsType; log_level=Info) where {F, valsType}
    prepopulation = AugmentedVectorFilter{F, ValueCacheElement{valsType}, Nothing}()
    add_to_set!(prepopulation, ValueCacheElement(; vals); log_level)
    return prepopulation
end

function add_columns_to_prepulation!(prepopulation, X, mop, scaler, lin_cons, vals; offset=1, log_level=Info)
    for ξ0 in Iterators.drop(eachcol(X), offset)
        _vals = deepcopy(vals)
        copyto!(cached_ξ(_vals), ξ0)
        project_into_box!(cached_ξ(_vals), lin_cons)
        scale!(cached_x(_vals), scaler, cached_ξ(_vals))
        eval_mop!(_vals, mop)
        add_to_set!(prepopulation, ValueCacheElement(; vals=_vals); log_level)
    end
    return prepopulation
end

function make_solution(
    vals, step_vals, step_cache, iteration_scalars, mod, mod_vals, ::Type{status_type}=Any,
    gen_id::Int=1,
    sol_id::Int=1
) where {status_type}
    sstructs = SolutionStructs(
        vals, 
        deepcopy(step_vals), 
        deepcopy(step_cache), 
        deepcopy(iteration_scalars), 
        deepcopy(mod), 
        deepcopy(mod_vals),
        Ref{status_type}(nothing),
        Ref(gen_id), 
        Ref(gen_id)
    )
    return sstructs
end

function singleton_population(::Type{F}, sol::S) where {F, S}
    population = AugmentedVectorFilter{F, S, Nothing}(; print_name="Population")
    unconditionally_add_to_set!(population, sol)
    return population
end

function _status_type(mop, mod, stop_crits)
    unwrapped_stype = Union{
        stop_type(mod), 
        stop_type(mop), 
        stop_crit_type(stop_crits),
        MaxIterStopping,
        InfeasibleStopping
    }
    return Union{Nothing, unwrapped_stype, WrappedStoppingCriterion{<:unwrapped_stype}}
end

function optimize_population!(
    population, ndset, optimizer_caches, algo_opts;
    check_filter=true
)
    @unpack max_iter = algo_opts
    @unpack log_level = algo_opts
    
    max_iter_stop_crit = MaxIterStopping(max_iter)
        restoration_seeds = AugmentedVectorFilter(;
        elems = empty(population.elems),
        gamma = population.gamma,
        tmp = deepcopy(population.tmp)
    )
    
    stop_optimization = false
    max_gen_id = 1

    while !stop_optimization
        remove_stale_elements!(population)  # optional, we should do some benchmarks ...
        
        sol, population_empty, _ , _max_gen_id = select_solution(population)
        if _max_gen_id >= max_gen_id
            max_gen_id = _max_gen_id    # for setting an id in restoration
        end
        if isnothing(sol)
            if population_empty
                sol, status = restore!(
                    population, restoration_seeds, ndset, optimizer_caches, algo_opts; 
                    max_gen_id
                )
                if status isa AbstractStoppingCriterion
                    log_stop_code(status, log_level)
                    mark_stale!(population, sol)
                    sol.status_ref[] = status
                    stop_optimization = true
                end
            else
                stop_optimization = true
            end
            continue
        else
            if sol.gen_id_ref[] > max_iter
                sol.status_ref[] = max_iter_stop_crit
                continue
            end
            sol.gen_id_ref[] += 1
            if check_filter
                if !is_filter_acceptable(ndset, sol)
                    mark_stale!(population, sol)
                    continue
                end
            end
            n_is_compatible = step_normally!(sol, ndset, optimizer_caches, algo_opts)
            if n_is_compatible isa AbstractStoppingCriterion
                sol.status_ref[] = n_is_compatible
                if unwrap_stop_crit(n_is_compatible) isa AbstractUltimateStoppingCriterion
                    stop_optimization = true
                end
                continue
            end

            if !n_is_compatible
                @logmsg log_level "… normal step not compatible. 1/2) Augmenting filter."
                add_to_filter_and_check_population!(ndset, population, sol; log_level)

                @logmsg log_level "2/2) Storing seed."
                add_to_set!(restoration_seeds, sol; log_level)
                remove_stale_elements!(restoration_seeds)
                continue
            end

            status = step_tangentially!(sol, ndset, optimizer_caches, algo_opts)
            if status isa AbstractStoppingCriterion
                sol.status_ref[] = status
                if unwrap_stop_crit(status) isa AbstractUltimateStoppingCriterion
                    stop_optimization = true
                end
                continue
            end
            status = test_trial_point!(sol, ndset, population, optimizer_caches, algo_opts)
            if status isa AbstractStoppingCriterion
                sol.status_ref[] = status
                if unwrap_stop_crit(status) isa AbstractUltimateStoppingCriterion
                    stop_optimization = true
                end
                continue
            end
        end
    end
    remove_stale_elements!(population)

    return population
end

function select_solution(population)
    next_sol = nothing
    is_empty = true
    max_theta = -Inf
    max_delta = -Inf
    min_gen_id, max_gen_id, _max_gen_id = max_gen_id_for_selection(population) 
    for sol in all_elements(population)
        is_stale(population, sol) && continue
        is_empty = false
        is_converged(sol) && continue
        sol.gen_id_ref[] > _max_gen_id && continue
        theta = cached_theta(sol)
        @unpack delta = sol.iteration_scalars
        if theta >= max_theta
            if (theta > max_theta || delta > max_delta)
                next_sol = sol
                max_theta = theta
                max_delta = delta 
            end
        end
    end
    return next_sol, is_empty, min_gen_id, max_gen_id
end

function max_gen_id_for_selection(population)
    min_gen_id, max_gen_id = gen_id_interval(population)
    gen_spread = max_gen_id - min_gen_id
    if gen_spread <= 3
        return min_gen_id, max_gen_id, max_gen_id
    end
    return min_gen_id, max_gen_id, min_gen_id + 3
end

function gen_id_interval(population)
    min_gen_id = typemax(Int)
    max_gen_id = typemin(Int)
    for sol in nondominated_elements(population)
        is_converged(sol) && continue
        sid = sol.gen_id_ref[]
        if sid <= min_gen_id
            min_gen_id = sid
        end
        if sid >= max_gen_id
            max_gen_id = sid
        end
    end
    return min_gen_id, max_gen_id
end

function step_tangentially!(sol, ndset, optimizer_caches, algo_opts; indent=0)
    @unpack log_level = algo_opts
    @unpack vals, step_vals, step_cache, iteration_scalars, mod, mod_vals = sol
    @unpack (
        stop_crits, mop, scaler, lin_cons, scaled_cons, vals_tmp, 
        crit_cache, trial_caches, iteration_status,
    ) = optimizer_caches
    @unpack delta = iteration_scalars
    
    @logmsg log_level "* Computing a descent step."
    
    do_descent_step!(
        step_cache, step_vals, delta, mop, mod, scaler, lin_cons, scaled_cons, vals, mod_vals;
        log_level
    )

    @logmsg log_level " - Criticality χ=$(step_vals.crit_ref[]), ‖d‖₂=$(LA.norm(step_vals.d)), ‖s‖₂=$(LA.norm(step_vals.s))."

    @ignoraise check_stopping_criterion(
        stop_crits, CheckPostDescentStep(), 
        mop, mod, scaler, lin_cons, scaled_cons, vals, vals_tmp,
        mod_vals, ndset, step_vals, step_cache, crit_cache, trial_caches, 
        iteration_status, iteration_scalars, stop_crits,
        algo_opts
    )
    ## For the convergence analysis to work, we also have to have the Criticality Routine:
    @ignoraise delta_new = criticality_routine!(
        mop, mod, scaler, lin_cons, scaled_cons, vals, vals_tmp,
        mod_vals, ndset, step_vals, step_cache, crit_cache, trial_caches, 
        iteration_status, iteration_scalars, stop_crits,
        algo_opts;
        indent = indent+1
    )
    delta = iteration_scalars.delta = delta_new
    return nothing
end

function step_normally!(sol, ndset, optimizer_caches, algo_opts; indent=0)
    @unpack log_level = algo_opts
    @unpack vals, step_vals, step_cache, iteration_scalars, mod, mod_vals = sol
    @unpack (
        stop_crits, mop, scaler, lin_cons, scaled_cons, vals_tmp, 
        crit_cache, trial_caches, iteration_status,
    ) = optimizer_caches
    @unpack delta = iteration_scalars
    @ignoraise check_stopping_criterion(
        stop_crits, CheckPreIteration(),
        mop, mod, scaler, lin_cons, scaled_cons, vals, vals_tmp,
        mod_vals, ndset, step_vals, step_cache, crit_cache, trial_caches, 
        iteration_status, iteration_scalars, stop_crits,
        algo_opts
    ) indent
   
    log_solution_values(sol, log_level)

    @logmsg log_level "* Updating Surrogates."
    
    @ignoraise update_models!(mod, delta, scaler, vals, scaled_cons; log_level, indent) indent
    @ignoraise eval_and_diff_mod!(mod_vals, mod, cached_x(vals)) indent

    do_normal_step!(
        step_cache, step_vals, delta, mop, mod, scaler, lin_cons, scaled_cons, vals, mod_vals;
        log_level, indent=0
    )
    return compatibility_test(step_vals.n, algo_opts, delta)
end

function log_solution_values(sstructs, log_level; normal_step=false)
    delta = sstructs.iteration_scalars.delta
    sol_id = sstructs.sol_id_ref[]
    gen_id = sstructs.gen_id_ref[]
    msg = """
        \n###########################
        # sol_id = $(sol_id)
        # gen_id = $(gen_id)
        ###########################
        Δ = $(delta)
        θ = $(cached_theta(sstructs))
        ξ = $(pretty_row_vec(cached_ξ(sstructs)))
        x = $(pretty_row_vec(cached_x(sstructs)))
        fx= $(pretty_row_vec(cached_fx(sstructs)))"""
    if normal_step
        @unpack step_vals = sstructs
        msg *= """
            \nn = $(pretty_row_vec(step_vals.n))
            xn= $(pretty_row_vec(step_vals.xn))"""
    end

    @logmsg log_level msg
    return nothing
end

function test_trial_point!(sol, ndset, population, optimizer_caches, algo_opts; indent=0)
    @unpack log_level = algo_opts
    @unpack vals, step_vals, step_cache, iteration_scalars, mod, mod_vals = sol
    @unpack (
        stop_crits, mop, scaler, lin_cons, scaled_cons, vals_tmp, 
        crit_cache, trial_caches, iteration_status,
    ) = optimizer_caches
 
    @ignoraise delta_new, is_good_trial_point, augment_filter = _test_trial_point!(
        sol, ndset,
        population, mop, scaler, trial_caches, vals_tmp, algo_opts; 
        indent
    )

    @ignoraise check_stopping_criterion(
        stop_crits, CheckPostIteration(),
        mop, mod, scaler, lin_cons, scaled_cons, vals, vals_tmp,
        mod_vals, ndset, step_vals, step_cache, crit_cache, trial_caches, 
        iteration_status, iteration_scalars, stop_crits,
        algo_opts
    )
   
    @ignoraise process_trial_point!(mod, vals_tmp, is_good_trial_point)
    
    if is_good_trial_point
        @logmsg log_level "$(indent_str(indent)) Trial point is accepted..."
        _sol = copy_solution_structs(sol)
        universal_copy!(_sol.vals, vals_tmp)
        add_to_set!(population, _sol; log_level, indent=indent+1)
        @logmsg log_level "$(indent_str(indent)) Trial point id = $(_sol.sol_id_ref[])"
    else
        @logmsg log_level "$(indent_str(indent)) Trial point is not accepted."
    end

    if augment_filter
        @logmsg log_level "$(indent_str(indent)) Adding $(sol.sol_id_ref[]) to filter."
        #mark_stale!(population, sol) 
        add_to_filter_and_check_population!(ndset, population, sol; log_level, indent=2)
    end
    return nothing
end

function _test_trial_point!(
    sol, ndset, 
    population, mop, scaler, trial_caches, vals_tmp , algo_opts;
    indent=0,
)
    @unpack vals, step_vals, mod_vals, mod, iteration_scalars = sol
    @unpack psi_theta, kappa_theta, nu_accept, nu_success, gamma_shrink, gamma_shrink_much, 
        gamma_grow, delta_max = algo_opts
    @unpack log_level = algo_opts

    @unpack diff_x, diff_fx, diff_fx_mod = trial_caches
    
    delta = delta_new = iteration_scalars.delta
    is_good_trial_point = false
    augment_filter = false
    
    tol = 2*mapreduce(eps, min, step_vals.s)
    if _delta_too_small(step_vals.s, tol)
        @logmsg log_level "Zero step. No trial point."
        
        universal_copy!(vals_tmp, vals)
        diff_x .= 0
        diff_fx .= 0
        diff_fx_mod .= 0

        delta_new = gamma_shrink_much * delta
        is_good_trial_point = false
        augment_filter = false
    else

        ## evaluate mop at trial point and store results in vals_tmp
        copyto!(cached_x(vals_tmp), step_vals.xs)
        xs = cached_x(vals_tmp)
        @ignoraise eval_mop!(vals_tmp, mop, scaler)
        
        ## make sure that model is also evaluated at trial point
        mxs = step_vals.fxs
        @ignoraise objectives!(mxs, mod, xs)
        x = cached_x(vals)
        fx = cached_fx(vals)
        fxs = cached_fx(vals_tmp)
        mx = cached_fx(mod_vals)
        θxs = cached_theta(vals_tmp)
      
        @. diff_x = x - xs
        @. diff_fx = fx - fxs
        @. diff_fx_mod = mx - mxs

        Δmx = maximum(diff_fx_mod)

        @logmsg log_level """
            \n$(indent_str(indent))- x - xs = $(pretty_row_vec(diff_x)), 
            $(indent_str(indent))- fx - fxs = $(pretty_row_vec(diff_fx)), 
            $(indent_str(indent))- mx - mxs = $(pretty_row_vec(diff_fx_mod))"""

        θx = cached_theta(vals)
    
        if !is_filter_acceptable(ndset, vals_tmp, vals)
            ## trial point rejected
            ## radius is reduced
            is_good_trial_point = false
            delta_new = gamma_shrink_much * delta
        else
            ## !(m(x) - κ⋅θ(x)^ψ >= m(x+s))
            ##if any(diff_fx_mod .< kappa_theta * θx^psi_theta)
            if Δmx .< kappa_theta * θx^psi_theta
                ## (theta step)
                ## trial point is accepted    
                is_good_trial_point = true
                delta_new = Δmx < 0 ? gamma_shrink_much * delta : delta
                ## sol is added to filter (sync population!)
                ## (adding sol to filter makes it incompatible with filter, 
                ## it is removed from population
                augment_filter = true
            else
                offset = Δmx * nu_accept
                if is_nondominated_with_offset(population, θxs, fxs, offset)
                    is_good_trial_point = true
                    delta_new = gamma_shrink * delta
                    if all(diff_fx .>= 0) && all(diff_fx_mod .>= 0)
                        if minimum(diff_fx) / Δmx >= nu_success
                            delta_new = min(delta_max, gamma_grow * delta)
                        end
                    end
                else
                    ## (innacceptable)
                    ## trial point is rejected
                    is_good_trial_point = false
                    ## radius is reduced much
                    delta_new = gamma_shrink_much * delta
                end
            end        
        end
    end
    iteration_scalars.delta = delta_new
    @logmsg log_level "$(indent_str(indent)) Δ_new = $(delta_new) (was Δ = $(delta))"
    return delta_new, is_good_trial_point, augment_filter
end

function is_nondominated_with_offset(population, θxs, fxs, offset)
    fxs = fxs .+ offset     # TODO cache
    for sol in nondominated_elements(population)
        if augmented_dominates(
            cached_theta(sol), cached_fx(sol),
            θxs, fxs
        )
            return false
        end
    end
    return true
end

function restore!(population, restoration_seeds, ndset, optimizer_caches, algo_opts; 
    max_gen_id,
    indent=0
)
    @unpack log_level = algo_opts
    @logmsg log_level """\n
    ~~~~~~~~~~~~~~~~~~~~~ RESTORATION ~~~~~~~~~~~~~~~~~~~~~"""


    #=
    best_index = 0
    best_n_norm = Inf
    for (sol_id, sol) in pairs(restoration_seeds.elems)
        is_stale(restoration_seeds, sol) && continue
        @unpack step_vals = sol
        if !(any(isnan.(step_vals.n)))
            n_norm = LA.norm(step_vals.n)
            if best_index <= 0
                best_index = sol_id
                best_n_norm = n_norm
            else
                if n_norm < best_n_norm
                    best_index = sol_id
                    best_n_norm = n_norm
                end
            end
        end                
    end
    =#

    _, best_index = findmin(cached_theta, restoration_seeds.elems)
    best_sol = restoration_seeds.elems[best_index]
    status = nothing

    @unpack (
        stop_crits, mop, scaler, lin_cons, scaled_cons, vals_tmp, 
        crit_cache, trial_caches, iteration_status,
    ) = optimizer_caches 

    sol = copy_solution_structs(best_sol)
    sol.gen_id_ref[] = max_gen_id + 1
    unconditionally_add_to_set!(population, sol)

    @unpack vals, step_vals, step_cache, iteration_scalars, mod, mod_vals = sol
    status = do_restoration(
        mop, mod, scaler, lin_cons, scaled_cons, vals, vals_tmp,
        mod_vals, ndset, step_vals, step_cache, crit_cache, trial_caches, 
        iteration_status, iteration_scalars, stop_crits,
        algo_opts
    )
    
    @logmsg log_level """\n
    ~~~~~~~~~~~~~~~~~~~~~ END RESTORATION ~~~~~~~~~~~~~~~~~"""
    
    return sol, status
end