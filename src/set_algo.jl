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
    return AugmentedVectorFilter{F, AugmentedVectorElement{F}, tmpType, NoFilterMeta}(;
        print_name="Filter", gamma, tmp)
end

function initialize_population(
    prepopulation, mod_vals, step_vals, step_cache, iteration_scalars, mop, mod, stop_crits
)
    ST = _status_type(mop, mod, stop_crits)
    vals = getfield(first(nondominated_elements(prepopulation)), :vals)
    sol = make_solution(
        vals, step_vals, step_cache, iteration_scalars, mod, mod_vals, ST; 
        gen_id = 0,
    )
    meta = FilterMetaMOP(;
        float_type = float_type(mop),
        dim_vars = dim_vars(mop), 
        dim_objectives = dim_objectives(mop),
        dim_nl_eq_constraints = dim_nl_eq_constraints(mop),
        dim_nl_ineq_constraints = dim_nl_ineq_constraints(mop),
        dim_lin_eq_constraints = dim_lin_eq_constraints(mop),
        dim_lin_ineq_constraints = dim_lin_ineq_constraints(mop),
    )
    population = singleton_population(float_type(mop), sol, meta)
    for elem in Iterators.drop(nondominated_elements(prepopulation), 1)
        sol = make_solution(
            elem.vals, step_vals, step_cache, iteration_scalars, mod, mod_vals, ST;
            gen_id = 0
        )
        unconditionally_add_to_set!(
            population, sol; 
            elem_identifier = get_identifier(elem)  # retain ids from prepopulation for consistent logging.
        )
    end
    return sol, population
end

function initialize_prepopulation(F, X, mop, scaler, lin_cons, vals; log_level=Info)
    prepopulation = make_prepopulation(F, vals; log_level)
    return add_columns_to_prepulation!(
        prepopulation, X, mop, scaler, lin_cons, vals;
        log_level,
        offset = 1
    )
end

function make_prepopulation(::Type{F}, vals::valsType; log_level=Info) where {F, valsType}
    prepopulation = AugmentedVectorFilter{F, ValueCacheElement{valsType}, Nothing, NoFilterMeta}()
    add_to_set!(prepopulation, ValueCacheElement(; vals); log_level)
    return prepopulation
end

function add_columns_to_prepulation!(
    prepopulation, X, mop, scaler, lin_cons, vals; 
    offset=1, log_level=Info
)
    for (i, ξ0) in enumerate(eachcol(X))
        i <= offset && continue

        _vals = deepcopy(vals)
        copyto!(cached_ξ(_vals), ξ0)
        project_into_box!(cached_ξ(_vals), lin_cons)
        scale!(cached_x(_vals), scaler, cached_ξ(_vals))
        @ignorebreak eval_mop!(_vals, mop)
        add_to_set!(
            prepopulation, ValueCacheElement(; vals=_vals); 
            log_level, check_elem = true,
            elem_identifier=i
        )
    end
    return prepopulation
end

function make_solution(
    vals, step_vals, step_cache, iteration_scalars, mod, mod_vals, ::Type{status_type}=Any;
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
        Ref(sol_id),
        Ref(false)
    )
    return sstructs
end

function singleton_population(::Type{F}, sol::S, meta::M) where {F, S, M}
    population = AugmentedVectorFilter{F, S, Nothing, M}(; 
        print_name="Population", meta)
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
        tmp = nothing #deepcopy(population.tmp)
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
                    gen_id = max_gen_id + 1
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
                    @logmsg log_level "💀 🦜 marking $(get_identifier(sol)) as stale."
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
                add_to_filter_and_mark_population!(ndset, population, sol; log_level)
                @logmsg log_level "2/2) Storing seed."
                #abc add_to_set!(restoration_seeds, sol; log_level, elem_identifier=get_identifier(sol))
                add_to_set!(restoration_seeds, copy_solution_structs(sol); log_level, elem_identifier=get_identifier(sol))
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
        #=
        @info """
        \nsol_id = $(sol.sol_id_ref[])
        gen_id = $(sol.gen_id_ref[])
        θ = $(cached_theta(sol))
        is_stale = $(is_stale(population, sol))"""
        =#
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

    if !sol.is_restored_ref[]
        @logmsg log_level "* Updating Surrogates."
        @ignoraise update_models!(mod, delta, scaler, vals, scaled_cons; log_level, indent) indent
    end
    sol.is_restored_ref[] = false
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

function test_trial_point!(
    sol, ndset, population, optimizer_caches, algo_opts, ver=Val(:version1); 
    indent=0,
    gen_id=nothing
)
    @unpack log_level = algo_opts
    @unpack vals, step_vals, step_cache, iteration_scalars, mod, mod_vals = sol
    @unpack (
        stop_crits, mop, scaler, lin_cons, scaled_cons, vals_tmp, 
        crit_cache, trial_caches, iteration_status,
    ) = optimizer_caches
 
    @ignoraise delta_new, is_good_trial_point, augment_filter, delta_child = _test_trial_point!(
        ver, sol, ndset,
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
        _sol.iteration_scalars.delta = delta_child
        if !isnothing(gen_id)
            _sol.gen_id_ref[] = gen_id
        end

        add_to_set!(population, _sol; log_level, indent=indent+1)
        @logmsg log_level "$(indent_str(indent)) Trial point id = $(_sol.sol_id_ref[])"
    else
        @logmsg log_level "$(indent_str(indent)) Trial point is not accepted."
    end

    if augment_filter
        @logmsg log_level "$(indent_str(indent)) Adding $(sol.sol_id_ref[]) to filter."
        add_to_filter_and_mark_population!(ndset, population, sol; log_level, indent=2)
    end
    return nothing
end

function _test_trial_point!(
    ::Val{:version1},
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
    return delta_new, is_good_trial_point, augment_filter, delta_new
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
    gen_id,
    augment_filter=false,
    indent=0
)
    @unpack log_level = algo_opts
    @logmsg log_level """\n
    ~~~~~~~~~~~~~~~~~~~~~ IT_RESTORATION ~~~~~~~~~~~~~~~~~~~~~"""

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
    if augment_filter
        add_to_filter_and_mark_population!(
            ndset, population, best_sol; 
            indent, log_level,
            elem_identifier = get_identifier(best_sol)
        )
    end
    log_solution_values(best_sol, log_level)
    status = nothing

    @unpack (
        stop_crits, mop, scaler, lin_cons, scaled_cons, vals_tmp, 
        crit_cache, trial_caches, iteration_status,
    ) = optimizer_caches 

    sol = copy_solution_structs(best_sol)
    sol.gen_id_ref[] = gen_id
    unconditionally_add_to_set!(population, sol)
    @logmsg log_level "$(indent_str(indent))Restoration point has id $(get_identifier(sol))."

    @unpack vals, step_vals, step_cache, iteration_scalars, mod, mod_vals = sol
    status = do_restoration(
        mop, mod, scaler, lin_cons, scaled_cons, vals, vals_tmp,
        mod_vals, ndset, step_vals, step_cache, crit_cache, trial_caches, 
        iteration_status, iteration_scalars, stop_crits,
        algo_opts
    )
    
    @logmsg log_level """\n
    ~~~~~~~~~~~~~~~~~~~~~ END IT_RESTORATION ~~~~~~~~~~~~~~~~~"""
    
    return sol, status
end
function optimize_many(
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

    # for the moment, only use minimum radius stopping (and max_iter)
    # TODO re-enable other criteria
    @unpack stop_delta_min = algo_opts
    stop_crits = MinimumRadiusStopping(; delta_min = stop_delta_min)
    @reset optimizer_caches.stop_crits = stop_crits

    F = float_type(mop)
    prepopulation = initialize_prepopulation(F, X, mop, scaler, lin_cons, vals; log_level)   
    sol, population = initialize_population(
        prepopulation, mod_vals, step_vals, step_cache, iteration_scalars, mop, mod, stop_crits
    )

    ndset = initialize_filter(F, sol, filter_gamma)

    restoration_seeds = AugmentedVectorFilter(;
        elems = empty(population.elems),
        gamma = population.gamma,
        tmp = nothing #deepcopy(population.tmp)
    )

    optimize_population!(population, ndset, restoration_seeds, optimizer_caches, algo_opts)
    return population
end

function reset_restoration_seeds!(restoration_seeds)
    empty!(restoration_seeds.elems)
    empty!(restoration_seeds.is_stale)
    restoration_seeds.min_scalar_val[] = Inf
    empty!(restoration_seeds.min_vector_vals)
    restoration_seeds.counter[] = 0
    nothing
end

function optimize_population!(
    population, ndset, restoration_seeds, optimizer_caches, algo_opts;
    check_filter=true
)
    @unpack max_iter = algo_opts
    @unpack log_level = algo_opts
    
    final_stop_crit = MaxIterStopping(max_iter)
    
    restoration_required = stop_optimization = false
    for it_id = 1 : max_iter
        remove_stale_elements!(population)  # optional, we should do some benchmarks ...
        remove_stale_elements!(ndset)
        stop_optimization && break

        @logmsg log_level """\n
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ~ 🕐 ITERATION $(it_id), len pop = $(length(population.elems))
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""

        if restoration_required
            sol, status = restore!(
                population, restoration_seeds, ndset, optimizer_caches, algo_opts; 
                gen_id=it_id,
                augment_filter=true
            )
            if status isa AbstractStoppingCriterion
                log_stop_code(status, log_level)
                mark_stale!(population, sol)
                sol.status_ref[] = status
                stop_optimization = true
            end
            sol.is_restored_ref[] = true
            restoration_required = false
            continue
        end

        #remove_stale_elements!(restoration_seeds)
        reset_restoration_seeds!(restoration_seeds)
        sorted_sol_oids = sortperm(population.elems; by=solution_sorting_vector, rev=true)

        for sol_objectid in sorted_sol_oids
            sol = population.elems[sol_objectid]
            is_stale(population, sol) && continue
            is_converged(sol) && continue
            sol.gen_id_ref[] >= it_id && continue
            
            if check_filter
                if !is_filter_acceptable(ndset, sol)
                    @logmsg log_level "💀 🦜 marking $(get_identifier(sol)) as stale."
                    mark_stale!(population, sol)
                    continue
                end
            end

            n_is_compatible = step_normally!(sol, ndset, optimizer_caches, algo_opts)
            if n_is_compatible isa AbstractStoppingCriterion
                sol.status_ref[] = n_is_compatible
                if is_ultimate_stop_crit(n_is_compatible) 
                    final_stop_crit = n_is_compatible
                    stop_optimization = true
                end
                continue
            end

            if !n_is_compatible
                #@logmsg log_level "… normal step not compatible. 1/2) Augmenting filter."
                #add_to_filter_and_mark_population!(ndset, population, sol; log_level)
                mark_stale!(population, sol)
                @logmsg log_level "Normal step incompatible, storing seed $(get_identifier(sol))."
                add_to_set!(
                    restoration_seeds, 
                    copy_solution_structs(sol);     # I think copying is very important, but I can't remember why
                    log_level, 
                    elem_identifier=get_identifier(sol) # use old id for consistent logging
                )
                continue
            end
            
            status = step_tangentially!(sol, ndset, optimizer_caches, algo_opts)
            if status isa AbstractStoppingCriterion
                sol.status_ref[] = status
                if is_ultimate_stop_crit(status) 
                    final_stop_crit = status
                    stop_optimization = true
                end
                continue
            end
            status = test_trial_point!(
                sol, ndset, population, optimizer_caches, algo_opts, Val(:version2);
                gen_id = it_id,
            )
            if status isa AbstractStoppingCriterion
                sol.status_ref[] = status
                if is_ultimate_stop_crit(status) 
                    final_stop_crit = status
                    stop_optimization = true
                end
                continue
            end
        end

        stop_optimization && continue

        restoration_required, stop_optimization = check_population(population)
        
    end
    @logmsg log_level "🏁 🏁 🏁 🏁 🏁 FINISHED 🏁 🏁 🏁 🏁 🏁 ."

    for sol in population.elems
        if !is_converged(sol)
            sol.status_ref[] = final_stop_crit
        end
    end

    return population
end

function solution_sorting_vector(sol)
    @unpack vals, iteration_scalars = sol
    return vcat(cached_theta(vals), iteration_scalars.delta, -sol.gen_id_ref[])
end

function check_population(population)
    ## if all elements are (stale or converged): stop
    ## ⇔ if there is an element neither stale nor converged: don't stop

    ## if all elements are stale, and there is no converged solution: restoration
    ## ⇔ if there is a nonstale element or there is a converged solution: dont do restoration

    stop_optimization = true
    any_converged = false
    for sol in population.elems
        if is_converged(sol)
            any_converged = true
            continue
        end
        is_stale(population, sol) && continue
        ## element not stale nor converged: don't stop
        stop_optimization = false
        break
    end

    restoration_required = false
    if stop_optimization
        if !any_converged
            restoration_required = true
            stop_optimization = false
        end
    end

    return restoration_required, stop_optimization
end


function _test_trial_point!(
    ::Val{:version2}, sol, ndset, 
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
    
    tol = 1.1*mapreduce(eps, min, step_vals.s)
    if _delta_too_small(step_vals.s, tol)
        @logmsg log_level "Zero step. No trial point."
        
        universal_copy!(vals_tmp, vals)
        diff_x .= 0
        diff_fx .= 0
        diff_fx_mod .= 0

        delta_child = delta_new = gamma_shrink_much * delta
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

        θx = cached_theta(vals)
        @logmsg log_level """
            \n$(indent_str(indent))- x - xs = $(pretty_row_vec(diff_x)), 
            $(indent_str(indent))- fx - fxs = $(pretty_row_vec(diff_fx)), 
            $(indent_str(indent))- mx - mxs = $(pretty_row_vec(diff_fx_mod)),
            $(indent_str(indent))- theta(xs)= $(cached_theta(vals_tmp)),
            $(indent_str(indent))- κθ^ψ     = $(kappa_theta * θx^psi_theta)"""
   
        
        ## `is_pos` is an index vector for the positive entries of the difference
        ## vector `del_mx`.
        ## We use it in case of partial descent.
        ## `Δmx` is the minimum surrogate objective reduction.
        is_pos = zeros(Bool, length(diff_fx_mod))
        Δmx = Inf
        for (l, del_mx) in enumerate(diff_fx_mod)
            del_mx <= 0 && continue
            is_pos[l] = true
            if del_mx < Δmx 
                Δmx = del_mx
            end
        end

        if !(is_filter_acceptable(ndset, vals_tmp, vals))
            ## trial point rejected
            ## radius is reduced
            @logmsg log_level "$(indent_str(indent)) Trial point not filter-compatible."
            is_good_trial_point = false
            delta_child = delta_new = gamma_shrink_much * delta
        else
            ## The single-objective f-step test is 
            ## m(x) - m(x+s) ≥ κ θ(x)^ψ
            ## ⇔
            ## m(x + s) + κ θ(x)^ψ ≤ m(x)
            ## In the multi-objective case, let's try
            ## m(x+s) + κθ(x)^ψ ≼ m(x).
            ## If this test fails, we deem the constraint violation too high and
            ## have a θ-step.
            ## For sake of simplicity, we test 
            ## any( m(x+s) + κθ(x)^ψ .> m(x) )
            ## If that is true, then we have θ-step.
            ##
            ## In case of partial descent, we have to respect the index set `is_pos`.
            theta_step_test_offset = kappa_theta * θx^psi_theta
            if isinf(Δmx) || any( mxs[is_pos] .+ theta_step_test_offset .> mx[is_pos] )
                ## (theta step)
                ## trial point is accepted    
                is_good_trial_point = true
                ## do not change radius # TODO could be smarter
                delta_child = delta_new = delta
                ## sol is added to filter (sync population!)
                ## (adding sol to filter makes it incompatible with filter, 
                ##   it should be/is (?) removed from population)
                augment_filter = true
            else
                offset = zero(fxs)  # TODO cache
                offset[is_pos] .= diff_fx_mod[is_pos] .* nu_accept
                if trial_point_population_acceptable(fxs, θxs, population, offset)
                    is_good_trial_point = true

                    delta_child = delta_new = gamma_shrink * delta
                    rho_success = minimum( diff_fx[is_pos] ./ diff_fx_mod[is_pos]; init=-Inf )
                    if rho_success >= nu_success
                        delta_child = min(delta_max, gamma_grow * delta)
                    end
                else
                    ## (innacceptable)
                    ## trial point is rejected
                    is_good_trial_point = false
                    ## radius is reduced much
                    delta_child = delta_new = gamma_shrink_much * delta
                end
            end        
        end
    end
    iteration_scalars.delta = delta_new
    @logmsg log_level "$(indent_str(indent)) Δ_new = $(delta_new), Δ_child = $(delta_child), (was Δ = $(delta))"
    return delta_new, is_good_trial_point, augment_filter, delta_child
end

function vector_dominates(lhs, rhs)
    return all( lhs .<= rhs ) && any( lhs .<= rhs )
end

function cosine_similarity(p1, p2)
    return LA.dot(p1, p2) / (LA.norm(p1) * LA.norm(p2))
end

function trial_point_population_acceptable(fxs, θxs, population, offset=0)
    fxs_shifted = fxs .+ offset     # TODO cache
    for sol in nondominated_elements(population)
        if augmented_dominates(
            cached_theta(sol), cached_fx(sol),
            θxs, fxs_shifted, 
        )
            return false
        end
    end
    return true
end