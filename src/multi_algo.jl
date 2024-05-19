# We want to iterate a whole population of solutions and their corresponding
# structures.
# For sake of simplicity, we don't care too much about optimizing allocations right now.
# We thus maintain objects in arrays and modify these arrays as needed.
abstract type AbstractMultiStatus end
struct IncompatibleNormalStep <: AbstractMultiStatus end
struct InacceptablePoint <: AbstractMultiStatus end

import UUIDs: UUID, uuid4
struct SolutionStructs{
    valsType, step_valsType, step_cacheType, 
    modType, mod_valsType, iteration_scalarsType,
    statusType
}
    vals :: valsType
    step_vals :: step_valsType
    step_cache :: step_cacheType
    iteration_scalars :: iteration_scalarsType
    mod :: modType
    mod_vals :: mod_valsType
    status :: statusType
    gen_id :: Base.RefValue{UUID}
    counter :: Base.RefValue{Int}
end

function copy_solution_structs(sstructs)
    @unpack vals, step_vals, step_cache, iteration_scalars, mod, mod_vals, status, gen_id, counter = sstructs
    return SolutionStructs(
        deepcopy(vals),
        deepcopy(step_vals),
        deepcopy(step_cache),
        deepcopy(iteration_scalars),
        CE.copy_model(mod),
        deepcopy(mod_vals),
        deepcopy(status),
        deepcopy(gen_id),
        deepcopy(counter)
    )
end

struct NondominatedSet{F<:AbstractFloat, E}
    fx_vecs :: Vector{Vector{F}}
    theta_vals :: Vector{F}

    gamma :: F

    extra :: E

    counter :: Base.RefValue{Int}
end

float_type(::NondominatedSet{F}) where F = F

keepat_extra!(ndset::NondominatedSet, ind)=keepat_extra!(ndset.extra, ind)
keepat_extra!(::Nothing, ind)=nothing
keepat_extra!(extra, ind)=keepat!(extra, ind)
push_extra!(ndset::NondominatedSet, ex)=push_extra!(ndset.extra, ex)
push_extra!(::Nothing, ex)=nothing
push_extra!(extra, ex)=push!(extra, ex)

filter_min_theta(ndset::NondominatedSet) = max(0, minimum(ndset.theta_vals; init=0))

function NondominatedSet(::Type{F}, extra = nothing; gamma=0) where F<:AbstractFloat
    return NondominatedSet(Vector{F}[], F[], F(gamma), extra, Ref(0))
end

function is_not_dominated(ndset::NondominatedSet, fx, theta)
    @unpack fx_vecs, theta_vals, gamma = ndset
    is_acceptable = true
    for (fx_i, theta_i) in zip(fx_vecs, theta_vals)
        if theta_i <= theta && all(fx_i .<= fx) && (theta_i < theta || any(fx_i .< fx))
            is_acceptable = false
            break
        end
    end
    return is_acceptable
end

function is_filter_acceptable(ndset::NondominatedSet, cache)
    return is_not_dominated(ndset, cached_fx(cache), cached_theta(cache))
end
function is_filter_acceptable(ndset::NondominatedSet, cache_test, cache_add)
    # assume that cache_add is already contained in ndset
    return is_filter_acceptable(ndset, cache_test)
end

set_counter!(ndset, ::Nothing)=nothing
function set_counter!(ndset, solution_structs)
    ndset.counter[] += 1
    solution_structs.counter[] = ndset.counter[]
    return nothing
end
function add_to_filter!(ndset::NondominatedSet, cache, extra=nothing)
    set_counter!(ndset, extra)
    return add_and_remove_dominated_no_check!(ndset, copy(cached_fx(cache)), cached_theta(cache), extra)
end

function add_to_filter!(ndset::NondominatedSet, sstructs::SolutionStructs)
    cache=sstructs.vals
    return add_and_remove_dominated_no_check!(ndset, cache, sstructs)
end

function add_and_remove_dominated!(ndset::NondominatedSet, fx, theta, extra=nothing)
    if is_not_dominated(ndset, fx, theta)
        return add_and_remove_dominated_no_check!(ndset, fx, theta, extra)
    end
    return nothing
end

function add_and_remove_dominated_no_check!(
    ndset::NondominatedSet, fx, theta, extra=nothing
)
    @unpack fx_vecs, theta_vals, gamma = ndset
    offset_j = gamma * theta
    fx_j = fx .- offset_j
    theta_j = max(0, theta - offset_j)
    
    remain_flags = fill(true, length(fx_vecs))
    for (i, (fx_i, theta_i)) in enumerate(zip(fx_vecs, theta_vals))
        if theta_j <= theta_i && all(fx_j .<= fx_i) && (theta_j < theta_i || any(fx_j .< fx_i))
            remain_flags[i] = false
        end
    end
    keepat!(fx_vecs, remain_flags)
    keepat!(theta_vals, remain_flags)
    keepat_extra!(ndset, remain_flags)
    push!(fx_vecs, fx_j)
    push!(theta_vals, theta_j)
    push_extra!(ndset, extra)
    return length(fx_vecs)
end

function optimize_set(
    X::RMat, 
    mop;
    algo_opts = AlgorithmOptions(), 
    user_callback = NoUserCallback()
)
    @unpack log_level, max_iter = algo_opts
    @reset algo_opts.max_iter = typemax(Int)
    optimizer_caches = initialize_structs(mop, @view(X[:, 1]), algo_opts, user_callback)

    if optimizer_caches isa AbstractStoppingCriterion
        return NondominatedSet(float_type(mop))
    end
    
    @unpack (
        mop, mod, scaler, lin_cons, scaled_cons, vals, vals_tmp,
        mod_vals, step_vals, step_cache, crit_cache, trial_caches, 
        iteration_status, iteration_scalars, stop_crits,
    ) = optimizer_caches

    #=
    _ST = Union{
        stop_crit_type(stop_crits), 
        stop_type(mop), 
        stop_type(mod), 
        IncompatibleNormalStep(), InfeasibleStopping()}
    ST = Base.RefValue{Union{Nothing, _ST, WrappedStoppingCriterion{<:_ST}}}
    =#
    ST = Base.RefValue{Any}
    it_id = uuid4()
    
    solution_structs = SolutionStructs(
        vals, step_vals, step_cache, iteration_scalars, mod, mod_vals, 
        ST(nothing), Ref(it_id), Ref(0)
    )
    filter = NondominatedSet(float_type(mop), typeof(solution_structs)[])
    add_to_filter!(filter, vals, solution_structs)

    for ξ0 = Iterators.drop(eachcol(X), 1)
        vals_x = deepcopy(vals)
        Base.copyto!(cached_ξ(vals_x), ξ0)
        project_into_box!(cached_ξ(vals_x), lin_cons)
        scale!(cached_x(vals_x), scaler, cached_ξ(vals_x))
        @ignorebreak eval_mop!(vals_x, mop)
        if is_filter_acceptable(filter, vals_x)
            solution_structs_x = copy_solution_structs(solution_structs)
            universal_copy!(solution_structs_x.vals, vals_x)
            add_to_filter!(filter, vals_x, solution_structs_x)
        end
    end

    for it_index = 1:max_iter
        @logmsg log_level """\n
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ITERATION $(it_index)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        """
        next_it_id = uuid4() 
        indent = 1
        is_compat = Dict{Int, Bool}(sol.counter[] => true for sol in filter.extra)
        any_not_converged = false
        any_is_compat = false

        #sort_fn(sol_structs) = !isnothing(sol_structs.status[]) ? -Inf : (sol_structs.gen_id[] != it_id ? -Inf : sol_structs.iteration_scalars.delta)
        #I = sortperm(filter.extra; by = sort_fn, rev = true)
        I = eachindex(filter.extra)
        for sol_index in I 
            solution_structs = filter.extra[sol_index]
            !isnothing(solution_structs.status[]) && continue
            solution_structs.gen_id[] != it_id && continue
            any_not_converged = true

            @unpack vals, step_vals, step_cache, iteration_scalars, mod, mod_vals = solution_structs
            delta = iteration_scalars.delta
            status = check_stopping_criterion(
                stop_crits, CheckPreIteration(),
                mop, mod, scaler, lin_cons, scaled_cons, vals, vals_tmp,
                mod_vals, filter, step_vals, step_cache, crit_cache, trial_caches, 
                iteration_status, iteration_scalars, stop_crits,
                algo_opts
            )
            if status isa AbstractStoppingCriterion
                solution_structs.status[] = status
                continue
            end

            @logmsg log_level """\n
            ###########################
            # sol_id = $(solution_structs.counter[]).
            ###########################
            Δ = $(delta)
            θ = $(cached_theta(vals))
            ξ = $(pretty_row_vec(cached_ξ(vals)))
            x = $(pretty_row_vec(cached_x(vals)))
            fx = $(pretty_row_vec(cached_fx(vals)))
            """

            @logmsg log_level "* Updating Surrogates."
            status = update_models!(mod, delta, scaler, vals, scaled_cons; log_level, indent)
            if status isa AbstractStoppingCriterion
                solution_structs.status[] = status
                continue
            end
            status = eval_and_diff_mod!(mod_vals, mod, cached_x(vals))
            if status isa AbstractStoppingCriterion
                solution_structs.status[] = status
                continue
            end

            do_normal_step!(
                step_cache, step_vals, delta, mop, mod, scaler, lin_cons, scaled_cons, vals, mod_vals;
                log_level, indent=0
            )

            n_is_compatible = compatibility_test(step_vals.n, algo_opts, delta)

            if !n_is_compatible
                @logmsg log_level "… normal step not compatible, $(step_vals.xn)."
                is_compat[solution_structs.counter[]] = false
                continue
            end
            any_is_compat = true
        end
        !any_not_converged && break
        
        there_are_solutions_with_current_id = true
        if !any_is_compat
            best_index = 0
            best_n_norm = Inf
            for (sol_index, solution_structs) = enumerate(filter.extra)
                !isnothing(solution_structs.status[]) && continue
                solution_structs.gen_id[] != it_id && continue
                @unpack step_vals = solution_structs
                
                solution_structs.status[] = IncompatibleNormalStep()

                if !(any(isnan.(step_vals.n)))
                    if best_index <= 0
                        best_index = sol_index
                    else
                        nnorm = LA.norm(step_vals.n)
                        if nnorm < best_n_norm
                            best_index = sol_index
                            best_n_norm = nnorm
                        end
                    end
                end
            end
            if best_index == 0
                best_index = 1
            end
            solution_structs = copy_solution_structs(filter.extra[best_index])
            @unpack vals, step_vals, step_cache, iteration_scalars, mod, mod_vals = solution_structs
            restoration_code = do_restoration(
                mop, mod, scaler, lin_cons, scaled_cons, vals, vals_tmp,
                mod_vals, filter, step_vals, step_cache, crit_cache, trial_caches, 
                iteration_status, iteration_scalars, stop_crits,
                algo_opts
            )
            if restoration_code isa AbstractStoppingCriterion
                return NondominatedSet(float_type(mop))
            end
            solution_structs.gen_id[] = next_it_id
            solution_structs.status[] = nothing
            add_to_filter!(filter, vals, solution_structs)
            there_are_solutions_with_current_id = false
        else
            while !(all(is_compat[sol.counter[]] for sol in filter.extra))
                for (sol_index, _solution_structs) = enumerate(filter.extra)
                    get(is_compat, _solution_structs.counter[], false) && continue
                    is_compat[_solution_structs.counter[]] = true

                    !isnothing(_solution_structs.status[]) && continue
                    _solution_structs.gen_id[] != it_id && continue
                
                    @logmsg log_level "## RESTORATION sol_index = $(solution_structs.counter[]). ##"
                    _solution_structs.status[] = IncompatibleNormalStep()

                    solution_structs = copy_solution_structs(_solution_structs)
                    @unpack vals, step_vals, step_cache, iteration_scalars, mod, mod_vals = solution_structs
                    restoration_code = do_restoration(
                        mop, mod, scaler, lin_cons, scaled_cons, vals, vals_tmp,
                        mod_vals, filter, step_vals, step_cache, crit_cache, trial_caches, 
                        iteration_status, iteration_scalars, stop_crits,
                        algo_opts; solve_nl_subproblem=false
                    )
                    if restoration_code isa AbstractStoppingCriterion
                        continue
                    end
                    solution_structs.gen_id[] = next_it_id
                    solution_structs.status[] = nothing
                    add_to_filter!(filter, vals, solution_structs)
                    is_compat[solution_structs.counter[]] = true
                    break
                end
            end
        end
        while there_are_solutions_with_current_id
            there_are_solutions_with_current_id = false
            #I = sortperm(filter.extra; by = sort_fn, rev = true)
            I = eachindex(filter.extra)
            for sol_index in I
                solution_structs = filter.extra[sol_index]
                !isnothing(solution_structs.status[]) && continue
                solution_structs.gen_id[] != it_id && continue
                
                there_are_solutions_with_current_id = true
                solution_structs.gen_id[] = next_it_id
                
                @unpack vals, step_vals, step_cache, iteration_scalars, mod, mod_vals = solution_structs
                @unpack delta = iteration_scalars

                @logmsg log_level """\n
                ###########################
                # sol_index = $(solution_structs.counter[]).
                ###########################
                Δ = $(delta)
                θ = $(cached_theta(vals))
                ξ = $(pretty_row_vec(cached_ξ(vals)))
                x = $(pretty_row_vec(cached_x(vals)))
                n = $(pretty_row_vec(step_vals.n))
                xn= $(pretty_row_vec(step_vals.xn))
                """

                @logmsg log_level "* Computing a descent step."
                do_descent_step!(
                    step_cache, step_vals, delta, mop, mod, scaler, lin_cons, scaled_cons, vals, mod_vals;
                    log_level
                )
        
                @logmsg log_level " - Criticality χ=$(step_vals.crit_ref[]), ‖d‖₂=$(LA.norm(step_vals.d)), ‖s‖₂=$(LA.norm(step_vals.s))."

                status = check_stopping_criterion(
                    stop_crits, CheckPostDescentStep(), 
                    mop, mod, scaler, lin_cons, scaled_cons, vals, vals_tmp,
                    mod_vals, filter, step_vals, step_cache, crit_cache, trial_caches, 
                    iteration_status, iteration_scalars, stop_crits,
                    algo_opts
                )
                if status isa AbstractStoppingCriterion
                    solution_structs.status[] = status
                    continue
                end
                ## For the convergence analysis to work, we also have to have the Criticality Routine:
                delta_new = criticality_routine!(
                    mop, mod, scaler, lin_cons, scaled_cons, vals, vals_tmp,
                    mod_vals, filter, step_vals, step_cache, crit_cache, trial_caches, 
                    iteration_status, iteration_scalars, stop_crits,
                    algo_opts;
                    indent
                )
                if delta_new isa AbstractStoppingCriterion
                    solution_structs.status[] = delta_new
                    continue
                end
                iteration_scalars.delta = delta_new
                delta = iteration_scalars.delta

                # Use `vals_tmp` to hold the true values at `xs`
                copyto!(cached_x(vals_tmp), step_vals.xs)
                status = eval_mop!(vals_tmp, mop, scaler)
                if status isa AbstractStoppingCriterion
                    solution_structs.status[] = status
                    continue
                end
                
                delta_new = test_trial_point!(
                    mop, mod, scaler, lin_cons, scaled_cons, vals, vals_tmp,
                    mod_vals, filter, step_vals, step_cache, crit_cache, trial_caches, 
                    iteration_status, iteration_scalars, stop_crits,
                    algo_opts;
                    indent, do_log_trial=false,
                )
                if delta_new isa AbstractStoppingCriterion
                    @logmsg log_level "Could not evaluate trial point because of status code:"
                    log_stop_code(delta_new, log_level)
                    solution_structs.status[] = delta_new
                    continue
                end
                @logmsg log_level """\n
                it_type = $(iteration_status.iteration_type)
                θ(x)   = $(cached_theta(vals))
                θ(x+s) = $(cached_theta(vals_tmp))
                f(x)-f(x+s) = $(pretty_row_vec(trial_caches.diff_fx))
                m(x)-m(x+s) = $(pretty_row_vec(trial_caches.diff_fx_mod))
                rho's       = $(pretty_row_vec(trial_caches.diff_fx ./ trial_caches.diff_fx_mod))
                """

                iteration_scalars.delta = delta_new
                delta = iteration_scalars.delta

                status = check_stopping_criterion(
                    stop_crits, CheckPostIteration(),
                    mop, mod, scaler, lin_cons, scaled_cons, vals, vals_tmp,
                    mod_vals, filter, step_vals, step_cache, crit_cache, trial_caches, 
                    iteration_status, iteration_scalars, stop_crits,
                    algo_opts
                )
                if status isa AbstractStoppingCriterion
                    solution_structs.status[] = status
                    continue
                end

                status = process_trial_point!(mod, vals_tmp, iteration_status)
                if status isa AbstractStoppingCriterion
                    solution_structs.status[] = status
                    continue
                end
            
                ## update filter
                if iteration_status.iteration_type != FILTER_FAIL
                    trial_sol = copy_solution_structs(solution_structs)
                    #=
                    if !_trial_point_accepted(iteration_status)
                        trial_sol.status[] = InacceptablePoint()
                    end
                    =#

                    universal_copy!(trial_sol.vals, vals_tmp)
                    add_to_filter!(filter, vals_tmp, trial_sol)
                end
                break
            end#for
        end#while
        it_id = next_it_id
    end

    return filter
end

function initialize_structs_for_multi_algo(X, mop, algo_opts, user_callback; indent=0)
    
    @unpack scaler_cfg, step_config, delta_init, delta_max = algo_opts
    require_fully_linear = algo_opts.require_fully_linear_models
    
    F = float_type(mop)
    X0 = ensure_float_type(X, F)

    n_vars, n_sols = size(X0)
    @assert n_vars == dim_vars(mop)
    @assert n_sols > 0
    lin_cons = init_lin_cons(mop)
    scaler = init_scaler(scaler_cfg, lin_cons, n_vars)
    scaled_cons = deepcopy(lin_cons)
    update_lin_cons!(scaled_cons, scaler, lin_cons)
   
    ## caches for working arrays x, fx, hx, gx, Ex, Ax, …
    value_caches = [WrappedMOPCache(init_value_caches(mop)) for _=1:n_sols]
     
    ## (perform 1 evaluation to set values already)
    for (j, ξ0) = enumerate(eachcol(X0))
        vals = value_caches[j]
        Base.copyto!(cached_ξ(vals), ξ0)
        project_into_box!(cached_ξ(vals), lin_cons)
        scale!(cached_x(vals), scaler, cached_ξ(vals))
        @ignoraise eval_mop!(vals, mop) indent
    end

    ndset = init_nondom_set!(value_caches, F)
    n_sols = length(value_caches)

    value_caches_tmp = deepcopy(value_caches)

    mod = init_models(mop, scaler; delta_max, require_fully_linear)
    mod_vals = init_value_caches(mod)
 
    ## pre-allocate working arrays for normal and descent step calculation:
    step_value_caches = [init_step_vals(vals) for vals in value_caches]
    step_tmp_caches = [
        init_step_cache(step_config, vals, mod_vals) for vals in value_caches
    ]

    crit_cache = CriticalityRoutineCache(;
        delta = F(NaN),
        num_crit_loops = -1,
        step_vals = deepcopy(first(step_value_caches)),
    )

    ## finally, compose information about the 0-th iteration for next_iterations:
    iteration_stats_template = IterationStatus(;
        iteration_type = INITIALIZATION,
        radius_update_result = INITIAL_RADIUS,
    )
    it_stats = fill(iteration_stats_template, n_sols)

    iteration_scalars_array = fill(
        IterationScalars{F}(; it_index = 0, delta = F(delta_init)), n_sols)

    #deltas = fill(F(delta_init), n_sols)

    trial_caches = TrialCaches{F}(;
        delta = F(NaN),
        diff_x = array(F, dim_vars(mop)),
        diff_fx = array(F, dim_objectives(mop)),
        diff_fx_mod = array(F, dim_objectives(mop))
    )
    stop_crits = stopping_criteria(algo_opts, user_callback)
    return (
        lin_cons, scaler, scaled_cons, value_caches, value_caches_tmp,
        ndset, mod, mod_vals, step_value_caches, step_tmp_caches,
        crit_cache, it_stats, iteration_scalars_array, trial_caches, stop_crits
    )
end

function init_nondom_set!(
    value_caches, ::Type{F}; 
) where F <: AbstractFloat
    ndset = NondominatedSet(F, Int[])
    fx_vecs = (cached_fx(vcache) for vcache in value_caches)
    theta_vals = (cached_theta(vcache) for vcache in value_caches)
    init_nondom_set!(ndset, fx_vecs, theta_vals)
    keepat!(value_caches, ndset.extra)
    return ndset 
end

function init_nondom_set!(ndset, fx_vecs, theta_vals)
    for (i, (fx, theta)) in enumerate(zip(fx_vecs, theta_vals))
        add_and_remove_dominated!(ndset, fx, theta, i)
    end
    return nothing
end

function nondominated_caches(ndset, @nospecialize(filter_fn = sol -> !(sol.status[] isa IncompatibleNormalStep)))
    ##sols = filter(sol -> !isa(sol.status[], IncompatibleNormalStep), ndset.extra)
    sols = filter(filter_fn, ndset.extra)
    return [sol.vals for sol in sols]
end
function nondominated_vars(ndset, args...)
    caches = nondominated_caches(ndset, args...)
    return mapreduce(cached_ξ, hcat, caches)
end
function nondominated_objectives(ndset, args...)
    caches = nondominated_caches(ndset, args...)
    return mapreduce(cached_fx, hcat, caches)
end
function nondominated_thetas(ndset, args...)
    caches = nondominated_caches(ndset, args...)
    return map(cached_theta, caches)
end