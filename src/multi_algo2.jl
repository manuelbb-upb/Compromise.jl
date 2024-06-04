import UUIDs: UUID, uuid4
import Dictionaries: Dictionary, getindices

struct SolutionStaleMark end

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

    status_ref :: Base.RefValue{statusType}
    gen_id_ref :: Base.RefValue{Int}
    sol_id_ref :: Base.RefValue{Int}
end

is_stale(::Any)=false
is_stale(::SolutionStaleMark)=true
is_stale(status_ref::Base.RefValue)=is_stale(status_ref[])
is_stale(sstructs::SolutionStructs)=is_stale(sstructs.status_ref)

is_not_stale(s)=!is_stale(s)

is_converged(::Any)=true
is_converged(::Nothing)=false
is_converged(status_ref::Base.RefValue)=is_converged(status_ref[])
is_converged(sstructs::SolutionStructs)=is_converged(sstructs.status_ref)

mark_stale!(status_ref::Base.RefValue)=(status_ref[] = SolutionStaleMark())
mark_stale!(sstructs::SolutionStructs)=mark_stale!(sstructs.status_ref)

function Base.show(io::IO, sstructs::SolutionStructs)
    @unpack sol_id_ref = sstructs
    print(io, "SolutionStructs(; sol_id=$(sol_id_ref[]))")
end

@forward SolutionStructs.vals cached_x(sols::SolutionStructs)
@forward SolutionStructs.vals cached_ξ(sols::SolutionStructs)
@forward SolutionStructs.vals cached_fx(sols::SolutionStructs)
@forward SolutionStructs.vals cached_hx(sols::SolutionStructs)
@forward SolutionStructs.vals cached_gx(sols::SolutionStructs)
@forward SolutionStructs.vals cached_ξ(sols::SolutionStructs)
@forward SolutionStructs.vals cached_Ax(sols::SolutionStructs)
@forward SolutionStructs.vals cached_Ex(sols::SolutionStructs)
@forward SolutionStructs.vals cached_Ex_min_c(sols::SolutionStructs)
@forward SolutionStructs.vals cached_theta(sols::SolutionStructs)

function copy_solution_structs(sstructs)
    @unpack vals, step_vals, step_cache, iteration_scalars, mod, mod_vals, status_ref, 
        gen_id_ref, sol_id_ref = sstructs
    return SolutionStructs(
        deepcopy(vals),
        deepcopy(step_vals),
        deepcopy(step_cache),
        deepcopy(iteration_scalars),
        CE.copy_model(mod),
        deepcopy(mod_vals),
        deepcopy(status_ref),
        deepcopy(gen_id_ref),
        deepcopy(sol_id_ref)
    )
end

struct NondominatedSet{F<:AbstractFloat, E}
    fx_vecs :: Vector{Vector{F}}
    theta_vals :: Vector{F}

    gamma :: F
    extra :: E
end

function NondominatedSet{F}(;
    fx_vecs = Vector{Vector{F}}(),
    theta_vals = Vector{F}(),
    gamma = 0,
    extra :: E = nothing
) where {F<:AbstractFloat, E}
    return NondominatedSet{F, E}(fx_vecs, theta_vals, gamma, extra)
end

float_type(::NondominatedSet{F}) where F = F

keepat_extra!(ndset::NondominatedSet, ind)=keepat_extra!(ndset.extra, ind)
keepat_extra!(::Nothing, ind)=nothing
keepat_extra!(extra, ind)=keepat!(extra, ind)
push_extra!(ndset::NondominatedSet, ex)=push_extra!(ndset.extra, ex)
push_extra!(::Nothing, ex)=nothing
push_extra!(extra, ex)=push!(extra, ex)

function init_nondom_set!(ndset, fx_vecs, theta_vals)
    for (i, (fx, theta)) in enumerate(zip(fx_vecs, theta_vals))
        add_and_remove_dominated!(ndset, fx, theta, i)
    end
    return nothing
end

filter_min_theta(ndset::NondominatedSet) = max(0, minimum(ndset.theta_vals; init=0))

function is_not_dominated(ndset::NondominatedSet, fx, theta)
    @unpack fx_vecs, theta_vals = ndset
    is_acceptable = true
    for (fx_i, theta_i) in zip(fx_vecs, theta_vals)
        if filter_dominates(theta_i, fx_i, theta, fx)
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
    theta = cached_theta(cache_add)
    fx = cached_fx(cache_add)
    theta_filter, fx_filter = filter_offset_values(ndset, fx, theta)
    if filter_dominates(theta_filter, fx_filter, cached_theta(cache_test), cached_fx(cache_test))
        return false
    end
    return is_filter_acceptable(ndset, cache_test)
end

function add_to_filter!(ndset::NondominatedSet, cache, extra=nothing; gamma=nothing)
    return add_and_remove_dominated_no_check!(
        ndset, copy(cached_fx(cache)), cached_theta(cache), extra; gamma)
end

function add_to_filter!(ndset::NondominatedSet, sstructs::SolutionStructs; kwargs...)
    cache = sstructs.vals
    sol_id_ref = sstructs.sol_id_ref
    return add_to_filter!(ndset, cache, sol_id_ref[]; kwargs...)
end

function add_and_remove_dominated!(
    ndset::NondominatedSet, fx, theta, extra=nothing; gamma=nothing)
    if is_not_dominated(ndset, fx, theta)
        return add_and_remove_dominated_no_check!(ndset, fx, theta, extra; gamma)
    end
    return nothing
end

function add_and_remove_dominated_no_check!(
    ndset::NondominatedSet, fx, theta, extra=nothing; gamma=nothing
)  
    theta_j, fx_j = filter_offset_values(ndset, fx, theta, gamma)
    return add_without_offset_and_remove_dominated_no_check!(ndset, fx_j, theta_j, extra)
end

function add_without_offset_and_remove_dominated_no_check!(
    ndset::NondominatedSet, fx_j, theta_j, extra=nothing
)
    @unpack fx_vecs, theta_vals = ndset
    
    remain_flags = fill(true, length(fx_vecs))
    for (i, (fx_i, theta_i)) in enumerate(zip(fx_vecs, theta_vals))
        if filter_dominates(theta_j, fx_j, theta_i, fx_i)
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

function filter_dominates(theta_filter, fx_filter, theta, fx)
    return (
        theta_filter <= theta &&
        all(fx_filter .<= fx) &&
        (
            theta_filter < theta ||
            any(fx_filter .< fx )
        )
    )
end

function filter_offset_values(ndset, fx, theta, gamma=nothing)
    if isnothing(gamma)
        gamma = ndset.gamma
    end
    offset_j = gamma * theta
    fx_j = fx .- offset_j
    theta_j = theta - offset_j
    #theta_j = max(0, theta_j)
    return theta_j, fx_j
end

function add_to_filter_and_mark_population!(ndset, population, sstructs;indent=0, log_level=Info)
    ## assume that without `sstructs` all elements of `population` that
    ## are not stale are filter-acceptable
    theta_new, fx_new = filter_offset_values(ndset, cached_fx(sstructs), cached_theta(sstructs))
    mark_dominated_as_stale!(population, theta_new, fx_new; indent, log_level)
    return add_without_offset_and_remove_dominated_no_check!(ndset, fx_new, theta_new, sstructs.sol_id_ref[])
end

function mark_dominated_as_stale!(population, theta_new, fx_new;indent=0, log_level=Info)
    for (sol_id, pstructs) in pairs(population.dict)
        is_stale(pstructs) && continue
        theta = cached_theta(pstructs)
        fx = cached_fx(pstructs)
        if filter_dominates(theta_new, fx_new, theta, fx)
            @logmsg log_level "$(indent_str(indent)) 💀 removing dominated solution $(sol_id)."
            #delete!(population.dict, sol_id)
            mark_stale!(pstructs)
        end
    end
    return nothing
end

struct CountedPopulation{S}
    dict :: Dictionary{Int, S}
    counter :: Base.RefValue{Int}
end

function is_not_dominated(population::CountedPopulation, fx, theta)
    is_acceptable = true
    for pstructs in population.dict
        theta_i = cached_theta(pstructs)
        fx_i = cached_fx(pstructs)
        if filter_dominates(theta_i, fx_i, theta, fx)
            is_acceptable = false
            break
        end
    end
    return is_acceptable
end

function Base.show(io::IO, population::CountedPopulation)
    print(io, "CountedPopulation(size=$(length(population.dict)), counter=$(population.counter[]))")
end

function add_to_population!(population, sstructs; prune=false, log_level=Info, indent=0)
    theta_new = cached_theta(sstructs)
    fx_new = cached_fx(sstructs)
    mark_dominated_as_stale!(population, theta_new, fx_new; log_level, indent)
    if prune
        remove_stale!(population)
    end
    return add_to_population_no_check!(population, sstructs)
end

function add_to_population_no_check!(population, sstructs)
    id = sstructs.sol_id_ref[] = population.counter[] += 1
    insert!(population.dict, id, sstructs)# population.dict[id] = sstructs
    return population
end

function singleton_population(sstructs::S) where S
    sol_id = sstructs.sol_id_ref[] = 1
    return CountedPopulation(Dictionary{Int,S}(Int[sol_id,], S[sstructs,]), Ref(1))
end

for fname in (
    :cached_x, :cached_ξ, :cached_fx, :cached_gx, :cached_hx,
    :cached_Ax, :cached_Ex,
    :cached_Ax_min_b, :cached_Ex_min_c,
    :cached_theta
)
    @eval function $(fname)(population::CountedPopulation)
        mapreduce($(fname), hcat, values(population.dict))
    end
end

function make_first_solution(
    vals, step_vals, step_cache, iteration_scalars, mod, mod_vals,
    ::Type{status_type}=Any
) where {status_type}
    sstructs = SolutionStructs(
        vals, step_vals, step_cache, iteration_scalars, mod, mod_vals,
        Ref{status_type}(nothing),
        Ref(1), 
        Ref(1)
    )
    return sstructs
end

function initialize_population(
    X, sstructs, mop, scaler, lin_cons;
)
    population = singleton_population(sstructs)
    @unpack vals = sstructs
    for ξ0 = Iterators.drop(eachcol(X), 1)
        _vals = deepcopy(vals)
        Base.copyto!(cached_ξ(_vals), ξ0)
        project_into_box!(cached_ξ(_vals), lin_cons)
        scale!(cached_x(_vals), scaler, cached_ξ(_vals))
        @ignorebreak eval_mop!(_vals, mop)
        if is_not_dominated(population, cached_fx(_vals), cached_theta(_vals))
            _sstructs = copy_solution_structs(sstructs)
            universal_copy!(_sstructs.vals, _vals)
            add_to_population_no_check!(population, _sstructs)
        end
    end
    
    return population
end
#=
function sync(population, ndset)
    #synced_dict = filter( ((id, sol),) -> id in ndset.extra, pairs(population.dict) )
    #synced_dict = getindices(population.dict, intersect(keys(population.dict), ndset.extra))
    #return CountedPopulation(synced_dict, population.counter)
    #intersect!(keys(population.dict), ndset.extra) # does not work! does not delete values as expected
    filter!( ((id, sol),) -> id in ndset.extra, pairs(population.dict) )
    return population
end
=#

function remove_stale!(population)
    pdict = population.dict
    filter!(is_not_stale, pdict)
    return population  
end

function _status_type(mop, mod, stop_crits)
    unwrapped_stype = Union{
        SolutionStaleMark,
        InfeasibleStopping,     # probably not used
        stop_type(mod), 
        stop_type(mop), 
        stop_crit_type(stop_crits)
    }
    return Union{Nothing, unwrapped_stype, WrappedStoppingCriterion{<:unwrapped_stype}}
end

function optimize_set(
    X::RMat, 
    mop;
    algo_opts = AlgorithmOptions(), 
    user_callback = NoUserCallback(),
    filter_gamma = 1e-6,
)
    @unpack log_level, max_iter = algo_opts
    @reset algo_opts.max_iter = typemax(Int)
    optimizer_caches = initialize_structs(mop, @view(X[:, 1]), algo_opts, user_callback)

    ndset = NondominatedSet{float_type(mop)}(gamma = filter_gamma, extra = Int[])
    if optimizer_caches isa AbstractStoppingCriterion
        return ndset
    end
    
    @unpack (
        mop, mod, scaler, lin_cons, scaled_cons, vals, vals_tmp,
        mod_vals, step_vals, step_cache, crit_cache, trial_caches, 
        iteration_status, iteration_scalars, stop_crits,
    ) = optimizer_caches

    sstructs = make_first_solution(
        vals, step_vals, step_cache, iteration_scalars, mod, mod_vals,
        _status_type(mop, mod, stop_crits)
    )
    population = initialize_population(X, sstructs, mop, scaler, lin_cons)

    for it_index = 1 : max_iter
        gen_id = it_index
        @logmsg log_level """\n
        ======================================================
        ==  ITERATION $(it_index)
        ==  pop size = $(length(population.dict))
        ======================================================
        ======================================================"""
        step_normally!(population, ndset, optimizer_caches, algo_opts; gen_id)
        restore_population!(population, ndset, optimizer_caches, algo_opts; gen_id,)

        propagate_population!(population, ndset, optimizer_caches, algo_opts; gen_id,)

        if isempty(population.dict)
            break
        end
        if all(is_converged, population.dict)
            break
        end
    end

    return population
end
function propagate_population!(population, ndset, optimizer_caches, algo_opts; gen_id,)
    @unpack mop, scaler, lin_cons, scaled_cons, vals_tmp, trial_caches,
        crit_cache, iteration_status, stop_crits = optimizer_caches    
    
    @logmsg algo_opts.log_level """\n
    ~~~~~~~~~~~~~~~~~~~~~ EVOLUTION ~~~~~~~~~~~~~~~~~~~~~"""
    return propagate_population!(
        population, ndset, 
        mop, scaler, lin_cons, scaled_cons,
        vals_tmp, trial_caches, crit_cache, iteration_status, stop_crits, algo_opts
        ;
        gen_id,
    )
end

function propagate_population!(
    population, ndset,
    mop, scaler, lin_cons, scaled_cons, vals_tmp, trial_caches,
    crit_cache, iteration_status, stop_crits, algo_opts; 
    gen_id, 
)        
    @unpack log_level = algo_opts
    for (sol_id, sstructs) in pairs(population.dict)
        @assert sol_id == sstructs.sol_id_ref[]
        is_stale(sstructs) && continue
        is_converged(sstructs) && continue
        sstructs.gen_id_ref[] > gen_id && continue
        log_solution_values(sstructs, log_level; normal_step=true)

        @unpack vals, step_vals, step_cache, iteration_scalars, mod, mod_vals = sstructs
        delta = iteration_scalars.delta
        @logmsg log_level "* Computing a descent step."
        do_descent_step!(
            step_cache, step_vals, delta, mop, mod, scaler, lin_cons, scaled_cons, vals, mod_vals;
            log_level
        )

        @logmsg log_level " - Criticality χ=$(step_vals.crit_ref[]), ‖d‖₂=$(LA.norm(step_vals.d)), ‖s‖₂=$(LA.norm(step_vals.s))."

        status = check_stopping_criterion(
            stop_crits, CheckPostDescentStep(), 
            mop, mod, scaler, lin_cons, scaled_cons, vals, vals_tmp,
            mod_vals, ndset, step_vals, step_cache, crit_cache, trial_caches, 
            iteration_status, iteration_scalars, stop_crits,
            algo_opts
        )
        if status isa AbstractStoppingCriterion
            sstructs.status_ref[] = status
            continue
        end
         ## For the convergence analysis to work, we also have to have the Criticality Routine:
        delta_new = criticality_routine!(
            mop, mod, scaler, lin_cons, scaled_cons, vals, vals_tmp,
            mod_vals, ndset, step_vals, step_cache, crit_cache, trial_caches, 
            iteration_status, iteration_scalars, stop_crits,
            algo_opts;
            indent = 1
        )
        if delta_new isa AbstractStoppingCriterion
            sstructs.status_ref[] = delta_new
            continue
        end
        delta = iteration_scalars.delta = delta_new

        test_trial_point!(
            sstructs, ndset, population,
            mop, scaler, lin_cons, scaled_cons, vals_tmp,
            step_cache, crit_cache, trial_caches, 
            iteration_status, stop_crits, algo_opts;
            gen_id, indent = 1
        ) 
    end
    remove_stale!(population)
end

function test_trial_point!(
    sstructs, ndset, population, 
    mop, scaler, lin_cons, scaled_cons, vals_tmp,
    step_cache, crit_cache, trial_caches, 
    iteration_status, stop_crits, algo_opts;
    gen_id, indent = 1
)
    @unpack log_level = algo_opts
    @unpack vals, step_vals, mod_vals, mod, iteration_scalars = sstructs
    
    status = _test_trial_point!(
        sstructs, ndset, 
        mop, scaler, trial_caches, vals_tmp, algo_opts; 
        indent=1
    )
    if status isa AbstractStoppingCriterion
        sstructs.status_ref[] = status
        return status        
    end
    is_good_trial_point, augment_filter = status

    status = check_stopping_criterion(
        stop_crits, CheckPostIteration(),
        mop, mod, scaler, lin_cons, scaled_cons, vals, vals_tmp,
        mod_vals, ndset, step_vals, step_cache, crit_cache, trial_caches, 
        iteration_status, iteration_scalars, stop_crits,
        algo_opts
    )
    if status isa AbstractStoppingCriterion
        sstructs.status_ref[] = status
        return status        
    end
    status = process_trial_point!(mod, vals_tmp, is_good_trial_point)
    if status isa AbstractStoppingCriterion
        sstructs.status_ref[] = status
        return status        
    end
    
    if is_good_trial_point
        @logmsg log_level "$(indent_str(indent)) Trial point is accepted..."
        _sstructs = copy_solution_structs(sstructs)
        universal_copy!(_sstructs.vals, vals_tmp)
        _sstructs.gen_id_ref[] = gen_id + 1
        add_to_population!(population, _sstructs; log_level, indent=2)
        @logmsg log_level "$(indent_str(indent)) Trial point id = $(_sstructs.sol_id_ref[])"
    else
        @logmsg log_level "$(indent_str(indent)) Trial point is not accepted."
    end

    if augment_filter
        @logmsg log_level "$(indent_str(indent)) Adding $(sstructs.sol_id_ref[]) to filter."
        #mark_stale!(sstructs) 
        add_to_filter_and_mark_population!(ndset, population, sstructs; log_level, indent=2)
    end
    return nothing
end

function _test_trial_point!(
    sstructs, ndset, 
    mop, scaler, trial_caches, vals_tmp , algo_opts;
    indent=0,
)
    @unpack vals, step_vals, mod_vals, mod, iteration_scalars = sstructs
    
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

    @unpack diff_x, diff_fx, diff_fx_mod = trial_caches
    @. diff_x = x - xs
    @. diff_fx = fx - fxs
    @. diff_fx_mod = mx - mxs
    
    rho_ratio = diff_fx ./ diff_fx_mod
    rho_accept = maximum(rho_ratio)
    rho_radius = minimum(rho_ratio)

    @unpack log_level = algo_opts
    @logmsg log_level """
        \n$(indent_str(indent))- x - xs = $(pretty_row_vec(diff_x)), 
        $(indent_str(indent))- fx - fxs = $(pretty_row_vec(diff_fx)), 
        $(indent_str(indent))- mx - mxs = $(pretty_row_vec(diff_fx_mod))
        $(indent_str(indent))- rho_accept = $(rho_accept)
        $(indent_str(indent))- rho_radius = $(rho_radius)"""

    θx = cached_theta(vals)
    @unpack psi_theta, kappa_theta, nu_accept, nu_success, gamma_shrink, gamma_shrink_much, 
        gamma_grow, delta_max = algo_opts
    delta = delta_new = iteration_scalars.delta
    is_good_trial_point = true
    augment_filter = false
    if !is_filter_acceptable(ndset, vals_tmp, vals)
        # trial point rejected
        # radius is reduced
        is_good_trial_point = false
        delta_new = gamma_shrink_much * delta
    else
        if any(diff_fx_mod .< kappa_theta * θx^psi_theta)
            # (theta step)
            # trial point is accepted    
            is_good_trial_point = true
            # radius is kept as it is or updated based on rho
            delta_new = delta
            # sstructs is added to filter (sync population!)
            # (adding sstructs to filter makes it incompatible with filter, 
            # it is removed from population
            augment_filter = true
        else
            if rho_accept >= nu_accept
                # trial point is accepted
                is_good_trial_point = true
                if rho_radius >= nu_accept
                    if rho_radius >= nu_success
                        # increase radius
                        delta_new = min(delta_max, gamma_grow * delta)
                    else
                        # only sligthly decrease radius
                        gamma_shrink_little = gamma_shrink + ((1 - gamma_shrink)/2)
                        delta_new = gamma_shrink_little * delta
                    end
                else
                    # decrease radius
                    delta_new = gamma_shrink * delta
                end
            else
                # (innacceptable)
                # trial point is rejected
                is_good_trial_point = false
                # radius is reduced much
                delta_new = gamma_shrink_much * delta
            end
        end        
    end
    @logmsg log_level "$(indent_str(indent)) Δ_new = $(delta_new) -- was Δ=$(delta)"
    iteration_scalars.delta = delta_new

    return is_good_trial_point, augment_filter
end
   

function restore_population!(population, ndset, optimizer_caches, algo_opts; gen_id)
    @unpack mop, scaler, lin_cons, scaled_cons, vals_tmp, trial_caches,
        crit_cache, iteration_status, stop_crits = optimizer_caches    
    return restore_population!(
        population, ndset, 
        mop, scaler, lin_cons, scaled_cons,
        vals_tmp, trial_caches, crit_cache, iteration_status, stop_crits, algo_opts
        ;
        gen_id,
    )
end
function restore_population!(
    population, ndset,
    mop, scaler, lin_cons, scaled_cons, vals_tmp, trial_caches,
    crit_cache, iteration_status, stop_crits, algo_opts; 
    gen_id, 
)        
    best_index = 0
    best_n_norm = Inf
    for (sol_id, sstructs) in pairs(population.dict)
        if is_not_stale(sstructs)
            best_index = 0
            break
        end
        sstructs.gen_id_ref[] > gen_id && continue

        @unpack step_vals = sstructs
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

    if !iszero(best_index)
        @logmsg algo_opts.log_level """\n
        ~~~~~~~~~~~~~~~~~~~~~ RESTORATION ~~~~~~~~~~~~~~~~~~~~~"""
    
        sstructs = copy_solution_structs(population.dict[best_index])
        add_to_population_no_check!(population, sstructs)    # to set `sol_id_ref`, `do_restoration!` might add to filter
        sstructs.gen_id_ref[] = gen_id + 1
        @unpack vals, step_vals, step_cache, iteration_scalars, mod, mod_vals = sstructs
        restoration_code = do_restoration(
            mop, mod, scaler, lin_cons, scaled_cons, vals, vals_tmp,
            mod_vals, ndset, step_vals, step_cache, crit_cache, trial_caches, 
            iteration_status, iteration_scalars, stop_crits,
            algo_opts
        )
        if restoration_code isa AbstractStoppingCriterion
            mark_stale!(sstructs)
        end    
    end
    
    return remove_stale!(population)
end

function step_normally!(
    population, ndset, optimizer_caches, algo_opts;
    gen_id
)
    @unpack mop, scaler, lin_cons, scaled_cons, vals_tmp, trial_caches,
        crit_cache, iteration_status, stop_crits = optimizer_caches
    @logmsg algo_opts.log_level """\n
    ~~~~~~~~~~~~~~~~~~~~~ NORMAL STEPS ~~~~~~~~~~~~~~~~~~~~~"""
    return step_normally!(
        population, ndset,
        mop, scaler, lin_cons, scaled_cons, vals_tmp, trial_caches,
        crit_cache, iteration_status, stop_crits,
        algo_opts
        ;
        gen_id
    )
end
             
function step_normally!(
    population, ndset,
    mop, scaler, lin_cons, scaled_cons, vals_tmp, trial_caches,
    crit_cache, iteration_status, stop_crits,
    algo_opts
    ; 
    check_filter=true,
    gen_id
)
    @unpack log_level = algo_opts
    indent = 0
    sort!(population.dict; by=cached_theta, rev=true)
    for (sol_id, sstructs) in pairs(population.dict)
        @assert sol_id == sstructs.sol_id_ref[]
        is_stale(sstructs) && continue
        is_converged(sstructs) && continue
        sstructs.gen_id_ref[] > gen_id && continue
        if check_filter
            if !is_filter_acceptable(ndset, sstructs)
                #delete!(population.dict, sol_id)
                mark_stale!(sstructs)
                continue
            end
        end

        @unpack vals, step_vals, step_cache, iteration_scalars, mod, mod_vals = sstructs
        delta = iteration_scalars.delta
        status = check_stopping_criterion(
            stop_crits, CheckPreIteration(),
            mop, mod, scaler, lin_cons, scaled_cons, vals, vals_tmp,
            mod_vals, ndset, step_vals, step_cache, crit_cache, trial_caches, 
            iteration_status, iteration_scalars, stop_crits,
            algo_opts
        )
        if status isa AbstractStoppingCriterion
            sstructs.status_ref[] = status
            continue
        end
        log_solution_values(sstructs, log_level)

        @logmsg log_level "* Updating Surrogates."
        status = update_models!(mod, delta, scaler, vals, scaled_cons; log_level, indent)
        if status isa AbstractStoppingCriterion
            sstructs.status_ref[] = status
            log_stop_code(status, log_level)
            continue
        end
        status = eval_and_diff_mod!(mod_vals, mod, cached_x(vals))
        if status isa AbstractStoppingCriterion
            sstructs.status_ref[] = status
            log_stop_code(status, log_level)
            continue
        end

        do_normal_step!(
            step_cache, step_vals, delta, mop, mod, scaler, lin_cons, scaled_cons, vals, mod_vals;
            log_level, indent=0
        )
        n_is_compatible = compatibility_test(step_vals.n, algo_opts, delta)

        if !n_is_compatible
            @logmsg log_level "… normal step not compatible. Augmenting filter."
            add_to_filter_and_mark_population!(ndset, population, sstructs; log_level)
        end
    end
    #remove_stale!(population)
    return population
end

function log_solution_values(sstructs, log_level; normal_step=false)
    delta = sstructs.iteration_scalars.delta
    sol_id = sstructs.sol_id_ref[]
    msg = """
        \n###########################
        # sol_id = $(sol_id).
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