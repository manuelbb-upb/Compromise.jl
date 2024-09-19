
function initialize_structs( 
    MOP::AbstractMOP, ξ0::RVec, algo_opts::AlgorithmOptions,
    user_callback :: AbstractStoppingCriterion = NoUserCallback();
    log_time::Bool=true
)

    @assert !isempty(ξ0) "Starting point array `x0` is empty."
    @assert dim_objectives(MOP) > 0 "Objective Vector dimension of problem is zero."
     
    n_vars = length(ξ0)
    @assert dim_vars(MOP) == n_vars "Mismatch in number of variables."

    ## IT_INITIALIZATION (Iteration 0)
    stats = @timed begin
        mop = initialize(MOP)
        initialize_structs_from_mop(mop, ξ0, algo_opts, user_callback)
    end
    if log_time 
        @logmsg algo_opts.log_level "Initialization complete after $(stats.time) sec."
    end
    return stats.value
end

function initialize_structs_from_mop(
    mop::AbstractMOP, ξ0::RVec, algo_opts::AlgorithmOptions,
    user_callback :: AbstractStoppingCriterion = NoUserCallback(),
)
    indent = 0
    T = float_type(mop)
    @unpack log_level = algo_opts
    NaNT = T(NaN)

    n_vars = Int(dim_vars(mop))     # Cthulhu tells me that `dim_vars` is unstable. TODO make a wrapper/safeguard
    ## struct holding constant linear constraint informaton (lb, ub, A_b, E_c)
    ## set these first, because they are needed to initialize a scaler 
    ## if `algo_opts.scaler_cfg==:box`:
    lin_cons = init_lin_cons(mop)

    ## initialize a scaler according to configuration
    scaler = init_scaler(algo_opts.scaler_cfg, lin_cons, n_vars)
    ## whenever the scaler changes, we have to re-scale the linear constraints
    scaled_cons = deepcopy(lin_cons)
    update_lin_cons!(scaled_cons, scaler, lin_cons)

    ## pre-allocate surrogates `mod`
    ## (they are not trained yet)
    mod = init_models(
        mop, scaler; 
        delta_max = algo_opts.delta_max,
        require_fully_linear = algo_opts.require_fully_linear_models
    )

    ## caches for working arrays x, fx, hx, gx, Ex, Ax, …
    vals = WrappedMOPCache(init_value_caches(mop))

    ## (perform 1 evaluation to set values already)
    Base.copyto!(cached_ξ(vals), ξ0)
    project_into_box!(cached_ξ(vals), lin_cons)
    scale!(cached_x(vals), scaler, cached_ξ(vals))
    @ignoraise eval_mop!(vals, mop) indent

    vals_tmp = deepcopy(vals)
     
    ## caches for surrogate value vectors fx, hx, gx, Dfx, Dhx, Dgx
    ## (values not set yet, only after training)
    mod_vals = init_value_caches(mod)
 
    ## pre-allocate working arrays for normal and descent step calculation:
    step_vals = init_step_vals(vals)
    step_cache = init_step_cache(algo_opts.step_config, vals, mod_vals)

    crit_cache = CriticalityRoutineCache(;
        delta = NaNT,
        num_crit_loops = -1,
        step_vals = deepcopy(step_vals),
    )

    ## initialize empty filter
    filter = StandardFilter{T}()

    ## caches for trial point checking:
    trial_caches = TrialCaches{T}(;
        delta=NaNT,
        diff_x = array(T, dim_vars(mop)),
        diff_fx = array(T, dim_objectives(mop)),
        diff_fx_mod = array(T, dim_objectives(mop))
    )

    ## finally, compose information about the 0-th iteration for next_iterations:
    iteration_status = IterationStatus{T}(;
        iteration_classification = IT_INITIALIZATION,
        rho = NaNT,
        rho_classification = RHO_NAN,
        radius_update_result = RADIUS_INITIAL,
    )

    @unpack delta_init = algo_opts
    iteration_scalars = IterationScalars{T}(;
        it_index = 0,
        delta = T(delta_init),
    )
    
    ## prepare stopping
    stop_crits = stopping_criteria(algo_opts, user_callback)

    return OptimizerCaches(;
        mop, mod, scaler, lin_cons, scaled_cons, vals, vals_tmp, mod_vals, filter, 
        step_vals, step_cache, crit_cache, trial_caches, iteration_status, iteration_scalars, 
        stop_crits
    )
end

function init_scaler(scaler_cfg::Symbol, lin_cons, n_vars)
    return init_scaler(Val(scaler_cfg), lin_cons, n_vars)
end

function init_scaler(::Val, lin_cons, n_vars)
    return IdentityScaler(n_vars)
end

function init_scaler(v::Val{:box}, lin_cons, n_vars)
    @unpack lb, ub = lin_cons
    return init_scaler(v, lb, ub, n_vars)
end
function init_scaler(::Val{:box}, lb, ub, n_vars)
    return init_box_scaler(lb, ub, n_vars)
end

function update_lin_cons!(scaled_cons, scaler, lin_cons)
    scale_lin_cons!(scaled_cons, scaler, lin_cons)
    return nothing
end

function init_step_vals(vals)
    T = float_type(vals) 
    return StepValueArrays(
        dim_vars(vals), 
        dim_objectives(vals), 
        T
    )
end