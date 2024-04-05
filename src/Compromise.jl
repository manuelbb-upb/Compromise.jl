module Compromise

# ## Imports

# ### External Dependencies
# We use the macros in `Parameters` quite often, because of their convenience:
import Parameters: @with_kw, @unpack
# Everything in this module needs at least some linear algebra:
import LinearAlgebra as LA
# Automatically annotated docstrings:
using DocStringExtensions

import Logging
import Logging: @logmsg, LogLevel, Info, Debug
import Printf: @sprintf

# Re-export symbols from important sub-modules
import Reexport: @reexport

# Make our types base equality on field value equality:
import StructHelpers: @batteries

# With the external dependencies available, we can include global type definitions and constants:
include("value_caches.jl")
include("types.jl")
include("utils.jl")
include("concurrent_locks.jl")

# #### Optimization Packages
# At some point, the choice of solver is meant to be configurable, with different
# extensions to choose from:
import JuMP     # LP and QP modelling for descent and normal steps
#src import COSMO    # actual QP solver
#src const DEFAULT_QP_OPTIMIZER=COSMO.Optimizer
import HiGHS
const DEFAULT_QP_OPTIMIZER=HiGHS.Optimizer
# For restoration we currently use `NLopt`. This is also meant to become 
# configurable...
import NLopt

# ### Interfaces and Algorithm Types
# Abstract types and interfaces to handle multi-objective optimization problems...
include("mop.jl")
# ... and how to model them:
include("surrogate.jl")
# The cache interface is defined separately:

# Tools to scale and unscale variables:
include("affine_scalers.jl")
# Implementations of Filter(s):
include("filter.jl")
# Types and methods to compute inexact normal steps and descent steps:
include("steps.jl")
# The restoration utilities:
include("restoration.jl")
# Trial point testing:
include("trial.jl")
# Criticality Routine has its own file too:
include("criticality_routine.jl")
# Stopping criteria:
include("stopping.jl")

# ### Internal Dependencies or Extensions
# Import operator types and interface definitions:
include("CompromiseEvaluators.jl")
using .CompromiseEvaluators
const CE = CompromiseEvaluators

# Import wrapper types to make user-provided functions conform to the operator interface:
include("evaluators/NonlinearFunctions.jl")
using .NonlinearFunctions
import .NonlinearFunctions: NonlinearParametricFunction
# Import the optional extension `ForwardDiffBackendExt`, if `ForwardDiff` is available:

using Requires
@static if !isdefined(Base, :get_extension)
    function __init__()
        @require ForwardDiff="f6369f11-7733-5829-9624-2563aa707210" begin
            include("../ext/ForwardDiffBackendExt/ForwardDiffBackendExt.jl")
            import .ForwardDiffBackendExt
        end
        @require ConcurrentUtils = "3df5f688-6c4c-4767-8685-17f5ad261477"
        include("../ext/ConcurrentRWLockExt/ConcurrentRWLockExt.jl")
        import .ConcurrentRWLockExt: ConcurrentRWLock
    end
end

# As of now, it is not easy to export stuff from extensions.
# We have this Getter instead:
function ForwardDiffBackend()
    if !isdefined(Base, :get_extension)
        if isdefined(@__MODULE__, :ForwardDiffBackendExt)
            return ForwardDiffBackendExt.ForwardDiffBackend()
        end
        return nothing
    else
        m = Base.get_extension(@__MODULE__, :ForwardDiffBackendExt)
        return isnothing(m) ? m : m.ForwardDiffBackend()
    end
end
function ConcurrentRWLock()
    if !isdefined(Base, :get_extension)
        if isdefined(@__MODULE__, :ConcurrentRWLock)
            return ConcurrentRWLock
        end
        return nothing
    else
        m = Base.get_extension(@__MODULE__, :ConcurrentRWLock)
        return isnothing(m) ? m : m.ConcurrentRWLock()
    end
end
export ForwardDiffBackend, ConcurrentRWLock

# Import Radial Basis Function surrogates:
include("evaluators/RBFModels/RBFModels.jl")
@reexport using .RBFModels

# Taylor Polynomial surrogates:
include("evaluators/TaylorPolynomialModels.jl")
@reexport using .TaylorPolynomialModels

# Exact “Surrogates”:
include("evaluators/ExactModels.jl")
@reexport using .ExactModels

# The helpers in `simple_mop.jl` depend on those model types:
include("simple_mop.jl")
export MutableMOP, add_objectives!, add_nl_ineq_constraints!, add_nl_eq_constraints!

# ## The Algorithm
# (This still neads some re-factoring...)

function init_scaler(scaler_cfg::Symbol, lin_cons)
    return init_scaler(Val(scaler_cfg), lin_cons)
end

function init_scaler(::Val{:box}, lin_cons)
    @unpack n_vars, lb, ub = lin_cons
    return init_box_scaler(lb, ub, n_vars)
end
init_scaler(::Val{:none}, lin_cons) = IdentityScaler(lin_cons.n_vars)

function init_lin_cons(mop)
    lb = lower_var_bounds(mop)
    ub = upper_var_bounds(mop)

    if !var_bounds_valid(lb, ub)
        error("Variable bounds inconsistent.")
    end

    A = lin_ineq_constraints_matrix(mop)
    b = lin_ineq_constraints_vector(mop)
    E = lin_eq_constraints_matrix(mop)
    c = lin_eq_constraints_vector(mop)

    return LinearConstraints(dim_vars(mop), lb, ub, A, b, E, c)
end

function scale_lin_cons!(trgt, scaler, lin_cons)
    @unpack A, b, E, c, lb, ub = lin_cons
    
    scale!(trgt.lb, scaler, lb)
    scale!(trgt.ub, scaler, ub)
    scale_eq!(trgt.A, trgt.b, scaler, A, b)
    scale_eq!(trgt.E, trgt.c, scaler, E, c)
    return nothing
end

function reset_lin_cons!(scaled_cons::LinearConstraints, lin_cons::LinearConstraints)
    for fn in fieldnames(LinearConstraints)
        universal_copy!(
            getfield(scaled_cons, fn), 
            getfield(lin_cons, fn)
        )
    end
    return nothing
end
function update_lin_cons!(scaled_cons, scaler, lin_cons)
    reset_lin_cons!(scaled_cons, lin_cons)
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

function compatibility_test_rhs(c_delta, c_mu, mu, Δ)
    return c_delta * min(Δ, c_mu + Δ^(1+mu))
end

function compatibility_test(n, c_delta, c_mu, mu, Δ)
    return LA.norm(n, Inf) <= compatibility_test_rhs(c_delta, c_mu, mu, Δ)
end

function compatibility_test(n, algo_opts, Δ)
    any(isnan.(n)) && return false
    @unpack c_delta, c_mu, mu = algo_opts
    return compatibility_test(n, c_delta, c_mu, mu, Δ)
end

function accept_trial_point!(vals, vals_tmp)
    return universal_copy!(vals, vals_tmp)
end

function stopping_criteria(algo_opts)
    return (
        MaxIterStopping(;num_max_iter=algo_opts.max_iter),
        MinimumRadiusStopping(;delta_min=algo_opts.stop_delta_min),
        ArgsRelTolStopping(;tol=algo_opts.stop_xtol_rel),
        ArgsAbsTolStopping(;tol=algo_opts.stop_xtol_abs),
        ValsRelTolStopping(;tol=algo_opts.stop_ftol_rel),
        ValsAbsTolStopping(;tol=algo_opts.stop_ftol_abs),
        CritAbsTolStopping(;
            crit_tol=algo_opts.stop_crit_tol_abs,
            theta_tol=algo_opts.stop_theta_tol_abs
        ),
        MaxCritLoopsStopping(;num=algo_opts.stop_max_crit_loops)
    )
end

function optimize(
    MOP::AbstractMOP, ξ0::RVec;
    algo_opts :: AlgorithmOptions = AlgorithmOptions(),
    @nospecialize(user_callback = NoUserCallback()),
)   
    return optimize_with_algo(MOP, algo_opts, ξ0; user_callback)
end

function optimize(
    MOP::AbstractMOP;
    algo_opts :: AlgorithmOptions = AlgorithmOptions(),
    @nospecialize(user_callback = NoUserCallback()),
)
    ξ0 = initial_vars(MOP)
    
    @assert !isnothing(ξ0) "`optimize` called without initial variable vector."

    return optimize_with_algo(MOP, algo_opts, ξ0; user_callback)
end

function optimize_with_algo(
    MOP::AbstractMOP, outer_opts::ThreadedOuterAlgorithmOptions, ξ0::RVecOrMat; 
    @nospecialize(user_callback = NoUserCallback()),
)
    _ξ0 = ξ0 isa RVec ? reshape(ξ0, :, 1) : ξ0

    algo_opts = outer_opts.inner_opts
    log_level = algo_opts.log_level

    ret_lock = ReentrantLock()
    ret = Any[]

    @logmsg log_level "Starting threaded evaluation of columns. (No Logging)."
    ncols = size(_ξ0, 2)
    Threads.@threads for i=1:ncols
        ξ0_i = lock(ret_lock) do
            @logmsg log_level "\tInspecting column $(i)."
            _ξ0[:, i]
        end
        r = Logging.with_logger(Logging.NullLogger()) do
            optimize_with_algo(MOP, algo_opts, ξ0_i; user_callback)
        end
        lock(ret_lock) do 
            push!(ret, r)
        end
    end

    return ret
end

function optimize_with_algo(
    MOP::AbstractMOP, algo_opts::AlgorithmOptions, ξ0; 
    @nospecialize(user_callback = NoUserCallback()),
)
    init_result = initialize_structs(MOP, ξ0, algo_opts)
    if !(isa(init_result, AbstractStoppingCriterion))
        (
            update_results, mop, mod, scaler, lin_cons, scaled_cons, 
            vals, vals_tmp, step_vals, mod_vals, filter, step_cache, crit_cache, stop_crits, 
            algo_opts
        ) = init_result

        _ξ0 = copy(cached_ξ(vals))

        stop_code = nothing
        @unpack log_level = algo_opts
        while true
            @ignorebreak stop_code log_level
            stop_code = do_iteration!(      # assume `copyto_model!`, otherwise, `mod, stop_code = do_iteration!...`
                update_results, mop, mod, scaler, lin_cons, scaled_cons, 
                vals, vals_tmp, step_vals, mod_vals, filter, step_cache, crit_cache, 
                stop_crits, algo_opts, user_callback
            )
        end
        return ReturnObject(_ξ0, vals, stop_code, mod)
    else
        return ReturnObject(ξ0, nothing, init_result, nothing)
    end
end

function initialize_structs( 
    MOP::AbstractMOP, ξ0::RVec, algo_opts :: AlgorithmOptions,
)
    @assert !isempty(ξ0) "Starting point array `x0` is empty."
    @assert dim_objectives(MOP) > 0 "Objective Vector dimension of problem is zero."
    
    @unpack log_level = algo_opts
    
    n_vars = length(ξ0)
    if dim_vars(MOP) != n_vars
        @error "`dim_vars(MOP)=$(dim_vars(mop))`, but `length(ξ0)=$(n_vars)`."
    end

    ## INITIALIZATION (Iteration 0)
    mop = initialize(MOP)
    T = float_type(mop)
    
    ## struct holding constant linear constraint informaton (lb, ub, A_b, E_c)
    ## set these first, because they are needed to initialize a scaler 
    ## if `algo_opts.scaler_cfg==:box`:
    lin_cons = init_lin_cons(mop)

    ## initialize a scaler according to configuration
    scaler = init_scaler(algo_opts.scaler_cfg, lin_cons)
    ## whenever the scaler changes, we have to re-scale the linear constraints
    scaled_cons = deepcopy(lin_cons)
    update_lin_cons!(scaled_cons, scaler, lin_cons)

    ## pre-allocate surrogates `mod`
    ## (they are not trained yet)
    mod = init_models(
        mop, n_vars, scaler; 
        delta_max = algo_opts.delta_max,
        require_fully_linear = algo_opts.require_fully_linear_models
    )

    ## caches for working arrays x, fx, hx, gx, Ex, Ax, …
    vals = init_value_caches(mop)

    ## (perform 1 evaluation to set values already)
    Base.copyto!(cached_ξ(vals), ξ0)
    project_into_box!(cached_ξ(vals), lin_cons)
    scale!(cached_x(vals), scaler, cached_ξ(vals))
    @ignoraise eval_mop!(vals, mop)

    vals_tmp = deepcopy(vals)
     
    ## caches for surrogate value vectors fx, hx, gx, Dfx, Dhx, Dgx
    ## (values not set yet, only after training)
    mod_vals = init_value_caches(mod)
 
    ## pre-allocate working arrays for normal and descent step calculation:
    step_vals = init_step_vals(vals)
    step_cache = init_step_cache(algo_opts.step_config, vals, mod_vals)

    crit_cache = CriticalityRoutineCache(
        deepcopy(mod_vals),
        deepcopy(step_vals),
        deepcopy(step_cache),
        universal_copy_model(mod)
    )

    ## initialize empty filter
    filter = StandardFilter{T}()

    ## finally, compose information about the 0-th iteration for next_iterations:
    update_results = init_update_results(T, n_vars, dim_objectives(mop), algo_opts.delta_init)
    
    ## prepare stopping
    stop_crits = stopping_criteria(algo_opts)
    return (
        update_results, mop, mod, scaler, lin_cons, scaled_cons, 
        vals, vals_tmp, step_vals, mod_vals, filter, step_cache, crit_cache, stop_crits, 
        algo_opts
    )
end

function do_iteration!(
    update_results, mop, mod, scaler, lin_cons, scaled_cons,
    vals, vals_tmp, step_vals, mod_vals, filter, step_cache, crit_cache, stop_crits, algo_opts,
    @nospecialize(user_callback);
)
    ## assumptions at the start of an iteration:
    ## * `vals` holds valid values for the stored argument vector `x`.
    ## * If the models `mod` are valid for `vals`, then `mod_vals` are valid, too.
    @unpack Δ_pre, Δ_post, it_index, point_has_changed = update_results
    last_it_stat = update_results.it_stat

    Δ = Δ_post
    radius_has_changed = Δ_pre != Δ_post
    it_index += 1

    @unpack log_level = algo_opts

    @ignoraise check_stopping_criteria(
        stop_crits, CheckPreIteration(), mop, scaler, lin_cons, scaled_cons, vals, filter, algo_opts;
        it_index, delta=Δ, indent=0
    )
    @ignoraise check_stopping_criterion(
        user_callback, CheckPreIteration(), mop, scaler, lin_cons, scaled_cons, vals, filter, algo_opts;
        it_index, delta=Δ, indent=0
    )

    @logmsg log_level """\n
    ###########################
    #  ITERATION $(it_index).
    ###########################
    Δ = $(Δ)
    θ = $(cached_theta(vals))
    ξ = $(pretty_row_vec(cached_ξ(vals)))
    x = $(pretty_row_vec(cached_x(vals)))
    fx = $(pretty_row_vec(cached_fx(vals)))
    """

    ## The models are valid
    ## - the last iteration was a successfull restoration iteration.
    ## - if the point has not changed and the models do not depend on the radius or  
    ##   radius has not changed neither.
    models_valid = (
        last_it_stat == RESTORATION || 
        !point_has_changed && !(depends_on_radius(mod) && radius_has_changed)
    )

    if !models_valid
        @logmsg log_level "* Updating Surrogates."
        @ignoraise update_models!(mod, Δ, mop, scaler, vals, scaled_cons, algo_opts; indent=0)
        @ignoraise eval_and_diff_mod!(mod_vals, mod, cached_x(vals))
    end
    @ignoraise do_normal_step!(
        step_cache, step_vals, Δ, mop, mod, scaler, lin_cons, scaled_cons, vals, mod_vals;
        it_index, log_level, indent=0
    )

    n_is_compatible = compatibility_test(step_vals.n, algo_opts, Δ)

    if !n_is_compatible
        ## Try to do a restoration
        @logmsg log_level "* Normal step incompatible. Trying restoration."
        add_to_filter!(filter, cached_theta(vals), cached_Phi(vals))
        @ignoraise do_restoration(
            mop, Δ, mod, scaler, lin_cons, scaled_cons,
            vals, vals_tmp, step_vals, mod_vals, filter, step_cache, update_results, 
            stop_crits, user_callback, algo_opts;
            it_index, indent=0
        )
    end

    @logmsg log_level "* Computing a descent step."
    @ignoraise do_descent_step!(
        step_cache, step_vals, Δ, mop, mod, scaler, lin_cons, scaled_cons, vals, mod_vals;
        log_level)
    
    @logmsg log_level " - Criticality χ=$(step_vals.crit_ref[]), ‖d‖₂=$(LA.norm(step_vals.d)), ‖s‖₂=$(LA.norm(step_vals.s))."

    @ignoraise check_stopping_criteria(
        stop_crits, CheckPostDescentStep(), mop, mod, scaler, lin_cons, scaled_cons, 
        vals, mod_vals, step_vals, filter, algo_opts; it_index, delta=Δ
    ) 
    @ignoraise check_stopping_criterion(
        user_callback, CheckPostDescentStep(), mop, mod, scaler, lin_cons, scaled_cons, 
        vals, mod_vals, step_vals, filter, algo_opts; it_index, delta=Δ
    )
   
    ## For the convergence analysis to work, we also have to have the Criticality Routine:
    @ignoraise Δ = criticality_routine(
        Δ, mod, step_vals, mod_vals, step_cache, crit_cache,
        mop, scaler, lin_cons, scaled_cons, vals, vals_tmp, stop_crits, algo_opts, 
        user_callback; it_index, indent=0
    )
    
    ## test if trial point is acceptable for filter and set missing meta data in `update_results`:
    @ignoraise test_trial_point!(
        update_results, vals_tmp, Δ, mop, scaler, filter, vals, mod_vals, step_vals, algo_opts;
        it_index, indent=0
    )

    @ignoraise process_trial_point!(mod, vals_tmp, update_results)

    ## update filter
    if update_results.it_stat == FILTER_ADD || update_results.it_stat == FILTER_ADD_SHRINK
        add_to_filter!(filter, cached_theta(vals_tmp), cached_Phi(vals_tmp))
    end

    @ignoraise finish_iteration(
        stop_crits, user_callback, update_results, mop, mod, scaler, lin_cons,
        scaled_cons, mod_vals, vals, vals_tmp, step_vals, filter, algo_opts; 
        it_index, indent=0
    )

    if update_results.point_has_changed
        ## accept trial point, mathematically ``xₖ₊₁ ← xₖ + sₖ``, pseudo-code `copyto!(vals, vals_tmp)`: 
        accept_trial_point!(vals, vals_tmp)
    end

    return nothing
end

function finish_iteration(
    stop_crits, user_callback,
    update_results, mop, mod, scaler, lin_cons,
    scaled_cons, mod_vals, vals, vals_tmp, step_vals, filter, algo_opts; 
    it_index, indent
)
    # Now, check stopping criteria.
    # It has to happen here, so that the criteria have access to both `vals` and `vals_tmp`.
    @ignoraise check_stopping_criteria(
        stop_crits, CheckPostIteration(), 
        update_results, mop, mod, scaler, lin_cons, scaled_cons,
        vals, mod_vals, vals_tmp, step_vals, filter, algo_opts;
        it_index, indent
    )
    @ignoraise check_stopping_criterion(
        user_callback, CheckPostIteration(), 
        update_results, mop, mod, scaler, lin_cons, scaled_cons,
        vals, mod_vals, vals_tmp, step_vals, filter, algo_opts;
        it_index, indent
    )
end

export optimize, AlgorithmOptions
export opt_vars, opt_objectives, opt_nl_eq_constraints, opt_nl_ineq_constraints,
    opt_lin_eq_constraints, opt_lin_ineq_constraints, opt_constraint_violation, opt_stop_code
end