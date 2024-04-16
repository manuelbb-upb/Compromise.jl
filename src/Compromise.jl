module Compromise

# ## Imports

# ### External Dependencies
# We use the macros in `Parameters` quite often, because of their convenience:
import UnPack
import UnPack: @unpack
import Parameters: @with_kw
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

import Accessors
import Accessors: PropertyLens
@reexport import Accessors: @set, @reset

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

# With the external dependencies available, we can include global type definitions and constants:
include("macros.jl")
include("globals.jl")

# ### Interfaces and Algorithm Types
# Abstract types and interfaces to handle multi-objective optimization problems...
include("mop.jl")
# ... and how to model them:
include("surrogate.jl")
# The cache interface is defined separately:
include("value_caches.jl")
# Tools to scale and unscale variables:
include("scaling.jl")
# Implementations of Filter(s):
include("filter.jl")
# Types and methods to compute inexact normal steps and descent steps:
include("steps.jl")
include("steepest_descent.jl")
# The restoration utilities:
include("restoration.jl")
# Trial point testing:
include("trial.jl")
# Criticality Routine has its own file too:
include("criticality_routine.jl")
# Stopping criteria:
include("stopping.jl")
# Pseudo lock:
include("concurrent_locks.jl")

# Miscellaneous functions to be included when all types are defined (but before sub-modules)
include("utils.jl")

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
        @require ConcurrentUtils = "3df5f688-6c4c-4767-8685-17f5ad261477" begin
            include("../ext/ConcurrentRWLockExt/ConcurrentRWLockExt.jl")
            import .ConcurrentRWLockExt
        end
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
        if isdefined(@__MODULE__, :ConcurrentRWLockExt)
            return ConcurrentRWLockExt.ConcurrentRWLock()
        end
        return nothing
    else
        m = Base.get_extension(@__MODULE__, :ConcurrentRWLockExt)
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
include("SimpleMOP/simple_mop.jl")

# ## The Algorithm
# (This still neads some re-factoring...)

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
    MOP::AbstractMOP, outer_opts::ThreadedOuterAlgorithmOptions, ξ0::RVecOrMat; 
    user_callback :: AbstractStoppingCriterion = NoUserCallback(),
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
    user_callback :: AbstractStoppingCriterion = NoUserCallback(),
)
    optimizer_caches = initialize_structs(MOP, ξ0, algo_opts, user_callback)
    return optimize!(optimizer_caches, algo_opts, ξ0)
end

function optimize!(optimizer_caches::OptimizerCaches, algo_opts, ξ0)
    @unpack vals = optimizer_caches
     
    _ξ0 = copy(cached_ξ(vals))    # for final ReturnObject

    stop_code = nothing
    while true
        time_start = time()
        @ignorebreak stop_code = do_iteration!(optimizer_caches, algo_opts)
        time_stop = time()
        @logmsg algo_opts.log_level "Iteration was $(time_stop - time_start) sec."
    end
    log_stop_code(stop_code, algo_opts.log_level)
    return ReturnObject(_ξ0, vals, stop_code, mod)
end

function optimize!(optimizer_caches::AbstractStoppingCriterion, algo_opts, ξ0)
    log_stop_code(optimizer_caches, algo_opts.log_level)
    return ReturnObject(ξ0, nothing, optimizer_caches, nothing)
end

function log_stop_code(crit, log_level)
    @logmsg log_level stop_message(crit)
end

function initialize_structs( 
    MOP::AbstractMOP{T}, ξ0::RVec, algo_opts::AlgorithmOptions,
    user_callback :: AbstractStoppingCriterion = NoUserCallback(),
) where T<:AbstractFloat

    @assert !isempty(ξ0) "Starting point array `x0` is empty."
    @assert dim_objectives(MOP) > 0 "Objective Vector dimension of problem is zero."
     
    n_vars = length(ξ0)
    @assert dim_vars(MOP) == n_vars "Mismatch in number of variables."

    ## INITIALIZATION (Iteration 0)
    stats = @timed begin
        mop = initialize(MOP)
        initialize_structs_from_mop(mop, ξ0, algo_opts, user_callback)
    end
    @logmsg algo_opts.log_level "Initialization complete after $(stats.time) sec."
    return stats.value
end
function initialize_structs_from_mop(
    mop::AbstractMOP{T}, ξ0::RVec, algo_opts::AlgorithmOptions,
    user_callback :: AbstractStoppingCriterion = NoUserCallback(),
) where T<:AbstractFloat
    indent = 0
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
    iteration_status = IterationStatus(;
        iteration_type = INITIALIZATION,
        radius_change = INITIAL_RADIUS,
        step_class = INITIAL_STEP,
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

function do_iteration!(
    optimizer_caches, algo_opts
)
    indent = 0

    @unpack (
        mop, mod, scaler, lin_cons, scaled_cons, vals, vals_tmp, mod_vals, filter, 
        step_vals, step_cache, crit_cache, trial_caches, iteration_status, iteration_scalars, 
        stop_crits, 
    ) = optimizer_caches
   
    @unpack log_level = algo_opts
    
    iteration_scalars.it_index += 1
    @unpack it_index, delta = iteration_scalars

    @ignoraise check_stopping_criterion(
        stop_crits, CheckPreIteration(), optimizer_caches, algo_opts) indent

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
    radius_has_changed = Int8(iteration_status.radius_change) >= 0
    models_valid = (
        iteration_status.iteration_type == RESTORATION ||
        !_trial_point_accepted(iteration_status) && !(depends_on_radius(mod) && radius_has_changed)
    )

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
        add_to_filter!(filter, cached_theta(vals), cached_Phi(vals))
        @ignoraise do_restoration(optimizer_caches, algo_opts) indent
    end

    @logmsg log_level "* Computing a descent step."
    @ignoraise do_descent_step!(
        step_cache, step_vals, delta, mop, mod, scaler, lin_cons, scaled_cons, vals, mod_vals;
        log_level
    ) indent
    
    @logmsg log_level " - Criticality χ=$(step_vals.crit_ref[]), ‖d‖₂=$(LA.norm(step_vals.d)), ‖s‖₂=$(LA.norm(step_vals.s))."

    @ignoraise check_stopping_criterion(
        stop_crits, CheckPostDescentStep(), optimizer_caches, algo_opts) indent
 
    ## For the convergence analysis to work, we also have to have the Criticality Routine:
    @ignoraise delta_new = criticality_routine!(optimizer_caches, algo_opts; indent) indent
    iteration_scalars.delta = delta_new
    delta = iteration_scalars.delta
    
    ## test if trial point is acceptable
    @ignoraise delta_new = test_trial_point!(optimizer_caches, algo_opts; indent) indent
    iteration_scalars.delta = delta_new
    delta = iteration_scalars.delta
    
    @ignoraise check_stopping_criterion(
        stop_crits, CheckPostIteration(), optimizer_caches, algo_opts) indent

    @ignoraise process_trial_point!(mod, vals_tmp, iteration_status) indent

    ## update filter
    if iteration_status.iteration_type == THETA_STEP
        add_to_filter!(filter, cached_theta(vals_tmp), cached_Phi(vals_tmp))
    end
   
    if _trial_point_accepted(iteration_status)
        ## accept trial point, mathematically ``xₖ₊₁ ← xₖ + sₖ``, pseudo-code `copyto!(vals, vals_tmp)`: 
        accept_trial_point!(vals, vals_tmp)
    end
    
    return nothing
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

export optimize, AlgorithmOptions
export opt_vars, opt_objectives, opt_nl_eq_constraints, opt_nl_ineq_constraints,
    opt_lin_eq_constraints, opt_lin_ineq_constraints, opt_constraint_violation, opt_stop_code
export MutableMOP, add_objectives!, add_nl_ineq_constraints!, add_nl_eq_constraints!
end