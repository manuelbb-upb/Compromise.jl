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
include("algo_init.jl")
include("main_algo.jl")
include("outer_algos.jl")
include("multi_algo2.jl")

export optimize, optimize_with_algo, 
    AlgorithmOptions, ThreadedOuterAlgorithmOptions, SequentialOuterAlgorithmOptions
export opt_cache, opt_vars, opt_objectives, opt_nl_eq_constraints, opt_nl_ineq_constraints,
    opt_lin_eq_constraints, opt_lin_ineq_constraints, opt_constraint_violation, opt_stop_code,
    opt_surrogate
export MutableMOP, add_objectives!, add_nl_ineq_constraints!, add_nl_eq_constraints!
export TypedMOP, NonlinearFunction
end