# # Constants and Types
# In this file, we define most global constants and (abstract) types.
#
# This file structure is chosen as to avoid parse conflicts:
# Were we to use a type before it is defined, we would get an error.
# Thus, if we want inter-type dependencies, we have to be careful.
# Note, however, that we are allowed to call a type constructor before the type definition,
# just like any other function call is allowed before the definition has been parsed.

## Constants
const MOI = JuMP.MOI

# The default floating point number type:
const DEFAULT_FLOAT_TYPE = Float64
const InfDEF = DEFAULT_FLOAT_TYPE(Inf)
const NaNDEF = DEFAULT_FLOAT_TYPE(NaN)
const ZERO_DEF = zero(DEFAULT_FLOAT_TYPE)
const ONE_DEF = one(DEFAULT_FLOAT_TYPE)

# ### Shorthands for Real Arrays
# Our algorithm operates on real-valued vectors and matrices, these are shorthands:
const RVec = AbstractVector{<:Real}
const RMat = AbstractMatrix{<:Real}
const RVecOrMat = Union{RVec, RMat}
const RVecOrNothing = Union{RVec, Nothing}
const RMatOrNothing = Union{RMat, Nothing}
const RVecOrRMatOrNothing = Union{RVecOrMat, Nothing}

# ### Abstract Types
# When we want the algorithm to be customized, we offer abstract super types.

# #### Multi-Objective Problems
# The algorithm operates on `AbstractMOP`s, where MOP stands for multi-objective optimization
# problem. It's interface is defined in "src/mop.jl".
# An MOP is surrogated by an `AbstractMOPSurrogate`. The interface is defined in "src/model.jl".
abstract type AbstractMOP end
abstract type AbstractMOPSurrogate end

# The results of an MOP are stored in mutable caches:
abstract type AbstractValueCache{F<:AbstractFloat} end
abstract type AbstractMOPCache{F} <: AbstractValueCache{F} end
abstract type AbstractMOPSurrogateCache{F} <: AbstractValueCache{F} end

# #### Scaling
# Supertypes for objects that scale variables.
# At the moment, we only use constant scaling, and I am unsure If we should distinguish between
# constant and variable scaling.
# Current scalers are defined in "src/diagonal_scalers.jl"
abstract type AbstractAffineScaler end

# #### Step Computation
# Supertypes to configure and cache descent step computation.
# A config is meant to be provided by the user, and the cache is passed internally.
# At the moment, there is only steepest descent with standard Armijo line-search, which
# is implmented in "descent.jl"
abstract type AbstractStepConfig end
abstract type AbstractStepCache end

# #### Stopping
# An `AbstractStoppingCriterion` is applicable at various points in the algorithm:
abstract type AbstractStoppingCriterion end
abstract type AbstractStopPoint end

# #### Other
# To enable thread-safe databases, we have an extension specializing an `AbstractReadWriteLock`:
abstract type AbstractReadWriteLock end

# ### Concrete Types

# Some configuration types have fields of a special nature:
# We want them to take on default values dependent on the float_type.
struct NumberWithDefault{F}
	val :: F
	is_default :: Bool
end
function Base.convert(::Type{NumberWithDefault{F}}, x::NumberWithDefault{F}) where F
	return x
end
function Base.convert(::Type{NumberWithDefault{F}}, x::NumberWithDefault{T}) where {F,T}
	return NumberWithDefault{F}(T(x.val), x.is_default)
end
#src function Base.convert(::Type{NumberWithDefault{F}}, num::T) where {F, T}
#src 	NumberWithDefault(convert(F, num), true)
#src end

# #### Includes

# The `AlgorithmOptions` have their own file, because the type is somewhat complicated:
include("algorithm_options.jl")

# #### More Definitions

# ## General Array Containers
struct StepValueArrays{T}
    "Normal step vector."
    n :: Vector{T}
    "Iterate + normal step."
    xn :: Vector{T}
    "(Scaled) descent step."
    d :: Vector{T}
    "Complete step (normal + descent)."
	s :: Vector{T}
    "Iterate + step."
    xs :: Vector{T}
    "Values of surrogate objectives at `xs`."
	fxs :: Vector{T}
    "Reference to approximate criticality value."
	crit_ref :: Base.RefValue{T}
end

function StepValueArrays(n_vars, n_objfs, T)
    return StepValueArrays{T}(
        zeros(T, n_vars),
        zeros(T, n_vars),
        zeros(T, n_vars),
        zeros(T, n_vars),
        zeros(T, n_vars),
		zeros(T, n_objfs),
		Ref(T(Inf)),
	)
end

struct LinearConstraints{
	LB<:Union{Nothing, <:AbstractVector}, 
	UB<:Union{Nothing, <:AbstractVector},
	AType<:Union{Nothing, <:AbstractMatrix}, 
	BType<:Union{Nothing, <:AbstractVector}, 
	EType<:Union{Nothing, <:AbstractMatrix}, 
	CType<:Union{Nothing, <:AbstractVector}, 
}
    lb :: LB
    ub :: UB
    A :: AType
	b :: BType
	E :: EType
	c :: CType
end
float_type(::LinearConstraints{F}) where F=F

function init_lin_cons(mop :: AbstractMOP)
	lb = lower_var_bounds(mop)
	ub = upper_var_bounds(mop)
	if !var_bounds_valid(lb, ub)
        error("Variable bounds inconsistent.")
    end

	A = lin_ineq_constraints_matrix(mop)
	E = lin_eq_constraints_matrix(mop)
	b = lin_ineq_constraints_vector(mop)
	c = lin_eq_constraints_vector(mop)
	return init_lin_cons(float_type(mop), lb, ub, A, b, E, c)
end

function init_lin_cons(::Type{F}, lb, ub, A, b, E, c) where F
	return LinearConstraints(
		ensure_float_type(lb, F),
		ensure_float_type(ub, F),
		ensure_float_type(A, F),
		ensure_float_type(b, F),
		ensure_float_type(E, F),
		ensure_float_type(c, F),
	)
end

@enum RADIUS_UPDATE :: Int8 begin
	SHRINK_FAIL = -2
	GROW_FAIL = -1
	INITIAL_RADIUS = 0
	SHRINK = 1
	GROW = 2
end

@enum ITERATION_TYPE :: UInt8 begin
	INITIALIZATION
	RESTORATION
	FILTER_FAIL
	F_STEP
	THETA_STEP
end

@enum STEP_CLASS :: Int8 begin
	INACCEPTABLE = 0
	ACCEPTABLE = 1 
	SUCCESSFUL = 2
	INITIAL_STEP = 3
end

Base.@kwdef mutable struct IterationStatus
	iteration_type :: ITERATION_TYPE
	radius_change :: RADIUS_UPDATE
	step_class :: STEP_CLASS
end

Base.@kwdef mutable struct IterationScalars{F}
	it_index :: Int
	delta :: F
end

Base.@kwdef mutable struct TrialCaches{F}
	delta :: F
	diff_x :: Vector{F}
	diff_fx :: Vector{F}
	diff_fx_mod :: Vector{F}
end

Base.@kwdef mutable struct CriticalityRoutineCache{F, SV}
	delta :: F
	num_crit_loops :: Int
	step_vals :: SV
end

Base.@kwdef struct OptimizerCaches{
	mopType <: AbstractMOP,
	modType <: AbstractMOPSurrogate,
	scalerType <: AbstractAffineScaler,
	lin_consType <: LinearConstraints,
	scaled_consType <: LinearConstraints,
	valsType, # <: WrappedMOPCache,
	mod_valsType <: AbstractMOPSurrogateCache,
	filterType, # <: StandardFilter,
	step_valsType <: StepValueArrays,
	step_cacheType,
	crit_cacheType <: CriticalityRoutineCache,
	trial_cachesType <: TrialCaches,
	iteration_statusType <: IterationStatus,
	iteration_scalarsType <: IterationScalars,
	stop_critsType <: AbstractStoppingCriterion
}
	mop :: mopType
	mod :: modType

	scaler :: scalerType

	lin_cons :: lin_consType
	scaled_cons :: scaled_consType

	vals :: valsType
	vals_tmp :: valsType

	mod_vals :: mod_valsType

	filter :: filterType

	step_vals :: step_valsType
	step_cache :: step_cacheType
	crit_cache :: crit_cacheType
	trial_caches :: trial_cachesType

	iteration_status :: iteration_statusType
	iteration_scalars :: iteration_scalarsType

	stop_crits :: stop_critsType
end

include("return_object.jl")