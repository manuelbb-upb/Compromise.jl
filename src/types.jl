# # Constants and Types
# In this file ("src/types.jl"), we define **all** abstract types and some concrete types.
# Every concrete type that is not defined here should only depend on types declared within
# this file.
const DEFAULT_FLOAT_TYPE = Float64

# ## Shorthands for Real Arrays
# Our algorithm operates on real-valued vectors and matrices, these are shorthands:
const RVec = AbstractVector{<:Real}
const RMat = AbstractMatrix{<:Real}
const RVecOrMat = Union{RVec, RMat}
const RVecOrNothing = Union{RVec, Nothing}
const RMatOrNothing = Union{RMat, Nothing}

# ## Multi-Objective Problems
# The algorithm operates on `AbstractMOP`s, where MOP stands for multi-objective optimization
# problem. It's interface is defined in "src/mop.jl".
# An MOP is surrogated by an `AbstractMOPSurrogate`. The interface is defined in "src/model.jl".
abstract type AbstractMOP end
abstract type AbstractMOPSurrogate end

# ## Scaling
# Supertypes for objects that scale variables.
# At the moment, we only use constant scaling, and I am unsure If we should distinguish between
# constant and variable scaling.
# Current scalers are defined in "src/diagonal_scalers.jl"
abstract type AbstractAffineScaler end

# ## Step Computation
# Supertypes to configure and cache descent step computation.
# A config is meant to be provided by the user, and the cache is passed internally.
# At the moment, there is only steepest descent with standard Armijo line-search, which
# is implmented in "descent.jl"
abstract type AbstractStepConfig end
abstract type AbstractStepCache end

include("algorithm_options.jl")
# ## General Array Containers

struct StepValueArrays{T}
    n :: Vector{T}
    xn :: Vector{T}
    d :: Vector{T}
	s :: Vector{T}
    xs :: Vector{T}
	fxs :: Vector{T}
	crit_ref :: Base.RefValue{T}
end

function universal_copy!(
	step_vals_trgt::StepValueArrays, step_vals_src::StepValueArrays
)
	for fn in fieldnames(StepValueArrays)
		trgt_fn = getfield(step_vals_trgt, fn)
		universal_copy!(trgt_fn, getfield(step_vals_src, fn))
	end
	return nothing
end

function StepValueArrays(n_vars, n_objfs, T)
    return StepValueArrays(
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
	F<:Number,
	LB<:Union{Nothing, <:AbstractVector{F}}, 
	UB<:Union{Nothing, <:AbstractVector{F}},
	AType<:Union{Nothing, <:AbstractMatrix{F}}, 
	BType<:Union{Nothing, <:AbstractVector{F}}, 
	EType<:Union{Nothing, <:AbstractMatrix{F}}, 
	CType<:Union{Nothing, <:AbstractVector{F}}, 
}
	_F :: Type{F}
	n_vars :: Int
    lb :: LB
    ub :: UB
    A :: AType
	b :: BType
	E :: EType
	c :: CType
end
float_type(::LinearConstraints{F}) where F=F

function universal_copy!(
	scaled_cons::LinearConstraints, lin_cons::LinearConstraints)
    for fn in fieldnames(LinearConstraints)
        universal_copy!(
            getfield(scaled_cons, fn), 
            getfield(lin_cons, fn)
        )
    end
    return nothing
end

function init_lin_cons(mop)
	F = float_type(mop)
    
	lb = ensure_float_type(lower_var_bounds(mop), F)
    ub = ensure_float_type(upper_var_bounds(mop), F)

    if !var_bounds_valid(lb, ub)
        error("Variable bounds inconsistent.")
    end

    A = ensure_float_type(lin_ineq_constraints_matrix(mop), F)
    b = ensure_float_type(lin_ineq_constraints_vector(mop), F)
    E = ensure_float_type(lin_eq_constraints_matrix(mop), F)
    c = ensure_float_type(lin_eq_constraints_vector(mop), F)

    return LinearConstraints(F, dim_vars(mop), lb, ub, A, b, E, c)
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
	INITIAL_STEP = -1
	INACCEPTABLE = 0
	ACCEPTABLE = 1 
	SUCCESSFUL = 2
end

Base.@kwdef mutable struct IterationStatus
	iteration_type :: ITERATION_TYPE
	radius_change :: RADIUS_UPDATE
	step_class :: STEP_CLASS
end

_trial_point_accepted(iteration_status)=_trial_point_accepted(iteration_status.step_class)
function _trial_point_accepted(step_class::STEP_CLASS)
	return Int8(step_class) > 0
end

Base.@kwdef mutable struct IterationScalars{F}
	it_index :: Int
	delta_pre :: F
	delta :: F
end

Base.@kwdef mutable struct TrialCaches{F}
	delta :: F
	diff_x :: Vector{F}
	diff_fx :: Vector{F}
	diff_fx_mod :: Vector{F}
end

mutable struct CriticalityRoutineCache{F, MV, SV}
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
	valsType <: WrappedMOPCache,
	mod_valsType <: AbstractMOPSurrogateCache,
	filterType <: StandardFilter,
	step_cacheType <: StepValueArrays,
	crit_cacheType <: CriticalityRoutineCache,
	trial_cachesType <: TrialCaches,
	iteration_statusType <: IterationStatus,
	iteration_scalarsType <: IterationScalars,
	stop_critsType
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

	step_cache :: step_cacheType
	crit_cache :: crit_cacheType
	trial_caches :: trial_cachesType

	iteration_status :: iteration_statusType
	iteration_scalars :: iteration_scalarsType

	stop_crits :: stop_critsType
end

struct ReturnObject{X, V, S, M}
	ξ0 :: X
	vals :: V
	stop_code :: S
	mod :: M
end

opt_surrogate(r::ReturnObject) = r.mod
opt_initial_vars(r::ReturnObject) = r.ξ0

opt_vars(r::ReturnObject)=opt_vars(r.vals)
opt_vars(::Nothing) = missing
opt_vars(v::AbstractMOPCache)=cached_ξ(v)

opt_objectives(r::ReturnObject)=opt_objectives(r.vals)
opt_objectives(::Nothing)=missing
opt_objectives(v::AbstractMOPCache)=cached_fx(v)

opt_nl_eq_constraints(r::ReturnObject)=opt_nl_eq_constraints(r.vals)
opt_nl_eq_constraints(::Nothing)=missing
opt_nl_eq_constraints(v::AbstractMOPCache)=cached_hx(v)

opt_nl_ineq_constraints(r::ReturnObject)=opt_nl_ineq_constraints(r.vals)
opt_nl_ineq_constraints(::Nothing)=missing
opt_nl_ineq_constraints(v::AbstractMOPCache)=cached_gx(v)

opt_lin_eq_constraints(r::ReturnObject)=opt_lin_eq_constraints(r.vals)
opt_lin_eq_constraints(::Nothing)=missing
opt_lin_eq_constraints(v::AbstractMOPCache)=cached_Ex_min_c(v)

opt_lin_ineq_constraints(r::ReturnObject)=opt_lin_ineq_constraints(r.vals)
opt_lin_ineq_constraints(::Nothing)=missing
opt_lin_ineq_constraints(v::AbstractMOPCache)=cached_Ax_min_b(v)

opt_constraint_violation(r::ReturnObject)=opt_constraint_violation(r.vals)
opt_constraint_violation(::Nothing)=missing
opt_constraint_violation(v::AbstractMOPCache)=cached_theta(v)

function opt_stop_code(r::ReturnObject)
	c = r.stop_code
	while c isa WrappedStoppingCriterion
		c = c.crit
	end
	return c
end

function Base.show(io::IO, ret::ReturnObject)
	print(io, """
	ReturnObject
	x0   = $(pretty_row_vec(opt_initial_vars(ret)))
	x*   = $(pretty_row_vec(opt_vars(ret)))
	f(x*)= $(pretty_row_vec(opt_objectives(ret)))
	code = $(opt_stop_code(ret))"""
	)
end