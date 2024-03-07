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
# Current scalers are defined in "src/affine_scalers.jl"
abstract type AbstractAffineScaler end
abstract type AbstractConstantAffineScaler <: AbstractAffineScaler end
abstract type AbstractDynamicAffineScaler <: AbstractAffineScaler end

# ## Step Computation
# Supertypes to configure and cache descent step computation.
# A config is meant to be provided by the user, and the cache is passed internally.
# At the moment, there is only steepest descent with standard Armijo line-search, which
# is implmented in "descent.jl"
abstract type AbstractStepConfig end
abstract type AbstractStepCache end

# ## AlgorithmOptions
"""
	AlgorithmOptions(; kwargs...)

Configure the optimization by passing keyword arguments:
$(TYPEDFIELDS)
"""
Base.@kwdef struct AlgorithmOptions{_T <: Number, SC}
	T :: Type{_T} = DEFAULT_FLOAT_TYPE

	"Configuration object for descent and normal step computation."
    step_config :: SC = SteepestDescentConfig()

	require_fully_linear_models :: Bool = true

	"Control verbosity by setting a min. level for `@logmsg`."
	log_level :: LogLevel = LogLevel(0)

	"Maximum number of iterations."
    max_iter :: Int = 500

    "Stop if the trust region radius is reduced to below `stop_delta_min`."
	stop_delta_min :: _T = eps(T)

	"Stop if the trial point ``xₜ`` is accepted and ``‖xₜ - x‖≤ δ‖x‖``."
	stop_xtol_rel :: _T = -Inf
	"Stop if the trial point ``xₜ`` is accepted and ``‖xₜ - x‖≤ ε``."
	stop_xtol_abs :: _T = -Inf
	"Stop if the trial point ``xₜ`` is accepted and ``‖f(xₜ) - f(x)‖≤ δ‖f(x)‖``."
	stop_ftol_rel :: _T = -Inf
	"Stop if the trial point ``xₜ`` is accepted and ``‖f(xₜ) - f(x)‖≤ ε``."
	stop_ftol_abs :: _T = -Inf

	"Stop if for the approximate criticality it holds that ``χ̂(x) <= ε`` and for the feasibility that ``θ <= δ``."
	stop_crit_tol_abs :: _T = -Inf
	"Stop if for the approximate criticality it holds that ``χ̂(x) <= ε`` and for the feasibility that ``θ <= δ``."
	stop_theta_tol_abs :: _T = eps(T)
	
	"Stop after the criticality routine has looped `stop_max_crit_loops` times."
	stop_max_crit_loops :: Int = 10

	# criticality test thresholds
	"Lower bound for criticality before entering Criticality Routine."
	eps_crit :: _T = 0.1
	"Lower bound for feasibility before entering Criticality Routine."
	eps_theta :: _T = 0.1
	"At the end of the Criticality Routine the radius is possibly set to `crit_B * χ`."
	crit_B :: _T = 100
	"Criticality Routine runs until `Δ ≤ crit_M * χ`."
	crit_M :: _T = 3*crit_B
	"Trust region shrinking factor in criticality loops."
	crit_alpha :: _T = 0.5
	
	# initialization
	"Initial trust region radius."
	delta_init :: _T = 0.5
	"Maximum trust region radius."
	delta_max :: _T = 2^5 * delta_init

	# trust region updates
	"Most severe trust region reduction factor."
	gamma_shrink_much :: _T = 0.1 	    # 0.1 is suggested by Fletcher et. al. 
	"Trust region reduction factor."
	gamma_shrink :: _T = 0.5 			# 0.5 is suggested by Fletcher et. al. 
	"Trust region enlargement factor."
	gamma_grow :: _T = 2.0 			# 2.0 is suggested by Fletcher et. al. 

	# acceptance test 
	"Whether to require *all* objectives to be reduced or not."
	strict_acceptance_test :: Bool = true
	"Acceptance threshold."
	nu_accept :: _T = 1e-4 			# 1e-2 is suggested by Fletcher et. al. 
	"Success threshold."
	nu_success :: _T = 0.4 			# 0.9 is suggested by Fletcher et. al. 
	
	# compatibilty parameters
	"Factor for normal step compatibility test. The smaller `c_delta`, the stricter the test."
	c_delta :: _T = 0.9 				# 0.7 is suggested by Fletcher et. al. 
	"Factor for normal step compatibility test. The smaller `c_mu`, the stricter the test for small radii."
	c_mu :: _T = 100.0 				# 100 is suggested by Fletcher et. al.
	"Exponent for normal step compatibility test. The larger `mu`, the stricter the test for small radii."
	mu :: _T = 0.01 					# 0.01 is suggested by Fletcher et. al.

	# model decrease / constraint violation test
	"Factor in the model decrease condition."
	kappa_theta :: _T = 1e-4 			# 1e-4 is suggested by Fletcher et. al. 
	"Exponent (for constraint violation) in the model decrease condition."
	psi_theta :: _T = 2.0

	"Configuration to determine variable scaling (if model supports it). Either `:box` or `:none`."
    scaler_cfg :: Symbol = :box

	"NLopt algorithm symbol for restoration phase."
    nl_opt :: Symbol = :GN_DIRECT_L_RAND    
end

## to be sure that equality is based on field values:
@batteries AlgorithmOptions selfconstructor=false

function AlgorithmOptions{T, SC}(
	typekw :: Type,
    step_config :: SC,
	require_fully_linear_models::Bool,
	log_level::LogLevel,
	max_iter::Integer,
	stop_delta_min::Real,
	stop_xtol_abs::Real,
	stop_ftol_rel::Real,
	stop_ftol_abs::Real,
	stop_crit_tol_abs :: Real,
	stop_theta_tol_abs :: Real,
	stop_max_crit_loops :: Integer,
	eps_crit :: Real, 
	eps_theta :: Real,
	crit_B :: Real,
	crit_M :: Real,
	crit_alpha :: Real,
	delta_init :: Real,
	delta_max :: Real,
	gamma_shrink_much :: Real,
	gamma_shrink :: Real,
	gamma_grow :: Real,
	strict_acceptance_test :: Bool,
	nu_accept::Real,
	nu_success :: Real,
	c_delta :: Real,
	c_mu :: Real,
	mu :: Real,
	kappa_theta :: Real,
	psi_theta :: Real, 
	scaler_cfg :: Symbol,
	nl_opt :: Symbol,
) where {T<:Real, SC}
	@assert scaler_cfg == :box || scaler_cfg == :none
	@assert string(nl_opt)[2] == 'N' "Restoration algorithm must be derivative free."
	return AlgorithmOptions{T, SC}(
		T,
		step_config,
		require_fully_linear_models,
		log_level,
		max_iter,
		stop_delta_min,
		stop_xtol_abs,
		stop_ftol_rel,
		stop_ftol_abs,
		stop_crit_tol_abs,
		stop_theta_tol_abs,
		stop_max_crit_loops,
		eps_crit, 
		eps_theta,
		crit_B,
		crit_M,
		crit_alpha,
		delta_init,
		delta_max,
		gamma_shrink_much,
		gamma_shrink,
		gamma_grow,
		strict_acceptance_test,
		nu_accept,
		nu_success,
		c_delta,
		c_mu,
		mu,
		kappa_theta,
		psi_theta, 
		scaler_cfg,
		nl_opt,
	)
end
function AlgorithmOptions(T::Type{_T}, step_config::SC, args...) where {_T <: Real, SC}
	return AlgorithmOptions{T, SC}(T, step_config, args...)
end
function AlgorithmOptions{T}(; kwargs...) where T<:Real
	AlgorithmOptions(; T, kwargs...)
end

# ## General Array Containers
"A struct holding values computed for or derived from an `AbstractMOP`."
Base.@kwdef struct ValueArrays{
	F<:AbstractFloat,
	FX<:AbstractVector{F},	# usually everything is a `Vector{F}`,
							# but we allow the type to be set with 
							# functions like `prealloc_objectives_vector`
	HX<:AbstractVector{F},	
	GX<:AbstractVector{F},
}
	"Unscaled variable vector used for evaluation."
    ξ :: Vector{F}
	"Internal (scaled) variable vector."
    x :: Vector{F}
	"Objective value vector."
    fx :: FX
	"Nonlinear equality constraints value vector."
    hx :: HX
	"Nonlinear inequality constraints value vector."
    gx :: GX
    
	Ex :: Vector{F}
    Ax :: Vector{F}

	Ax_min_b :: Vector{F}
	Ex_min_c :: Vector{F}
	
	"Reference to maximum constraint violation."
    theta_ref :: Base.RefValue{F}
	"Reference to maximum function value."
    phi_ref :: Base.RefValue{F}

	# meta data fields
	n_vars :: Int = length(x)
	dim_objectives :: Int = length(fx)
	dim_lin_eq_constraints :: Int = length(Ex)
	dim_lin_ineq_constraints :: Int = length(Ax)
	dim_nl_eq_constraints :: Int = length(hx)
	dim_nl_ineq_constraints :: Int = length(gx)
end

function universal_copy!(
	vals_trgt::ValueArrays, 
	vals_src::ValueArrays
)
	for fn in fieldnames(ValueArrays)
		universal_copy!(
			getfield(vals_trgt, fn),
			getfield(vals_src, fn)
		)
	end
	return nothing
end

struct SurrogateValueArrays{
	F <: AbstractFloat,
	FX <: AbstractVector{F},
	HX <: AbstractVector{F},
	GX <: AbstractVector{F},
	DFX <: AbstractMatrix{F},
	DHX <: AbstractMatrix{F},
	DGX <: AbstractMatrix{F}
}
    fx :: FX
    hx :: HX
    gx :: GX
    Dfx :: DFX
    Dhx :: DHX
    Dgx :: DGX
end

function universal_copy!(
	mod_vals_trgt::SurrogateValueArrays, 
	mod_vals_src::SurrogateValueArrays
)
	for fn in fieldnames(SurrogateValueArrays)
		trgt_fn = getfield(mod_vals_trgt, fn)
		universal_copy!(trgt_fn, getfield(mod_vals_src, fn))
	end
	return nothing
end

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
	LB<:Union{Nothing, AbstractVector}, 
	UB<:Union{Nothing, AbstractVector},
	AType<:Union{Nothing, AbstractMatrix}, 
	BType<:Union{Nothing, AbstractVector}, 
	EType<:Union{Nothing, AbstractMatrix}, 
	CType<:Union{Nothing, AbstractVector}, 
}
	n_vars :: Int
    lb :: LB
    ub :: UB
    A :: AType
	b :: BType
	E :: EType
	c :: CType
end

@enum IT_STAT begin
	RESTORATION = -3
	FILTER_FAIL = -2
	INACCEPTABLE = -1
	INITIALIZATION = 0
	FILTER_ADD_SHRINK = 1
	FILTER_ADD = 2
	ACCEPTABLE = 3
	SUCCESSFUL = 4
	CRITICAL_LOOP = 5
end

Base.@kwdef mutable struct UpdateResults{F<:AbstractFloat}
	it_index :: Int
	Δ_pre :: F
	Δ_post :: F

	it_stat :: IT_STAT

	point_has_changed :: Bool

	diff_x :: Vector{F}
	diff_fx :: Vector{F}
	diff_fx_mod :: Vector{F}

	norm2_x :: F
	norm2_fx :: F
	norm2_fx_mod :: F
end

function init_update_results(T, n_vars, n_objfs, delta_init)
	NaNT = T(NaN)
	return UpdateResults(
		it_index = 0,
		Δ_pre = NaNT,
		Δ_post = T(delta_init),
		it_stat = INITIALIZATION,
		point_has_changed = true,
		diff_x = fill(NaNT, n_vars),
		diff_fx = fill(NaNT, n_objfs),
		diff_fx_mod = fill(NaNT, n_objfs),
		norm2_x = NaNT,
		norm2_fx = NaNT,
		norm2_fx_mod = NaNT
	)
end

mutable struct CriticalityRoutineCache{MV, SV, SC, M}
	mod_vals :: MV
	step_vals :: SV
	step_cache :: SC
	mod :: M
end

#=
The iteration status already encodes much information:

| it_stat        | value | trial_point_accepted | radius_changed |
|----------------|-------|----------------------|----------------|
| RESTORATION       | -3    | yes [^1]             | no [^2]        |
| FILTER_FAIL       | -2    | no                   | yes            |
| INACCEPTABLE      | -1    | no                   | yes            |
| INITIALIZATION    | 0     | yes or no [^3]       | yes [^4]       |
| FILTER_ADD_SHRINK | 1     | yes                  | yes            |
| FILTER_ADD        | 2     | yes                  | no             |
| ACCEPTABLE        | 3     | yes                  | yes            |
| SUCCESSFUL        | 4     | yes                  | yes or no [^6] |
| CRITICAL_LOOP		| 5		| yes or no			   | yes

[^1]: We interpret the result of the restoration procedure as a trial point.
[^2]: Radius change in restoration iterations is up for debate.
[^3]: At initialization, the trial point is copied from the initial values.
[^4]: We go from “no radius” to “initial radius” and trigger a model update.
[^5]: In a `FILTER_ADD` iteration, the radius may be decreased if sufficient decrease fails.
[^6]: We can only update up until the maximum allowed radius.

Because of all these nuances, we rather use specific flags for acceptance and radius updates
in the main loop.
=#

struct ReturnObject{V, S}
	vals :: V
	stop_code :: S
end
 
opt_vars(r::ReturnObject)=r.vals.ξ
opt_objectives(r::ReturnObject)=r.vals.fx
opt_nl_eq_constraints(r::ReturnObject)=r.vals.hx
opt_nl_ineq_constraints(r::ReturnObject)=r.vals.gx
opt_lin_eq_constraints(r::ReturnObject)=r.vals.Ex_min_c
opt_lin_ineq_constraints(r::ReturnObject)=r.vals.Ax_min_b
opt_constraint_violation(r::ReturnObject)=r.theta_ref[]
function opt_stop_code(r::ReturnObject)
	c = r.stop_code
	while c isa WrappedStoppingCriterion
		c = c.crit
	end
	return c
end

