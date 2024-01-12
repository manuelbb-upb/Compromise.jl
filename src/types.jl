# # Constants and Types
# In this file ("src/types.jl"), we define **all** abstract types and some concrete types.
# Every concrete type that is not defined here should only depend on types declared within
# this file.

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

# Supertypes to indicate if a model supports scaling:
abstract type AbstractScalingIndicator end
abstract type AbstractAffineScalingIndicator <: AbstractScalingIndicator end

# Available return values for `supports_scaling(model)`:
struct NoScaling <: AbstractScalingIndicator end
struct ConstantAffineScaling <: AbstractAffineScalingIndicator end
struct DynamicAffineScaling <: AbstractAffineScalingIndicator end

# ## Step Computation
# Supertypes to configure and cache descent step computation.
# A config is meant to be provided by the user, and the cache is passed internally.
# At the moment, there is only steepest descent with standard Armijo line-search, which
# is implmented in "descent.jl"
abstract type AbstractStepConfig end
abstract type AbstractStepCache end

# ## AlgorithmOptions
@with_kw struct AlgorithmOptions{SC}
	log_level :: LogLevel = LogLevel(0)

    max_iter :: Int = 500

    "Stop if the trust region radius is reduced to below `stop_delta_min`."
	stop_delta_min :: Float64 = eps(Float64)

	"Stop if the trial point ``xₜ`` is accepted and ``‖xₜ - x‖≤ δ‖x‖``."
	stop_xtol_rel :: Float64 = -Inf
	"Stop if the trial point ``xₜ`` is accepted and ``‖xₜ - x‖≤ ε``."
	stop_xtol_abs :: Float64 = -Inf
	"Stop if the trial point ``xₜ`` is accepted and ``‖f(xₜ) - f(x)‖≤ δ‖f(x)‖``."
	stop_ftol_rel :: Float64 = -Inf
	"Stop if the trial point ``xₜ`` is accepted and ``‖f(xₜ) - f(x)‖≤ ε``."
	stop_ftol_abs :: Float64 = -Inf

	"Stop if for the approximate criticality it holds that ``χ̂(x) <= ε`` and for the feasibility that ``θ <= δ``."
	stop_crit_tol_abs :: Float64 = eps(Float64)
	"Stop if for the approximate criticality it holds that ``χ̂(x) <= ε`` and for the feasibility that ``θ <= δ``."
	stop_theta_tol_abs :: Float64 = eps(Float64)
	
	"Stop after the criticality routine has looped `stop_max_crit_loops` times."
	stop_max_crit_loops :: Int = 1

	# criticality test thresholds
	eps_crit :: Float64 = 0.1
	eps_theta :: Float64 = 0.1
	crit_B :: Float64 = 1000
	crit_M :: Float64 = 3000
	crit_alpha :: Float64 = 0.5
	
	# initialization
	delta_init :: Float64 = 0.5
	delta_max :: Float64 = 2^5 * delta_init

	# trust region updates
	gamma_shrink_much :: Float64= 0.1 	# 0.1 is suggested by Fletcher et. al. 
	gamma_shrink :: Float64 = 0.5 			# 0.5 is suggested by Fletcher et. al. 
	gamma_grow :: Float64 = 2.0 			# 2.0 is suggested by Fletcher et. al. 

	# acceptance test 
	strict_acceptance_test :: Bool = true
	nu_accept :: Float64 = 0.01 			# 1e-2 is suggested by Fletcher et. al. 
	nu_success :: Float64 = 0.9 			# 0.9 is suggested by Fletcher et. al. 
	
	# compatibilty parameters
	c_delta :: Float64 = 0.7 				# 0.7 is suggested by Fletcher et. al. 
	c_mu :: Float64 = 100.0 				# 100 is suggested by Fletcher et. al.
	mu :: Float64 = 0.01 					# 0.01 is suggested by Fletcher et. al.

	# model decrease / constraint violation test
	kappa_theta :: Float64 = 1e-4 			# 1e-4 is suggested by Fletcher et. al. 
	psi_theta :: Float64 = 2.0

	nlopt_restoration_algo :: Symbol = :LN_COBYLA

    scaler_cfg :: Symbol = :box

    step_config :: SC = SteepestDescentConfig()

    nl_opt :: Symbol = :LN_COBYLA

    @assert scaler_cfg == :box || scaler_cfg == :none
    @assert string(nl_opt)[2] == 'N' "Restoration algorithm must be derivative free."
end

# ## General Array Containers 

"A struct holding values computed for or derived from an `AbstractMOP`."
struct ValueArrays{X, FX, HX, GX, EX, AX, THETA, PHI}
	"Internal (scaled) variable vector."
    ξ :: X
	"Unscaled variable vector used for evaluation."
    x :: X
	"Objective value vector."
    fx :: FX
	"Nonlinear equality constraints value vector."
    hx :: HX
	"Nonlinear inequality constraints value vector."
    gx :: GX
	"Linear equality constraints residual."
    Eres :: EX
    Ex :: EX
	"Linear inequality constraints residual."
    Ares :: AX
    Ax :: AX
	"Reference to maximum constraint violation."
    θ :: THETA
	"Reference to maximum function value."
    Φ :: PHI
end

function Base.eltype(
    ::ValueArrays{X, FX, HX, GX, EX, AX, THETA, PHI}
) where {X, FX, HX, GX, EX, AX, THETA, PHI}
	T = eltype(X)
	return reduce(promote_modulo_nothing, (FX, HX, GX, EX, AX, THETA, PHI); init=T)
end

struct SurrogateValueArrays{FX, HX, GX, DFX, DHX, DGX}
    fx :: FX
    hx :: HX
    gx :: GX
    Dfx :: DFX
    Dhx :: DHX
    Dgx :: DGX
end
function Base.copyto!(step_vals_trgt::SurrogateValueArrays, step_vals_src::SurrogateValueArrays)
	for fn in fieldnames(SurrogateValueArrays)
		trgt_fn = getfield(step_vals_trgt, fn)
		isnothing(trgt_fn) && continue
		copyto!(trgt_fn, getfield(step_vals_src, fn))
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

function Base.copyto!(step_vals_trgt::StepValueArrays, step_vals_src::StepValueArrays)
	for fn in fieldnames(StepValueArrays)
		fn == :crit_ref && continue
		trgt_fn = getfield(step_vals_trgt, fn)
		isnothing(trgt_fn) && continue
		copyto!(trgt_fn, getfield(step_vals_src, fn))
	end
	step_vals_trgt.crit_ref[] = step_vals_src.crit_ref[]
	return nothing
end

function StepValueArrays(x, fx)
    T = Base.promote_eltype(x, fx)
    nin = length(x)
    nout = length(fx)
    return StepValueArrays(
        zeros(T, nin),
        zeros(T, nin),
        zeros(T, nin),
        zeros(T, nin),
        zeros(T, nin),
        zeros(T, nout),
		Ref(T(Inf))
    )
end

struct LinearConstraints{LB, UB, ABEQ, ABINEQ}
    lb :: LB
    ub :: UB
    A_b :: ABEQ
    E_c :: ABINEQ
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

Base.@kwdef mutable struct IterationMeta{F}
	it_index :: Int
	Δ_pre :: F
	Δ_post :: F

	crit_val :: F
	num_crit_loops :: Int

	it_stat_pre :: IT_STAT
	it_stat_post :: IT_STAT

	point_has_changed :: Bool

	vals_diff_vec :: Vector{F}
	mod_vals_diff_vec :: Vector{F}

	args_diff_len :: F
	vals_diff_len :: F
	mod_vals_diff_len :: F
end

function init_iter_meta(T, n_objfs, algo_opts)
	return IterationMeta(;
		it_index = 0,
		Δ_pre = T(NaN),
		Δ_post = T(algo_opts.delta_init),
		crit_val = T(Inf),
		num_crit_loops = 0,
		it_stat_pre = INITIALIZATION,
		it_stat_post = INITIALIZATION,
		point_has_changed = true,
		vals_diff_vec = fill(T(NaN), n_objfs),
		mod_vals_diff_vec = fill(T(NaN), n_objfs),
		args_diff_len = T(NaN),
		vals_diff_len = T(NaN),
		mod_vals_diff_len = T(NaN)
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

@enum RET_CODE begin
	INFEASIBLE = -1
    CONTINUE = 0
	CRITICAL = 1
	BUDGET = 2
	TOLERANCE_X = 3
	TOLERANCE_F = 4
end

const DEFAULT_PRECISION=Float32