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
@with_kw struct AlgorithmOptions{QPOPT}
    max_iter :: Int = 100

    "Stop if the trust region radius is reduced to below `stop_delta_min`."
	stop_delta_min = eps(Float64)

	"Stop if the trial point ``xₜ`` is accepted and ``‖xₜ - x‖≤ δ‖x‖``."
	stop_xtol_rel = 1e-5
	"Stop if the trial point ``xₜ`` is accepted and ``‖xₜ - x‖≤ ε``."
	stop_xtol_abs = -Inf
	"Stop if the trial point ``xₜ`` is accepted and ``‖f(xₜ) - f(x)‖≤ δ‖f(x)‖``."
	stop_ftol_rel = 1e-5
	"Stop if the trial point ``xₜ`` is accepted and ``‖f(xₜ) - f(x)‖≤ ε``."
	stop_ftol_abs = -Inf

	"Stop if for the approximate criticality it holds that ``χ̂(x) <= ε`` and for the feasibility that ``θ <= δ``."
	stop_crit_tol_abs = eps(Float64)
	"Stop if for the approximate criticality it holds that ``χ̂(x) <= ε`` and for the feasibility that ``θ <= δ``."
	stop_theta_tol_abs = eps(Float64)
	
	"Stop after the criticality routine has looped `stop_max_crit_loops` times."
	stop_max_crit_loops = 1

	# criticality test thresholds
	eps_crit = 0.1
	eps_theta = 0.1
	crit_B = 1000
	crit_M = 3000
	crit_alpha = 0.5
	
	# initialization
	delta_init = 0.5
	delta_max = 2^5 * delta_init

	# trust region updates
	gamma_shrink_much = 0.1 	# 0.1 is suggested by Fletcher et. al. 
	gamma_shrink = 0.5 			# 0.5 is suggested by Fletcher et. al. 
	gamma_grow = 2.0 			# 2.0 is suggested by Fletcher et. al. 

	# acceptance test 
	strict_acceptance_test = true
	nu_accept = 0.01 			# 1e-2 is suggested by Fletcher et. al. 
	nu_success = 0.9 			# 0.9 is suggested by Fletcher et. al. 
	
	# compatibilty parameters
	c_delta = 0.7 				# 0.7 is suggested by Fletcher et. al. 
	c_mu = 100.0 				# 100 is suggested by Fletcher et. al.
	mu = 0.01 					# 0.01 is suggested by Fletcher et. al.

	# model decrease / constraint violation test
	kappa_theta = 1e-4 			# 1e-4 is suggested by Fletcher et. al. 
	psi_theta = 2.0

	nlopt_restoration_algo = :LN_COBYLA

    scaler_cfg :: Symbol = :box

    qp_opt :: QPOPT = DEFAULT_QP_OPTIMIZER

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

struct StepValueArrays{T}
    n :: Vector{T}
    xn :: Vector{T}
    d :: Vector{T}
    xs :: Vector{T}
    fxs :: Vector{T}
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
        zeros(T, nout)
    )
end

struct LinearConstraints{LB, UB, ABEQ, ABINEQ}
    lb :: LB
    ub :: UB
    Ab :: ABEQ
    Ec :: ABINEQ
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