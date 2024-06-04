"""
	AlgorithmOptions(; kwargs...)

Configure the optimization by passing keyword arguments:
$(TYPEDFIELDS)
"""
Base.@kwdef struct AlgorithmOptions{T <: AbstractFloat, SC, SCALER_CFG_TYPE}
	float_type :: Type{T} = DEFAULT_FLOAT_TYPE

	"Configuration object for descent and normal step computation."
    step_config :: SC = SteepestDescentConfig(; float_type)

	"Configuration to determine variable scaling (if model supports it). Either `:box` or `:none`."
    scaler_cfg :: SCALER_CFG_TYPE = Val(:box)

	require_fully_linear_models :: Bool = true

	"Control verbosity by setting a min. level for `@logmsg`."
	log_level :: LogLevel = LogLevel(0)

	"Maximum number of iterations."
    max_iter :: Int = 500

    "Stop if the trust region radius is reduced to below `stop_delta_min`."
	stop_delta_min :: NumberWithDefault{T} = NumberWithDefault(eps(float_type), true)

	"Stop if the trial point ``xₜ`` is accepted and ``‖xₜ - x‖≤ δ‖x‖``."
	stop_xtol_rel :: T = -Inf
	"Stop if the trial point ``xₜ`` is accepted and ``‖xₜ - x‖≤ ε``."
	stop_xtol_abs :: T = -Inf
	"Stop if the trial point ``xₜ`` is accepted and ``‖f(xₜ) - f(x)‖≤ δ‖f(x)‖``."
	stop_ftol_rel :: T = -Inf
	"Stop if the trial point ``xₜ`` is accepted and ``‖f(xₜ) - f(x)‖≤ ε``."
	stop_ftol_abs :: T = -Inf

	"Stop if for the approximate criticality it holds that ``χ̂(x) <= ε`` and for the feasibility that ``θ <= δ``."
	stop_crit_tol_abs :: T = -Inf
	"Stop if for the approximate criticality it holds that ``χ̂(x) <= ε`` and for the feasibility that ``θ <= δ``."
	stop_theta_tol_abs :: NumberWithDefault{T} = NumberWithDefault(eps(float_type), true)
	
	"Stop after the criticality routine has looped `stop_max_crit_loops` times."
	stop_max_crit_loops :: Int = 10

	# criticality test thresholds
	"Lower bound for criticality before entering Criticality Routine."
	eps_crit :: T = 0.01
	"Lower bound for feasibility before entering Criticality Routine."
	eps_theta :: T = 0.05
	"At the end of the Criticality Routine the radius is possibly set to `crit_B * χ`."
	crit_B :: T = 100
	"Criticality Routine runs until `Δ ≤ crit_M * χ`."
	crit_M :: T = 3*crit_B
	"Trust region shrinking factor in criticality loops."
	crit_alpha :: T = 0.1

	backtrack_in_crit_routine :: Bool = true
	
	# initialization
	"Initial trust region radius."
	delta_init :: T = 0.5
	"Maximum trust region radius."
	delta_max :: T = 2^5 * delta_init

	# trust region updates
	"Most severe trust region reduction factor."
	gamma_shrink_much :: T = 0.1 	    # 0.1 is suggested by Fletcher et. al. 
	"Trust region reduction factor."
	gamma_shrink :: T = 0.5 			# 0.5 is suggested by Fletcher et. al. 
	"Trust region enlargement factor."
	gamma_grow :: T = 2.0 			# 2.0 is suggested by Fletcher et. al. 

	# acceptance test 
	"Whether to require *all* objectives to be reduced or not."
	trial_mode ::Union{Val{:max_diff}, Val{:min_rho}, Val{:max_rho}} = Val(:max_diff)
	"Acceptance threshold."
	nu_accept :: T = 1e-4 			# 1e-2 is suggested by Fletcher et. al. 
	"Success threshold."
	nu_success :: T = 0.4 			# 0.9 is suggested by Fletcher et. al. 
	
	# compatibilty parameters
	"Factor for normal step compatibility test. The smaller `c_delta`, the stricter the test."
	c_delta :: T = 0.99 				# 0.7 is suggested by Fletcher et. al. 
	"Factor for normal step compatibility test. The smaller `c_mu`, the stricter the test for small radii."
	c_mu :: T = 100.0 				# 100 is suggested by Fletcher et. al.
	"Exponent for normal step compatibility test. The larger `mu`, the stricter the test for small radii."
	mu :: T = 0.01 					# 0.01 is suggested by Fletcher et. al.

	# model decrease / constraint violation test
	"Factor in the model decrease condition."
	kappa_theta :: T = 1e-4 			# 1e-4 is suggested by Fletcher et. al. 
	"Exponent (for constraint violation) in the model decrease condition."
	psi_theta :: T = 2.0
end

## to be sure that equality is based on field values:
@batteries AlgorithmOptions selfconstructor=false

## outer constructor to automatically convert kwarg values to correct type:
function AlgorithmOptions(
	:: Type{float_type},
    step_config :: SC,
	scaler_cfg :: SCALER_CFG_TYPE,
	require_fully_linear_models,
	log_level,
	max_iter,
	stop_delta_min,
	stop_xtol_rel,
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
	backtrack_in_crit_routine,
	delta_init,
	delta_max,
	gamma_shrink_much,
	gamma_shrink,
	gamma_grow,
	trial_mode,
	nu_accept,
	nu_success,
	c_delta,
	c_mu,
	mu,
	kappa_theta,
	psi_theta,
) where {float_type<:AbstractFloat, SC, SCALER_CFG_TYPE}
	@assert scaler_cfg isa AbstractAffineScaler || scaler_cfg isa Val || scaler_cfg == :box || scaler_cfg == :none
	if trial_mode isa Symbol
		trial_mode = Val(Symbol)
	end
	return AlgorithmOptions{float_type, SC, SCALER_CFG_TYPE}(
		float_type,
		step_config,
		scaler_cfg,
		require_fully_linear_models,
		log_level,
		max_iter,
		ensure_NumberWithDefault(stop_delta_min, false),	# if `stop_delta_min` is a `NumberWithDefault`, then `false` is ignored
																# if `stop_delta_min` is a Number, then we assume it to be non-default
		stop_xtol_rel,
		stop_xtol_abs,
		stop_ftol_rel,
		stop_ftol_abs,
		stop_crit_tol_abs,
		ensure_NumberWithDefault(stop_theta_tol_abs, false),
		stop_max_crit_loops,
		eps_crit, 
		eps_theta,
		crit_B,
		crit_M,
		crit_alpha,
		backtrack_in_crit_routine,
		delta_init,
		delta_max,
		gamma_shrink_much,
		gamma_shrink,
		gamma_grow,
		trial_mode,
		nu_accept,
		nu_success,
		c_delta,
		c_mu,
		mu,
		kappa_theta,
		psi_theta, 
	)
end
float_type(::AlgorithmOptions{F}) where F = F
ensure_NumberWithDefault(num, is_default)=NumberWithDefault(num, is_default)
ensure_NumberWithDefault(num::NumberWithDefault, is_default)=num

function change_float_type(
	x::NumberWithDefault{T}, 
	::Union{PropertyLens{:stop_delta_min}, PropertyLens{:stop_theta_tol_abs}}, 
	::Type{new_float_type}
) where {T, new_float_type}
	if x.is_default
		return NumberWithDefault(eps(new_float_type), true)
	end
	return convert(NumberWithDefault{new_float_type}, x)
end
function change_float_type(
	x::Number,
	::Union{PropertyLens{:stop_delta_min}, PropertyLens{:stop_theta_tol_abs}}, 
	::Type{new_float_type}
) where new_float_type
	return NumberWithDefault{new_float_type}(x, false)
end

function UnPack.unpack(opts::AlgorithmOptions, ::Val{field}) where field
	p = getproperty(opts, field)
	return unpack_property(p)
end
unpack_property(p)=p
unpack_property(p::NumberWithDefault)=p.val
Base.@kwdef struct ThreadedOuterAlgorithmOptions{A}
	inner_opts :: A = AlgorithmOptions()
end

Base.@kwdef struct SequentialOuterAlgorithmOptions{A}
	sort_by_delta :: Bool = true
	delta_factor :: Float64 = 0.0
	initial_nondominance_testing :: Bool = false
	nondominance_testing_offset :: Int = typemax(Int)
	log_level :: LogLevel = Info
	final_nondominance_testing :: Bool = true
	inner_opts :: A = AlgorithmOptions(; log_level)
end
@batteries SequentialOuterAlgorithmOptions selfconstructor=false