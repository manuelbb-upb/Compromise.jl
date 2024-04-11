# ## AlgorithmOptions
"""
	AlgorithmOptions(; kwargs...)

Configure the optimization by passing keyword arguments:
$(TYPEDFIELDS)
"""
Base.@kwdef struct AlgorithmOptions{_T <: Number, SC, SCALER_CFG_TYPE}
	T :: Type{_T} = DEFAULT_FLOAT_TYPE

	"Configuration object for descent and normal step computation."
    step_config :: SC = SteepestDescentConfig()

	"Configuration to determine variable scaling (if model supports it). Either `:box` or `:none`."
    scaler_cfg :: SCALER_CFG_TYPE = Val(:box)

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
	eps_theta :: _T = 0.05
	"At the end of the Criticality Routine the radius is possibly set to `crit_B * χ`."
	crit_B :: _T = 100
	"Criticality Routine runs until `Δ ≤ crit_M * χ`."
	crit_M :: _T = 3*crit_B
	"Trust region shrinking factor in criticality loops."
	crit_alpha :: _T = 0.1

	backtrack_in_crit_routine :: Bool = true
	
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

	"NLopt algorithm symbol for restoration phase."
    nl_opt :: Symbol = :GN_DIRECT_L_RAND    
end

## to be sure that equality is based on field values:
@batteries AlgorithmOptions selfconstructor=false

function AlgorithmOptions{T, SC, SCALER_CFG_TYPE}(
	typekw :: Type,
    step_config :: SC,
	scaler_cfg :: SCALER_CFG_TYPE,
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
	backtrack_in_crit_routine :: Bool,
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
	nl_opt :: Symbol,
) where {T<:Real, SC, SCALER_CFG_TYPE}
	@assert scaler_cfg isa AbstractAffineScaler || scaler_cfg isa Val || scaler_cfg == :box || scaler_cfg == :none
	@assert string(nl_opt)[2] == 'N' "Restoration algorithm must be derivative free."
	return AlgorithmOptions{T, SC, SCALER_CFG_TYPE}(
		T,
		step_config,
		scaler_cfg,
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
		backtrack_in_crit_routine,
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
		nl_opt,
	)
end
function AlgorithmOptions(T::Type{_T}, step_config::SC, scaler_cfg::ST, args...) where {_T <: Real, SC, ST}
	return AlgorithmOptions{T, SC, ST}(T, step_config, scaler_cfg, args...)
end
function AlgorithmOptions{T}(; kwargs...) where T<:Real
	AlgorithmOptions(; T, kwargs...)
end

Base.@kwdef struct ThreadedOuterAlgorithmOptions{A}
	inner_opts :: A = AlgorithmOptions()
end