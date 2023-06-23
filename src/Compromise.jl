module Compromise

# import Parameters: @with_kw
# import LinearAlgebra as LA

include("CompromiseEvaluators.jl")
using .CompromiseEvaluators

#=
const Vec = AbstractVector{<:Real}
const Mat = AbstractMatrix{<:Real}

@enum IT_STAT begin
	RESTORATION = -3
	FILTER_FAIL = -2
	INACCEPTABLE = -1
	INITIALIZATION = 0
	FILTER_ADD = 1
	ACCEPTABLE = 2
	SUCCESSFUL = 3
end

# An `AbstractEvaluator` is a glorified wrapper around vector-vector functions.
# First, it maps a variable vector to a state vector.
abstract type AbstractEvaluator end

precision(::AbstractEvaluator)::Type{<:AbstractFloat}=Float64

dim_objectives(::AbstractEvaluator)::Int=0
dim_nl_eq_constraints(::AbstractEvaluator)::Int=0
dim_nl_ineq_constraints(::AbstractEvaluator)::Int=0

lower_var_bounds(::AbstractEvaluator)::Union{Nothing, Vec}=nothing
upper_var_bounds(::AbstractEvaluator)::Union{Nothing, Vec}=nothing

eval_objectives!(::Vec, ::AbstractEvaluator, ::Vec)=nothing
eval_nl_eq_constraints!(::Vec, ::AbstractEvaluator, ::Vec)=nothing
eval_nl_ineq_constraints!(::Vec, ::AbstractEvaluator, ::Vec)=nothing

jacT_objectives!(::Mat, ::AbstractEvaluator, ::Vec)=nothing
jacT_nl_eq_constraints!(::Mat, ::AbstractEvaluator, ::Vec)=nothing
jacT_nl_ineq_constraints!(::Mat, ::AbstractEvaluator, ::Vec)=nothing

A_eq(::AbstractEvaluator)::Union{Mat, Nothing}=nothing
A_ineq(::AbstractEvaluator)::Union{Mat, Nothing}=nothing
b_eq(::AbstractEvaluator)::Union{Vec, Nothing}=nothing
b_ineq(::AbstractEvaluator)::Union{Vec, Nothing}=nothing

function dim_lin_eq_constraints(ev::AbstractEvaluator)
    A = A_eq(ev)
    isnothing(A) && return 0
    return size(A, 1)
end

function dim_lin_ineq_constraints(ev::AbstractEvaluator)
    A = A_ineq(ev)
    isnothing(A) && return 0
    return size(A, 1)
end

function eval_lin_constraints!(y, x, A, b)
    A, b = data
    LA.mul!(y, A, x)
    if !isnothing(b)
        y .-= b
    end
    return nothing
end
function eval_lin_eq_constraints!(y::Vec, ev::AbstractEvaluator, x::Vec)
    A = A_eq(ev)
    isnothing(A) && return
    b = b_eq(ev)
    return eval_lin_constraints!(y, x, A, b)
end
function eval_lin_ineq_constraints!(y::Vec, ev::AbstractEvaluator, x::Vec)
    A = A_ineq(ev)
    isnothing(A) && return
    b = b_ineq(ev)
    return eval_lin_constraints!(y, x, A, b)
end

for func_type in (
    :objectives!,
    :nl_eq_constraints!, 
    :nl_ineq_constraints!,
    :lin_eq_constraints!,
    :lin_ineq_constraints!,
)
    eval_and_jacT_method_name = Symbol("eval_and_jacT_", func_type)
    eval_method_name = Symbol("eval_", func_type)
    jacT_method_name = Symbol("jacT_", func_type)
    dim_method_name = Symbol("dim_$(func_type)"[1:end-1])

    @eval function $(eval_and_jacT_method_name)(y::Vec, jacT::Mat, ev::AbstractEvaluator, x::Vec)
        $(eval_method_name)(y, ev, x)
        $(jacT_method_name)(jacT, ev, x)
        return nothing
    end

    eval_outputs_method_name = Symbol("eval_outputs_", func_type)
    
    @eval begin
        """
            $($(eval_outputs_method_name))(y, evaluator, x, out_indices)
            
        Evaluate the outputs with indices `out_indices` of `evaluator` at input x and store 
        the corresponding results in `y`.
        """
        function $(eval_outputs_method_name)(y::Vec, ev::AbstractEvaluator, x::Vec, out_indices)
            _y = Vector{precision(ev)}(undef, $(dim_method_name)(ev))
            $(eval_method_name)(_y, ev, x)
            y .= _y[out_indices]
            return nothing
        end
    end
end

dim_eq_constraints(ev) = dim_lin_eq_constraints(ev) + dim_nl_eq_constraints(ev)
dim_ineq_constraints(ev) = dim_lin_ineq_constraints(ev) + dim_nl_ineq_constraints(ev)

function prealloc_fx(ev::AbstractEvaluator)
    T = precision(ev)
    return Vector{T}(undef, dim_objectives(ev))
end

function prealloc_ex(ev::AbstractEvaluator)
    T = precision(ev)
    return Vector{T}(undef, dim_eq_constraints(ev))
end

function prealloc_ix(ev::AbstractEvaluator)
    T = precision(ev)
    return Vector{T}(undef, dim_ineq_constraints(ev))
end

function prealloc_Dfx(ev::AbstractEvaluator, nvars)
    T = precision(ev)
    return Matrix{T}(undef, nvars, dim_objectives(ev))
end

function prealloc_Dex(ev::AbstractEvaluator, nvars)
    T = precision(ev)
    return Matrix{T}(undef, nvars, dim_eq_constraints(ev))
end

function prealloc_Dix(ev::AbstractEvaluator, nvars)
    T = precision(ev)
    return Matrix{T}(undef, nvars, dim_ineq_constraints(ev))
end

Base.@kwdef mutable struct DevEvaluator! <: AbstractEvaluator
    objectives! :: Union{Function, Nothing} = nothing
    nl_eq_constraints! :: Union{Function, Nothing} = nothing
    nl_ineq_constraints! :: Union{Function, Nothing} = nothing
    
    jacT_objectives! :: Union{Function, Nothing} = nothing
    jacT_nl_eq_constraints! :: Union{Function, Nothing} = nothing
    jacT_nl_ineq_constraints! :: Union{Function, Nothing} = nothing
    
    A_eq :: Union{Nothing, Mat} = nothing 
    A_ineq :: Union{Nothing, Mat} = nothing
    b_eq :: Union{Nothing, Vec} = nothing
    b_ineq :: Union{Nothing, Vec} = nothing

    dim_objectives :: Int = 0
    dim_nl_eq_constraints :: Int = 0
    dim_nl_ineq_constraints :: Int = 0
    dim_lin_eq_constraints :: Int = isnothing(A_eq) ? 0 : size(A_eq, 1)
    dim_lin_ineq_constraints :: Int = isnothing(A_eq) ? 0 : size(A_eq, 1)

end

for prop_name in (
    :dim_objectives,
    :dim_nl_eq_constraints,
    :dim_nl_ineq_constraints,
    :dim_lin_eq_constraints,
    :dim_lin_ineq_constraints
)
    $(prop_name)(ev::DevEvaluator!) = getfield(ev, $(Meta.quot(prop_name)))
end

@with_kw struct AlgorithmConfig
	# stopping criteria
	max_iter = 100

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

	# backtracking:
	normalize_grads = false	
	armijo_strict = strict_acceptance_test
	armijo_min_stepsize = eps(Float64)
	armijo_backtrack_factor = .75
	armijo_rhs_factor = 1e-3
end	

# ξ = Tx + b
# x = T⁻¹(ξ - b)
struct AffineVarScaler{
    bType, TType1<:AbstractMatrix, TType2<:AbstractMatrix
}
    T :: TType1
    Tinv :: TType2 
    b :: bType
end

init_scaler(nvars, lb, ub) = AffineVarScalar(LA.I(nvars), LA.I(nvars), 0)

function init_scaler(nvars, lb::Vec, ub::Vec)
    if any(isinf.(lb)) || any(isinf.(ub))
        return init_scaler(nvars, nothing, nothing)
    end
    w = ub .- lb

    T = LA.Diagonal(1 ./ w)
    Tinv = LA.Diagonal(w)
    b = - lb ./ w

    return AffineVarScalar(T, Tinv, b)
end


function optimize(ev::AbstractEvaluator, x0; algo_config :: AlgorithmConfig)
    @assert !isempty(x0) "Starting point array `x0` is empty."
    @assert dim_objectives(ev) > 0 "Objective Vector dimension of problem is zero."
    T = precision(ev)

    nvars = length(x0)
    _lb = lower_var_bounds(ev)
    _ub = upper_var_bounds(ev)
    lb = isnothing(_lb) ? fill(T(-Inf), nvars) : T.(_lb)
    ub = isnothing(_ub) ? fill(T(Inf), nvars) : T.(_ub)
    @assert all( lb .<= ub ) "All lower bounds must be less than or equal to upper bounds."
    var_scaler = init_scaler(nvars, _lb, _ub)

    x = T.(x0)
    fx = prealloc_fx(ev)
    ex = prealloc_ex(ev)
    ix = prealloc_ix(ev)

    Dfx = prealloc_Dfx(ev, nvars)
    Dex = prealloc_Dex(ev, nvars)
    Dix = prealloc_Dix(ev, nvars)

    d = similar(x)
    n = similar(x)

end
=#
end
