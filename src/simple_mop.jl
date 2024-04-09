# # Basic Proof-Of-Concept Implementations of Multi-Objective Problems

# ## Scaled Evaluation
# The optimizer might internally apply a scaling to the variables.
# The user-provided multi-objective problem does not know about this scaling
# and we have to consider it when evaluating and differntiating the problem
# functions.
#
# Currently we restrict ourselves to affine-linear transformations.
# Given `scaler::AbstractAffineScaler`,
# consider the unscaled variable vector ``ξ``.
# The scaled variable is ``x = Tξ + t``, where ``T`` is the `scaling_matrix(scaler)`
# and ``t`` is the `scaling_offset(scaler)`.
# Unscaling happens via ``ξ = Sx + s``, where ``S=T^{-1}`` is 
# `unscaling_matrix(scaler)` and ``s=-T^{-1}t`` is 
# `unscaling_offset(scaler)`.
#
# A `ScaledOperator` wraps some other `AbstractNonlinearOperator` and enables caching
# of (un-)scaling results as well taking care of the chain rule for derivatives:

import .CompromiseEvaluators: FuncCallCounter, AbstractNonlinearOperatorWrapper

struct RecountedOperator{O} <: AbstractNonlinearOperatorWrapper
    op :: O
    num_func_calls :: Union{FuncCallCounter, Nothing}
    num_grad_calls :: Union{FuncCallCounter, Nothing}
    num_hess_calls :: Union{FuncCallCounter, Nothing}
end

function RecountedOperator(op; replace_counters::Bool=false)
    if !replace_counters
        return RecountedOperator(op, nothing, nothing, nothing)
    end
    return RecountedOperator(op, FuncCallCounter(), FuncCallCounter(), FuncCallCounter())
end

recount_operator(op::Nothing; kwargs...) = nothing
recount_operator(op; kwargs...)=RecountedOperator(op; kwargs...)

CE.wrapped_operator(sop::RecountedOperator)=sop.op

function CE.func_call_counter(sop::RecountedOperator, v::Val{0})
    isnothing(sop.num_func_calls) && return CE.func_call_counter(sop.op, v)
    return sop.num_func_calls
end
function CE.func_call_counter(sop::RecountedOperator, v::Val{1})
    isnothing(sop.num_grad_calls) && return CE.func_call_counter(sop.op, v)
    return sop.num_grad_calls
end
function CE.func_call_counter(sop::RecountedOperator, v::Val{2})
    isnothing(sop.num_hess_calls) && return CE.func_call_counter(sop.op, v)
    return sop.num_hess_calls
end
struct ScaledOperator{F<:AbstractFloat, O, S} <: AbstractNonlinearOperatorWrapper
    op :: O
    scaler :: S
    ## cache for unscaled site
    ξ :: Matrix{F}
end
CE.wrapped_operator(sop::ScaledOperator)=sop.op
function CE.operator_chunk_size(sop::ScaledOperator)
    return min(size(sop.ξ, 2), CE.operator_chunk_size(sop.op))
end

function CE.preprocess_inputs(sop::ScaledOperator, x::RVec)
    @unpack scaler, ξ = sop
    _ξ = @view(ξ[:, 1])
    copyto!(_ξ, x)
    apply_scaling!(_ξ, scaler, InverseScaling())
    return _ξ
end

function CE.preprocess_inputs(sop::ScaledOperator, x::RMat)
    @unpack scaler, ξ, op = sop
    n_x = size(x, 2) 
    _ξ = @view(ξ[:, 1:n_x])
    copyto!(_ξ, x)
    apply_scaling!(_ξ, scaler, InverseScaling())
    return _ξ
end

function CE.postprocess_grads!(Dy, sop::ScaledOperator, x_pre, x_post)
    @unpack scaler = sop
    scale_grads!(Dy, scaler)
    return nothing
end
function CE.postprocess_hessians!(H, sop::ScaledOperator, x_pre, x_post)
    @unpack scaler = sop
    scale_hessians!(H, scaler)
end

# ## `SimpleMOP`
# In this section, we will define two kinds of simple problem structures:
abstract type SimpleMOP <: AbstractMOP end

# A `MutableMOP` is meant to be initialized empty and built up iteratively.
"""
    MutableMOP(; num_vars, kwargs...)

Initialize a multi-objective problem with `num_vars` variables.

# Functions
There can be exactly one (possibly vector-valued) objective function,
one nonlinear equality constraint function, and one nonlinear inequality constraint function.
For now, they have to be of type `NonlinearFunction`.
You could provide these functions with the keyword-arguments `objectives`,
`nl_eq_constraints` or `nl_ineq_constraints` or set the fields of the same name.
To conveniently add user-provided functions, there are helper functions, 
like `add_objectives!`.

## LinearConstraints
Box constraints are defined by the vectors `lb` and `ub`.
Linear equality constraints ``Ex=c`` are defined by the matrix `E` and the vector `c`.
Inequality constraints read ``Ax≤b`` and use `A` and `b`.

# Surrogate Configuration
Use the keyword-arguments `mcfg_objectives` to provide an `AbstractSurrogateModelConfig`
to define how the objectives should be modelled.
By default, we assume `ExactModelConfig()`, which requires differentiable objectives.
"""
@with_kw mutable struct MutableMOP <: SimpleMOP
    objectives :: Union{Nothing, NonlinearFunction} = nothing
    nl_eq_constraints :: Union{Nothing, NonlinearFunction} = nothing
    nl_ineq_constraints :: Union{Nothing, NonlinearFunction} = nothing

    reset_call_counters :: Bool = true

    num_vars :: Int

    x0 :: Union{Nothing, RVec, RMat} = nothing

    dim_objectives :: Int = 0
    dim_nl_eq_constraints :: Int = 0
    dim_nl_ineq_constraints :: Int = 0

    mcfg_objectives :: AbstractSurrogateModelConfig = ExactModelConfig()
    mcfg_nl_eq_constraints :: AbstractSurrogateModelConfig = ExactModelConfig()
    mcfg_nl_ineq_constraints :: AbstractSurrogateModelConfig = ExactModelConfig()

    lb :: Union{Nothing, Vector{Float64}} = nothing
    ub :: Union{Nothing, Vector{Float64}} = nothing

    E :: Union{Nothing, Matrix{Float64}} = nothing
    c :: Union{Nothing, Vector{Float64}} = nothing
    A :: Union{Nothing, Matrix{Float64}} = nothing
    b :: Union{Nothing, Vector{Float64}} = nothing
end
MutableMOP(num_vars::Int)=MutableMOP(;num_vars)

# The `TypedMOP` looks nearly the same, but is strongly typed and immutable.
# We initialize a `TypedMOP` from a `MutableMOP` before optimization for 
# performance reasons:
struct TypedMOP{
    O, NLEC,NLIC,
    XType,
    MTO, MTNLEC, MTNLIC,
    LB, UB, 
    EType, CType, AType, BType
} <: SimpleMOP
    objectives :: O
    nl_eq_constraints :: NLEC
    nl_ineq_constraints :: NLIC

    reset_call_counters :: Bool

    num_vars :: Int

    x0 :: XType
    
    dim_objectives :: Int
    dim_nl_eq_constraints :: Int
    dim_nl_ineq_constraints :: Int

    mcfg_objectives :: MTO
    mcfg_nl_eq_constraints :: MTNLEC
    mcfg_nl_ineq_constraints :: MTNLIC 

    lb :: LB
    ub :: UB

    E :: EType
    c :: CType
    A :: AType
    b :: BType
end

# This initialization really is just a forwarding of all fields:
function initialize(mop::SimpleMOP)
    replace_counters = mop.reset_call_counters
    return TypedMOP(
        recount_operator(mop.objectives; replace_counters),
        recount_operator(mop.nl_eq_constraints; replace_counters),
        recount_operator(mop.nl_ineq_constraints; replace_counters), 
        mop.reset_call_counters,
        mop.num_vars,
        mop.x0,
        mop.dim_objectives,
        mop.dim_nl_eq_constraints,
        mop.dim_nl_ineq_constraints,
        mop.mcfg_objectives,
        mop.mcfg_nl_eq_constraints,
        mop.mcfg_nl_ineq_constraints,
        mop.lb,
        mop.ub,
        mop.E, mop.c,
        mop.A, mop.b
   )
end

# ### Iterative Problem Setup

# First, let's define some helpers to parse model configuration specifications:
# If a configuration is given already, do nothing:
parse_mcfg(mcfg::AbstractSurrogateModelConfig)=mcfg
# We can also allow Symbols:
parse_mcfg(mcfg_symb::Symbol)=parse_mcfg(Val(mcfg_symb))
parse_mcfg(::Val{:exact})=ExactModelConfig()
parse_mcfg(::Val{:rbf})=RBFConfig()
parse_mcfg(::Val{:taylor1})=TaylorPolynomialConfig(;degree=1)
parse_mcfg(::Val{:taylor2})=TaylorPolynomialConfig(;degree=2)
function parse_mcfg(::Val{:rbfLocked})
    database_rwlock = Compromise.ConcurrentRWLock()
    if isnothing(database_rwlock)
        error("Cannot add RBF using read-write-lock without `ConcurrentUtils`.")
    end
    cfg = RBFConfig(; database_rwlock)
    return cfg
end
# The default value `nothing` redirects to an `ExactModelConfig`:
parse_mcfg(::Nothing)=parse_mcfg(:exact)

# Add function is the backend for the helper functions `add_objectives!` etc.
"""
    add_function!(func_field, mop, op, model_cfg; dim_out, backend=NoBackend())

Add the operator `op` to `mop` at `func_field` and use model configuration `model_cfg`.
Keyword argument `dim_out::Int` is mandatory.
E.g., `add_function!(:objectives, mop, op, :rbf; dim_out=2)` adds `op`
as the bi-valued objective to `mop`.
"""
function add_function!(
    func_field::Symbol, mop::MutableMOP, op::AbstractNonlinearOperator, 
    model_cfg::Union{AbstractSurrogateModelConfig, Nothing, Symbol}=nothing;
    dim_out::Int
)

    setfield!(mop, func_field, op)
    setfield!(mop, Symbol("dim_", func_field), dim_out)

    mod = parse_mcfg(model_cfg)
    setfield!(mop, Symbol("mcfg_", func_field), mod)
    return nothing
end

# We now generate the derived helper functions `add_objectives!`,
# `add_nl_ineq_constraints!` and `add_nl_eq_constraints!`.
# Here, we additionally allow for `Function`s to be used instead of `NonlinearFunction`s.
function add_objectives!(mop, args...; kwargs...) end
function add_nl_eq_constraints!(mop, args...; kwargs...) end
function add_nl_ineq_constraints!(mop, args...; kwargs...) end

for (fntype, typenoun) in (
    (:objectives, "objectives"),
    (:nl_eq_constraints, "nonlinear equality constraints"),
    (:nl_ineq_constraints, "nonlinear inequality constraints")
)
    add_fn = Symbol("add_", fntype, "!")
    @eval begin
        function $(add_fn)(
            mop::MutableMOP, op::NonlinearFunction, model_cfg=nothing; dim_out::Int)
            return add_function!($(Meta.quot(fntype)), mop, op, model_cfg; dim_out)
        end#function
        
        """
            $($(add_fn))(mop::MutableMOP, func, model_cfg=nothing; 
                dim_out::Int, kwargs...)

        Set function `func` to return the $($(typenoun)) vector of `mop`.
        Argument `model_cfg` is optional and specifies the surrogate models for `func`.
        Can be `nothing`, a Symbol (`:exact`, `:rbf`, `taylor1`, `taylor2`), or an
        `AbstractSurrogateModelConfig` object.

        All functions can be in-place, see keyword arguments `func_iip`.
        
        Keyword argument `dim_out` is mandatory and corresponds to the length of the result
        vector.
        The other `kwargs...` are passed to the inner `AbstractNonlinearOperator` as is.
        For options and defaults see [`NonlinearParametricFunction`](@ref).
        """
        function $(add_fn)(
            mop::MutableMOP, func::Function, model_cfg=nothing; dim_out::Int, kwargs...
        )
            if dim_out > 0
                op = NonlinearFunction(; func, kwargs...)
                return add_function!($(Meta.quot(fntype)), mop, op, model_cfg; dim_out)
            else
                @warn "`dim_out` must be positive. Not adding function to problem."
                return nothing
            end
        end

        # Define methods to allow hand-crafted gradient functions without keyword
        # arguments:
        #src #TODO allow for passing Hessians...

        """
            $($(add_fn))(mop::MutableMOP, func, grads, model_cfg=nothing; 
                dim_out::Int, kwargs...)

        Set function `func` to return the $($(typenoun)) vector of `mop`.
        Argument `model_cfg` is optional and specifies the surrogate models for `func`.
        Can be `nothing`, a Symbol (`:exact`, `:rbf`, `taylor1`, `taylor2`), or an
        `AbstractSurrogateModelConfig` object.
        `grads` should be a function mapping a vector to the transposed jacobian of `func`.
        
        All functions can be in-place, see keyword arguments `func_iip` and `grads_iip`.
        
        Keyword argument `dim_out` is mandatory and corresponds to the length of the result
        vector.
        The other `kwargs...` are passed to the inner `AbstractNonlinearOperator` as is.
        For options and defaults see [`NonlinearParametricFunction`](@ref).
        """
        function $(add_fn)(
            mop::MutableMOP, func::Function, grads::Function, model_cfg=nothing; dim_out::Int, kwargs...
        )
            return $(add_fn)(mop, func, model_cfg; dim_out, grads, kwargs...)
        end
    
        """
            $($(add_fn))(mop::MutableMOP, func, grads, func_and_grads, model_cfg=nothing; 
                dim_out::Int, kwargs...)

        Set function `func` to return the $($(typenoun)) vector of `mop`.
        Argument `model_cfg` is optional and specifies the surrogate models for `func`.
        Can be `nothing`, a Symbol (`:exact`, `:rbf`, `taylor1`, `taylor2`), or an
        `AbstractSurrogateModelConfig` object.
        `grads` should be a function mapping a vector to the transposed jacobian of `func`,
        while `func_and_grads` returns a primal vector and the gradients at the same time.
        
        All functions can be in-place, see keyword arguments `func_iip`, `grads_iip` and 
        `func_and_grads_iip`.
        
        Keyword argument `dim_out` is mandatory and corresponds to the length of the result
        vector.
        The other `kwargs...` are passed to the inner `AbstractNonlinearOperator` as is.
        For options and defaults see [`NonlinearParametricFunction`](@ref).
        """
        function $(add_fn)(
            mop::MutableMOP, func::Function, grads::Function, func_and_grads::Function, 
            model_cfg=nothing; 
            dim_out::Int, kwargs...
        )
            return $(add_fn)(mop, func, model_cfg; dim_out, grads, func_and_grads, kwargs...)
        end
    end#@eval
end

# ### Interface Implementation
# Define the most basic Getters:
float_type(::SimpleMOP) = Float64

initial_vars(mop::SimpleMOP) = mop.x0

# Box constraints are easy:
lower_var_bounds(mop::SimpleMOP)=mop.lb
upper_var_bounds(mop::SimpleMOP)=mop.ub

# To access functions, retrieve the corresponding fields of the MOP:
dim_vars(mop::SimpleMOP)=mop.num_vars
for dim_func in (:dim_objectives, :dim_nl_eq_constraints, :dim_nl_ineq_constraints)
    @eval function $(dim_func)(mop::SimpleMOP)
        return getfield(mop, $(Meta.quot(dim_func)))
    end
end

# Linear constraints are returned only if the matrix is not nothing.
# If the offset vector is nothing, we return zeros:
lin_eq_constraints_matrix(mop::Union{TypedMOP, MutableMOP})=mop.E
lin_eq_constraints_vector(mop::Union{TypedMOP, MutableMOP})=mop.c
lin_ineq_constraints_matrix(mop::Union{TypedMOP, MutableMOP})=mop.A
lin_ineq_constraints_vector(mop::Union{TypedMOP, MutableMOP})=mop.b

# Because we store `AbstractNonlinearOperator`s, evaluation can simply be redirected:
function eval_objectives!(y::RVec, mop::SimpleMOP, x::RVec)
    return func_vals!(y, mop.objectives, x)
end
function eval_nl_eq_constraints!(y::RVec, mop::SimpleMOP, x::RVec)
    return func_vals!(y, mop.nl_eq_constraints, x)
end
function eval_nl_ineq_constraints!(y::RVec, mop::SimpleMOP, x::RVec)
    return func_vals!(y, mop.nl_ineq_constraints, x)
end

# ## Surrogate Modelling of `SimpleMOP`

# For `SimpleMOP`, the surrogates are stored in the type-stable structure
# `SimpleMOPSurrogate`, with exactly one surrogate for each kind of problem 
# function.
# We additionally store the operators, because we might wrap them as
# `ScaledOperator` and don't want to modify the original problem.
# (Also, I am not sure if the scaler is available when the problem is initialized).
struct SimpleMOPSurrogate{
    O, NLEC, NLIC, MO, MNLEC, MNLIC,
} <: AbstractMOPSurrogate
    objectives :: O
    nl_eq_constraints :: NLEC
    nl_ineq_constraints :: NLIC
    
    num_vars :: Int
    dim_objectives :: Int
    dim_nl_eq_constraints :: Int
    dim_nl_ineq_constraints :: Int

    mod_objectives :: MO
    mod_nl_eq_constraints :: MNLEC
    mod_nl_ineq_constraints :: MNLIC
end

# The float_type is again fixed:
float_type(mod::SimpleMOPSurrogate)=Float64

# Getters are generated automatically:
dim_vars(mod::SimpleMOPSurrogate)=mod.num_vars
for dim_func in (:dim_objectives, :dim_nl_eq_constraints, :dim_nl_ineq_constraints)
    @eval function $(dim_func)(mod::SimpleMOPSurrogate)
        return getfield(mod, $(Meta.quot(dim_func)))
    end
end

# The model container is dependent on the trust-region if any of the models is:
function depends_on_radius(mod::SimpleMOPSurrogate)
    return (
        (!isnothing(mod.mod_objectives) && CE.depends_on_radius(mod.mod_objectives)) ||
        (!isnothing(mod.mod_nl_eq_constraints) && CE.depends_on_radius(mod.mod_nl_eq_constraints)) ||
        (!isnothing(mod.mod_nl_ineq_constraints) && CE.depends_on_radius(mod.mod_nl_ineq_constraints))
    )
end

# At the moment, I have not yet thought about what we would need to change for dynamic scaling:
supports_scaling(::Type{<:SimpleMOPSurrogate})=ConstantAffineScaling()

# Evaluation redirects to evaluation of the surrogates:
function eval_objectives!(y::RVec, mod::SimpleMOPSurrogate, x::RVec)
    return func_vals!(y, mod.mod_objectives, x)
end
function eval_nl_eq_constraints!(y::RVec, mod::SimpleMOPSurrogate, x::RVec)
    return func_vals!(y, mod.mod_nl_eq_constraints, x)
end
function eval_nl_ineq_constraints!(y::RVec, mod::SimpleMOPSurrogate, x::RVec)
    return func_vals!(y, mod.mod_nl_ineq_constraints, x)
end
# Gradients:
function grads_objectives!(Dy::RMat, mod::SimpleMOPSurrogate, x::RVec)
    return func_grads!(Dy, mod.mod_objectives, x)
end
function grads_nl_eq_constraints!(Dy::RMat, mod::SimpleMOPSurrogate, x::RVec)
    return func_grads!(Dy, mod.mod_nl_eq_constraints, x)
end
function grads_nl_ineq_constraints!(Dy::RMat, mod::SimpleMOPSurrogate, x::RVec)
    return func_grads!(Dy, mod.mod_nl_ineq_constraints, x)
end
function eval_and_grads_objectives!(y::RVec, Dy::RMat, mod::SimpleMOPSurrogate, x::RVec)
    return func_vals_and_grads!(y, Dy, mod.mod_objectives, x)
end
function eval_and_grads_nl_eq_constraints!(y::RVec, Dy::RMat, mod::SimpleMOPSurrogate, x::RVec)
    return func_vals_and_grads!(y, Dy, mod.mod_nl_eq_constraints, x)
end
function eval_and_grads_nl_ineq_constraints!(y::RVec, Dy::RMat, mod::SimpleMOPSurrogate, x::RVec)
    return func_vals_and_grads!(y, Dy, mod.mod_nl_ineq_constraints, x)
end
# ### Initialization and Training
# These helpers make an operator respect scaling by returning `ScaledOperator`:
scale_wrap_op(scaler::IdentityScaler, op, mcfg, dim_in, dim_out, T)=op
function scale_wrap_op(
    scaler, op, mcfg, dim_in, dim_out, T;
)
    isnothing(op) && return op
    op_cs = CE.operator_chunk_size(op)
    if isinf(op_cs) || op_cs <= 0
        op_cs = dim_in + 1
    end
    ξ = zeros(T, dim_in, op_cs)
    return ScaledOperator(op, scaler, ξ)
end

# We use the `ScaledOperator` in the `SimpleMOPSurrogate` to make writing models
# as easy as possible. 
# Modellers should always assume working in the scaled domain and not be bothered
# with transformations...
function init_models(mop::SimpleMOP, scaler; kwargs...)
    d_objf = dim_objectives(mop)
    d_nl_eq = dim_nl_eq_constraints(mop)
    d_nl_ineq = dim_nl_ineq_constraints(mop)
    d_in = mop.num_vars

    objf = scale_wrap_op(
        scaler, mop.objectives, mop.mcfg_objectives, d_in, d_objf, Float64;
    )
    nl_eq = scale_wrap_op(
        scaler, mop.nl_eq_constraints, mop.mcfg_nl_eq_constraints, d_in, d_nl_eq, Float64;
    )
    nl_ineq = scale_wrap_op(
        scaler, mop.nl_ineq_constraints, mop.mcfg_nl_ineq_constraints, d_in, d_nl_ineq, Float64;
    )

    mobjf = isnothing(objf) ? nothing : init_surrogate(
        mop.mcfg_objectives, objf, d_in, d_objf, nothing, Float64; kwargs...)
    mnl_eq = isnothing(nl_eq) ? nothing : init_surrogate(
        mop.mcfg_nl_eq_constraints, nl_eq, d_in, d_nl_eq, nothing, Float64; kwargs...)
    mnl_ineq = isnothing(nl_ineq) ? nothing : init_surrogate(
        mop.mcfg_nl_ineq_constraints, nl_ineq, d_in, d_nl_ineq, nothing, Float64; kwargs...)

    return SimpleMOPSurrogate(
        objf, nl_eq, nl_ineq, d_in, d_objf, d_nl_eq, d_nl_ineq, mobjf, mnl_eq, mnl_ineq)
end

# The sub-models are trained separately:
function update_models!(
    mod::SimpleMOPSurrogate, Δ, scaler, vals, scaled_cons;
    log_level::LogLevel, indent::Int
)
    x = cached_x(vals)
    @unpack lb, ub = scaled_cons
    if !isnothing(mod.mod_objectives)
        @ignoraise update!(
            mod.mod_objectives, mod.objectives, Δ, x, cached_fx(vals), lb, ub; log_level, indent)
    end
    if !isnothing(mod.mod_nl_eq_constraints)
        @ignoraise update!(
            mod.mod_nl_eq_constraints, mod.nl_eq_constraints, Δ, x, cached_hx(vals), lb, ub; log_level, indent)
    end
    if !isnothing(mod.mod_nl_ineq_constraints)
        @ignoraise update!(
            mod.mod_nl_ineq_constraints, mod.nl_ineq_constraints, Δ, x, cached_gx(vals), lb, ub; log_level, indent)
    end
    return nothing
end

function universal_copy(mod::SimpleMOPSurrogate)
    mobjf = isnothing(mod.mod_objectives) ? nothing : CE.universal_copy_model(mod.mod_objectives)
    mnleq = isnothing(mod.nl_eq_constraints) ? nothing :  CE.universal_copy_model(mod.mod_nl_eq_constraints)
    mnlineq = isnothing(mod.nl_ineq_constraints) ? nothing : CE.universal_copy_model(mod.mod_nl_ineq_constraints)
    return SimpleMOPSurrogate(
        mod.objectives,
        mod.nl_eq_constraints,
        mod.nl_ineq_constraints,
        mod.num_vars,
        mod.dim_objectives,
        mod.dim_nl_eq_constraints,
        mod.dim_nl_ineq_constraints,
        mobjf,
        mnleq,
        mnlineq,
    )
end

function universal_copy!(
    mod_trgt::SimpleMOPSurrogate, mod_src::SimpleMOPSurrogate
)
    if !isnothing(mod_trgt.mod_objectives)
        CE.universal_copy_model!(mod_trgt.mod_objectives, mod_src.mod_objectives)
    end
    if !isnothing(mod_trgt.mod_nl_eq_constraints)
        CE.universal_copy_model!(mod_trgt.mod_nl_eq_constraints, mod_src.mod_nl_eq_constraints)
    end
    if !isnothing(mod_trgt.mod_nl_ineq_constraints)
        CE.universal_copy_model!(mod_trgt.mod_nl_ineq_constraints, mod_src.mod_nl_ineq_constraints)
    end
    return mod_trgt
end

function process_trial_point!(mod::SimpleMOPSurrogate, vals_trial, update_results)
    xtrial = cached_x(vals_trial)
    isnext = update_results.point_has_changed
    if !isnothing(mod.mod_objectives)
        CE.process_trial_point!(
            mod.mod_objectives, xtrial, cached_fx(vals_trial), isnext)
    end
    if !isnothing(mod.mod_nl_eq_constraints)
        CE.process_trial_point!(
            mod.mod_nl_eq_constraints, xtrial, cached_hx(vals_trial), isnext)
    end
    if !isnothing(mod.mod_nl_ineq_constraints)
        CE.process_trial_point!(
            mod.mod_nl_ineq_constraints, xtrial, cached_gx(vals_trial), isnext)
    end
    return nothing
end

include("simple_caches.jl")