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

import .CompromiseEvaluators: FuncCallCounter, AbstractNonlinearOperatorWrapper, BudgetExhausted

struct RecountedOperator{O} <: AbstractNonlinearOperatorWrapper
    op :: O
    num_func_calls :: Union{FuncCallCounter, Nothing}
    num_grad_calls :: Union{FuncCallCounter, Nothing}
    num_hess_calls :: Union{FuncCallCounter, Nothing}
end
@batteries RecountedOperator selfconstructor=false

recount_operator(op::Nothing, ::Val{true}) = nothing
recount_operator(op::Nothing, ::Val{false}) = nothing
recount_operator(op, new_counters::Val{false})=RecountedOperator(op, nothing, nothing, nothing)
function recount_operator(op, new_counters::Val{true})
    RecountedOperator(op, FuncCallCounter(), FuncCallCounter(), FuncCallCounter())
end
function recount_operator(op::RecountedOperator, new_counters::Val{false})
    return op
end
function recount_operator(op::RecountedOperator, new_counters::Val{true})
    return recount_operator(op.op, new_counters)
end

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
@batteries ScaledOperator selfconstructor=false
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

stop_type(::SimpleMOP) = BudgetExhausted
_reset_call_counters(::SimpleMOP) = Val{true}()

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

    num_vars :: Int = -1

    x0 :: Union{Nothing, RVec, RMat} = nothing

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
@batteries MutableMOP selfconstructor=false
MutableMOP(num_vars::Int)=MutableMOP(;num_vars)
_reset_call_counters(mop::MutableMOP)=Val(mop.reset_call_counters)

# The `TypedMOP` looks nearly the same, but is strongly typed and immutable.
# We initialize a `TypedMOP` from a `MutableMOP` before optimization for 
# performance reasons:
Base.@kwdef struct TypedMOP{
    O, NLEC, NLIC,
    rccType,
    XType,
    MTO, MTNLEC, MTNLIC,
    LB, UB, 
    EType, CType, AType, BType
} <: SimpleMOP
    objectives :: O = nothing
    nl_eq_constraints :: NLEC = nothing
    nl_ineq_constraints :: NLIC = nothing

    reset_call_counters :: rccType = Val{true}()

    num_vars :: Int = -1

    x0 :: XType = nothing
    
    mcfg_objectives :: MTO = ExactModelConfig()
    mcfg_nl_eq_constraints :: MTNLEC = ExactModelConfig()
    mcfg_nl_ineq_constraints :: MTNLIC = ExactModelConfig()

    lb :: LB = nothing
    ub :: UB = nothing

    E :: EType = nothing
    c :: CType = nothing
    A :: AType = nothing
    b :: BType = nothing
end
@batteries TypedMOP selfconstructor=false
_reset_call_counters(mop::TypedMOP) = mop.reset_call_counters

# This initialization really is just a forwarding of all fields:
function initialize(mop::SimpleMOP)
    replace_counters = _reset_call_counters(mop)
    new_objectives = recount_operator(mop.objectives, replace_counters)
    new_nl_eq_constraints = recount_operator(mop.nl_eq_constraints, replace_counters)
    new_nl_ineq_constraints = recount_operator(mop.nl_ineq_constraints, replace_counters)
    return TypedMOP(
        new_objectives,
        new_nl_eq_constraints,
        new_nl_ineq_constraints,
        mop.reset_call_counters,
        mop.num_vars,
        mop.x0,
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
    field_val::Val{field}, mop::MutableMOP, op::AbstractNonlinearOperator, 
    model_cfg::Union{AbstractSurrogateModelConfig, Nothing, Symbol}=nothing;
) where field

    setproperty!(mop, field, op)
    mod_cfg = parse_mcfg(model_cfg)
    _mcfg_field!(mop, field_val, mod_cfg)
    return nothing
end
_mcfg_field!(mop::SimpleMOP, ::Val{:objectives}, cfg)= (mop.mcfg_objectives = cfg)
_mcfg_field!(mop::SimpleMOP, ::Val{:nl_eq_constraints}, cfg) = (mop.mcfg_nl_eq_constraints = cfg)
_mcfg_field!(mop::SimpleMOP, ::Val{:nl_ineq_constraints}, cfg) = (mop.mcfg_nl_ineq_constraints = cfg)

# We now generate the derived helper functions `add_objectives!`,
# `add_nl_ineq_constraints!` and `add_nl_eq_constraints!`.
# Here, we additionally allow for `Function`s to be used instead of `NonlinearFunction`s.
function add_objectives!(mop, args...; kwargs...) 
    @warn("`add_objectives!` fallback.") 
end
function add_nl_eq_constraints!(mop, args...; kwargs...)
    @warn("`add_nl_eq_constraints!` fallback.") 
end
function add_nl_ineq_constraints!(mop, args...; kwargs...)
    @warn("`add_nl_ineq_constraints!` fallback.")
end

for (fntype, typenoun) in (
    (:objectives, "objectives"),
    (:nl_eq_constraints, "nonlinear equality constraints"),
    (:nl_ineq_constraints, "nonlinear inequality constraints")
)
    add_fn = Symbol("add_", fntype, "!")
    @eval begin
        function $(add_fn)(
            mop::MutableMOP, op::NonlinearFunction, model_cfg=nothing,
        )
            return add_function!(Val($(Meta.quot(fntype))), mop, op, parse_mcfg(model_cfg))
        end#function

        """
            $($(add_fn))(mop::MutableMOP, func, model_cfg=nothing; 
                dim_out::Int, kwargs...)

        Set function `func` to return the $($(typenoun)) vector of `mop`.
        Argument `model_cfg` is optional and specifies the surrogate models for `func`.
        Can be `nothing`, a Symbol (`:exact`, `:rbf`, `taylor1`, `taylor2`), or an
        `AbstractSurrogateModelConfig` object.

        All functions can be in-place, see keyword argument `func_iip`.
        
        Keyword argument `dim_out` is mandatory and corresponds to the length of the result
        vector.
        If `dim_vars(mop) <= 0`, then `dim_in` is also mandatory.
        The other `kwargs...` are passed to the inner `AbstractNonlinearOperator` as is.
        For options and defaults see [`NonlinearParametricFunction`](@ref).
        """
        function $(add_fn)(
            mop::MutableMOP, func::Function, model_cfg=nothing; dim_out::Int, kwargs...
        )
            if dim_out > 0
                dim_in = get(kwargs, :dim_in, nothing)
                if isnothing(dim_in)
                    dim_in_mop = dim_vars(mop)
                    if dim_in_mop <= 0
                        @warn "`dim_in` not specified, please give positive integer. Not adding function to problem."
                        return nothing
                    end
                    dim_in = dim_in_mop
                end
                dim_in_mop = dim_vars(mop)
                if dim_in != dim_in_mop
                    if dim_in_mop <= 0
                        mop.num_vars = dim_in
                    else
                        @warn "`mop` has $(dim_in_mop) variables, but function has `dim_in`=$(dim_in). Not adding function to problem."
                        return nothing
                    end
                end

                op = NonlinearFunction(; func, dim_out, kwargs..., dim_in)  # rightmost argument has precedence
                return add_function!(Val($(Meta.quot(fntype))), mop, op, model_cfg)
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
# #### “Mutable” TypedMOP
function Accessors.set(
    mop::TypedMOP, 
    lens::Union{
        PropertyLens{:objectives}, 
        PropertyLens{:nl_eq_constraintss}, 
        PropertyLens{:nl_ineq_constraints},
    },
    op::AbstractNonlinearOperator
)
    dim_in_op = CE.operator_dim_in(op)
    patch = (;)
    patch = Accessors.insert(patch, lens, op)
    if dim_vars(mop) <= 0
        patch = Accessors.insert(patch, PropertyLens{:num_vars}(), dim_in_op)
    else
        @assert dim_vars(mop) == CE.operator_dim_in(op)
    end
    return Accessors.setproperties(mop, patch)
end

function Accessors.set(
    mop::TypedMOP, 
    lens::Union{
        PropertyLens{:mcfg_objectives}, 
        PropertyLens{:mcfg_nl_eq_constraintss}, 
        PropertyLens{:mcfg_nl_ineq_constraints},
    },
    mcfg::Union{Symbol, AbstractSurrogateModelConfig, Nothing}
)
    patch = (;)
    patch = Accessors.insert(patch, lens, parse_mcfg(mcfg))
    return Accessors.setproperties(mop, patch)
end

# ### Interface Implementation
# Define the most basic Getters:
float_type(::SimpleMOP) = Float64

initial_vars(mop::SimpleMOP) = mop.x0

# Box constraints are easy:
lower_var_bounds(mop::SimpleMOP)=mop.lb
upper_var_bounds(mop::SimpleMOP)=mop.ub

# To access functions, retrieve the corresponding fields of the MOP:
dim_vars(mop::SimpleMOP)=mop.num_vars::Int
dim_objectives(mop::SimpleMOP)=simple_op_dim_out(mop.objectives)::Int
dim_nl_eq_constraints(mop::SimpleMOP)=simple_op_dim_out(mop.nl_eq_constraints)::Int
dim_nl_ineq_constraints(mop::SimpleMOP)=simple_op_dim_out(mop.nl_ineq_constraints)::Int
simple_op_dim_out(::Nothing)=0
simple_op_dim_out(op)=CE.operator_dim_out(op)

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
   
    mod_objectives :: MO
    mod_nl_eq_constraints :: MNLEC
    mod_nl_ineq_constraints :: MNLIC
end
@batteries SimpleMOPSurrogate selfconstructor=false
function CE.copy_model(mod::SimpleMOPSurrogate)
    @unpack objectives, nl_eq_constraints, nl_ineq_constraints, num_vars, mod_objectives,
        mod_nl_eq_constraints, mod_nl_ineq_constraints = mod
    return SimpleMOPSurrogate(
        objectives,
        nl_eq_constraints,
        nl_ineq_constraints,
        num_vars,
        CE.copy_model(mod_objectives),
        CE.copy_model(mod_nl_eq_constraints),
        CE.copy_model(mod_nl_ineq_constraints)
    )
end

# The float_type is again fixed:
float_type(mod::SimpleMOPSurrogate)=Float64
stop_type(mod::SimpleMOPSurrogate)=Union{BudgetExhausted, RBFModels.RBFConstructionImpossible}

# Getters are generated automatically:
dim_vars(mod::SimpleMOPSurrogate)=mod.num_vars::Int
dim_objectives(mop::SimpleMOPSurrogate)=simple_op_dim_out(mop.objectives)::Int
dim_nl_eq_constraints(mop::SimpleMOPSurrogate)=simple_op_dim_out(mop.nl_eq_constraints)::Int
dim_nl_ineq_constraints(mop::SimpleMOPSurrogate)=simple_op_dim_out(mop.nl_ineq_constraints)::Int

# The model container is dependent on the trust-region if any of the models is:
function depends_on_radius(mod::SimpleMOPSurrogate)
    return (
        simple_model_depends_on_radius(mod.mod_objectives) ||
        simple_model_depends_on_radius(mod.mod_nl_eq_constraints) ||
        simple_model_depends_on_radius(mod.mod_nl_ineq_constraints) 
    )
end
simple_model_depends_on_radius(::Nothing)=false
simple_model_depends_on_radius(mod)=CE.depends_on_radius(mod)

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
scale_wrap_op(scaler::IdentityScaler, op, mcfg, T)=op
function scale_wrap_op(
    scaler, op, mcfg, T;
)
    isnothing(op) && return op
    op_cs = CE.operator_chunk_size(op)
    dim_in = CE.operator_dim_in(op)
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
    objf = scale_wrap_op(
        scaler, mop.objectives, mop.mcfg_objectives, Float64;
    )
    nl_eq = scale_wrap_op(
        scaler, mop.nl_eq_constraints, mop.mcfg_nl_eq_constraints, Float64;
    )
    nl_ineq = scale_wrap_op(
        scaler, mop.nl_ineq_constraints, mop.mcfg_nl_ineq_constraints, Float64;
    )

    mobjf = init_simple_surrogate(objf, mop.mcfg_objectives; kwargs...)
    mnl_eq = init_simple_surrogate(nl_eq, mop.mcfg_nl_eq_constraints; kwargs...)
    mnl_ineq = init_simple_surrogate(nl_ineq, mop.mcfg_nl_ineq_constraints; kwargs...)

    return SimpleMOPSurrogate(
        objf, 
        nl_eq, 
        nl_ineq, 
        mop.num_vars, 
        mobjf, 
        mnl_eq, 
        mnl_ineq
    )
end
init_simple_surrogate(op::Nothing, cfg; kwargs...)=nothing
function init_simple_surrogate(op, cfg; kwargs...) 
    return init_surrogate(cfg, op, nothing, Float64; kwargs...)
end
# The sub-models are trained separately:
function update_models!(
    mod::SimpleMOPSurrogate, Δ, scaler, vals, scaled_cons;
    log_level::LogLevel, indent::Int
)
    x = cached_x(vals)
    @unpack lb, ub = scaled_cons
    if !isnothing(mod.mod_objectives)
        @logmsg log_level "$(indent_str(indent))(Objectives)"
        @ignoraise update!(
            mod.mod_objectives, mod.objectives, Δ, x, cached_fx(vals), lb, ub; log_level, indent)
    end
    if !isnothing(mod.mod_nl_eq_constraints)
        @logmsg log_level "$(indent_str(indent))(Eq. Constraints)"
        @ignoraise update!(
            mod.mod_nl_eq_constraints, mod.nl_eq_constraints, Δ, x, cached_hx(vals), lb, ub; log_level, indent)
    end
    if !isnothing(mod.mod_nl_ineq_constraints)
        @logmsg log_level "$(indent_str(indent))(Ineq. Constraints)"
        @ignoraise update!(
            mod.mod_nl_ineq_constraints, mod.nl_ineq_constraints, Δ, x, cached_gx(vals), lb, ub; log_level, indent)
    end
    return nothing
end

function process_trial_point!(mod::SimpleMOPSurrogate, vals_trial, isnext)
    xtrial = cached_x(vals_trial)
    #isnext = _trial_point_accepted(iteration_status)
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