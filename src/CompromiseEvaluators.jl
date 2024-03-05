#src This file is meant to be parsed by Literate.jl
# # Module `CompromiseEvaluators`
# This file provides a submodule defining abstract types and interfaces for evaluation
# of vector-vector-functions and surrogate models.

module CompromiseEvaluators #src

import Logging: Debug

# ## `AbstractNonlinearOperator` Interface
#=
An object subtyping `AbstractNonlinearOperator` represents a function mapping real-valued
vectors to real-valued vectors.
The interface defines methods to evaluate such a function.
These methods are used internally by Compromise, and we made the decision to assume
in-place functions.
If the user has out-of-place functions, they have to transform them accordingly.
Alternatively, this functionality can be provided by utility types implementing the interface.
=#

abstract type AbstractNonlinearOperator end

# A function can have parameters that are constant in a single optimization run.
# Previously, there was a type hierachy like
#=
```julia
abstract type AbstractNonlinearOperatorWithParams <: AbstractNonlinearOperator end
abstract type AbstractNonlinearOperatorNoParams <: AbstractNonlinearOperator end
```
=#
# **This is no longer the case!**
# We now have a “trait”:
abstract type AbstractNonlinearOperatorTrait end

abstract type AbstractNonlinearOperatorManyColumnsTrait <: AbstractNonlinearOperatorTrait end
struct OperatorSequential <: AbstractNonlinearOperatorManyColumnsTrait end
struct OperatorParallel <: AbstractNonlinearOperatorManyColumnsTrait end
optrait_multi(op::AbstractNonlinearOperator) = OperatorSequential()

abstract type AbstractNonlinearOperatorHasParamsTrait <: AbstractNonlinearOperatorTrait end
struct IsParametricOperator <: AbstractNonlinearOperatorHasParamsTrait end
struct IsNonparametricOperator <: AbstractNonlinearOperatorHasParamsTrait end

function optrait_params(op::AbstractNonlinearOperator)::NonlinearOperatorHasParamsTrait
    return IsNonparametricOperator()
end

import ..Compromise: RVec, RVecOrMat, RMat, @ignoraise

# Evaluation of derivatives is optional if evaluation-based models are used.
# We have functions to indicate if an operator implements `eval_grads!`, `eval_hessians!`:
function provides_grads(op::AbstractNonlinearOperator)
    return false
end
function provides_hessians(op::AbstractNonlinearOperator)
    return false
end

# In certain situations (nonlinear subproblems relying on minimization of scalar-valued 
# objective or constraint compoments) it might be beneficial if only certain outputs
# of a vector-function could be evaluated.
# The method `supports_partial_evaluation` signals this feature.
# If it returns `true`, the feature is assumed to be available for derivatives as well.
# In this situation, the type should implment methods starting with `partial_`, see below
# for details.
abstract type NonlinearOperatorPartialEvaluationTrait <: AbstractNonlinearOperatorTrait end
struct OperatorSupportsPartialEvaluation <: NonlinearOperatorPartialEvaluationTrait end
struct OperatorOnlyFullEvaluation <: NonlinearOperatorPartialEvaluationTrait end
optrait_partial(op::AbstractNonlinearOperator)=OperatorOnlyFullEvaluation()

# The methods below should be implemented to evaluate parameter dependent operators: 
"""
    eval_op!(y, op, x, p)

Evaluate the operator `op` at variable vector `x` with parameters `p`
and mutate the target vector `y` to contain the result.
"""
function eval_op!(y::RVec, op::AbstractNonlinearOperator, x::RVec, p)
    return error("No implementation of `eval_op!` for operator $op.")
end

"""
    eval_op!(y, op, x)

Evaluate the parameterless operator `op` at variable vector `x`
and mutate the target vector `y` to contain the result.
"""
function eval_op!(y::RVec, op::AbstractNonlinearOperator, x::RVec)
    return error("No implementation of `eval_op!` for operator $op.")
end

"""
    eval_grads!(Dy, op, x, p)

Compute the gradients of the operator `op` at variable vector `x` with parameters `p` 
and mutate the target matrix `Dy` to contain the gradients w.r.t. `x` in its columns. 
That is, `Dy` is the transposed Jacobian at `x`.
"""
function eval_grads!(Dy, op::AbstractNonlinearOperator, x, p)
    return error("No implementation of `eval_grads!` for operator $op.")
end

"""
    eval_grads!(Dy, op, x)

Compute the gradients of the parameterless operator `op` at variable vector `x`
and mutate the target matrix `Dy` to contain the gradients w.r.t. `x` in its columns. 
That is, `Dy` is the transposed Jacobian at `x`.
"""
function eval_grads!(Dy, op::AbstractNonlinearOperator, x)
    return error("No implementation of `eval_grads!` for operator $op.")
end


# If Hessian matrices are needed, implement `eval_hessians!(H, op, x, p)`.
# Assume `H` to be a 3D array, where the last index iterates over the function outputs.
# That is, `H[:,:,1]` is a matrix containing second order partial derivatives of the first 
# output, `H[:,:,2]` has second order derivatives for the second output, and so forth...
# Moreover, in unlike with `eval_grads!`, the Hessian information should be stored in 
# correct order - `H[i,j,k]` should correspond to `∂yₖ/∂xᵢ∂xⱼ`.
# After `eval_grads!(D, op, x, p)`, the column `D[:,k]` contains partial derivatives 
# `∂₁yₖ, ∂₂yₖ, …, ∂ₘyₖ`, the gradient of `y[k]`.
# After `eval_hessians!(H, op, x, p)`, the “column” `H[:, j, k]` contains
# `∂₁(∂ⱼyₖ), ∂₂(∂ⱼyₖ), …, ∂ₘ(∂ⱼyₖ)`, the gradient of `Dy[j, k]`.

"""
    eval_hessians!(H, op, x, p)

Compute the Hessians of the operator `op` at variable vector `x` with parameters `p` 
and mutate the target array `H` to contain the Hessians along its last index.
That is, `H[:,:,i]` is the Hessian at `x` and `p` w.r.t. `x` of output `i`.
"""
function eval_hessians!(H, op::AbstractNonlinearOperator, x, p)
    return error("No implementation of `eval_hessians!` for operator $op.")
end

"""
    eval_hessians!(H, op, x, p)

Compute the Hessians of the parameterless operator `op` at variable vector `x`
and mutate the target array `H` to contain the Hessians along its last index.
That is, `H[:,:,i]` is the Hessian at `x` and `p` w.r.t. `x` of output `i`.
"""
function eval_hessians!(H, op::AbstractNonlinearOperator, x)
    return error("No implementation of `eval_hessians!` for operator $op.")
end

# Some operators might support partial evaluation. 
# They should implement these methods, if `supports_partial_evaluation` returns `true`.
# The argument `outputs` is an iterable of output indices, assuming `1` to be the first output.
# `y` is the full length vector, and `partial_op!` should set `y[outputs]`.
function eval_op!(y::RVec, op::AbstractNonlinearOperator, x::RVec, outputs::Vector{Bool})
    return error("Partial evaluation not implemented.")
end
function eval_op!(y::RVec, op::AbstractNonlinearOperator, x::RVec, params, outputs::Vector{Bool})
    return error("Partial evaluation not implemented.")
end

function eval_grads!(Dy, op::AbstractNonlinearOperator, x, outputs::Vector{Bool})
    return error("Partial Jacobian not implemented.")
end
function eval_grads!(Dy, op::AbstractNonlinearOperator, x, params, outputs::Vector{Bool})
    return error("Partial Jacobian not implemented.")
end

# Optional Parallel Evaluation:
function eval_op!(Y::RMat, op::AbstractNonlinearOperator, X::RMat)
    error("`eval_op!` not implemented for matrices.")
end
function eval_op!(Y::RMat, op::AbstractNonlinearOperator, X::RMat, params)
    error("`eval_op!` not implemented for matrices.")
end
function eval_op!(Y::RMat, op::AbstractNonlinearOperator, X::RMat, params, outputs::Vector{Bool})
    error("`eval_op!` not implemented for matrices.")
end

# The combined forward-function `eval_op_and_grads!` is derived from `eval_op!` and 
# `eval_grads!`, but can be customized easily:

function eval_op_and_grads!(y, Dy, op::AbstractNonlinearOperator, x, args...)
    @ignoraise eval_op!(y, op, x, args...)
    @ignoraise eval_grads!(Dy, op, x, args...)
    return nothing
end

function eval_op_and_grads_and_hessians!(y, Dy, H, op::AbstractNonlinearOperator, x, args...)
    @ignoraise eval_op_and_grads!(y, Dy, op, x, args...)
    @ignoraise eval_hessians!(H, op, x, args...)
    return nothing
end

# # Safe-Guarded Functions
function func_vals!(
    y::RVecOrMat, op::AbstractNonlinearOperator, x::RVecOrMat, params=nothing, outputs::Union{Nothing,Vector{Bool}}=nothing
)
    return _func_vals_multi!(y, op::AbstractNonlinearOperator, optrait_multi(op), x, params, outputs)
end

# If vectors are used, we can forward to `eval_op!` in both cases (OperatorParallel(), OperatorSequential):
function _func_vals_multi!(
    y::RVec, op::AbstractNonlinearOperator, ::Union{OperatorParallel, OperatorSequential}, x::RVec, params, outputs
)
    @ignoraise _func_vals!(y, op, optrait_params(op), optrait_partial(op), x, params, outputs)
    return nothing
end
# For matrices, we can forward only if `OperatorParallel()` trait is present:
function _func_vals_multi!(
    y::RMat, op::AbstractNonlinearOperator, ::OperatorParallel, x::RMat, params, outputs
)
    @ignoraise _func_vals!(y, op, optrait_params(op), optrait_partial(op), x, params, outputs)
    return nothing
end
# Otherwise, we have a simple fallback:
function _func_vals_multi!(
    Y::RMat, op::AbstractNonlinearOperator, ::OperatorSequential, X::RMat, params, outputs
)
    for (y, x) = zip(eachcol(Y), eachcol(X))
        @ignoraise _func_vals!(y, op, optrait_params(op), optrait_partial(op), x, params, outputs)
    end
end

function func_grads!(
    Dy, op::AbstractNonlinearOperator, x, params=nothing, outputs::Union{Nothing,Vector{Bool}}=nothing
)
    @ignoraise _func_grads!(Dy, op, optrait_params(op), optrait_partial(op), x, params, outputs)
    return nothing
end

function func_vals_and_grads!(
    y, Dy, op::AbstractNonlinearOperator, x, params=nothing, outputs::Union{Nothing,Vector{Bool}}=nothing
)
    @ignoraise _func_vals_and_grads!(y, Dy, op, optrait_params(op), optrait_partial(op), x, params, outputs)
    return nothing
end

function func_hessians!(
    H, op::AbstractNonlinearOperator, x, params=nothing, outputs::Union{Nothing,Vector{Bool}}=nothing
)
    @ignoraise _func_hessians!(H, op, optrait_params(op), optrait_partial(op), x, params, outputs)
    return nothing
end

function func_vals_and_grads_and_hessians!(
    y, Dy, H, op::AbstractNonlinearOperator, x, params=nothing, outputs::Union{Nothing,Vector{Bool}}=nothing
)
    @ignoraise _func_vals_and_grads_and_hessians!(y, Dy, H, op, optrait_params(op), optrait_partial(op), x, params, outputs)
    return nothing
end

for (new_name, inner_name, mod_args, in_arg) in 
    (
        (:_func_vals!, :eval_op!, (:y,), :x),
        (:_func_grads!, :eval_grads!, (:Dy,), :x),
        (:_func_hessians!, :eval_hessians!, (:Dy,), :x),
        (:_func_vals_and_grads!, :eval_op_and_grads!, (:y, :Dy,), :x),
        (:_func_vals_and_grads_and_hessians!, :eval_op_and_grads_and_hessians!, (:y, :Dy, :H), :x),
    )

    inner_args = Any[in_arg,]
    for (i, params_trait) in enumerate((:(::IsNonparametricOperator), :(::IsParametricOperator)))
        if i == 2
            push!(inner_args, :params)
        end
        for (j, partial_trait) in enumerate((:(::OperatorOnlyFullEvaluation), :(::OperatorSupportsPartialEvaluation)))
            _inner_args = copy(inner_args)
            if j == 2
                push!(_inner_args, :outputs)
            end

            @eval function $(new_name)(
                $(mod_args...), op::AbstractNonlinearOperator, $(params_trait), $(partial_trait), 
                x::RVecOrMat, params, outputs
            )
                $(inner_name)($(mod_args...), op, $(_inner_args...))
            end
        end
    end
end

# ## Types and Methods for Surrogate Models

# An `AbstractSurrogateModel` is similar to `AbstractNonlinearOperator`.
# Such a surrogate model is always non-parametric, as parameters of operators are assumed 
# to be fix in-between optimization runs.
abstract type AbstractSurrogateModel <: AbstractNonlinearOperator end
optrait_params(::AbstractSurrogateModel)=IsNonparametricOperator()
optrait_partial(::AbstractSurrogateModel)=OperatorOnlyFullEvaluation()
optrait_multi(::AbstractSurrogateModel)=OperatorSequential()

provides_grads(::AbstractSurrogateModel)=true

# We also want to be able to define the behavior of models with light-weight objects:
abstract type AbstractSurrogateModelConfig end
num_parallel_evals(::AbstractSurrogateModelConfig)::Integer=1

# ### Indicator Methods
# Functions to indicate the order of the surrogate model:
requires_grads(::AbstractSurrogateModelConfig)=false
requires_hessians(::AbstractSurrogateModelConfig)=false

# A function to indicate that a model should be updated when the trust region has changed:
depends_on_radius(::AbstractSurrogateModel)=true

# ### Initialization and Modification

# The choice to don't separate between a model and its parameters (like `Lux.jl` does) 
# is historic.
# There are pros and cons to both approaches.
# The most obvious point in favor of how it is now are the unified evaluation interfaces.
# However, for the Criticality Routine we might need to copy models and retrain them for 
# smaller trust-region radii.
# That is why we require implementation of `copy_model(source_model)` and 
# `copyto_model!(trgt_model, src_model)`.
# A modeller should take care to really only copy parameter arrays and pass other large
# objects, such as databases, by reference so as to avoid a large memory-overhead.
# Moreover, we only need copies for radius-dependent models!
# You can ignore those methods otherwise.

# A surrogate is initialized from its configuration and the operator it is meant to model:
"""
    init_surrogate(
        model_config, nonlin_op, dim_in, dim_out, params, T
    )

Return a model subtyping `AbstractSurrogateModel`, as defined by 
`model_config::AbstractSurrogateModelConfig`, for the nonlinear operator `nonlin_op`.
The operator (and model) has input dimension `dim_in` and output dimension `dim_out`.
`params` is the current parameter object for `nonlin_op` and is cached.
`T` is a subtype of `AbstractFloat` to indicate precision of cache arrays.
"""
function init_surrogate(
    ::AbstractSurrogateModelConfig, op, dim_in, dim_out, params, T;
    require_fully_linear::Bool=true, 
    delta_max::Union{Number, AbstractVector{<:Number}}=Inf,
)::AbstractSurrogateModel
    return nothing
end

# A function to return a copy of a model. Should be implemented if 
# `depends_on_radius` returns `true`.
# Note, that the returned object does not have to be an “independent” copy, we allow 
# for shared objects (like mutable database arrays or something of that sort)...
copy_model(mod_src)=deepcopy(mod_src)

# A function to copy parameters between source and target models, like `Base.copy!` or 
# `Base.copyto!`. Relevant mostly for trainable parameters.
copyto_model!(mod_trgt::AbstractSurrogateModel, mod_src::AbstractSurrogateModel)=mod_trgt

function _copy_model(mod)
    depends_on_radius(mod) && return copy_model(mod)
    return mod
end

function _copyto_model!(mod_trgt, mod_src)
    depends_on_radius(mod_trgt) && return copyto_model!(mod_trgt, mod_src)
    return mod_trgt
end

# Because parameters are implicit, updates are in-place operations:
"""
    update!(surrogate_model, nonlinear_operator, Δ, x, fx, lb, ub)

Update the model on a trust region of size `Δ` in a box with lower left corner `lb`
and upper right corner `ub` (in the scaled variable domain)
`x` is a sub-vector of the current iterate conforming to the inputs of `nonlinear_operator`
in the scaled domain. `fx` are the outputs of `nonlinear_operator` at `x`.
"""
function update!(
    surr::AbstractSurrogateModel, op, Δ, x, fx, lb, ub; log_level, kwargs...
)
    return nothing    
end

# ### Evaluation
# Mandatory:
#=
```julia
eval_op!(y::RVec, surr::AbstractSurrogateModel, x::RVec)=nothing
eval_grads!(Dy, surr::AbstractSurrogateModel, x)=nothing
eval_op_and_grads!(y, Dy, surr::AbstractSurrogateModel, x)
```
=#
# ## Module Exports
export AbstractNonlinearOperator 
export AbstractSurrogateModel, AbstractSurrogateModelConfig
export supports_partial_evaluation, provides_grads, provides_hessians, requires_grads, requires_hessians
export func_vals!, func_grads!, func_hessians!, func_vals_and_grads!, func_vals_and_grads_and_hessians!
export eval_op!, eval_grads!, eval_hessians!, eval_op_and_grads!, eval_op_and_grads_and_hessians!
export func_vals!, func_grads!, func_vals_and_grads!, func_vals_and_grads_and_hessians!
export init_surrogate, update!

end#module