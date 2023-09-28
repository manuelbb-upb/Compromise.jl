# Module `CompromiseEvaluators`
This file provides a submodule defining abstract types and interfaces for evaluation
of vector-vector-functions and surrogate models.

## `AbstractNonlinearOperator` Interface
An object subtyping `AbstractNonlinearOperator` represents a function mapping real-valued
vectors to real-valued vectors.
The interface defines methods to evaluate such a function.
These methods are used internally by Compromise, and we made the decision to assume
in-place functions.
If the user has out-of-place functions, they have to transform them accordingly.
Alternatively, this functionality can be provided by utility types implementing the interface.

````julia
abstract type AbstractNonlinearOperator end
````

A function can have parameters that are constant in a single optimization run, and
the type structure reflects this distinction:

````julia
abstract type AbstractNonlinearOperatorWithParams <: AbstractNonlinearOperator end
abstract type AbstractNonlinearOperatorNoParams <: AbstractNonlinearOperator end
````

### Common Methods
Both, `AbstractNonlinearOperatorWithParams` and `AbstractNonlinearOperatorNoParams`
have methods like `eval_op!`.
The signatures look different, though.
That is why there is a separate section for both types.
The below methods have the same signature for both operator supertypes:

Evaluation of derivatives is optional if evaluation-based models are used.
We have functions to indicate if an operator implements `eval_grads!`, `eval_hessians!`:

````julia
provides_grads(op::AbstractNonlinearOperator)=false
provides_hessians(op::AbstractNonlinearOperator)=false
````

In certain situations (nonlinear subproblems relying on minimization of scalar-valued
objective or constraint compoments) it might be beneficial if only certain outputs
of a vector-function could be evaluated.
The method `supports_partial_evaluation` signals this feature.
If it returns `true`, the feature is assumed to be available for derivatives as well.
In this situation, the type should implment methods starting with `partial_`, see below
for details.

````julia
supports_partial_evaluation(op::AbstractNonlinearOperator) = false
````

### Methods for `AbstractNonlinearOperatorWithParams`
The methods below should be implemented to evaluate parameter dependent operators:

````julia
"""
    eval_op!(y, op, x, p)

Evaluate the operator `op` at variable vector `x` with parameters `p`
and mutate the target vector `y` to contain the result.
"""
function eval_op!(y, op::AbstractNonlinearOperatorWithParams, x, p)
    return error("No implementation of `eval_op!` for operator $op.")
end

"""
    eval_grads!(Dy, op, x, p)

Compute the gradients of the operator `op` at variable vector `x` with parameters `p`
and mutate the target matrix `Dy` to contain the gradients w.r.t. `x` in its columns.
That is, `Dy` is the transposed Jacobian at `x`.
"""
function eval_grads!(Dy, op::AbstractNonlinearOperatorWithParams, x, p)
    return error("No implementation of `eval_grads!` for operator $op.")
end
````

The combined forward-function `eval_op_and_grads!` is derived from `eval_op!` and
`eval_grads!`, but can be customized easily:

````julia
function eval_op_and_grads!(y, Dy, op::AbstractNonlinearOperatorWithParams, x, p)
    eval_op!(y, val, x, p)
    eval_grads!(Dy, val, x, p)
    return nothing
end
````

If Hessian matrices are needed, implement `eval_hessians!(H, op, x, p)`.
Assume `H` to be a 3D array, where the last index iterates over the function outputs.
That is, `H[:,:,1]` is a matrix containing second order partial derivatives of the first
output, `H[:,:,2]` has second order derivatives for the second output, and so forth...
Moreover, in unlike with `eval_grads!`, the Hessian information should be stored in
correct order - `H[i,j,k]` should correspond to `∂yₖ/∂xᵢ∂xⱼ`.
After `eval_grads!(D, op, x, p)`, the column `D[:,k]` contains partial derivatives
`∂₁yₖ, ∂₂yₖ, …, ∂ₘyₖ`, the gradient of `y[k]`.
After `eval_hessians!(H, op, x, p)`, the “column” `H[:, j, k]` contains
`∂₁(∂ⱼyₖ), ∂₂(∂ⱼyₖ), …, ∂ₘ(∂ⱼyₖ)`, the gradient of `Dy[j, k]`.

````julia
"""
    eval_hessians!(H, op, x, p)

Compute the Hessians of the operator `op` at variable vector `x` with parameters `p`
and mutate the target array `H` to contain the Hessians along its last index.
That is, `H[:,:,i]` is the Hessian at `x` and `p` w.r.t. `x` of output `i`.
"""
function eval_hessians!(H, op::AbstractNonlinearOperatorWithParams, x, p)
    return error("No implementation of `eval_hessians!` for operator $op.")
end
````

The combined forward-function `eval_op_and_grads_and_hessians!`
is derived from `eval_op_and_grads!` and `eval_hessians!`,
but can be customized easily:

````julia
function eval_op_and_grads!(y, Dy, H, op::AbstractNonlinearOperatorWithParams, x, p)
    eval_op_and_grads!(Dy, y, op, x, p)
    eval_hessians!(H, val, x, p)
    return nothing
end
````

Some operators might support partial evaluation.
They should implement these methods, if `supports_partial_evaluation` returns `true`.
The argument `outputs` is an iterable of output indices, assuming `1` to be the first output.
`y` is the full length vector, and `partial_op!` should set `y[outputs]`.

````julia
function partial_op!(y, op::AbstractNonlinearOperatorWithParams, x, p, outputs)
    # length(y)==length(outputs)
    return error("Partial evaluation not implemented.")
end
function partial_grads!(Dy, op::AbstractNonlinearOperatorWithParams, x, p, outputs)
    # size(Dy)==(length(x), length(outputs))
    return error("Partial Jacobian not implemented.")
end
function partial_op_and_grads!(y, Dy, op::AbstractNonlinearOperatorWithParams, x, p, outputs)
    partial_op!(y, op, x, p, outputs)
    partial_grads!(Dy, op, x, p, outputs)
    return nothing
end
function partial_hessians!(H, op::AbstractNonlinearOperatorWithParams, x, p, outputs)
    return error("Partial Hessians not implemented.")
end
function partial_op_and_grads_and_hessians!(y, Dy, H, op::AbstractNonlinearOperatorWithParams, x, p, outputs)
    partial_op_and_grads!(y, Dy, op, x, p, outputs)
    partial_hessians!(H, op, x, p, outputs)
    return nothing
end
````

From the above, we derive safe-guarded functions, that can be used to pass `outputs`
whenever convenient.
Note, that these are defined for `AbstractNonlinearOperator`.
By implementing the parametric-interface for `AbstractNonlinearOperatorNoParams`,
they work out-of-the box for non-paremetric operators, too:

````julia
function func_vals!(y, op::AbstractNonlinearOperator, x, p, outputs=nothing)
    if !isnothing(outputs) && supports_partial_evaluation(op)
        partial_op!(y, op, x, p, outputs)
    end
    return eval_op!(y, op, x, p)
end
function func_grads!(Dy, op::AbstractNonlinearOperator, x, p, outputs=nothing)
    if !isnothing(outputs) && supports_partial_evaluation(op)
        partial_grads!(Dy, op, x, p, outputs)
    end
    return eval_grads!(Dy, op, x, p)
end
function func_vals_and_grads!(y, Dy, op::AbstractNonlinearOperator, x, p, outputs=nothing)
    if !isnothing(outputs) && supports_partial_evaluation(op)
        partial_op_and_grads!(y, Dy, op, x, p, outputs)
    end
    return eval_op_and_grads!(y, Dy, op, x, p)
end
function func_hessians!(H, op::AbstractNonlinearOperator, x, p, outputs=nothing)
    if !isnothing(outputs) && supports_partial_evaluation(op)
        partial_hessians!(H, op, x, p, outputs)
    end
    return eval_hessians!(H, op, x, p)
end
function func_vals_and_grads_and_hessians!(
    y, Dy, H, op::AbstractNonlinearOperator, x, p, outputs=nothing)
    if !isnothing(outputs) && supports_partial_evaluation(op)
        partial_op_and_grads_and_hessians!(y, Dy, H, op, x, p, outputs)
    end
    eval_op_and_grads_and_hessians!(y, Dy, H, op, x, p)
end
````

### Methods `AbstractNonlinearOperatorNoParams`

The interface for operators without parameters is very similar to what's above.
In fact, we could always treat it as a special case of `AbstractNonlinearOperatorWithParams`
and simply ignore the parameter vector.
However, in some situation it might be desirable to have the methods without `p`
readily at hand.
This also makes writing extensions a tiny bit easier.

````julia
function eval_op!(y, op::AbstractNonlinearOperatorNoParams, x)
    return error("No implementation of `eval_op!` for operator $op.")
end
function eval_grads!(Dy, op::AbstractNonlinearOperatorNoParams, x)
    return error("No implementation of `eval_grads!` for operator $op.")
end
````

Optional, derived method for values and gradients:

````julia
function eval_op_and_grads!(y, Dy, op::AbstractNonlinearOperatorNoParams, x)
    eval_op!(y, op, x)
    eval_grads!(Dy, op, x)
    return nothing
end
````

Same for Hessians:

````julia
function eval_hessians!(H, op::AbstractNonlinearOperatorNoParams, x)
    return error("No implementation of `eval_hessians!` for operator $op.")
end
function eval_op_and_grads_and_hessians!(y, Dy, H, op::AbstractNonlinearOperatorNoParams, x)
    eval_op_and_grads!(y, Dy, op, x)
    eval_hessians!(H, op, x)
    return nothing
end
````

Some operators might support partial evaluation.
They should implement these methods, if `supports_partial_evaluation` returns `true`:

````julia
function partial_op!(y, op::AbstractNonlinearOperatorNoParams, x, outputs)
    return error("Partial evaluation not implemented.")
end
function partial_grads!(Dy, op::AbstractNonlinearOperatorNoParams, x, outputs)
    return error("Partial Jacobian not implemented.")
end
function partial_op_and_grads!(y, Dy, op::AbstractNonlinearOperatorNoParams, x, outputs)
    partial_op!(y, op, x, outputs)
    partial_grads!(Dy, op, x, outputs)
    return nothing
end
function partial_hessians!(H, op::AbstractNonlinearOperatorNoParams, x, outputs)
    return error("Partial Hessians not implemented.")
end
function partial_op_and_grads_and_hessians!(y, Dy, H, op::AbstractNonlinearOperatorNoParams, x, outputs)
    partial_op_and_grads!(y, Dy, op, x, outputs)
    partial_hessians!(H, op, x, outputs)
    return nothing
end
````

The safe-guarded methods simply forward to the parametric versions and pass `nothing`
parameters:

````julia
function func_vals!(y, op::AbstractNonlinearOperatorNoParams, x, outputs=nothing)
    return func_vals!(y, op, x, nothing, outputs)
end
function func_grads!(Dy, op::AbstractNonlinearOperatorNoParams, x, outputs=nothing)
    return func_grads!(Dy, op, x, nothing, outputs)
end
function func_vals_and_grads!(y, Dy, op::AbstractNonlinearOperatorNoParams, x, outputs=nothing)
    return func_vals_and_grads!(y, Dy, op, x, nothing, outputs)
end
function func_hessians!(H, op::AbstractNonlinearOperatorNoParams, x, outputs=nothing)
    return func_hessians!(H, op, x, nothing, outputs)
end
function func_vals_and_grads_and_hessians!(
    y, Dy, H, op::AbstractNonlinearOperatorNoParams, x, outputs=nothing
)
    return func_vals_and_grads_and_hessians!(y, Dy, H, op, x, nothing, outputs)
end
````

#### Parameter-Methods for Non-Parametric Operators
To also be able to use non-parametric operators in the more general setting,
implement the parametric-interface:

````julia
function eval_op!(y, op::AbstractNonlinearOperatorNoParams, x, p)
    return eval_op!(y, op, x)
end
function eval_grads!(Dy, op::AbstractNonlinearOperatorNoParams, x, p)
    return eval_grads!(Dy, op, x)
end
function eval_op_and_grads!(y, Dy, op::AbstractNonlinearOperatorNoParams, x, p)
    return eval_op_and_grads!(y, Dy, op, x)
end
function eval_hessians!(H, op::AbstractNonlinearOperatorNoParams, x, p)
    return eval_hessians!(H, op, x)
end
function eval_op_and_grads_and_hessians!(y, Dy, H, op::AbstractNonlinearOperatorNoParams, x, p)
    return eval_op_and_grads_and_hessians!(y, Dy, H, op, x)
end
````

Partial evaluation or differentiation:

````julia
function partial_op!(y, op::AbstractNonlinearOperatorNoParams, x, p, outputs)
    return partial_op!(y, op, x, outputs)
end
function partial_grads!(Dy, op::AbstractNonlinearOperatorNoParams, x, p, outputs)
    return partial_grads!(Dy, op, x, outputs)
end
function partial_op_and_grads!(y, Dy, op::AbstractNonlinearOperatorNoParams, x, p, outputs)
    return partial_op_and_grads!(y, Dy, op, x, outputs)
end
function partial_hessians!(H, op::AbstractNonlinearOperatorNoParams, x, p, outputs)
    return partial_hessians!(H, op, x, outputs)
end
function partial_op_and_grads_and_hessians!(
    y, Dy, H, op::AbstractNonlinearOperatorNoParams, x, p, outputs
)
    return partial_op_and_grads_and_hessians!(y, Dy, H, op, x, outputs)
end
````

#### Non-Parametric Methods for Parametric Operators
These should only be used when you know what you are doing, as we set the parameters
to `nothing`.
This is safe only if we now that an underlying function is not-parametric, but somehow
wrapped in something implementing `AbstractNonlinearOperatorWithParams`.

````julia
function eval_op!(y, op::AbstractNonlinearOperatorWithParams, x)
    return eval_op!(y, op, x, nothing)
end
function eval_grads!(Dy, op::AbstractNonlinearOperatorWithParams, x)
    return eval_grads!(Dy, op, x, nothing)
end

function eval_op_and_grads!(y, Dy, op::AbstractNonlinearOperatorWithParams, x)
    return eval_op_and_grads!(y, Dy, op, x, nothing)
end

function eval_hessians!(H, op::AbstractNonlinearOperatorWithParams, x)
    return eval_hessians!(H, op, x, nothing)
end

function eval_op_and_grads_and_hessians!(y, Dy, H, op::AbstractNonlinearOperatorWithParams, x)
    return eval_op_and_grads_and_hessians!(y, Dy, H, op, x, nothing)
end

function partial_op!(y, op::AbstractNonlinearOperatorWithParams, x, outputs)
    return partial_op!(y, op, x, nothing, outputs)
end
function partial_grads!(Dy, op::AbstractNonlinearOperatorWithParams, x, outputs)
    return partial_grads!(Dy, op, x, nothing, outputs)
end
function partial_op_and_grads!(y, Dy, op::AbstractNonlinearOperatorWithParams, x, outputs)
    return partial_op_and_grads!(y, Dy, op, x, nothing, outputs)
end
function partial_hessians!(H, op::AbstractNonlinearOperatorWithParams, x, outputs)
    return partial_hessians!(H, op, x, nothing, outputs)
end
function partial_op_and_grads_and_hessians!(
    y, Dy, H, op::AbstractNonlinearOperatorWithParams, x, outputs
)
    return partial_op_and_grads_and_hessians!(y, Dy, H, op, x, nothing, outputs)
end
````

## Types and Methods for Surrogate Models

An `AbstractSurrogateModel` is similar to `AbstractNonlinearOperator`.
Such a surrogate model is always non-parametric, as parameters of operators are assumed
to be fix in between runs.

````julia
abstract type AbstractSurrogateModel end
````

We also want to be able to define the behavior of models with light-weight objects:

````julia
abstract type AbstractSurrogateModelConfig end
````

### Indicator Methods
Functions to indicate the order of the surrogate model:

````julia
requires_grads(::AbstractSurrogateModelConfig)=false
requires_hessians(::AbstractSurrogateModelConfig)=false
````

A function to indicate that a model should be updated when the trust region has changed:

````julia
depends_on_radius(::AbstractSurrogateModel)=true
````

### Initialization and Modification

The choice to don't separate between a model and its parameters (like `Lux.jl` does)
is historic.
There are pros and cons to both approaches.
The most obvious point in favor of how it is now are the unified evaluation interfaces.
However, for the Criticality Routine we might need to copy models and retrain them for
smaller trust-region radii.
That is why we require implementation of `copy_model(source_model)` and
`copyto_model!(trgt_model, src_model)`.
A modeller should take care to really only copy parameter arrays and pass other large
objects, such as databases, by reference so as to avoid a large memory-overhead.
Moreover, we only need copies for radius-dependent models!
You can ignore those methods otherwise.

A surrogate is initialized from its configuration and the operator it is meant to model:

````julia
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
    ::AbstractSurrogateModelConfig, op, dim_in, dim_out, params, T)::AbstractSurrogateModel
    return nothing
end
````

A function to return a copy of a model. Should be implemented if
`depends_on_radius` returns `true`.
Note, that the returned object does not have to be an “independent” copy, we allow
for shared objects (like mutable database arrays or something of that sort)...

````julia
copy_model(mod_src)=deepcopy(mod_src)
````

A function to copy parameters between source and target models, like `Base.copy!` or
`Base.copyto!`. Relevant mostly for trainable parameters.

````julia
copyto_model!(mod_trgt::AbstractSurrogateModel, mod_src::AbstractSurrogateModel)=mod_trgt

function _copy_model(mod)
    depends_on_radius(mod) && return copy_model(mod)
    return mod
end

function _copyto_model!(mod_trgt, mod_src)
    depends_on_radius(mod_trgt) && return copyto_model!(mod_trgt, mod_src)
    return mod_trgt
end
````

Because parameters are implicit, updates are in-place operations:

````julia
"""
    update!(surrogate_model, nonlinear_operator, Δ, x, fx, lb, ub)

Update the model on a trust region of size `Δ` in a box with lower left corner `lb`
and upper right corner `ub` (in the scaled variable domain)
`x` is a sub-vector of the current iterate conforming to the inputs of `nonlinear_operator`
in the scaled domain. `fx` are the outputs of `nonlinear_operator` at `x`.
"""
function update!(
    surr::AbstractSurrogateModel, op, Δ, x, fx, lb, ub; kwargs...
)
    return nothing
end
````

### Evaluation
In place evaluation and differentiation, similar to `AbstractNonlinearOperatorNoParams`.
Mandatory:

````julia
function model_op!(y, surr::AbstractSurrogateModel, x)
    return nothing
end
````

Mandatory:

````julia
function model_grads!(Dy, surr::AbstractSurrogateModel, x)
    return nothing
end
````

Optional:

````julia
function model_op_and_grads!(y, Dy, surr::AbstractSurrogateModel, x)
    model_op!(y, surr, x )
    model_grads!(Dy, surr, x)
    return nothing
end
````

#### Optional Partial Evaluation

````julia
supports_partial_evaluation(::AbstractSurrogateModel)=false
function model_op!(y, surr::AbstractSurrogateModel, x, outputs)
    return error("Surrogate model does not support partial evaluation.")
end
function model_grads!(y, surr::AbstractSurrogateModel, x, outputs)
    return error("Surrogate model does not support partial Jacobian.")
end
function model_op_and_grads!(y, Dy, surr::AbstractSurrogateModel, x, outputs)
    model_op!(y, surr, x, outputs)
    model_grads!(y, surr, x, outputs)
    return nothing
end
````

#### Safe-guarded, internal Methods
The methods below are used in the algorithm and have the same signature as
the corresponding methods for `AbstractNonlinearOperator`.
Thus, we do not have to distinguish types in practice.

````julia
function func_vals!(y, surr::AbstractSurrogateModel, x, p, outputs=nothing)
    if !isnothing(outputs)
        if supports_partial_evaluation(surr)
            return model_op!(y, surr, x, outputs)
        # else
        #     @warn "Partial evaluation not supported by surrogate model."
        end
    end
    return model_op!(y, surr, x)
end
function func_grads!(Dy, surr::AbstractSurrogateModel, x, p, outputs=nothing)
    if !isnothing(outputs)
        if supports_partial_evaluation(surr)
            return model_grads!(Dy, surr, x, outputs)
        # else
        #     @warn "Partial evaluation not supported by surrogate model."
        end
    end
    return model_grads!(Dy, surr, x)
end
function func_vals_and_grads!(y, Dy, surr::AbstractSurrogateModel, x, p, outputs=nothing)
    if !isnothing(outputs)
        if supports_partial_evaluation(surr)
            return model_op_and_grads!(y, Dy, surr, x, outputs)
        # else
        #     @warn "Partial evaluation not supported by surrogate model."
        end
    end
    return model_op_and_grads!(y, Dy, surr, x)
end
````

## Module Exports

````julia
export AbstractNonlinearOperator, AbstractNonlinearOperatorNoParams, AbstractNonlinearOperatorWithParams
export AbstractSurrogateModel, AbstractSurrogateModelConfig
export supports_partial_evaluation, provides_grads, provides_hessians, requires_grads, requires_hessians
export func_vals!, func_grads!, func_hessians!, func_vals_and_grads!, func_vals_and_grads_and_hessians!
export eval_op!, eval_grads!, eval_hessians!, eval_op_and_grads!, eval_op_and_grads_and_hessians!
export model_op!, model_grads!, model_op_and_grads!
export init_surrogate, update!

end#module
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

