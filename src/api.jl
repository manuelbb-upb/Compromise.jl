#=
# Problem Structure

## Outline

Internally, the evaluation of objective and constraint functions uses a 
mechanism that is quite involved.
The optimization problem as a whole is refered to as a `Problem` (similar to 
Models in `JuMP`).
A problem provides an abstraction to an underlying directed acyclic graph (DAG)
describing relations between problem functions.
In contrast to `MathOptInterface`, for example, all functions in the graph
are exactly that, `Function`s, not expressions.
The nodes in the graph map their inputs and outputs to (internal) variables.
We do all this to enable the surrogatizating of sub-components of the DAG.

=#

#=
## Indices
Inspired by MathOptInterface, we index variables and functions with objects 
of custom index types.
They are meant to enable querying and evaluation of a `Problem`.
=#

# All index types inherit from `AbstractIndex`.
abstract type AbstractIndex end
# Any index should be broadcasted by reference:
Base.broadcastable(i::AbstractIndex) = Ref(i)

# We proceed to build a hierachy of index types.
# First, `ScalarIndex` is the index type for any internally used scalar
# variable, i.e., primal problem variables, internal variables and scaled 
# variables.

struct ScalarIndex <: AbstractIndex
    value::Int
end

#=
# ## Inner Evaluators
As stated above, objectives and constraints are assumed to be Julia functions.
One advantage of this approach is that we can possibly use the same kind of functions for 
surrogate models as well.
To have a unified calling API, we define an interface for `AbstractInnerEvaluator`s, 
corresponding to vector-vector functions on real numbers.
As our optimizer can be derivative-free, an inner evaluator needs some evaluation method.
But we also allow second or first degree Taylor models. Then, we need first and maybe even 
second order derivative information as well.
The user should be able to supply function handles for derivatives.
Alternatively, we want to delegate the task to an `AbstractDifferentiation` backend.
=#

abstract type AbstractInnerEvaluator end

# It is mandatory to implement the evaluation method at
# some vector `x::AbstractVector{<:Real}`, into some target array `y::AbstractVector`.
# In general, mandatory methods start with a leading underscore.
# Optional methods might as well, and everything without a leading underscore
# is a wrapper that is used in the algorithm and employs additional checks.
function _func_at_vec!(y, func::AbstractInnerEvaluator, x)
    error("Method not yet defined.")
end

## internal method. stricter typing etc.
func_at_vec!(y::Vec, func::AbstractInnerEvaluator, x::Vec)=_func_at_vec!(y, func, x) 

# From this method, some defaults are derived, which can of course be fine-tuned for
# custom evaluators to make them more efficient.

function _func_at_mat!(mat_of_ys, func::AbstractInnerEvaluator, mat_of_xs)
    for (y,x) in zip(eachcol(mat_of_ys), eachcol(mat_of_xs))
        func_at_vec!(y, func, x)
    end
    return nothing
end

function func_at_mat!(mat_of_ys::Mat, func::AbstractInnerEvaluator, mat_of_xs::Mat)
    @assert(
        size(mat_of_ys, 2) == size(mat_of_xs, 2),
        "Input and output matrix must have same number of columns."
    )
    _func_at_mat!(mat_of_ys, func, mat_of_xs)
end

## (optional, but recommended)
function _func_output_at_vec!(y, output_index, func::AbstractInnerEvaluator, x)
    y_ = similar(y)
    _func_at_vec!(y, func, x)
    y[output_index] .= y_[output_index]
    return nothing
end

function func_output_at_vec!(
    y::Vec, output_index::Integer, func::AbstractInnerEvaluator, x::Vec)
    @assert 1 <= output_index <= length(y) "Invalid output index for target array."
    return _func_output_at_vec!(y, output_index, func, x)
end

# ### Differentiation
## (optional)
_provides_first_order_derivatives(::AbstractInnerEvaluator)=false
_provides_second_order_derivatives(::AbstractInnerEvaluator)=false

# If `func::AbstractInnerEvaluator` provides first order derivatives it
# has to implement `_jacobian_at_vec!`.

_jacobian_at_vec!(jac, func::AbstractInnerEvaluator, x)=error("Method not yet implemented.")
function jacobian_at_vec!(jac::Mat, func::AbstractInnerEvaluator, x::Vec)
    @assert(
        size(jac, 2)==length(x),
        "Number of columns in Jacobian must match length of input vector."
    )
    return _jacobian_at_vec!(jac, func, x)
end

# TODO                                                                                  #src
# make conversion between jacobian and gradient more versatile                          #src

## (optional)
function _gradient_of_output_at_vec!(g, output_index, func::AbstractInnerEvaluator, x)
    jac = zeros(eltype(g), length(g), length(x))
    _jacobian_at_vec!(jac, func, x)
    g .= jac[output_index, :]
    return nothing
end

function gradient_of_output_at_vec!(
    g::Vec, output_index::Integer, func::AbstractInnerEvaluator, x::Vec)
    return _gradient_of_output_at_vec!(g, output_index, func, x)
end

# Additionally, it might prove beneficial to evaluate the function and take
# its derivatives at the same time:
## (optional)
function _func_and_jacobian_at_vec!(y, jac, func::AbstractInnerEvaluator, x)
    _func_at_vec!(y, func, x)
    _jacobian_at_vec!(jac, func, y)
    return nothing
end

function func_and_jacobian_at_vec!(y::Vec, jac::Mat, func::AbstractInnerEvaluator, x::Vec)
    @assert(
        size(jac) == (length(y), length(x)),
        "Dimension mismatch between Jacobian and input and output vectors."
    )
    return _func_and_jacobian_at_vec!(y, jac, func, x)    
end

# If `func::AbstractInnerEvaluator` provides second order derivatives
# it has to implement `hessian_of_output_at_vec!`.

function _hessian_of_output_at_vec!(hess, output_index, func::AbstractInnerEvaluator, x)
    return nothing
end

function hessian_of_output_at_vec!(
    hess::Mat, output_index::Integer, func::AbstractInnerEvaluator, x::Vec)
    n = length(x)
    @assert size(hess) == (n, n) "Dimension of Hessian does not match length of input."
    _hessian_of_output_at_vec!(hess, output_index, func, x)
end

## (optional)
function _func_and_gradient_and_hessian_of_output_at_vec!(
    y, g, hess, output_index, func::AbstractInnerEvaluator, x)
    _func_output_at_vec!(y, output_index, func, x)
    _gradient_of_output_at_vec!(g, output_index, func, x)
    _hessian_of_output_at_vec!(hess, output_index, func, x)
    return nothing
end

function _func_and_gradient_and_hessian_of_output_at_vec!(
    y::Vec, g::Vec, hess::Mat, output_index::Integer, func::AbstractInnerEvaluator, x::Vec)
    N = length(x)
    K = length(y)
    @assert length(g) == N "Length of gradient array and input must match."
    @assert size(hess) == (N, N) "Dimensions of Hessian and input must match."
    @assert 1 <= output_index <= K "Invalid output index for target array."

    return _func_and_gradient_and_hessian_of_output_at_vec!(y, g, hess, output_index, func, x)
end

# ## Outer Evaluators
# A glorified dictionary with metadata:

struct SourceIndices{S,XI,YI}
    source::S
    xind::XI
    yind::YI
end

struct OuterEvaluator{
    IndIn,
    AIE<:AbstractInnerEvaluator,
    X<:AbstractVector{<:Real},
    Y<:AbstractVector{<:Real},
    J<:Union{Nothing,AbstractMatrix{<:Real}},
    H<:Union{Nothing,AbstractMatrix{<:Real}}
    IP,
}
    func::AIE

    num_evals::Base.RefValue{Int}
    num_inputs::Int
    num_outputs::Int

    ## an iterable of either integer indices or index sets or tuples 
    ## of source OuterEvaluators and their output indices
    input_indices::IndIn
    
    ## caches
    x::X
    y::Y
    jac::J
    hessians::H

    func_hash::Base.RefValue{UInt64}
    jac_hash::Base.RefValue{UInt64}
end

inner_func!(evaluator)=func_at_vec!(evaluator.y, evaluator.func, evaluator.y)

function set_x_with_indices!(x_cache, ind, x, x_hash)
    x_cache[ind] .= x[ind]
    return nothing
end

function set_x_with_indices!(x_cache, ind::SourceIndices, x, x_hash)
    source = ind.source
    func_at_vec!(source, x, x_hash)
    x_cache[ind.xind] .= source.y[ind.yind]
    return nothing
end

function _func_at_vec!(evaluator::OuterEvaluator, x, x_hash)
    for ind in evaluator.input_indices
        set_x_with_indices!(evaluator.x, ind, x, x_hash)
    end
    inner_func!(evaluator)
    return nothing
end

function func_at_vec!(evaluator::OuterEvaluator, x, x_hash)
    if evaluator.func_hash[] == x_hash
        return nothing
    end
    _func_at_vec!(evaluator, x, x_hash)
    evaluator.func_hash[] = x_hash
    return nothing
end

function func_at_vec!(evaluator::OuterEvaluator, x)
    return func_at_vec!(evaluator, x, hash(x))
end

function _jacobian_at_vec!(evaluator::OuterEvaluator, x, x_hash)
    for ind in evaluator.input_indices
        set_x_with_indices!(evaluator.x, ind, x, x_hash)
    end
end