module NonlinearFunctions

include("abstract_autodiff.jl")

import ..Compromise: RVec, RMat, RVecOrMat, @ignoraise
import ..Compromise.CompromiseEvaluators as CE
import ..Compromise.CompromiseEvaluators: FuncCallCounter

using Parameters: @with_kw

# ## Wrapper for Parametric Functions
@doc """
    NonlinearParametricFunction(; 
        func, grads=nothing, hessians=nothing, 
        func_and_grads=nothing, func_and_grads_and_hessians=nothing,
        backend=nothing,
        func_iip=true, grads_iip=true, hessians_iip=true, 
        func_and_grads_iip=true, func_and_grads_and_hessians_iip=true)

A flexible function wrapper to conveniently query evaluations and derivatives of user
provided functions.

If the user provided function is used in a derivative-free alogrithm, only `func` has 
to be provided. The flag `func_iip` indicates its signature:
If `func_iip==true`, the function should mutate the target array and 
have signature `func!(y, x, p)`. Otherwise, `y = func(x, p)`.

Should gradients be needed, function handles can be provided, and the respective 
flags indicate the following signatures:
* `grads_iip == true` implies `grads!(Dy, x, p)`, otherwise `Dy = grads(x, p)`.
* `hessians_iip == true` implies `hessians!(H, x, p)`, otherwise `H = hessians(x, p)`.
* `func_and_grads_iip == true` implies `func_and_grads!(y, Dy, x, p)`,  
   otherwise `y, Dy = func_and_grads(x, p)`.
* `func_and_grads_and_hessians_iip == true` implies  
  `func_and_grads_and_hessians!(y, Dy, H, x, p)`,  
   else `y, Dy, H = func_and_grads_and_hessians(x, p)`.

Alternatively (or additionally), an `AbstractAutoDiffBackend` can be passed to 
compute the derivatives if the relevant field `isnothing`.
"""
@with_kw struct NonlinearParametricFunction{
    F, G, H, FG, FGH, B
} <: CE.AbstractNonlinearOperator
    func :: F
    grads :: G = nothing
    hessians :: H = nothing
    func_and_grads :: FG = nothing
    func_and_grads_and_hessians :: FGH = nothing
    backend :: B = NoBackend()
    func_is_multi :: Bool = false

    func_iip :: Bool = false # true -> func!(y, x, p), false -> y = func(x, p)
    grads_iip :: Bool = false # true -> grads!(Dy, x, p), false -> Dy = grads(x, p)
    hessians_iip :: Bool = false # true -> hessians!(H, x, p), false -> H = hessians(x, p)
    func_and_grads_iip :: Bool = grads_iip  # true -> func_and_grads!(y, Dy, x, p), false -> y, Dy = func_and_grads(x, p)
    func_and_grads_and_hessians_iip :: Bool = hessians_iip # true -> func_and_grads_and_hessians!(y, Dy, H, x, p), false -> y, Dy, H = func_and_grads_and_hessians(x, p)

    num_func_calls :: FuncCallCounter = FuncCallCounter()
    num_grad_calls :: FuncCallCounter = FuncCallCounter()
    num_hess_calls :: FuncCallCounter = FuncCallCounter()
 
    max_func_calls :: Int = typemax(Int)
    max_grad_calls :: Int = typemax(Int)
    max_hess_calls :: Int = typemax(Int)

    name :: Union{Nothing, String} = nothing
    
    # Hessians should imply gradients
    @assert (
        isnothing(backend) && isnothing(hessians) && isnothing(func_and_grads_and_hessians)
    ) || (
        !isnothing(backend) || !isnothing(grads) || !isnothing(func_and_grads)
    ) "If Hessians can be computed, then gradients should be computed too."
end

CE.operator_has_params(op::NonlinearParametricFunction)=true
CE.operator_can_partial(op::NonlinearParametricFunction)=false
CE.operator_can_eval_multi(op::NonlinearParametricFunction)=op.func_is_multi
CE.operator_has_name(op::NonlinearParametricFunction)=!isnothing(op.name)
CE.operator_name(op::NonlinearParametricFunction)=op.name

function CE.provides_grads(op::NonlinearParametricFunction)
    !isnothing(op.backend) && return true
    !isnothing(op.grads) && return true
    !isnothing(op.func_and_grads) && return true
    return false
end
function CE.provides_hessians(op::NonlinearParametricFunction)
    !isnothing(op.backend) && return true
    !isnothing(op.hessians) && return true
    !isnothing(op.func_and_grads_and_hessians) && return true
    return false
end

function CE.func_call_counter(op::NonlinearParametricFunction, ::Val{0})
    return op.num_func_calls
end
function CE.func_call_counter(op::NonlinearParametricFunction, ::Val{1})
    return op.num_grad_calls
end
function CE.func_call_counter(op::NonlinearParametricFunction, ::Val{2})
    return op.num_hess_calls
end

function CE.max_num_calls(op::NonlinearParametricFunction, ::Val{0})
    return op.max_func_calls
end
function CE.max_num_calls(op::NonlinearParametricFunction, ::Val{1})
    return op.max_grad_calls
end
function CE.max_num_calls(op::NonlinearParametricFunction, ::Val{2})
    return op.max_hess_calls
end

function CE.eval_op!(y::RVec, op::NonlinearParametricFunction, x::RVec, p)
    if op.func_iip 
        op.func(y, x, p)
    else
        y .= op.func(x, p)
    end
    return nothing
end

function CE.eval_op!(y::RMat, op::NonlinearParametricFunction, x::RMat, p)
    if !op.func_is_multi
        error("NonlinearParametricFunction has `func_is_multi==false`. This method should not have been called.")
    end
    if op.func_iip 
        op.func(y, x, p)
    else
        y .= op.func(x, p)
    end
    return nothing
end

function CE.eval_grads!(Dy, op::NonlinearParametricFunction, x, p)
    if !isnothing(op.grads)
        if op.grads_iip
            op.grads(Dy, op.grads!, x, p)
        else
            Dy .= op.grads(x, p)
        end
    elseif !isnothing(op.func_and_grads)
        @debug "Allocating temporary output array for `eval_grads!`."
        if op.func_and_grads_iip
            y = similar(Dy, size(Dy, 1))
            op.func_and_grads(y, Dy, x, p)
        else
            _, _Dy = op.func_and_grads(x, p)
            Dy .= _Dy
        end
    else
        ad_grads!(Dy, op.backend, op.func, x, p, Val(op.func_iip))
    end
    return nothing
end

function CE.eval_hessians!(H, op::NonlinearParametricFunction, x, p)
    if !isnothing(op.hessians)
        if op.hessians_iip
            op.hessians(H, x, p)
        else
            _H = op.hessians(x, p)
            H .= _H
        end
    elseif !isnothing(op.func_and_grads_and_hessians)
        @debug "Allocating temporary output arrays for `eval_hessians!`."
        if op.func_and_grads_and_hessians_iip
            y = similar(H, size(H, 3))
            Dy = similar(H, size(H, 1), length(y))
            op.func_and_grads_and_hessians(y, Dy, H, x, p)
        else
            _, _, _H = op.func_and_grads_and_hessians(x, p)
            H .= _H
        end
    else
        ad_hessians!(H, op.backend, op.func, x, p, Val(op.func_iip))
    end
    return nothing
end

function CE.eval_op_and_grads!(y, Dy, op::NonlinearParametricFunction, x, p)
    if !isnothing(op.func_and_grads)
        if op.func_and_grads_iip
            op.func_and_grads(y, Dy, x, p)
        else
            @debug "Allocating temporary output arrays for `eval_op_and_grads!`."
            _y, _Dy = op.func_and_grads(x, p)
            y .= _y
            Dy .= _Dy
        end
    elseif !isnothing(op.grads)
        @ignoraise CE.eval_op!(y, op, x, p)
        @ignoraise CE.eval_grads!(Dy, op, x, p)
    else
        ad_op_and_grads!(y, Dy, op.backend, op.func, x, p, Val(op.func_iip))
    end
    
   return nothing
end

function CE.eval_op_and_grads_and_hessians!(y, Dy, H, op::NonlinearParametricFunction, x, p)
  
    if !isnothing(op.func_and_grads_and_hessians)
        if op.func_and_grads_and_hessians_iip
            op.func_and_grads_and_hessians(y, Dy, H, x, p)
        else
            @debug "Allocating temporary output arrays for `eval_op_and_grads_and_hessians!`."
            _y, _Dy, _H = op.func_and_grads_and_hessians(x, p)
            y .= _y
            Dy .= _Dy
            H .= _H
        end
    elseif !isnothing(op.hessians)
        @ignoraise CE.eval_op_and_grads!(y, Dy, op, x, p)
        @ignoraise CE.eval_hessians!(H, op, x, p)
    else
        ad_op_and_grads_and_hessians!(y, Dy, H, op.backend, op.func, x, p, Val(op.func_iip))
    end

    return nothing
end

# ## Wrapper for Non-Parametric Functions
# Historically, `NonlinearParametricFunction` was implemented first: 
# An `AbstractNonlinearOperator` with `IsParametricOperator()` trait to 
# automatically make user defined functions compatible with the 
# `AbstractNonlinearOperator` interface.
# The user functions were also expected to take a parameter vector `p` last.
# Sometimes, we do not need that, and to make life easier for the user, we
# provide `NonlinearFunction`.
# 1) Its keyword-argument constructor intercepts all functions and wraps them 
#    in `MakeParametric`, to make them conform to the implementation of the
#    evaluation methods for `NonlinearParametricFunction`.
# 2) It then automatically supports the interface for `AbstractNonlinearOperator`
#    with `IsNonparametricOperator()` through the wrapped field.
#
# This whole affair is a bit convoluted, and tidying things up is a TODO.

## helper to intercept and modify functions without parameters and make them accept some
struct MakeParametric{F} <: Function
    func :: F
end

# Re-define evaluation to take parameters.
# For the general case, with arbitrarily many arguments,
# a `@generated` function appears to work reasonably well...
(@nospecialize(f::MakeParametric))(x, p) = f.func(x)
(@nospecialize(f::MakeParametric))(y, x, p) = f.func(y, x)
(@nospecialize(f::MakeParametric))(y, Dy, x, p) = f.func(y, Dy, x)
(@nospecialize(f::MakeParametric))(y, Dy, Hy, x, p) = f.func(y, Dy, Hy, x)

# `NonlinearFunction` simply wraps a `NonlinearParametricFunction`.
struct NonlinearFunction{
    WF<:CE.AbstractNonlinearOperator} <: CE.AbstractNonlinearOperatorWrapper
    wrapped_function :: WF
end

CE.wrapped_operator(op::NonlinearFunction)=op.wrapped_function
preprocess_inputs(op::NonlinearFunction, x::RVec, p)=(x, nothing)
preprocess_inputs(op::NonlinearFunction, x::RMat, p)=(x, nothing)

# The magic happens with the keyword-argument constructor.
# We intercept user functions without parameters and wrap them first.
# How we parse the `kwargs` is not necessarily performant, so this should not be 
# done in hot-loops:
function NonlinearFunction(; kwargs...)
    new_kwargs = Dict{Any, Any}(kwargs...)
    for fn in (
        :func, :grads, :hessians, :func_and_grads, :func_and_grads_and_hessians
    )
        if haskey(kwargs, fn)
            new_kwargs[fn] = MakeParametric(kwargs[fn])
        end
    end
    return NonlinearFunction(NonlinearParametricFunction(;new_kwargs...))
end

export AbstractAutoDiffBackend, NoBackend
export ad_grads!, ad_hessians!, ad_op_and_grads!, ad_op_and_grads_and_hessians!
export NonlinearFunction, NonlinearParametricFunction
end#module