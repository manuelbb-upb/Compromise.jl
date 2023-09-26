module NonlinearFunctions

include("abstract_autodiff.jl")

using ..Compromise
using ..Compromise.CompromiseEvaluators
const CE = CompromiseEvaluators
#=
if !isdefined(Base, :get_extension)
    using Requires
end

@static if !isdefined(Base, :get_extension)
    function __init__()
        @require ForwardDiff="f6369f11-7733-5829-9624-2563aa707210" begin
            include("../../ext/ForwardDiffBackendExt/ForwardDiffBackendExt.jl")
            using .ForwardDiffBackendExt
        end
    end
end
=#

using Parameters: @with_kw

# ## Wrapper for Parametric Functions

"""
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

@with_kw struct NonlinearParametricFunction{F, G, H, FG, FGH, B} <: AbstractNonlinearOperatorWithParams
    func :: F
    grads :: G = nothing
    hessians :: H = nothing
    func_and_grads :: FG = nothing
    func_and_grads_and_hessians :: FGH = nothing
    backend :: B = NoBackend()

    func_iip :: Bool = true # true -> func!(y, x, p), false -> y = func(x, p)
    grads_iip :: Bool = true # true -> grads!(Dy, x, p), false -> Dy = grads(x, p)
    hessians_iip :: Bool = true # true -> hessians!(H, x, p), false -> H = hessians(x, p)
    func_and_grads_iip :: Bool = grads_iip  # true -> func_and_grads!(y, Dy, x, p), false -> y, Dy = func_and_grads(x, p)
    func_and_grads_and_hessians_iip :: Bool = hessians_iip # true -> func_and_grads_and_hessians!(y, Dy, H, x, p), false -> y, Dy, H = func_and_grads_and_hessians(x, p)
end

function CE.provides_grads(op::NonlinearParametricFunction)
    return !isnothing(op.grads) || !isnothing(op.func_and_grads) || !(op.backend isa NoBackend)
end
function CE.provides_hessians(op::NonlinearParametricFunction)
    return !isnothing(op.hessians) || !isnothing(op.func_and_grads_and_hessians) || 
        !(op.backend isa NoBackend)
end

function CE.eval_op!(y, op::NonlinearParametricFunction, x, p)
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
        eval_op!(y, op, x, p)
        eval_grads!(Dy, op, x, p)
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
            _y, _Dy, _H = op.func_and_grads(x, p)
            y .= _y
            Dy .= _Dy
            H .= _H
        end
    elseif !isnothing(op.hessians)
        eval_op_and_grads!(y, Dy, op, x, p)
        eval_hessians!(H, op, x, p)
    else
        ad_op_and_grads_and_hessians!(y, Dy, H, op.backend, op.func, x, p, Val(op.func_iip))
    end
    return nothing
end

# ## Wrapper for Non-Parametric Functions
# Historically, `NonlinearParametricFunction` was implemented first: 
# An `AbstractNonlinearOperatorWithParams`, to automatically make user defined functions
# compatible with the `AbstractNonlinearOperator` interface.
# The user functions were also expected to take a parameter vector `p` last.
# Sometimes, we do not need that, and to make life easier for the user, we
# provide `NonlinearFunction`.
# 1) Its keyword-argument constructor intercepts all functions and wraps them 
# in `NoParams`, to make them conform to the implementation of the methods of
# `NonlinearParametricFunction`.
# 2) It then automatically supports the interface for `AbstractNonlinearOperatorWithParams`
# through its `wrapped_function`.
# 3) For the sake of convenience, the interface for 
# `AbstractNonlinearOperatorNoParams` is also available automatically.
#
# This whole affair is a bit convoluted, and tidying things up is a TODO.

## helper to intercept and modify functions without parameters and make them accept some
struct NoParams{F}
    func :: F
end

# Re-define evaluation to take parameters.
# For the general case, with arbitrarily many arguments,
# a `@generated` function appears to work reasonably well...
(f::NoParams)(x, p) = f.func(x)
(f::NoParams)(y, x, p) = f.func(y, x)
(f::NoParams)(y, Dy, x, p) = f.func(y, Dy, x)
(f::NoParams)(y, Dy, Hy, x, p) = f.func(y, Dy, Hy, x)

# `NonlinearFunction` simply wraps a `NonlinearParametricFunction`.
struct NonlinearFunction{WF<:AbstractNonlinearOperator} <: AbstractNonlinearOperatorNoParams
    wrapped_function :: WF
end

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
            new_kwargs[fn] = NoParams(kwargs[fn])
        end
    end
    return NonlinearFunction(NonlinearParametricFunction(;new_kwargs...))
end

# Why do a 5-minute task by hand, if you can spend 5 hours automating it?

macro forward(call_ex)
    @assert call_ex.head == :call
    func_name = call_ex.args[1]
    func_args = Any[]
    kwargs = nothing
    for arg_ex in call_ex.args[2:end]
        if arg_ex isa Symbol
            push!(func_args, arg_ex)
        elseif Meta.isexpr(arg_ex, :(::))
            if arg_ex.args[2] == :(NonlinearFunction)
                push!(func_args, :(getfield($(arg_ex.args[1]), :wrapped_function)))
            else
                push!(func_args, arg_ex.args[1])
            end
        elseif Meta.isexpr(arg_ex, :parameters)
            kwargs = arg_ex.args
        end
    end
    if isnothing(kwargs)
        return quote 
            function $(func_name)($(call_ex.args[2:end]...))
                return $(func_name)($(func_args...))
            end
        end |> esc
    else
        return quote 
            function $(func_name)($(call_ex.args[2:end]...))
                return $(func_name)($(func_args...); $(kwargs...))
            end
        end |> esc
    end
end

@forward CE.supports_partial_evaluation(op::NonlinearFunction)
@forward CE.provides_grads(op::NonlinearFunction)
@forward CE.provides_hessians(op::NonlinearFunction)
@forward CE.eval_op!(y, op::NonlinearFunction, x)
@forward CE.eval_grads!(Dy, op::NonlinearFunction, x)
@forward CE.eval_hessians!(H, op::NonlinearFunction, x)
@forward CE.eval_op_and_grads!(y, Dy, op::NonlinearFunction, x)
@forward CE.eval_op_and_grads_and_hessians!(y, Dy, H, op::NonlinearFunction, x)

export AbstractAutoDiffBackend, NoBackend
export ad_grads!, ad_hessians!, ad_op_and_grads!, ad_op_and_grads_and_hessians!
export NonlinearFunction, NonlinearParametricFunction
end#module