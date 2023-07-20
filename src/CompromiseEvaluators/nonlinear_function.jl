# requires "wrapped_function.jl"

# Historically, `WrappedFunction` was implemented first: 
# An `AbstractNonlinearOperatorWithParams`, to automatically make user defined functions
# compatible with the `AbstractNonlinearOperator` interface.
# The user functions were also expected to take a parameter vector `p` last.
# Sometimes, we do not need that, and to make life easier for the user, we
# provide `NonlinearFunction`.
# 1) Its keyword-argument constructor intercepts all functions and wraps them 
# in `NoParams`, to make them conform to the implementation of the methods of
# `WrappedFunction`.
# 2) It then automatically supports the interface for `AbstractNonlinearOperatorWithParams`
# through its `wrapped_function`.
# 3) For the sake of convenience, we can also implement the interface for 
# `AbstractNonlinearOperatorNoParams`.
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

# `NonlinearFunction` simply wraps a `WrappedFunction`.
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
    return NonlinearFunction(WrappedFunction(;new_kwargs...))
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

@forward supports_partial_evaluation!(op::NonlinearFunction)
@forward provides_grads(op::NonlinearFunction)
@forward provides_hessians(op::NonlinearFunction)
@forward eval_op!(y, op::NonlinearFunction, x)
@forward eval_grads!(Dy, op::NonlinearFunction, x)
@forward eval_hessians!(H, op::NonlinearFunction, x)
@forward eval_op_and_grads!(y, Dy, op::NonlinearFunction, x)
@forward eval_op_and_grads_and_hessians!(y, Dy, H, op::NonlinearFunction, x)