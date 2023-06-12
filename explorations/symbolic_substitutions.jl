#=
At some point in time, the solver (or an extension thereof) should support:
* objectives that are defined as symbolic expressions
* re-instantiation of parameter-dependent symbolic objectives
  [init interface for reduced overhead in repeated solves](https://github.com/SciML/Optimization.jl/issues/352)
* modelling of sub-components of those expressions
=#

# ## Symbolics
#=
The first difficulty in doing it with Symbolics alone stems from the fact that 
`@register_symbolic` (same as `@register`, one of them is deprecated) cannot use run-time
symbols for function names. A workaround is to use one more outer `@eval` and interpolate
a generated symbol.
Second, nested systems are more complicated. Maybe ModelingToolkit is better suited.
=#

using Pkg
Pkg.activate(@__DIR__)

using Parameters: @with_kw
using Symbolics

abstract type AbstractModelConfig end

struct ConstantModelConfig <: AbstractModelConfig end

@with_kw struct SymObjf
    objf_expr :: Num
    subsys_func :: Function
    subsys_cfg :: AbstractModelConfig = ConstantModelConfig()
end

function expensive_function(x)
    sleep(1)
    return 1
end

num_vars = 2
@variables x[1:num_vars]

@register expensive_function(x::AbstractVector)::Int
objf_expr = x[1] + expensive_function(x)

objf = SymObjf(; objf_expr, subsys_func = expensive_function)

struct SymModel{F}
    func :: F
end

model_function(f::Function, ::ConstantModelConfig) = x -> 1

function make_model(objf::SymObjf, x_sym_vec)
    model_cfg = objf.subsys_cfg
    subsys_func_mod = model_function(objf.subsys_func, model_cfg)
    @show func_name = gensym(string(typeof(model_cfg).name.name))
    subsys_func_new = @eval begin 
        $(func_name)(x::AbstractVector) = $(subsys_func_mod)(x)
        @eval @register $(func_name)(x::AbstractVector)::Real
        $(func_name)
    end
    model_expr = Symbolics.simplify(substitute(objf.objf_expr, Dict(objf.subsys_func => subsys_func_new)))
    model_func = build_function(model_expr, x_sym_vec; expression = Val{false})
    return SymModel(model_func)
end

function pseudooptimize(objf::SymObjf, x_sym_vec)
    # in every iteration:
    mod = make_model(objf, x_sym_vec)
    return mod
end

mod = pseudooptimize(objf, x)
mod.func(rand(2))

#%%
using ModelingToolkit
using Optimization
const MTK=ModelingToolkit

@variable y
@named inner_sys = OptimizationSystem(0, [x[1], y], [], constraints=[y ~ sin(x[1])] )

outer = 1 + inner_sys.y + x[1] + x[2]

@named outer_sys = OptimizationSystem(outer, [x[1], x[2], inner_sys.y], [], systems = [inner_sys])

@named inner_sys = NonlinearSystem([y ~ sin(x[1])], [x[1], y], [])
@variables objf_val
@named outer_sys = NonlinearSystem([objf_val ~ inner_sys.y + 1 + sum(x)], [x[1], x[2]], [], systems=[inner_sys,])