module ExactModels

using ..Compromise.CompromiseEvaluators
const CE = CompromiseEvaluators
import Compromise: @ignoraise, RVec

Base.@kwdef struct ExactModel{O, P} <: AbstractSurrogateModel
    op :: O
    params :: P = nothing
end

struct ExactModelConfig <: AbstractSurrogateModelConfig end

CE.depends_on_radius(::ExactModel)=false
CE.requires_grads(::ExactModelConfig)=true
CE.requires_hessians(::ExactModelConfig)=false

function CE.init_surrogate(::ExactModelConfig, op, dim_in, dim_out, params, T; kwargs...)
    return ExactModel(op, params)
end

function CE.eval_op!(y::RVec, surr::ExactModel, x::RVec)
    return func_vals!(y, surr.op, x, surr.params)
end
function CE.eval_grads!(Dy, surr::ExactModel, x)
    return func_grads!(Dy, surr.op, x, surr.params)
end
function CE.eval_op_and_grads!(y, Dy, surr::ExactModel, x)
    return func_vals_and_grads!(y, Dy, surr.op, x, surr.params)
end

function CE.update!(
    surr::ExactModel, op, Δ, x, fx, lb, ub; kwargs...
)
    #src return CE.check_num_calls(op, (1,2)) #TODO think about this
    return nothing
end
export ExactModel, ExactModelConfig

end#module