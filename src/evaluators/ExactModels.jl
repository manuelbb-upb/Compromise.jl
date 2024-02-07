module ExactModels

using ..Compromise.CompromiseEvaluators
const CE = CompromiseEvaluators
import Compromise: @serve

Base.@kwdef struct ExactModel{O,P} <: AbstractSurrogateModel#{O<:AbstractNonlinearOperator, P}
    op :: O
    params :: P = nothing
end

struct ExactModelConfig <: AbstractSurrogateModelConfig end

CE.depends_on_radius(::ExactModel)=false
CE.requires_grads(::ExactModelConfig)=true
CE.requires_hessians(::ExactModelConfig)=false

function CE.init_surrogate(::ExactModelConfig, op, dim_in, dim_out, params, T)
    return ExactModel(op, params)
end

function CE.model_op!(y, surr::ExactModel, x)
    #eval_op!(y, surr.op, x, surr.params)
    # if `surr.op` has enforce_max_calls==true then func_vals checks for max_calls
    # if it does not, we could/should do it here...
    @serve CE.check_num_calls(surr.op, 1; force=true)
    return func_vals!(y, surr.op, x, surr.params)
end
function CE.model_grads!(Dy, surr::ExactModel, x)
    #return eval_grads!(Dy, surr.op, x, surr.params)
    @serve CE.check_num_calls(surr.op, 2; force=true)
    return func_grads!(Dy, surr.op, x, surr.params)
end
function CE.model_op_and_grads!(y, Dy, surr::ExactModel, x)
    #return eval_op_and_grads!(y, Dy, surr.op, x, surr.params)
    @serve CE.check_num_calls(surr.op, (1,2); force=true)
    return func_vals_and_grads!(y, Dy, surr.op, x, surr.params)
end
CE.supports_partial_evaluation(::ExactModel)=false

function CE.update!(
    surr::ExactModel, op, Δ, x, fx, lb, ub; kwargs...
)
    #src return CE.check_num_calls(op, (1,2)) #TODO think about this
    return nothing
end
#=
CE.copy_model(mod::ExactModel)=mod
CE.copyto_model!(mod_trgt::ExactModel, mod_src::ExactModel)=mod_trgt
=#
export ExactModel, ExactModelConfig

end#module