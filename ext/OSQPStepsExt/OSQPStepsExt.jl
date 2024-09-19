module OSQPStepsExt

import Compromise: dim_vars, init_jump_model
import Compromise: AbstractJuMPModelConfig
import Compromise as C
import JuMP
import JuMP: MOI
import StructHelpers: @batteries
import OSQP

Base.@kwdef struct OSQPOptimizerConfig <: AbstractJuMPModelConfig
    silent :: Bool = true
    polishing :: Bool = true
    max_time :: Union{Int, Nothing} = nothing
    eps_rel :: Union{Float64, Nothing, Missing} = missing
    eps_abs :: Union{Float64, Nothing, Missing} = missing
    max_iter :: Union{Nothing, Int} = nothing
end
@batteries OSQPOptimizerConfig selfconstructor=false

function init_jump_model(osqp_jump_cfg::OSQPOptimizerConfig, vals, delta)
    opt = JuMP.Model(OSQP.Optimizer)
    if osqp_jump_cfg.silent
        JuMP.set_silent(opt)
    end
    time_limit = if !isnothing(osqp_jump_cfg.max_time) && osqp_jump_cfg.max_time > 0
        osqp_jump_cfg.max_time
    else
        num_vars = dim_vars(vals)
        max(10, 2*num_vars)
    end |> Float64
    JuMP.set_attribute(opt, "time_limit", time_limit)
    
    JuMP.set_attribute(opt, "adaptive_rho_interval", 25)

    if osqp_jump_cfg.polishing
        JuMP.set_attribute(opt, "polishing", true)
    end

    if !isnothing(osqp_jump_cfg.max_iter) && osqp_jump_cfg.max_iter > 0
        JuMP.set_attribute(opt, "max_iter", osqp_jump_cfg.max_iter)
    end

    if !isnothing(osqp_jump_cfg.eps_rel)
        if !ismissing(osqp_jump_cfg.eps_rel) 
            if osqp_jump_cfg.eps_rel > 0
                JuMP.set_attribute(opt, "eps_rel", osqp_jump_cfg.eps_rel)
            end
        else
            eps_rel = delta >= 1e-3 ? 1e-4 : delta >= 1e-6 ? 1e-5 : 1e-6
            JuMP.set_attribute(opt, "eps_rel", eps_rel)
        end
    end

    if !isnothing(osqp_jump_cfg.eps_abs)
        if !ismissing(osqp_jump_cfg.eps_abs) 
            if osqp_jump_cfg.eps_abs > 0
                JuMP.set_attribute(opt, "eps_abs", osqp_jump_cfg.eps_abs)
            end
        else
            eps_abs = delta >= 1e-3 ? 1e-4 : delta >= 1e-6 ? 1e-5 : 1e-6
            JuMP.set_attribute(opt, "eps_abs", eps_abs)
        end
    end
    
    return opt
end

end