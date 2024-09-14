module HiGHSStepsExt

import Compromise as C
import JuMP
import JuMP: MOI
import StructHelpers: @batteries
import HiGHS

Base.@kwdef struct HiGHSOptimizerConfig <: C.AbstractJuMPModelConfig
    silent :: Bool = true
    ## restricting the number of threads might be necessary for thread-safety in case
    ## of parallel solver runs
    num_threads :: Union{Nothing, Int} = 1
end
@batteries HiGHSOptimizerConfig selfconstructor=false

function C.init_jump_model(highs_jump_cfg::HiGHSOptimizerConfig, vals, delta)
    opt = JuMP.Model(HiGHS.Optimizer)

    if highs_jump_cfg.silent
        JuMP.set_silent(opt)
    end
    if !isnothing(highs_jump_cfg.num_threads) && highs_jump_cfg.num_threads > 0
        JuMP.set_attribute(opt, MOI.NumberOfThreads(), highs_jump_cfg.num_threads)
    end

    num_vars = C.dim_vars(vals)
    JuMP.set_attribute(opt, "time_limit", max(10, Float64(2*num_vars)))
    
    return opt
end

end