using Compromise
using Test

@testset "Stopping Criteria" begin
    function setup_mop()
        mop = MutableMOP(;num_vars=2)

        function objective_function(x)
            return [
                sum( (x .- 1).^2 ),
                sum( (x .+ 1).^2 ),
            ]
        end

        add_objectives!(
            mop, objective_function, :rbf; 
            dim_out=2, func_iip=false,
        )
        
        return mop
    end

    #%%
    mop = setup_mop()
    final_vals, stop_code = optimize(
        mop, zeros(2);
        algo_opts = AlgorithmOptions(;
            max_iter=1,
            stop_delta_min=-Inf,
            stop_xtol_rel=-Inf,
            stop_xtol_abs=-Inf,
            stop_ftol_rel=-Inf,
            stop_ftol_abs=-Inf,
            stop_crit_tol_abs=-Inf,
            stop_theta_tol_abs=-Inf,
            stop_max_crit_loops=typemax(Int)
        )
    )
    @test stop_code isa Compromise.MaxIterStopping

    #%%
    mop = setup_mop()
    final_vals, stop_code = optimize(
        mop, zeros(2);
        algo_opts = AlgorithmOptions(;
            max_iter=typemax(Int),
            stop_delta_min=1e-2,
            stop_xtol_rel=-Inf,
            stop_xtol_abs=-Inf,
            stop_ftol_rel=-Inf,
            stop_ftol_abs=-Inf,
            stop_crit_tol_abs=-Inf,
            stop_theta_tol_abs=-Inf,
            stop_max_crit_loops=typemax(Int)
        )
    )
    @test stop_code isa Compromise.MinimumRadiusStopping

    #%%
    ## Relative argument stopping criterion does only apply if 
    ## a trial point is accepted. 
    ## We thus should not start at a critical site:
    mop = setup_mop()
    final_vals, stop_code = optimize(
        mop, [π, -ℯ];
        algo_opts = AlgorithmOptions(;
            max_iter=typemax(Int),
            stop_delta_min=-Inf,
            stop_xtol_rel=1e-2,
            stop_xtol_abs=-Inf,
            stop_ftol_rel=-Inf,
            stop_ftol_abs=-Inf,
            stop_crit_tol_abs=-Inf,
            stop_theta_tol_abs=-Inf,
            stop_max_crit_loops=typemax(Int)
        )
    )
    @test stop_code isa Compromise.ArgsRelTolStopping

    #%%
    mop = setup_mop()
    final_vals, stop_code = optimize(
        mop, [π, -ℯ];
        algo_opts = AlgorithmOptions(;
            max_iter=typemax(Int),
            stop_delta_min=-Inf,
            stop_xtol_rel=-Inf,
            stop_xtol_abs=1e-2,
            stop_ftol_rel=-Inf,
            stop_ftol_abs=-Inf,
            stop_crit_tol_abs=-Inf,
            stop_theta_tol_abs=-Inf,
            stop_max_crit_loops=typemax(Int)
        )
    )
    @test stop_code isa Compromise.ArgsAbsTolStopping

    #%%
    mop = setup_mop()
    final_vals, stop_code = optimize(
        mop, [π, -ℯ];
        algo_opts = AlgorithmOptions(;
            max_iter=typemax(Int),
            stop_delta_min=-Inf,
            stop_xtol_rel=-Inf,
            stop_xtol_abs=-Inf,
            stop_ftol_rel=0.1,
            stop_ftol_abs=-Inf,
            stop_crit_tol_abs=-Inf,
            stop_theta_tol_abs=-Inf,
            stop_max_crit_loops=typemax(Int)
        )
    )
    @test stop_code isa Compromise.ValsRelTolStopping

    #%%
    mop = setup_mop()
    final_vals, stop_code = optimize(
        mop, [π, -ℯ];
        algo_opts = AlgorithmOptions(;
            max_iter=typemax(Int),
            stop_delta_min=-Inf,
            stop_xtol_rel=-Inf,
            stop_xtol_abs=-Inf,
            stop_ftol_rel=-Inf,
            stop_ftol_abs=0.1,
            stop_crit_tol_abs=-Inf,
            stop_theta_tol_abs=-Inf,
            stop_max_crit_loops=typemax(Int)
        )
    )
    @test stop_code isa Compromise.ValsAbsTolStopping

    #%%
    mop = setup_mop()
    final_vals, stop_code = optimize(
        mop, [π, -ℯ];
        algo_opts = AlgorithmOptions(;
            max_iter=typemax(Int),
            stop_delta_min=-Inf,
            stop_xtol_rel=-Inf,
            stop_xtol_abs=-Inf,
            stop_ftol_rel=-Inf,
            stop_ftol_abs=-Inf,
            stop_crit_tol_abs=1e-4,
            stop_theta_tol_abs=1e-4,
            stop_max_crit_loops=typemax(Int)
        )
    )
    @test stop_code isa Compromise.CritAbsTolStopping

    #%%
    struct MyCallback <: Compromise.AbstractStoppingCriterion end
    Compromise.check_pre_iteration(::MyCallback)=true
        
    function Compromise.evaluate_stopping_criterion(
        crit::MyCallback,
        Δ, mop, mod, scaler, lin_cons, scaled_cons,
        vals, vals_tmp, step_vals, mod_vals, filter, iter_meta, step_cache, algo_opts,
    )
        if vals.x == [π, -ℯ]
            return crit
        end
        return nothing
    end

    mop = setup_mop()
    final_vals, stop_code = optimize(
        mop, [π, -ℯ];
        algo_opts = AlgorithmOptions(;
            max_iter=typemax(Int),
            stop_delta_min=-Inf,
            stop_xtol_rel=-Inf,
            stop_xtol_abs=-Inf,
            stop_ftol_rel=-Inf,
            stop_ftol_abs=-Inf,
            stop_crit_tol_abs=-Inf,
            stop_theta_tol_abs=-Inf,
            stop_max_crit_loops=typemax(Int)
        ),
        user_callback = MyCallback()
    )
    @test stop_code isa MyCallback
end
