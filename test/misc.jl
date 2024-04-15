using Compromise

@testset "Taylor Polynomials deg 2" begin
    tcfg = TaylorPolynomialConfig(;degree=2)
    function func(x)
        return [
            sum( (x .- 1).^2 ),
            sum( (x .+ 1).^2 ),
        ]
    end

    function grads(x)
        return 2 .* hcat( x .- 1, x .+ 1 )
    end

    function hessians(x)
        n = length(x)
        H = zeros(eltype(x), n, n, 2)
        for l in [1,2]
            for i=1:n
                H[i, i, l] = 2
            end
        end
        return H
    end

    op = Compromise.NonlinearFunction(;func, grads, hessians)
    tp = Compromise.init_surrogate(tcfg, op, 2, 2, nothing, Float64)

    x = rand(2)
    y = zeros(2)
    Compromise.func_vals!(y, op, x)
    @test y ≈ func(x)
    Compromise.update!(tp, op, nothing, x, y, nothing, nothing)
    z = zeros(2)
    Compromise.func_vals!(z, tp, x)

    @test y ≈ z

    Dy = zeros(2,2)
    Compromise.func_grads!(Dy, tp, x)
    @test Dy ≈ grads(x)

    Dy .= 0
    y .= 0
    Compromise.func_vals_and_grads!(y, Dy, tp, x)
    @test y ≈ func(x)
    @test Dy ≈ grads(x)

    Hy = zeros(2,2,2)
    Dy .= 0
    y .= 0
    Compromise.func_vals_and_grads_and_hessians!(y, Dy, Hy, op, x)
    @test y ≈ func(x)
    @test Dy ≈ grads(x)
    @test Hy ≈ hessians(x)
end

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
    ret = optimize(
        mop, rand(2);
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
    @test opt_stop_code(ret) isa Compromise.MaxIterStopping

    #%%
    mop = setup_mop()
    ret = optimize(
        mop, rand(2);
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
    @test opt_stop_code(ret) isa Compromise.MinimumRadiusStopping

    #%%
    ## Relative argument stopping criterion does only apply if 
    ## a trial point is accepted. 
    ## We thus should not start at a critical site:
    mop = setup_mop()
    ret = optimize(
        mop, [π, -ℯ];
        algo_opts = AlgorithmOptions(;
            eps_crit = -1.0,        # don't enter criticality loop
            max_iter=typemax(Int),
            stop_delta_min=-Inf,
            stop_xtol_rel=1e-1,
            stop_xtol_abs=-Inf,
            stop_ftol_rel=-Inf,
            stop_ftol_abs=-Inf,
            stop_crit_tol_abs=-Inf,
            stop_theta_tol_abs=-Inf,
            stop_max_crit_loops=typemax(Int)
        )
    )
    @test opt_stop_code(ret) isa Compromise.ArgsRelTolStopping

    #%%
    mop = setup_mop()
    ret = optimize(
        mop, [π, -ℯ];
        algo_opts = AlgorithmOptions(;
            eps_crit = -1.0,
            max_iter=typemax(Int),
            stop_delta_min=-Inf,
            stop_xtol_rel=-Inf,
            stop_xtol_abs=1e-1,
            stop_ftol_rel=-Inf,
            stop_ftol_abs=-Inf,
            stop_crit_tol_abs=-Inf,
            stop_theta_tol_abs=-Inf,
            stop_max_crit_loops=typemax(Int)
        )
    )
    @test opt_stop_code(ret) isa Compromise.ArgsAbsTolStopping

    #%%
    mop = setup_mop()
    ret = optimize(
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
    @test opt_stop_code(ret) isa Compromise.ValsRelTolStopping

    #%%
    mop = setup_mop()
    ret = optimize(
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
    @test opt_stop_code(ret) isa Compromise.ValsAbsTolStopping

    #%%
    mop = setup_mop()
    ret = optimize(
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
    @test opt_stop_code(ret) isa Compromise.CritAbsTolStopping

    #%%
    struct MyCallback <: Compromise.AbstractStoppingCriterion end
            
    function Compromise.check_stopping_criterion(
        crit::MyCallback, ::Compromise.CheckPreIteration,
        mop, scaler, lin_cons, scaled_cons,
        vals, filter, algo_opts;
        indent::Int, it_index::Int, delta::Real
    )
        if Compromise.cached_x(vals) ≈ [π, -ℯ]
            return crit
        end
        return nothing
    end

    mop = setup_mop()
    ret = optimize(
        mop, [π, -ℯ];
        algo_opts = AlgorithmOptions(;
            max_iter=5,
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
    @test_broken opt_stop_code(ret) isa MyCallback
end

@testset "Max Eval Stopping" begin

    algo_opts = AlgorithmOptions(;
        stop_delta_min=1e-10,
        stop_xtol_rel=-Inf,
        stop_xtol_abs=-Inf,
        stop_ftol_rel=-Inf,
        stop_ftol_abs=-Inf,
        stop_crit_tol_abs=-Inf,
        stop_theta_tol_abs=-Inf,
        stop_max_crit_loops=20,
    )

    fn_counter = Ref(0)
    dfn_counter = Ref(0)

    function objective_function(x)
        fn_counter[] += 1
        return [
            sum( (x .- 1).^2 ),
            sum( (x .+ 1).^2 ),
        ]
    end

    function grads_objectives_function(x)
        dfn_counter[] += 1
        return 2 .* hcat( x .- 1, x .+ 1 )
    end

    function hess_objectives_function(x)
        n = length(x)
        H = zeros(eltype(x), n, n, 2)
        for l in [1,2]
            for i=1:n
                H[i, i, l] = 2
            end
        end
        return H
    end

    mop = MutableMOP(;num_vars=2)
    add_objectives!(
        mop, objective_function, grads_objectives_function, :exact; 
        dim_out=2, max_func_calls=10, 
    )
    ret = optimize(mop, [π, -ℯ]; algo_opts)

    @test fn_counter[] == 10
    @test opt_stop_code(ret) isa  Compromise.CompromiseEvaluators.BudgetExhausted

    fn_counter[] = 0
    dfn_counter[] = 0
    
    mop = MutableMOP(;num_vars=2)
    add_objectives!(
        mop, objective_function, grads_objectives_function, :exact; 
        dim_out=2, max_grad_calls=1 
    )
    ret = optimize(mop, [π, -ℯ]; algo_opts)
    @test dfn_counter[] == 1
    @test opt_stop_code(ret) isa  Compromise.CompromiseEvaluators.BudgetExhausted

    mop = MutableMOP(;num_vars=2)
    add_objectives!(
        mop, objective_function, grads_objectives_function, :exact; 
        dim_out=2, max_grad_calls=1, 
    )
    ret = optimize(mop, [π, -ℯ]; algo_opts)
    @test dfn_counter[] == 2
    @test opt_stop_code(ret) isa  Compromise.CompromiseEvaluators.BudgetExhausted

    fn_counter[] = 0
    dfn_counter[] = 0
    
    mop = MutableMOP(;num_vars=2)
    add_objectives!(
        mop, objective_function, grads_objectives_function, :exact; 
        dim_out=2, max_func_calls=100
    )
    ret = optimize(mop, [π, -ℯ]; algo_opts)

    @test fn_counter[] == 100
    @test opt_stop_code(ret) isa  Compromise.CompromiseEvaluators.BudgetExhausted

    fn_counter[] = 0
    dfn_counter[] = 0
    
    mop = MutableMOP(;num_vars=2)
    add_objectives!(
        mop, objective_function, grads_objectives_function, :rbf; 
        dim_out=2, max_func_calls=10
    )
    ret = optimize(mop, [π, -ℯ]; algo_opts)
    @test fn_counter[] == 10
    @test opt_stop_code(ret) isa  Compromise.CompromiseEvaluators.BudgetExhausted

    fn_counter[] = 0
    dfn_counter[] = 0
    
    mop = MutableMOP(;num_vars=2)
    add_objectives!(
        mop, objective_function, grads_objectives_function, :taylor1; 
        dim_out=2, max_func_calls=10
    )
    ret = optimize(mop, [π, -ℯ]; algo_opts)
    @test fn_counter[] == 10
    @test opt_stop_code(ret) isa  Compromise.CompromiseEvaluators.BudgetExhausted

    fn_counter[] = 0
    dfn_counter[] = 0
    
    mop = MutableMOP(;num_vars=2)
    add_objectives!(
        mop, objective_function, grads_objectives_function, :taylor1; 
        dim_out=2, max_grad_calls=5
    )
    ret = optimize(mop, [π, -ℯ]; algo_opts)
    @test dfn_counter[] <= 5
    @test opt_stop_code(ret) isa  Compromise.CompromiseEvaluators.BudgetExhausted
    
    fn_counter[] = 0
    dfn_counter[] = 0

    mop = MutableMOP(;num_vars=2)
    add_objectives!(
        mop, objective_function, grads_objectives_function, :taylor2; 
        dim_out=2, max_func_calls=10, 
        hessians=hess_objectives_function, hessians_iip=false 
    )
    ret = optimize(mop, [π, -ℯ]; algo_opts)
    @test fn_counter[] <= 10
    @test opt_stop_code(ret) isa  Compromise.CompromiseEvaluators.BudgetExhausted

    fn_counter[] = 0
    dfn_counter[] = 0

    mop = MutableMOP(;num_vars=2)
    add_objectives!(
        mop, objective_function, grads_objectives_function, :taylor2; 
        dim_out=2, max_grad_calls=2, 
        hessians=hess_objectives_function, hessians_iip=false 
    )
    ret = optimize(mop, [π, -ℯ]; algo_opts)
    @test dfn_counter[] == 2
    @test opt_stop_code(ret) isa  Compromise.CompromiseEvaluators.BudgetExhausted
    
    mop = MutableMOP(;num_vars=2)
    add_objectives!(
        mop, objective_function, grads_objectives_function, :taylor2; 
        dim_out=2, max_hess_calls=2, 
        hessians=hess_objectives_function, hessians_iip=false 
    )
    ret = optimize(mop, [π, -ℯ]; algo_opts)
    @test opt_stop_code(ret) isa  Compromise.CompromiseEvaluators.BudgetExhausted

end

@testset "Variable Bounds" begin
    algo_opts = AlgorithmOptions(; stop_delta_min=1e-11)

    function objective_function(x)
        return [
            sum( (x .- 1).^2 ),
            sum( (x .+ 1).^2 ),
        ]
    end

    lb = -10 * rand(2)
    ub = 5 * rand(2)

    mop = MutableMOP(; num_vars=2, lb, ub)
    @test mop.lb == lb
    @test mop.ub == ub

    randx(n=2) = lb .+ (ub .- lb) .* rand(n)
    add_objectives!(mop, objective_function, :rbf; dim_out=2, func_iip=false)
    ret = optimize(mop, randx(); algo_opts)
    ξ = opt_vars(ret)
    @test all(lb .- 1e-5 .<= ξ)
    @test all(ξ .<= ub .- 1e-5)
end