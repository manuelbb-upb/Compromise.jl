using Compromise
@testset  "SteepestDescentConfig" begin
    for backtracking_factor in (-1.0, -1//2, 0, 1f0, 2)
        @test_throws AssertionError Compromise.SteepestDescentConfig(;
            backtracking_factor
        )
    end
    for rhs_factor in (-1.0, -1//2, 0, 1f0, 2)
        @test_throws AssertionError Compromise.SteepestDescentConfig(;
            rhs_factor
        )
    end
    for descent_step_norm in (0, 1, 2, 3)
        @test_throws AssertionError Compromise.SteepestDescentConfig(;
            descent_step_norm
        )
    end
    for normal_step_norm in (0, 1, 3)
        @test_throws AssertionError Compromise.SteepestDescentConfig(;
            normal_step_norm
        )
    end

    cfg_default = Compromise.SteepestDescentConfig()
    @test cfg_default.backtracking_factor == 1//2
    @test cfg_default.rhs_factor == Compromise.DEFAULT_FLOAT_TYPE(0.001)
    @test cfg_default.normalize_gradients == false
    @test cfg_default.strict_backtracking == true
    @test cfg_default.descent_step_norm == Inf
    @test cfg_default.normal_step_norm == 2
    @test cfg_default.qp_opt == Compromise.DEFAULT_QP_OPTIMIZER

    cfg_2 = Compromise.SteepestDescentConfig(
        backtracking_factor = 0.5,
        rhs_factor = Compromise.DEFAULT_FLOAT_TYPE(0.001),
        normalize_gradients = false,
        strict_backtracking = true, 
        descent_step_norm = Inf,
        normal_step_norm = 2,
        qp_opt = Compromise.DEFAULT_QP_OPTIMIZER
    )

    @test cfg_default == cfg_2
end

@testset "AlgorithmOptions" begin
    # TODO
    opts = AlgorithmOptions()
    _opts = deepcopy(opts)

    @test opts == _opts
    @test opts.stop_crit_tol_abs isa Float32

    opts = AlgorithmOptions(; T = Float16)
    @test opts.stop_crit_tol_abs isa Float16
    
    opts = AlgorithmOptions{Float64}(; T = Float16)
    @test opts.stop_crit_tol_abs isa Float16

    opts = AlgorithmOptions{Float64}()
    @test opts.stop_crit_tol_abs isa Float64

    opts = AlgorithmOptions{Float64}(; stop_crit_tol_abs=1f0)
    @test opts.stop_crit_tol_abs isa Float64
end

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
    final_vals, stop_code = optimize(
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
    @test stop_code.crit isa Compromise.MaxIterStopping

    #%%
    mop = setup_mop()
    final_vals, stop_code = optimize(
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
    @test stop_code.crit isa Compromise.MinimumRadiusStopping

    #%%
    ## Relative argument stopping criterion does only apply if 
    ## a trial point is accepted. 
    ## We thus should not start at a critical site:
    mop = setup_mop()
    final_vals, stop_code = optimize(
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
    @test stop_code.crit isa Compromise.ArgsRelTolStopping

    #%%
    mop = setup_mop()
    final_vals, stop_code = optimize(
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
    @test stop_code.crit isa Compromise.ArgsAbsTolStopping

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
    @test stop_code.crit isa Compromise.ValsRelTolStopping

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
    @test stop_code.crit isa Compromise.ValsAbsTolStopping

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
    @test stop_code.crit isa Compromise.CritAbsTolStopping

    #%%
    struct MyCallback <: Compromise.AbstractStoppingCriterion end
    Compromise.check_pre_iteration(::MyCallback)=true
        
    function Compromise.evaluate_stopping_criterion(
        crit::MyCallback,
        Δ, mop, mod, scaler, lin_cons, scaled_cons,
        vals, vals_tmp, step_vals, mod_vals, filter, iter_meta, step_cache, algo_opts,
    )
        if vals.x ≈ [π, -ℯ]
            return crit
        end
        return nothing
    end

    mop = setup_mop()
    final_vals, stop_code = optimize(
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
    @test stop_code.crit isa MyCallback
end

@testset "Max Eval Stopping" begin

    algo_opts = AlgorithmOptions(;
        max_iter=typemax(Int),
        stop_delta_min=-Inf,
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
    final_vals, stop_code = optimize(mop, [π, -ℯ]; algo_opts)

    @test fn_counter[] == 10
    @test stop_code.crit isa Compromise.NonlinearFunctions.BudgetExhausted

    fn_counter[] = 0
    dfn_counter[] = 0
    
    mop = MutableMOP(;num_vars=2)
    add_objectives!(
        mop, objective_function, grads_objectives_function, :exact; 
        dim_out=2, max_grad_calls=1 
    )
    final_vals, stop_code = optimize(mop, [π, -ℯ]; algo_opts)
    @test dfn_counter[] == 1
    @test stop_code.crit isa Compromise.NonlinearFunctions.BudgetExhausted

    mop = MutableMOP(;num_vars=2)
    add_objectives!(
        mop, objective_function, grads_objectives_function, :exact; 
        dim_out=2, max_grad_calls=1, 
    )
    final_vals, stop_code = optimize(mop, [π, -ℯ]; algo_opts)
    @test dfn_counter[] == 2
    @test stop_code.crit isa Compromise.NonlinearFunctions.BudgetExhausted

    fn_counter[] = 0
    dfn_counter[] = 0
    
    mop = MutableMOP(;num_vars=2)
    add_objectives!(
        mop, objective_function, grads_objectives_function, :exact; 
        dim_out=2, max_func_calls=100
    )
    final_vals, stop_code = optimize(mop, [π, -ℯ]; algo_opts)

    @test fn_counter[] == 100
    @test stop_code.crit isa Compromise.NonlinearFunctions.BudgetExhausted

    fn_counter[] = 0
    dfn_counter[] = 0
    
    mop = MutableMOP(;num_vars=2)
    add_objectives!(
        mop, objective_function, grads_objectives_function, :rbf; 
        dim_out=2, max_func_calls=10
    )
    final_vals, stop_code = optimize(mop, [π, -ℯ]; algo_opts)
    @test fn_counter[] == 10
    @test stop_code.crit isa Compromise.NonlinearFunctions.BudgetExhausted

    fn_counter[] = 0
    dfn_counter[] = 0
    
    mop = MutableMOP(;num_vars=2)
    add_objectives!(
        mop, objective_function, grads_objectives_function, :taylor1; 
        dim_out=2, max_func_calls=10
    )
    final_vals, stop_code = optimize(mop, [π, -ℯ]; algo_opts)
    @test fn_counter[] == 10
    @test stop_code.crit isa Compromise.NonlinearFunctions.BudgetExhausted

    fn_counter[] = 0
    dfn_counter[] = 0
    
    mop = MutableMOP(;num_vars=2)
    add_objectives!(
        mop, objective_function, grads_objectives_function, :taylor1; 
        dim_out=2, max_grad_calls=10
    )
    final_vals, stop_code = optimize(mop, [π, -ℯ]; algo_opts)
    @test dfn_counter[] == 10
    @test stop_code.crit isa Compromise.NonlinearFunctions.BudgetExhausted
    
    fn_counter[] = 0
    dfn_counter[] = 0

    mop = MutableMOP(;num_vars=2)
    add_objectives!(
        mop, objective_function, grads_objectives_function, :taylor2; 
        dim_out=2, max_func_calls=10, 
        hessians=hess_objectives_function, hessians_iip=false 
    )
    final_vals, stop_code = optimize(mop, [π, -ℯ]; algo_opts)
    @test fn_counter[] <= 10
    @test stop_code.crit isa Compromise.NonlinearFunctions.BudgetExhausted

    fn_counter[] = 0
    dfn_counter[] = 0

    mop = MutableMOP(;num_vars=2)
    add_objectives!(
        mop, objective_function, grads_objectives_function, :taylor2; 
        dim_out=2, max_grad_calls=2, 
        hessians=hess_objectives_function, hessians_iip=false 
    )
    final_vals, stop_code = optimize(mop, [π, -ℯ]; algo_opts)
    @test dfn_counter[] == 2
    @test stop_code.crit isa Compromise.NonlinearFunctions.BudgetExhausted
    
    mop = MutableMOP(;num_vars=2)
    add_objectives!(
        mop, objective_function, grads_objectives_function, :taylor2; 
        dim_out=2, max_hess_calls=2, 
        hessians=hess_objectives_function, hessians_iip=false 
    )
    final_vals, stop_code = optimize(mop, [π, -ℯ]; algo_opts)
    @test stop_code.crit isa Compromise.NonlinearFunctions.BudgetExhausted

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
    final_vals, _ = optimize(mop, randx(); algo_opts)
    ξ = final_vals.ξ
    @test all(lb .- 1e-5 .<= ξ)
    @test all(ξ .<= ub .- 1e-5)
end