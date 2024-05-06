using Compromise
import Compromise: SteepestDescentConfig

@testset  "SteepestDescentConfig" begin
    for backtracking_factor in (-2, -1.0, -0.5, 0.0, 1f0, Float16(2.0))
        @test_throws AssertionError SteepestDescentConfig(;
            backtracking_factor
        )
    end
    for rhs_factor in (-1.0, -1//2, 0, 1f0, 2)
        @test_throws AssertionError SteepestDescentConfig(;
            rhs_factor
        )
    end
    for descent_step_norm in (0, 1, 2, 3)
        @test_throws AssertionError SteepestDescentConfig(;
            descent_step_norm
        )
    end
    for normal_step_norm in (0, 1, 3)
        @test_throws AssertionError SteepestDescentConfig(;
            normal_step_norm
        )
    end

    function test_cfg(cfg; float_type = Compromise.DEFAULT_FLOAT_TYPE)
        @test cfg.float_type == float_type
        @test cfg.backtracking_factor == 1//2
        @test isapprox(cfg.rhs_factor, 1e-3; rtol=eps(Float16))
        @test cfg.normalize_gradients == false
        @test cfg.strict_backtracking == true
        @test cfg.descent_step_norm == Inf
        @test cfg.normal_step_norm == 2
        @test cfg.qp_opt == Compromise.DEFAULT_QP_OPTIMIZER
    end
    
    cfg_default = SteepestDescentConfig()
    cfg_16 = SteepestDescentConfig(; float_type=Float16)
    cfg_32 = SteepestDescentConfig(; float_type=Float32)
    cfg_64 = SteepestDescentConfig(; float_type=Float64)
    cfg_big = SteepestDescentConfig(; float_type=BigFloat)

    test_cfg(cfg_default)
    test_cfg(cfg_16; float_type=Float16)
    test_cfg(cfg_32; float_type=Float32)
    test_cfg(cfg_64; float_type=Float64)
    test_cfg(cfg_big; float_type=BigFloat)

    cfg_copy = deepcopy(cfg_32)
    @test cfg_copy == cfg_32
    @test isequal(cfg_copy, cfg_32)

    @reset cfg_32.float_type = Float16
    test_cfg(cfg_32; float_type = Float16)
    cfg_64 = @set cfg_16.float_type = Float64
    test_cfg(cfg_64; float_type=Float64)
end


@testset "AlgorithmOptions" begin
    # TODO
    opts = AlgorithmOptions()
    _opts = deepcopy(opts)

    @test opts == _opts
    @test isequal(opts, _opts)
    
    function test_opts(opts; float_type=Compromise.DEFAULT_FLOAT_TYPE)
        @test opts.float_type == float_type
        @test opts.step_config isa SteepestDescentConfig{float_type}
        @test opts.scaler_cfg == Val(:box)
        @test opts.require_fully_linear_models
        @test opts.max_iter == 500
        @test opts.stop_delta_min.val ≈ eps(float_type)
        @test opts.stop_delta_min.is_default
        @test opts.stop_xtol_rel == -Inf
        @test opts.stop_xtol_abs == -Inf
        @test opts.stop_ftol_rel == -Inf
        @test opts.stop_ftol_abs == -Inf
        @test opts.stop_crit_tol_abs == -Inf
        @test opts.stop_theta_tol_abs.val ≈ eps(float_type)
        @test opts.stop_theta_tol_abs.is_default
        @test opts.stop_max_crit_loops == 10
        @test opts.eps_crit ≈ .1 rtol=eps(Float16)
        @test opts.eps_theta ≈ .05 rtol=eps(Float16)
        @test opts.crit_B ≈ 100 rtol=eps(Float16)
        @test opts.crit_M ≈ 3*opts.crit_B rtol=eps(Float16)
        @test opts.crit_alpha ≈ .1 rtol=eps(Float16)
        @test opts.backtrack_in_crit_routine
        @test opts.delta_init ≈ .5 rtol=eps(Float16)
        @test opts.delta_max ≈ 2^5 * opts.delta_init rtol=eps(Float16)
        @test opts.gamma_shrink_much ≈ .1 rtol=eps(Float16)
        @test opts.gamma_shrink ≈ .5 rtol=eps(Float16)
        @test opts.gamma_grow ≈ 2 rtol=eps(Float16)
        @test opts.strict_acceptance_test
        @test opts.nu_accept ≈ 1e-4 rtol=eps(Float16)
        @test opts.nu_success ≈ .4 rtol=eps(Float16)
        @test opts.c_delta ≈ .9 rtol=eps(Float16)
        @test opts.c_mu ≈ 100 rtol=eps(Float16)
        @test opts.mu ≈ .01 rtol=eps(Float16)
        @test opts.kappa_theta ≈ 1e-4 rtol=eps(Float16)
        @test opts.psi_theta ≈ 2 rtol=eps(Float16)
#        @test opts.nl_opt == :GN_DIRECT_L_RAND
    end

    opts16 = AlgorithmOptions(; float_type=Float16)
    opts32 = AlgorithmOptions(; float_type=Float32)
    opts64 = AlgorithmOptions(; float_type=Float64)
    test_opts(opts16; float_type=Float16)
    test_opts(opts32; float_type=Float32)
    test_opts(opts64; float_type=Float64)

    @reset opts16.float_type = Float32
    test_opts(opts16; float_type=Float32)
    opts_big = @set opts64.float_type = BigFloat
    test_opts(opts_big; float_type=BigFloat)

    @reset opts_big.stop_xtol_rel = .1
    @test opts_big.stop_xtol_rel ≈ .1
    @reset opts_big.stop_delta_min = .1
    @test opts_big.stop_delta_min.val ≈ .1
    @test !opts_big.stop_delta_min.is_default
end