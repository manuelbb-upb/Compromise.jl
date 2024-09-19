using Compromise
using Test
include("TestProblems.jl")
TP = TestProblems
#%%
tp = TP._test_problem(Val(1), 2, Float64)
mop = TP.to_mutable_mop(tp; mcfg=:rbf)
ret = optimize(mop, tp.x0)
xopt = opt_vars(ret)
@test xopt[1] ≈ xopt[2]

#%%
tp = TP._test_problem(Val(2), 3, Float64)
mop = TP.to_mutable_mop(tp; mcfg=:rbf)
ret = optimize(mop, tp.x0)
xopt = opt_vars(ret)
@test sum(xopt) ≈ 1 rtol=1e-4
@test all(tp.lb .<= opt_vars(ret) .<= tp.ub)

#%%
tp = TP._test_problem(Val(3), 3, Float64)
mop = TP.to_mutable_mop(tp; mcfg=:rbf)
ret = optimize(mop, tp.x0)
xopt = opt_vars(ret)
@test sum(xopt) ≈ 1 rtol=1e-4
@test all(tp.lb .<= opt_vars(ret) .<= tp.ub)
@test all(tp.A * opt_vars(ret) .<= tp.b)
#%%
tp = TP._test_problem(Val(4), 3, Float64)
mop = TP.to_mutable_mop(tp; mcfg=:rbf)
ret = optimize(mop, tp.x0)
xopt = opt_vars(ret)
@test sum(xopt) ≈ 1 rtol=1e-4
@test all(tp.lb .<= opt_vars(ret) .<= tp.ub)
@test all(tp.A * opt_vars(ret) .<= tp.b)
#%%
tp = TP._test_problem(Val(5), 3, Float64)
mop = TP.to_mutable_mop(tp; mcfg=:rbf)
#ret = optimize(mop, tp.x0; algo_opts = AlgorithmOptions(; c_delta=0.1))
ret = optimize(mop, tp.x0)
xopt = opt_vars(ret)
@test sum(xopt) ≈ 1 rtol=1e-4
@test all(tp.lb .<= opt_vars(ret) .<= tp.ub)
@test all(tp.A * opt_vars(ret) .<= tp.b)
#%%
tp = TP._test_problem(Val(6), 3, Float64)
mop = TP.to_mutable_mop(tp; mcfg=:rbf)
ret = optimize(mop, tp.x0)
@test opt_stop_code(ret) isa Compromise.InfeasibleStopping