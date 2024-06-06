import Compromise as C
import Compromise: @unpack

includet("TestProblems.jl")
TP = TestProblems

tp = TP._test_problem(Val(3), 2)
_mop = TP.to_mutable_mop(tp; max_func_calls=10000)
X = [3.42e-01; 9.23e-01;;]
X = _mop.lb .+ (_mop.ub .- _mop.lb) .* rand(2, 50)
algo_opts = C.AlgorithmOptions(; 
    nu_success=.8,
    nu_accept=1e-3,
    max_iter=100, 
    stop_delta_min=1e-5,
    step_config = C.SteepestDescentConfig(;
        backtracking_mode = Val(:all)
    )
)

population = C.optimize_set(X, _mop; algo_opts);
population2 = C.optimize_many(X, _mop; algo_opts);