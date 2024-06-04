import Compromise as C
import Compromise: NondominatedSet, init_nondom_set!
import Compromise: @unpack

includet("TestProblems.jl")
TP = TestProblems
#%%

x = -4 .+ 8 .* rand(2, 500)
objf = x -> [ sum((x .- 1).^2); sum((x .+ 1).^2) ]
fx = mapreduce(objf, hcat, eachcol(x))
constr = x -> sum( (x .- 1.25).^2 ) - 0.5^2
theta = max.(0, map(constr, eachcol(x)))

ndset = NondominatedSet{Float64}(extra = Int[])
init_nondom_set!(ndset, eachcol(fx), theta)
rflags = ndset.extra
#%%
z = vcat(theta', fx)
_z = vcat(theta[rflags]', fx[:, rflags])
#%%
tp = TP._test_problem(Val(5), 2)
_mop = TP.to_mutable_mop(tp; max_func_calls=100)
X = _mop.lb .+ (_mop.ub .- _mop.lb) .* rand(2, 10)
algo_opts = C.AlgorithmOptions(; 
    nu_success=.8,
    max_iter=20, 
    stop_delta_min=1e-6,
    step_config = C.SteepestDescentConfig(;
        backtracking_mode = Val(:all)
    )
)

population2 = C.optimize_many(X, _mop; algo_opts);