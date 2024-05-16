import Compromise as C
import Compromise: NondominatedSet, init_nondom_set!
import Compromise: @unpack

include("TestProblems.jl")
TP = TestProblems
#%%

x = -4 .+ 8 .* rand(2, 100)
objf = x -> [ sum((x .- 1).^2); sum((x .+ 1).^2) ]
fx = mapreduce(objf, hcat, eachcol(x))
constr = x -> sum( (x).^2 ) - 0.5^2
theta = max.(0, map(constr, eachcol(x)))

ndset = NondominatedSet(Float64, Int[])
init_nondom_set!(ndset, eachcol(fx), theta)
rflags = ndset.extra
#%%
z = vcat(theta', fx)
_z = vcat(theta[rflags]', fx[:, rflags])
#%%
mop = TP.to_mutable_mop(TP._test_problem(Val(6), 2))
X = mop.lb .+ (mop.ub .- mop.lb) .* rand(2, 50)
algo_opts = C.AlgorithmOptions(; max_iter=1)

C.optimize_set(X, mop; algo_opts);