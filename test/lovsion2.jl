using Compromise
import ForwardDiff as FD
using Test
#let
begin
mop = MutableMOP(; 
    num_vars=2,
    lb = fill(-.5, 2),
    ub = [0.0, 0.5]
)
objf!(y, x) = begin
    y[1] = x[2]
    y[2] = - (x[2]-x[1]^3)/(x[1]+1)
    nothing
end
add_objectives!(
    mop, objf!, :rbf; 
    dim_out=2, func_iip=true, 
    #backend=ForwardDiffBackend()
)

algo_opts = AlgorithmOptions(;
    eps_crit=1e-3,
    stop_delta_min=1e-12, 
    stop_max_crit_loops=10,
)
x0 = [
    -0.125,
    -0.3888888889,
]
#r = optimize(mop, x0; algo_opts=AlgorithmOptions(stop_max_crit_loops=2))
end