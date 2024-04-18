import Compromise: solve_steepest_descent_problem
import Compromise: DEFAULT_QP_OPTIMIZER
using Test
import LinearAlgebra as LA
#%%
lb = nothing
ub = nothing
E = nothing
c = nothing
A = nothing
b = nothing
Axn = zeros(0)
Dgx_n = zeros(0)
Dgx = zeros(2, 0)
Dhx = zeros(2, 0)
normalize_gradients=false
qp_opt = DEFAULT_QP_OPTIMIZER
ball_norm = Inf

Dfx = [ 
    0;
    -1;;
]
xn = zeros(2)
χ, d = solve_steepest_descent_problem(
    xn, Dfx, lb, ub, E, Axn, A, b, Dhx, Dgx_n, Dgx, qp_opt; 
    normalize_gradients, ball_norm
)
@test χ ≈ 1
@test d[2] ≈ 1
@test LA.norm(d, Inf) ≈ 1
#%%
Dfx = [
    -10      10;
    -1      -1
]
χ, d = solve_steepest_descent_problem(
    xn, Dfx, lb, ub, E, Axn, A, b, Dhx, Dgx_n, Dgx, qp_opt; 
    normalize_gradients=false, ball_norm
)
@test LA.norm(d, Inf) ≈ 1
@test χ ≈ 1

_χ, d = solve_steepest_descent_problem(
    xn, Dfx, lb, ub, E, Axn, A, b, Dhx, Dgx_n, Dgx, qp_opt; 
    normalize_gradients=true, ball_norm
)
@test abs(_χ - χ/10) < 1e-2

#%%
Dfx = [0 -1]'
# Axn[1] + d[1] <= -0.5
Axn = [0,]
A = [1 0]
b = [-0.5]

χ, d = solve_steepest_descent_problem(
    xn, Dfx, lb, ub, E, Axn, A, b, Dhx, Dgx_n, Dgx, qp_opt; 
    normalize_gradients=false,  ball_norm
)
@test d[1] <= -0.5 + 1e-5
#%%
Axn = zeros(0)
A = b = nothing
Dgx_n = [0.5,]
Dgx = [1 0]'
χ, d = solve_steepest_descent_problem(
    xn, Dfx, lb, ub, E, Axn, A, b, Dhx, Dgx_n, Dgx, qp_opt; 
    normalize_gradients=false, ball_norm
)
@test d[1] <= -0.5 + 1e-5
#%%
Axn = [-.5]
A = [0 1]
b = [-.5]
χ, d = solve_steepest_descent_problem(
    xn, Dfx, lb, ub, E, Axn, A, b, Dhx, Dgx_n, Dgx, qp_opt; 
    normalize_gradients=false,  ball_norm
)
@test abs(χ) <= 1e-6
#%%
Dgx_n = [-0.75, 0.2]
Dgx = [
    1   0
    0   1
]
χ, d = solve_steepest_descent_problem(
    xn, Dfx, lb, ub, E, Axn, A, b, Dhx, Dgx_n, Dgx, qp_opt; 
    normalize_gradients=false, ball_norm
)