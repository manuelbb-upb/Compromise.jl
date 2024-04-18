using Compromise
using Test
includet("TestProblems.jl")
const TP = TestProblems
const C = Compromise

#%%
ensure_matrix(A, mop)=A
ensure_matrix(::Nothing, mop)=Matrix{C.float_type(mop)}(undef, 0, C.dim_vars(mop))
ensure_vector(b, mop)=b
ensure_vector(::Nothing, mop)=Vector{C.float_type(mop)}(undef, 0)
#%%
simplex_mops = TP.all_simplex_mops(3)
for (i,mop) in enumerate(simplex_mops)

    rmop = C.RestorationMOP(mop, Inf)

    @test C.dim_vars(rmop) == C.dim_vars(mop) + 1
    @test C.dim_nl_ineq_constraints(rmop) == C.dim_nl_ineq_constraints(mop) + 2 * C.dim_nl_eq_constraints(mop)
    @test C.dim_lin_ineq_constraints(rmop) == C.dim_lin_ineq_constraints(mop) + 2 * C.dim_lin_eq_constraints(mop)

    E = ensure_matrix(C.lin_eq_constraints_matrix(mop), mop)
    A = ensure_matrix(C.lin_ineq_constraints_matrix(mop), mop)
    c = ensure_vector(C.lin_eq_constraints_vector(mop), mop)
    b = ensure_vector(C.lin_ineq_constraints_vector(mop), mop)

    G = ensure_matrix(C.lin_ineq_constraints_matrix(rmop), rmop)
    g = ensure_vector(C.lin_ineq_constraints_vector(rmop), rmop)

    __G = vcat(E, -E, A)
    _G = hcat(fill(-1, size(__G, 1)), __G) 

    _g = vcat(c, -c, b)
    @test G ≈ _G
    @test g ≈ _g

end

mop = last(simplex_mops)
v = C.init_vals(mop)
v.x .= rand(C.dim_vars(mop))

lincons = C.init_lin_cons(mop)
scaler = C.init_scaler(:box, lincons)
C.eval_mop!(v, mop, scaler)

theta_k = v.theta_ref[]
rmop = C.RestorationMOP(rmop, theta_k)
rv = C.init_vals(rmop)
rv.x[1] = v.theta_ref[]
rv.x[2:end] .= v.x

C.eval_mop!(rv, rmop)

C.eval_objectives!(rv.fx, rmop, rv.x)