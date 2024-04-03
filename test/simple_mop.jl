import Compromise as C
using Test

function test_simple_mop(
    @nospecialize(mop);
    n_vars,
    n_objfs,
    n_nl_eq,
    n_nl_ineq,
    n_lin_eq,
    n_lin_ineq,
    x0 = nothing,
    lb = nothing,
    ub = nothing,
    mcfg_objfs=C.ExactModelConfig(),
    mcfg_nl_eq=C.ExactModelConfig(),
    mcfg_nl_ineq=C.ExactModelConfig(),
)
    @test C.dim_vars(mop) == n_vars
    @test C.float_type(mop) == Float64

    @test C.dim_objectives(mop) == n_objfs 
    @test C.dim_nl_eq_constraints(mop) == n_nl_eq
    @test C.dim_nl_ineq_constraints(mop) == n_nl_ineq
    @test C.dim_lin_eq_constraints(mop) == n_lin_eq
    @test C.dim_lin_ineq_constraints(mop) == n_lin_ineq
    
    if iszero(n_objfs)
        @test isnothing(mop.objectives)
    end
    if iszero(n_nl_eq)
        @test isnothing(mop.nl_eq_constraints)
    end
    if iszero(n_nl_ineq)
        @test isnothing(mop.nl_ineq_constraints)
    end
    
    if iszero(n_lin_eq) 
        @test isnothing(mop.E) || size(mop.E, 1) == n_lin_eq
    else
        @test !isnothing(mop.E)
        @test !isnothing(mop.c)
    end
    if iszero(n_lin_ineq) 
        @test isnothing(mop.A) || size(mop.A, 1) == n_lin_eq
    else
        @test !isnothing(mop.A) 
        @test !isnothing(mop.b) 
    end

    if !isnothing(mop.E) && !isnothing(mop.c)
        @test size(mop.E, 1) == length(mop.c)
        @test length(mop.c) == n_lin_eq
    end

    if !isnothing(mop.A) && !isnothing(mop.b)
        @test size(mop.A, 1) == length(mop.b)
        @test length(mop.b) == n_lin_ineq
    end

    if isnothing(x0)
        @test isnothing(mop.x0)
    else
        @test x0 == mop.x0
    end
    if isnothing(lb)
        @test isnothing(mop.lb)
    else
        @test lb == mop.lb
    end
    if isnothing(ub)
        @test isnothing(mop.ub)
    else
        @test ub == mop.ub
    end

    if n_objfs > 0
        @test mop.mcfg_objectives == mcfg_objfs
    end
    if n_nl_eq > 0
        @test mop.mcfg_nl_eq_constraints == mcfg_nl_eq
    end
    if n_nl_ineq > 0
        @test mop.mcfg_nl_ineq_constraints == mcfg_nl_ineq
    end

    test_simple_cache(mop)
    return nothing
end

function test_simple_cache(@nospecialize(mop))
    cache = C.init_value_caches(mop)
    @test cache isa C.SimpleValueCache
    F = C.float_type(mop)
    @test C.float_type(cache) == F
    @test cache.x isa Vector{F}
    @test cache.ξ isa Vector{F}
    @test cache.fx isa Vector{F}
    @test cache.gx isa Vector{F}
    @test cache.hx isa Vector{F}
    @test cache.Ax isa Vector{F}
    @test cache.Ex isa Vector{F}
    @test cache.Ax_min_b isa Vector{F}
    @test cache.Ex_min_c isa Vector{F}
    @test cache.theta_ref isa Base.RefValue{F}
    @test cache.phi_ref isa Base.RefValue{F}

    @test isequal(C.cached_ξ(cache), cache.ξ)
    @test isequal(C.cached_x(cache), cache.x)
    @test isequal(C.cached_fx(cache), cache.fx)
    @test isequal(C.cached_gx(cache), cache.gx)
    @test isequal(C.cached_hx(cache), cache.hx)
    @test isequal(C.cached_Ex(cache), cache.Ex)
    @test isequal(C.cached_Ax(cache), cache.Ax)
    @test isequal(C.cached_Ex_min_c(cache), cache.Ex_min_c)
    @test isequal(C.cached_Ax_min_b(cache), cache.Ax_min_b)
    
    @test isequal(C.cached_theta(cache), cache.theta_ref[])
    C.cached_theta!(cache, 3.14)
    @test C.cached_theta(cache) == 3.14
    @test isequal(C.cached_theta(cache), cache.theta_ref[])
    
    @test isequal(C.cached_Phi(cache), cache.phi_ref[])
    C.cached_Phi!(cache, 3.14)
    @test C.cached_Phi(cache) == 3.14
    @test isequal(C.cached_Phi(cache), cache.phi_ref[])

    @test C.dim_vars(cache) == length(cache.x)
    @test C.dim_vars(cache) == length(cache.ξ)
    @test C.dim_vars(cache) == C.dim_vars(mop)

    @test C.dim_objectives(cache) == length(cache.fx)
    @test C.dim_objectives(cache) == C.dim_objectives(mop)
    
    @test C.dim_nl_eq_constraints(cache) == length(cache.hx)
    @test C.dim_nl_eq_constraints(cache) == C.dim_nl_eq_constraints(mop)
    
    @test C.dim_nl_ineq_constraints(cache) == length(cache.gx)
    @test C.dim_nl_ineq_constraints(cache) == C.dim_nl_ineq_constraints(mop)

end
#%%
let
n_vars = 1
mop = C.MutableMOP(n_vars)
test_simple_mop(
    mop;
    n_vars,
    n_objfs=0,
    n_nl_eq=0,
    n_nl_ineq=0,
    n_lin_eq=0,
    n_lin_ineq=0,
    x0=nothing,
    lb=nothing,
    ub=nothing
)
end
#%%
function constant_op(n_vars, n_out)
    if n_out <= 0
        return nothing
    else
        return C.NonlinearFunction(;
            func = x -> ones(n_out),
            func_iip = false,
            grads = x -> zeros(n_vars, n_out),
            grads_iip = false,
            hessians = x -> zeros(n_vars, n_vars, n_out),
            hessians_iip = false
        )
    end
end

function zero_lin_cons(n_vars, n_out)
    if n_out <= 0
        return nothing, nothing
    else
        return (zeros(n_out, n_vars), zeros(n_out))
    end
end
#%%

for (n_vars, n_objfs, n_nl_eq, n_nl_ineq, n_lin_eq, n_lin_ineq) in Iterators.product(
    1:5, 1:3, 0:2, 0:2, 0:2, 0:2
)
    for set_var_bounds in (false, true)
    for set_x0 in (false, true)

    if set_var_bounds
        ub = rand(n_vars)
        lb = -ub
    else
        ub = lb = nothing
    end
    x0 = if set_x0
        if set_var_bounds
            lb .+ (ub .- lb) .* rand(n_vars)
        else
            rand(n_vars)
        end
    else
        nothing
    end

    E, c = zero_lin_cons(n_vars, n_lin_eq)
    A, b = zero_lin_cons(n_vars, n_lin_ineq)

    mop = C.MutableMOP(;
        num_vars = n_vars,
        dim_objectives = n_objfs,
        dim_nl_eq_constraints = n_nl_eq,
        dim_nl_ineq_constraints = n_nl_ineq,
        objectives = constant_op(n_vars, n_objfs),
        nl_eq_constraints = constant_op(n_vars, n_nl_eq),
        nl_ineq_constraints = constant_op(n_vars, n_nl_ineq),
        x0, lb, ub, E, c, A, b
    )
    
    test_simple_mop(
        mop;
        n_vars, n_objfs, n_nl_eq, n_nl_ineq, 
        n_lin_eq, n_lin_ineq, x0, lb, ub,
    )

    _mop = C.initialize(mop)
    @test _mop isa C.TypedMOP
    test_simple_mop(
        _mop;
        n_vars, n_objfs, n_nl_eq, n_nl_ineq, 
        n_lin_eq, n_lin_ineq, x0, lb, ub,
    )

    mop = C.MutableMOP(n_vars)
    mop.x0 = x0
    mop.lb = lb
    mop.ub = ub
    C.add_objectives!(mop, constant_op(n_vars, n_objfs), :rbf; dim_out=n_objfs)
    C.add_nl_eq_constraints!(mop, constant_op(n_vars, n_nl_eq), :rbf; dim_out=n_nl_eq)
    C.add_nl_ineq_constraints!(mop, constant_op(n_vars, n_nl_ineq), :rbf; dim_out=n_nl_ineq)
    mop.A = A
    mop.b = b
    mop.E = E
    mop.c = c

    test_simple_mop(
        mop;
        n_vars, n_objfs, n_nl_eq, n_nl_ineq, 
        n_lin_eq, n_lin_ineq, x0, lb, ub,
        mcfg_objfs = C.RBFConfig(),
        mcfg_nl_eq = C.RBFConfig(),
        mcfg_nl_ineq = C.RBFConfig(),
    )

    _mop = C.initialize(mop)
    @test _mop isa C.TypedMOP
    test_simple_mop(
        _mop;
        n_vars, n_objfs, n_nl_eq, n_nl_ineq, 
        n_lin_eq, n_lin_ineq, x0, lb, ub,
        mcfg_objfs = C.RBFConfig(),
        mcfg_nl_eq = C.RBFConfig(),
        mcfg_nl_ineq = C.RBFConfig(),
    ) 
end
end
end

# TODO test surrogate and cache