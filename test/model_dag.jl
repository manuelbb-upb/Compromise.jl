using Test
import Compromise as C
CE = C.CompromiseEvaluators

#%%
model = CE.Model()

x = CE.add_variable!(model)

@test x isa CE.VariableNode
@test x.index == 1
@test length(CE.var_nodes(model)) == 1

function CE.eval_op!(y, ::Val{UInt64(1)}, x)
    y[1] = sin(x[1])
    return nothing
end

tmp_y = ones(1)
tmp_x = [π]
CE.eval_op!(tmp_y, Val(UInt64(1)), tmp_x)
@test iszero(tmp_y[1])

s = CE.add_operator!(model, 1, x, nothing; dim_out = 1)
@test s isa Vector{CE.StateNode}
@test length(s) == 1
@test length(CE.state_nodes(model)) == 1
@test only(s) == only(CE.state_nodes(model))
@test length(CE.op_nodes(model)) == 1

op_node = only(CE.op_nodes(model))
@test op_node.dispatch_index == 1

# check edges:
@test only(op_node.in_nodes) == x
@test only(op_node.out_nodes) == only(s)
@test only(x.out_nodes) == op_node
@test only(only(s).in_nodes) == op_node

p = CE.add_parameter!(model, π)
@test only(CE.param_nodes(model)) == p

func2 = (_x, _p) -> cos(@show(_x[1]) + @show(_p[1]))
CE.@operator(model, ξ[1]=func2(x, p))
@test ξ isa Vector{CE.StateNode}
@test length(ξ) == 1
@test length(CE.op_nodes(model)) == 2

ξold = ξ
CE.@operator(model, ξ[1]=func2(x, p))
@test ξold != ξ
@test length(CE.op_nodes(model)) == 3

opp_node = last(CE.op_nodes(model))
tmp_y = ones(1)
tmp_x = zeros(1)
CE.eval_op!(tmp_y, Val(opp_node.dispatch_index), vcat(tmp_x, p.value))
@test tmp_y[1] == cos(π)
#%%
dag = CE.initialize(model; check_cycles=true)

for n in model.nodes
    j = n.array_index
    i = dag.dag_indices[j]
    m = dag.nodes[i]
    @test i == m.array_index
    @test j == m.src_index
end

@test length(dag.primals) == 5  # 1 var + 3 states + 1 param (in that order)
@test all(isnan.(dag.primals[1:end-1]))
@test last(dag.primals) ≈ π

x0 = [1.2,]
CE.eval_node(dag, s[1], x0; prepare_grads=false)

@test dag.primals[1] ≈ 1.2
@test dag.primals[2] ≈ sin(1.2)
@test dag.primals[5] ≈ π
@test isnan(dag.primals[3])
@test isnan(dag.primals[4])

@test length(dag.x_hash) == 4 # 1 var + 3 states
@test dag.x_hash[1] == dag.x_hash[2]
@test dag.x_hash[1] != dag.x_hash[3]
@test dag.x_hash[3] == dag.x_hash[4] == 0

CE.eval_node(dag, ξold[1], x0; prepare_grads=false)
@test dag.primals[1] ≈ 1.2
@test dag.primals[2] ≈ sin(1.2)
@test dag.primals[3] ≈ cos(1.2 + π)
@test isnan(dag.primals[4])
@test dag.primals[5] ≈ π

@test dag.x_hash[1] == dag.x_hash[3]
@test dag.x_hash[1] != dag.x_hash[4]

CE.eval_node(dag, ξ[1], x0; prepare_grads=false)
@test dag.primals[1] ≈ 1.2
@test dag.primals[2] ≈ sin(1.2)
@test dag.primals[3] ≈ cos(1.2 + π)
@test dag.primals[4] ≈ dag.primals[3]
@test dag.primals[5] ≈ π
@test all(map(isequal(dag.x_hash[1]), dag.x_hash))

dag.primals .= 0
CE.eval_node(dag, ξ[1], x0; prepare_grads=false)
@test all(iszero.(dag.primals))     # no values computed because of `dag.x_hash`
dag.x_hash .= 0
CE.eval_node(dag, ξ[1], x0; prepare_grads=false)
@test dag.primals[4] ≈ cos(1.2)     # parameter was reset to 0 by `dag.primals .= 0`
#%%
dag.x_hash .= 0
@test_throws Exception CE.eval_node(dag, ξ[1], x0)

xdag = dag.nodes[dag.dag_indices[x.array_index]]
ξdag = dag.nodes[dag.dag_indices[ξ[1].array_index]]
opn = only(CE.predecessors(dag, ξdag))

function CE.eval_grads!(Dy, ::Val{opn.special_index}, x_and_p)
    @assert size(Dy) == (1, 1)
    Dy[1] = -sin(x_and_p[1])
    return nothing
end

dag.x_hash .= 0
CE.eval_node(dag, ξ[1], x0)
@test dag.partials[xdag.array_index, ξdag.array_index] ≈ -sin(x0[end])

#%% example from wikipedia
wikimod = CE.Model()
w1 = CE.add_variable!(wikimod)
w2 = CE.add_variable!(wikimod)

function CE.eval_op!(y, ::Val{UInt64(3)}, x)
    y[1] = prod(x)
    return nothing
end
function CE.eval_grads!(Dy, ::Val{UInt64(3)}, x)
    @assert size(Dy) == (2,1)
    Dy[1] = x[2]
    Dy[2] = x[1]
    return nothing
end
w3 = only(CE.add_operator!(wikimod, 3, [w1, w2], nothing; dim_out = 1))

function CE.eval_op!(y, ::Val{UInt64(4)}, x)
    y[1] = sin(only(x))
    return nothing
end
function CE.eval_grads!(Dy, ::Val{UInt64(4)}, x)
    @assert size(Dy) == (1,1)
    Dy[1] = cos(only(x))
    return nothing
end
w4 = only(CE.add_operator!(wikimod, 4, w1, nothing; dim_out = 1))

function CE.eval_op!(y, ::Val{UInt64(5)}, x)
    y[1] = sum(x)
    return nothing
end
function CE.eval_grads!(Dy, ::Val{UInt64(5)}, x)
    @assert size(Dy) == (2,1)
    Dy[:] .= 1
    return nothing
end
w5 = only(CE.add_operator!(wikimod, 5, [w3,w4], nothing; dim_out = 1))

wikidag = CE.initialize(wikimod)
CE.eval_node(wikidag, w5, [1.5, 2.5])

@test wikidag.primals[5] ≈ sin(1.5) + 1.5*2.5
CE.pullback_node(wikidag, w5)
@test wikidag.partials[1, 5] ≈ cos(1.5) + 2.5
@test wikidag.partials[2, 5] ≈ 1.5

#%% Rosenbrock for Hessians

rosmod = CE.Model()
x1 = CE.add_variable!(rosmod)
x2 = CE.add_variable!(rosmod)

# g1 = x1
function CE.eval_op!(y, ::Val{UInt64(11)}, x)
    y[1] = x[1]
    return nothing
end
function CE.eval_grads!(Dy, ::Val{UInt64(11)}, x)
    Dy[1,1] = 1
    return nothing
end
function CE.eval_hessians!(H, ::Val{UInt64(11)}, x)
    H[1,1,1] = 0
    return nothing
end
g1 = CE.add_operator!(rosmod, 11, x1, nothing; dim_out = 1)

# g2 = x1^2 - x2
function CE.eval_op!(y, ::Val{UInt64(12)}, x)
    y[1] = x[1]^2 - x[2]
    return nothing
end
function CE.eval_grads!(Dy, ::Val{UInt64(12)}, x)
    Dy[1,1] = 2*x[1]
    Dy[2,1] = -1
    return nothing
end
function CE.eval_hessians!(H, ::Val{UInt64(12)}, x)
    H[1,1,1] = 2
    H[2,1,1] = 0
    return nothing
end
g2 = CE.add_operator!(rosmod, 12, [x1; x2], nothing; dim_out = 1)


# f(g1, g2) = (1-g1)^2 + 100*g2^2
function CE.eval_op!(y, ::Val{UInt64(13)}, g)
    y[1] = (1-g[1])^2 + 100*g[2]^2
    return nothing
end
function CE.eval_grads!(Dy, ::Val{UInt64(13)}, g)
    Dy[1,1] = -2*(1-g[1])
    Dy[2,1] = 200*g[2]
    return nothing
end
function CE.eval_hessians!(H, ::Val{UInt64(13)}, g)
    H[1,1,1] = 2
    H[2,1,1] = 200
    return nothing
end

y = CE.add_operator!(rosmod, 13, [g1; g2], nothing; dim_out = 1)[end]

rosdag = CE.initialize(rosmod)

x0 = [ℯ, -π]
CE.eval_node(rosdag, y, x0)

x1i = rosdag.dag_indices[x1.array_index]
x2i = rosdag.dag_indices[x2.array_index]
yi = rosdag.dag_indices[y.array_index]

@test rosdag.primals[yi] ≈ (1-x0[1])^2 + 100*(x0[1]^2 - x0[2])^2

CE.pullback_node(rosdag, y)
dy1 = -400*(x0[2]-x0[1]^2)*x0[1] -2*(1-x0[1])
dy2 = 200*(x0[2]-x0[1]^2)
@test rosdag.partials[x1i, yi] ≈ dy1 
@test rosdag.partials[x2i, yi] ≈ dy2

CE.forward_hessian(rosdag, y)