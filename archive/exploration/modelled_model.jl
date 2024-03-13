using Test
import Compromise as C
CE = C.CompromiseEvaluators

#%%
model = CE.Model()

x = CE.add_variable!(model)

@test x isa CE.VariableNode
@test x.index == 1
@test length(CE.var_nodes(model)) == 1

op1 = CE.NonlinearParametricFunction(; func = (y, x, p) -> y[1] = sin(x[1]), func_iip=true)

tmp_y = ones(1)
tmp_x = [π]
CE.eval_op!(tmp_y, op1, tmp_x, nothing)
@test iszero(tmp_y[1])

s = CE.add_operator!(model, op1, x, nothing; dim_out = 1)
@test s isa Vector{CE.StateNode}
@test length(s) == 1
@test length(CE.state_nodes(model)) == 1
@test only(s) == only(CE.state_nodes(model))
@test length(CE.op_nodes(model)) == 1

op_node = only(CE.op_nodes(model))

# check edges:
@test only(op_node.in_nodes) == x
@test only(op_node.out_nodes) == only(s)
@test only(x.out_nodes) == op_node
@test only(only(s).in_nodes) == op_node

p = CE.add_parameter!(model, π)
@test only(CE.param_nodes(model)) == p

func2 = (_x, _p) -> cos(@show(_x[1]) + @show(_p[1]))
op2 = CE.NonlinearParametricFunction(;func=func2, func_iip=false)
ξ = CE.add_state!(model)
CE.add_operator!(model, op2, x, p, ξ)
@test ξ isa CE.StateNode
@test length(CE.op_nodes(model)) == 2
#%%

opp_node = last(CE.op_nodes(model))
tmp_y = ones(1)
tmp_x = zeros(1)
CE.eval_op!(tmp_y, opp_node.operator, tmp_x, p.value)
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

@test length(dag.primals) == 4  # 1 var + 2 states + 1 param (in that order)
@test all(isnan.(dag.primals[1:end-1]))
@test last(dag.primals) ≈ π

x0 = [1.2,]
CE.eval_node(dag, s[1], x0; prepare_grads=false)

@test dag.primals[1] ≈ 1.2
@test dag.primals[2] ≈ sin(1.2)
@test dag.primals[4] ≈ π
@test isnan(dag.primals[3])

@test length(dag.x_hash) == 3 # 1 var + 2 states
@test dag.x_hash[1] == dag.x_hash[2]
@test dag.x_hash[1] != dag.x_hash[3]
@test dag.x_hash[3] == 0

CE.eval_node(dag, ξ, x0; prepare_grads=false)
@test dag.primals[1] ≈ 1.2
@test dag.primals[2] ≈ sin(1.2)
@test dag.primals[3] ≈ cos(1.2 + π)
@test dag.primals[4] ≈ π
@test all(map(isequal(dag.x_hash[1]), dag.x_hash))

dag.primals .= 0
CE.eval_node(dag, ξ, x0; prepare_grads=false)
@test all(iszero.(dag.primals))     # no values computed because of `dag.x_hash`
dag.x_hash .= 0
CE.eval_node(dag, ξ, x0; prepare_grads=false)
@test dag.primals[3] ≈ cos(1.2)     # parameter was reset to 0 by `dag.primals .= 0`
#%%
dag.x_hash .= 0
@test_throws Exception CE.eval_node(dag, ξ[1], x0)

xdag = dag.nodes[dag.dag_indices[x.array_index]]
ξdag = dag.nodes[dag.dag_indices[ξ.array_index]]
opn = only(CE.predecessors(dag, ξdag))
op = dag.operators[opn.special_index]
function CE.eval_op_and_grads!(y, Dy, op::typeof(op), x, p)
    CE.eval_op!(y, op, x, p)
    @assert size(Dy) == (1, 1)
    Dy[1] = -sin(x[1])
    return nothing
end

CE.reset_hashes!(dag)
CE.eval_node(dag, ξ, x0; prepare_grads=true)
@test dag.partials[xdag.array_index, ξdag.array_index] ≈ -sin(x0[end])

function CE.eval_op_and_grads_and_hessians!(y, Dy, H, op::typeof(op), x, p)
    CE.eval_op_and_grads!(y, Dy, op, x, p)
    H[1,1,1] = -cos(x[1])
    return nothing
end

CE.reset_hashes!(dag)
CE.eval_node(dag, ξ, x0; prepare_grads=true, prepare_hessians=true)
@test dag.partials2[1,1,dag.dag_indices[ξ.array_index]] ≈ -cos(x0[1])
#%%
using ForwardDiff

wikimod = CE.Model()
w1 = CE.add_variable!(wikimod)
w2 = CE.add_variable!(wikimod)

w3_op = CE.NonlinearParametricFunction(; func=(x,p) -> prod(x), func_iip=false, backend=CE.ForwardDiffBackend())

w3_y = zeros(1)
CE.eval_op!(w3_y, w3_op, [1, 2], [])
@test only(w3_y) == 2

w3_Dy = zeros(2, 1)
CE.eval_op_and_grads!(w3_y, w3_Dy, w3_op, [2, 3], Float64[])
@test only(w3_y) == 6
@test w3_Dy == [3; 2;;]

w3_H = zeros(2, 2, 1)
CE.eval_op_and_grads_and_hessians!(w3_y, w3_Dy, w3_H, w3_op, [2, 3], Float64[])
@test w3_H[:, :, 1] == [0 1; 1 0]

w3 = only(CE.add_operator!(wikimod, w3_op, [w1, w2], nothing; dim_out = 1))

w4_op = CE.NonlinearParametricFunction(; func = (y, x, p) -> y[1] = sin(only(x)), func_iip=true, backend=CE.ForwardDiffBackend())

w4_y = zeros(1)
CE.eval_op!(w4_y, w4_op, [1], [])
@test only(w4_y) ≈ sin(1)
w4_Dy = zeros(1, 1)
CE.eval_op_and_grads!(w4_y, w4_Dy, w4_op, [1], [])
@test only(w4_Dy) ≈ cos(1)
w4_Hy = zeros(1, 1, 1)
CE.eval_op_and_grads_and_hessians!(w4_y, w4_Dy, w4_Hy, w4_op, [1], [])
@test only(w4_Hy) ≈ -sin(1)

w4 = only(CE.add_operator!(wikimod, w4_op, [w1], nothing; dim_out = 1))

w5_op = CE.NonlinearParametricFunction(; func=(x, p) -> sum(x), func_iip=false, backend=CE.ForwardDiffBackend())
w5 = only(CE.add_operator!(wikimod, w5_op, [w3, w4], nothing; dim_out = 1))
#%%
wikidag = CE.initialize(wikimod)
CE.eval_node(wikidag, w5, [1.5, 2.5])

@test wikidag.primals[5] ≈ sin(1.5) + 1.5*2.5
@test_throws Exception CE.pullback_node(wikidag, w5)

CE.eval_node(wikidag, w5, [1.5, 2.5]; prepare_grads=true)
CE.pullback_node(wikidag, w5)
@test wikidag.partials[1, 5] ≈ cos(1.5) + 2.5
@test wikidag.partials[2, 5] ≈ 1.5
