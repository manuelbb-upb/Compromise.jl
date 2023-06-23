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
eval_op!(tmp_y, Val(UInt64(1)), tmp_x)
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
#%%
func2 = (_x, _p) -> cos(@show(_x[1]) + @show(_p[1]))
if length(CE.op_nodes(model))==1
    CE.@operator(model, ξ[1]=func2(x, p))
    @test ξ isa Vector{CE.StateNode}
    @test length(ξ) == 1
    @test length(CE.op_nodes(model)) == 2
end

if length(CE.op_nodes(model))==2
    ξold = ξ
    CE.@operator(model, ξ[1]=func2(x, p))
    @test ξold != ξ
    @test length(CE.op_nodes(model)) == 3
end

opp_node = last(CE.op_nodes(model))
tmp_y = ones(1)
tmp_x = zeros(1)
CE.eval_op!(tmp_y, Val(opp_node.dispatch_index), vcat(tmp_x, p.value))
@test tmp_y[1] == cos(π)