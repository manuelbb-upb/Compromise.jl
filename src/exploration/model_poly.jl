using Test
using ForwardDiff
import Compromise as C
CE = C.CompromiseEvaluators
#%%

mod = CE.Model()
x = CE.add_variable!(mod)

opA_primal_counter = 0
opA_grads_counter = 0
opA = CE.WrappedFunction(;
    func= (x,p) -> begin global opA_primal_counter += 1; only(x)^2 end, func_iip=false, 
    grads=(x,p) -> begin global opA_grads_counter += 1; [2*x;;] end, grads_iip=false
)
yA = zeros(1)
DyA = zeros(1,1)
CE.eval_op_and_grads!(yA, DyA, opA, [0.5,], [])
@test only(yA) == 0.25
@test only(DyA) == 1
@test opA_primal_counter == opA_grads_counter == 1

y = CE.add_operator!(mod, opA, x, nothing; dim_out=1) |> only
opB = CE.WrappedFunction(;
    func=(x,p) -> sum(x), grads=(x, p) -> fill(1, length(x), 1), func_iip=false, grads_iip=false)

z = CE.add_operator!(mod, opB, [x,y], nothing; dim_out=1) |> only
#%%
opA_primal_counter = 1
opA_grads_counter = 1
dag = CE.initialize(mod)

CE.forward_node(dag, z, [0.5,])
@assert dag.primals == [0.5, 0.25, 0.75]
@assert all(iszero, dag.partials)
@test opA_primal_counter == 2
@test opA_grads_counter == 1
# does hashing and early breaking work?
CE.forward_node(dag, z, [0.5,])
@test opA_primal_counter == 2
CE.forward_node(dag, z, [0.25,])
@test opA_primal_counter == 3
CE.forward_node(dag, z, [0.25,]; check_fw_hashes=false)
@test opA_primal_counter == 4
CE.forward_node(dag, z, [0.25,]; prepare_grads=true)
@test opA_primal_counter == 5   # if gradients are required, we automatically do `eval_op_and_grads!`
@test opA_grads_counter == 2
#%%
CE.reset!(dag)
opA_primal_counter = 4
opA_grads_counter = 2

CE.forward_node(dag, z, [0.5,]; prepare_grads=true)
@assert dag.primals == [0.5, 0.25, 0.75]
@assert dag.partials == [
    0 1 1;
    0 0 1;
    0 0 0;
]
@test opA_primal_counter == 5
@test opA_grads_counter == 3

# again, due to our forward hash tree, calling with the same input should be a “no-op”:
CE.forward_node(dag, z, [0.5,]; prepare_grads=true)
@test opA_primal_counter == 5
@test opA_grads_counter == 3

CE.forward_node(dag, z, [0.25,]; prepare_grads=true)
@test opA_primal_counter == 6
@test opA_grads_counter == 4

#%%
CE.reset!(dag)
CE.forward_node(dag, z, [0.25,]; prepare_grads=true)

opA_primal_counter = 6
opA_grads_counter = 4

old_dzdx_hash = dag.grads_hashes[1, 3]
CE.pullback_node(dag, z)

@test opA_primal_counter == 6
@test opA_grads_counter == 4
# derivative of z = x + y = x + x^2 at x = 0.25 is dz/dx = 1 + 2*0.25 = 1.5
@test dag.partials[1, 3] == 1
@test dag.grads[1, 3] == 1.5
@test old_dzdx_hash != dag.grads_hashes[1, 3]

new_old_dzdx_hash = dag.grads_hashes[1, 3]
# messing with gradient values without resetting, `dx_hash`.
# if hashing works as intended, values should not change after pullback despite being wrong:
dag.partials[1,3] = 1
CE.pullback_node(dag, z)
@test dag.partials[1, 3] == 1
@test dag.grads[1, 3] == 1.5
@test new_old_dzdx_hash == dag.grads_hashes[1,3]

CE.pullback_node(dag, x)
@test dag.partials[1,1] == 0
@test dag.grads[1,1] == 1
CE.pullback_node(dag, y)
@test dag.partials[:,2] == [0.5, 0, 0]
@test dag.grads[:,2] == [0.5, 1.0, 0]

#%%
CE.reset!(dag)
CE.forward_node(dag, z, [0.125,]; prepare_grads=true)
CE.pullback_node(dag, x)
CE.pullback_node(dag, y)
CE.pullback_node(dag, z)