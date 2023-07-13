module CompromiseEvaluators #src
using Requires  #src
function __init__() #src
    @require ForwardDiff="f6369f11-7733-5829-9624-2563aa707210" include("forwarddiff_backend.jl") #src
end #src

# # Introduction
# Before anything else, load some helpers:

include("utils.jl")

# This module defines tools to deal with evaluation and differentiation of multiple related
# multi-input multi-output (MIMO) functions.
# We can vizualize the computations for a MIMO function as a directed acyclic graph (DAG).
# Multiple MIMO functions might share subexpressions, in which case their values are leaves
# of a common DAG. Otherwise, we have DAG with disjoint components, which is just fine 
# for our use case.
# There are two DAG types in this module, and both are influenced by structures in 
# **MathOptInterface** (and possibly JuMP).
# First, `Model` is what the user creates interactively to set up the optimization problem.
# Second, at the beginning of the optimization procedure, a `Model` is analyzed and 
# transformed into a `DAG`, which allows for performant (and cached) evaluation of value
# nodes, as well as differentiation.

# ## The `Model`
# To setup a model and visualize the flow of computation, we distinguish two classes of 
# nodes in a DAG: (scalar) value nodes, which represent a single real number, and 
# operator nodes, which take some value nodes in and fill some outgoing value nodes.
# We can setup a type hierachy for this:
abstract type AbstractNode end
abstract type AbstractValueNode <: AbstractNode end
abstract type AbstractOperatorNode <: AbstractNode end

# ### Value Nodes
# In our modeling framework, we allow for 3 types of scalar value nodes: 
# * variables (function inputs)
# * parameters (similar to variables, but usually constant in certain loops)
# * states, which is our shorthand term for dependent variables.
# Each operator node introduces unique states, and each state can only have a single 
# preceeding operator node.

# Each `AbstractValueNode` has a common field `array_index`, which defaults to nothing.
# It is set by problem construction or mutation methods and allows for retrieval of `Model`
# nodes in the processed `DAG`.
# The fields `in_nodes` and `out_nodes` store relations to other nodes.

## generic getter methods
in_nodes(n::AbstractNode)=getfield(n, :in_nodes)
out_nodes(n::AbstractNode)=getfield(n, :out_nodes)

import Parameters: @with_kw_noshow, @with_kw

@with_kw_noshow mutable struct VariableNode <: AbstractValueNode
	index :: Int
	array_index :: Union{Int, Nothing} = nothing
    out_nodes :: Vector{<:AbstractNode} = AbstractNode[]
end
"""
    VariableNode(i::Int)

A model variable with index `i`. The index has no special meaning except to sort and 
distinguish variables.
"""
VariableNode(i::Int)=VariableNode(;index=i)
## minimal printing
Base.show(io::IO, node::VariableNode)=print(io, "Variable($(node.index))")
## custom getter methods
in_nodes(n::VariableNode)=AbstractNode[]

# Parameter nodes are very similar to variable nodes except that they store a value:
@with_kw_noshow mutable struct ParameterNode{R<:Real} <: AbstractValueNode
	index :: Int
    value :: R
	array_index :: Union{Int, Nothing} = nothing
    out_nodes :: Vector{<:AbstractNode} = AbstractNode[]
end
"""
    ParameterNode(i::Int, val::Real)

A model parameter with index `i` and initial value `val`. 
The index has no special meaning except to sort and distinguish parameters.
"""
ParameterNode(i::Int, val::Real)=ParameterNode(;index=i, value=val)
## minimal printing
Base.show(io::IO, node::ParameterNode)=print(io, "Parameter($(node.index)){$(node.value)}")
## custom getter methods
in_nodes(n::ParameterNode)=AbstractNode[]

# State nodes can actually have ingoing edges:
@with_kw_noshow mutable struct StateNode <: AbstractValueNode
	index :: Int
	array_index :: Union{Int, Nothing} = nothing
    in_nodes :: Vector{<:AbstractNode} = AbstractNode[]
    out_nodes :: Vector{<:AbstractNode} = AbstractNode[]
end

"""
    StateNode(i::Int)

A dependent variable of a model, with index `i`. 
The index has no special meaning except to sort and distinguish states.
"""
StateNode(i::Int) = StateNode(;index=i)
## minimal printing
Base.show(io::IO, node::StateNode)=print(io, "State($(node.index))")

function Base.show(io::IO, nodes::AbstractVector{<:AbstractNode})
    print(io, "[" * join((repr(n) for n in nodes), ", ") * "]")
end

# Helper functions for constructing the DAG later on, to determine if a scalar node is 
# can be “dependent” (`VariableNode` or `StateNode`):
is_scalar_node(::AbstractNode) = false
is_scalar_node(::AbstractValueNode) = true
is_dep_node(::AbstractNode) = false
is_dep_node(::Union{VariableNode, StateNode}) = true

# ### Operator Nodes
# At the moment, there are two types of operator nodes:
# * the `NonlinearOperatorNode`, to setup a problem and perform computations,
# * and the `SurrogateModelNode`, used to indicate that a sub-graph is meant to be modelled with  
#   surrogate models during optimization.

# The `SurrogateModelNode` type has an `index` for distinction, and references an 
# `AbstractSurrogateModel` as described in the included file:
include("surrogate_models.jl")

@with_kw_noshow mutable struct SurrogateModelNode <: AbstractOperatorNode
    index :: Int
    array_index :: Union{Int, Nothing} = nothing
    
    surrogate :: Any

    in_nodes :: Vector{<:AbstractNode} = AbstractNode[]
    out_nodes :: Vector{<:AbstractNode} = AbstractNode[]

    num_in :: Int = length(in_nodes)
    num_out :: Int = length(out_nodes) 
end

# For now, let's rather focus on the `NonlinearOperatorNode`.
# Like the value nodes, it has an `index` for distinction, and an `array_index` for
# retrieval in the processed `DAG`.
# It also references some `AbstractNonlinearOperator` (imported from the file 
# `nonlinear_operators.jl`) and metadata:

include("nonlinear_operators.jl")

@with_kw_noshow mutable struct NonlinearOperatorNode <: AbstractOperatorNode
	index :: Int
	array_index :: Union{Int, Nothing} = nothing

    operator :: Any

    in_nodes :: Vector{<:AbstractNode} = AbstractNode[]
    out_nodes :: Vector{<:AbstractNode} = AbstractNode[]

    num_deps :: Int = count(n -> n isa Union{VariableNode, StateNode}, in_nodes)
    num_params :: Int = length(in_nodes) - num_deps
    num_out :: Int = length(out_nodes) 
end

## minimal printing
function Base.show(io::IO, node::NonlinearOperatorNode)
    print(io, "Operator{($(node.num_deps),$(node.num_params))->$(node.num_out)}($(node.index))")
end

# ### `Model` Type

# The model is composed successively by the user.
# There is an array for nodes of all types and dedicated arrays storing integer 
# indices according to node type.

@with_kw_noshow struct Model
    nodes :: Vector{AbstractNode} = []    
	var_indices :: Vector{Int} = []
    param_indices :: Vector{Int} = []
	state_indices :: Vector{Int} = []
	op_indices :: Vector{Int} = []
    model_indices :: Vector{Int} = []
end

# These are some helpers to get sub-vectors of nodes of the same type:
var_nodes(model::Model) = view(model.nodes, model.var_indices)
param_nodes(model::Model) = view(model.nodes, model.param_indices)
state_nodes(model::Model) = view(model.nodes, model.state_indices)
op_nodes(model::Model) = view(model.nodes, model.op_indices)
model_nodes(model::Model) = view(model.nodes, model.model_indices)

# To setup a model, the file "model_helpers.jl" has some utility functions,
# like `add_variable!`, `add_state!`, `add_parameter!` and `add_operator!`:
include("model_helpers.jl") # TODO re-enable

# ## Evaluation Graph `DAG`
# We could already use the `Model` to evaluate MIMO functions.
# That would be slow and complicated.
# We can at least work on the “slow” issue.
# The considerations are similar to those in 
# [MathOptInterface's Nonlinear Design](http://jump.dev/MathOptInterface.jl/stable/submodules/Nonlinear/overview/#Expression-graph-representation).

# Internally, our graph uses nodes of type `MetaNode`.
# A `MetaNode` only stores integer values, the meaning of which depends on the 
# type of node it shadows:
@enum META_NODE_TYPE::UInt8 begin
    META_NODE_VAR   = 1
    META_NODE_STATE = 2
    META_NODE_PARAM = 3
    META_NODE_OP    = 4
    META_NODE_MOD   = 5
end

# The `array_index` of a node `n::MetaNode` indicates its position in a list of nodes of 
# the DAG.
struct MetaNode
	ntype :: META_NODE_TYPE
	array_index :: Int      # index in `nodes::Vector{MetaNode}` of `DAG`
    src_index :: Int        # array index of source node in model
	special_index :: Int    # functional index into sub-arrays in `DAG`
    num_deps :: Int         # number of non-parameter inputs
end

## helpers for construction:
is_dep_node(meta_node) = UInt8(meta_node.ntype) <= 2
is_scalar_node(meta_node) = UInt8(meta_node.ntype) <= 3

## minimal printing
function Base.show(io::IO, n::MetaNode)
    print(io, 
        "MetaNode{" *
        cpad(string(n.ntype)[11:end],5) * "}(" *     # VAR, PARAM, STATE, OP, MOD,
        "ind=$(n.array_index)," *           # ind=1,
        "src=$(n.src_index)," *             # src=2
        "special=$(n.special_index)" *      # special=5
    ")")
end

# The `special_index` of node `n::MetaNode` refers to:
# * The index of a value in the `primals` array of a DAG if `n.ntype == META_NODE_VAR`, 
#   `n.ntype==META_NODE_PARAM` or  `n.ntype==META_NODE_STATE`.
#   (For these types of nodes it also holds that `n.array_index == n.special_index`.)
# * The index of a variable value array during evaluation if `n.ntype==META_NODE_VAR`,
# * The dispatch index for `eval_op!` and `eval_grads!` if `n.ntype==META_NODE_OP`.
# * The index of a model in the `models` array of a DAG if `n.ntype==META_NODE_MOD`.
#
# To store edge information we make use of sparse matrices:
import SparseArrays: SparseMatrixCSC, spzeros, dropstored!, nnz, nzrange, rowvals, nonzeros

# The `DAG` basically is a vector with `MetaNode`s, adjacency information and some meta-data.
# To avoid complicated filtering, there are two adjacency matrices: 
# One for predecessors (or “input”) and one for successors (or “outputs”) in the graph.
# Their array_indices are stored in order and column-wise.

import SparseArrayKit as SAK
Base.@kwdef struct DAG
    ## meta-data
    nvars :: Int
    nparams :: Int
    nstates :: Int
    nops :: Int
    nmodels :: Int

    nodes :: Vector{MetaNode} 

    sorting_indices :: Union{Vector{Int}, Nothing}
    dag_indices :: Vector{Int}

    ## partial adjacency matrix. if node `n` with index `j=n.array_index`
    ## has predecessors in the computational graph, store their indices in column `j`
    adj_pred :: SparseMatrixCSC{Int, Int}
    ## partial adjacency matrix, but with successor indices stored in columns
    adj_succ :: SparseMatrixCSC{Int, Int}
    ## NOTE
    ## previously, I used a single adjecency matrix, where 
    ## the predecessor indices of node with index j where stored as negative 
    ## values, and the successor indices as positive entries.
    ## This required akward filtering and lead to allocations during evaluation.
    
    primals :: Vector{Float64}
    partials :: SparseMatrixCSC{Float64, Int}
    partials2 :: SAK.SparseArray{Float64, 3}
    hessian_indices :: SparseMatrixCSC{Int, Int}

    p_hash :: Base.Ref{UInt64}
    x_hash :: Vector{UInt64}
    dx_hash :: SparseMatrixCSC{UInt64, Int}
    ddx_hash :: SAK.SparseArray{UInt64, 3}

    operators :: Vector{Any}
    surrogates :: Vector{Any}
end

## minimal printing
function Base.show(io::IO, dag::DAG)
    print(io, """
    DAG with 
    * $(dag.nvars) variables, $(dag.nparams) parameters and $(dag.nstates) states;
    * $(dag.nops) operator nodes and $(dag.nmodels) surrogate models.""")
end

# When creating a `DAG` from a `Model`, things become easier if we now the nodes of 
# the model to have consecutive array indices.
# The below preprocessing function makes sure of that:
function consistent_array_indices!(model)
    node_counters = fill(0, 5)  # index array for position in sub-arrays `var_indices`, `param_indices`, ...

    ## comparison function to sort operator nodes so that variables and states come 
    ## before parameters:
    input_lt_func(in1, in2) = false
    input_lt_func(in1::Union{VariableNode,StateNode}, in2::ParameterNode) = true

    ## inspect all `array_index` values and update sub-arrays to reflect possible changes:
    for (ni, n) in enumerate(model.nodes)
        ai = n.array_index
        if ai != ni
            @warn "`array_index` of node $(n) does not correspond to its position. Resetting it."
            n.array_index = ni
        end
        if n isa VariableNode
            j = node_counters[1] += 1
            model.var_indices[j] = ni
        elseif n isa ParameterNode
            j = node_counters[2] += 1
            model.param_indices[j] = ni
        elseif n isa StateNode
            j = node_counters[3] += 1
            model.state_indices[j] = ni
        elseif n isa NonlinearOperatorNode
            j = node_counters[4] += 1
            model.op_indices[j] = ni
            ## sort input nodes by `input_lt_func`
            sort!(n.in_nodes; lt=input_lt_func)
        elseif n isa SurrogateModelNode
            j = node_counters[5] += 1
            model.model_indices[j] = ni
        end
    end
    return nothing
end

function initialize(model; sort_nodes=false, check_cycles=true)
	
	nvars = length(model.var_indices)
	nstates = length(model.state_indices)
	nparams = length(model.param_indices)
    ndeps = nvars + nstates  # scalar nodes that vary, for which we can perform differentiation
 	nvals = ndeps + nparams
	nops = length(model.op_indices)
    nmodels = length(model.model_indices)
	nnodes = nvals + nops + nmodels
    
    @assert nnodes == length(model.nodes) "Model arrays contain more indices than there are nodes in `nodes`."
    consistent_array_indices!(model)
    
    ## Constructing array of `MetaNode`s for the DAG
    ## it's a bit more complicated then it seems necessary on the first glance,
    ## because I want the `special_index` of variable nodes to preceed those of 
    ## state nodes and parameter nodes...
	nodes = Vector{MetaNode}(undef, nnodes)
    ## `primals[i]` is the value of a scalar node `n` with `n.special_index==i`:
    primals = fill(NaN, nvals) 
    model_indices = Vector{Int}(undef, nnodes)
    ### do the scalar nodes:
    ai = 0  
    for ni in model.var_indices
        ai += 1
        nodes[ai] = MetaNode(META_NODE_VAR, ai, ni, ai, 0) # special_index == array_index
        model_indices[ai] = ni
    end
    for ni in model.state_indices
        ai += 1
        nodes[ai] = MetaNode(META_NODE_STATE, ai, ni, ai, 0)
        model_indices[ai] = ni
    end
    for ni in model.param_indices
        ai += 1
        nodes[ai] = MetaNode(META_NODE_PARAM, ai, ni, ai, 1)
        n = model.nodes[ni]
        primals[ai] = n.value
        model_indices[ai] = ni
    end
    @assert ai == nvals

    ## now the operator nodes
    operators = Any[]
    surrogates = Any[]
    for ni in model.op_indices
        ai += 1
        n = model.nodes[ni]
        oi = pushend!(operators, n.operator)
        nodes[ai] = MetaNode(META_NODE_OP, ai, ni, oi, n.num_deps)
        model_indices[ai] = ni
    end
    for ni in model.model_indices
        ai += 1
        n = model.nodes[ni]
        oi = pushend!(surrogates, n.surrogate)
        nodes[ai] = MetaNode(META_NODE_MOD, ai, ni, oi, n.num_in)   # TODO `num_deps`
        model_indices[ai] = ni
    end
    dag_indices = sortperm(model_indices)
    
    ## build partial adjacency matrices, because edge information is otherwise deleted with 
    ## Kahn's Algorithm below
    adj_pred = spzeros(Int, nvals, nnodes)
    adj_succ = spzeros(Int, nvals, nnodes)
    max_in = 0
    max_out = 0
    for meta_n in nodes
        meta_ni = meta_n.array_index
        n = model.nodes[meta_n.src_index]
        for (pos, m) in enumerate(in_nodes(n))
            mi = m.array_index
            meta_mi = dag_indices[mi]
            adj_pred[pos, meta_ni] = meta_mi
            if pos >= max_in
                max_in = pos
            end
        end
        for (pos, m) in enumerate(out_nodes(n))
            mi = m.array_index
            meta_mi = dag_indices[mi]
            adj_succ[pos, meta_ni] = meta_mi
            if pos >= max_out
                max_out = pos
            end
        end
    end

    adj_in = copy(adj_pred)
    adj_out = copy(adj_succ)
    if sort_nodes
        ## Kahn's Algorithm
        ## see https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm
        L = Int[]   # array of indices for sorted nodes
        ## array of nodes without ingoing edges:
        S = vcat(dag_indices[model.var_indices], dag_indices[model.param_indices]) 
        while !isempty(S)
            ni = pop!(S)        # remove a node `n` from `S`
            push!(L, ni)        # add `n` to `L`
            ## for each node `m` with an edge `e` from `n` to `m` do ...
            for (mpos, mi) in enumerate(nonzeros(adj_out[:, ni]))   # DON'T use views here! `dropstored!` does not like that...
                ## remove edge `e` from the graph
                ## a) remove `m` from `n.out_nodes`
                dropstored!(adj_out, mpos, ni)
                ## b) remove `n` from `m.in_nodes`
                for (npos, _n) in enumerate(nonzeros(adj_in[:, mi]))
                    if ni == _n
                        dropstored!(adj_in, npos, mi)
                    end
                end
            
                ## if `m` has no other incoming edges, then place it in `S`
                if nnz(view(adj_in, :, mi)) == 0
                    push!(S, mi)
                end
            end
        end
        if check_cycles
            if nnz(adj_in) > 0 || nnz(adj_out) > 0
                error("Model graph contains a cycle.")
            end
        end
        sorting_indices = L
    else
        sorting_indices = nothing
    end

    ## `partials[i, j]` means something like ∂nⱼ/∂nᵢ:
    ## column `j` contains the partial derivatives of scalar node `n` with `n.array_index==j`
    ## with respect to `m` with `m.array_index==i`:
	partials = spzeros(Float64, ndeps, ndeps)
    partials2 = SAK.SparseArray{Float64}(undef, ndeps, ndeps, ndeps)

    hessian_indices = spzeros(Int, ndeps+1, nops)

    p_hash = Ref(zero(UInt64))          # reset when changing parameters
    
    ## `x_hash[k] == h` indicates that primal value has been set for node `n` with 
    ## `n.array_index == k` for argument `x` with `h == hash(p_hash[], hash(x))`.
	x_hash = fill(p_hash[], ndeps)
    ## `dx_hash[i, k] == h` indicates that the partial derivative of node `n` with 
    ## `n.array_index == k` hab been computed with respect to node `m` with 
    ## `m.array_index == i` for argument `x` with `h == hash(p_hash[], hash(x))`.
	dx_hash = spzeros(UInt64, ndeps, ndeps)
    ## Like `dx_hash` but for second order derivatives
    ddx_hash = SAK.SparseArray{Float64}(undef, ndeps, ndeps, ndeps)
	
	return DAG(;
		nvars, nparams, nstates, nops, nmodels,
		nodes,
        dag_indices,
        sorting_indices,
		adj_pred,
		adj_succ,
        primals, 
		partials,
        partials2,
        hessian_indices,
        p_hash,
		x_hash,
		dx_hash,
        ddx_hash,
        operators,
        surrogates
	)
end

function reset_hashes!(dag)
    dag.x_hash .= 0
    dag.dx_hash .= 0
    dag.ddx_hash .= 0
    return nothing
end

# ## Evaluation

# ### Recursive Forward Pass

# We evaluate the subgraph of a DAG `dag` until reaching a target node and cache the 
# intermediate values.
# Beforehand, we set the terminal conditions (variable values) to also enable partial 
# evaluation of subgraphs that don't recurse down to variable nodes.

# First, helper functions to set bespoke terminal conditions:
function set_variables!(dag, x)
    x_hash = hash(x)
    dag.primals[1:dag.nvars] .= x
    dag.x_hash[1:dag.nvars] .= x_hash 
    return nothing 
end

function set_parameter!(dag, array_index, v)
    dag.primals[array_index] = v
    return nothing
end

# Because at initialization, we store sufficient metadata, we can process a node  
# of the original `Model` by looking up the its index in `dag.dag_indices`:
function model_node_to_dag_node(dag, node::AbstractNode)
    return dag.nodes[dag.dag_indices[node.array_index]]
end

function set_parameter!(dag, node::ParameterNode, v)
    node.value = v
    meta_node = model_node_to_dag_node(dag, node)
    return set_parameter!(dag, meta_node.array_index, v)
end

# These helper functions allow us to query predecessor and successor nodes in a DAG 
# by looking at the adjecency matrices:
"Given a node `n`, return iterable of `array_index` that preceed `n` (its inputs)."
@views pred_indices(dag, node) = nonzeros(dag.adj_pred)[nzrange(dag.adj_pred, node.array_index)]
"Given a node `n`, return iterable of `array_index` that succeed `n` (its outputs)."
@views succ_indices(dag, node) = nonzeros(dag.adj_succ)[nzrange(dag.adj_succ, node.array_index)]
@views predecessors(dag, node) = dag.nodes[pred_indices(dag, node)]
@views successors(dag, node) = dag.nodes[succ_indices(dag, node)]

# The recursive evaluation procedure starts at the target `node`.
# To gather its inputs, we have to first evaluate its successors, until reaching 
# terminal nodes.

function forward_node(
    dag, node::AbstractNode, x=nothing; 
    mode=:no_surrogates, prepare_grads=false, prepare_hessians=false, kwargs...
)
    meta_node = model_node_to_dag_node(dag, node)
    return forward_node(dag, meta_node, x; mode, prepare_grads, prepare_hessians, kwargs...)
end

function forward_node(
    dag, node::AbstractNode, x=nothing; 
    mode=:no_surrogates, prepare_grads=false, prepare_hessians=false, kwargs...
)
    if !isnothing(x)
        set_variables!(dag, x)
    end
    return _forward_node(dag, node; mode, prepare_grads, prepare_hessians, kwargs...)
end

# How to evaluate a node depends on its type.
# `_forward_node` simply redirects to specific methods
function _forward_node(dag, node::MetaNode; kwargs...)
	if node.ntype == META_NODE_VAR
		_forward_var_node(dag, node; kwargs...)
	elseif node.ntype == META_NODE_PARAM
		_forward_param_node(dag, node; kwargs...)
	elseif node.ntype == META_NODE_STATE
		_forward_state_node(dag, node; kwargs...)
	elseif node.ntype == META_NODE_OP
		eval_op_node(dag, node; kwargs...)
	elseif node.ntype == META_NODE_MOD
		eval_model_node(dag, node; kwargs...)
	end
    return nothing
end

# For parameter nodes, there is nothing to do, there values are assumed fixed and set in
# primals:
function _forward_param_node(args...; kwargs...) end
# For variable nodes, we also have a no-op, because we already called `set_variables!`:
function _forward_var_node(args...; kwargs...) end

# For state nodes, we recursively visit their successors:
function _forward_state_node(dag, node; kwargs...)
    operator_indices = pred_indices(dag, node)
    for op_array_index in operator_indices
        _forward_node(
            dag, dag.nodes[op_array_index]; caller_array_index=node.array_index, kwargs...)
    end
    return nothing
end

# For parameter nodes, there is nothing to do, there values are assumed fixed and set in
# primals:
function eval_param_node(args...; kwargs...) end

# The value of a variable node is looked up in the input vector `x` according to its 
# `secial index`:
function eval_var_node(dag, node, x, x_hash; kwargs...)
    dag.x_hash[node.array_index] == x_hash && return nothing
    dag.primals[node.special_index] = x[node.special_index]
    dag.x_hash[node.array_index] = x_hash
    return nothing
end

# For state nodes, we recursively visit their successors:
function eval_state_node(dag, node, x, x_hash; kwargs...)
    operator_indices = pred_indices(dag, node)
    for op_array_index in operator_indices
        eval_node(
            dag, dag.nodes[op_array_index], x, x_hash; 
            caller_array_index=node.array_index, kwargs...)
    end
    return nothing
end

# Evaluation of non-scalar nodes is most complicated, as we have to actually call the
# underlying functions and map their inputs and outputs to arrays of the `DAG`.
#
# This is a helper function. Suppose, we want to only evaluate a single output 
# of a vector-valued function. 
# The output corresponds to a node with `caller_array_index`, and `_partial_output_i_or_nothing`
# either returns its position in the array `output_indices` of an operator node, or 
# `nothing`:
function _partial_output_i_or_nothing(can_partial, output_indices, caller_array_index)
    if !can_partial || isnothing(caller_array_index)
        return nothing
    end
    return findfirst(isequal(caller_array_index), output_indices)
end

@views function inputs_and_hash(dag_primals, input_indices, node)
    dep_indices = input_indices[1:node.num_deps]
    param_indices = input_indices[node.num_deps+1:end] # for `n.ntype==META_NODE_MOD`, this should be empty
    X = dag_primals[dep_indices]
    P = dag_primals[param_indices]
    x_hash = hash(X,hash(P))
    return dep_indices, param_indices, X, P, x_hash
end

function set_hashes!(hash_arr, hash_val, all_indices, sub_index)
    if isnothing(sub_index)
        hash_arr[all_indices] .= hash_val
    else
        for i in sub_index
            j = all_indices[i]
            hash_arr[j] = hash_val
        end
    end
end

@views function _forward_op_node(
    dag, node; mode, prepare_grads, prepare_hessians, caller_array_index, kwargs...
)
    ## If we are evaluating the “shadow graph”, return early:
    mode != :no_surrogates && return nothing

    ## First, we recursively visit all predecessor nodes
    input_indices = pred_indices(dag, node)
    for n in dag.nodes[input_indices]
        _forward_node(dag, n; mode, prepare_grads, prepare_hessians, kwargs...)
    end

    ## We next inspect, whether we have to update the outputs to the operator node
    ## based on the hash of the successor with `caller_array_index`, indicating the hash
    ## of the previous input for which values have been computed:
    dep_indices, param_indices, X, P, x_hash = inputs_and_hash(dag.primals, input_indices, node)
    old_hash = dag.x_hash[caller_array_index]
    needs_vals = x_hash != old_hash

    ## We can similarly determine whether to re-compute derivative information.
    ## Derivatives depend on the first `node.num_deps` inputs:
    needs_grads = prepare_grads && x_hash != dag.dx_hash[caller_array_index]
    needs_hessians = prepare_hessians && x_hash != dag.ddx_hash[caller_array_index]
        
    ## Finally, do everything that is needed by calling a subfunction 
    ## that takes the operator as its first argument:
    __forward_functional(
        dag.operators[node.special_index], X, P, x_hash, dep_indices, output_indices, 
        caller_array_index, dag.primals, dag.partials, dag.partials2, dag.x_hash, dag.dx_hash,
        dag.ddx_hash, needs_vals, needs_grads, needs_hessians)

    return nothing
end

@views function __forward_functional(
    functional, X, P, x_hash, dep_indices, output_indices, caller_array_index, 
    primals, partials, partials2, x_hash_arr, dx_hash_arr, ddx_hash_arr, 
    needs_vals, needs_grads, needs_hessians
)
    if needs_vals || needs_grads || needs_hessians
        partial_output_i = _partial_output_i_or_nothing(
            supports_partial_evaluation(functional), output_indices, caller_array_index
        )   # `nothing`, if partial evaluation is not supported, `i` otherwise 

        y = primals[output_indices]

        if needs_grads || needs_hessians
            Dy = partials[dep_indices, output_indices]
            if needs_hessians
                H = partials2[dep_indices, dep_indices, output_indices]
                func_vals_and_grads_and_hessians!(y, Dy, H, functional, X, P, partial_output_i)
                set_hashes!(ddx_hash_arr, x_hash, output_indices, partial_output_i)
            else
                func_vals_and_grads!(y, Dy, functional, X, P, partial_output_i)
            end
            set_hashes!(dx_hash_arr, x_hash, output_indices, partial_output_i)
        else
            func_vals!(y, functional, X, P, partial_output_i)
        end
        set_hashes!(x_hash_arr, x_hash, output_indices, partial_output_i)
    end     
    return nothing
end

function eval_node(dag, node::AbstractNode, x; mode=:no_surrogates, kwargs...)
    meta_node = dag.nodes[dag.dag_indices[node.array_index]]
    return eval_node(dag, meta_node, x; mode, kwargs...)
end

function eval_node(dag, node::MetaNode, x; mode=:no_surrogates, kwargs...)
    x_hash = hash(mode, hash(dag.p_hash, hash(x)))
    return eval_node(dag, node, x, x_hash; mode, kwargs...)
end

function eval_node(dag, node::MetaNode, x, x_hash; kwargs...)
#src    if is_dep_node(node)
#src        ni = node.array_index
#src	    if x_hash == dag.x_hash[ni]
#src		    return nothing
#src	    end
#src    else
#src        ni = -1
#src    end
	
	if node.ntype == META_NODE_VAR
		eval_var_node(dag, node, x, x_hash; kwargs...)
	elseif node.ntype == META_NODE_PARAM
		eval_param_node(dag, node, x, x_hash; kwargs...)
	elseif node.ntype == META_NODE_STATE
		eval_state_node(dag, node, x, x_hash; kwargs...)
	elseif node.ntype == META_NODE_OP
		eval_op_node(dag, node, x, x_hash; kwargs...)
	elseif node.ntype == META_NODE_MOD
		eval_model_node(dag, node, x, x_hash; kwargs...)
	end
    
#src    if ni > 0
#src        dag.x_hash[ni] = x_hash
#src    end

    return nothing
end

# For parameter nodes, there is nothing to do, there values are assumed fixed and set in
# primals:
function eval_param_node(args...; kwargs...) end

# The value of a variable node is looked up in the input vector `x` according to its 
# `secial index`:
function eval_var_node(dag, node, x, x_hash; kwargs...)
    dag.x_hash[node.array_index] == x_hash && return nothing
    dag.primals[node.special_index] = x[node.special_index]
    dag.x_hash[node.array_index] = x_hash
    return nothing
end

# For state nodes, we recursively visit their successors:
function eval_state_node(dag, node, x, x_hash; kwargs...)
    operator_indices = pred_indices(dag, node)
    for op_array_index in operator_indices
        eval_node(
            dag, dag.nodes[op_array_index], x, x_hash; 
            caller_array_index=node.array_index, kwargs...)
    end
    return nothing
end

# Evaluation of operator nodes is most complicated, as we have to actually call the
# underlying functions and map their inputs and outputs to arrays of the `DAG`.
#
# This is a helper function. Suppose, we want to only evaluate a single output 
# of a vector-valued function. 
# The output corresponds to a node with `caller_array_index`, and `_partial_output_i_or_nothing`
# either returns its position in the array `output_indices` of an operator node, or 
# `nothing`:
function _partial_output_i_or_nothing(can_partial, output_indices, caller_array_index)
    if !can_partial || isnothing(caller_array_index)
        return nothing
    end
    return findfirst(isequal(caller_array_index), output_indices)
end
# To actually index `output_indices` with the position, `_partial_output_indices` might 
# be a bit more useful. It will always return an iterable index:
function _partial_output_indices(can_partial, output_indices, caller_array_index)
    i = _partial_output_i_or_nothing(can_partial, output_indices, caller_array_index)
    isnothing(i) && return output_indices
    return [i,]
end

# Convenience functions for checking hashes:
@views x_hashs_invalid(dag, x_hash, output_index) = any(dag.x_hash[output_index] .!= x_hash)
@views function dx_hashs_invalid(dag, x_hash, output_index, dep_indices)
    any( dag.dx_hash[dep_indices, output_index] .!= x_hash )
end
@views function ddx_hashs_invalid(dag, x_hash, output_index, dep_indices)
    any( dag.ddx_hash[dep_indices, dep_indices, output_index] .!= x_hash )
end

function eval_model_node(args...; mode, kwargs...)
    if mode == :no_surrogates
        return nothing
    else
        ## TODO
    end
end

@views function _eval_functional_node(
    dag, @nospecialize(functional), x, x_hash, node, input_indices, output_indices;
    mode, prepare_grads=false, prepare_hessians=false, caller_array_index=nothing, kwargs...
)
    is_surrogate = mode != :no_surrogates

    ## recursively set inputs for current operator
    for n in dag.nodes[input_indices]
        eval_node(dag, n, x, x_hash; mode, prepare_grads, prepare_hessians, kwargs...)
    end

    ## compute output(s)   
    output_indices = succ_indices(dag, node)
    partial_output_i = _partial_output_i_or_nothing(
        supports_partial_evaluation(functional),output_indices, caller_array_index)
    partial_output_indices = isnothing(partial_output_i) ? output_indices : output_indices[partial_output_i]

    dep_indices = input_indices[1:node.num_deps]        

    needs_vals = x_hashs_invalid(dag, x_hash, partial_output_indices) 
    needs_grads = prepare_grads && dx_hashs_invalid(dag, x_hash, partial_output_indices, dep_indices)
    needs_hessians = prepare_hessians && !is_surrogate && ddx_hashs_invalid(
        dag, x_hash, partial_output_indices, dep_indices)

    if needs_vals || needs_grads || needs_hessians
        param_indices = input_indices[node.num_deps+1:end] # for `n.ntype==META_NODE_MOD`, this should be empty
        X = dag.primals[dep_indices]
        P = dag.primals[param_indices]

        y = dag.primals[output_indices]

        if needs_grads || needs_hessians
            Dy = dag.partials[dep_indices, output_indices]
            if needs_hessians
                H = dag.partials2[dep_indices, dep_indices, output_indices]
                func_vals_and_grads_and_hessians!(y, Dy, H, functional, X, P, partial_output_i)
                display(Dy) 
                _prepare_hessians!(dag.hessian_indices, dag.partials, node.special_index, dep_indices)
                dag.ddx_hash[dep_indices, dep_indices, partial_output_indices] .= x_hash
            else
                func_vals_and_grads!(y, Dy, functional, X, P, partial_output_i)
            end
            dag.dx_hash[dep_indices, partial_output_indices] .= x_hash
        else
            func_vals!(y, functional, X, P, partial_output_i)
            dag.x_hash[partial_output_indices] .= x_hash
        end
    end     

    return nothing
end

@views function _prepare_hessians!(dag_hessian_indices, dag_partials, op_special_index, dep_indices)
    @show op_special_index
    if length(nzrange(dag_hessian_indices, op_special_index)) == 0
        ## compute indices for computation of output hessians
        k = 1
        Ind = dag_hessian_indices[:, op_special_index]
        Ind[k] = -1  # “parity bit” to avoid re-computations

        Dy_rows = rowvals(dag_partials)
        for j in dep_indices
            for l in Dy_rows[nzrange(dag_partials, j)]
                if !(l in Ind[1:k])
                    k += 1
                    Ind[k] = l
                end
            end
        end 
        @show Ind
    end
    return nothing
end

@views function eval_op_node(dag, node, x, x_hash; 
    caller_array_index=nothing, prepare_grads=false, prepare_hessians=false,
    mode, kwargs...
)
    if mode==:no_surrogates
        ## recursively set inputs for current operator
        input_indices = pred_indices(dag, node)
        output_indices = succ_indices(dag, node)
        functional = dag.operators[node.special_index]
        return _eval_functional_node(
            dag, functional, x, x_hash, node, input_indices, output_indices;
            mode, prepare_grads, prepare_hessians, caller_array_index, kwargs...
        )
    end

    return nothing
end

# ### Recursive Pullback(s)
#
# In performing “pullback”, we evaluate the chain rule by recursively substituting 
# partial derivatives of the outer function.
# That is, we propagate cotangents: starting with `∂n/∂n=1`, 
# for every predecessor `mᵢ` of `n`, we find
# `∂n/∂mᵢ = ∂n/∂n ∂n/∂mᵢ`. If `mᵢ` depends on `kⱼ`, then `∂n/∂kⱼ = ∂n/∂mᵢ ∂mᵢ/∂kⱼ`, 
# and so forth.

function pullback_node(dag, node::AbstractNode; mode=:no_surrogates, kwargs...)
    meta_node = dag.nodes[dag.dag_indices[node.array_index]]
    return pullback_node(dag, meta_node; mode, kwargs...)
end

# Below, `diff_index` is the array index of the node for which we compute all its partial 
# derivatives:
function pullback_node(dag, node::MetaNode; mode=:no_surrogates, kwargs...)
    @assert is_scalar_node(node) "Can only pullback on scalar nodes."
    diff_index = node.array_index
    dag.partials[diff_index, diff_index] = 1
    x_hash = dag.x_hash[diff_index]
    dag.dx_hash[diff_index, diff_index] = x_hash
    return pullback_node(dag, node, diff_index, x_hash; mode, kwargs...)
end

function pullback_node(dag, node, diff_index, x_hash; mode, kwargs...)
    
#src 	if is_dep_node(node)
#src         ni = node.array_index
#src 	
#src 	    if dag.x_hash[ni] != x_hash
#src 		    error("Fractured evaluation tree, gradients not valid. Perform forward pass up until node with array index $diff_index.")
#src 	    end
#src     end

    if node.ntype == META_NODE_VAR
		return pullback_var_node(dag, node, diff_index, x_hash; mode, kwargs...)
    elseif node.ntype == META_NODE_PARAM
		return pullback_param_node(dag, node, diff_index, x_hash; mode, kwargs...)
	elseif node.ntype == META_NODE_STATE
		return pullback_state_node(dag, node, diff_index, x_hash; mode, kwargs...)
	elseif node.ntype == META_NODE_OP
		return pullback_op_node(dag, node, diff_index, x_hash; mode, kwargs...)
    elseif node.ntype == META_NODE_MOD
		return pullback_model_node(dag, node, diff_index, x_hash; mode, kwargs...)
	end
end

function pullback_param_node(dag, node, diff_index, x_hash; mode, kwargs...) end

function pullback_var_node(dag, node, diff_index, x_hash; mode, kwargs...) 
	return _sum_successor_partials(dag, node, diff_index, x_hash)
end

function pullback_state_node(dag, node, diff_index, x_hash; mode, kwargs...)
	_sum_successor_partials(dag, node, diff_index, x_hash)
    for op_array_index in pred_indices(dag, node)
        pullback_node(
            dag, dag.nodes[op_array_index], diff_index, x_hash; mode, kwargs...)
    end
end

#=
In `_sum_successor_partials`, we want to compute `∂z/∂n`, with `z.array_index==diff_index`.
We assume the partial derivatives of all successors of `n` to be set with respect to `n`.
To get `∂z/∂n` we have to sum all these partial derivatives, which is what
we do below.
The other function, `_partials_term`, is just a helper to have a tidier for loop.
=#

function _partials_term(dag_partials, dag_dx_hash, x_hash, node_index, diff_index, successor_node)
    si = successor_node.array_index
    if is_dep_node(successor_node) && dag_dx_hash[si, diff_index] != x_hash 
        error("""
            Inconsistent partial derivative information for node `n = nodes[$(node_index)]`,
            while pulling back on node `z=nodes[$(diff_index)]`.
            For `s=nodes[$(si)],` the value `∂s/∂n` has been set for a different
            input than was primal for `z`.""")
    end
	return dag_partials[si, diff_index] * dag_partials[node_index, si]
end

function _sum_successor_partials(dag, node, diff_index, x_hash)
	ni = node.array_index
    
    ## In the very first call to `pullback_node`, we query
    ## ∂z/∂z, which is already seeded, and we want to ignore 
    ## successor information: 
	ni == diff_index && return nothing
	
    ## If the hash is set already, we can also return early:
	dag.dx_hash[ni, diff_index] == x_hash && return nothing

    ## We have to actually compute the derivative, if we are here...
	## Let node with `diff_index` be `z`, let current node be `n`.
	## We assume the cotangents of all successors, ∂sᵢ/∂n, to be available to sum them up:
	## ∂z/∂n = ̄n = Σ ∂z/∂s ∂s/∂n
	dzdn = zero(eltype(dag.partials))
	for outn in successors(dag, node)
		if !is_scalar_node(outn) 
			for _outn in successors(dag, outn)
                dzdn += _partials_term(dag.partials, dag.dx_hash, x_hash, ni, diff_index, _outn)
            end
        else
            dzdn += _partials_term(dag.partials, dag.dx_hash, x_hash, ni, diff_index, outn)
        end
	end
	dag.partials[ni, diff_index] = dzdn
    dag.dx_hash[ni, diff_index] = x_hash
	return nothing
end

function pullback_model_node(dag, node, diff_index, x_hash; mode, kwargs...) end # TODO

function pullback_op_node(dag, node, diff_index, x_hash; mode, kwargs...)
	input_idx = pred_indices(dag, node)
	
	for in_idx in input_idx
		n = dag.nodes[in_idx]
		pullback_node(dag, n, diff_index, x_hash; mode, kwargs...)
	end
	
	return nothing
end

# ### Forward Hessians

#=
To compute Hessian matrices, Wikipedia gives us 
[Faà di Brunos formula](https://en.wikipedia.org/wiki/Chain_rule#Higher_derivatives_of_multivariable_functions):
```math
\frac{
    ∂y
}{
    ∂x_i ∂x_j
}
= 
\sum_k
    \left(
        \frac{
            ∂y
        }{
            ∂u_k
        }
        \frac{
            ∂^2 u_k
        }{
            ∂x_i ∂x_j
        }
    \right)
+
\sum_{k,ℓ}
    \left(
        \frac{
            ∂^2 y
        }{
            ∂u_k ∂u_ℓ
        }
        \frac{
            ∂u_k
        }{
            ∂x_i
        }
        \frac{
            ∂u_ℓ
        }{
            ∂x_j
        }
    \right).
```
This corresponds to the matrix formula found in [this article](https://arxiv.org/pdf/1911.13292.pdf):
```math
H(f∘g) = (∇g)ᵀ ⋅ Hf(g) ⋅ ∇g + ∑ₖ ∂ₖf Hgₖ.
```

!!! note
    I am not 100 % sure everything works out for non-symmetric Hessians though...
=#

#=
The function `forward_hessian` recursively computes the Hessian information for the 
target `node`.
Originally, we called `eval_hessians!` in operator nodes.
But this should rather be done in the forward pass by setting `prepare_hessians=true`
for `eval_node`.
=#

function forward_hessian(dag, node::AbstractNode; kwargs...)
    meta_node = dag.nodes[dag.dag_indices[node.array_index]]
    return forward_hessian(dag, meta_node; kwargs...)
end

function forward_hessian(dag, node::MetaNode; kwargs...)
    @assert is_scalar_node(node) "Can only compute Hessian of scalar nodes for now."
    #src # TODO for everything that is not a scalar we can iterate over outputs maybe
    diff_index = node.array_index
    x_hash = dag.x_hash[diff_index]
    return forward_hessian(dag, node, diff_index, x_hash; kwargs...)
end

function forward_hessian(dag, node, diff_index, x_hash; kwargs...)
	#src if is_dep_node(node)
    #src        ni = node.array_index
	#src     if dag.x_hash[ni] != x_hash
	#src 	    error("Fractured evaluation tree, gradients not valid. Perform forward pass up until node with array index $diff_index.")
	#src     end
    #src        if dag.ddx_hash[ni] == x_hash
    #src            return nothing
    #src        end
    #src    else
    #src        ni = -1
    #src    end 
	
	if node.ntype == META_NODE_STATE
		forward_hessian_state(dag, node, diff_index, x_hash; kwargs...)
	elseif node.ntype == META_NODE_OP
		forward_hessian_op(dag, node, diff_index, x_hash; kwargs...)
	end
    #src if ni > 0
    #src     dag.ddx_hash[ni] = x_hash
    #src end
    return nothing
end

function forward_hessian_state(dag, node, diff_index, x_hash; kwargs...)
    op_indices = pred_indices(dag, node)
    for op_array_index in op_indices
        forward_hessian_op(
            dag, dag.nodes[op_array_index], diff_index, x_hash; caller_array_index=node.array_index, kwargs...)
    end
    return nothing
end

import LinearAlgebra as LA
@views function forward_hessian_op(
    dag, node, diff_index, x_hash; 
    check_hessians_set=false,
    caller_array_index=nothing,
    kwargs...
)
    input_indices = pred_indices(dag, node)

    ## set Hessians of inputs
    for in_ind in input_indices
        forward_hessian(dag, dag.nodes[in_ind], diff_index, x_hash)
    end

    output_indices = succ_indices(dag, node)
    functional = dag.operators[node.special_index]
    partial_output_i = _partial_output_i_or_nothing(
        supports_partial_evaluation(functional),output_indices, caller_array_index)
    partial_output_indices = isnothing(partial_output_i) ? output_indices : output_indices[partial_output_i]

    dep_indices = input_indices[1:node.num_deps]

    _prepare_hessians!(dag.hessian_indices, dag.partials, node.special_index, dep_indices)

    if check_hessians_set
        functional = dag.operators[node.special_index]
        if ddx_hashs_invalid(dag, x_hash, partial_output_indices, dep_indices)
            param_indices = input_indices[node.num_deps+1:end]
            X = dag.primals[dep_indices]
            P = dag.primals[param_indices]
            H = dag.partials2[dep_indices, dep_indices, output_indices]
            eval_hessians!(H, functional, X, P, partial_output_i)
        end
    end
   
    ij_indices = nonzeros(dag.hessian_indices)[nzrange(dag.hessian_indices, node.special_index)[2:end]]
    @show ij_indices
    if !isempty(ij_indices)
        for out_ind in partial_output_indices
            Hy = dag.partials2[ij_indices, ij_indices, out_ind]     # target array
            Hy_u = dag.partials2[dep_indices, dep_indices, out_ind] # hessian of output `y` w.r.t. inputs u_1, …, u_k
            JuT = dag.partials[ij_indices, dep_indices]             # transposed Jacobian of inputs
            Hy .= JuT * Hy_u * JuT'    # TODO can we do some in-place stuff here?
            
            for k in dep_indices
                dy_uk = dag.partials[k, out_ind]
                Huk = dag.partials2[ij_indices, ij_indices, k]
                Hy .+= dy_uk .* Huk
            end
        end
    end

	return nothing
end

end