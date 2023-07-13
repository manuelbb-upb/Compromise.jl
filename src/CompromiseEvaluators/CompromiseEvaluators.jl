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
Base.@kwdef struct DAG{T<:AbstractFloat}
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
    
    primals :: Vector{T}
    partials :: SparseMatrixCSC{T, Int}       # array to store partial derivative information of non-scalar nodes
    grads :: SparseMatrixCSC{T, Int}          # gradients computed from partials in pullback
    partials2 :: SAK.SparseArray{T, 3}        # array to store second order partial derivatives
    hessians :: SAK.SparseArray{T, 3}         # hessians computed from partials in pullback

    #src # TODO (“Transposed Jacobians”)
    #src after introducing partials and grads seperately, we would rather like to have
    #src Jacobians in `partials` to query columns in pullback OR use `SparseMatricesCSR`
    #src and keep storing transposed Jacobians...

    primals_hashes :: Vector{UInt}
    partials_hashes :: SparseMatrixCSC{UInt, Int}
    grads_hashes :: SparseMatrixCSC{UInt, Int}
    partials2_hashes :: SAK.SparseArray{UInt, 3}
    hessians_hashes :: SAK.SparseArray{UInt, 3} 

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

# Some helper functions to build a `DAG` from a `Model`: \
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

# Now a function to build an array of `MetaNode`s for the DAG.
# It's a bit more complicated then it seems necessary on the first glance,
# because I want the `special_index` of variable nodes to preceed those of 
# state nodes and parameter nodes...
function meta_node_arrays(model, nvals, nnodes)
	nodes = Vector{MetaNode}(undef, nnodes)
    ## `primals[i]` is the value of a scalar node `n` with `n.special_index==i`:
    primals = fill(NaN, nvals) 
    model_indices = Vector{Int}(undef, nnodes)
    ### 1) do the scalar nodes:
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

    ## 2) now the operator nodes
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
    return nodes, primals, operators, surrogates, dag_indices
end

# The adjacency matrices contain information edge informaton for predecessors and 
# successors seperately.
# The matrices are rectangular, because a scalar node can only have a non-scalar node 
# as a successor and vice versa.
function adjacency_matrices(dag_nodes, model_nodes, dag_indices, nvals, nnodes)
    adj_pred = spzeros(Int, nvals, nnodes)
    adj_succ = spzeros(Int, nvals, nnodes)
    #scr max_in = 0
    #src max_out = 0
    for meta_n in dag_nodes
        meta_ni = meta_n.array_index
        n = model_nodes[meta_n.src_index]
        for (pos, m) in enumerate(in_nodes(n))
            mi = m.array_index
            meta_mi = dag_indices[mi]
            adj_pred[pos, meta_ni] = meta_mi
            #src if pos >= max_in
            #src     max_in = pos
            #src end
        end
        for (pos, m) in enumerate(out_nodes(n))
            mi = m.array_index
            meta_mi = dag_indices[mi]
            adj_succ[pos, meta_ni] = meta_mi
            #src if pos >= max_out
            #src     max_out = pos
            #src end
        end
    end
    return adj_pred, adj_succ
end

# The graph structure allows us to sort the nodes in topological order, so that we could
# theoretically evaluate everything in order.
# At the moment, I do not use this sorting, but rather employ recursion:
function topological_sort(
    adj_pred, adj_succ, dag_indices, model_var_indices, model_param_indices; 
    sort_nodes, check_cycles
)
    ## copy matrices, because edges get deleted
    adj_in = copy(adj_pred)
    adj_out = copy(adj_succ)
    if sort_nodes
        ## Kahn's Algorithm
        ## see https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm
        L = Int[]   # array of indices for sorted nodes
        ## array of nodes without ingoing edges:
        S = vcat(dag_indices[model_var_indices], dag_indices[model_param_indices]) 
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
    return sorting_indices
end

# The complete initialization routine:
function initialize(model, T::Type{<:AbstractFloat}=Float64; sort_nodes=false, check_cycles=true)
	
    ## parse number of nodes of different types
	nvars = length(model.var_indices)
	nstates = length(model.state_indices)
	nparams = length(model.param_indices)
    ndeps = nvars + nstates  # scalar nodes that vary, for which we can perform differentiation
 	nvals = ndeps + nparams
	nops = length(model.op_indices)
    nmodels = length(model.model_indices)
	nnodes = nvals + nops + nmodels
    
    @assert nnodes == length(model.nodes) "Model arrays contain more indices than there are nodes in `nodes`."

    ## modify nodes such that `array_index` values are consecutive
    consistent_array_indices!(model)
    
    nodes, primals, operators, surrogates, dag_indices = meta_node_arrays(model, nvals, nnodes)
    
    ## build partial adjacency matrices
    adj_pred, adj_succ = adjacency_matrices(nodes, model.nodes, dag_indices, nvals, nnodes)
    
    ## Kahn's Algorithm below
    sorting_indices = topological_sort(
        adj_pred, adj_succ, dag_indices, model.var_indices, model.param_indices;
        sort_nodes, check_cycles
    )
    
    ## Meaning of `partials` and `grads`:
    ## Assume some operator with `l` inputs maps nodes `n₁, …, nₗ`
    ## to nodes `m₁, …, mₖ`.
    ## `partials` stores the partial derivatives of the operator column-wise.
    ## `partials[i, j]` is `∂ᵢmⱼ`, but not the total derivative of `mⱼ` with respect to `nᵢ`.
    ## That is stored in `grads[i, j]` after pullback.
	partials = spzeros(T, ndeps, ndeps)
	grads = spzeros(T, ndeps, ndeps)
    partials2 = SAK.SparseArray{T}(undef, ndeps, ndeps, ndeps)
    hessians = SAK.SparseArray{T}(undef, ndeps, ndeps, ndeps)
 
    ## Hash arrays, Meaning is explained below.
	primals_hashes = zeros(UInt, nvals)
	partials_hashes = spzeros(UInt, ndeps, ndeps)
	grads_hashes = spzeros(UInt, ndeps, ndeps)

    ## Like `dx_hash` but for second order derivatives
    partials2_hashes = SAK.SparseArray{UInt}(undef, ndeps, ndeps, ndeps)
    hessians_hashes = SAK.SparseArray{UInt}(undef, ndeps, ndeps, ndeps)
	
	return DAG(;
		nvars, nparams, nstates, nops, nmodels,
		nodes,
        sorting_indices,
        dag_indices,
		adj_pred,
		adj_succ,
        primals, 
		partials,
        grads,
        partials2,
        hessians,
        primals_hashes,
        partials_hashes,
        grads_hashes,
        partials2_hashes,
        hessians_hashes,
        operators,
        surrogates
	)
end

function reset!(dag)
    dag.primals_hashes .= 0
    dag.partials_hashes .= 0
    dag.grads_hashes .= 0
    dag.partials2_hashes .= 0
    dag.hessians_hashes .= 0
    dag.primals .= 0
    dag.partials .= 0
    dag.grads .= 0
    dag.partials2 .= 0
    dag.hessians .= 0
    return nothing
end

# ## Evaluation

# ### Recursive Forward Pass

# We evaluate the subgraph of a DAG `dag` until reaching a target node and cache the 
# intermediate values.
# Beforehand, we set the terminal conditions (variable values) to also enable partial 
# evaluation of subgraphs that don't recurse down to variable nodes.

# First, helper functions to set bespoke terminal conditions:
function set_primal!(dag_primals, dag_primals_hashes, array_index, v)
    dag_primals[array_index] = v
    dag_primals_hashes[array_index] = hash(v)
end

function set_variables!(dag, x)
    for (i,xi) = enumerate(x)
        set_primal!(dag.primals, dag.primals_hashes, i, xi)
    end
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
    return set_primal!(dag.primals, dag.primals_hashes, meta_node.array_index, v)
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
    mode=:no_surrogates, prepare_grads=false, prepare_hessians=false, 
    check_fw_hashes::Bool=true,
    kwargs...
)
    meta_node = model_node_to_dag_node(dag, node)
    return forward_node(
        dag, meta_node, x; mode, prepare_grads, prepare_hessians, check_fw_hashes, kwargs...
    )
end

function forward_node(
    dag, node::MetaNode, x=nothing; 
    mode=:no_surrogates, prepare_grads=false, prepare_hessians=false, 
    check_fw_hashes::Bool=true, 
    kwargs...
)
    !isnothing(x) && set_variables!(dag, x)
    
    return _forward_node(
        dag, node; mode, prepare_grads, prepare_hessians, check_fw_hashes, kwargs...
    )
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
		_forward_op_node(dag, node; kwargs...)
	elseif node.ntype == META_NODE_MOD
		_forward_mod_node(dag, node; kwargs...)
	end
    return nothing
end

# For parameter nodes, there is nothing to do, there values are assumed fixed and set in
# primals:
function _forward_param_node(args...; kwargs...) end
# For variable nodes, we also have a no-op, because we already called `set_variables!`:
function _forward_var_node(args...; kwargs...) end

# For state nodes, we recursively visit their predecessors.
# Most of the time, in the DAG a state should only have one node as a direct predecessor,
# an operator node.
# However, we allow for nodes to be modelled by model nodes, so we have to 
# distinguish both cases here:
function _forward_state_node(dag, node; mode, kwargs...)
    op_nodes = predecessors(dag, node)
    if length(op_nodes) > 1
        for op_n in op_nodes
            if mode == :no_surrogates && op_n.ntype != META_NODE_OP ||
                mode != :no_surrogates && op_n.ntype == META_NODE_MOD
                continue
            end
            _forward_node(
                dag, op_n; caller_array_index=node.array_index, mode, kwargs...)
            break
        end
    else
        _forward_node(
                dag, only(op_nodes); caller_array_index=node.array_index, mode, kwargs...)
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

@views function op_forward_hash(dag_primals_hashes, op_predecessor_indices, op_ntype)
    return reduce(hash, dag_primals_hashes[op_predecessor_indices]; init=(op_ntype))
end

function op_forward_hash(dag, op_node) 
    return op_forward_hash(dag.primals_hashes, pred_indices(dag, op_node), op_node.ntype)
end

@views function node_hash(dag_xhash, input_indices, ntype)
    input_hashes = dag_xhash[input_indices]
    return reduce(hash, input_hashes; init=hash(ntype))
end

@views function node_input_indices(input_indices, node)
    return input_indices[1:node.num_deps], input_indices[node.num_deps+1:end]
end

@views function node_vectors(dag_primals, input_indices, node)
    dep_indices, param_indices = node_input_indices(input_indices, node)
    X = dag_primals[dep_indices]
    P = dag_primals[param_indices]
    return dep_indices, param_indices, X, P
end

@views function _primals_hashes!(dag_primals_hashes, fw_hash, output_indices, sub_index::Nothing)
    dag_primals_hashes[output_indices] .= fw_hash
    nothing
end
@views function _primals_hashes!(dag_primals_hashes, fw_hash, output_indices, sub_index::Int)
    dag_primals_hashes[output_indices[sub_index]] = fw_hash
    nothing
end
primals_hashes!(dag_primals_hashes, fw_hash::Nothing, output_indices, sub_index)=nothing
function primals_hashes!(dag_primals_hashes, fw_hash, output_indices, sub_index)
    return _primals_hashes!(dag_primals_hashes, fw_hash, output_indices, sub_index)
end

@views function _partials_hashes!(dag_partials_hashes, fw_hash, dep_indices, output_indices, sub_index::Nothing)
    dag_partials_hashes[dep_indices, output_indices] .= fw_hash
    nothing
end
@views function _partials_hashes!(dag_partials_hashes, fw_hash, dep_indices, output_indices, sub_index::Int)
    dag_partials_hashes[dep_indices, output_indices[sub_index]] = fw_hash
    nothing
end
partials_hashes!(dag_primals_hashes, fw_hash::Nothing, dep_indices, output_indices, sub_index)=nothing
function partials_hashes!(dag_primals_hashes, fw_hash, dep_indices, output_indices, sub_index)
    return _partials_hashes!(dag_primals_hashes, fw_hash, dep_indices, output_indices, sub_index)
end

@views function _partials2_hashes!(dag_partials2_hashes, fw_hash, dep_indices, output_indices, sub_index::Nothing)
    dag_partials2_hashes[dep_indices, dep_indices, output_indices] .= fw_hash
    nothing
end
@views function _partials2_hashes!(dag_partials2_hashes, fw_hash, dep_indices, output_indices, sub_index::Int)
    dag_partials2_hashes[dep_indices, dep_indices, output_indices[sub_index]] = fw_hash
    nothing
end
partials2_hashes!(dag_primals_hashes, fw_hash::Nothing, dep_indices, output_indices, sub_index)=nothing
function partials2_hashes!(dag_primals_hashes, fw_hash, dep_indices, output_indices, sub_index)
    return _partials2_hashes!(dag_primals_hashes, fw_hash, dep_indices, output_indices, sub_index)
end

@views function _forward_op_node(
    dag, node; prepare_grads, prepare_hessians, caller_array_index, check_fw_hashes, kwargs...
)
    ## First, we recursively visit all predecessor nodes
    input_indices = pred_indices(dag, node)
    for n in dag.nodes[input_indices]
        _forward_node(dag, n; prepare_grads, prepare_hessians, check_fw_hashes, kwargs...)
    end

    ## We next inspect, whether we have to update the outputs to the operator node
    ## based on the hash of the successor with `caller_array_index`, indicating the hash
    ## of the previous input for which values have been computed:
    dep_indices, param_indices, X, P = node_vectors(dag.primals, input_indices, node)
    if check_fw_hashes
        new_hash = op_forward_hash(dag.primals_hashes, input_indices, node.ntype)
        old_hash = dag.primals_hashes[caller_array_index]
        
        needs_vals = new_hash != old_hash

        ## We can similarly determine whether to re-compute derivative information.
        ## Derivatives depend on the first `node.num_deps` inputs:
        needs_grads = prepare_grads && any(!isequal(new_hash), dag.partials_hashes[dep_indices, caller_array_index])
        needs_hessians = prepare_hessians && any(!isequal(new_hash), dag.partials2_hashes[dep_indices, dep_indices, caller_array_index])
    else
        new_hash = nothing
        needs_vals = true
        needs_grads = prepare_grads
        needs_hessians = prepare_hessians
    end
    ## Finally, do everything that is needed by calling a subfunction 
    ## that takes the operator as its first argument:
    output_indices = succ_indices(dag, node)
    __forward_functional(
        dag.operators[node.special_index], X, P, new_hash, dep_indices, output_indices, 
        caller_array_index, dag.primals, dag.partials, dag.partials2, dag.primals_hashes, dag.partials_hashes,
        dag.partials2_hashes, needs_vals, needs_grads, needs_hessians)

    return nothing
end

@views function __forward_functional(
    functional, X, P, fw_hash, dep_indices, output_indices, caller_array_index, 
    primals, partials, partials2, dag_primals_hashes, dag_partials_hashes, dag_partials2_hashes, 
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
                set_dxxhashes!(dag_partials2_hashes, fw_hash, dep_indices, output_indices, partial_output_i)
            else
                func_vals_and_grads!(y, Dy, functional, X, P, partial_output_i)
            end
            partials_hashes!(dag_partials_hashes, fw_hash, dep_indices, output_indices, partial_output_i)
        else
            func_vals!(y, functional, X, P, partial_output_i)
        end
        primals_hashes!(dag_primals_hashes, fw_hash, output_indices, partial_output_i)
    end     
    return nothing
end

# The forward pass for surrogates works quite similar:
function _forward_mod_node(dag, node; kwargs...) end # TODO

# ### Recursive Pullback(s)
#
# In performing “pullback”, we evaluate the chain rule by recursively substituting 
# partial derivatives of the outer function.
# That is, we propagate cotangents: starting with `∂z/∂z=1`, 
# for every predecessor `nᵢ` of `z` (if `z` is the only successor of `nᵢ`) we find
# `∂z/∂nᵢ = ∂z/∂z ∂z/∂nᵢ`. If `nᵢ` depends on `kⱼ`, then `∂z/∂kⱼ = ∂z/∂nᵢ ∂nᵢ/∂kⱼ`, 
# and so forth.
# If a node `n` has multiple successors `sₗ`, then
# `∂z/∂n = Σₗ ∂z/∂sₗ ∂sₗ/∂n`.

function pullback_node(
    dag, node::AbstractNode; mode=:no_surrogates, check_fw_hashes::Bool=true, kwargs...
)
    meta_node = model_node_to_dag_node(dag, node)
    return pullback_node(dag, meta_node; mode, check_fw_hashes, kwargs...)
end

# Below, `diff_index` is the array index of the node for which we compute all its partial 
# derivatives.
# The entry `dag.partials_hashes[i,j]` is a hash indicating for what input the value ∂nⱼ/∂nᵢ is valid.
# For the starting node `z`, it is seeded with the current primal hash of `z`.
# In the backward pass, we then build a reversed Hash tree:
# Starting with `h0 = hash(∂z/∂z)`, for every successor `n` of `z`, its hash value for 
# ∂z/∂n is set to `hash(hash(∂z/∂s₀), hash(hash(∂z/∂s₁), …))` for all successors `sₗ` of `n`.
# The hash arrays allow us to check for value consistency in case of partial re-evaluation 
# of the DAG, and to return early, supposing hashing is cheaper than summing
# the partial derivatives.

function pullback_node(
    dag, node::MetaNode; mode=:no_surrogates, check_fw_hashes::Bool=true, kwargs...
)
    @assert is_scalar_node(node) "Can only pullback on scalar nodes."
    diff_index = node.array_index               # index of node `z` for which partial derivatives are computed
    
    dag.grads[diff_index, diff_index] = 1       # set ∂z/∂z = 1
    fw_hash = dag.primals_hashes[diff_index]             
    dag.grads_hashes[diff_index, diff_index] = hash(fw_hash)
    
    fw_hash_vec = check_fw_hashes ? zeros(UInt, dag.nops + dag.nmodels) : nothing

    return pullback_node(dag, node, diff_index, fw_hash_vec; mode, kwargs...)
end

function scalar_succ_indices(dag, snode)
    return Iterators.flatten( succ_indices(dag, onode) for onode in successors(dag, snode) )
end

function dep_backward_hash(dag, node, diff_index)::UInt
    reduce(hash, dag.grads_hashes[si, diff_index] for si in scalar_succ_indices(dag, node))
end

function sum_successor_partials(dag::DAG{T}, node, diff_index, fw_hash_vec) where T
    ni = node.array_index
    dzdn = zero(T)
    for op_node in successors(dag,  node)
        dzdn += sum_op_partials(dag, op_node, ni, diff_index, fw_hash_vec)
    end
    return dzdn
end

@views function sum_op_partials(dag, op_node, ni, diff_index, fw_hash_vec)
    output_indices = succ_indices(dag, op_node)
    check_op_fw_hashes(dag, op_node, ni, output_indices, fw_hash_vec)
    return dag.grads[output_indices, diff_index]'dag.partials[ni, output_indices] 
    #src # TODO performance suboptimal for column-major stored dag.partials and dag.partials_hashes
end

check_op_fw_hashes(dag, op_node, ni, output_indices, fw_hash_vec::Nothing)=nothing
function check_op_fw_hashes(dag, op_node, ni, output_indices, fw_hash_vec)
    op_index = op_node.special_index
    if op_node.ntype == META_NODE_MOD
        op_index += dag.nops
    end
    if iszero(fw_hash_vec[op_index])
        fw_hash_vec[op_index] = op_forward_hash(dag, op_node)
    end
    op_hash = fw_hash_vec[op_index]
    if any(!isequal(op_hash), dag.partials_hashes[ni, output_indices])
        error("Partial derivative of node $(si) with respect to $(ni) does not match 
                hashes of operator node $(op_node.array_index).")
    end
    return nothing
end

function _pullback_dep_node(
    dag, node, diff_index, fw_hash_vec
)
    if node.array_index != diff_index
        ni = node.array_index
        new_bw_hash = dep_backward_hash(dag, node, diff_index)
        old_bw_hash = dag.grads_hashes[ni, diff_index]
        if new_bw_hash != old_bw_hash
            dag.grads[ni, diff_index] = sum_successor_partials(
                dag, node, diff_index, fw_hash_vec)
            dag.grads_hashes[ni, diff_index] = new_bw_hash
        end
    end
    return nothing
end

function pullback_var_node(dag, node, diff_index, fw_hash_vec; kwargs...)
    _pullback_dep_node(dag, node, diff_index, fw_hash_vec)
end

function pullback_state_node(dag, node, diff_index, fw_hash_vec; mode, kwargs...)
    _pullback_dep_node(dag, node, diff_index, fw_hash_vec)
    
    op_nodes = predecessors(dag, node)
    if length(op_nodes) > 1
        for op_n in op_nodes
            if mode == :no_surrogates && op_n.ntype != META_NODE_OP ||
                mode != :no_surrogates && op_n.ntype == META_NODE_MOD
                continue
            end
            pullback_node(
                dag, op_n, diff_index, fw_hash_vec; mode, kwargs...)
            break
        end
    else
        pullback_node(
                dag, only(op_nodes), diff_index, fw_hash_vec; mode, kwargs...)
    end
    return nothing
end

function pullback_op_node(dag, node, diff_index, fw_hash_vec; kwargs...)
    for in_node in predecessors(dag, node)
		pullback_node(dag, in_node, diff_index, fw_hash_vec; kwargs...)
	end
    return nothing
end

pullback_param_node(dag, node, diff_index, fw_hash_vec; kwargs...)=nothing
pullback_mod_node(dag, node, diff_index, fw_hash_vec; kwargs...)=nothing    # TODO

function pullback_node(dag, node, diff_index, fw_hash_vec; kwargs...)
    if node.ntype == META_NODE_VAR
		return pullback_var_node(dag, node, diff_index, fw_hash_vec; kwargs...)
    elseif node.ntype == META_NODE_PARAM
		return pullback_param_node(dag, node, diff_index, fw_hash_vec; kwargs...)
	elseif node.ntype == META_NODE_STATE
		return pullback_state_node(dag, node, diff_index, fw_hash_vec; kwargs...)
	elseif node.ntype == META_NODE_OP
		return pullback_op_node(dag, node, diff_index, fw_hash_vec; kwargs...)
    elseif node.ntype == META_NODE_MOD
		return pullback_mod_node(dag, node, diff_index, fw_hash_vec; kwargs...)
	end
end
#=
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
=#
end