module CompromiseEvaluators

# # Introduction
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

import Parameters: @with_kw_noshow

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
# * the `ValueDispatchNode`, to setup a problem and perform computations,
# * and the `ModelNode`, used to indicate that a sub-graph is meant to be modelled with  
#   surrogate models during optimization.

# The `ModelNode` type has an `index` for distinction, and references a model config:
abstract type AbstractModelConfig end
mutable struct ModelNode <: AbstractOperatorNode
    index :: Int
    cfg :: AbstractModelConfig
    array_index :: Union{Int, Nothing} # = nothing
    modelled_nodes :: AbstractVector{<:AbstractValueNode}
    in_nodes :: Vector{<:AbstractNode}# = AbstractNode[]
    out_nodes :: Vector{<:AbstractNode}# = AbstractNode[]
end

# For now, let's rather focus on the `ValueDispatchNode`.
# Like the value nodes, it has an `index` for distinction, and an `array_index` for
# retrieval in the processed `DAG`.
# But it also has a `dispatch_index`, which assumes our internal operator interface to 
# be implemented for `Val(dispatch_index)`.

@with_kw_noshow mutable struct ValueDispatchNode <: AbstractOperatorNode
	index :: Int
	dispatch_index :: UInt64
	array_index :: Union{Int, Nothing} = nothing
    ## meta-data for easy retrieval
    n_in :: Int
    n_pars :: Int
	n_out :: Int
    in_nodes :: Vector{<:AbstractNode} = AbstractNode[]
    out_nodes :: Vector{<:AbstractNode} = AbstractNode[]
end
## minimal printing
const super_chars = Base.ImmutableDict(
    0 => '⁰', 
    1 => '¹',
    2 => '²',
    3 => '³', 
    4 => '⁴', 
    5 => '⁵', 
    6 => '⁶', 
    7 => '⁷',
    8 => '⁸',
    9 => '⁹'
)
const sub_chars = Base.ImmutableDict(
    0 => '₀',
    1 => '₁',
    2 => '₂', 
    3 => '₃',
    4 => '₄',
    5 => '₅',
    6 => '₆',
    7 => '₇',
    8 => '₈', 
    9 => '₉'
)
super(num)=join(super_chars[d] for d in reverse(digits(num)))
sub(num)=join(sub_chars[d] for d in reverse(digits(num)))
Base.show(io::IO, node::ValueDispatchNode)=print(io, "Operator$(sub(node.n_in))$(super(node.n_out))($(node.index))")

# #### Operator Interface for `ValueDispatchNode`
# Namely, for every operator in our model we wish the following to functions to be defined:

"""
    eval_op!(y, ::Val{i}, x_and_p)

Evaluate the operator with special index `i::UInt64` at variable vector `x` with parameters `p`
and mutate the target vector `y` to contain the result.
"""
function eval_op!(y, ::Val{I}, x_and_p) where I
    return error("No implementation of `eval_op!` for operator with special index $I.")
end

"""
    eval_grads!(Dy, ::Val{i}, x_and_p)

Compute the gradients of the operator with special index `i` at variable vector `x` 
with parameters `p` and mutate the target matrix `Dy` to contain the gradients w.r.t. `x`
in its columns. That is, `Dy` is the transposed Jacobian at `x`.
"""
function eval_grads!(Dy, ::Val{I}, x_and_p) where I
    return error("No implementation of `eval_grads!` for operator with special index $I.")
end

# The combined forward-function `eval_op_and_grads!` is derived from `eval_op!` and 
# `eval_grads!`, but can be customized easily:
function eval_op_and_grads!(y, Dy, val, x_and_p; do_grads=false)
    eval_op!(y, val, x_and_p)
    if do_grads
        eval_grads!(Dy, val, x_and_p)
    end
    return nothing
end

# If Hessian matrices are needed, implement `eval_hessians!(H, ::Val{I}, x_and_p)`
# for a operator dispatching on `I`.
# Assume `H` to be a vector of matrices, each matrix corresponding to the hessian of 
# some operator output.
# That is, `H[:,:,1]` is a matrix containing second order partial derivatives of the first output,
# `H[:,:,2]` has second order derivatives for the second output, and so forth...
# Moreover, in contrast to `eval_grads!`, the Hessian information should be stored in 
# correct order- `H[i,j,k]` should correspond to `∂yₖ/∂xᵢ∂xⱼ`:

"""
    eval_hessians!(H, ::Val{i}, x_and_p)

Compute the Hessians of the operator with special index `i` at variable vector `x` 
with parameters `p` and mutate the target array `H` to contain the Hessians 
along its last index.
That is, `H[:,:,i]` is the Hessian at `x` and `p` w.r.t. `x` of output `i`.
"""
function eval_hessians!(H, ::Val{I}, x_and_p) where I
    return error("No implementation of `eval_hessians!` for operator with special index $I.")
end

# Some operators might support partial evaluation. 
# They should implement these methods:
supports_partial_evaluation(val) = false
supports_partial_jacobian(val) = false
## length(y)==length(outputs)
eval_op!(y, ::Val{I}, x_and_p, outputs) where{I}=error("Partial evaluation not implemented.")
## size(Dy)==(length(x), length(outputs))
eval_grads!(Dy, ::Val{I}, x_and_p, outputs) where{I}=error("Partial Jacobian not implemented.")

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
include("model_helpers.jl")

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
    META_NODE_MODEL = 5
end

# The `array_index` of a node `n::MetaNode` indicates its position in a list of nodes of 
# the DAG.
struct MetaNode
	ntype :: META_NODE_TYPE
	array_index :: Int
    src_index :: Int
	special_index :: UInt64
end

## helpers for construction:
is_dep_node(meta_node) = UInt8(meta_node.ntype) <= 2
is_scalar_node(meta_node) = UInt8(meta_node.ntype) <= 3

## minimal printing
function cpad(str, len)
    l = length(str)
    l >= len && return str
    del = len - l
    p, r = divrem(del, 2)
    p1 = l + p
    p2 = p1 + p + r
    return rpad(lpad(str, p1), p2)
end

function Base.show(io::IO, n::MetaNode)
    print(io, 
        "MetaNode{" *
        cpad(string(n.ntype)[11:end],5) * "}(" *     # VAR, PARAM, STATE, OP, MODEL,
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
# * The index of a model in the `models` array of a DAG if `n.ntype==META_NODE_MODEL`.
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

    pred_is_dep :: BitMatrix
    ## NOTE
    ## previously, I used a single adjecency matrix, where 
    ## the predecessor indices of node with index j where stored as negative 
    ## values, and the successor indices as positive entries.
    ## This required akward filtering and lead to allocations during evaluation.
    
    primals :: Vector{Float64}
    partials :: SparseMatrixCSC{Float64, Int}
    partials2 :: SAK.SparseArray{Float64, 3}
    
    p_hash :: Base.Ref{UInt64}
    x_hash :: Vector{UInt64}
    dx_hash :: SparseMatrixCSC{UInt64, Int}
    ddx_hash :: Vector{UInt64}

    models :: Vector{Any}
end

## minimal printing
function Base.show(io::IO, dag::DAG)
    print(io, """
    DAG with 
    * $(dag.nvars) variables, $(dag.nparams) parameters and $(dag.nstates) states;
    * $(dag.nops) operator nodes and $(dag.nmodels) surrogate models.""")
end
#=
@views function node_indices(adj::SparseMatrixCSC, j)
    return rowvals(adj)[nzrange(adj,j)]
end
=#

function consistent_array_indices!(model)
    node_counters = fill(0, 5)
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
        elseif n isa ValueDispatchNode
            j = node_counters[4] += 1
            model.op_indices[j] = ni
        elseif n isa ModelNode
            j = node_counters[5] += 1
            model.model_indices[j] = ni
        end
    end
    return nothing
end

function initialize(model; sort_nodes=true, check_cycles=false)
	
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
    si = 0  # special_index == array_index
    for ni in model.var_indices
        si += 1
        nodes[si] = MetaNode(META_NODE_VAR, si, ni, si)
        model_indices[si] = ni
    end
    for ni in model.state_indices
        si += 1
        nodes[si] = MetaNode(META_NODE_STATE, si, ni, si)
        model_indices[si] = ni
    end
    for ni in model.param_indices
        si += 1
        nodes[si] = MetaNode(META_NODE_PARAM, si, ni, si)
        n = model.nodes[ni]
        primals[si] = n.value
        model_indices[si] = ni
    end
    @assert si == nvals

    ## now the operater nodes
    for ni in model.op_indices
        si += 1
        n = model.nodes[ni]
        nodes[si] = MetaNode(META_NODE_OP, si, ni, n.dispatch_index)
        model_indices[si] = ni
    end
    for ni in model.model_indices
        si += 1
        n = model.nodes[ni]
        nodes[si] = MetaNode(META_NODE_MODEL, si, ni, n.index)
        model_indices[si] = ni
    end
    dag_indices = sortperm(model_indices)
    
    ## build partial adjacency matrices, because edge information is otherwise deleted with 
    ## Kahn's Algorithm below
    adj_pred = spzeros(Int, nvals, nnodes)
    adj_succ = spzeros(Int, nvals, nnodes)
    _pred_is_dep = zeros(Bool, nvals, nnodes)
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
            if is_dep_node(m)
                _pred_is_dep[pos, meta_ni] = true
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
    pred_is_dep = _pred_is_dep[1:max_in, :]

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

    p_hash = Ref(zero(UInt64))
	x_hash = fill(p_hash[], ndeps)
	dx_hash = spzeros(UInt64, ndeps, ndeps)
    ddx_hash = copy(x_hash)
	
	return DAG(;
		nvars, nparams, nstates, nops, nmodels,
		nodes,
        dag_indices,
        sorting_indices,
		adj_pred,
		adj_succ,
        pred_is_dep,
		primals, 
		partials,
        partials2,
        p_hash,
		x_hash,
		dx_hash,
        ddx_hash,
        models = Any[]
	)
end

# ## Evaluation

# ### Recursive Forward Pass

# Evaluate the subgraph of a DAG `dag` until reaching a target node and cache the 
# intermediate values.

"Given a node `n`, return iterable of `array_index` that preceed `n` (its inputs)."
@views pred_indices(dag, node) = nonzeros(dag.adj_pred)[nzrange(dag.adj_pred, node.array_index)]
"Given a node `n`, return iterable of `array_index` that succeed `n` (its outputs)."
@views succ_indices(dag, node) = nonzeros(dag.adj_succ)[nzrange(dag.adj_succ, node.array_index)]

@views predecessors(dag, node) = dag.nodes[pred_indices(dag, node)]
@views successors(dag, node) = dag.nodes[succ_indices(dag, node)]

function eval_node(dag, node::AbstractNode, x; kwargs...)
    meta_node = dag.nodes[dag.dag_indices[node.array_index]]
    return eval_node(dag, meta_node, x; kwargs...)
end

function eval_node(dag, node::MetaNode, x; kwargs...)
    x_hash = hash(dag.p_hash, hash(x))
    return eval_node(dag, node, x, x_hash; kwargs...)
end

function eval_node(dag, node::MetaNode, x, x_hash; kwargs...)
    if is_dep_node(node)
        ni = node.array_index
	    if x_hash == dag.x_hash[ni]
		    return nothing
	    end
    else
        ni = -1
    end
	
	if node.ntype == META_NODE_VAR
		eval_var_node(dag, node, x, x_hash; kwargs...)
	elseif node.ntype == META_NODE_PARAM
		eval_param_node(dag, node, x, x_hash; kwargs...)
	elseif node.ntype == META_NODE_STATE
		eval_state_node(dag, node, x, x_hash; kwargs...)
	elseif node.ntype == META_NODE_OP
		eval_op_node(dag, node, x, x_hash; kwargs...)
	elseif node.ntype == META_NODE_MODEL
		eval_model_node(dag, node, x, x_hash; kwargs...)
	end

    if ni > 0
        dag.x_hash[ni] = x_hash
    end

    return nothing
end

function eval_param_node(args...; kwargs...) end
function eval_model_node(args...; kwargs...) end    # TODO

@views function eval_var_node(dag, node, x, x_hash; kwargs...)
    dag.primals[node.special_index] = x[node.special_index]
    return nothing
end

@views function eval_state_node(dag, node, x, x_hash; kwargs...)
    op_array_index = only(pred_indices(dag, node))
    return eval_op_node(dag, dag.nodes[op_array_index], x, x_hash, node.array_index; kwargs...)
end

ensure_int_vec(i::Integer) = [i,]
ensure_int_vec(arr::AbstractVector{<:Integer}) = arr
@views function eval_op_node(dag, node, x, x_hash, out_array_index=nothing; prepare_grads=true, kwargs...)
    ## recursively set inputs for current operator
	input_indices = pred_indices(dag, node)
	for n in dag.nodes[input_indices]
		eval_node(dag, n, x, x_hash; prepare_grads, kwargs...)
	end
	
    ## compute output(s)   
	dispatch_val = Val(node.special_index)  # value to dispatch `eval_op!` on
    x_and_p = dag.primals[input_indices]    # input vector for `eval_op`
    output_indices = succ_indices(dag, node)
    output_pos = nothing
    if supports_partial_evaluation(dispatch_val) && !isnothing(out_array_index)
        output_pos = findfirst(isequal(out_array_index), output_indices)
        if !isnothing(output_pos)
            ## without `@views` wrapping this function, we would need `ensure_int_vec`
            ## to index dag.primals correctly here
            eval_op!(dag.primals[output_indices[output_pos]], dispatch_val, x_and_p, output_pos)
        end
    end
    if isnothing(output_pos)
	    eval_op!(dag.primals[output_indices], dispatch_val, x_and_p)
    end
    
    ## compute gradients
    if prepare_grads
        nin = length(input_indices)
        grad_inputs = input_indices[dag.pred_is_dep[1:nin, node.array_index]]
        if supports_partial_jacobian(dispatch_val) && !isnothing(output_pos)
            jacT = dag.partials[grad_inputs, output_indices[output_pos]]
	        eval_grads!(jacT, dispatch_val, x_and_p, output_pos)
        else
	        jacT = dag.partials[grad_inputs, output_indices]
	        eval_grads!(jacT, dispatch_val, x_and_p)
        end
    end

	return nothing
end

function pullback_node(dag, node::AbstractNode)
    meta_node = dag.nodes[dag.dag_indices[node.array_index]]
    return pullback_node(dag, meta_node)
end

function pullback_node(dag, node::MetaNode)
    @assert is_scalar_node(node) "Can only pullback on scalar nodes."
    diff_index = node.array_index
    dag.partials[diff_index, diff_index] = 1
    x_hash = dag.x_hash[diff_index]
    return pullback_node(dag, node, diff_index, x_hash)
end

function pullback_node(dag, node, diff_index, x_hash)
	if is_dep_node(node)
        ni = node.array_index
	
	    if dag.x_hash[ni] != x_hash
		    error("Fractured evaluation tree, gradients not valid. Perform forward pass up until node with array index $diff_index.")
	    end
    end

    if node.ntype == META_NODE_VAR
		return pullback_var_node(dag, node, diff_index, x_hash)
    elseif node.ntype == META_NODE_PARAM
		return pullback_param_node(dag, node, diff_index, x_hash)
	elseif node.ntype == META_NODE_STATE
		return pullback_state_node(dag, node, diff_index, x_hash)
	elseif node.ntype == META_NODE_OP
		return pullback_op_node(dag, node, diff_index, x_hash)
    elseif node.ntype == META_NODE_MODEL
		return pullback_model_node(dag, node, diff_index, x_hash)
	end
end

function pullback_param_node(dag, node, diff_index, x_hash) end
function pullback_model_node(dag, node, diff_index, x_hash) end # TODO

function pullback_var_node(dag, node, diff_index, x_hash) 
	return _pullback_set_partials(dag, node, diff_index, x_hash)
end

function pullback_state_node(dag, node, diff_index, x_hash)
	_pullback_set_partials(dag, node, diff_index, x_hash)
	op_array_index = only(pred_indices(dag, node))
	op_node = dag.nodes[op_array_index]
	return pullback_op_node(dag, op_node, diff_index, x_hash)
end

@views function _pullback_set_partials(dag, node, diff_index, x_hash)
	ni = node.array_index

	ni == diff_index && return nothing
	
	dag.dx_hash[ni, diff_index] == x_hash && return nothing

	## let node with `diff_index` be `z`
	## let current node be `n`
	## we assume the cotangents of all successors, ∂pᵢ/∂n, to be available
	## we can then sum them up:
	## ∂z/∂n = ̄n = Σ ∂z/∂p ∂p/∂n
	dzdn = 0.0
	for outn in successors(dag, node)
		if !is_scalar_node(outn) 
			for _outn in successors(dag, outn)
				si = _outn.array_index
                ## ∂z/∂n += ∂z/∂pᵢ * ∂pᵢ/∂n
		        dzdn += dag.partials[si, diff_index] * dag.partials[ni, si]
			end
		else
			si = outn.array_index
		    dzdn += dag.partials[si, diff_index] * dag.partials[ni, si] 
		end
	end
	dag.partials[ni, diff_index] = dzdn

    dag.dx_hash[ni, diff_index] = x_hash
	return nothing
end

@views function pullback_op_node(dag, node, diff_index, x_hash)
	# NOTE I do not check or set dag.dx_hash[node.array_index] for operator nodes
	# this is so that each nodes x_hash can be checked against the leave node's x_hash
	# and a error is thrown for fractured trees
	
	input_idx = pred_indices(dag, node)
	
	for in_idx in input_idx
		n = dag.nodes[in_idx]
		pullback_node(dag, n, diff_index, x_hash)
	end
	
	return nothing
end

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
H(f∘g) = (∇g)ᵀ ⋅ Hf(g) ⋅ ∇g + ∑ₖ ∂ₖf Hgₖ
```
=#

function forward_hessian(dag, node::AbstractNode)
    meta_node = dag.nodes[dag.dag_indices[node.array_index]]
    return forward_hessian(dag, meta_node)
end

function forward_hessian(dag, node::MetaNode)
    @assert is_scalar_node(node) "Can only compute Hessian of scalar nodes."
    diff_index = node.array_index
    x_hash = dag.x_hash[diff_index]
    return forward_hessian(dag, node, diff_index, x_hash)
end

function forward_hessian(dag, node, diff_index, x_hash)
	if is_dep_node(node)
        ni = node.array_index
	
	    if dag.x_hash[ni] != x_hash
		    error("Fractured evaluation tree, gradients not valid. Perform forward pass up until node with array index $diff_index.")
	    end
        if dag.ddx_hash[ni] == x_hash
            return nothing
        end
    else
        ni = -1
    end
	
	if node.ntype == META_NODE_STATE
		forward_hessian_state(dag, node, diff_index, x_hash)
	elseif node.ntype == META_NODE_OP
		forward_hessian_op(dag, node, diff_index, x_hash)
	end
    if ni > 0
        dag.ddx_hash[ni] = x_hash
    end
    return nothing
end

@views function forward_hessian_state(dag, node, diff_index, x_hash)
    op_array_index = only(pred_indices(dag, node))
    return forward_hessian_op(dag, dag.nodes[op_array_index], diff_index, x_hash)
end

import LinearAlgebra as LA
@views function forward_hessian_op(dag, node, diff_index, x_hash)
    input_indices = pred_indices(dag, node)

    # set Hessians of inputs
    for in_ind in input_indices
        forward_hessian(dag, dag.nodes[in_ind], diff_index, x_hash)
    end

    output_indices = succ_indices(dag, node)
    
    x_and_p = dag.primals[input_indices]    # input vector for `eval_hess!`
    dispatch_val = Val(node.special_index)  # value to dispatch `eval_hess!` on

    nin = length(input_indices)
    hess_inputs = input_indices[dag.pred_is_dep[1:nin, node.array_index]]
	
    H = dag.partials2[hess_inputs, hess_inputs, output_indices]
    eval_hessians!(H, dispatch_val, x_and_p)

    _JgT = dag.partials[:, hess_inputs]
    ij_indices = setdiff(unique(rowvals(_JgT)), hess_inputs)       # TODO store these indices in some array of `dag` during pullback...
    
    for out_ind in output_indices
        Hy = dag.partials2[ij_indices, ij_indices, out_ind]     # target array
        Hy_u = dag.partials2[hess_inputs, hess_inputs, out_ind] # hessian of output `y` w.r.t. inputs u_1, …, u_k
        JuT = dag.partials[ij_indices, hess_inputs]             # transposed Jacobian of inputs
        Hy .= JuT * Hy_u * Ju'    # TODO can we do some in-place stuff here?
        for k in hess_inputs
            dy_uk = dag.partials[k, out_ind]
            Huk = dag.partials2[ij_indices, ij_indices, k]
            Hy += dy_uk .* Huk
        end
    end

	return nothing
end

end