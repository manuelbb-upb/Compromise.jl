next_var_index(model) = length(model.var_indices) + 1
next_state_index(model) = length(model.state_indices) + 1
next_param_index(model) = length(model.param_indices) + 1
next_op_index(model) = length(model.op_indices) + 1
next_mod_index(model) = length(model.model_indices) + 1

function add_scalar_node!(model, arr_name, node)
    i = pushend!(model.nodes, node)
    push!(getfield(model, arr_name), i)
    node.array_index = i
    return node
end

function add_variable!(model)
    x = VariableNode(next_var_index(model))
    return add_scalar_node!(model, :var_indices, x)
end

function add_state!(model)
    x = StateNode(next_state_index(model))
    return add_scalar_node!(model, :state_indices, x)
end

function add_parameter!(model, val)
    x = ParameterNode(next_param_index(model), val)
    return add_scalar_node!(model, :param_indices, x)
end

## These helper functions assume `eval_op!` and possible `eval_grads!` to be defined.
ensure_vec(::Nothing)=AbstractNode[]
ensure_vec(x) = [x,]
ensure_vec(x::AbstractVector) = x

function add_operator!(model, operator::AbstractNonlinearOperator, _x, _p, _ξ)
    x = ensure_vec(_x)
    p = ensure_vec(_p)
    ξ = ensure_vec(_ξ)
    ## check if any state ξi in ξ is already the successor of some operator
    for ξi in ξ
        if !isempty(ξi.in_nodes) && any(isa.(ξi.in_nodes, NonlinearOperatorNode))
            @warn "State $(ξi) already is the successor of an operator node. Doing nothing."
            return nothing
        end
    end
    
    ## setup the operator node
    index = next_op_index(model)
	f = NonlinearOperatorNode(; 
        index, operator,
        in_nodes = vcat(x, p),
        out_nodes = ξ,    
    )
	i = pushend!(model.nodes, f)
    push!(model.op_indices, i)
    f.array_index = i
    
    ## modify edges of input and output nodes
    for xi in x
        push!(xi.out_nodes, f)
    end
    for pj in p
        push!(pj.out_nodes, f)
    end
    for ξi in ξ
        push!(ξi.in_nodes, f)
    end

    return ξ
end

function add_operator!(model, operator::AbstractNonlinearOperator, x, p; dim_out::Int)
    ## add dependent variables ...
	ξ = [add_state!(model) for _=1:dim_out]
    return add_operator!(model, operator, x, p, ξ)
end

function add_surrogate!(model, surrogate::AbstractSurrogateModel, _x, _ξ)
    x = ensure_vec(_x)
    ξ = ensure_vec(_ξ)
    
    ## setup the operator node
    index = next_mod_index(model)
	f = SurrogateModelNode(; 
        index, surrogate,
        in_nodes = x,
        out_nodes = ξ,    
    )
	i = pushend!(model.nodes, f)
    push!(model.model_indices, i)
    f.array_index = i
    
    ## modify edges of input and output nodes
    for xi in x
        push!(xi.out_nodes, f)
    end
    for ξi in ξ
        push!(ξi.in_nodes, f)
    end

    return ξ
end
#=
function _eval_op_expr(mod, func_name, func_arg_x, func_arg_p)
    return quote
        dispatch_index = time_ns()
        num_x = length(ensure_vec($(esc(func_arg_x))))
        if isnothing($(esc(func_arg_p))) || length(ensure_vec($(esc(func_arg_p)))) == 0
            arg_expr = (:x,)
        else 
            arg_expr = (:x, :p)
        end
        Base.eval(
            $(mod),
            quote 
                function (::$($(typeof(eval_op!))))(y, v::Val{$(dispatch_index)}, x, p)
                    y .= $($(esc(func_name)))($(arg_expr...))
                    return nothing
                end
            end
        )
    end
end

function _parse_func_ex(func_ex)
    if length(func_ex.args) == 2
        func_name, func_arg_x = func_ex.args
        func_arg_p = :nothing
    else
        func_name, func_arg_x, func_arg_p = func_ex.args
    end
    return func_name, func_arg_x, func_arg_p
end

function _op_expr_dim(mod, model_ex, func_ex, dim_out)
    @assert dim_out isa Integer "Output dimension must be integer."
    @assert(
        Meta.isexpr(func_ex, :call, 2) || Meta.isexpr(func_ex, :call, 3),
        "Operator expression must correspond to a function call with 1 or 2 argument(s)."
    )
    
    func_name, func_arg_x, func_arg_p = _parse_func_ex(func_ex)
        
    return quote
        $(_eval_op_expr(mod, func_name, func_arg_x, func_arg_p))
        add_operator!(
            $(esc(model_ex)), dispatch_index, $(esc(func_arg_x)), $(esc(func_arg_p)); dim_out = $(dim_out))
    end
end

function _op_expr_eq(mod, src, model_ex, ex)
    ## @operator(model, ξ[1:5]=func(x))
    ## @operator(model, ξ=func(x))
    ## @operator(model, ξ=func(x[1:5]))
    ## @operator(model, ξ[1:3]=func(x[1:5]))

    lhs_ex, func_ex = ex.args
    @assert(
        Meta.isexpr(func_ex, :call, 2) || Meta.isexpr(func_ex, :call, 3),
        "Operator expression must correspond to a function call with 1 or 2 argument(s)."
    )
    
    ## extract symbol naming the lhs array
    if lhs_ex isa Symbol
        lhs_ex = Expr(:ref, lhs_ex, Expr(:call, :(:), 1, 1))
    end
    @assert Meta.isexpr(lhs_ex, :ref) "Invalid left-hand-side expression."
    lhs_arr_symb = lhs_ex.args[1]
    if lhs_ex.args[2] isa Integer
        lb_ex = ub_ex = lhs_ex.args[2]
    elseif Meta.isexpr(lhs_ex.args[2], :call, 3) && lhs_ex.args[2].args[1] == :(:)
        lb_ex, ub_ex = lhs_ex.args[2].args[[2,3]]
    else
        error("Invalid left-hand-side expression.")
    end
    
    func_name, func_arg_x, func_arg_p = _parse_func_ex(func_ex)
    return quote
        dim_out = $(esc(ub_ex)) - $(esc(lb_ex)) + 1
        $(_eval_op_expr(mod, func_name, func_arg_x, func_arg_p))
            $(esc(lhs_arr_symb)) = add_operator!(
            $(esc(model_ex)), dispatch_index, $(esc(func_arg_x)), $(esc(func_arg_p)); dim_out)
        #=
        $(esc(lhs_arr_symb)) = Base.eval(
            $(mod),
            Expr(
                :macrocall,
                Expr(:(.), $(@__MODULE__), QuoteNode(Symbol("@operator"))),
                $(src),
                $(Meta.quot(model_ex)),
                $(Meta.quot(func_ex)),
                dim_out
            )
        )=#
    end#quote
end

## @operator(model, func(x), dim_out)
## @operator(model, func(x, p), dim_out)
## @operator(model, ξ[1:5]=func(x))
## @operator(model, ξ[1:5]=func(x, p))
## @operator!(model, func!(ξ, x))            # TODO
## @operator!(model, func!(ξ, x, p))         # TODO
## @operator!(model, func!(ξ[1:3], x[1:5]))  # TODO
macro operator(model_ex, exs...)
    if length(exs) == 1
        ex = only(exs)
        if Meta.isexpr(ex, :(=), 2) || Meta.isexpr(ex, :(.=), 2)
            ## @operator(model, ξ[1:5]=func(x))
            ## @operator(model, ξ=func(x))
            ## @operator(model, ξ=func(x[1:5]))
            ## @operator(model, ξ[1:3]=func(x[1:5]))
            return _op_expr_eq(__module__, __source__, model_ex, ex)
        end
    elseif length(exs) == 2
        ## @operator(model, func(x), dim_out)
        func_ex, dim_out = exs
        return _op_expr_dim(__module__, model_ex, func_ex, dim_out)
    end

    return error("@operator macro cannot parse expression.")
end
=#