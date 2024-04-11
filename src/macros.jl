
"""
	@ignoraise a, b, c = critical_function(args...) [indent=0]

Evaluate the right-hand side.
If it returns an `AbstractStoppingCriterion`, make sure it is wrapped and return it.
Otherwise, unpack the returned values into the left-hand side.

Also valid:
```julia
@ignoraise critical_function(args...)
@ignoraise critical_function(args...) indent_var
```
The indent expression must evaluate to an `Int`.
"""
macro ignoraise(ex, indent_ex=0)
	has_lhs, lhs, rhs = _parse_ignoraise_expr(ex)
	
	return quote
		ret_val = $(rhs)
		if ret_val isa AbstractStoppingCriterion
			return wrap_stop_crit(
				ret_val, $(QuoteNode(__source__)), $(esc(indent_ex))
			)
		end
		$(if has_lhs
			:($(lhs) = wrapped)
		else
			:(ret_val = nothing)
		end)
	end
end

"""
	@ignorebreak ret_var = critical_function(args...)

Similar to `@ignoraise`, but instead of returning if `critical_function` returns 
an `AbstractStoppingCriterion`, we `break`.
This allows for post-processing before eventually returning.
`ret_var` is optional, but in constrast to `@ignoraise`, we unpack unconditionally,
so length of return values should match length of left-hand side expression.
"""
macro ignorebreak(ex, indent_ex=0)
	has_lhs, lhs, rhs = _parse_ignoraise_expr(ex)
		
	return quote
		ret_val = $(rhs)
		ret_val = wrap_stop_crit(
			ret_val, $(QuoteNode(__source__)), $(esc(indent_ex)))
		do_break = ret_val isa AbstractStoppingCriterion
		$(if has_lhs
			:($(lhs) = ret_val)
		else
			:(ret_val = nothing)
		end)
		do_break && break		
	end
end


macro forward(type_ex, call_ex)
	@assert Meta.isexpr(type_ex, :(.), 2)
	type_name, wrapped_fn_name = type_ex.args
    @assert call_ex.head == :call
    func_name = call_ex.args[1]
    func_args = Any[]
    kwargs = nothing
    for arg_ex in call_ex.args[2:end]
        if arg_ex isa Symbol
            push!(func_args, arg_ex)
        elseif Meta.isexpr(arg_ex, :(::))
			if length(arg_ex.args) < 2
				pushfirst!(arg_ex.args, gensym())
			end
			vt = arg_ex.args[2]
            if vt == type_name
                push!(func_args, :(getfield($(arg_ex.args[1]), $(wrapped_fn_name))))
            else
                push!(func_args, arg_ex.args[1])
            end
        elseif Meta.isexpr(arg_ex, :parameters)
            kwargs = arg_ex.args
        end
    end
    if isnothing(kwargs)
        return quote 
            function $(func_name)($(call_ex.args[2:end]...))
                return $(func_name)($(func_args...))
            end
        end |> esc
    else
        return quote 
            function $(func_name)($(call_ex.args[2:end]...))
                return $(func_name)($(func_args...); $(kwargs...))
            end
        end |> esc
    end
end