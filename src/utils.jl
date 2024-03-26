ensure_float_type(::Nothing, ::Type)=nothing
function ensure_float_type(arr::AbstractArray{F}, ::Type{F}) where F
	return arr
end

function ensure_float_type(arr::AbstractArray{F}, ::Type{T}) where {F,T}
	return T.(arr)
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

function vec2str(x, max_entries=typemax(Int), digits=10)
	x_end = min(length(x), max_entries)
	_x = x[1:x_end]
	_, i = findmax(abs.(_x))
	len = length(string(trunc(_x[i]))) + digits + 1
	x_str = "[\n"
	for xi in _x
		x_str *= "\t" * lpad(string(round(xi; digits)), len, " ") * ",\n"
	end
	x_str *= "]"
	return x_str
end

macro ignoraise(ex, loglevelex=nothing)
	has_lhs = false
	if Meta.isexpr(ex, :(=), 2)
		lhs, rhs = esc.(ex.args)
		has_lhs = true
	else
		rhs = esc(ex)
	end
	
	return quote
		ret_val = $(rhs)
		if isa(ret_val, AbstractStoppingCriterion)
			wrapped = if !isa(ret_val, WrappedStoppingCriterion)
				WrappedStoppingCriterion(ret_val, $(QuoteNode(__source__)), false)
			else
				ret_val
			end
			$(if !isnothing(loglevelex)
				quote
					if !wrapped.has_logged
						stop_msg = stop_message(wrapped.crit)
						if !isnothing(stop_msg)
							@logmsg $(esc(loglevelex)) stop_msg
						end
						wrapped.has_logged = true
					end
				end
			end)
			return wrapped
		end
		$(if has_lhs
			:($(lhs) = ret_val)
		else
			:(ret_val = nothing)
		end)
	end
end

macro ignorebreak(ex, loglevelex=nothing)
	has_lhs = false
	if Meta.isexpr(ex, :(=), 2)
		lhs, rhs = esc.(ex.args)
		has_lhs = true
	else
		rhs = esc(ex)
	end
	
	return quote
		ret_val = $(rhs)
		do_break = isa(ret_val, AbstractStoppingCriterion)
		$(if !isnothing(loglevelex)
			quote
				if ret_val isa WrappedStoppingCriterion
					wrapped = ret_val
					if !wrapped.has_logged
						stop_msg = stop_message(wrapped.crit)
						if !isnothing(stop_msg)
							@logmsg $(esc(loglevelex)) stop_msg
						end
						wrapped.has_logged = true
					end
				end
			end
		end)
		$(if has_lhs
			:($(lhs) = ret_val)
		else
			:(ret_val = nothing)
		end)
		do_break && break		
	end
end

universal_copy!(trgt, src)=nothing
function universal_copy!(trgt::AbstractArray{T, N}, src::AbstractArray{F, N}) where{T, F, N}
	copyto!(trgt, src)
	return nothing
end
function universal_copy!(trgt::Base.RefValue{T}, src::Base.RefValue{F}) where {T, F}
	trgt[] = src[]
	nothing
end

"""
	`var_bounds_valid(lb, ub)`
Return `true` if lower bounds `lb` and upper bounds `ub` are consistent.
"""
var_bounds_valid(lb, ub)=true
var_bounds_valid(lb::Nothing, ub::RVec)=!(any(isequal(-Inf), ub))
var_bounds_valid(lb::RVec, ub::Nothing)=!(any(isequal(Inf), lb))
var_bounds_valid(lb::RVec, ub::RVec)=all(lb .<= ub)

function project_into_box!(x, lb, ub)
	project_into_lower_bounds!(x, lb)
	project_into_upper_bounds!(x, ub)
end

project_into_lower_bounds!(x, ::Nothing)=nothing
project_into_lower_bounds!(x, lb)=begin
	x .= max.(x, lb)
	nothing
end
project_into_upper_bounds!(x, ::Nothing)=nothing
project_into_upper_bounds!(x, ub)=begin
	x .= min.(x, ub)
	nothing
end

function project_into_box!(x, lin_cons::LinearConstraints)
	project_into_lower_bounds!(x, lin_cons.lb)
	project_into_upper_bounds!(x, lin_cons.ub)
end

const SUPERSCRIPT_DICT = Base.ImmutableDict(
	0 => "⁰",
	1 => "¹",
	2 => "²",
	3 => "³",
	4 => "⁴",
	5 => "⁵",
	6 => "⁶",
	7 => "⁷",
	8 => "⁸",
	9 => "⁹"
)

const SUBSCRIPT_DICT = Base.ImmutableDict(
	0 => "₀",
	1 => "₁",
	2 => "₂",
	3 => "₃",
	4 => "₄",
	5 => "₅",
	6 => "₆",
	7 => "₇",
	8 => "₈",
	9 => "₉"
)

function supscript(num::Integer)
	return join((SUPERSCRIPT_DICT[i] for i in reverse(digits(num))), "")
end
function subscript(num::Integer)
	return join((SUBSCRIPT_DICT[i] for i in reverse(digits(num))), "")
end

function pretty_row_vec(
	x::AbstractVector;
	cutoff=80
)
	repr_str = "["
	lenx = length(x)
	for (i, xi) in enumerate(x)
		xi_str = @sprintf("%.2e", xi)
		if length(repr_str) + length(xi_str) >= cutoff
			repr_str *= "..."
			break
		end
		repr_str *= xi_str
		if i < lenx
			repr_str *= ", "
		end
	end
	repr_str *= "]"
	return repr_str
end
pretty_row_vec(x)=string(x)